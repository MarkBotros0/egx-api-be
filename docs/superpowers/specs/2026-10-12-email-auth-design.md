# Email accounts, open sign-up, Google + Apple sign-in, SMTP email — design

Date: 2026-10-12 · Repos: `egx-api-be` (this spec), `egx-api-fe`, `egx-api-mobile`

## Goal

Replace admin-made username/password accounts with email accounts that anyone can create:
email + password (verified by an emailed link), Google (web + mobile), or Apple (iOS app only).
Admins invite by email instead of copy-pasting generated passwords; users set their own
password from an emailed link and can reset a forgotten one themselves.

## Decisions (from the user)

- Login identity is the **email**. Usernames are removed.
- **Open sign-up with email verification.** Email+password accounts cannot sign in until the
  link is clicked. Google/Apple emails count as verified.
- Sign-up asks for a **name**; the UI shows `name || email`.
- **Existing users are deleted** (with all their data) by a one-off script the user runs.
  The only admin is `mark3essam@gmail.com`.
- SMTP via a **Gmail app password**, configured through generic `SMTP_*` env vars.
- **Forgot password** is self-service.
- Our backend keeps issuing its own JWT (HS256, 30 days, `AUTH_SECRET`). No auth vendor, no Firebase.
- Apple sign-in is iOS-app only (not web, not Android).

## Non-goals

Universal/app links (email links open the web), 2FA, changing email address, changing password
while signed in (use Forgot password), Apple on web/Android, social providers other than Google/Apple.

## 1. Data model (Postgres, `app/core/db.py`)

`users` is recreated in this shape (new installs get it from `init_db`; the reset script
drops the old table first):

| column | type | notes |
|---|---|---|
| `id` | TEXT PK | uuid4, unchanged meaning — every other table keys on it |
| `email` | TEXT UNIQUE NOT NULL | stored trimmed + lowercased |
| `name` | TEXT NOT NULL DEFAULT '' | ≤ 80 chars |
| `password_hash` | TEXT NULL | NULL = no password (Google/Apple-only or invited-not-yet-set) |
| `email_verified` | BOOLEAN NOT NULL DEFAULT FALSE | |
| `google_sub` | TEXT UNIQUE NULL | Google `sub` claim |
| `apple_sub` | TEXT UNIQUE NULL | Apple `sub` claim |
| `apple_refresh_token` | TEXT NULL | for revocation on account deletion |
| `role` | TEXT NOT NULL DEFAULT 'user' | still written ONLY by `AUTH_ADMINS` seeding |
| `is_active` | BOOLEAN NOT NULL DEFAULT TRUE | |
| `last_email_at` | TEXT NULL | when we last emailed this user — drives the 60 s cooldown |
| `created_at` | TEXT NOT NULL | ISO-8601 UTC `…Z` |

`auth_tokens` (email-link tokens):

| column | type | notes |
|---|---|---|
| `token_hash` | TEXT PK | SHA-256 hex of the raw token; the raw token is never stored |
| `user_id` | TEXT NOT NULL | |
| `purpose` | TEXT NOT NULL | `verify` \| `reset` \| `invite` |
| `expires_at` | TEXT NOT NULL | verify 24 h, reset 1 h, invite 7 days |
| `used_at` | TEXT NULL | set on first use; a used token is rejected |
| `created_at` | TEXT NOT NULL | |

Raw tokens: `secrets.token_urlsafe(32)`. Issuing a new token of a purpose for a user deletes
that user's older unused tokens of the same purpose (only the newest link works).

**User-owned tables** (deleted with the user, both by account deletion and by the reset
script): `portfolio`, `portfolio_sales`, `portfolio_dividends`, `user_settings`, `watchlist`,
`push_devices`, `alert_preferences`, `alert_log`, `auth_tokens`. A test asserts this list
matches every table in `init_db` that has a `user_id` column, so a future table can't be missed.

## 2. Seeding and the reset script

- `AUTH_ADMINS` becomes a comma-separated list of **emails**. On boot, each listed email is
  created if missing (`email_verified=TRUE`, no password, `name=''`) and stamped `admin`; others
  are demoted only when the var is non-empty (existing rule kept). Production value:
  `AUTH_ADMINS=mark3essam@gmail.com`.
- `AUTH_USERS` is removed (code, docs, `.env.example`). The admin signs in with Google (Gmail)
  or uses Forgot password.
- `scripts/reset_users_for_email_auth.py`: prints row counts per user-owned table, requires the
  user to type `DELETE ALL USERS`, then in one transaction deletes all rows of the user-owned
  tables and drops `users`, then runs `init_db()` + seeding. Run by the user against
  production; never run automatically.

## 3. API (`app/routers/auth.py`, `app/core/auth.py`, new `app/core/email.py`, `app/core/oauth.py`)

All new auth routes are added to `PUBLIC_ENDPOINTS`; `tests/test_auth_gate.py` keeps sweeping
the real route table. Passwords: 8–128 chars (existing bcrypt-over-SHA-256 hashing).
Email validation: trimmed, lowercased, one `@`, a dot in the domain, ≤ 254 chars.

| Route | Body | Result |
|---|---|---|
| `POST /api/auth/signup` | `{name, email, password}` | Always `200 {"status":"check_email"}`. New email → create unverified user + send verify email. Existing **unverified** email → replace its password hash and name, resend verify. Existing **verified** email → send an "you already have an account — sign in or reset your password" email. |
| `POST /api/auth/resend-verification` | `{email}` | Always `200`. Sends a new verify link if the account exists and is unverified. |
| `POST /api/auth/verify-email` | `{token}` | Marks verified, consumes token, returns `{access_token, token_type, user}` (signed in). |
| `POST /api/auth/login` | `{email, password}` | `200 {access_token, token_type, user}`; `401 "Invalid email or password"` (same message for unknown email, wrong password, no password set); `403 {"detail":"email_not_verified"}` when the password is right but email unverified; `403` for disabled. |
| `POST /api/auth/forgot-password` | `{email}` | Always `200`. Sends a `reset` link if an active account exists. |
| `POST /api/auth/set-password` | `{token, password}` | Accepts `reset` and `invite` tokens. Sets the hash, sets `email_verified=TRUE`, consumes the token, returns `{access_token, token_type, user}`. |
| `POST /api/auth/google` | `{idToken, name?}` | `200 {access_token, token_type, user}`. |
| `POST /api/auth/apple` | `{identityToken, authorizationCode?, name?}` | `200 {access_token, token_type, user}`. |
| `GET /api/auth/me` | — | `{id, email, name, role}` |
| `DELETE /api/auth/me` | — | Deletes the caller and all their rows; revokes the Apple token if stored. `204`. The **last active admin cannot delete themselves** (`409`). |

Every session response is `{access_token, token_type: "bearer", user}` (the shape clients already parse); `user` is the same `{id, email, name, role}` as `/me`.

Token errors (`verify-email`, `set-password`): unknown, used, expired, or wrong purpose →
`400 {"detail":"invalid_or_expired_link"}`. Disabled user → `403`.

**Cooldown:** every email to a user (verify, reset, invite, already-registered) first checks
`users.last_email_at`; if it is less than 60 s ago nothing is sent and the endpoint still returns
its normal response. A successful send sets `last_email_at`.

### Google and Apple verification (`app/core/oauth.py`)

Both use `PyJWT`'s `PyJWKClient` (requirements: `pyjwt[crypto]`), JWKS cached in-process.
- **Google:** JWKS `https://www.googleapis.com/oauth2/v3/certs`; `iss` ∈
  {`https://accounts.google.com`, `accounts.google.com`}; `aud` ∈ `GOOGLE_CLIENT_IDS`
  (comma-separated: web, Android, iOS); require `email_verified == true`. Name from `name`.
- **Apple:** JWKS `https://appleid.apple.com/auth/keys`; `iss == https://appleid.apple.com`;
  `aud` ∈ `APPLE_CLIENT_IDS` (production: `com.markbotros.borsacompass` ONLY). Expo Go's
  shared audience `host.exp.Exponent` is rejected (401) unless the dev-only flag
  `APPLE_ALLOW_EXPO_GO=1` is set on a development backend — every Expo Go project receives
  tokens with that audience, so accepting it in production would let anyone's Expo Go app
  link to an account by email. The token/revoke calls use the first `APPLE_CLIENT_IDS` entry
  that is not `host.exp.Exponent`. Apple sends `email` only on first
  sign-in and the name never in the token — the app passes `name` from the native response.
  If `authorizationCode` is present and `APPLE_TEAM_ID`/`APPLE_KEY_ID`/`APPLE_PRIVATE_KEY` are
  set, exchange it at `https://appleid.apple.com/auth/token` (client secret = ES256 JWT) and
  store `apple_refresh_token`. On account deletion, POST it to
  `https://appleid.apple.com/auth/revoke` — best-effort: a failure is logged, never blocks deletion.
- A provider that isn't configured (empty client-id var) returns `503`.

### Account resolution for Google/Apple (`provider` = google|apple, `sub`, `email`)

1. A user with that `{provider}_sub` exists → sign in (403 if disabled).
2. Else a user with that email exists → link: set `{provider}_sub`. If that user was
   **unverified**, also set `password_hash = NULL` (prevents a pre-registered password from
   surviving a takeover) and `email_verified = TRUE`. Fill `name` if empty.
3. Else create `{email, name, email_verified=TRUE, {provider}_sub}`.
4. Apple with no email in the token and no matching sub → `400` (Apple only omits email on
   repeat sign-ins, which step 1 handles).

### Email (`app/core/email.py`)

`smtplib.SMTP(SMTP_HOST, SMTP_PORT)` + `starttls()` + login; envs `SMTP_HOST`
(default `smtp.gmail.com`), `SMTP_PORT` (default `587`), `SMTP_USER`, `SMTP_PASSWORD`.
The sender is always `"Borsa Compass <SMTP_USER>"`. Multipart plain + minimal HTML.
Links use the FIRST entry of `FRONTEND_ORIGIN` (the production web origin; the same variable
sets CORS): `{origin}/verify-email?token=…`, `{origin}/set-password?token=…` (reset and
invite share the page). It is read raw, with no localhost default. Missing SMTP or
`FRONTEND_ORIGIN` → `503 "Email is not configured"` on
routes that must send (signup, invite, admin reset); the "always 200" routes also 503 rather
than pretend. An SMTP failure → `502 "Could not send email"`; the token row is deleted so the
cooldown doesn't block a retry. Four templates: verify, reset, invite ("{admin name} invited you
to Borsa Compass"), already-registered. Tests swap the sender for an in-memory outbox.

## 4. Admin API (`app/routers/users.py`)

- `GET /api/users` → `{id, email, name, role, is_active, email_verified, methods:[password|google|apple], created_at}`.
- `POST /api/users` `{email, name?}` → creates an unverified user with no password, sends an
  `invite` email, `201`. Existing email → `409`, except an active account nobody has set up
  (unverified, no password, no Google/Apple), which is re-invited (`201`, `429` inside the
  cooldown) — a failed invite email must not strand the address.
- `POST /api/users/{id}/send-reset` → sends a `reset` email (`invite` if the user has no
  password and never verified). Replaces `POST /api/users/{id}/password`.
- `PATCH` (active) and `DELETE` keep the self/last-admin guards. `DELETE` now removes all
  user-owned rows (shared helper with `DELETE /api/auth/me`).
- No API writes `role` (existing grep test stays).
- Generated passwords are removed (`generate_password`, related tests and docs).

## 5. Web (`egx-api-fe`)

- `lib/auth.ts`, `lib/api.ts`, `lib/types.ts`: `AuthUser = {id, email, name, role}`;
  `displayName(u) = u.name || u.email`; client functions for every route above. These sync to mobile.
- Pages: `/login` (email, password, "Continue with Google", links to `/signup` and
  `/forgot-password`; on `email_not_verified` show "Check your inbox" + Resend), `/signup`,
  `/forgot-password`, `/verify-email` (auto-submits the token, then lands signed in on `/`),
  `/set-password` (new password + confirm, then signed in on `/`). All reachable signed out
  (middleware allowlist). Neutral confirmation copy that never reveals whether an email exists.
- Google: Google Identity Services script, `NEXT_PUBLIC_GOOGLE_CLIENT_ID`; button hidden when unset.
- Header/Navbar shows `displayName`. New **Account** section (reached from the header user
  menu) with name, email, and "Delete account" (type `DELETE` to confirm → `DELETE /api/auth/me`
  → logout).
- Admin page: Invite form (email + optional name), user rows with methods/verified badges,
  "Send password reset", disable/delete. `PasswordRevealDialog` and the credential copy block are deleted.

## 6. Mobile (`egx-api-mobile`)

- Shared logic via `npm run sync:shared`.
- Screens: Login (email/password, Google, Apple on iOS, links), Sign up, Forgot password,
  "Check your inbox" state with Resend. Verify / set-password links open the web.
- Google: `@react-native-google-signin/google-signin` (config plugin; `webClientId` +
  `iosClientId` from `EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID` / `EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID`).
  Native module → **not in Expo Go**: the button is not rendered when `isExpoGo()` (CLAUDE.md
  exception list gains this second entry).
- Apple: `expo-apple-authentication`, iOS only, `ios.usesAppleSignIn: true` in app config;
  rendered only when `isAvailableAsync()` is true.
- Header shows `displayName`; Account section with Delete account (same confirm).
- Admin screen mirrors the web admin; `PasswordRevealSheet` and `credentials.ts` are deleted.

## 7. Env vars

Backend: `AUTH_ADMINS` (emails), `FRONTEND_ORIGIN` (comma-separated; the FIRST entry must be
the production web origin — it is the base of every emailed link, and the list is the CORS
allow-list), `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `GOOGLE_CLIENT_IDS`
(web, Android, iOS client ids), `APPLE_CLIENT_IDS` (the bundle id only in production),
`APPLE_TEAM_ID`, `APPLE_KEY_ID`, `APPLE_PRIVATE_KEY` (needed for App Store review: deleting
an account must revoke the Apple grant). Dev backends only: `APPLE_ALLOW_EXPO_GO=1` — never
in production. Removed: `AUTH_USERS`, `FE_BASE_URL`, `SMTP_FROM`.
Web: `NEXT_PUBLIC_GOOGLE_CLIENT_ID`. Mobile: `EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID`,
`EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID`.

## 8. Testing

Backend (pytest, existing fixtures):
- signup → verify → login; login before verify → 403 `email_not_verified`.
- Enumeration: signup / forgot / resend give identical responses for existing vs unknown emails.
- Tokens: expired, reused, wrong purpose, newer-token-invalidates-older → `invalid_or_expired_link`.
- Cooldown: second request inside 60 s sends nothing.
- Google/Apple: tokens signed with a test RSA key served by a stubbed JWKS client — wrong
  `aud`, wrong `iss`, expired, Google `email_verified=false` rejected; the 3 resolution paths;
  **takeover case**: unverified password account + Google sign-in with the same email → linked,
  `password_hash` NULL, old password no longer logs in.
- Deletion removes rows from every user-owned table; table-list completeness test; last-admin guard.
- Seeding from `AUTH_ADMINS` emails; no `SET role` in any router (existing).
- Auth-gate sweep passes with the new public routes.
- SMTP unconfigured → 503; send failure → 502 and token row removed.

Web/mobile: component tests for each new screen and the admin invite/reset flow, provider SDKs
mocked; `npm run verify` (mobile) and lint/tsc/tests (web) green.

Manual (user, on devices): Google on web, Android dev build, iOS dev build; Apple on iOS;
real Gmail delivery of all four emails.

## 9. Rollout order

One order, used everywhere (this spec, the reset script, CLAUDE.md):

1. Set the backend env vars on Vercel (`FRONTEND_ORIGIN` with the production web origin first,
   `SMTP_*`, `AUTH_ADMINS`, `GOOGLE_CLIENT_IDS`, `APPLE_*`) and the web's
   `NEXT_PUBLIC_GOOGLE_CLIENT_ID`. Before production, deploy a Vercel PREVIEW of the backend
   with real SMTP and confirm a forgot-password email actually arrives (background sends after
   the response are unverified on Vercel's Python runtime).
2. Run `scripts/reset_users_for_email_auth.py` against production (from the email-auth code).
3. Deploy the backend.
4. Deploy the web; the admin signs in with Google or Forgot password.
Then a mobile dev build for device tests. Sign-in is broken between steps 2 and 4 (old
clients use username login) — accepted, since every account is reset anyway.

## 10. User setup checklist

Gmail app password; Google Cloud OAuth consent screen + 3 OAuth clients (web with the FE origin,
Android with the EAS SHA-1, iOS with the bundle id); Apple Developer: enable Sign in with Apple
on `com.markbotros.borsacompass`, create a Sign in with Apple key (`.p8`); set env vars on Vercel
(backend + web) and EAS (mobile).

## Changes during implementation

Review of the backend tasks changed the behaviour below; it supersedes the text above where they differ. (The session response shape correction `{token, user}` -> `{access_token, token_type, user}` is already applied in §3.)

1. **Sign-up credentials live on the verify link, not the account.** `auth_tokens` gained `pending_password_hash` and `pending_name`. A new email+password sign-up creates the user with `password_hash` NULL; the password and name are applied only when that link is used (`verify-email`). This replaces §3's "existing unverified email: replace its password hash and name":
   - Re-signing-up for an unverified email never changes the account: outside the 60 s cooldown it emails a NEW verify link (older unused ones are deleted) carrying the new pending password; inside the cooldown nothing happens; a disabled account gets nothing.
   - `resend-verification` re-sends the pending password of the newest live verify link; with none live it sends nothing.
   - `verify-email` on an already-verified account consumes the link and changes nothing.
   - Login of an unverified account returns `403 email_not_verified` only if the password matches the live pending password, otherwise `401`. Unknown emails and passwordless accounts still run a dummy bcrypt check (timing).
   - `set-password` (reset or invite) also deletes any unused verify links, so an old sign-up link can never overwrite a password set later.
   - Accepted residual: if someone signs up with your email and you click that unsolicited confirmation, you confirm their password; resetting your password evicts them.
2. **Password set/reset revokes sessions.** `users.sessions_valid_after` is stamped by `set-password`; `get_current_user` rejects tokens whose `iat` is earlier (`401 "Session expired — sign in again"`).
3. **Background email for enumeration-sensitive routes.** `forgot-password` and `resend-verification` send through `BackgroundMailer`, so known and unknown emails respond in the same time; SMTP failures there are logged, not `502`. `signup`, admin invite and admin send-reset stay synchronous (`502` on failure).
4. **SMTP** uses STARTTLS with a certificate-verifying context; a non-numeric `SMTP_PORT` answers `503`.
5. **OAuth hardening.** Tokens are decoded with 60 s clock leeway; JWKS fetch failures answer `503 "<Provider> sign-in is temporarily unavailable"` (not `401`), with a 5 s JWKS timeout. Apple's email is used only when Apple marks it `email_verified` (otherwise `sub`-only; an unknown `sub` without an email is `400`). Disabled accounts are rejected before any linking write. Linking an unverified account also deletes its pending verify links and takes the provider's name.
6. **Account deletion** lives in `AccountStore.delete_account` (one transaction over `USER_OWNED_TABLES`, then `users`) and `flows.delete_account`, shared by `DELETE /api/auth/me` and admin `DELETE /api/users/{id}`. The `SET role` grep test now scans every file in `app/routers/`; `users.py` writes no password (test-pinned).
7. **Final review fixes.**
   - Apple: `host.exp.Exponent` is rejected unless `APPLE_ALLOW_EXPO_GO=1` (dev only); exchange/revoke use the first non-Expo-Go audience (see §3).
   - Email env: `FE_BASE_URL` and `SMTP_FROM` are gone; links use the first `FRONTEND_ORIGIN` entry (see §3, §7).
   - Sign-up creates the account with `name = ""`; the name arrives only through `pending_name` on verify, and verify emails greet plain "Hi,". Failure logs name the recipient's domain only.
   - Re-inviting an account nobody set up resends the invite (see §4).
   - **Abuse limits** (`app/core/limits.py`, counters in a `rate_limits (key, window_start, count)` table bumped by one atomic upsert per fixed window): signup / resend-verification / forgot-password allow 5 requests per IP (first `X-Forwarded-For` entry) per 10 minutes, then answer `check_email` and send nothing; every send attempt (all purposes) counts against `email:global`, 300 per UTC day — over it nothing is sent, public routes still answer `check_email`, admin routes `429 "Daily email limit reached — try again tomorrow."`; login allows 10 failed attempts per IP per 15 minutes, then `429 "Too many attempts — try again in a few minutes."` before checking the password.
   - **Pruning** (`limits.prune_auth`, best-effort from the nightly `POST /api/pe/refresh`): links used or expired over a day ago, rate-limit windows over two days old, and accounts over a week old that are unverified, passwordless, unlinked, role user, without a live link and owning no rows.
   - Not yet verified (I3): background sends after the response on Vercel's Python runtime. Verify a forgot-password email arrives on a Vercel preview deploy before production (rollout step 1).
