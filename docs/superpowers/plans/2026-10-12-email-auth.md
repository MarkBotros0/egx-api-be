# Email Accounts + Google/Apple Sign-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace admin-made username accounts with email accounts: open sign-up with email verification, Google (web + mobile) and Apple (iOS) sign-in, SMTP emails for verification / invites / password reset, self-service account deletion.

**Architecture:** The backend keeps issuing its own HS256 JWT. A new `AccountStore` owns all `users`/`auth_tokens` SQL; `auth_flows.py` holds every flow as plain functions over a store + mailer, so tests run against an in-memory store and an outbox mailer. Google/Apple tokens are verified with PyJWT against the providers' JWKS. The web adds 5 pages and a GIS button; mobile adds sign-up/forgot screens, native Google (dev build only) and Apple (iOS) buttons, and an Account screen.

**Tech Stack:** FastAPI + psycopg + PyJWT[crypto] + smtplib (backend); Next.js 14 (web); Expo SDK 57, `@react-native-google-signin/google-signin`, `expo-apple-authentication`, jest-expo + RNTL (mobile).

**Spec:** `egx-api-be/docs/superpowers/specs/2026-10-12-email-auth-design.md`

## Global Constraints

- Three repos, one branch `email-auth` in each, in worktrees under each repo's `.worktrees/email-auth`: backend `/home/mark/projects/egx-api/egx-api-be` (base `master`), web `/home/mark/projects/egx-api/egx-api-fe` (base `master`), mobile `/home/mark/projects/egx-api/egx-api-mobile` (base `main`).
- `egx-api-fe/CLAUDE.md` and `egx-api-be/CLAUDE.md` must stay byte-identical: edit one, `cp` to the other, `cmp`, commit in both.
- Emails are stored trimmed + lower-cased; ≤ 254 chars; must match `^[^@\s]+@[^@\s]+\.[^@\s]+$`. Passwords 8–128 chars, never trimmed. Names: whitespace collapsed, ≤ 80 chars.
- Link-token lifetimes exactly: `verify` 24 h, `reset` 1 h, `invite` 7 days. Raw token `secrets.token_urlsafe(32)`, stored only as SHA-256 hex.
- Email cooldown: at most one email per user per **60 s** (`users.last_email_at`); inside it, send nothing and return the normal response.
- Error contract: login wrong creds → `401 "Invalid email or password"`; unverified → `403 {"detail":"email_not_verified"}`; disabled → `403 "This account has been disabled."`; bad/used/expired link → `400 {"detail":"invalid_or_expired_link"}`; SMTP/FE_BASE_URL missing → `503 "Email is not configured"`; SMTP failure → `502 "Could not send email"`; provider not configured → `503`.
- Session responses keep the EXISTING shape: `{"access_token": <jwt>, "token_type": "bearer", "user": {id, email, name, role}}` (see Ruling 1).
- No API route writes `role`; the string `SET role` must not appear in `app/routers/`.
- Apple sign-in: iOS app only. Google: web + Android + iOS. No Firebase.
- Backend tests never touch the network; Postgres only in the opt-in `tests/test_accounts_pg.py` (skipped unless `TEST_DATABASE_URL`).
- Backend venv: `/tmp/claude-1000/-home-mark-projects-egx-api/f3f9fba8-4a93-4720-b415-a22413998dfa/scratchpad/bevenv/bin/python` (if missing: `python3 -m venv <path> && <path>/bin/pip install -r requirements.txt pytest`). Run single files directly; the FULL suite (~2.5 min) MUST run in tmux: `tmux kill-session -t egx-be-tests 2>/dev/null; tmux new -d -s egx-be-tests "cd <worktree> && <venv python> -m pytest -q 2>&1 | tee ~/logs/egx-be-tests.log"`.
- Web checks: `npx tsc --noEmit` and `npm run lint`; `npm run build` in tmux (`egx-fe-build`, `~/logs/egx-fe-build.log`). The web has no unit-test runner — do not add one.
- Mobile: deps ONLY via `npx expo install <pkg>`; `npm run verify` green before every mobile commit (run in tmux `egx-mobile-verify`, log `~/logs/egx-mobile-verify.log`); `expo lint` may rewrite `tsconfig.json` → `git checkout tsconfig.json`. `src/shared/` is generated: change `egx-api-fe/src/app/lib/`, then `npm run sync:shared` in mobile. Mobile UI rules from `egx-api-mobile/CLAUDE.md` apply (≥44pt targets, `Screen` owns `px-4`, ternaries not `&&` with strings/numbers, hexes only from `COLORS`).
- The mobile app must still RUN in Expo Go: Google button not rendered there.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV
  ```
- Never push. Never bare `git stash`. Never `pkill -f`. Never run the reset script against any database.

## Rulings made while planning (deviations from / additions to the spec text)

1. **Session response shape** stays `{access_token, token_type, user}` (spec §3 wrote `{token, user}`); both clients already parse `access_token`. Task 7 corrects the spec.
2. **Testability:** flows take an `AccountStore`-shaped object and a `Mailer`; tests use `tests/account_fakes.py` (`MemoryAccountStore`, `OutboxMailer`). Real SQL is covered by FakeDB parameter tests plus an opt-in Postgres test run once via Docker in Task 1.
3. **Legacy table guard:** if `users` still has a `username` column, `seed_users_from_env` prints a warning and returns; the app needs the reset script. `init_db` does not raise (market data must keep working).
4. **JWT** drops the `username` claim: payload is `{sub, iat, exp}`; `get_current_user` requires only `sub`.
5. **Middleware (web):** `/login`, `/signup`, `/forgot-password` bounce signed-in visitors home; `/verify-email` and `/set-password` are reachable signed in OR out (a link from an email must always work).
6. **`isExpoGo()` moves** from `src/features/alerts/push.ts` to `src/lib/runtime.ts` (push.ts re-exports it) so auth code doesn't import `expo-notifications`.
7. **Google native module is required lazily** (`require()` inside the press handler path, behind `!isExpoGo()`), so importing the login screen in Expo Go never touches it.
8. **`app.config.ts`** is added in mobile to inject the Google iOS URL scheme from `GOOGLE_IOS_URL_SCHEME` at build time; without it the plugin is skipped.
9. **Admin "Send reset"** sends `invite` when the account has no password and is unverified, else `reset` (spec §4).
10. **Account page/screen route is `/account`** on both clients; mobile reaches it from a header `person-circle-outline` button; web from the Navbar name.

## Review Focus

1. Pre-registration takeover: an unverified password account for `victim@x.com`, then a Google sign-in for the same verified email → account linked, `password_hash` cleared, old password rejected. Pinned in Task 5.
2. Enumeration: `signup`, `resend-verification`, `forgot-password` return byte-identical bodies for unknown, unverified and verified emails. Pinned in Task 4.
3. A link clicked twice, or after a newer link was sent, must fail with `invalid_or_expired_link`, and a `reset` token must not work on `verify-email`. Pinned in Task 4.
4. Email typed with capitals/spaces (`"  Mark@Gmail.COM "`) must sign in to the account created as `mark@gmail.com` — on every route (signup, login, forgot, google link-by-email, admin invite). Pinned in Tasks 1, 4, 5, 6.
5. Deleting an account must remove rows from EVERY user-owned table, including ones added later. Pinned by the completeness test in Task 1 and the deletion test in Task 6.

---

## File map

**Backend (`egx-api-be`)**
- Create `app/core/accounts.py` — validation, `Account`, `AccountStore`, link tokens, `USER_OWNED_TABLES` (Task 1).
- Modify `app/core/db.py` — new `users`, `auth_tokens` (Task 1).
- Create `app/core/email.py` — `Mailer`, `SmtpMailer`, templates, `get_mailer` (Task 2).
- Modify `app/core/auth.py` — `CurrentUser(email,name)`, token, seeding by email, legacy guard; later remove `generate_password` (Tasks 3, 6).
- Create `scripts/reset_users_for_email_auth.py` (Task 3).
- Create `app/core/auth_flows.py` — every flow (Tasks 4–6).
- Rewrite `app/routers/auth.py` (Tasks 4–6), `app/routers/users.py` (Task 6).
- Create `app/core/oauth.py` — Google/Apple verification + Apple token exchange/revoke (Task 5).
- Tests: `tests/account_fakes.py`, `tests/test_accounts_store.py`, `tests/test_accounts_pg.py`, `tests/test_email.py`, `tests/test_auth_flows.py`, `tests/test_oauth.py`, `tests/test_auth_routes.py`; update `tests/test_users_and_roles.py`, `tests/test_auth_gate.py`, `tests/test_alerts_router.py`.
- `requirements.txt` (`pyjwt[crypto]`), `CLAUDE.md` (Task 7).

**Web (`egx-api-fe`)**
- `src/app/lib/auth.ts`, `src/app/lib/api.ts` (Task 8).
- `src/app/components/AuthProvider.tsx`, `Navbar.tsx`, `src/app/login/page.tsx`, create `src/app/components/AuthCard.tsx` (Task 8).
- Pages `signup`, `forgot-password`, `verify-email`, `set-password`; modify `src/middleware.ts` (Task 9).
- Create `src/app/components/GoogleSignInButton.tsx` (Task 10).
- Create `src/app/account/page.tsx` (Task 11).
- Admin: modify `src/app/admin/page.tsx`, `AdminUsersTable.tsx`; replace `CreateUserModal.tsx` with `InviteUserModal.tsx`; delete `PasswordRevealDialog.tsx` (Task 8).

**Mobile (`egx-api-mobile`)**
- `src/shared/*` (sync), `src/auth/tokenStore.ts`, `src/auth/AuthProvider.tsx`, `src/features/auth/LoginForm.tsx`, admin feature (Task 12).
- Create `src/features/auth/SignupForm.tsx`, `ForgotPasswordForm.tsx`, `CheckInbox.tsx`, `AuthShell.tsx`, routes `src/app/signup.tsx`, `src/app/forgot-password.tsx`; modify `RootNavigator.tsx`, `src/app/login.tsx` (Task 13).
- Create `src/lib/runtime.ts`, `src/features/auth/GoogleButton.tsx`, `AppleButton.tsx`, `SocialButtons.tsx`, `app.config.ts`; modify `app.json`, `push.ts`, `.env.example` (Task 14).
- Create `src/features/account/AccountScreen.tsx`, `AccountHeaderButton.tsx`, route `src/app/account.tsx` (Task 15).
- `CLAUDE.md`, `docs/release.md` (Task 16). Final cross-repo verification: Task 17.

---

## Task 0: Worktrees

- [ ] **Step 1: Create the three worktrees**

```bash
for r in egx-api-be egx-api-fe egx-api-mobile; do
  cd /home/mark/projects/egx-api/$r
  grep -qx '.worktrees/' .gitignore || { echo '.worktrees/' >> .gitignore; git add .gitignore; git commit -qm "chore: ignore .worktrees

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"; }
  git worktree add .worktrees/email-auth -b email-auth
done
# sync-shared.mjs reads ../egx-api-fe relative to the mobile root; point that at the fe WORKTREE
ln -s /home/mark/projects/egx-api/egx-api-fe/.worktrees/email-auth /home/mark/projects/egx-api/egx-api-mobile/.worktrees/egx-api-fe
(cd /home/mark/projects/egx-api/egx-api-mobile/.worktrees/email-auth && npm ci)
(cd /home/mark/projects/egx-api/egx-api-fe/.worktrees/email-auth && npm ci)
``` Expected: three worktrees on branch `email-auth`. All later paths are relative to the matching worktree.

---

## Task 1: Account store and schema (backend)

**Files:**
- Create: `app/core/accounts.py`
- Modify: `app/core/db.py` (the `users` block at the top of `init_db`)
- Create: `tests/account_fakes.py`, `tests/test_accounts_store.py`, `tests/test_accounts_pg.py`

**Interfaces:**
- Produces: `normalize_email(raw) -> str`, `clean_password(raw) -> str`, `clean_name(raw) -> str`, `iso(dt) -> str`, `hash_link_token(raw) -> str`, constants `PURPOSE_VERIFY|RESET|INVITE`, `TOKEN_LIFETIMES`, `EMAIL_COOLDOWN`, `USER_OWNED_TABLES`, `PROVIDERS`; `Account` (frozen dataclass, `.methods`), `public_user(a) -> dict`; `AccountStore(db)` with `by_id`, `by_email`, `by_provider(provider, sub)`, `list_all()`, `holdings_counts()`, `count_active_admins()`, `create(*, email, name, password_hash, email_verified, now, google_sub=None, apple_sub=None) -> Account`, `update(user_id, **fields)`, `issue_token(user_id, purpose, now) -> str`, `discard_token(raw)`, `consume_token(raw, purposes, now) -> Optional[str]`, `delete_account(user_id)`, `users_table_is_legacy() -> bool`.
- Test fakes: `MemoryAccountStore` (same methods), `OutboxMailer` (`.sent: list[dict]`, `.fail: bool`).

- [ ] **Step 1: Write the fakes**

`tests/account_fakes.py`:
```python
"""
In-memory stand-ins for AccountStore and the Mailer, so flow tests exercise
real behaviour (tokens expire, get consumed, rows get deleted) without
Postgres or SMTP. Imported top-level like route_table.py / alert_fakes.py.
"""

from __future__ import annotations

import secrets
import uuid
from dataclasses import replace
from typing import Optional

from app.core.accounts import (
    TOKEN_LIFETIMES,
    USER_OWNED_TABLES,
    Account,
    hash_link_token,
    iso,
)
from app.core.email import EmailSendFailed


class MemoryAccountStore:
    def __init__(self):
        self.users: dict[str, Account] = {}
        self.tokens: dict[str, dict] = {}          # token_hash -> row
        self.holdings: dict[str, int] = {}
        self.deleted_from: list[tuple[str, str]] = []  # (table, user_id)

    # -- reads --
    def by_id(self, user_id):
        return self.users.get(user_id)

    def by_email(self, email):
        return next((u for u in self.users.values() if u.email == email), None)

    def by_provider(self, provider, sub):
        attr = f"{provider}_sub"
        return next((u for u in self.users.values() if getattr(u, attr) == sub), None)

    def list_all(self):
        return sorted(self.users.values(), key=lambda u: u.created_at)

    def holdings_counts(self):
        return dict(self.holdings)

    def count_active_admins(self):
        return sum(1 for u in self.users.values() if u.role == "admin" and u.is_active)

    def users_table_is_legacy(self):
        return False

    # -- writes --
    def create(self, *, email, name, password_hash, email_verified, now,
               google_sub=None, apple_sub=None, role="user"):
        acc = Account(
            id=str(uuid.uuid4()), email=email, name=name, password_hash=password_hash,
            email_verified=email_verified, google_sub=google_sub, apple_sub=apple_sub,
            apple_refresh_token=None, role=role, is_active=True, last_email_at=None,
            created_at=iso(now),
        )
        self.users[acc.id] = acc
        return acc

    def update(self, user_id, **fields):
        self.users[user_id] = replace(self.users[user_id], **fields)

    def set_role(self, user_id, role):
        """Test-only: the real store has no way to write a role."""
        self.users[user_id] = replace(self.users[user_id], role=role)

    def issue_token(self, user_id, purpose, now):
        for h, row in list(self.tokens.items()):
            if row["user_id"] == user_id and row["purpose"] == purpose and row["used_at"] is None:
                del self.tokens[h]
        raw = secrets.token_urlsafe(32)
        self.tokens[hash_link_token(raw)] = {
            "user_id": user_id, "purpose": purpose,
            "expires_at": iso(now + TOKEN_LIFETIMES[purpose]), "used_at": None,
            "created_at": iso(now),
        }
        return raw

    def discard_token(self, raw):
        self.tokens.pop(hash_link_token(raw), None)

    def consume_token(self, raw, purposes, now):
        row = self.tokens.get(hash_link_token(raw or ""))
        if (not row or row["used_at"] is not None or row["expires_at"] <= iso(now)
                or row["purpose"] not in purposes):
            return None
        row["used_at"] = iso(now)
        return row["user_id"]

    def delete_account(self, user_id):
        for t in USER_OWNED_TABLES:
            self.deleted_from.append((t, user_id))
        self.tokens = {h: r for h, r in self.tokens.items() if r["user_id"] != user_id}
        self.users.pop(user_id, None)


class OutboxMailer:
    def __init__(self):
        self.sent: list[dict] = []
        self.fail = False

    def send(self, to, subject, text, html):
        if self.fail:
            raise EmailSendFailed("boom")
        self.sent.append({"to": to, "subject": subject, "text": text, "html": html})

    def last_link(self) -> Optional[str]:
        import re
        m = re.search(r"https?://\S+token=([\w-]+)", self.sent[-1]["text"])
        return m.group(1) if m else None
```

NOTE: `app.core.email` is created in Task 2. Until then, put a temporary minimal module so imports work — create `app/core/email.py` now with only:
```python
class EmailSendFailed(Exception):
    """SMTP refused or failed; routes turn this into 502."""
```
Task 2 extends this file.

- [ ] **Step 2: Write the failing store tests**

`tests/test_accounts_store.py`:
```python
"""
AccountStore + validation. SQL is checked against a scriptable FakeDB (no
Postgres); real-Postgres behaviour is covered by test_accounts_pg.py.

Run from egx-api-be:  python -m pytest tests/test_accounts_store.py -v
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from alert_fakes import FakeDB
from app.core.accounts import (
    USER_OWNED_TABLES,
    AccountStore,
    clean_name,
    clean_password,
    hash_link_token,
    iso,
    normalize_email,
)

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("raw,expected", [
    ("  Mark@Gmail.COM ", "mark@gmail.com"),
    ("a.b+c@sub.example.org", "a.b+c@sub.example.org"),
])
def test_emails_are_trimmed_and_lowercased(raw, expected):
    assert normalize_email(raw) == expected


@pytest.mark.parametrize("bad", ["", "no-at-sign", "a@b", "a b@c.com", "@x.com", "x@" + "a" * 250 + ".com", None])
def test_invalid_emails_are_rejected(bad):
    with pytest.raises(HTTPException) as e:
        normalize_email(bad)
    assert e.value.status_code == 400


def test_passwords_are_length_checked_and_never_trimmed():
    assert clean_password("  spaced  ") == "  spaced  "
    for bad in ("short", "x" * 129, "", None):
        with pytest.raises(HTTPException):
            clean_password(bad)


def test_names_collapse_whitespace_and_cap_at_80():
    assert clean_name("  Mark   Botros ") == "Mark Botros"
    assert len(clean_name("x" * 200)) == 80
    assert clean_name(None) == ""


def test_iso_is_utc_with_z():
    assert iso(NOW) == "2026-10-12T09:00:00Z"


def test_by_email_queries_the_normalised_value():
    db = FakeDB()
    AccountStore(db).by_email("mark@gmail.com")
    sql, params = db.log[-1]
    assert "WHERE email = %s" in sql and params == ("mark@gmail.com",)


def test_by_provider_rejects_unknown_provider_columns():
    with pytest.raises(ValueError):
        AccountStore(FakeDB()).by_provider("facebook; DROP TABLE users", "x")


def test_update_refuses_role_and_unknown_fields():
    store = AccountStore(FakeDB())
    for bad in ({"role": "admin"}, {"email": "x@y.z"}, {"nope": 1}):
        with pytest.raises(ValueError):
            store.update("u1", **bad)


def test_issue_token_stores_only_the_hash_and_replaces_older_unused():
    db = FakeDB()
    raw = AccountStore(db).issue_token("u1", "reset", NOW)
    stmts = [(s, p) for s, p in db.log]
    delete = next(p for s, p in stmts if s.startswith("DELETE FROM auth_tokens"))
    assert delete == ("u1", "reset")
    insert = next(p for s, p in stmts if "INSERT INTO auth_tokens" in s)
    assert insert[0] == hash_link_token(raw) and raw not in insert
    assert insert[3] == "2026-10-12T10:00:00Z"   # reset = 1 h


def test_consume_token_is_one_atomic_update():
    db = FakeDB([("UPDATE auth_tokens", [("u1",)])])
    assert AccountStore(db).consume_token("raw", {"verify"}, NOW) == "u1"
    sql, params = db.log[-1]
    assert "used_at IS NULL" in sql and "expires_at > %s" in sql and "RETURNING user_id" in sql
    assert params[1] == hash_link_token("raw") and params[3] == ["verify"]


def test_consume_token_returns_none_when_nothing_matched():
    assert AccountStore(FakeDB()).consume_token("raw", {"verify"}, NOW) is None


def test_delete_account_clears_every_owned_table_then_the_user():
    db = FakeDB()
    AccountStore(db).delete_account("u1")
    tables = [re.match(r"DELETE FROM (\w+)", s).group(1) for s, _ in db.log]
    assert tables == list(USER_OWNED_TABLES) + ["users"]
    # Sales and dividends go before holdings, as the undo path depends on.
    assert tables.index("portfolio_sales") < tables.index("portfolio")


def test_user_owned_tables_matches_every_user_id_table_in_init_db():
    """A table added later with a user_id column must be added to USER_OWNED_TABLES."""
    src = (Path(__file__).resolve().parents[1] / "app" / "core" / "db.py").read_text(encoding="utf-8")
    bodies = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\s*\)\s*\n?\s*\"\"\"", src, re.S)
    owned = {name for name, body in bodies if re.search(r"^\s*user_id\s+TEXT", body, re.M)}
    assert owned, "regex found no tables — fix the test, not the list"
    assert owned == set(USER_OWNED_TABLES)


def test_methods_lists_the_ways_an_account_can_sign_in():
    from account_fakes import MemoryAccountStore
    s = MemoryAccountStore()
    a = s.create(email="a@b.co", name="", password_hash="h", email_verified=True, now=NOW, google_sub="g")
    assert a.methods == ["password", "google"]
```
(`FakeDB` from `tests/alert_fakes.py` has no `transaction()`; add one there in this step, identical to the one in `tests/test_users_and_roles.py`:)
```python
    @contextmanager
    def transaction(self):
        outer = self

        class _Tx:
            def execute(self, sql, params=()):
                outer.log.append((sql, params))
                return _Result(outer._answer(sql))

        yield _Tx()
```
(`contextmanager` is already imported in that file.)

- [ ] **Step 3: Run to verify failure**

Run: `<venv python> -m pytest tests/test_accounts_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.accounts'`.

- [ ] **Step 4: Write `app/core/accounts.py`**

```python
"""
Email accounts: input validation, the `users` / `auth_tokens` SQL, and the
list of tables a user owns.

Flows (sign-up, emailed links, Google/Apple) live in core/auth_flows.py and
talk to an AccountStore-shaped object, so their tests run against
tests/account_fakes.MemoryAccountStore instead of Postgres.

Link tokens: the raw token goes into the emailed URL and is NEVER stored —
only its SHA-256. Issuing a new token of a purpose deletes the user's older
unused ones, so only the newest link works. Consuming is one UPDATE …
RETURNING, so a link clicked twice at once still works exactly once.
"""

import hashlib
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from fastapi import HTTPException

PURPOSE_VERIFY = "verify"
PURPOSE_RESET = "reset"
PURPOSE_INVITE = "invite"
TOKEN_LIFETIMES = {
    PURPOSE_VERIFY: timedelta(hours=24),
    PURPOSE_RESET: timedelta(hours=1),
    PURPOSE_INVITE: timedelta(days=7),
}
EMAIL_COOLDOWN = timedelta(seconds=60)

PASSWORD_MIN = 8
PASSWORD_MAX = 128
NAME_MAX = 80
EMAIL_MAX = 254
PROVIDERS = ("google", "apple")

# Every table with a user_id column. Deleting an account clears these in this
# order (sales/dividends before holdings, as the undo path expects), then the
# users row. tests/test_accounts_store.py fails if init_db grows a user_id
# table that is missing here.
USER_OWNED_TABLES = (
    "portfolio_sales",
    "portfolio_dividends",
    "portfolio",
    "watchlist",
    "user_settings",
    "push_devices",
    "alert_preferences",
    "alert_log",
    "auth_tokens",
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(raw) -> str:
    email = (raw or "").strip().lower()
    if len(email) > EMAIL_MAX or not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return email


def clean_password(raw) -> str:
    password = raw or ""
    if not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be {PASSWORD_MIN}–{PASSWORD_MAX} characters.",
        )
    return password


def clean_name(raw) -> str:
    return " ".join((raw or "").split())[:NAME_MAX]


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_link_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Account:
    id: str
    email: str
    name: str
    password_hash: Optional[str]
    email_verified: bool
    google_sub: Optional[str]
    apple_sub: Optional[str]
    apple_refresh_token: Optional[str]
    role: str
    is_active: bool
    last_email_at: Optional[str]
    created_at: str

    @property
    def methods(self) -> list:
        out = []
        if self.password_hash:
            out.append("password")
        if self.google_sub:
            out.append("google")
        if self.apple_sub:
            out.append("apple")
        return out


def public_user(a: Account) -> dict:
    return {"id": a.id, "email": a.email, "name": a.name, "role": a.role}


_COLUMNS = (
    "id, email, name, password_hash, email_verified, google_sub, apple_sub, "
    "apple_refresh_token, role, is_active, last_email_at, created_at"
)
_UPDATABLE = {
    "name", "password_hash", "email_verified", "google_sub", "apple_sub",
    "apple_refresh_token", "is_active", "last_email_at",
}


def _row(row) -> Optional[Account]:
    if not row:
        return None
    return Account(
        id=row[0], email=row[1], name=row[2] or "", password_hash=row[3],
        email_verified=bool(row[4]), google_sub=row[5], apple_sub=row[6],
        apple_refresh_token=row[7], role=row[8] or "user", is_active=bool(row[9]),
        last_email_at=row[10], created_at=row[11],
    )


def _provider_column(provider: str) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider!r}")
    return f"{provider}_sub"


class AccountStore:
    """All SQL for `users` and `auth_tokens`. The pool runs autocommit."""

    def __init__(self, db):
        self.db = db

    # -- reads --
    def by_id(self, user_id: str) -> Optional[Account]:
        return _row(self.db.execute(
            f"SELECT {_COLUMNS} FROM users WHERE id = %s", (user_id,)).fetchone())

    def by_email(self, email: str) -> Optional[Account]:
        return _row(self.db.execute(
            f"SELECT {_COLUMNS} FROM users WHERE email = %s", (email,)).fetchone())

    def by_provider(self, provider: str, sub: str) -> Optional[Account]:
        col = _provider_column(provider)
        return _row(self.db.execute(
            f"SELECT {_COLUMNS} FROM users WHERE {col} = %s", (sub,)).fetchone())

    def list_all(self) -> list:
        rows = self.db.execute(
            f"SELECT {_COLUMNS} FROM users ORDER BY created_at").fetchall()
        return [_row(r) for r in rows]

    def holdings_counts(self) -> dict:
        rows = self.db.execute(
            "SELECT user_id, COUNT(*) FROM portfolio GROUP BY user_id").fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def count_active_admins(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = TRUE").fetchone()
        return int(row[0]) if row else 0

    def users_table_is_legacy(self) -> bool:
        """True while `users` still has the pre-email `username` column."""
        row = self.db.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'users' AND column_name = 'username'").fetchone()
        return bool(row)

    # -- writes --
    def create(self, *, email: str, name: str, password_hash: Optional[str],
               email_verified: bool, now: datetime, google_sub: Optional[str] = None,
               apple_sub: Optional[str] = None) -> Account:
        user_id = str(uuid.uuid4())
        created = iso(now)
        self.db.execute(
            "INSERT INTO users (id, email, name, password_hash, email_verified, "
            "google_sub, apple_sub, role, is_active, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, 'user', TRUE, %s)",
            (user_id, email, name, password_hash, email_verified, google_sub,
             apple_sub, created),
        )
        self.db.commit()
        return Account(
            id=user_id, email=email, name=name, password_hash=password_hash,
            email_verified=email_verified, google_sub=google_sub, apple_sub=apple_sub,
            apple_refresh_token=None, role="user", is_active=True, last_email_at=None,
            created_at=created,
        )

    def update(self, user_id: str, **fields) -> None:
        """Set columns on one user. `role` is deliberately not updatable."""
        if not fields:
            return
        bad = set(fields) - _UPDATABLE
        if bad:
            raise ValueError(f"not updatable: {sorted(bad)}")
        assignments = ", ".join(f"{k} = %s" for k in fields)
        self.db.execute(
            f"UPDATE users SET {assignments} WHERE id = %s",
            (*fields.values(), user_id),
        )
        self.db.commit()

    def issue_token(self, user_id: str, purpose: str, now: datetime) -> str:
        raw = secrets.token_urlsafe(32)
        with self.db.transaction() as tx:
            tx.execute(
                "DELETE FROM auth_tokens WHERE user_id = %s AND purpose = %s AND used_at IS NULL",
                (user_id, purpose),
            )
            tx.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, purpose, expires_at, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (hash_link_token(raw), user_id, purpose,
                 iso(now + TOKEN_LIFETIMES[purpose]), iso(now)),
            )
        return raw

    def discard_token(self, raw: str) -> None:
        self.db.execute("DELETE FROM auth_tokens WHERE token_hash = %s", (hash_link_token(raw),))
        self.db.commit()

    def consume_token(self, raw: str, purposes: Iterable[str], now: datetime) -> Optional[str]:
        """Mark a live token used and return its user_id, or None."""
        row = self.db.execute(
            "UPDATE auth_tokens SET used_at = %s WHERE token_hash = %s AND used_at IS NULL "
            "AND expires_at > %s AND purpose = ANY(%s) RETURNING user_id",
            (iso(now), hash_link_token(raw or ""), iso(now), list(purposes)),
        ).fetchone()
        self.db.commit()
        return row[0] if row else None

    def delete_account(self, user_id: str) -> None:
        # No table has a foreign key to users, so nothing cascades — without
        # this the rows survive as invisible orphans. One transaction, so a
        # half-deleted user is impossible.
        with self.db.transaction() as tx:
            for table in USER_OWNED_TABLES:
                tx.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM users WHERE id = %s", (user_id,))
```

- [ ] **Step 5: Change the schema in `app/core/db.py`**

Replace the `CREATE TABLE IF NOT EXISTS users (...)` statement AND the comment + two `ALTER TABLE users ADD COLUMN IF NOT EXISTS role/is_active` lines after it with:
```python
    # Email accounts (2026-10-12). A database that still has the old
    # username-based table keeps it — CREATE IF NOT EXISTS skips — and
    # seed_users_from_env refuses to run until
    # scripts/reset_users_for_email_auth.py has replaced it.
    #
    # `role` is NOT settable through any API: it is stamped from the
    # AUTH_ADMINS env var at boot (core/auth.seed_users_from_env).
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            password_hash TEXT,
            email_verified BOOLEAN NOT NULL DEFAULT FALSE,
            google_sub TEXT UNIQUE,
            apple_sub TEXT UNIQUE,
            apple_refresh_token TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            last_email_at TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # Emailed links (verify / reset / invite). Only the SHA-256 of the raw
    # token is stored — see core/accounts.py.
    db.execute("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            purpose TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT NOT NULL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id, purpose)")
```

- [ ] **Step 6: Run the store tests**

Run: `<venv python> -m pytest tests/test_accounts_store.py -q`
Expected: PASS. If `test_user_owned_tables_matches_every_user_id_table_in_init_db` reports a mismatch, the regex or list is wrong — print `owned` and fix whichever disagrees with `db.py` (do NOT weaken the assertion).

- [ ] **Step 7: Write the opt-in Postgres test**

`tests/test_accounts_pg.py`:
```python
"""
AccountStore against a REAL Postgres. Skipped unless TEST_DATABASE_URL is set.

    docker run -d --rm --name egx-auth-pg -e POSTGRES_PASSWORD=pw -p 55432:5432 postgres:16
    TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55432/postgres \
        python -m pytest tests/test_accounts_pg.py -v
    docker stop egx-auth-pg

Never point this at a real database: it drops tables.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    from app.core import db as db_mod
    from app.core.accounts import USER_OWNED_TABLES, AccountStore

    monkeypatch.setattr(db_mod, "_pool", None)
    monkeypatch.setattr(db_mod, "_initialized", True)
    d = db_mod._DB(db_mod._get_pool())
    for t in (*USER_OWNED_TABLES, "users"):
        d.execute(f"DROP TABLE IF EXISTS {t}")
    db_mod.init_db(d)
    return AccountStore(d)


def test_create_find_update(store):
    a = store.create(email="a@b.co", name="A", password_hash="h", email_verified=False, now=NOW)
    assert store.by_email("a@b.co").id == a.id
    store.update(a.id, email_verified=True, google_sub="g1")
    assert store.by_provider("google", "g1").email_verified is True
    assert store.users_table_is_legacy() is False


def test_tokens_expire_are_single_use_and_replaced(store):
    a = store.create(email="a@b.co", name="", password_hash=None, email_verified=False, now=NOW)
    old = store.issue_token(a.id, "verify", NOW)
    new = store.issue_token(a.id, "verify", NOW)
    assert store.consume_token(old, {"verify"}, NOW) is None
    assert store.consume_token(new, {"reset"}, NOW) is None
    assert store.consume_token(new, {"verify"}, NOW) == a.id
    assert store.consume_token(new, {"verify"}, NOW) is None
    late = store.issue_token(a.id, "reset", NOW)
    assert store.consume_token(late, {"reset"}, NOW + timedelta(hours=2)) is None


def test_delete_account_removes_tokens_and_row(store):
    a = store.create(email="a@b.co", name="", password_hash=None, email_verified=False, now=NOW)
    store.issue_token(a.id, "verify", NOW)
    store.delete_account(a.id)
    assert store.by_id(a.id) is None
```

- [ ] **Step 8: Run it once against Docker Postgres**

```bash
docker run -d --rm --name egx-auth-pg -e POSTGRES_PASSWORD=pw -p 55432:5432 postgres:16 && sleep 4
TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55432/postgres <venv python> -m pytest tests/test_accounts_pg.py -v
docker stop egx-auth-pg
```
Expected: 3 passed. If Docker is unavailable, record that in the report; the file still skips cleanly (`<venv python> -m pytest tests/test_accounts_pg.py -q` → `3 skipped`).

- [ ] **Step 9: Commit**

```bash
git add app/core/accounts.py app/core/email.py app/core/db.py tests/account_fakes.py tests/alert_fakes.py tests/test_accounts_store.py tests/test_accounts_pg.py
git commit -m "feat(auth): email account store, auth_tokens table, user-owned table list

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 2: Email sending (backend)

**Files:**
- Modify: `app/core/email.py` (created as a stub in Task 1)
- Test: `tests/test_email.py`

**Interfaces:**
- Produces: `EmailNotConfigured(HTTPException 503)`, `EmailSendFailed(Exception)`, `Mailer` protocol `send(to, subject, text, html) -> None`, `SmtpMailer.from_env() -> SmtpMailer` (raises `EmailNotConfigured`), `get_mailer() -> Mailer` (FastAPI dependency), `link(path, token) -> str` (raises `EmailNotConfigured` without `FE_BASE_URL`), templates each returning `(subject, text, html)`: `verify_message(name, url)`, `reset_message(name, url)`, `invite_message(name, inviter, url)`, `already_registered_message(name, login_url, reset_url)`; `login_url() -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_email.py`:
```python
"""
SMTP mailer + templates. smtplib is replaced with a recorder; nothing leaves
the machine.

Run from egx-api-be:  python -m pytest tests/test_email.py -v
"""

import pytest

from app.core import email as email_mod
from app.core.email import (
    EmailNotConfigured,
    EmailSendFailed,
    SmtpMailer,
    already_registered_message,
    invite_message,
    link,
    reset_message,
    verify_message,
)


class _FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, pw):
        self.calls.append(("login", user, pw))

    def send_message(self, msg):
        self.calls.append(("send", msg))


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-pw")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)
    _FakeSMTP.instances.clear()
    monkeypatch.setattr(email_mod.smtplib, "SMTP", _FakeSMTP)


def test_defaults_to_gmail_starttls(smtp_env):
    SmtpMailer.from_env().send("to@x.com", "Subj", "plain", "<p>html</p>")
    s = _FakeSMTP.instances[-1]
    assert (s.host, s.port) == ("smtp.gmail.com", 587)
    assert s.calls[0] == "starttls"
    assert s.calls[1] == ("login", "me@gmail.com", "app-pw")
    msg = s.calls[2][1]
    assert msg["To"] == "to@x.com"
    assert msg["From"] == "Borsa Compass <me@gmail.com>"
    assert msg["Subject"] == "Subj"
    assert msg.is_multipart()


def test_missing_credentials_is_503(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    with pytest.raises(EmailNotConfigured) as e:
        SmtpMailer.from_env()
    assert e.value.status_code == 503


def test_smtp_errors_become_email_send_failed(smtp_env, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(email_mod.smtplib, "SMTP", boom)
    with pytest.raises(EmailSendFailed):
        SmtpMailer.from_env().send("to@x.com", "s", "t", "h")


def test_links_use_fe_base_url_without_double_slash(monkeypatch):
    monkeypatch.setenv("FE_BASE_URL", "https://app.example.com/")
    assert link("/set-password", "abc") == "https://app.example.com/set-password?token=abc"


def test_links_require_fe_base_url(monkeypatch):
    monkeypatch.delenv("FE_BASE_URL", raising=False)
    with pytest.raises(EmailNotConfigured):
        link("/verify-email", "abc")


@pytest.mark.parametrize("build", [
    lambda: verify_message("Mark", "https://x/verify-email?token=t"),
    lambda: reset_message("", "https://x/set-password?token=t"),
    lambda: invite_message("Sara", "Mark", "https://x/set-password?token=t"),
    lambda: already_registered_message("", "https://x/login", "https://x/forgot-password"),
])
def test_every_template_has_subject_text_and_html_with_the_link(build):
    subject, text, html = build()
    assert subject and "Borsa Compass" in (subject + text)
    assert "https://x/" in text and "https://x/" in html


def test_html_escapes_names():
    _, _, html = invite_message("<b>x</b>", "<script>", "https://x/set-password?token=t")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_greeting_falls_back_when_name_is_empty():
    _, text, _ = verify_message("", "https://x/verify-email?token=t")
    assert text.startswith("Hi,")
```

- [ ] **Step 2: Run to verify failure**

Run: `<venv python> -m pytest tests/test_email.py -q`
Expected: FAIL — `ImportError: cannot import name 'EmailNotConfigured'`.

- [ ] **Step 3: Write `app/core/email.py`**

```python
"""
Outgoing email over SMTP (a Gmail app password in production).

Env: SMTP_HOST (default smtp.gmail.com), SMTP_PORT (587), SMTP_USER,
SMTP_PASSWORD, SMTP_FROM (default "Borsa Compass <SMTP_USER>"), and
FE_BASE_URL — every emailed link points at the web app, which hosts
/verify-email and /set-password for both clients.

Routes that must send depend on get_mailer(); a missing config is a 503,
never a silent success. Tests replace the mailer with
tests/account_fakes.OutboxMailer.
"""

import html as _html
import os
import smtplib
from email.message import EmailMessage
from typing import Protocol

from fastapi import HTTPException

APP_NAME = "Borsa Compass"


class EmailNotConfigured(HTTPException):
    def __init__(self):
        super().__init__(status_code=503, detail="Email is not configured")


class EmailSendFailed(Exception):
    """SMTP refused or failed; routes turn this into 502."""


class Mailer(Protocol):
    def send(self, to: str, subject: str, text: str, html: str) -> None: ...


class SmtpMailer:
    def __init__(self, host: str, port: int, user: str, password: str, sender: str):
        self.host, self.port = host, port
        self.user, self.password, self.sender = user, password, sender

    @classmethod
    def from_env(cls) -> "SmtpMailer":
        user = os.environ.get("SMTP_USER", "").strip()
        password = os.environ.get("SMTP_PASSWORD", "").strip()
        if not user or not password:
            raise EmailNotConfigured()
        return cls(
            host=os.environ.get("SMTP_HOST", "").strip() or "smtp.gmail.com",
            port=int(os.environ.get("SMTP_PORT", "").strip() or 587),
            user=user,
            password=password,
            sender=os.environ.get("SMTP_FROM", "").strip() or f"{APP_NAME} <{user}>",
        )

    def send(self, to: str, subject: str, text: str, html: str) -> None:
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
        try:
            with smtplib.SMTP(self.host, self.port, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(msg)
        except (smtplib.SMTPException, OSError) as e:
            print(f"[email] send to {to} failed: {e}")
            raise EmailSendFailed(str(e)) from e


def get_mailer() -> Mailer:
    """FastAPI dependency. Overridden in tests."""
    return SmtpMailer.from_env()


def _base_url() -> str:
    base = os.environ.get("FE_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise EmailNotConfigured()
    return base


def link(path: str, token: str) -> str:
    return f"{_base_url()}{path}?token={token}"


def login_url() -> str:
    return f"{_base_url()}/login"


def forgot_url() -> str:
    return f"{_base_url()}/forgot-password"


def _greeting(name: str) -> str:
    return f"Hi {name}," if name else "Hi,"


def _render(name: str, paragraphs: list, button_label: str, url: str, footer: str):
    text = "\n\n".join([_greeting(name), *paragraphs, f"{button_label}: {url}", footer,
                        f"— {APP_NAME}"])
    esc = _html.escape
    body = "".join(f"<p>{esc(p)}</p>" for p in paragraphs)
    html = (
        f'<div style="font-family:Arial,sans-serif;font-size:15px;color:#111">'
        f"<p>{esc(_greeting(name))}</p>{body}"
        f'<p><a href="{esc(url)}" style="display:inline-block;padding:10px 18px;'
        f'background:#4488ff;color:#fff;border-radius:8px;text-decoration:none">'
        f"{esc(button_label)}</a></p>"
        f'<p style="color:#666;font-size:13px">{esc(footer)}<br>'
        f'If the button does not work, open this link: {esc(url)}</p>'
        f"<p>— {APP_NAME}</p></div>"
    )
    return text, html


def verify_message(name: str, url: str):
    text, html = _render(
        name, [f"Confirm your email address to finish creating your {APP_NAME} account."],
        "Confirm email", url, "This link expires in 24 hours. If you did not sign up, ignore this email.")
    return f"Confirm your {APP_NAME} email", text, html


def reset_message(name: str, url: str):
    text, html = _render(
        name, ["Someone asked to reset the password for this account."],
        "Choose a new password", url,
        "This link expires in 1 hour. If it wasn't you, ignore this email — your password stays the same.")
    return f"Reset your {APP_NAME} password", text, html


def invite_message(name: str, inviter: str, url: str):
    who = inviter or "An admin"
    text, html = _render(
        name, [f"{who} invited you to {APP_NAME}. Choose a password to activate your account. "
               "You can also sign in with Google using this email address."],
        "Set your password", url, "This invitation expires in 7 days.")
    return f"{who} invited you to {APP_NAME}", text, html


def already_registered_message(name: str, login: str, reset: str):
    text, html = _render(
        name, ["Someone tried to create an account with this email, but you already have one.",
               f"If it was you, sign in here: {login}"],
        "Reset your password", reset, "If it wasn't you, ignore this email.")
    return f"You already have a {APP_NAME} account", text, html
```
Also remove the Task-1 stub text (the `EmailSendFailed` class above replaces it).

- [ ] **Step 4: Run the tests**

Run: `<venv python> -m pytest tests/test_email.py tests/test_accounts_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/email.py tests/test_email.py
git commit -m "feat(auth): SMTP mailer and email templates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 3: Identity switch to email + seeding + reset script (backend)

**Files:**
- Modify: `app/core/auth.py` (`CurrentUser`, `create_access_token`, `_load_user`, `get_current_user`, seeding)
- Create: `scripts/reset_users_for_email_auth.py`
- Modify tests: `tests/test_users_and_roles.py`, `tests/test_auth_gate.py`, `tests/test_alerts_router.py`

**Interfaces:**
- Consumes: `AccountStore.users_table_is_legacy()`, `iso` (Task 1).
- Produces: `CurrentUser(id, email, name="", role="user", is_active=True)`; `create_access_token(user_id: str) -> str`; `seed_users_from_env(db)` reading `AUTH_ADMINS` as emails.

- [ ] **Step 1: Update `CurrentUser`, token and loader in `app/core/auth.py`**

```python
@dataclass
class CurrentUser:
    id: str
    email: str
    name: str = ""
    role: str = ROLE_USER
    is_active: bool = True

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN
```

```python
def create_access_token(user_id: str) -> str:
    secret = _require_secret()
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=TOKEN_LIFETIME_DAYS)).timestamp()),
    }
    # Role, email and active-state are deliberately NOT claims. Tokens live
    # 30 days, so a claim would mean disabling a user does nothing for a
    # month — see get_current_user, which reads them from the row every time.
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
```

`_load_user`:
```python
    row = get_db().execute(
        "SELECT id, email, name, role, is_active FROM users WHERE id = %s",
        (user_id,),
    ).fetchone()
    if not row or not row[4]:
        return None
    return CurrentUser(id=row[0], email=row[1], name=row[2] or "",
                       role=row[3] or ROLE_USER, is_active=bool(row[4]))
```

In `get_current_user`, replace the payload checks with:
```python
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
```

- [ ] **Step 2: Replace seeding**

Replace `_parse_admin_usernames` and `seed_users_from_env` with:
```python
def _parse_admin_emails() -> set:
    raw = os.environ.get("AUTH_ADMINS", "").strip()
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def seed_users_from_env(db) -> None:
    """Create and promote the admins listed in AUTH_ADMINS (emails).

    `AUTH_ADMINS=mark3essam@gmail.com` — comma-separated emails. A listed
    email with no account gets one: verified, NO password (sign in with
    Google, or use Forgot password). Nothing else is ever created here, and
    no password ever lives in an env var.

    AUTH_ADMINS is authoritative for admin status and is re-applied on every
    boot, so it survives a DB reset and you cannot lock yourself out by
    fumbling a role in the database. Demotion of unlisted users only happens
    when AUTH_ADMINS is non-empty — a blank or unset var must never strip
    every admin and leave the app unmanageable.
    """
    from app.core.accounts import AccountStore, iso

    admins = _parse_admin_emails()
    if not admins:
        return
    if AccountStore(db).users_table_is_legacy():
        print("[auth] users table predates email accounts — run "
              "scripts/reset_users_for_email_auth.py; seeding skipped")
        return

    now = iso(datetime.now(timezone.utc))
    for email in sorted(admins):
        db.execute(
            "INSERT INTO users (id, email, name, password_hash, email_verified, role, "
            "is_active, created_at) VALUES (%s, %s, '', NULL, TRUE, %s, TRUE, %s) "
            "ON CONFLICT (email) DO NOTHING",
            (str(uuid.uuid4()), email, ROLE_ADMIN, now),
        )
    db.execute(
        "UPDATE users SET role = %s WHERE email = ANY(%s) AND role <> %s",
        (ROLE_ADMIN, sorted(admins), ROLE_ADMIN),
    )
    db.execute(
        "UPDATE users SET role = %s WHERE NOT (email = ANY(%s)) AND role <> %s",
        (ROLE_USER, sorted(admins), ROLE_USER),
    )
    db.commit()
```
Delete `AUTH_USERS` handling entirely. Update the module docstring if it mentions usernames. Leave `generate_password` in place for now (Task 6 removes it with its last caller).

- [ ] **Step 3: Update existing tests for the new identity**

In `tests/test_users_and_roles.py`, `tests/test_auth_gate.py`, `tests/test_alerts_router.py`:
- `CurrentUser(id=..., username="x", ...)` → `CurrentUser(id=..., email="x@example.com", ...)`.
- `create_access_token("user-1", "sara")` → `create_access_token("user-1")`.
- Rows faked for `_load_user` change from `(id, username, role, is_active)` to `(id, email, name, role, is_active)`, e.g. `("user-1", "sara@example.com", "Sara", "user", True)`.
- Delete the seeding tests that exercise `AUTH_USERS` (`test_seeding_does_not_overwrite_an_existing_password`, `test_seeding_creates_a_user_that_does_not_exist_yet`, `test_a_new_user_named_in_auth_admins_is_created_as_admin`) and replace the AUTH_ADMINS tests with:

```python
def _seed_db(legacy=False):
    return FakeDB([("information_schema.columns", [(1,)] if legacy else [])])


def test_auth_admins_creates_missing_admins_verified_without_password(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", " Mark3essam@Gmail.com ")
    db = _seed_db()
    seed_users_from_env(db)
    insert = next(p for s, p in db.log if s.startswith("INSERT INTO users"))
    assert insert[1] == "mark3essam@gmail.com"
    assert insert[2] == ROLE_ADMIN
    sql = next(s for s, _ in db.log if s.startswith("INSERT INTO users"))
    assert "NULL, TRUE" in sql and "ON CONFLICT (email) DO NOTHING" in sql


def test_auth_admins_promotes_the_listed_emails(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", "mark@x.com,sara@x.com")
    db = _seed_db()
    seed_users_from_env(db)
    promote = next(p for s, p in db.log if "SET role" in s and "NOT (" not in s)
    assert promote[0] == ROLE_ADMIN and promote[1] == ["mark@x.com", "sara@x.com"]


def test_auth_admins_demotes_anyone_no_longer_listed(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", "mark@x.com")
    db = _seed_db()
    seed_users_from_env(db)
    demote = next(p for s, p in db.log if "NOT (email = ANY" in s)
    assert demote[0] == ROLE_USER and demote[1] == ["mark@x.com"]


def test_empty_auth_admins_touches_nothing(monkeypatch):
    monkeypatch.delenv("AUTH_ADMINS", raising=False)
    db = _seed_db()
    seed_users_from_env(db)
    assert db.log == []


def test_a_legacy_username_table_skips_seeding(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", "mark@x.com")
    db = _seed_db(legacy=True)
    seed_users_from_env(db)
    assert not any(s.startswith(("INSERT", "UPDATE")) for s, _ in db.log)
```
Keep `test_get_current_user_does_not_trust_role_from_the_token` but assert the payload keys are exactly `{"sub", "iat", "exp"}`.

- [ ] **Step 4: Write the reset script**

`scripts/reset_users_for_email_auth.py`:
```python
"""
ONE-OFF: delete every user and everything they own, then recreate `users`
in the email-account shape and seed AUTH_ADMINS.

    DATABASE_URL=... AUTH_ADMINS=mark3essam@gmail.com \
        python scripts/reset_users_for_email_auth.py

It prints row counts and asks you to type DELETE ALL USERS. Run it from the
email-auth code, BEFORE deploying that code. There is no undo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import db as db_mod  # noqa: E402
from app.core.accounts import USER_OWNED_TABLES  # noqa: E402
from app.core.auth import seed_users_from_env  # noqa: E402

CONFIRM = "DELETE ALL USERS"


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is not set")
        return 2
    db_mod._initialized = True          # do not run init_db against the old table yet
    db = db_mod.get_db()

    tables = (*USER_OWNED_TABLES, "users")
    print("Rows that will be deleted:")
    for t in tables:
        exists = db.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0]
        n = db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] if exists else 0
        print(f"  {t:22s} {n}")
    if input(f'Type "{CONFIRM}" to continue: ').strip() != CONFIRM:
        print("Aborted — nothing changed.")
        return 1

    with db.transaction() as tx:
        for t in USER_OWNED_TABLES:
            if tx.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0]:
                tx.execute(f"DELETE FROM {t}")
        tx.execute("DROP TABLE IF EXISTS users")
    db_mod.init_db(db)
    seed_users_from_env(db)
    admins = db.execute("SELECT email, role FROM users").fetchall()
    print(f"Done. Users now: {admins}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
Do NOT run it. (Optionally dry-run it against the Task 1 Docker Postgres and answer "no" to confirm it aborts.)

- [ ] **Step 5: Run the affected tests**

Run: `<venv python> -m pytest tests/test_users_and_roles.py tests/test_auth_gate.py tests/test_alerts_router.py tests/test_accounts_store.py -q`
Expected: PASS except tests that exercise `app/routers/users.py` create/reset/list (they still assert username behaviour). Mark exactly those with `@pytest.mark.skip(reason="rewritten in Task 6")` — list them in the report; Task 6 deletes or rewrites every one.

- [ ] **Step 6: Commit**

```bash
git add app/core/auth.py scripts/reset_users_for_email_auth.py tests/
git commit -m "feat(auth): identify users by email; AUTH_ADMINS takes emails; reset script

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 4: Email + password flows and routes (backend)

**Files:**
- Create: `app/core/auth_flows.py`
- Rewrite: `app/routers/auth.py`
- Modify: `app/core/auth.py` (`PUBLIC_ENDPOINTS`)
- Tests: `tests/test_auth_flows.py`, `tests/test_auth_routes.py`, `tests/test_auth_gate.py`

**Interfaces:**
- Consumes: Task 1 store API, Task 2 mailer/templates, `hash_password`, `verify_password`, `create_access_token(user_id)`.
- Produces (`app/core/auth_flows.py`): `session(account) -> dict`, `signup(store, mailer, *, name, email, password, now)`, `resend_verification(store, mailer, *, email, now)`, `verify_email(store, *, token, now) -> Account`, `login(store, *, email, password) -> Account`, `forgot_password(store, mailer, *, email, now)`, `set_password(store, *, token, password, now) -> Account`, `send_link(store, mailer, account, purpose, now, inviter="") -> bool`. Constant `CHECK_EMAIL = {"status": "check_email"}`.
- Router dependency: `get_store() -> AccountStore` in `app/routers/auth.py` (tests override via `app.dependency_overrides`).

- [ ] **Step 1: Write the failing flow tests**

`tests/test_auth_flows.py`:
```python
"""
Email/password flows against MemoryAccountStore + OutboxMailer.

Run from egx-api-be:  python -m pytest tests/test_auth_flows.py -v
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from account_fakes import MemoryAccountStore, OutboxMailer
from app.core import auth_flows as flows
from app.core.auth import hash_password

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(minutes=2)


@pytest.fixture(autouse=True)
def fe_base(monkeypatch):
    monkeypatch.setenv("FE_BASE_URL", "https://app.test")


@pytest.fixture
def store():
    return MemoryAccountStore()


@pytest.fixture
def mailer():
    return OutboxMailer()


def _signup(store, mailer, email="Mark@Gmail.com ", now=NOW):
    flows.signup(store, mailer, name=" Mark ", email=email, password="correct-horse", now=now)


def test_signup_creates_unverified_and_emails_a_verify_link(store, mailer):
    _signup(store, mailer)
    acc = store.by_email("mark@gmail.com")
    assert acc and not acc.email_verified and acc.name == "Mark"
    assert mailer.sent[-1]["to"] == "mark@gmail.com"
    assert "/verify-email?token=" in mailer.sent[-1]["text"]


def test_login_before_verifying_is_403_email_not_verified(store, mailer):
    _signup(store, mailer)
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="mark@gmail.com", password="correct-horse")
    assert e.value.status_code == 403 and e.value.detail == "email_not_verified"


def test_verify_then_login(store, mailer):
    _signup(store, mailer)
    acc = flows.verify_email(store, token=mailer.last_link(), now=NOW)
    assert acc.email_verified
    assert flows.login(store, email="  MARK@gmail.com", password="correct-horse").id == acc.id


@pytest.mark.parametrize("email,password", [
    ("mark@gmail.com", "wrong-password"),
    ("nobody@gmail.com", "correct-horse"),
])
def test_bad_credentials_share_one_message(store, mailer, email, password):
    _signup(store, mailer)
    flows.verify_email(store, token=mailer.last_link(), now=NOW)
    with pytest.raises(HTTPException) as e:
        flows.login(store, email=email, password=password)
    assert e.value.status_code == 401 and e.value.detail == "Invalid email or password"


def test_account_without_password_cannot_password_login(store):
    store.create(email="g@x.com", name="", password_hash=None, email_verified=True, now=NOW, google_sub="g")
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="g@x.com", password="anything-long")
    assert e.value.status_code == 401


def test_disabled_account_is_403(store):
    a = store.create(email="d@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"), email_verified=True, now=NOW)
    store.update(a.id, is_active=False)
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="d@x.com", password="pw-pw-pw-pw")
    assert e.value.status_code == 403 and "disabled" in e.value.detail


def test_responses_do_not_reveal_whether_an_email_exists(store, mailer):
    """Review focus 2: identical return values for unknown / unverified / verified."""
    store.create(email="v@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"), email_verified=True, now=NOW)
    store.create(email="u@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"), email_verified=False, now=NOW)
    for email in ("new@x.com", "u@x.com", "v@x.com"):
        assert flows.signup(store, mailer, name="", email=email, password="another-pass", now=NOW) == flows.CHECK_EMAIL
        assert flows.resend_verification(store, mailer, email=email, now=LATER) == flows.CHECK_EMAIL
        assert flows.forgot_password(store, mailer, email=email, now=LATER + timedelta(minutes=2)) == flows.CHECK_EMAIL


def test_signup_for_a_verified_email_sends_already_registered_and_keeps_the_password(store, mailer):
    store.create(email="v@x.com", name="", password_hash=hash_password("original-pw"), email_verified=True, now=NOW)
    flows.signup(store, mailer, name="", email="v@x.com", password="attacker-pw", now=NOW)
    assert "already have" in mailer.sent[-1]["subject"]
    assert flows.login(store, email="v@x.com", password="original-pw")


def test_signup_again_while_unverified_replaces_password_and_resends(store, mailer):
    _signup(store, mailer)
    flows.signup(store, mailer, name="New", email="mark@gmail.com", password="second-pass", now=LATER)
    flows.verify_email(store, token=mailer.last_link(), now=LATER)
    assert flows.login(store, email="mark@gmail.com", password="second-pass")
    assert store.by_email("mark@gmail.com").name == "New"


def test_cooldown_sends_nothing_inside_60_seconds(store, mailer):
    _signup(store, mailer)
    flows.resend_verification(store, mailer, email="mark@gmail.com", now=NOW + timedelta(seconds=30))
    assert len(mailer.sent) == 1
    flows.resend_verification(store, mailer, email="mark@gmail.com", now=NOW + timedelta(seconds=61))
    assert len(mailer.sent) == 2


def test_forgot_password_then_set_password_signs_in_and_verifies(store, mailer):
    _signup(store, mailer)                     # unverified
    flows.forgot_password(store, mailer, email="mark@gmail.com", now=LATER)
    acc = flows.set_password(store, token=mailer.last_link(), password="brand-new-pw", now=LATER)
    assert acc.email_verified
    assert flows.login(store, email="mark@gmail.com", password="brand-new-pw").id == acc.id


@pytest.mark.parametrize("case", ["reused", "expired", "wrong_purpose", "superseded", "garbage"])
def test_bad_links_are_invalid_or_expired(store, mailer, case):
    """Review focus 3."""
    _signup(store, mailer)
    token = mailer.last_link()
    if case == "reused":
        flows.verify_email(store, token=token, now=NOW)
        call = lambda: flows.verify_email(store, token=token, now=NOW)
    elif case == "expired":
        call = lambda: flows.verify_email(store, token=token, now=NOW + timedelta(hours=25))
    elif case == "wrong_purpose":
        call = lambda: flows.set_password(store, token=token, password="whatever-pw", now=NOW)
    elif case == "superseded":
        flows.resend_verification(store, mailer, email="mark@gmail.com", now=LATER)
        call = lambda: flows.verify_email(store, token=token, now=LATER)
    else:
        call = lambda: flows.verify_email(store, token="not-a-token", now=NOW)
    with pytest.raises(HTTPException) as e:
        call()
    assert e.value.status_code == 400 and e.value.detail == "invalid_or_expired_link"


def test_send_failure_is_502_and_frees_the_token_and_cooldown(store, mailer):
    mailer.fail = True
    with pytest.raises(HTTPException) as e:
        _signup(store, mailer)
    assert e.value.status_code == 502
    assert store.tokens == {}
    assert store.by_email("mark@gmail.com").last_email_at is None


def test_missing_fe_base_url_is_503(store, mailer, monkeypatch):
    monkeypatch.delenv("FE_BASE_URL")
    with pytest.raises(HTTPException) as e:
        _signup(store, mailer)
    assert e.value.status_code == 503


def test_session_shape(store):
    a = store.create(email="a@x.com", name="A", password_hash=None, email_verified=True, now=NOW)
    s = flows.session(a)
    assert s["token_type"] == "bearer" and s["access_token"]
    assert s["user"] == {"id": a.id, "email": "a@x.com", "name": "A", "role": "user"}
```
Add at the top of the file: `import os; os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-flows-0123456789")` (before importing `auth_flows`).

- [ ] **Step 2: Run to verify failure**

Run: `<venv python> -m pytest tests/test_auth_flows.py -q`
Expected: FAIL — `ImportError: cannot import name 'auth_flows'`.

- [ ] **Step 3: Write `app/core/auth_flows.py`**

```python
"""
Every account flow, as plain functions over an AccountStore-shaped `store`
and a Mailer. Routers adapt HTTP to these; tests drive them directly with
tests/account_fakes.

Public flows that take an email (signup, resend, forgot) ALWAYS return
CHECK_EMAIL, whatever exists — the response must not reveal which emails
have accounts. Errors that do leak (login's email_not_verified) need the
right password first.
"""

from datetime import datetime
from typing import Optional

from fastapi import HTTPException

from app.core import email as mail
from app.core.accounts import (
    EMAIL_COOLDOWN,
    PURPOSE_INVITE,
    PURPOSE_RESET,
    PURPOSE_VERIFY,
    Account,
    clean_name,
    clean_password,
    iso,
    normalize_email,
    public_user,
)
from app.core.auth import create_access_token, hash_password, verify_password

CHECK_EMAIL = {"status": "check_email"}
INVALID_LINK = "invalid_or_expired_link"
DISABLED = "This account has been disabled."
BAD_CREDENTIALS = "Invalid email or password"


def session(account: Account) -> dict:
    return {
        "access_token": create_access_token(account.id),
        "token_type": "bearer",
        "user": public_user(account),
    }


def _in_cooldown(account: Account, now: datetime) -> bool:
    return bool(account.last_email_at) and account.last_email_at > iso(now - EMAIL_COOLDOWN)


def _deliver(store, mailer, account: Account, now: datetime, message, token: Optional[str]) -> None:
    subject, text, html = message
    try:
        mailer.send(account.email, subject, text, html)
    except mail.EmailSendFailed:
        if token:
            store.discard_token(token)
        raise HTTPException(status_code=502, detail="Could not send email")
    store.update(account.id, last_email_at=iso(now))


def send_link(store, mailer, account: Account, purpose: str, now: datetime,
              inviter: str = "") -> bool:
    """Email a fresh link of `purpose`. False (and nothing sent) inside the cooldown."""
    if _in_cooldown(account, now):
        return False
    path = "/verify-email" if purpose == PURPOSE_VERIFY else "/set-password"
    mail.link(path, "probe")   # 503 before a token exists if FE_BASE_URL is missing
    token = store.issue_token(account.id, purpose, now)
    url = mail.link(path, token)
    if purpose == PURPOSE_VERIFY:
        message = mail.verify_message(account.name, url)
    elif purpose == PURPOSE_RESET:
        message = mail.reset_message(account.name, url)
    else:
        message = mail.invite_message(account.name, inviter, url)
    _deliver(store, mailer, account, now, message, token)
    return True


def signup(store, mailer, *, name: str, email: str, password: str, now: datetime) -> dict:
    email = normalize_email(email)
    password = clean_password(password)
    name = clean_name(name)
    existing = store.by_email(email)
    if existing is None:
        account = store.create(email=email, name=name, password_hash=hash_password(password),
                               email_verified=False, now=now)
        send_link(store, mailer, account, PURPOSE_VERIFY, now)
    elif not existing.email_verified:
        # Whoever proves the inbox owns the account; until then the latest
        # sign-up's password and name win.
        store.update(existing.id, password_hash=hash_password(password), name=name)
        send_link(store, mailer, store.by_id(existing.id), PURPOSE_VERIFY, now)
    elif not _in_cooldown(existing, now):
        message = mail.already_registered_message(existing.name, mail.login_url(), mail.forgot_url())
        _deliver(store, mailer, existing, now, message, None)
    return CHECK_EMAIL


def resend_verification(store, mailer, *, email: str, now: datetime) -> dict:
    account = store.by_email(normalize_email(email))
    if account and account.is_active and not account.email_verified:
        send_link(store, mailer, account, PURPOSE_VERIFY, now)
    return CHECK_EMAIL


def forgot_password(store, mailer, *, email: str, now: datetime) -> dict:
    account = store.by_email(normalize_email(email))
    if account and account.is_active:
        send_link(store, mailer, account, PURPOSE_RESET, now)
    return CHECK_EMAIL


def _redeem(store, token: str, purposes: set, now: datetime) -> Account:
    user_id = store.consume_token(token, purposes, now)
    account = store.by_id(user_id) if user_id else None
    if account is None:
        raise HTTPException(status_code=400, detail=INVALID_LINK)
    if not account.is_active:
        raise HTTPException(status_code=403, detail=DISABLED)
    return account


def verify_email(store, *, token: str, now: datetime) -> Account:
    account = _redeem(store, token, {PURPOSE_VERIFY}, now)
    store.update(account.id, email_verified=True)
    return store.by_id(account.id)


def set_password(store, *, token: str, password: str, now: datetime) -> Account:
    password = clean_password(password)           # validate BEFORE burning the link
    account = _redeem(store, token, {PURPOSE_RESET, PURPOSE_INVITE}, now)
    # Clicking an emailed link proves the inbox, so this also verifies.
    store.update(account.id, password_hash=hash_password(password), email_verified=True)
    return store.by_id(account.id)


def login(store, *, email: str, password: str) -> Account:
    try:
        email = normalize_email(email)
    except HTTPException:
        raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)
    account = store.by_email(email)
    if not account or not account.password_hash or not verify_password(password or "", account.password_hash):
        raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)
    if not account.is_active:
        raise HTTPException(status_code=403, detail=DISABLED)
    if not account.email_verified:
        raise HTTPException(status_code=403, detail="email_not_verified")
    return account
```

- [ ] **Step 4: Run the flow tests**

Run: `<venv python> -m pytest tests/test_auth_flows.py -q`
Expected: PASS.

- [ ] **Step 5: Rewrite `app/routers/auth.py`**

```python
"""
Authentication endpoints. Logic lives in core/auth_flows.py; this file only
adapts HTTP to it.

Public (no token — listed in core/auth.PUBLIC_ENDPOINTS):
  POST /api/auth/signup               {name, email, password}  -> check_email
  POST /api/auth/resend-verification  {email}                  -> check_email
  POST /api/auth/verify-email         {token}                  -> session
  POST /api/auth/login                {email, password}        -> session
  POST /api/auth/forgot-password      {email}                  -> check_email
  POST /api/auth/set-password         {token, password}        -> session
Signed in:
  GET  /api/auth/me                                            -> user
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core import auth_flows as flows
from app.core.accounts import AccountStore
from app.core.auth import CurrentUser, get_current_user
from app.core.db import get_db
from app.core.email import Mailer, get_mailer

router = APIRouter()


def get_store() -> AccountStore:
    """FastAPI dependency. Overridden in tests with MemoryAccountStore."""
    return AccountStore(get_db())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SignupRequest(BaseModel):
    name: Optional[str] = ""
    email: str
    password: str


class EmailRequest(BaseModel):
    email: str


class TokenRequest(BaseModel):
    token: str


class LoginRequest(BaseModel):
    email: str
    password: str


class SetPasswordRequest(BaseModel):
    token: str
    password: str


@router.post("/api/auth/signup")
def signup(body: SignupRequest, store=Depends(get_store), mailer: Mailer = Depends(get_mailer)):
    return flows.signup(store, mailer, name=body.name or "", email=body.email,
                        password=body.password, now=_now())


@router.post("/api/auth/resend-verification")
def resend_verification(body: EmailRequest, store=Depends(get_store), mailer: Mailer = Depends(get_mailer)):
    return flows.resend_verification(store, mailer, email=body.email, now=_now())


@router.post("/api/auth/verify-email")
def verify_email(body: TokenRequest, store=Depends(get_store)):
    return flows.session(flows.verify_email(store, token=body.token, now=_now()))


@router.post("/api/auth/login")
def login(body: LoginRequest, store=Depends(get_store)):
    return flows.session(flows.login(store, email=body.email, password=body.password))


@router.post("/api/auth/forgot-password")
def forgot_password(body: EmailRequest, store=Depends(get_store), mailer: Mailer = Depends(get_mailer)):
    return flows.forgot_password(store, mailer, email=body.email, now=_now())


@router.post("/api/auth/set-password")
def set_password(body: SetPasswordRequest, store=Depends(get_store)):
    return flows.session(flows.set_password(store, token=body.token, password=body.password, now=_now()))


@router.get("/api/auth/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "role": user.role}
```

- [ ] **Step 6: Open the public routes**

In `app/core/auth.py`:
```python
PUBLIC_ENDPOINTS = frozenset({
    # The ways in. None of them returns data without a valid credential or
    # emailed link; the email-taking ones always answer "check_email".
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/signup"),
    ("POST", "/api/auth/resend-verification"),
    ("POST", "/api/auth/verify-email"),
    ("POST", "/api/auth/forgot-password"),
    ("POST", "/api/auth/set-password"),
    ("POST", "/api/pe/refresh"),      # scheduled; guarded by PE_REFRESH_SECRET
    ("POST", "/api/cron/risk_snapshot"),  # scheduled; guarded by CRON_SECRET
})
```
Update `tests/test_auth_gate.py::test_only_login_and_the_crons_are_public` → rename to `test_only_the_ways_in_and_the_crons_are_public` and assert the exact set above.

- [ ] **Step 7: Route tests**

`tests/test_auth_routes.py`:
```python
"""
HTTP layer for the auth routes: TestClient + dependency overrides (memory
store, outbox mailer). The auth gate middleware runs for real.

Run from egx-api-be:  python -m pytest tests/test_auth_routes.py -v
"""

import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-routes-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pytest
from fastapi.testclient import TestClient

from account_fakes import MemoryAccountStore, OutboxMailer
from app.core.email import get_mailer
from app.main import app
from app.routers.auth import get_store


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setenv("FE_BASE_URL", "https://app.test")
    store, mailer = MemoryAccountStore(), OutboxMailer()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_mailer] = lambda: mailer
    yield TestClient(app), store, mailer
    app.dependency_overrides.clear()


def test_signup_verify_login_round_trip(ctx):
    client, store, mailer = ctx
    r = client.post("/api/auth/signup", json={"name": "Mark", "email": "Mark@Gmail.com", "password": "correct-horse"})
    assert r.status_code == 200 and r.json() == {"status": "check_email"}

    r = client.post("/api/auth/login", json={"email": "mark@gmail.com", "password": "correct-horse"})
    assert r.status_code == 403 and r.json()["detail"] == "email_not_verified"

    r = client.post("/api/auth/verify-email", json={"token": mailer.last_link()})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["user"]["email"] == "mark@gmail.com"

    # /me goes through get_current_user, which reads the real DB — it is
    # covered in test_users_and_roles; here only the login response is checked.
    r = client.post("/api/auth/login", json={"email": " MARK@gmail.com ", "password": "correct-horse"})
    assert r.status_code == 200 and r.json()["user"]["id"] == store.by_email("mark@gmail.com").id


def test_public_routes_need_no_token(ctx):
    client, _, _ = ctx
    for path, body in [
        ("/api/auth/forgot-password", {"email": "x@y.com"}),
        ("/api/auth/resend-verification", {"email": "x@y.com"}),
    ]:
        assert client.post(path, json=body).status_code == 200


def test_bad_link_is_400(ctx):
    client, _, _ = ctx
    r = client.post("/api/auth/set-password", json={"token": "nope", "password": "long-enough-pw"})
    assert r.status_code == 400 and r.json()["detail"] == "invalid_or_expired_link"


def test_me_requires_a_token(ctx):
    client, _, _ = ctx
    assert client.get("/api/auth/me").status_code == 401
```
If `app.main` import fails in tests because seeding touches `get_db()` at import, it is already wrapped in try/except there (prints and continues) — no change needed. If the middleware's `authenticate_request` is what returns 401 for `/me`, that is the expected assertion.

- [ ] **Step 8: Run backend tests touched so far**

Run: `<venv python> -m pytest tests/test_auth_flows.py tests/test_auth_routes.py tests/test_auth_gate.py tests/test_users_and_roles.py -q`
Expected: PASS (Task-3 skips still skipped).

- [ ] **Step 9: Commit**

```bash
git add app/core/auth_flows.py app/routers/auth.py app/core/auth.py tests/
git commit -m "feat(auth): sign-up, email verification, login by email, forgot/set password

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 5: Google and Apple sign-in (backend)

**Files:**
- Create: `app/core/oauth.py`
- Modify: `app/core/auth_flows.py` (add `oauth_sign_in`), `app/routers/auth.py` (2 routes), `app/core/auth.py` (`PUBLIC_ENDPOINTS` +2), `requirements.txt`
- Tests: `tests/test_oauth.py`, extend `tests/test_auth_flows.py`, `tests/test_auth_gate.py`

**Interfaces:**
- Produces (`app/core/oauth.py`): `Identity(sub: str, email: Optional[str], name: str)`; `verify_google(id_token: str, jwks=None) -> Identity`; `verify_apple(identity_token: str, jwks=None) -> Identity`; `exchange_apple_code(code: str) -> Optional[str]` (refresh token, None when not configured or on failure); `revoke_apple_token(refresh_token: str) -> None` (never raises); `_client_ids(env_var) -> list[str]`.
- Produces (`auth_flows`): `oauth_sign_in(store, *, provider, identity, name_hint="", now) -> Account`.

- [ ] **Step 1: Add the dependency**

In `requirements.txt` change `pyjwt>=2.9.0` → `pyjwt[crypto]>=2.9.0`, then `<venv>/bin/pip install "pyjwt[crypto]>=2.9.0"`.

- [ ] **Step 2: Write the failing verification tests**

`tests/test_oauth.py`:
```python
"""
Google / Apple ID-token verification with a locally generated RSA key and a
stub JWKS client — no network.

Run from egx-api-be:  python -m pytest tests/test_oauth.py -v
"""

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app.core import oauth

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StubJWKS:
    def __init__(self, key):
        self.key = key.public_key()

    def get_signing_key_from_jwt(self, token):
        return type("K", (), {"key": self.key})()


def _token(claims, key=KEY):
    base = {"iat": int(time.time()), "exp": int(time.time()) + 600}
    return jwt.encode({**base, **claims}, key, algorithm="RS256")


@pytest.fixture(autouse=True)
def ids(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_IDS", "web-id, android-id")
    monkeypatch.setenv("APPLE_CLIENT_IDS", "com.markbotros.borsacompass,host.exp.Exponent")


GOOD_GOOGLE = {"iss": "https://accounts.google.com", "aud": "android-id", "sub": "g-1",
               "email": "Mark@Gmail.com", "email_verified": True, "name": "Mark B"}
GOOD_APPLE = {"iss": "https://appleid.apple.com", "aud": "com.markbotros.borsacompass",
              "sub": "a-1", "email": "x@privaterelay.appleid.com"}


def test_google_happy_path_normalises_email():
    ident = oauth.verify_google(_token(GOOD_GOOGLE), jwks=StubJWKS(KEY))
    assert ident == oauth.Identity(sub="g-1", email="mark@gmail.com", name="Mark B")


def test_google_accepts_the_bare_issuer_too():
    ident = oauth.verify_google(_token({**GOOD_GOOGLE, "iss": "accounts.google.com"}), jwks=StubJWKS(KEY))
    assert ident.sub == "g-1"


@pytest.mark.parametrize("bad", [
    {"aud": "someone-else"},
    {"iss": "https://evil.example"},
    {"exp": int(time.time()) - 10},
    {"email_verified": False},
    {"email": None},
])
def test_google_rejects(bad):
    with pytest.raises(HTTPException) as e:
        oauth.verify_google(_token({**GOOD_GOOGLE, **bad}), jwks=StubJWKS(KEY))
    assert e.value.status_code == 401


def test_google_rejects_a_token_signed_by_another_key():
    with pytest.raises(HTTPException):
        oauth.verify_google(_token(GOOD_GOOGLE, key=OTHER), jwks=StubJWKS(KEY))


def test_google_not_configured_is_503(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_IDS", "")
    with pytest.raises(HTTPException) as e:
        oauth.verify_google(_token(GOOD_GOOGLE), jwks=StubJWKS(KEY))
    assert e.value.status_code == 503


def test_apple_happy_path_and_missing_email():
    assert oauth.verify_apple(_token(GOOD_APPLE), jwks=StubJWKS(KEY)).email == "x@privaterelay.appleid.com"
    no_email = {k: v for k, v in GOOD_APPLE.items() if k != "email"}
    assert oauth.verify_apple(_token(no_email), jwks=StubJWKS(KEY)).email is None


def test_apple_accepts_expo_go_audience():
    assert oauth.verify_apple(_token({**GOOD_APPLE, "aud": "host.exp.Exponent"}), jwks=StubJWKS(KEY)).sub == "a-1"


@pytest.mark.parametrize("bad", [{"aud": "com.other.app"}, {"iss": "https://accounts.google.com"}])
def test_apple_rejects(bad):
    with pytest.raises(HTTPException):
        oauth.verify_apple(_token({**GOOD_APPLE, **bad}), jwks=StubJWKS(KEY))


def test_apple_exchange_is_skipped_when_not_configured(monkeypatch):
    for v in ("APPLE_TEAM_ID", "APPLE_KEY_ID", "APPLE_PRIVATE_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert oauth.exchange_apple_code("code") is None


def test_revoke_never_raises(monkeypatch):
    monkeypatch.setattr(oauth, "_apple_post", lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setenv("APPLE_TEAM_ID", "T"); monkeypatch.setenv("APPLE_KEY_ID", "K")
    monkeypatch.setenv("APPLE_PRIVATE_KEY", "not-a-key")
    oauth.revoke_apple_token("rt")   # no exception
```

- [ ] **Step 3: Run to verify failure**

Run: `<venv python> -m pytest tests/test_oauth.py -q`
Expected: FAIL — `ImportError: cannot import name 'oauth'`.

- [ ] **Step 4: Write `app/core/oauth.py`**

```python
"""
Google and Apple sign-in: verify the provider's signed ID token ourselves
(PyJWT + the provider's JWKS). No SDK, no Firebase.

Env:
  GOOGLE_CLIENT_IDS  comma-separated OAuth client ids (web, Android, iOS)
  APPLE_CLIENT_IDS   comma-separated audiences: the bundle id, plus
                     host.exp.Exponent so Expo Go works in development
  APPLE_TEAM_ID, APPLE_KEY_ID, APPLE_PRIVATE_KEY (.p8 contents)
                     optional; enable the code→refresh-token exchange that
                     lets account deletion revoke Apple's grant
"""

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import HTTPException

GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
APPLE_JWKS = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"


@dataclass(frozen=True)
class Identity:
    sub: str
    email: Optional[str]
    name: str


def _client_ids(env_var: str) -> list:
    return [c.strip() for c in os.environ.get(env_var, "").split(",") if c.strip()]


@lru_cache(maxsize=2)
def _jwks(url: str):
    return jwt.PyJWKClient(url, cache_keys=True)


def _decode(token: str, jwks, audiences: list, issuers, label: str) -> dict:
    try:
        key = jwks.get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256"], audience=audiences,
                            options={"require": ["exp", "iat", "sub", "iss", "aud"]})
    except Exception:
        raise HTTPException(status_code=401, detail=f"{label} sign-in failed")
    if claims.get("iss") not in issuers:
        raise HTTPException(status_code=401, detail=f"{label} sign-in failed")
    return claims


def verify_google(id_token: str, jwks=None) -> Identity:
    audiences = _client_ids("GOOGLE_CLIENT_IDS")
    if not audiences:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")
    claims = _decode(id_token, jwks or _jwks(GOOGLE_JWKS), audiences, GOOGLE_ISSUERS, "Google")
    email = (claims.get("email") or "").strip().lower()
    if not email or claims.get("email_verified") is not True:
        raise HTTPException(status_code=401, detail="Google sign-in failed")
    return Identity(sub=str(claims["sub"]), email=email, name=claims.get("name") or "")


def verify_apple(identity_token: str, jwks=None) -> Identity:
    audiences = _client_ids("APPLE_CLIENT_IDS")
    if not audiences:
        raise HTTPException(status_code=503, detail="Apple sign-in is not configured")
    claims = _decode(identity_token, jwks or _jwks(APPLE_JWKS), audiences, (APPLE_ISSUER,), "Apple")
    email = (claims.get("email") or "").strip().lower() or None
    # Apple never puts the name in the token; the app sends it (first sign-in only).
    return Identity(sub=str(claims["sub"]), email=email, name="")


def _apple_configured() -> bool:
    return all(os.environ.get(v, "").strip() for v in ("APPLE_TEAM_ID", "APPLE_KEY_ID", "APPLE_PRIVATE_KEY"))


def _apple_client_secret() -> str:
    now = int(time.time())
    key = os.environ["APPLE_PRIVATE_KEY"].replace("\\n", "\n")
    return jwt.encode(
        {"iss": os.environ["APPLE_TEAM_ID"], "iat": now, "exp": now + 300,
         "aud": APPLE_ISSUER, "sub": _client_ids("APPLE_CLIENT_IDS")[0]},
        key, algorithm="ES256", headers={"kid": os.environ["APPLE_KEY_ID"]},
    )


def _apple_post(url: str, fields: dict) -> dict:
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def exchange_apple_code(code: str) -> Optional[str]:
    """Trade the one-time authorization code for a refresh token, or None."""
    if not code or not _apple_configured() or not _client_ids("APPLE_CLIENT_IDS"):
        return None
    try:
        out = _apple_post(APPLE_TOKEN_URL, {
            "client_id": _client_ids("APPLE_CLIENT_IDS")[0],
            "client_secret": _apple_client_secret(),
            "code": code,
            "grant_type": "authorization_code",
        })
        return out.get("refresh_token")
    except Exception as e:
        print(f"[oauth] apple code exchange failed: {e}")
        return None


def revoke_apple_token(refresh_token: str) -> None:
    """Best-effort: account deletion must never fail because Apple is down."""
    if not refresh_token or not _apple_configured():
        return
    try:
        _apple_post(APPLE_REVOKE_URL, {
            "client_id": _client_ids("APPLE_CLIENT_IDS")[0],
            "client_secret": _apple_client_secret(),
            "token": refresh_token,
            "token_type_hint": "refresh_token",
        })
    except Exception as e:
        print(f"[oauth] apple revoke failed: {e}")
```

- [ ] **Step 5: Run verification tests**

Run: `<venv python> -m pytest tests/test_oauth.py -q`
Expected: PASS.

- [ ] **Step 6: Add the resolution flow tests to `tests/test_auth_flows.py`**

```python
from app.core.oauth import Identity


def test_oauth_creates_a_verified_account(store):
    acc = flows.oauth_sign_in(store, provider="google",
                              identity=Identity(sub="g1", email="new@x.com", name="New"), now=NOW)
    assert acc.email_verified and acc.google_sub == "g1" and acc.name == "New" and acc.password_hash is None


def test_oauth_finds_by_sub_first(store):
    first = flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a1", email="r@privaterelay.appleid.com", name=""), now=NOW)
    again = flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a1", email=None, name=""), now=NOW)
    assert again.id == first.id


def test_oauth_links_a_verified_password_account_by_email_and_keeps_its_password(store):
    a = store.create(email="mark@gmail.com", name="Mark", password_hash=hash_password("keep-this-pw"), email_verified=True, now=NOW)
    acc = flows.oauth_sign_in(store, provider="google", identity=Identity(sub="g1", email="mark@gmail.com", name="Other"), now=NOW)
    assert acc.id == a.id and acc.google_sub == "g1" and acc.name == "Mark"
    assert flows.login(store, email="mark@gmail.com", password="keep-this-pw")


def test_oauth_takeover_of_an_unverified_account_clears_the_password(store, mailer):
    """Review focus 1."""
    flows.signup(store, mailer, name="", email="victim@x.com", password="attacker-pw", now=NOW)
    acc = flows.oauth_sign_in(store, provider="google", identity=Identity(sub="g9", email="victim@x.com", name="Victim"), now=NOW)
    assert acc.email_verified and acc.password_hash is None and acc.name == "Victim"
    with pytest.raises(HTTPException):
        flows.login(store, email="victim@x.com", password="attacker-pw")


def test_oauth_disabled_account_is_403(store):
    a = store.create(email="d@x.com", name="", password_hash=None, email_verified=True, now=NOW, google_sub="g1")
    store.update(a.id, is_active=False)
    with pytest.raises(HTTPException) as e:
        flows.oauth_sign_in(store, provider="google", identity=Identity(sub="g1", email="d@x.com", name=""), now=NOW)
    assert e.value.status_code == 403


def test_apple_without_email_and_unknown_sub_is_400(store):
    with pytest.raises(HTTPException) as e:
        flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a-new", email=None, name=""), now=NOW)
    assert e.value.status_code == 400


def test_name_hint_fills_an_empty_name(store):
    acc = flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a1", email="a@x.com", name=""),
                              name_hint="  Sara  K ", now=NOW)
    assert acc.name == "Sara K"
```

- [ ] **Step 7: Implement `oauth_sign_in` in `app/core/auth_flows.py`**

```python
def oauth_sign_in(store, *, provider: str, identity, name_hint: str = "", now: datetime) -> Account:
    """Resolve a verified Google/Apple identity to an account (spec §3)."""
    name = clean_name(identity.name or name_hint)
    account = store.by_provider(provider, identity.sub)
    if account is None:
        if not identity.email:
            raise HTTPException(status_code=400, detail="Your Apple ID did not share an email address.")
        email = normalize_email(identity.email)
        account = store.by_email(email)
        if account is None:
            account = store.create(email=email, name=name, password_hash=None, email_verified=True,
                                   now=now, **{f"{provider}_sub": identity.sub})
        else:
            fields = {f"{provider}_sub": identity.sub}
            if not account.email_verified:
                # The provider just proved the inbox. A password set by whoever
                # pre-registered this email must not survive that.
                fields.update(password_hash=None, email_verified=True)
            if not account.name and name:
                fields["name"] = name
            store.update(account.id, **fields)
            account = store.by_id(account.id)
    elif not account.name and name:
        store.update(account.id, name=name)
        account = store.by_id(account.id)
    if not account.is_active:
        raise HTTPException(status_code=403, detail=DISABLED)
    return account
```
(`store.create` must accept `google_sub=`/`apple_sub=` keyword args — it does.)

- [ ] **Step 8: Routes + public entries**

Append to `app/routers/auth.py` (and add both to the docstring list):
```python
from app.core import oauth


class GoogleRequest(BaseModel):
    idToken: str
    name: Optional[str] = ""


class AppleRequest(BaseModel):
    identityToken: str
    authorizationCode: Optional[str] = None
    name: Optional[str] = ""


@router.post("/api/auth/google")
def google_sign_in(body: GoogleRequest, store=Depends(get_store)):
    identity = oauth.verify_google(body.idToken)
    return flows.session(flows.oauth_sign_in(store, provider="google", identity=identity,
                                             name_hint=body.name or "", now=_now()))


@router.post("/api/auth/apple")
def apple_sign_in(body: AppleRequest, store=Depends(get_store)):
    identity = oauth.verify_apple(body.identityToken)
    account = flows.oauth_sign_in(store, provider="apple", identity=identity,
                                  name_hint=body.name or "", now=_now())
    if body.authorizationCode and not account.apple_refresh_token:
        refresh = oauth.exchange_apple_code(body.authorizationCode)
        if refresh:
            store.update(account.id, apple_refresh_token=refresh)
    return flows.session(account)
```
Add `("POST", "/api/auth/google")` and `("POST", "/api/auth/apple")` to `PUBLIC_ENDPOINTS` and to the exact-set assertion in `tests/test_auth_gate.py`.

Route test in `tests/test_auth_routes.py`:
```python
def test_google_route_signs_in(ctx, monkeypatch):
    client, store, _ = ctx
    from app.core import oauth
    monkeypatch.setattr(oauth, "verify_google", lambda tok: oauth.Identity(sub="g1", email="g@x.com", name="G"))
    r = client.post("/api/auth/google", json={"idToken": "anything"})
    assert r.status_code == 200 and r.json()["user"]["email"] == "g@x.com"
    assert store.by_provider("google", "g1")


def test_apple_route_stores_refresh_token_once(ctx, monkeypatch):
    client, store, _ = ctx
    from app.core import oauth
    monkeypatch.setattr(oauth, "verify_apple", lambda tok: oauth.Identity(sub="a1", email="a@x.com", name=""))
    calls = []
    monkeypatch.setattr(oauth, "exchange_apple_code", lambda code: calls.append(code) or "rt-1")
    for _ in range(2):
        assert client.post("/api/auth/apple", json={"identityToken": "t", "authorizationCode": "c", "name": "Ann"}).status_code == 200
    acc = store.by_provider("apple", "a1")
    assert acc.apple_refresh_token == "rt-1" and acc.name == "Ann" and calls == ["c"]
```

- [ ] **Step 9: Run**

Run: `<venv python> -m pytest tests/test_oauth.py tests/test_auth_flows.py tests/test_auth_routes.py tests/test_auth_gate.py -q`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add requirements.txt app/core/oauth.py app/core/auth_flows.py app/routers/auth.py app/core/auth.py tests/
git commit -m "feat(auth): Google and Apple sign-in with account linking

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 6: Account deletion and the admin API (backend)

**Files:**
- Modify: `app/core/auth_flows.py` (`delete_account`, `invite`, `admin_send_reset`), `app/routers/auth.py` (`DELETE /api/auth/me`), rewrite `app/routers/users.py`, `app/core/auth.py` (remove `generate_password`, `_PASSWORD_ALPHABET`, `GENERATED_PASSWORD_LENGTH`)
- Tests: `tests/test_users_and_roles.py` (rewrite the admin-API section; remove Task-3 skips), `tests/test_auth_flows.py`

**Interfaces:**
- Produces (`auth_flows`): `delete_account(store, account_id, *, revoke=oauth.revoke_apple_token) -> None` (409 for the last active admin); `invite(store, mailer, *, email, name, inviter, now) -> Account` (409 if the email exists); `admin_send_reset(store, mailer, *, user_id, now) -> str` (returns `"invite"` or `"reset"`; 404 unknown; 429 `"An email was sent less than a minute ago."` inside cooldown).
- Admin API JSON per user: `{id, email, name, role, is_active, email_verified, methods, created_at, holdings_count}`.

- [ ] **Step 1: Flow tests (append to `tests/test_auth_flows.py`)**

```python
def test_delete_account_removes_everything_and_revokes_apple(store):
    a = store.create(email="a@x.com", name="", password_hash=None, email_verified=True, now=NOW, apple_sub="a1")
    store.update(a.id, apple_refresh_token="rt")
    revoked = []
    flows.delete_account(store, a.id, revoke=revoked.append)
    assert store.by_id(a.id) is None and revoked == ["rt"]
    from app.core.accounts import USER_OWNED_TABLES
    assert {t for t, uid in store.deleted_from if uid == a.id} == set(USER_OWNED_TABLES)


def test_last_active_admin_cannot_delete_themselves(store):
    a = store.create(email="boss@x.com", name="", password_hash=None, email_verified=True, now=NOW)
    store.set_role(a.id, "admin")
    with pytest.raises(HTTPException) as e:
        flows.delete_account(store, a.id, revoke=lambda rt: None)
    assert e.value.status_code == 409


def test_invite_creates_unverified_passwordless_and_emails_set_password(store, mailer):
    acc = flows.invite(store, mailer, email=" Sara@X.com", name="Sara", inviter="Mark", now=NOW)
    assert acc.email == "sara@x.com" and not acc.email_verified and acc.password_hash is None
    assert "Mark invited you" in mailer.sent[-1]["subject"]
    done = flows.set_password(store, token=mailer.last_link(), password="sara-pw-123", now=NOW)
    assert done.email_verified


def test_invite_for_an_existing_email_is_409(store, mailer):
    store.create(email="sara@x.com", name="", password_hash=None, email_verified=True, now=NOW)
    with pytest.raises(HTTPException) as e:
        flows.invite(store, mailer, email="sara@x.com", name="", inviter="", now=NOW)
    assert e.value.status_code == 409


def test_admin_send_reset_picks_invite_or_reset(store, mailer):
    invited = flows.invite(store, mailer, email="i@x.com", name="", inviter="", now=NOW)
    assert flows.admin_send_reset(store, mailer, user_id=invited.id, now=LATER) == "invite"
    full = store.create(email="f@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"), email_verified=True, now=NOW)
    assert flows.admin_send_reset(store, mailer, user_id=full.id, now=LATER) == "reset"
    with pytest.raises(HTTPException) as e:
        flows.admin_send_reset(store, mailer, user_id=full.id, now=LATER)
    assert e.value.status_code == 429
```

- [ ] **Step 2: Implement in `app/core/auth_flows.py`**

```python
from app.core import oauth


def delete_account(store, account_id: str, *, revoke=None) -> None:
    account = store.by_id(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="User not found")
    if account.role == "admin" and account.is_active and store.count_active_admins() <= 1:
        raise HTTPException(
            status_code=409,
            detail="You are the last active admin — no one could administer the app.")
    store.delete_account(account.id)
    if account.apple_refresh_token:
        (revoke or oauth.revoke_apple_token)(account.apple_refresh_token)


def invite(store, mailer, *, email: str, name: str, inviter: str, now: datetime) -> Account:
    email = normalize_email(email)
    if store.by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    account = store.create(email=email, name=clean_name(name), password_hash=None,
                           email_verified=False, now=now)
    send_link(store, mailer, account, PURPOSE_INVITE, now, inviter=inviter)
    return store.by_id(account.id)


def admin_send_reset(store, mailer, *, user_id: str, now: datetime) -> str:
    account = store.by_id(user_id)
    if account is None:
        raise HTTPException(status_code=404, detail="User not found")
    purpose = PURPOSE_INVITE if (not account.password_hash and not account.email_verified) else PURPOSE_RESET
    if not send_link(store, mailer, account, purpose, now):
        raise HTTPException(status_code=429, detail="An email was sent less than a minute ago.")
    return purpose
```

- [ ] **Step 3: `DELETE /api/auth/me` in `app/routers/auth.py`**

```python
from fastapi import Response


@router.delete("/api/auth/me", status_code=204)
def delete_me(user: CurrentUser = Depends(get_current_user), store=Depends(get_store)):
    flows.delete_account(store, user.id)
    return Response(status_code=204)
```
(Signed-in route: NOT added to `PUBLIC_ENDPOINTS`.)

- [ ] **Step 4: Rewrite `app/routers/users.py`**

```python
"""
User administration — the admin tab's backend.

GET    /api/users                   — list every user
POST   /api/users                   — invite {email, name?}: creates the account, emails a set-password link
POST   /api/users/{id}/send-reset   — email a reset link (an invite link if they never set up)
PATCH  /api/users/{id}              — enable / disable
DELETE /api/users/{id}              — delete the user and everything they own

Every route requires an admin. `role` is NOT settable here: admin status comes
only from the AUTH_ADMINS env var at boot (core/auth.seed_users_from_env), so
privilege escalation through this API is structurally impossible.
No password ever passes through this API.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core import auth_flows as flows
from app.core.auth import ROLE_ADMIN, CurrentUser, require_admin
from app.core.email import Mailer, get_mailer
from app.routers.auth import get_store

router = APIRouter()


class InviteRequest(BaseModel):
    email: str
    name: Optional[str] = ""


class UserPatch(BaseModel):
    is_active: bool


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _out(account, holdings: dict) -> dict:
    return {
        "id": account.id, "email": account.email, "name": account.name,
        "role": account.role, "is_active": account.is_active,
        "email_verified": account.email_verified, "methods": account.methods,
        "created_at": account.created_at, "holdings_count": holdings.get(account.id, 0),
    }


def _target(store, user_id: str):
    account = store.by_id(user_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    return account


def _guard_not_self(target_id: str, admin: CurrentUser, verb: str) -> None:
    if target_id == admin.id:
        raise HTTPException(status_code=400, detail=f"You cannot {verb} your own account.")


def _guard_not_last_admin(store, target, verb: str) -> None:
    """An app with no active admin can never be administered again."""
    if target.role == ROLE_ADMIN and target.is_active and store.count_active_admins() <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"You cannot {verb} the last active admin — no one could administer the app.")


@router.get("/api/users")
def list_users(admin: CurrentUser = Depends(require_admin), store=Depends(get_store)):
    holdings = store.holdings_counts()
    return {"users": [_out(a, holdings) for a in store.list_all()]}


@router.post("/api/users", status_code=201)
def invite_user(body: InviteRequest, admin: CurrentUser = Depends(require_admin),
                store=Depends(get_store), mailer: Mailer = Depends(get_mailer)):
    account = flows.invite(store, mailer, email=body.email, name=body.name or "",
                           inviter=admin.name or admin.email, now=_now())
    return {"user": _out(account, {})}


@router.post("/api/users/{user_id}/send-reset")
def send_reset(user_id: str, admin: CurrentUser = Depends(require_admin),
               store=Depends(get_store), mailer: Mailer = Depends(get_mailer)):
    sent = flows.admin_send_reset(store, mailer, user_id=user_id, now=_now())
    return {"id": user_id, "sent": sent}


@router.patch("/api/users/{user_id}")
def set_user_active(user_id: str, body: UserPatch, admin: CurrentUser = Depends(require_admin),
                    store=Depends(get_store)):
    target = _target(store, user_id)
    if not body.is_active:
        _guard_not_self(user_id, admin, "disable")
        _guard_not_last_admin(store, target, "disable")
    store.update(user_id, is_active=bool(body.is_active))
    return _out(store.by_id(user_id), store.holdings_counts())


@router.delete("/api/users/{user_id}")
def delete_user(user_id: str, admin: CurrentUser = Depends(require_admin), store=Depends(get_store)):
    target = _target(store, user_id)
    _guard_not_self(user_id, admin, "delete")
    _guard_not_last_admin(store, target, "delete")
    flows.delete_account(store, user_id)
    return {"deleted": user_id, "email": target.email}
```
Remove `generate_password`, `_PASSWORD_ALPHABET`, `GENERATED_PASSWORD_LENGTH` and their comment from `app/core/auth.py`.

- [ ] **Step 5: Rewrite the admin-API tests in `tests/test_users_and_roles.py`**

Delete every test that depends on generated passwords, usernames, `_clean_username`, `_clean_password`, `reset_password`, or the old FakeDB `_fetch_user` rows (including the Task-3 skips). Keep: `require_admin` tests, `test_every_users_route_requires_admin_not_merely_a_login`, `test_the_admin_api_cannot_grant_admin` (update it to call `invite_user` and keep the `"SET role" not in src` check), `get_current_user` tests, weights tests, SQL-scoping tests. Add, calling the route functions directly with a `MemoryAccountStore`:

```python
from datetime import datetime, timezone
from account_fakes import MemoryAccountStore, OutboxMailer
import app.routers.users as users_mod

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)


def _store_with(*roles):
    s = MemoryAccountStore()
    accs = []
    for i, role in enumerate(roles):
        a = s.create(email=f"u{i}@x.com", name="", password_hash=None, email_verified=True, now=T0)
        if role == "admin":
            s.set_role(a.id, "admin")
        accs.append(s.users[a.id])
    return s, accs


def _admin(acc):
    return CurrentUser(id=acc.id, email=acc.email, role=ROLE_ADMIN)


def test_cannot_disable_or_delete_yourself():
    s, (me, _) = _store_with("admin", "admin")
    for call in (lambda: users_mod.set_user_active(me.id, users_mod.UserPatch(is_active=False), _admin(me), s),
                 lambda: users_mod.delete_user(me.id, _admin(me), s)):
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == 400


def test_cannot_disable_the_last_active_admin():
    s, (boss, other_admin) = _store_with("admin", "admin")
    s.update(other_admin.id, is_active=False)
    actor = CurrentUser(id="someone-else", email="x@x.com", role=ROLE_ADMIN)
    with pytest.raises(HTTPException):
        users_mod.set_user_active(boss.id, users_mod.UserPatch(is_active=False), actor, s)


def test_delete_user_removes_every_owned_table():
    s, (boss, victim) = _store_with("admin", "user")
    users_mod.delete_user(victim.id, _admin(boss), s)
    from app.core.accounts import USER_OWNED_TABLES
    assert {t for t, uid in s.deleted_from if uid == victim.id} == set(USER_OWNED_TABLES)


def test_listing_shows_methods_and_never_a_password_hash():
    s, (boss,) = _store_with("admin")
    s.update(boss.id, password_hash="secret-hash", google_sub="g")
    out = users_mod.list_users(_admin(boss), s)["users"][0]
    assert out["methods"] == ["password", "google"]
    assert "secret-hash" not in str(out) and "password_hash" not in out


def test_invite_route_uses_admin_name_as_inviter(monkeypatch):
    monkeypatch.setenv("FE_BASE_URL", "https://app.test")
    s, (boss,) = _store_with("admin")
    mailer = OutboxMailer()
    actor = CurrentUser(id=boss.id, email=boss.email, name="Mark", role=ROLE_ADMIN)
    out = users_mod.invite_user(users_mod.InviteRequest(email="NEW@x.com", name="N"), actor, s, mailer)
    assert out["user"]["email"] == "new@x.com" and "Mark invited you" in mailer.sent[-1]["subject"]
```
Route functions are called positionally in the order `(path params…, body, admin, store[, mailer])` exactly as declared in Step 4.

- [ ] **Step 6: Run the FULL backend suite (tmux)**

```bash
mkdir -p ~/logs; tmux kill-session -t egx-be-tests 2>/dev/null
tmux new -d -s egx-be-tests "cd <be worktree> && <venv python> -m pytest -q 2>&1 | tee ~/logs/egx-be-tests.log"
```
Wait for it (`tail -3 ~/logs/egx-be-tests.log`). Expected: all pass, 0 failed; `test_accounts_pg.py` skipped. `grep -rn "SET role" app/routers` → no output. `grep -rn "generate_password\|username" app` → only historical comments, if any (remove them).

- [ ] **Step 7: Commit**

```bash
git add app/ tests/
git commit -m "feat(auth): self-service account deletion; admin invites and reset emails replace generated passwords

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 7: Backend docs (backend + web CLAUDE.md mirror)

**Files:**
- Modify: `egx-api-be/CLAUDE.md` → copy to `egx-api-fe/CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-10-12-email-auth-design.md` (Ruling 1)

- [ ] **Step 1: Rewrite the auth section of `CLAUDE.md`**

Replace the `### Auth, roles and user management` section body (keep the heading) with prose covering, in this order:
1. Accounts are emails (stored lower-case). Ways in: email+password (must verify email first), Google (`/api/auth/google`, web + mobile), Apple (`/api/auth/apple`, iOS app only). All return `{access_token, token_type, user:{id,email,name,role}}`; same 30-day HS256 JWT with claims `{sub, iat, exp}` only.
2. Open sign-up. `signup` / `resend-verification` / `forgot-password` always answer `{"status":"check_email"}` — never reveal whether an email exists. Login with the right password but unverified email → `403 email_not_verified`.
3. Emailed links: `auth_tokens` stores SHA-256 only; verify 24 h, reset 1 h, invite 7 days; newest link of a purpose wins; single use; `invalid_or_expired_link` otherwise. Links point at the WEB (`FE_BASE_URL`) `/verify-email` and `/set-password`. 60-second per-user email cooldown (`users.last_email_at`).
4. Account linking rule (provider sub → verified email link → create); linking to an UNVERIFIED account clears its password (pre-registration takeover).
5. Admin status ONLY from `AUTH_ADMINS` (emails). A listed email with no account gets one, verified, with no password. `AUTH_USERS` no longer exists. No API writes `role` (grep test).
6. Admin API: invite by email, send reset, enable/disable, delete. No passwords pass through it.
7. `DELETE /api/auth/me` deletes the caller and every `USER_OWNED_TABLES` row (completeness test pins the list), revokes Apple's grant best-effort; the last active admin cannot.
8. Code map: `core/accounts.py` (store, validation), `core/auth_flows.py` (every flow), `core/oauth.py`, `core/email.py`, `routers/auth.py`, `routers/users.py`; tests use `tests/account_fakes.py`.
9. Env vars: `AUTH_ADMINS`, `FE_BASE_URL`, `SMTP_HOST` (smtp.gmail.com), `SMTP_PORT` (587), `SMTP_USER`, `SMTP_PASSWORD` (Gmail app password), `SMTP_FROM`, `GOOGLE_CLIENT_IDS`, `APPLE_CLIENT_IDS`, `APPLE_TEAM_ID`, `APPLE_KEY_ID`, `APPLE_PRIVATE_KEY`. Web: `NEXT_PUBLIC_GOOGLE_CLIENT_ID`.
10. One-off migration: `scripts/reset_users_for_email_auth.py` (deletes all users and their data; typed confirmation; run before deploying).

Also, in the same file (it documents the web too):
- Fix the auth-gate paragraph's `PUBLIC_ENDPOINTS` list (8 auth routes + 2 crons).
- Frontend middleware paragraph: signed-out pages are `/login`, `/signup`, `/forgot-password` (signed-in visitors bounced home) plus `/verify-email` and `/set-password` (always reachable — an emailed link must work whoever is signed in).
- Admin tab: Invite form (email + optional name) and "Send reset email"; method badges (Password/Google/Apple, Invited, Unverified). Delete the `PasswordRevealDialog` bullet and the copy-credentials explanation entirely.
- New: `/account` page (name, email, Log out, Delete account with typed `DELETE`), reached from the Navbar name; `GoogleSignInButton` (Google Identity Services, hidden without `NEXT_PUBLIC_GOOGLE_CLIENT_ID`); shared `AuthCard` chrome for the five signed-out pages.
- Remove every remaining mention of usernames and generated passwords.

- [ ] **Step 2: Correct the spec**

In the spec §3 table, replace `{token, user}` with `{access_token, token_type, user}` in every row and the sentence below the table.

- [ ] **Step 3: Mirror and commit in both repos**

```bash
cp CLAUDE.md <fe worktree>/CLAUDE.md && cmp CLAUDE.md <fe worktree>/CLAUDE.md
git add CLAUDE.md docs/superpowers/specs/2026-10-12-email-auth-design.md && git commit -m "docs: email accounts, sign-up, Google/Apple, SMTP

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
(cd <fe worktree> && git add CLAUDE.md && git commit -m "docs: mirror backend CLAUDE.md (email accounts)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV")
```

---
## Task 8: Web — shared auth lib and every consumer (login, provider, navbar, admin)

All paths relative to the **fe** worktree. These lib files are shared with mobile (Task 12 syncs them).

**Files:**
- Modify: `src/app/lib/auth.ts`, `src/app/lib/api.ts` (Auth + User administration sections)
- Modify: `src/app/components/AuthProvider.tsx`, `src/app/components/Navbar.tsx`, `src/app/login/page.tsx`
- Create: `src/app/components/AuthCard.tsx`
- Admin: modify `src/app/admin/page.tsx`, `src/app/components/AdminUsersTable.tsx`; create `src/app/components/InviteUserModal.tsx`; delete `CreateUserModal.tsx`, `PasswordRevealDialog.tsx`

**Interfaces:**
- Produces (`lib/auth.ts`): `AuthUser {id, email, name, role}`, `asRole`, `displayName(u)`, `parseAuthUser(raw) -> AuthUser | null`, `looksLikeEmail(s)`, `PASSWORD_MIN_LENGTH = 8`, `PASSWORD_MAX_LENGTH = 128`, `EMAIL_NOT_VERIFIED`, `INVALID_LINK`.
- Produces (`lib/api.ts`): `LoginError(message, code?)`, `Session {token, user}`, `loginRequest(email, password)`, `signupRequest(name, email, password)`, `resendVerification(email)`, `forgotPassword(email)`, `verifyEmailRequest(token)`, `setPasswordRequest(token, password)`, `googleSignInRequest(idToken, name?)`, `appleSignInRequest({identityToken, authorizationCode?, name?})`, `fetchMe()`, `deleteMyAccount()`; admin: `SignInMethod`, `ManagedUser`, `fetchUsers()`, `inviteUser(email, name)`, `sendUserReset(id)`, `setUserActive(id, isActive)`, `deleteUser(id)`.
- Produces (`AuthProvider`): context gains `completeSignIn(session: Session): void`; `login(email, password)` throws `LoginError` (with `.code`).
- Produces (`AuthCard.tsx`): `AuthCard({title, subtitle?, children, footer?})`, `Field(props)`, `FormError({message})`, `FormNotice({children})`, `SubmitButton({busy, label, busyLabel})`, `inputClass`.

- [ ] **Step 1: `src/app/lib/auth.ts`**

```ts
/** Shared with egx-api-mobile — keep platform-neutral (see apiConfig.ts). */

export type UserRole = "user" | "admin";

export interface AuthUser {
  id: string;
  email: string;
  /** May be empty — show displayName(), never `name` alone. */
  name: string;
  role: UserRole;
}

export const PASSWORD_MIN_LENGTH = 8;
export const PASSWORD_MAX_LENGTH = 128;

/** Backend error codes the forms branch on (detail strings, not messages). */
export const EMAIL_NOT_VERIFIED = "email_not_verified";
export const INVALID_LINK = "invalid_or_expired_link";

/** Anything that isn't literally "admin" is a plain user. */
export function asRole(value: unknown): UserRole {
  return value === "admin" ? "admin" : "user";
}

export function displayName(u: Pick<AuthUser, "name" | "email">): string {
  return u.name.trim() || u.email;
}

/** A user object from the API or storage, or null if it isn't one (e.g. a pre-email cached user). */
export function parseAuthUser(raw: unknown): AuthUser | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.id !== "string" || typeof r.email !== "string") return null;
  return {
    id: r.id,
    email: r.email,
    name: typeof r.name === "string" ? r.name : "",
    role: asRole(r.role),
  };
}

/** Client-side hint only — the backend is the judge. Same shape it checks. */
export function looksLikeEmail(s: string): boolean {
  return /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(s.trim());
}
```

- [ ] **Step 2: `src/app/lib/api.ts` — imports and the Auth section**

Change the import to `import { parseAuthUser, type AuthUser } from "./auth";` and replace everything from `// ---- Auth ----` to the end of `fetchMe` with:
```ts
// ---- Auth ----

/** A rejected auth call: the message to show, and the backend's code when it sent one. */
export class LoginError extends Error {
  constructor(message: string, public readonly code?: string) {
    super(message);
  }
}

export interface Session {
  token: string;
  user: AuthUser;
}

const AUTH_CODE_MESSAGES: Record<string, string> = {
  email_not_verified: "Confirm your email first — check your inbox for the link.",
  invalid_or_expired_link: "This link has expired or was already used. Request a new one.",
};

/**
 * The public auth routes. Deliberately NOT routed through fetchJSON: a 401
 * here is a wrong password and must surface on the form, not trigger the
 * "your session expired" sign-out path.
 */
async function authPost(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(`${base()}/auth/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      (typeof data?.detail === "string" && data.detail) ||
      (typeof data?.error === "string" && data.error) ||
      "";
    const code = detail in AUTH_CODE_MESSAGES ? detail : undefined;
    throw new LoginError(
      code ? AUTH_CODE_MESSAGES[code] : detail || "Something went wrong. Try again.",
      code
    );
  }
  return data;
}

function toSession(data: unknown): Session {
  const d = data as { access_token?: unknown; user?: unknown } | null;
  const user = parseAuthUser(d?.user);
  if (typeof d?.access_token !== "string" || !user) {
    throw new LoginError("Unexpected response from the server.");
  }
  return { token: d.access_token, user };
}

export async function loginRequest(email: string, password: string): Promise<Session> {
  return toSession(await authPost("login", { email, password }));
}

/** Always resolves the same way whether or not the email exists (the backend says "check_email"). */
export async function signupRequest(name: string, email: string, password: string): Promise<void> {
  await authPost("signup", { name, email, password });
}

export async function resendVerification(email: string): Promise<void> {
  await authPost("resend-verification", { email });
}

export async function forgotPassword(email: string): Promise<void> {
  await authPost("forgot-password", { email });
}

export async function verifyEmailRequest(token: string): Promise<Session> {
  return toSession(await authPost("verify-email", { token }));
}

export async function setPasswordRequest(token: string, password: string): Promise<Session> {
  return toSession(await authPost("set-password", { token, password }));
}

export async function googleSignInRequest(idToken: string, name?: string): Promise<Session> {
  return toSession(await authPost("google", { idToken, name: name ?? "" }));
}

export async function appleSignInRequest(input: {
  identityToken: string;
  authorizationCode?: string | null;
  name?: string;
}): Promise<Session> {
  return toSession(
    await authPost("apple", {
      identityToken: input.identityToken,
      authorizationCode: input.authorizationCode ?? null,
      name: input.name ?? "",
    })
  );
}

/** GET /auth/me — re-reads the role from the DB. A 401 signs out like any call. */
export async function fetchMe(): Promise<AuthUser> {
  const user = parseAuthUser(await fetchJSON<unknown>(`${base()}/auth/me`));
  if (!user) throw new Error("Malformed /auth/me response");
  return user;
}

/** DELETE /auth/me — deletes the account and everything in it. */
export async function deleteMyAccount(): Promise<void> {
  await fetchJSON<unknown>(`${base()}/auth/me`, { method: "DELETE" });
}
```

- [ ] **Step 3: `src/app/lib/api.ts` — the User administration section**

Replace from `// ---- User administration (admin only) ----` through `deleteUser` with:
```ts
// ---- User administration (admin only) ----

export type SignInMethod = "password" | "google" | "apple";

export interface ManagedUser {
  id: string;
  email: string;
  name: string;
  role: "user" | "admin";
  is_active: boolean;
  email_verified: boolean;
  methods: SignInMethod[];
  created_at: string;
  holdings_count: number;
}

export async function fetchUsers(): Promise<{ users: ManagedUser[] }> {
  return fetchJSON<{ users: ManagedUser[] }>(`${base()}/users`);
}

/** Creates the account and emails them a set-password link. 409 if the email exists. */
export async function inviteUser(email: string, name: string): Promise<{ user: ManagedUser }> {
  return fetchJSON(`${base()}/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, name }),
  });
}

/** Emails a reset link — or a fresh invite if they never set up. 429 inside the 60 s cooldown. */
export async function sendUserReset(id: string): Promise<{ id: string; sent: "invite" | "reset" }> {
  return fetchJSON(`${base()}/users/${encodeURIComponent(id)}/send-reset`, { method: "POST" });
}

export async function setUserActive(id: string, isActive: boolean): Promise<ManagedUser> {
  return fetchJSON<ManagedUser>(`${base()}/users/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_active: isActive }),
  });
}

export async function deleteUser(id: string): Promise<{ deleted: string; email: string }> {
  return fetchJSON(`${base()}/users/${encodeURIComponent(id)}`, { method: "DELETE" });
}
```

- [ ] **Step 4: `src/app/components/AuthCard.tsx`**

```tsx
"use client";

import type { InputHTMLAttributes, ReactNode } from "react";

/** Shared chrome for the signed-out pages: login, signup, forgot, verify, set-password. */

export const inputClass =
  "w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-[16px] text-white outline-none transition-colors focus:border-accent/50 focus:ring-1 focus:ring-accent/20 md:text-sm";

export function AuthCard({
  title,
  subtitle,
  children,
  footer,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="flex min-h-[70vh] items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm rounded-2xl border border-white/10 bg-charcoal p-6 shadow-xl">
        <div className="mb-6 text-center">
          <h1 className="text-2xl font-bold text-white">{title}</h1>
          {subtitle ? <p className="mt-1 text-sm text-white/50">{subtitle}</p> : null}
        </div>
        {children}
        {footer ? <div className="mt-6 text-center text-sm text-white/50">{footer}</div> : null}
      </div>
    </div>
  );
}

export function Field({ label, ...props }: { label: string } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-white/60">{label}</span>
      <input {...props} className={inputClass} />
    </label>
  );
}

export function FormError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div role="alert" className="rounded-lg border border-loss/30 bg-loss/10 px-3 py-2 text-xs text-loss">
      {message}
    </div>
  );
}

export function FormNotice({ children }: { children: ReactNode }) {
  return (
    <div role="status" className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-xs text-white/80">
      {children}
    </div>
  );
}

export function SubmitButton({ busy, label, busyLabel }: { busy: boolean; label: string; busyLabel: string }) {
  return (
    <button
      type="submit"
      disabled={busy}
      className="min-h-[44px] w-full rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-charcoal-dark transition-opacity disabled:opacity-50"
    >
      {busy ? busyLabel : label}
    </button>
  );
}
```
(`loss`/`accent` here are existing theme tokens; `loss` marks an error message, which is a real negative state, matching the current login page.)

- [ ] **Step 5: `src/app/components/AuthProvider.tsx`**

- Import: `import { loginRequest, type Session } from "../lib/api";` and `import { parseAuthUser, type AuthUser, type UserRole } from "../lib/auth";` (drop `asRole`).
- `AuthCtx`: `login: (email: string, password: string) => Promise<void>;` and add `completeSignIn: (session: Session) => void;`.
- `readStoredUser()`: body becomes `try { const raw = localStorage.getItem(USER_KEY); return raw ? parseAuthUser(JSON.parse(raw)) : null; } catch { return null; }` (keep the optimistic-only comment).
- In the `/auth/me` effect, replace the `data?.username` branch with `const fresh = res.ok ? parseAuthUser(data) : null; if (fresh) { …same body… }`.
- Replace `login` with:
```tsx
  const completeSignIn = useCallback((session: Session) => {
    try {
      localStorage.setItem(TOKEN_KEY, session.token);
      localStorage.setItem(USER_KEY, JSON.stringify(session.user));
    } catch {}
    setPresenceCookie(true);
    setToken(session.token);
    setUser(session.user);
    setError(null);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      setError(null);
      try {
        completeSignIn(await loginRequest(email, password));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not sign in");
        throw e;
      }
    },
    [completeSignIn]
  );
```
- Add `completeSignIn` to the memo value + deps and to the no-provider fallback (`completeSignIn: () => {}`).
- In the unauthorized handler, keep `/login` exempt AND add the other signed-out pages: `const PUBLIC = ["/login", "/signup", "/forgot-password", "/verify-email", "/set-password"]; if (!PUBLIC.includes(path)) …`.

- [ ] **Step 6: `src/app/components/Navbar.tsx`**

Import `displayName` from `../lib/auth`. Replace the desktop `<span … title={user?.username}>{user?.username}</span>` with:
```tsx
            <Link
              href="/account"
              className="max-w-[160px] truncate text-xs text-white/50 transition-colors hover:text-white"
              title={user ? `${displayName(user)} — account` : "Account"}
            >
              {user ? displayName(user) : "Account"}
            </Link>
```
and the mobile logout `title` → `title={user ? \`Log out (${displayName(user)})\` : "Log out"}`. Add, just before the mobile logout button, a mobile account link:
```tsx
          <Link
            href="/account"
            aria-label="Account"
            className="flex h-9 w-9 items-center justify-center rounded-md text-white/60 transition-colors hover:bg-white/5 hover:text-white md:hidden"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="4" />
              <path d="M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1" />
            </svg>
          </Link>
```

- [ ] **Step 7: `src/app/login/page.tsx`**

```tsx
"use client";

import { Suspense, useState, FormEvent, useEffect } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "../components/AuthProvider";
import { AuthCard, Field, FormError, FormNotice, SubmitButton } from "../components/AuthCard";
import { LoginError, resendVerification } from "../lib/api";
import { EMAIL_NOT_VERIFIED } from "../lib/auth";

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login, isAuthenticated } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [unverified, setUnverified] = useState(false);
  const [resent, setResent] = useState(false);

  const next = searchParams?.get("next") || "/";

  useEffect(() => {
    if (isAuthenticated) router.replace(next);
  }, [isAuthenticated, next, router]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setErr(null);
    setUnverified(false);
    setResent(false);
    setSubmitting(true);
    try {
      await login(email.trim(), password);
      router.replace(next);
    } catch (e: unknown) {
      if (e instanceof LoginError && e.code === EMAIL_NOT_VERIFIED) setUnverified(true);
      else setErr(e instanceof Error ? e.message : "Could not sign in");
    } finally {
      setSubmitting(false);
    }
  };

  const onResend = async () => {
    try {
      await resendVerification(email.trim());
      setResent(true);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Could not send the email");
    }
  };

  return (
    <form onSubmit={onSubmit} className="space-y-4">
      {/* GoogleSignInButton is added here in Task 10 */}
      <Field label="Email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
      <Field label="Password" type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
      <div className="-mt-2 text-right">
        <Link href="/forgot-password" className="text-xs text-accent hover:underline">
          Forgot password?
        </Link>
      </div>
      {unverified ? (
        <FormNotice>
          Confirm your email first — we sent a link to {email.trim()}.{" "}
          {resent ? (
            <span className="font-semibold">Sent again.</span>
          ) : (
            <button type="button" onClick={onResend} className="font-semibold text-accent underline">
              Resend the link
            </button>
          )}
        </FormNotice>
      ) : null}
      <FormError message={err} />
      <SubmitButton busy={submitting} label="Sign in" busyLabel="Signing in…" />
    </form>
  );
}

export default function LoginPage() {
  return (
    <AuthCard
      title="Sign in"
      subtitle="Access your portfolio and watchlist."
      footer={
        <>
          New here?{" "}
          <Link href="/signup" className="text-accent hover:underline">
            Create an account
          </Link>
        </>
      }
    >
      <Suspense fallback={null}>
        <LoginForm />
      </Suspense>
    </AuthCard>
  );
}
```

- [ ] **Step 8: Admin — `InviteUserModal.tsx` (replaces `CreateUserModal.tsx`)**

`git rm src/app/components/CreateUserModal.tsx src/app/components/PasswordRevealDialog.tsx`, then create `src/app/components/InviteUserModal.tsx`:
```tsx
"use client";

import { FormEvent, useState } from "react";
import { inputClass } from "./AuthCard";

/**
 * Full-screen on mobile, centred card on desktop — matching AddHoldingForm.
 * No password field: the person gets an email and chooses their own.
 */
export default function InviteUserModal({
  onInvite,
  onClose,
}: {
  onInvite: (email: string, name: string) => Promise<void>;
  onClose: () => void;
}) {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setErr(null);
    setSubmitting(true);
    try {
      await onInvite(email.trim(), name.trim());
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Could not send the invitation");
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-charcoal-dark md:items-center md:justify-center md:bg-black/70 md:backdrop-blur-sm">
      <div
        className="flex-1 overflow-y-auto p-4 md:max-h-[90vh] md:w-full md:max-w-md md:flex-none md:rounded-2xl md:border md:border-white/10 md:bg-charcoal md:p-6"
        style={{ paddingTop: "max(1rem, env(safe-area-inset-top))" }}
      >
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-lg font-bold text-white">Invite user</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="flex h-9 w-9 items-center justify-center rounded-md text-white/50 transition-colors hover:bg-white/5 hover:text-white"
          >
            ✕
          </button>
        </div>
        <form onSubmit={onSubmit} className="space-y-4">
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-white/60">Email</span>
            <input type="email" required autoComplete="off" value={email} onChange={(e) => setEmail(e.target.value)} className={inputClass} />
          </label>
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-white/60">Name (optional)</span>
            <input type="text" autoComplete="off" value={name} onChange={(e) => setName(e.target.value)} className={inputClass} />
          </label>
          <p className="text-xs text-white/40">
            They get an email with a link to choose their password. The link works for 7 days.
          </p>
          {err ? (
            <div className="rounded-lg border border-loss/30 bg-loss/10 px-3 py-2 text-xs text-loss">{err}</div>
          ) : null}
          <button
            type="submit"
            disabled={submitting}
            className="min-h-[44px] w-full rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-charcoal-dark transition-opacity disabled:opacity-50"
          >
            {submitting ? "Sending…" : "Send invitation"}
          </button>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: Admin — `AdminUsersTable.tsx`**

- Rename the prop `onResetPassword` → `onSendReset` everywhere in the file; the button label becomes `Send reset email`.
- Wherever the username is rendered (desktop row and mobile card), render instead:
```tsx
<div className="min-w-0">
  <div className="truncate text-sm text-white">{u.name || u.email}</div>
  {u.name ? <div className="truncate text-xs text-white/40">{u.email}</div> : null}
</div>
```
- Add, next to `RoleBadge`, and render `<MethodBadges u={u} />` in a new "Sign-in" column (desktop) and under the name (mobile):
```tsx
const METHOD_LABEL: Record<ManagedUser["methods"][number], string> = {
  password: "Password",
  google: "Google",
  apple: "Apple",
};

function MethodBadges({ u }: { u: ManagedUser }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {u.methods.length === 0 ? <span className="text-xs text-white/40">Invited</span> : null}
      {u.methods.map((m) => (
        <span key={m} className="rounded-md border border-white/10 px-1.5 py-0.5 text-[10px] text-white/60">
          {METHOD_LABEL[m]}
        </span>
      ))}
      {u.email_verified ? null : (
        <span className="rounded-md border border-white/20 px-1.5 py-0.5 text-[10px] text-white/50">Unverified</span>
      )}
    </span>
  );
}
```
- Any `aria-label`/`title` that interpolated `u.username` uses `u.email`.

- [ ] **Step 10: Admin — `src/app/admin/page.tsx`**

- Imports: replace `CreateUserModal`/`PasswordRevealDialog` with `import InviteUserModal from "../components/InviteUserModal";`; api imports become `deleteUser, fetchUsers, inviteUser, sendUserReset, setUserActive, type ManagedUser`.
- Remove `Revealed`, `revealed` state and its dialog. Rename `showCreate` → `showInvite`; button text `+ Invite user`; FAB `aria-label="Invite user"`.
- Add `const [notice, setNotice] = useState<string | null>(null);` rendered above the error block:
```tsx
      {notice ? (
        <div role="status" className="mb-4 rounded-lg border border-accent/30 bg-accent/10 px-3 py-2.5 text-sm text-white/80">
          {notice}
        </div>
      ) : null}
```
- Handlers:
```tsx
  const handleInvite = async (email: string, name: string) => {
    const res = await inviteUser(email, name);
    setShowInvite(false);
    setNotice(`Invitation sent to ${res.user.email}.`);
    await load();
  };

  const handleSendReset = async (u: ManagedUser) => {
    if (!confirm(`Email ${u.email} a link to choose a new password? Their current password keeps working until they use it.`)) return;
    setBusyId(u.id);
    setError(null);
    setNotice(null);
    try {
      const res = await sendUserReset(u.id);
      setNotice(res.sent === "invite" ? `Invitation re-sent to ${u.email}.` : `Reset link sent to ${u.email}.`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not send the email");
    } finally {
      setBusyId(null);
    }
  };
```
- `handleDelete` confirm text uses `u.email`.
- `<AdminUsersTable onSendReset={handleSendReset} …/>`; `{showInvite && <InviteUserModal onInvite={handleInvite} onClose={() => setShowInvite(false)} />}`.

- [ ] **Step 11: Typecheck + lint**

Run: `npx tsc --noEmit && npm run lint`
Expected: no errors. `grep -rn "username\|generated_password\|PasswordReveal\|CreateUserModal" src` → no matches.

- [ ] **Step 12: Commit**

```bash
git add -A src
git commit -m "feat(web): email sign-in, shared auth API for sign-up/links/Google/Apple, invite-based admin

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 9: Web — sign-up, forgot, verify-email and set-password pages

**Files:**
- Create: `src/app/signup/page.tsx`, `src/app/forgot-password/page.tsx`, `src/app/verify-email/page.tsx`, `src/app/set-password/page.tsx`
- Modify: `src/middleware.ts`

**Interfaces:**
- Consumes: Task 8 (`AuthCard`, `Field`, `FormError`, `FormNotice`, `SubmitButton`, `signupRequest`, `forgotPassword`, `verifyEmailRequest`, `setPasswordRequest`, `useAuth().completeSignIn`, `PASSWORD_MIN_LENGTH`, `looksLikeEmail`).

- [ ] **Step 1: `src/middleware.ts`**

Replace `const PUBLIC_PATHS = ["/login"];` and the `isPublicPath` branch with:
```ts
// Signed-out pages. A signed-in visitor is sent home from these…
const SIGNED_OUT_ONLY = ["/login", "/signup", "/forgot-password"];
// …but NOT from these: a link from an email must work whoever is signed in.
const EMAIL_LINK_PAGES = ["/verify-email", "/set-password"];
```
```ts
  const matches = (list: string[]) =>
    list.some((p) => pathname === p || pathname.startsWith(`${p}/`));

  if (isPublicFile || matches(EMAIL_LINK_PAGES)) return NextResponse.next();

  if (matches(SIGNED_OUT_ONLY)) {
    // Already signed in? These pages have nothing to offer — send them in.
    if (hasAuth) {
      const home = req.nextUrl.clone();
      home.pathname = "/";
      home.search = "";
      return NextResponse.redirect(home);
    }
    return NextResponse.next();
  }
```
Update the file comment's "Only these are reachable signed out" sentence to list the five pages.

- [ ] **Step 2: `src/app/signup/page.tsx`**

```tsx
"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { AuthCard, Field, FormError, FormNotice, SubmitButton } from "../components/AuthCard";
import { signupRequest } from "../lib/api";
import { PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH, looksLikeEmail } from "../lib/auth";

export default function SignupPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sentTo, setSentTo] = useState<string | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (!looksLikeEmail(email)) return setErr("Enter a valid email address.");
    if (password.length < PASSWORD_MIN_LENGTH) return setErr(`Use at least ${PASSWORD_MIN_LENGTH} characters for your password.`);
    setBusy(true);
    try {
      await signupRequest(name.trim(), email.trim(), password);
      setSentTo(email.trim());
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Could not create the account");
    } finally {
      setBusy(false);
    }
  };

  const footer = (
    <>
      Already have an account?{" "}
      <Link href="/login" className="text-accent hover:underline">
        Sign in
      </Link>
    </>
  );

  if (sentTo) {
    return (
      <AuthCard title="Check your inbox" footer={footer}>
        <FormNotice>
          We sent a link to <span className="font-semibold">{sentTo}</span>. Open it to finish creating your
          account. It works for 24 hours.
        </FormNotice>
      </AuthCard>
    );
  }

  return (
    <AuthCard title="Create account" subtitle="Track your portfolio and watchlist." footer={footer}>
      {/* GoogleSignInButton is added here in Task 10 */}
      <form onSubmit={onSubmit} className="space-y-4">
        <Field label="Name" autoComplete="name" value={name} onChange={(e) => setName(e.target.value)} maxLength={80} />
        <Field label="Email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        <Field
          label={`Password (at least ${PASSWORD_MIN_LENGTH} characters)`}
          type="password"
          autoComplete="new-password"
          required
          minLength={PASSWORD_MIN_LENGTH}
          maxLength={PASSWORD_MAX_LENGTH}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <FormError message={err} />
        <SubmitButton busy={busy} label="Create account" busyLabel="Creating…" />
      </form>
    </AuthCard>
  );
}
```

- [ ] **Step 3: `src/app/forgot-password/page.tsx`**

```tsx
"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { AuthCard, Field, FormError, FormNotice, SubmitButton } from "../components/AuthCard";
import { forgotPassword } from "../lib/api";
import { looksLikeEmail } from "../lib/auth";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (!looksLikeEmail(email)) return setErr("Enter a valid email address.");
    setBusy(true);
    try {
      await forgotPassword(email.trim());
      setSent(true);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Could not send the email");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AuthCard
      title="Reset password"
      subtitle={sent ? undefined : "We'll email you a link to choose a new one."}
      footer={
        <Link href="/login" className="text-accent hover:underline">
          Back to sign in
        </Link>
      }
    >
      {sent ? (
        // Same words whether or not the account exists — see the backend's check_email.
        <FormNotice>
          If an account exists for <span className="font-semibold">{email.trim()}</span>, a reset link is on its
          way. It works for 1 hour.
        </FormNotice>
      ) : (
        <form onSubmit={onSubmit} className="space-y-4">
          <Field label="Email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          <FormError message={err} />
          <SubmitButton busy={busy} label="Send reset link" busyLabel="Sending…" />
        </form>
      )}
    </AuthCard>
  );
}
```

- [ ] **Step 4: `src/app/verify-email/page.tsx`**

```tsx
"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "../components/AuthProvider";
import { AuthCard, FormError } from "../components/AuthCard";
import { verifyEmailRequest } from "../lib/api";

function Verify() {
  const token = useSearchParams()?.get("token") ?? "";
  const router = useRouter();
  const { completeSignIn } = useAuth();
  const [err, setErr] = useState<string | null>(token ? null : "This link is incomplete. Open it again from the email.");
  // React 18 dev double-runs effects; a verify link is single-use, so the
  // second call would turn success into "already used".
  const started = useRef(false);

  useEffect(() => {
    if (!token || started.current) return;
    started.current = true;
    verifyEmailRequest(token)
      .then((session) => {
        completeSignIn(session);
        router.replace("/");
      })
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : "Could not confirm your email"));
  }, [token, completeSignIn, router]);

  return err ? (
    <div className="space-y-4">
      <FormError message={err} />
      <Link href="/login" className="block text-center text-sm text-accent hover:underline">
        Go to sign in — you can resend the link from there
      </Link>
    </div>
  ) : (
    <p className="text-center text-sm text-white/60">Confirming…</p>
  );
}

export default function VerifyEmailPage() {
  return (
    <AuthCard title="Confirm email">
      <Suspense fallback={null}>
        <Verify />
      </Suspense>
    </AuthCard>
  );
}
```

- [ ] **Step 5: `src/app/set-password/page.tsx`**

```tsx
"use client";

import { FormEvent, Suspense, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "../components/AuthProvider";
import { AuthCard, Field, FormError, SubmitButton } from "../components/AuthCard";
import { setPasswordRequest } from "../lib/api";
import { PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH } from "../lib/auth";

function SetPasswordForm() {
  const token = useSearchParams()?.get("token") ?? "";
  const router = useRouter();
  const { completeSignIn } = useAuth();
  const [password, setPassword] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(token ? null : "This link is incomplete. Open it again from the email.");

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (password.length < PASSWORD_MIN_LENGTH) return setErr(`Use at least ${PASSWORD_MIN_LENGTH} characters.`);
    if (password !== confirmPw) return setErr("The two passwords don't match.");
    setBusy(true);
    try {
      completeSignIn(await setPasswordRequest(token, password));
      router.replace("/");
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Could not set the password");
      setBusy(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="space-y-4">
      <Field label="New password" type="password" autoComplete="new-password" required minLength={PASSWORD_MIN_LENGTH} maxLength={PASSWORD_MAX_LENGTH} value={password} onChange={(e) => setPassword(e.target.value)} />
      <Field label="Repeat password" type="password" autoComplete="new-password" required value={confirmPw} onChange={(e) => setConfirmPw(e.target.value)} />
      <FormError message={err} />
      <SubmitButton busy={busy} label="Save and sign in" busyLabel="Saving…" />
      <Link href="/forgot-password" className="block text-center text-xs text-white/50 hover:underline">
        Link expired? Request a new one
      </Link>
    </form>
  );
}

export default function SetPasswordPage() {
  return (
    <AuthCard title="Choose a password">
      <Suspense fallback={null}>
        <SetPasswordForm />
      </Suspense>
    </AuthCard>
  );
}
```

- [ ] **Step 6: Typecheck, lint, build**

Run: `npx tsc --noEmit && npm run lint`, then the build in tmux:
```bash
tmux kill-session -t egx-fe-build 2>/dev/null; tmux new -d -s egx-fe-build "cd <fe worktree> && npm run build 2>&1 | tee ~/logs/egx-fe-build.log"
```
Expected: build succeeds; the route list includes `/signup`, `/forgot-password`, `/verify-email`, `/set-password`.

- [ ] **Step 7: Commit**

```bash
git add -A src
git commit -m "feat(web): sign-up, forgot-password, verify-email and set-password pages

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 10: Web — Continue with Google

**Files:**
- Create: `src/app/components/GoogleSignInButton.tsx`
- Modify: `src/app/login/page.tsx`, `src/app/signup/page.tsx`, `.env.example` (create if missing)

**Interfaces:**
- Consumes: `googleSignInRequest(idToken)`, `useAuth().completeSignIn`.
- Produces: `GoogleSignInButton({ onSignedIn: () => void })` — renders nothing when `NEXT_PUBLIC_GOOGLE_CLIENT_ID` is unset.

- [ ] **Step 1: `src/app/components/GoogleSignInButton.tsx`**

```tsx
"use client";

import Script from "next/script";
import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "./AuthProvider";
import { googleSignInRequest } from "../lib/api";

const CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? "";

interface GoogleId {
  initialize(cfg: { client_id: string; callback: (r: { credential?: string }) => void; ux_mode?: string }): void;
  renderButton(el: HTMLElement, opts: Record<string, unknown>): void;
}
declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } };
  }
}

/**
 * Google Identity Services button. Google hands us a signed ID token; the
 * backend verifies it and returns our own session — no NextAuth, no Firebase.
 */
export default function GoogleSignInButton({ onSignedIn }: { onSignedIn: () => void }) {
  const { completeSignIn } = useAuth();
  const slot = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onCredential = useCallback(
    async (r: { credential?: string }) => {
      if (!r.credential) return;
      setErr(null);
      try {
        completeSignIn(await googleSignInRequest(r.credential));
        onSignedIn();
      } catch (e: unknown) {
        setErr(e instanceof Error ? e.message : "Google sign-in failed");
      }
    },
    [completeSignIn, onSignedIn]
  );

  useEffect(() => {
    if (!ready || !slot.current || !window.google) return;
    window.google.accounts.id.initialize({ client_id: CLIENT_ID, callback: onCredential, ux_mode: "popup" });
    window.google.accounts.id.renderButton(slot.current, {
      theme: "filled_black",
      size: "large",
      shape: "pill",
      text: "continue_with",
      width: slot.current.offsetWidth || 320,
    });
  }, [ready, onCredential]);

  if (!CLIENT_ID) return null;

  return (
    <div className="space-y-2">
      <Script src="https://accounts.google.com/gsi/client" strategy="afterInteractive" onLoad={() => setReady(true)} onReady={() => setReady(true)} />
      <div ref={slot} className="flex min-h-[44px] w-full justify-center" />
      {err ? <p role="alert" className="text-center text-xs text-loss">{err}</p> : null}
      <div className="flex items-center gap-3 py-1 text-xs text-white/30">
        <span className="h-px flex-1 bg-white/10" />
        or
        <span className="h-px flex-1 bg-white/10" />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Mount it**

In `login/page.tsx` replace the `{/* GoogleSignInButton is added here in Task 10 */}` comment with `<GoogleSignInButton onSignedIn={() => router.replace(next)} />` (import default from `../components/GoogleSignInButton`). In `signup/page.tsx` replace the comment with `<div className="mb-4"><GoogleSignInButton onSignedIn={() => router.replace("/")} /></div>` and add `const router = useRouter();` (`import { useRouter } from "next/navigation";`).

- [ ] **Step 3: Env docs**

Add to `.env.example` (create with existing vars if absent — check `git ls-files | grep env`):
```
# Google Identity Services OAuth client id (Web application type). Empty hides the button.
NEXT_PUBLIC_GOOGLE_CLIENT_ID=
```

- [ ] **Step 4: Typecheck, lint, build (tmux as in Task 9)**

Expected: green. With `NEXT_PUBLIC_GOOGLE_CLIENT_ID` unset the login page renders exactly as before (no divider).

- [ ] **Step 5: Commit**

```bash
git add -A src .env.example
git commit -m "feat(web): Continue with Google on sign-in and sign-up

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 11: Web — Account page with delete account

**Files:**
- Create: `src/app/account/page.tsx`

**Interfaces:**
- Consumes: `useAuth()` (`user`, `logout`), `deleteMyAccount()`, `displayName`.

- [ ] **Step 1: `src/app/account/page.tsx`**

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../components/AuthProvider";
import { deleteMyAccount } from "../lib/api";
import { displayName } from "../lib/auth";
import { inputClass } from "../components/AuthCard";

const CONFIRM_WORD = "DELETE";

export default function AccountPage() {
  const { user, logout } = useAuth();
  const router = useRouter();
  const [confirmText, setConfirmText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (!user) return null;

  const onDelete = async () => {
    setErr(null);
    setBusy(true);
    try {
      await deleteMyAccount();
      logout();
      router.replace("/login");
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "Could not delete the account");
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-xl px-4 py-6 md:py-8">
      <h1 className="text-2xl font-bold text-white">Account</h1>

      <section className="mt-6 rounded-xl border border-white/10 bg-charcoal p-4">
        <div className="text-lg font-semibold text-white">{displayName(user)}</div>
        <div className="mt-1 text-sm text-white/50">{user.email}</div>
        <button
          onClick={() => {
            logout();
            router.replace("/login");
          }}
          className="mt-4 min-h-[44px] rounded-lg border border-white/10 px-4 text-sm text-white/70 transition-colors hover:bg-white/5"
        >
          Log out
        </button>
      </section>

      <section className="mt-6 rounded-xl border border-loss/30 p-4">
        <h2 className="text-base font-semibold text-white">Delete account</h2>
        <p className="mt-1 text-sm text-white/50">
          Permanently deletes your account, portfolio, sale and dividend history, watchlist and alert settings. This
          cannot be undone.
        </p>
        <label className="mt-4 block">
          <span className="mb-1 block text-xs font-medium text-white/60">Type {CONFIRM_WORD} to confirm</span>
          <input value={confirmText} onChange={(e) => setConfirmText(e.target.value)} autoComplete="off" className={inputClass} />
        </label>
        {err ? <p role="alert" className="mt-3 text-xs text-loss">{err}</p> : null}
        <button
          onClick={onDelete}
          disabled={busy || confirmText !== CONFIRM_WORD}
          className="mt-4 min-h-[44px] w-full rounded-lg bg-loss px-4 text-sm font-semibold text-white transition-opacity disabled:opacity-40"
        >
          {busy ? "Deleting…" : "Delete my account"}
        </button>
      </section>
    </div>
  );
}
```
(`loss` marks a destructive action — the one place this app uses it outside price data, already used for errors and Delete buttons in the admin table.)

- [ ] **Step 2: Typecheck, lint, build (tmux)**

Expected: green; `/account` in the route list.

- [ ] **Step 3: Browser smoke test (Playwright MCP)**

Start `npm run dev` in tmux (`egx-fe-dev`, `~/logs/egx-fe-dev.log`, with `NEXT_PUBLIC_API_BASE_URL` pointing at nothing reachable is fine). With the Playwright MCP tools: open `/login`, `/signup`, `/forgot-password`, `/set-password?token=x`, `/verify-email` (no token) at 390×844; confirm each renders, the links between them work, and client-side validation messages appear (bad email on signup, short password). Screenshot `/signup` and `/login` to the scratchpad. Stop the dev server (`tmux kill-session -t egx-fe-dev`).

- [ ] **Step 4: Commit**

```bash
git add -A src
git commit -m "feat(web): account page with self-service account deletion

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 12: Mobile — shared sync, session plumbing, login by email, admin invites

All paths relative to the **mobile** worktree. `scripts/sync-shared.mjs` reads `../egx-api-fe/src/app/lib` relative to the mobile root; Task 0 created the symlink `.worktrees/egx-api-fe → <fe worktree>` so that path resolves to the fe **worktree**, and `npm run sync:shared` / `npm run verify` work normally on this branch.

**Files:**
- Sync: `src/shared/auth.ts`, `src/shared/api.ts`
- Modify: `src/auth/tokenStore.ts`, `src/auth/AuthProvider.tsx`, `src/features/auth/LoginForm.tsx`, `src/test-utils/render.tsx` (fake user shape)
- Admin: modify `src/features/admin/AdminScreen.tsx`, `UsersList.tsx`; replace `CreateUserSheet.tsx` with `InviteUserSheet.tsx`; delete `PasswordRevealSheet.tsx`, `credentials.ts` (move `formatAdded` into `UsersList.tsx`); update tests in `src/features/admin/__tests__/` and `src/features/auth/__tests__/LoginForm.test.tsx`

**Interfaces:**
- Consumes: Task 8 shared API.
- Produces (`AuthProvider`): `login(email, password)` (throws `LoginError` with `.code`), `completeSignIn(session: Session): Promise<void>`, `deleteAccount(): Promise<void>` (calls `deleteMyAccount`, then the same cleanup as logout minus the push DELETE — the server already removed the devices).

- [ ] **Step 1: Sync shared files** — `npm run sync:shared`, then `npm run typecheck` to list every break.

- [ ] **Step 2: `src/auth/tokenStore.ts`**

Replace `parseUser`'s body with `try { return raw ? parseAuthUser(JSON.parse(raw)) : null; } catch { return null; }` and import `parseAuthUser` (drop `asRole`). Keep the optimistic-only comment.

- [ ] **Step 3: `src/auth/AuthProvider.tsx`**

- Import `deleteMyAccount, fetchMe, LoginError, loginRequest, type Session` from `@/shared/api`.
- `AuthCtx`: `login: (email: string, password: string) => Promise<void>; completeSignIn: (session: Session) => Promise<void>; deleteAccount: () => Promise<void>;`
- Implement:
```tsx
  const completeSignIn = useCallback(async (session: Session) => {
    await saveSession(session.token, session.user);
    setUser(session.user);
    setError(null);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      setError(null);
      try {
        await completeSignIn(await loginRequest(email, password));
      } catch (e) {
        setError(e instanceof LoginError ? e.message : UNREACHABLE);
        throw e;
      }
    },
    [completeSignIn]
  );

  const deleteAccount = useCallback(async () => {
    // The server deletes this phone's push registration with the account,
    // so there is no unregister call here — only local cleanup.
    await deleteMyAccount();
    await clearSession();
    setUser(null);
    setError(null);
  }, []);
```
Add them to the memo value and deps.
- Add tests in `src/auth/__tests__/` (extend the existing AuthProvider test file if present; else create `AuthProvider.email.test.tsx`) using `src/test-utils/fetch.ts`: (a) `login` posts `{email, password}` to `/auth/login` and stores the session; (b) a `403 {detail:"email_not_verified"}` rejects with `LoginError` whose `code === "email_not_verified"`; (c) `deleteAccount` sends `DELETE /auth/me` and signs out.

- [ ] **Step 4: `src/features/auth/LoginForm.tsx`**

```tsx
import { useState } from "react";
import { Pressable, View } from "react-native";
import { Link } from "expo-router";

import { useAuth } from "@/auth/AuthProvider";
import { LoginError, resendVerification } from "@/shared/api";
import { EMAIL_NOT_VERIFIED } from "@/shared/auth";
import { AppText, Button, TextField } from "@/ui";

export function LoginForm() {
  const { login, error } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [unverified, setUnverified] = useState(false);
  const [resent, setResent] = useState(false);

  const canSubmit = email.trim().length > 0 && password.length > 0 && !submitting;

  const onSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setUnverified(false);
    setResent(false);
    try {
      await login(email.trim(), password);
    } catch (e) {
      // `error` from useAuth carries the message; unverified gets its own panel.
      if (e instanceof LoginError && e.code === EMAIL_NOT_VERIFIED) setUnverified(true);
    } finally {
      setSubmitting(false);
    }
  };

  const onResend = async () => {
    try {
      await resendVerification(email.trim());
    } finally {
      setResent(true);
    }
  };

  return (
    <View className="gap-4">
      <TextField
        label="Email"
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="email-address"
        autoComplete="email"
        textContentType="emailAddress"
        returnKeyType="next"
      />
      <TextField
        label="Password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        autoComplete="current-password"
        textContentType="password"
        returnKeyType="go"
        onSubmitEditing={onSubmit}
      />
      <Link href="/forgot-password" asChild>
        <Pressable accessibilityRole="link" className="min-h-[44px] justify-center self-end">
          <AppText className="text-xs text-accent">Forgot password?</AppText>
        </Pressable>
      </Link>
      {unverified ? (
        <View className="gap-2 rounded-lg border border-accent/30 bg-accent/10 px-3 py-2">
          <AppText className="text-xs text-white/80">{`Confirm your email first — we sent a link to ${email.trim()}.`}</AppText>
          {resent ? (
            <AppText className="font-sans-semibold text-xs text-white/80">Sent again.</AppText>
          ) : (
            <Button variant="secondary" label="Resend the link" onPress={() => void onResend()} />
          )}
        </View>
      ) : error ? (
        <View className="rounded-lg border border-loss/30 bg-loss/10 px-3 py-2">
          <AppText className="text-xs text-loss">{error}</AppText>
        </View>
      ) : null}
      <Button label={submitting ? "Signing in…" : "Sign in"} accessibilityLabel="Sign in" onPress={onSubmit} disabled={!canSubmit} loading={submitting} />
    </View>
  );
}
```
The `/forgot-password` route is created in Task 13; until then use `href={"/forgot-password" as never}` with the comment `// typed route lands in Task 13` (Task 13 removes the cast).

Update `src/features/auth/__tests__/LoginForm.test.tsx`: label `Username` → `Email`; the submitted call becomes `login("mark@gmail.com", "pw")` for input `"  mark@gmail.com "` (trimmed, NOT lower-cased — the backend normalises); add a test that a rejected `LoginError("…", "email_not_verified")` shows "Confirm your email first" and a "Resend the link" button that calls `resendVerification`.

- [ ] **Step 5: Admin**

- `git rm src/features/admin/PasswordRevealSheet.tsx src/features/admin/credentials.ts src/features/admin/__tests__/PasswordRevealSheet.test.tsx src/features/admin/__tests__/credentials.test.ts`; move `formatAdded` (copy its body from the deleted `credentials.ts` first) into `UsersList.tsx` as a module function, and its tests into `UsersList.test.tsx`.
- `git mv src/features/admin/CreateUserSheet.tsx src/features/admin/InviteUserSheet.tsx` (and its test). Change it to fields **Email** (`keyboardType="email-address"`, `autoCapitalize="none"`) and **Name (optional)**, no password field, submit label `Send invitation`, helper text "They get an email with a link to choose their password. The link works for 7 days.", prop `onInvite: (email: string, name: string) => Promise<void>`.
- `UsersList.tsx`: prop `onResetPassword` → `onSendReset`, label `Send reset email`; title shows `u.name || u.email` with `u.email` under it when a name exists; accessibility label `${u.email}, show actions`; add method badges (Password / Google / Apple, "Invited" when none, "Unverified" when `!u.email_verified`) styled like `RoleBadge`'s neutral variant (`border-white/10`, `text-white/60`).
- `AdminScreen.tsx`: drop `Revealed`/`pending`/`revealed`/`creating` and the reveal sheet; `handleInvite(email, name)` calls `inviteUser`, closes the sheet, sets a `notice` string `Invitation sent to ${res.user.email}.` rendered as an accent panel above the error; `handleSendReset(u)` confirms (`Email ${u.email} a link to choose a new password?`, confirmLabel `Send`), calls `sendUserReset`, notice `Reset link sent to …` / `Invitation re-sent to …`; delete confirm text uses `u.email`; FAB `accessibilityLabel="Invite user"`.
- Update the admin tests to the new labels and calls (`inviteUser`, `sendUserReset`), and add: inviting shows the notice; a 409 from `inviteUser` shows its message in the sheet.

- [ ] **Step 6: Fix remaining test fixtures**

`grep -rn "username" src` → update every fake user (e.g. `src/test-utils/render.tsx`) to `{ id, email: "mark@example.com", name: "Mark", role }`.

- [ ] **Step 7: Verify (tmux) and commit**

```bash
tmux kill-session -t egx-mobile-verify 2>/dev/null; tmux new -d -s egx-mobile-verify "cd <mobile worktree> && npm run verify 2>&1 | tee ~/logs/egx-mobile-verify.log"
```
Expected: all green (`git checkout tsconfig.json` if lint rewrote it). Then:
```bash
git add -A src
git commit -m "feat(mobile): email sign-in, session plumbing for sign-up/links/social, invite-based admin

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 13: Mobile — sign-up and forgot-password screens

**Files:**
- Create: `src/features/auth/SignupForm.tsx`, `src/features/auth/ForgotPasswordForm.tsx`, `src/features/auth/CheckInbox.tsx`, `src/features/auth/AuthShell.tsx`, routes `src/app/signup.tsx`, `src/app/forgot-password.tsx`
- Modify: `src/navigation/RootNavigator.tsx`, `src/app/login.tsx`, `src/features/auth/LoginForm.tsx` (remove the `as never` cast)
- Tests: `src/features/auth/__tests__/SignupForm.test.tsx`, `ForgotPasswordForm.test.tsx`; extend `src/__tests__/routes/guard.test.tsx`

**Interfaces:**
- Consumes: `signupRequest`, `forgotPassword`, `PASSWORD_MIN_LENGTH`, `looksLikeEmail`.
- Produces: `AuthShell({ title, subtitle?, children, footer? })` (the login route's card chrome, reused by all three routes); `CheckInbox({ email, body })`.

- [ ] **Step 1: Tests first**

`src/features/auth/__tests__/SignupForm.test.tsx` (use `renderWithProviders` and the fetch helper as the existing LoginForm test does):
```tsx
// 1. Submitting with an invalid email shows "Enter a valid email address." and makes no request.
// 2. A 7-character password shows "Use at least 8 characters for your password." and makes no request.
// 3. Valid input POSTs /auth/signup with {name, email, password} (email trimmed) and then shows
//    "Check your inbox" with the email in the text.
// 4. A server error message (e.g. 503 "Email is not configured") is shown in the error panel.
```
`ForgotPasswordForm.test.tsx`:
```tsx
// 1. Valid email POSTs /auth/forgot-password and shows
//    "If an account exists for <email>, a reset link is on its way."
// 2. Invalid email shows the validation message and makes no request.
```
Write these as real RNTL tests in the style of `LoginForm.test.tsx` (`fireEvent.changeText` on `getByLabelText("Email")`, `fireEvent.press(getByRole("button", { name: "Create account" }))`, `await findByText(...)`, assert on the mocked fetch's URL and JSON body).

`guard.test.tsx`: signed out, `/signup` and `/forgot-password` render their screens; signed in, they are not reachable (fall back to the tabs) — mirror the existing `/login` cases.

- [ ] **Step 2: Run to verify failure** — `npx jest src/features/auth src/__tests__/routes/guard.test.tsx` → FAIL (modules missing).

- [ ] **Step 3: Implement**

`src/features/auth/AuthShell.tsx`:
```tsx
import type { ReactNode } from "react";
import { View } from "react-native";

import { AppText, Screen } from "@/ui";

/** The signed-out card: login, sign-up and forgot-password share it. */
export function AuthShell({ title, subtitle, children, footer }: { title: string; subtitle?: string; children: ReactNode; footer?: ReactNode }) {
  return (
    <Screen edges={["top", "bottom"]} contentClassName="flex-grow justify-center">
      <View className="rounded-2xl border border-white/10 bg-charcoal p-6">
        <View className="mb-6 items-center">
          <AppText variant="title" accessibilityRole="header">{title}</AppText>
          {subtitle ? <AppText variant="muted" className="mt-1 text-center">{subtitle}</AppText> : null}
        </View>
        {children}
        {footer ? <View className="mt-6 items-center">{footer}</View> : null}
      </View>
    </Screen>
  );
}
```

`src/features/auth/CheckInbox.tsx`:
```tsx
import { View } from "react-native";
import { AppText } from "@/ui";

export function CheckInbox({ body }: { body: string }) {
  return (
    <View accessibilityRole="summary" className="rounded-lg border border-accent/30 bg-accent/10 px-3 py-3">
      <AppText className="text-sm text-white/80">{body}</AppText>
    </View>
  );
}
```

`src/features/auth/SignupForm.tsx`:
```tsx
import { useState } from "react";
import { View } from "react-native";

import { signupRequest } from "@/shared/api";
import { PASSWORD_MIN_LENGTH, looksLikeEmail } from "@/shared/auth";
import { errorMessage } from "@/lib/errors";
import { AppText, Button, TextField } from "@/ui";

import { CheckInbox } from "./CheckInbox";

export function SignupForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sentTo, setSentTo] = useState<string | null>(null);

  const onSubmit = async () => {
    setErr(null);
    if (!looksLikeEmail(email)) return setErr("Enter a valid email address.");
    if (password.length < PASSWORD_MIN_LENGTH) return setErr(`Use at least ${PASSWORD_MIN_LENGTH} characters for your password.`);
    setBusy(true);
    try {
      await signupRequest(name.trim(), email.trim(), password);
      setSentTo(email.trim());
    } catch (e) {
      setErr(errorMessage(e, "Could not create the account"));
    } finally {
      setBusy(false);
    }
  };

  if (sentTo) {
    return (
      <View className="gap-3">
        <AppText accessibilityRole="header" className="font-sans-semibold text-base text-white">Check your inbox</AppText>
        <CheckInbox body={`We sent a link to ${sentTo}. Open it to finish creating your account, then sign in here. It works for 24 hours.`} />
      </View>
    );
  }

  return (
    <View className="gap-4">
      <TextField label="Name" value={name} onChangeText={setName} autoComplete="name" textContentType="name" maxLength={80} returnKeyType="next" />
      <TextField label="Email" value={email} onChangeText={setEmail} autoCapitalize="none" autoCorrect={false} keyboardType="email-address" autoComplete="email" textContentType="emailAddress" returnKeyType="next" />
      <TextField label={`Password (at least ${PASSWORD_MIN_LENGTH} characters)`} value={password} onChangeText={setPassword} secureTextEntry autoComplete="new-password" textContentType="newPassword" returnKeyType="go" onSubmitEditing={onSubmit} />
      {err ? (
        <View className="rounded-lg border border-loss/30 bg-loss/10 px-3 py-2">
          <AppText className="text-xs text-loss">{err}</AppText>
        </View>
      ) : null}
      <Button label={busy ? "Creating…" : "Create account"} accessibilityLabel="Create account" onPress={onSubmit} disabled={busy} loading={busy} />
    </View>
  );
}
```
(If `TextField`'s label prop doesn't feed `accessibilityLabel`, the tests query by placeholder instead — check `src/ui/TextField.tsx` and follow `LoginForm.test.tsx`.)

`src/features/auth/ForgotPasswordForm.tsx`: same structure — one Email field, button `Send reset link`, on success `<CheckInbox body={\`If an account exists for ${email}, a reset link is on its way. It works for 1 hour. Open it on this phone or any browser.\`} />`.

Routes:
```tsx
// src/app/signup.tsx
import { Link } from "expo-router";
import { Pressable } from "react-native";
import { AuthShell } from "@/features/auth/AuthShell";
import { SignupForm } from "@/features/auth/SignupForm";
import { AppText } from "@/ui";

export default function SignupScreen() {
  return (
    <AuthShell
      title="Create account"
      subtitle="Track your portfolio and watchlist."
      footer={
        <Link href="/login" asChild>
          <Pressable accessibilityRole="link" className="min-h-[44px] justify-center">
            <AppText className="text-sm text-white/50">Already have an account? <AppText className="text-sm text-accent">Sign in</AppText></AppText>
          </Pressable>
        </Link>
      }
    >
      {/* Social buttons are added in Task 14 */}
      <SignupForm />
    </AuthShell>
  );
}
```
`src/app/forgot-password.tsx`: `AuthShell title="Reset password" subtitle="We'll email you a link to choose a new one."` with `<ForgotPasswordForm />` and a "Back to sign in" link footer.
`src/app/login.tsx`: use `AuthShell title="Sign in" subtitle="Access your portfolio and watchlist."` with `<LoginForm />`, footer "New here? Create an account" → `/signup`, and a `{/* Social buttons are added in Task 14 */}` comment above the form.

`RootNavigator.tsx` signed-out group:
```tsx
        <Stack.Protected guard={!isAuthenticated}>
          <Stack.Screen name="login" />
          <Stack.Screen name="signup" />
          <Stack.Screen name="forgot-password" />
        </Stack.Protected>
```
Remove the `as never` cast in `LoginForm.tsx`.

- [ ] **Step 4: Verify (tmux, as Task 12) and commit**

```bash
git add -A src
git commit -m "feat(mobile): sign-up and forgot-password screens

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 14: Mobile — Google and Apple buttons

**Files:**
- Create: `src/lib/runtime.ts`, `src/features/auth/GoogleButton.tsx`, `src/features/auth/AppleButton.tsx`, `src/features/auth/SocialButtons.tsx`, `app.config.ts`
- Modify: `src/features/alerts/push.ts` (re-export `isExpoGo`), `app.json` (`ios.usesAppleSignIn`), `src/app/login.tsx`, `src/app/signup.tsx`, `.env.example`, `jest.setup.ts` / `src/test-utils/` (mocks)
- Tests: `src/features/auth/__tests__/SocialButtons.test.tsx`

**Interfaces:**
- Consumes: `googleSignInRequest`, `appleSignInRequest`, `useAuth().completeSignIn`.
- Produces: `isExpoGo()` in `@/lib/runtime`; `SocialButtons()` — renders Google (not in Expo Go, needs `EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID`) and Apple (iOS, `AppleAuthentication.isAvailableAsync()`), plus an "or" divider only when at least one renders.

- [ ] **Step 1: Install**

```bash
npx expo install @react-native-google-signin/google-signin expo-apple-authentication
```

- [ ] **Step 2: `src/lib/runtime.ts`** and re-export

```ts
import Constants, { ExecutionEnvironment } from "expo-constants";

/** True inside Expo Go, where custom native modules (push, Google sign-in) do not exist. */
export function isExpoGo(): boolean {
  return Constants.executionEnvironment === ExecutionEnvironment.StoreClient;
}
```
In `src/features/alerts/push.ts`, delete the local `isExpoGo` definition and add `export { isExpoGo } from "@/lib/runtime";` (keep the existing doc comment above it). Existing alerts tests must still pass unchanged.

- [ ] **Step 3: App config**

`app.json` → `"ios": { …, "usesAppleSignIn": true }` and add `"expo-apple-authentication"` to `plugins`.

`app.config.ts`:
```ts
import type { ConfigContext, ExpoConfig } from "expo/config";

/**
 * app.json plus build-time values that must not be committed. The Google
 * Sign-In plugin needs the iOS URL scheme (the reversed iOS client id); when
 * GOOGLE_IOS_URL_SCHEME is unset the plugin is skipped and the button stays
 * hidden on iOS (EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID unset too).
 */
export default ({ config }: ConfigContext): ExpoConfig => {
  const scheme = process.env.GOOGLE_IOS_URL_SCHEME;
  const plugins = [...(config.plugins ?? [])];
  if (scheme) plugins.push(["@react-native-google-signin/google-signin", { iosUrlScheme: scheme }]);
  return { ...(config as ExpoConfig), plugins };
};
```
Check: `npx expo config --type public | grep -n "usesAppleSignIn\|google-signin\|apple-authentication"` → Apple entries present; Google plugin present only with the env var set (`GOOGLE_IOS_URL_SCHEME=com.googleusercontent.apps.test npx expo config --type public | grep google-signin`).

- [ ] **Step 4: Mocks for jest**

Create `src/test-utils/socialAuthMock.ts`:
```ts
export const googleMock = {
  GoogleSignin: {
    configure: jest.fn(),
    hasPlayServices: jest.fn(async () => true),
    signIn: jest.fn(async () => ({ type: "success", data: { idToken: "google-id-token", user: { name: "Mark" } } })),
  },
  isSuccessResponse: (r: { type: string }) => r.type === "success",
  isErrorWithCode: () => false,
  statusCodes: { SIGN_IN_CANCELLED: "SIGN_IN_CANCELLED", IN_PROGRESS: "IN_PROGRESS" },
};

export const appleMock = {
  isAvailableAsync: jest.fn(async () => true),
  signInAsync: jest.fn(async () => ({
    identityToken: "apple-identity-token",
    authorizationCode: "apple-code",
    fullName: { givenName: "Mark", familyName: "B" },
  })),
  AppleAuthenticationScope: { FULL_NAME: 0, EMAIL: 1 },
  AppleAuthenticationButton: () => null,
  AppleAuthenticationButtonType: { CONTINUE: 1 },
  AppleAuthenticationButtonStyle: { WHITE: 0 },
};
```
and in `jest.setup.ts`:
```ts
jest.mock("@react-native-google-signin/google-signin", () => jest.requireActual("./src/test-utils/socialAuthMock").googleMock);
jest.mock("expo-apple-authentication", () => jest.requireActual("./src/test-utils/socialAuthMock").appleMock);
```
List the new mock in `CLAUDE.md`'s test-utils line (Task 16).

- [ ] **Step 5: Tests first** — `src/features/auth/__tests__/SocialButtons.test.tsx`:
```tsx
// 1. In Expo Go (mock @/lib/runtime isExpoGo → true) the Google button is not rendered.
// 2. With EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID set and not Expo Go, pressing "Continue with Google"
//    POSTs /auth/google with {idToken: "google-id-token", name: "Mark"} and calls completeSignIn.
// 3. A cancelled Google sign-in (signIn resolves {type: "cancelled"}) makes no request and shows no error.
// 4. On iOS (Platform.OS mocked to "ios") with isAvailableAsync → true, the Apple button is rendered;
//    pressing it POSTs /auth/apple with {identityToken, authorizationCode, name: "Mark B"}.
// 5. On Android the Apple button is not rendered.
// 6. A backend error (401 "Google sign-in failed") is shown under the buttons.
// 7. With neither button available, SocialButtons renders nothing (no "or" divider).
```
Write them as real RNTL tests (set `process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID` in `beforeEach`, restore in `afterEach`; trigger the Apple button by rendering the component's own `Pressable` wrapper, see Step 6).

- [ ] **Step 6: Implement**

`src/features/auth/GoogleButton.tsx`:
```tsx
import { useState } from "react";

import { useAuth } from "@/auth/AuthProvider";
import { isExpoGo } from "@/lib/runtime";
import { errorMessage } from "@/lib/errors";
import { googleSignInRequest } from "@/shared/api";
import { Button } from "@/ui";

const WEB_CLIENT_ID = process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID ?? "";
const IOS_CLIENT_ID = process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID ?? "";

export function googleAvailable(): boolean {
  return !!WEB_CLIENT_ID && !isExpoGo();
}

type GoogleModule = typeof import("@react-native-google-signin/google-signin");

let configured = false;
function google(): GoogleModule {
  // Required lazily: the native module does not exist in Expo Go, and a
  // top-level import would crash the login screen there.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const mod: GoogleModule = require("@react-native-google-signin/google-signin");
  if (!configured) {
    mod.GoogleSignin.configure({ webClientId: WEB_CLIENT_ID, iosClientId: IOS_CLIENT_ID || undefined });
    configured = true;
  }
  return mod;
}

export function GoogleButton({ onError }: { onError: (msg: string | null) => void }) {
  const { completeSignIn } = useAuth();
  const [busy, setBusy] = useState(false);

  const onPress = async () => {
    onError(null);
    setBusy(true);
    try {
      const mod = google();
      await mod.GoogleSignin.hasPlayServices();
      const res = await mod.GoogleSignin.signIn();
      if (!mod.isSuccessResponse(res)) return; // cancelled
      const idToken = res.data.idToken;
      if (!idToken) throw new Error("Google did not return an ID token.");
      await completeSignIn(await googleSignInRequest(idToken, res.data.user.name ?? ""));
    } catch (e) {
      const mod = google();
      if (mod.isErrorWithCode(e) && e.code === mod.statusCodes.SIGN_IN_CANCELLED) return;
      onError(errorMessage(e, "Google sign-in failed"));
    } finally {
      setBusy(false);
    }
  };

  return <Button variant="secondary" label={busy ? "Signing in…" : "Continue with Google"} accessibilityLabel="Continue with Google" onPress={onPress} disabled={busy} loading={busy} />;
}
```
`src/features/auth/AppleButton.tsx`:
```tsx
import * as AppleAuthentication from "expo-apple-authentication";
import { useEffect, useState } from "react";
import { Platform, View } from "react-native";

import { useAuth } from "@/auth/AuthProvider";
import { errorMessage } from "@/lib/errors";
import { appleSignInRequest } from "@/shared/api";

export function useAppleAvailable(): boolean {
  const [available, setAvailable] = useState(false);
  useEffect(() => {
    if (Platform.OS !== "ios") return;
    let live = true;
    AppleAuthentication.isAvailableAsync().then((ok) => live && setAvailable(ok)).catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  return available;
}

export function AppleButton({ onError }: { onError: (msg: string | null) => void }) {
  const { completeSignIn } = useAuth();

  const onPress = async () => {
    onError(null);
    try {
      const cred = await AppleAuthentication.signInAsync({
        requestedScopes: [AppleAuthentication.AppleAuthenticationScope.FULL_NAME, AppleAuthentication.AppleAuthenticationScope.EMAIL],
      });
      if (!cred.identityToken) throw new Error("Apple did not return a token.");
      // Apple shares the name only on the FIRST sign-in, and never in the token.
      const name = [cred.fullName?.givenName, cred.fullName?.familyName].filter(Boolean).join(" ");
      await completeSignIn(
        await appleSignInRequest({ identityToken: cred.identityToken, authorizationCode: cred.authorizationCode, name })
      );
    } catch (e) {
      if ((e as { code?: string })?.code === "ERR_REQUEST_CANCELED") return;
      onError(errorMessage(e, "Apple sign-in failed"));
    }
  };

  return (
    <View accessibilityRole="button" accessibilityLabel="Continue with Apple" className="h-11 w-full">
      <AppleAuthentication.AppleAuthenticationButton
        buttonType={AppleAuthentication.AppleAuthenticationButtonType.CONTINUE}
        buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.WHITE}
        cornerRadius={8}
        style={{ width: "100%", height: 44 }}
        onPress={() => void onPress()}
      />
    </View>
  );
}
```
(For test 4 the mocked `AppleAuthenticationButton` renders null; make the mock render a `Pressable` with `testID="apple-button"` calling `props.onPress` instead — update `socialAuthMock.ts` accordingly using `require("react-native").Pressable` and `React.createElement`.)

`src/features/auth/SocialButtons.tsx`:
```tsx
import { useState } from "react";
import { View } from "react-native";

import { AppText } from "@/ui";

import { AppleButton, useAppleAvailable } from "./AppleButton";
import { GoogleButton, googleAvailable } from "./GoogleButton";

/** Google (dev/store builds only) and Apple (iOS only), then an "or" divider. */
export function SocialButtons() {
  const [err, setErr] = useState<string | null>(null);
  const apple = useAppleAvailable();
  const google = googleAvailable();
  if (!apple && !google) return null;
  return (
    <View className="mb-4 gap-3">
      {apple ? <AppleButton onError={setErr} /> : null}
      {google ? <GoogleButton onError={setErr} /> : null}
      {err ? <AppText className="text-center text-xs text-loss">{err}</AppText> : null}
      <View className="flex-row items-center gap-3 py-1">
        <View className="h-px flex-1 bg-white/10" />
        <AppText className="text-xs text-white/30">or</AppText>
        <View className="h-px flex-1 bg-white/10" />
      </View>
    </View>
  );
}
```
Replace the `{/* Social buttons are added in Task 14 */}` comments in `login.tsx` and `signup.tsx` with `<SocialButtons />`.

`.env.example`: add
```
# Google OAuth client ids. WEB is required for Google sign-in on Android and iOS (it's the
# token audience); IOS is the iOS-type client. Empty WEB hides the button.
EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID=
EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID=
# Build-time only (app.config.ts): reversed iOS client id, e.g. com.googleusercontent.apps.123-abc
GOOGLE_IOS_URL_SCHEME=
```

- [ ] **Step 7: Verify (tmux) and commit**

Also confirm the app still starts in Expo Go terms: `npx expo export --platform ios --output-dir /tmp/claude-1000/-home-mark-projects-egx-api/f3f9fba8-4a93-4720-b415-a22413998dfa/scratchpad/export-check` in tmux (`egx-mobile-export`) succeeds (bundle builds with the lazy require). Delete the export dir after.
```bash
git add -A src app.json app.config.ts .env.example jest.setup.ts package.json package-lock.json
git commit -m "feat(mobile): Continue with Google (dev/store builds) and Apple (iOS)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 15: Mobile — Account screen with delete account

**Files:**
- Create: `src/features/account/AccountScreen.tsx`, `src/features/account/AccountHeaderButton.tsx`, route `src/app/account.tsx`
- Modify: `src/navigation/RootNavigator.tsx`, `src/app/(tabs)/_layout.tsx`, `src/lib/links.ts` (`routes.account`)
- Tests: `src/features/account/__tests__/AccountScreen.test.tsx`, extend `src/__tests__/routes/guard.test.tsx`

**Interfaces:**
- Consumes: `useAuth()` (`user`, `logout`, `deleteAccount`), `displayName`.

- [ ] **Step 1: Tests first**

```tsx
// AccountScreen.test.tsx
// 1. Shows displayName(user) and user.email.
// 2. "Delete my account" is disabled until the field contains exactly DELETE.
// 3. Typing DELETE and pressing it calls deleteAccount once.
// 4. A rejected deleteAccount (409 "You are the last active admin — …") shows the message and stays on screen.
// 5. "Log out" calls logout.
```
guard.test.tsx: `/account` reachable signed in, not signed out.

- [ ] **Step 2: Implement**

`src/features/account/AccountScreen.tsx`:
```tsx
import { useState } from "react";
import { View } from "react-native";

import { useAuth } from "@/auth/AuthProvider";
import { errorMessage } from "@/lib/errors";
import { displayName } from "@/shared/auth";
import { AppText, Button, Screen, TextField } from "@/ui";

const CONFIRM_WORD = "DELETE";

export function AccountScreen() {
  const { user, logout, deleteAccount } = useAuth();
  const [confirmText, setConfirmText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (!user) return null;

  const onDelete = async () => {
    setErr(null);
    setBusy(true);
    try {
      await deleteAccount(); // the root guard moves us to /login
    } catch (e) {
      setErr(errorMessage(e, "Could not delete the account"));
      setBusy(false);
    }
  };

  return (
    <Screen edges={[]}>
      <View className="gap-6 py-6">
        <View className="rounded-xl border border-white/10 bg-charcoal p-4">
          <AppText className="font-sans-semibold text-lg text-white">{displayName(user)}</AppText>
          <AppText className="mt-1 text-sm text-white/50">{user.email}</AppText>
          <View className="mt-4">
            <Button variant="secondary" label="Log out" onPress={() => void logout()} />
          </View>
        </View>

        <View className="rounded-xl border border-loss/30 p-4">
          <AppText accessibilityRole="header" className="font-sans-semibold text-base text-white">Delete account</AppText>
          <AppText className="mt-1 text-sm text-white/50">
            Permanently deletes your account, portfolio, sale and dividend history, watchlist and alert settings. This cannot be undone.
          </AppText>
          <View className="mt-4">
            <TextField label={`Type ${CONFIRM_WORD} to confirm`} value={confirmText} onChangeText={setConfirmText} autoCapitalize="characters" autoCorrect={false} />
          </View>
          {err ? <AppText className="mt-3 text-xs text-loss">{err}</AppText> : null}
          <View className="mt-4">
            <Button variant="danger" label={busy ? "Deleting…" : "Delete my account"} accessibilityLabel="Delete my account" onPress={onDelete} disabled={busy || confirmText !== CONFIRM_WORD} loading={busy} />
          </View>
        </View>
      </View>
    </Screen>
  );
}
```
`src/app/account.tsx`: `export { AccountScreen as default } from "@/features/account/AccountScreen";` (match how `src/app/alerts.tsx` re-exports its screen).

`AccountHeaderButton.tsx`: an `IconButton` with `icon="person-circle-outline"`, `accessibilityLabel="Account"`, `onPress={() => router.push(routes.account)}` (mirror `AlertsHeaderButton.tsx`). Add `account: "/account"` to `routes` in `src/lib/links.ts`.

`(tabs)/_layout.tsx` header: replace the log-out `IconButton` with `<AccountHeaderButton />` (log out now lives on the Account screen; drop the unused `useAuth` import).

`RootNavigator.tsx`: inside the signed-in group add `<Stack.Screen name="account" options={{ ...header, title: "Account" }} />`.

Update any test that pressed the header "Log out" button to go through the Account screen instead.

- [ ] **Step 3: Verify (tmux) and commit**

```bash
git add -A src
git commit -m "feat(mobile): Account screen with self-service account deletion

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 16: Mobile docs and release checklist

**Files:**
- Modify: `CLAUDE.md`, `docs/release.md`

- [ ] **Step 1: `CLAUDE.md`**

- "Everything must RUN in Expo Go" rule: exceptions are now (1) remote push and (2) Google sign-in — `isExpoGo()` lives in `src/lib/runtime.ts`; the Google module is required lazily in `GoogleButton.tsx` and must never be imported at top level. Apple sign-in works in Expo Go (iOS) because the backend accepts the `host.exp.Exponent` audience.
- New `## Accounts` section: sign-in is by email; screens login / signup / forgot-password (signed-out guard) and `/account` (signed-in); emailed verify/reset links open the WEB app, then the user signs in here; `AuthProvider` exposes `login`, `completeSignIn`, `deleteAccount`; `SocialButtons` decides what renders (Apple: iOS + `isAvailableAsync`; Google: not Expo Go + `EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID`); Log out moved from the tab header to the Account screen; Admin = invite by email + send reset email (no passwords shown).
- Test-utils list: add `socialAuthMock`.
- Environment: add `EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID`, `EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID`, build-time `GOOGLE_IOS_URL_SCHEME` (read by `app.config.ts`).
- Remove the Admin paragraph's "Generated passwords are shown once; copy block…" sentence.

- [ ] **Step 2: `docs/release.md`** — add a "Sign-in providers" section: Google Cloud OAuth consent screen; Android OAuth client with the EAS SHA-1 (`npx eas-cli@latest credentials` → Android → keystore SHA-1) and package `com.markbotros.borsacompass`; iOS OAuth client with bundle id; set the three env vars in EAS environments; Apple Developer: enable "Sign in with Apple" on the App ID, then rebuild; Apple requires in-app account deletion (done: Account screen).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/release.md
git commit -m "docs(mobile): accounts, social sign-in and release steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV"
```

---

## Task 17: Final verification across the three branches

- [ ] **Step 1:** Backend full suite in tmux (`egx-be-tests`) → 0 failed. `grep -rn "SET role" app/routers` → nothing.
- [ ] **Step 2:** Web `npx tsc --noEmit && npm run lint` and `npm run build` (tmux `egx-fe-build`) → green.
- [ ] **Step 3:** Mobile `npm run verify` (tmux `egx-mobile-verify`) → green (it includes `check:shared` against the fe worktree).
- [ ] **Step 4:** End-to-end against a local backend: start Docker Postgres (Task 1 command), run the backend with `uvicorn app.main:app --port 8000` in tmux (`egx-be-dev`) with `DATABASE_URL` → that Postgres, `AUTH_SECRET`, `AUTH_ADMINS=admin@example.com`, `FE_BASE_URL=http://localhost:3000`, and SMTP pointed at a local debug server (`python -m aiosmtpd -n -l localhost:1025` in tmux `egx-smtp`, `pip install aiosmtpd` into the venv; set `SMTP_HOST=localhost SMTP_PORT=1025 SMTP_USER=x SMTP_PASSWORD=x`, and for this run only make `SmtpMailer` skip `starttls`/`login` when `SMTP_HOST=localhost` — add that branch with a comment, it is harmless in production). Run the web dev server against it. With Playwright MCP: sign up → read the verify link from the SMTP log → open it → land signed in → log out → forgot password → set-password link → sign in with the new password → Account → delete → signed out, and signing in again fails. Record the result; stop every tmux session and the Docker container.
- [ ] **Step 5:** Report: commits per repo, test counts, and the user's remaining setup checklist (spec §10) plus the rollout order (spec §9).
