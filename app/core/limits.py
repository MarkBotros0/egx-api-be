"""
Abuse limits for the public auth surface, and the nightly pruning of what it
leaves behind. Serverless-safe: every counter lives in Postgres
(`rate_limits`, one atomic upsert per hit — AccountStore.hit), never in
process memory, because each Vercel request may land on a fresh instance.

- Per IP: signup, resend-verification and forgot-password allow
  PUBLIC_EMAIL_IP_LIMIT requests per PUBLIC_EMAIL_IP_WINDOW. Over it the route
  still answers check_email and sends nothing (the answer must not change).
- Global: every send attempt, whatever its purpose (admin invites and resets
  included), counts against EMAIL_DAILY_BUDGET per UTC day. Over it nothing is
  sent; public routes still answer check_email, admin routes 429. Gmail SMTP
  allows a few hundred recipients a day; this keeps a script from spending
  that quota and taking password reset down for everyone.
- Login: LOGIN_FAIL_LIMIT failed attempts per IP per LOGIN_FAIL_WINDOW, then
  429 before the password is even checked.

Windows are fixed (floor(now / window)), not sliding: a burst straddling a
boundary can get up to twice a limit, which is fine for abuse control.
"""

from datetime import datetime, timedelta

from fastapi import HTTPException

PUBLIC_EMAIL_IP_LIMIT = 5
PUBLIC_EMAIL_IP_WINDOW = timedelta(minutes=10)

EMAIL_BUDGET_KEY = "email:global"
EMAIL_DAILY_BUDGET = 300
EMAIL_BUDGET_WINDOW = timedelta(days=1)
EMAIL_BUDGET_EXCEEDED = "Daily email limit reached — try again tomorrow."

LOGIN_FAIL_LIMIT = 10
LOGIN_FAIL_WINDOW = timedelta(minutes=15)
TOO_MANY_ATTEMPTS = "Too many attempts — try again in a few minutes."

# Pruning ages (see prune_auth).
SPENT_TOKEN_AGE = timedelta(days=1)
RATE_LIMIT_AGE = timedelta(days=2)
ABANDONED_ACCOUNT_AGE = timedelta(days=7)


class EmailBudgetExceeded(HTTPException):
    """Raised before a send when the day's budget is spent. Public flows
    swallow it (their answer must not change); admin routes surface the 429."""

    def __init__(self):
        super().__init__(status_code=429, detail=EMAIL_BUDGET_EXCEEDED)


def client_ip(request) -> str:
    """The caller's address: the first X-Forwarded-For entry (Vercel sets it
    to the real client), else the socket peer."""
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


def allow_public_email(store, route: str, ip: str, now: datetime) -> bool:
    """Count this request; False once the IP is over its limit for `route`."""
    return store.hit(f"{route}:{ip}", PUBLIC_EMAIL_IP_WINDOW, now) <= PUBLIC_EMAIL_IP_LIMIT


def spend_email_budget(store, now: datetime) -> None:
    """Count one send attempt against the global daily budget."""
    if store.hit(EMAIL_BUDGET_KEY, EMAIL_BUDGET_WINDOW, now) > EMAIL_DAILY_BUDGET:
        print(f"[email] daily email limit ({EMAIL_DAILY_BUDGET}) reached — not sending")
        raise EmailBudgetExceeded()


def check_login_allowed(store, ip: str, now: datetime) -> None:
    if store.count(f"login:{ip}", LOGIN_FAIL_WINDOW, now) >= LOGIN_FAIL_LIMIT:
        raise HTTPException(status_code=429, detail=TOO_MANY_ATTEMPTS)


def record_login_failure(store, ip: str, now: datetime) -> None:
    store.hit(f"login:{ip}", LOGIN_FAIL_WINDOW, now)


def prune_auth(store, now: datetime) -> dict:
    """Delete auth leftovers; called daily from the /api/pe/refresh cron.

    - auth_tokens used, or expired, more than a day ago;
    - rate_limits windows older than two days;
    - accounts nobody ever used: unverified, no password, no Google/Apple,
      role user, no live link, created over a week ago, owning no rows in any
      user-owned table — junk sign-ups and expired invites.
    """
    return {
        "tokens": store.prune_tokens(now - SPENT_TOKEN_AGE),
        "rate_limits": store.prune_rate_limits(now - RATE_LIMIT_AGE),
        "users": store.prune_abandoned_accounts(now - ABANDONED_ACCOUNT_AGE, now),
    }
