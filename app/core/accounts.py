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
