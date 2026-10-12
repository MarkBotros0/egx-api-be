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
    _UPDATABLE,
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
        bad = set(fields) - _UPDATABLE          # same guard as AccountStore.update
        if bad:
            raise ValueError(f"not updatable: {sorted(bad)}")
        self.users[user_id] = replace(self.users[user_id], **fields)

    def set_role(self, user_id, role):
        """Test-only: the real store has no way to write a role."""
        self.users[user_id] = replace(self.users[user_id], role=role)

    def issue_token(self, user_id, purpose, now, pending_password_hash=None, pending_name=None):
        for h, row in list(self.tokens.items()):
            if row["user_id"] == user_id and row["purpose"] == purpose and row["used_at"] is None:
                del self.tokens[h]
        raw = secrets.token_urlsafe(32)
        self.tokens[hash_link_token(raw)] = {
            "user_id": user_id, "purpose": purpose,
            "expires_at": iso(now + TOKEN_LIFETIMES[purpose]), "used_at": None,
            "created_at": iso(now),
            "pending_password_hash": pending_password_hash, "pending_name": pending_name,
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
        return (row["user_id"], row["pending_password_hash"], row["pending_name"])

    def latest_pending(self, user_id, now):
        live = [r for r in self.tokens.values()
                if r["user_id"] == user_id and r["purpose"] == "verify" and r["used_at"] is None
                and r["expires_at"] > iso(now) and r["pending_password_hash"] is not None]
        if not live:
            return None
        r = max(live, key=lambda r: r["created_at"])
        return (r["pending_password_hash"], r["pending_name"] or "")

    def delete_tokens(self, user_id, purpose):
        self.tokens = {h: r for h, r in self.tokens.items()
                       if not (r["user_id"] == user_id and r["purpose"] == purpose and r["used_at"] is None)}

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
