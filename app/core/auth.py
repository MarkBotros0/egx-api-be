"""
Authentication core — password hashing, JWT signing, and FastAPI dependency
for extracting the current user from a Bearer token.

Rotating AUTH_SECRET invalidates every previously issued token because
HS256 signature verification will fail, so every client is forced back
through /api/auth/login.
"""

import hashlib
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, status


AUTH_SECRET = os.environ.get("AUTH_SECRET", "")
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME_DAYS = 30

ROLE_USER = "user"
ROLE_ADMIN = "admin"

# Ambiguous glyphs removed: a generated password gets read off a screen and
# typed by hand, so 0/O and 1/l/I cost support time for no entropy worth
# keeping. 16 chars of this alphabet is ~91 bits.
_PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
GENERATED_PASSWORD_LENGTH = 16


@dataclass
class CurrentUser:
    id: str
    username: str
    role: str = ROLE_USER
    is_active: bool = True

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


def generate_password(length: int = GENERATED_PASSWORD_LENGTH) -> str:
    """A random password for an admin to hand to a user. Never stored raw."""
    return "".join(secrets.choice(_PASSWORD_ALPHABET) for _ in range(length))


def _prehash(password: str) -> bytes:
    # SHA-256 → 32 bytes, comfortably under bcrypt's 72-byte input cap.
    # Eliminates length errors for arbitrarily long passwords without
    # silently truncating (which would collide different passwords).
    return hashlib.sha256(password.encode("utf-8")).digest()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("utf-8"))
    except Exception:
        return False


def _require_secret() -> str:
    if not AUTH_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AUTH_SECRET is not configured",
        )
    return AUTH_SECRET


def create_access_token(user_id: str, username: str) -> str:
    secret = _require_secret()
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=TOKEN_LIFETIME_DAYS)).timestamp()),
    }
    # Role and active-state are deliberately NOT claims. Tokens live 30 days,
    # so a claim would mean disabling a user does nothing for a month — see
    # get_current_user, which reads them from the row on every request.
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    secret = _require_secret()
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def _load_user(user_id: str) -> Optional[CurrentUser]:
    """Read the live user row. Returns None if missing or disabled."""
    from app.core.db import get_db

    row = get_db().execute(
        "SELECT id, username, role, is_active FROM users WHERE id = %s",
        (user_id,),
    ).fetchone()
    if not row or not row[3]:
        return None
    return CurrentUser(id=row[0], username=row[1], role=row[2] or ROLE_USER,
                       is_active=bool(row[3]))


def get_current_user(authorization: Optional[str] = Header(None)) -> CurrentUser:
    """
    Resolve the caller, reading role and active-state from the DB.

    The row is re-read on every request rather than trusted from the token.
    Tokens live 30 days, so carrying `role`/`is_active` as claims would mean an
    admin disabling a user changes nothing for a month — "disable" would be a
    lie. This is one indexed primary-key lookup on a pooled connection.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    payload = _decode_token(token)
    user_id = payload.get("sub")
    username = payload.get("username")
    if not user_id or not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = _load_user(user_id)
    if user is None:
        # Same message for "deleted" and "disabled" — the distinction is not
        # the caller's business, and either way they must log in again.
        raise HTTPException(status_code=401, detail="Account is not active")
    return user


def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[CurrentUser]:
    """
    The caller if they present a valid token, otherwise None — never raises.

    /api/analysis needs this: the dashboard is a public page (middleware.ts
    guards only /portfolio), so requiring auth there would break anonymous
    browsing. Anonymous callers score under the global/default weights.
    """
    if not authorization:
        return None
    try:
        return get_current_user(authorization)
    except HTTPException:
        return None


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------------------------------------------------------------------------
# The app-wide gate
# ---------------------------------------------------------------------------

# Every /api/* path is CLOSED unless it appears here. The allowlist is spelled
# as exact (method, path) pairs so a route cannot be opened by accident.
#
# Default-deny is the point: a router added later is locked until someone
# deliberately opens it. The failure mode becomes "it 401s and I notice
# immediately" rather than "it has been serving the whole dataset to the
# internet since the day it shipped" — which is what nine of these endpoints
# were doing before this gate existed.
PUBLIC_ENDPOINTS = frozenset({
    ("POST", "/api/auth/login"),      # the way in
    ("POST", "/api/pe/refresh"),      # scheduled; guarded by PE_REFRESH_SECRET
    ("POST", "/api/cron/risk_snapshot"),  # scheduled; guarded by CRON_SECRET
})


def is_public_endpoint(method: str, path: str) -> bool:
    """True for the handful of routes that must work without a user token."""
    # Preflights carry no Authorization header by design — blocking them breaks
    # CORS for every browser call, including the login request itself.
    if method.upper() == "OPTIONS":
        return True
    return (method.upper(), path.rstrip("/") or "/") in PUBLIC_ENDPOINTS


def authenticate_request(method: str, path: str, authorization: Optional[str]):
    """
    Resolve the caller for a raw request, or return None if the path is open.

    Raises HTTPException(401) when a closed path has no valid, active user.
    Shared by the ASGI middleware and its tests so the policy is stated once.
    """
    if is_public_endpoint(method, path):
        return None
    return get_current_user(authorization)


def _parse_admin_usernames() -> set:
    raw = os.environ.get("AUTH_ADMINS", "").strip()
    return {u.strip().lower() for u in raw.split(",") if u.strip()}


def seed_users_from_env(db) -> None:
    """Create users from AUTH_USERS and stamp admin roles from AUTH_ADMINS.

    Format: `AUTH_USERS=alice:pw1,bob:pw2` — each entry `username:password`.
            `AUTH_ADMINS=alice` — comma-separated usernames.

    Passwords here are for BOOTSTRAP only. An existing user's password hash is
    never touched: admins reset passwords through /api/users, and rewriting the
    hash on every boot (as this used to do) would silently revert every reset
    on the next cold start. To change a password, use the admin tab.

    AUTH_ADMINS is authoritative for admin status and is re-applied on every
    boot, so it survives a DB reset and you cannot lock yourself out by
    fumbling a role in the database. Demotion of unlisted users only happens
    when AUTH_ADMINS is non-empty — a blank or unset var must never strip
    every admin and leave the app unmanageable.
    """
    admins = _parse_admin_usernames()
    raw = os.environ.get("AUTH_USERS", "").strip()

    now = datetime.utcnow().isoformat() + "Z"
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        username, password = entry.split(":", 1)
        username = username.strip().lower()
        password = password.strip()
        if not username or not password:
            continue

        existing = db.execute(
            "SELECT id FROM users WHERE username = %s", (username,)
        ).fetchone()
        if not existing:
            db.execute(
                "INSERT INTO users (id, username, password_hash, created_at, role, is_active) "
                "VALUES (%s, %s, %s, %s, %s, TRUE)",
                (str(uuid.uuid4()), username, hash_password(password), now,
                 ROLE_ADMIN if username in admins else ROLE_USER),
            )

    if admins:
        db.execute(
            "UPDATE users SET role = %s WHERE username = ANY(%s) AND role <> %s",
            (ROLE_ADMIN, list(admins), ROLE_ADMIN),
        )
        db.execute(
            "UPDATE users SET role = %s WHERE NOT (username = ANY(%s)) AND role <> %s",
            (ROLE_USER, list(admins), ROLE_USER),
        )

    db.commit()
