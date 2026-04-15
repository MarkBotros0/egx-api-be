"""
Authentication core — password hashing, JWT signing, and FastAPI dependency
for extracting the current user from a Bearer token.

Rotating AUTH_SECRET invalidates every previously issued token because
HS256 signature verification will fail, so every client is forced back
through /api/auth/login.
"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Header, HTTPException, status
from passlib.context import CryptContext


AUTH_SECRET = os.environ.get("AUTH_SECRET", "")
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME_DAYS = 30

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass
class CurrentUser:
    id: str
    username: str


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _pwd_context.verify(password, password_hash)
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
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    secret = _require_secret()
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(authorization: Optional[str] = Header(None)) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    payload = _decode_token(token)
    user_id = payload.get("sub")
    username = payload.get("username")
    if not user_id or not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return CurrentUser(id=user_id, username=username)


def seed_users_from_env(db) -> None:
    """Upsert users from the AUTH_USERS env var.

    Format: `AUTH_USERS=alice:pw1,bob:pw2`

    Each entry is `username:password`. Existing users get their password hash
    refreshed on every boot, so rotating a password is as simple as editing
    the env var and restarting the backend. Users not listed in AUTH_USERS
    are left alone — remove them manually from the DB if needed.
    """
    raw = os.environ.get("AUTH_USERS", "").strip()
    if not raw:
        return

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

        password_hash = hash_password(password)
        existing = db.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            db.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (password_hash, username),
            )
        else:
            db.execute(
                "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), username, password_hash, now),
            )
    db.commit()
