"""
User administration — the admin tab's backend.

GET    /api/users                  — list every user
POST   /api/users                  — create one, generating a password
POST   /api/users/{id}/password    — reset a password
PATCH  /api/users/{id}             — enable / disable
DELETE /api/users/{id}             — delete the user and everything they own

Every route requires an admin. `role` is NOT settable here: admin status comes
only from the AUTH_ADMINS env var at boot (core/auth.seed_users_from_env), so
privilege escalation through this API is structurally impossible.

A generated password is returned EXACTLY ONCE, in the response to the call that
created it. Only the bcrypt hash is stored, so it can never be read back.
"""

import re
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.auth import (
    ROLE_ADMIN,
    ROLE_USER,
    CurrentUser,
    generate_password,
    hash_password,
    require_admin,
)
from app.core.db import get_db

router = APIRouter()

_USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,32}$")
_MIN_PASSWORD_LENGTH = 8


class UserCreate(BaseModel):
    username: str
    password: Optional[str] = None


class PasswordReset(BaseModel):
    password: Optional[str] = None


class UserPatch(BaseModel):
    is_active: bool


def _clean_username(raw: str) -> str:
    username = (raw or "").strip().lower()
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3–32 characters, using a–z, 0–9, dot, dash or underscore.",
        )
    return username


def _clean_password(raw: Optional[str]) -> tuple:
    """Return (password, was_generated). Generates when none was supplied."""
    if raw is None or not raw.strip():
        return generate_password(), True
    password = raw.strip()
    if len(password) < _MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {_MIN_PASSWORD_LENGTH} characters.",
        )
    return password, False


def _fetch_user(db, user_id: str):
    row = db.execute(
        "SELECT id, username, role, is_active FROM users WHERE id = %s",
        (user_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    return {"id": row[0], "username": row[1], "role": row[2] or ROLE_USER,
            "is_active": bool(row[3])}


def _guard_not_self(target_id: str, admin: CurrentUser, verb: str) -> None:
    if target_id == admin.id:
        raise HTTPException(status_code=400, detail=f"You cannot {verb} your own account.")


def _guard_not_last_admin(db, target: dict, verb: str) -> None:
    """An app with no active admin can never be administered again."""
    if target["role"] != ROLE_ADMIN or not target["is_active"]:
        return
    row = db.execute(
        "SELECT COUNT(*) FROM users WHERE role = %s AND is_active = TRUE",
        (ROLE_ADMIN,),
    ).fetchone()
    if row and int(row[0]) <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"You cannot {verb} the last active admin — no one could administer the app.",
        )


@router.get("/api/users")
def list_users(admin: CurrentUser = Depends(require_admin)):
    try:
        db = get_db()
        # LEFT JOIN so a user with no holdings still appears, and counting only
        # open positions (quantity > 0) to match what the portfolio page shows.
        rows = db.execute(
            "SELECT u.id, u.username, u.role, u.is_active, u.created_at, "
            "       COUNT(p.id) FILTER (WHERE p.quantity > 0) "
            "FROM users u LEFT JOIN portfolio p ON p.user_id = u.id "
            "GROUP BY u.id, u.username, u.role, u.is_active, u.created_at "
            "ORDER BY u.created_at ASC"
        ).fetchall()
        return {
            "users": [
                {
                    "id": r[0],
                    "username": r[1],
                    "role": r[2] or ROLE_USER,
                    "is_active": bool(r[3]),
                    "created_at": r[4],
                    "holdings_count": int(r[5] or 0),
                }
                for r in rows
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/users", status_code=201)
def create_user(body: UserCreate, admin: CurrentUser = Depends(require_admin)):
    try:
        username = _clean_username(body.username)
        password, generated = _clean_password(body.password)

        db = get_db()
        if db.execute(
            "SELECT id FROM users WHERE username = %s", (username,)
        ).fetchone():
            raise HTTPException(status_code=409, detail=f"Username already taken: {username}")

        user_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        db.execute(
            "INSERT INTO users (id, username, password_hash, created_at, role, is_active) "
            "VALUES (%s, %s, %s, %s, %s, TRUE)",
            (user_id, username, hash_password(password), now, ROLE_USER),
        )
        db.commit()

        return {
            "user": {
                "id": user_id,
                "username": username,
                "role": ROLE_USER,
                "is_active": True,
                "created_at": now,
                "holdings_count": 0,
            },
            # Returned once and never again — only the hash is stored.
            "generated_password": password if generated else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/users/{user_id}/password")
def reset_password(
    user_id: str,
    body: PasswordReset,
    admin: CurrentUser = Depends(require_admin),
):
    try:
        password, generated = _clean_password(body.password)
        db = get_db()
        target = _fetch_user(db, user_id)

        db.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(password), user_id),
        )
        db.commit()

        return {
            "id": user_id,
            "username": target["username"],
            "generated_password": password if generated else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/users/{user_id}")
def set_user_active(
    user_id: str,
    body: UserPatch,
    admin: CurrentUser = Depends(require_admin),
):
    try:
        db = get_db()
        target = _fetch_user(db, user_id)

        if not body.is_active:
            _guard_not_self(user_id, admin, "disable")
            _guard_not_last_admin(db, target, "disable")

        db.execute(
            "UPDATE users SET is_active = %s WHERE id = %s",
            (bool(body.is_active), user_id),
        )
        db.commit()
        return {**target, "is_active": bool(body.is_active)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/users/{user_id}")
def delete_user(user_id: str, admin: CurrentUser = Depends(require_admin)):
    try:
        db = get_db()
        target = _fetch_user(db, user_id)
        _guard_not_self(user_id, admin, "delete")
        _guard_not_last_admin(db, target, "delete")

        # No table has a foreign key to `users`, so nothing cascades — without
        # this the rows survive as invisible orphans. Sales and dividends go
        # before holdings, matching the direction the undo path depends on,
        # and it is all one transaction so a half-deleted user is impossible.
        with db.transaction() as tx:
            tx.execute("DELETE FROM portfolio_sales WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM portfolio_dividends WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM portfolio WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM watchlist WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM user_settings WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM users WHERE id = %s", (user_id,))

        return {"deleted": user_id, "username": target["username"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
