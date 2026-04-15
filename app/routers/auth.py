"""
Authentication endpoints.

POST /api/auth/login  — exchange username + password for a JWT
GET  /api/auth/me     — return the user identified by the current token

There is no registration endpoint by design — users are seeded from the
`AUTH_USERS` env var at startup (see `app.core.auth.seed_users_from_env`).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import (
    CurrentUser,
    create_access_token,
    get_current_user,
    verify_password,
)
from app.core.db import get_db

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/api/auth/login")
def login(body: LoginRequest):
    username = body.username.lower().strip()
    db = get_db()
    row = db.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if not row or not verify_password(body.password, row[2]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user_id, user_name = row[0], row[1]
    token = create_access_token(user_id, user_name)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user_id, "username": user_name},
    }


@router.get("/api/auth/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return {"id": user.id, "username": user.username}
