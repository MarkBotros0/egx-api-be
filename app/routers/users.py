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
