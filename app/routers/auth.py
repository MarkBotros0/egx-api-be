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
  POST /api/auth/google               {idToken, name?}         -> session
  POST /api/auth/apple                {identityToken, authorizationCode?, name?} -> session
Signed in:
  GET  /api/auth/me                                            -> user
  DELETE /api/auth/me                                          -> 204 (deletes the account and everything it owns)
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Response
from pydantic import BaseModel

from app.core import auth_flows as flows
from app.core import oauth
from app.core.accounts import AccountStore
from app.core.auth import CurrentUser, get_current_user
from app.core.db import get_db
from app.core.email import BackgroundMailer, Mailer, get_mailer

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
def resend_verification(body: EmailRequest, background: BackgroundTasks, store=Depends(get_store),
                        mailer: Mailer = Depends(get_mailer)):
    # Delivery runs after the response so timing does not reveal whether the email exists.
    return flows.resend_verification(store, BackgroundMailer(mailer, background), email=body.email, now=_now())


@router.post("/api/auth/verify-email")
def verify_email(body: TokenRequest, store=Depends(get_store)):
    return flows.session(flows.verify_email(store, token=body.token, now=_now()))


@router.post("/api/auth/login")
def login(body: LoginRequest, store=Depends(get_store)):
    return flows.session(flows.login(store, email=body.email, password=body.password))


@router.post("/api/auth/forgot-password")
def forgot_password(body: EmailRequest, background: BackgroundTasks, store=Depends(get_store),
                    mailer: Mailer = Depends(get_mailer)):
    return flows.forgot_password(store, BackgroundMailer(mailer, background), email=body.email, now=_now())


@router.post("/api/auth/set-password")
def set_password(body: SetPasswordRequest, store=Depends(get_store)):
    return flows.session(flows.set_password(store, token=body.token, password=body.password, now=_now()))


@router.get("/api/auth/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name, "role": user.role}


@router.delete("/api/auth/me", status_code=204)
def delete_me(user: CurrentUser = Depends(get_current_user), store=Depends(get_store)):
    flows.delete_account(store, user.id)
    return Response(status_code=204)


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
