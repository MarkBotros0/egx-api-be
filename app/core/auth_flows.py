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
