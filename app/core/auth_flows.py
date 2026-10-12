"""
Every account flow, as plain functions over an AccountStore-shaped `store`
and a Mailer. Routers adapt HTTP to these; tests drive them directly with
tests/account_fakes.

Public flows that take an email (signup, resend, forgot) ALWAYS return
CHECK_EMAIL, whatever exists — the response must not reveal which emails
have accounts. Errors that do leak (login's email_not_verified) need the
right password first.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException

from app.core import email as mail
from app.core import oauth
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
# Verified against when there is no real hash, so a wrong email costs the same
# bcrypt time as a wrong password.
_DUMMY_HASH = hash_password("dummy-password-for-timing")


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
              inviter: str = "", pending_password_hash: Optional[str] = None,
              pending_name: Optional[str] = None) -> bool:
    """Email a fresh link of `purpose`. False (and nothing sent) inside the cooldown."""
    if _in_cooldown(account, now):
        return False
    path = "/verify-email" if purpose == PURPOSE_VERIFY else "/set-password"
    mail.link(path, "probe")   # 503 before a token exists if FRONTEND_ORIGIN is unset
    token = store.issue_token(account.id, purpose, now, pending_password_hash, pending_name)
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
    """
    The password and name asked for live on the emailed verify link, never on
    the account: whoever clicks the link proves the inbox AND chooses the
    password, so a stranger re-signing-up with a victim's address cannot
    swap in their own password under the victim's pending link.
    """
    email = normalize_email(email)
    password = clean_password(password)
    name = clean_name(name)
    existing = store.by_email(email)
    if existing is None:
        account = store.create(email=email, name=name, password_hash=None,
                               email_verified=False, now=now)
        send_link(store, mailer, account, PURPOSE_VERIFY, now,
                  pending_password_hash=hash_password(password), pending_name=name)
    elif not existing.email_verified:
        if existing.is_active:
            # send_link is a no-op inside the cooldown, leaving the earlier link untouched.
            send_link(store, mailer, existing, PURPOSE_VERIFY, now,
                      pending_password_hash=hash_password(password), pending_name=name)
    elif not _in_cooldown(existing, now):
        message = mail.already_registered_message(existing.name, mail.login_url(), mail.forgot_url())
        _deliver(store, mailer, existing, now, message, None)
    return CHECK_EMAIL


def resend_verification(store, mailer, *, email: str, now: datetime) -> dict:
    account = store.by_email(normalize_email(email))
    if account and account.is_active and not account.email_verified:
        pending = store.latest_pending(account.id, now)
        if pending:
            send_link(store, mailer, account, PURPOSE_VERIFY, now,
                      pending_password_hash=pending[0], pending_name=pending[1])
    return CHECK_EMAIL


def forgot_password(store, mailer, *, email: str, now: datetime) -> dict:
    account = store.by_email(normalize_email(email))
    if account and account.is_active:
        send_link(store, mailer, account, PURPOSE_RESET, now)
    return CHECK_EMAIL


def _redeem(store, token: str, purposes: set, now: datetime) -> tuple:
    consumed = store.consume_token(token, purposes, now)
    account = store.by_id(consumed[0]) if consumed else None
    if account is None:
        raise HTTPException(status_code=400, detail=INVALID_LINK)
    if not account.is_active:
        raise HTTPException(status_code=403, detail=DISABLED)
    return account, consumed[1], consumed[2]


def verify_email(store, *, token: str, now: datetime) -> Account:
    account, pending_hash, pending_name = _redeem(store, token, {PURPOSE_VERIFY}, now)
    if account.email_verified:
        # A leftover sign-up link must never overwrite credentials set since.
        return account
    fields = {"email_verified": True}
    if pending_hash is not None:
        fields["password_hash"] = pending_hash
    if pending_name:
        fields["name"] = pending_name
    store.update(account.id, **fields)
    return store.by_id(account.id)


def set_password(store, *, token: str, password: str, now: datetime) -> Account:
    password = clean_password(password)           # validate BEFORE burning the link
    account, _, _ = _redeem(store, token, {PURPOSE_RESET, PURPOSE_INVITE}, now)
    # Clicking an emailed link proves the inbox, so this also verifies. It also
    # ends every session issued before now (the old password's holders).
    store.update(account.id, password_hash=hash_password(password), email_verified=True,
                 sessions_valid_after=iso(now))
    # No sign-up link carrying pending credentials may outlive a password being set.
    store.delete_tokens(account.id, PURPOSE_VERIFY)
    return store.by_id(account.id)


def login(store, *, email: str, password: str, now: Optional[datetime] = None) -> Account:
    now = now or datetime.now(timezone.utc)
    try:
        email = normalize_email(email)
    except HTTPException:
        raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)
    account = store.by_email(email)
    password = password or ""
    stored = account.password_hash if account else None
    if account and not stored and not account.email_verified:
        # Sign-up parks the password on the verify link; a right password
        # there earns the "verify your email" prompt, anything else is a 401.
        pending = store.latest_pending(account.id, now)
        stored = pending[0] if pending else None
    ok = verify_password(password, stored or _DUMMY_HASH) and bool(stored)
    if not account or not ok:
        raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)
    if not account.is_active:
        raise HTTPException(status_code=403, detail=DISABLED)
    if not account.email_verified:
        raise HTTPException(status_code=403, detail="email_not_verified")
    return account


def oauth_sign_in(store, *, provider: str, identity, name_hint: str = "", now: datetime) -> Account:
    """Resolve a verified Google/Apple identity to an account (spec §3)."""
    name = clean_name(identity.name or name_hint)
    account = store.by_provider(provider, identity.sub)
    if account is None:
        if not identity.email:
            raise HTTPException(status_code=400, detail="Your Apple ID did not share an email address.")
        email = normalize_email(identity.email)
        account = store.by_email(email)
        if account is None:
            return store.create(email=email, name=name, password_hash=None, email_verified=True,
                                now=now, **{f"{provider}_sub": identity.sub})
    if not account.is_active:
        raise HTTPException(status_code=403, detail=DISABLED)   # before any write
    fields = {}
    if getattr(account, f"{provider}_sub") != identity.sub:
        fields[f"{provider}_sub"] = identity.sub
    if not account.email_verified:
        # The provider just proved the inbox. A password set by whoever
        # pre-registered this email must not survive that...
        fields.update(password_hash=None, email_verified=True)
        # ...nor a pending sign-up link carrying the password it would set...
        store.delete_tokens(account.id, PURPOSE_VERIFY)
        # ...nor the name they chose.
        if name:
            fields["name"] = name
    if "name" not in fields and not account.name and name:
        fields["name"] = name
    if fields:
        store.update(account.id, **fields)
        account = store.by_id(account.id)
    return account


def delete_account(store, account_id: str, *, revoke=None) -> None:
    account = store.by_id(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="User not found")
    if account.role == "admin" and account.is_active and store.count_active_admins() <= 1:
        raise HTTPException(
            status_code=409,
            detail="You are the last active admin — no one could administer the app.")
    store.delete_account(account.id)
    if account.apple_refresh_token:
        (revoke or oauth.revoke_apple_token)(account.apple_refresh_token)


def invite(store, mailer, *, email: str, name: str, inviter: str, now: datetime) -> Account:
    email = normalize_email(email)
    if store.by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    account = store.create(email=email, name=clean_name(name), password_hash=None,
                           email_verified=False, now=now)
    send_link(store, mailer, account, PURPOSE_INVITE, now, inviter=inviter)
    return store.by_id(account.id)


def admin_send_reset(store, mailer, *, user_id: str, now: datetime) -> str:
    account = store.by_id(user_id)
    if account is None:
        raise HTTPException(status_code=404, detail="User not found")
    purpose = PURPOSE_INVITE if (not account.password_hash and not account.email_verified) else PURPOSE_RESET
    if not send_link(store, mailer, account, purpose, now):
        raise HTTPException(status_code=429, detail="An email was sent less than a minute ago.")
    return purpose
