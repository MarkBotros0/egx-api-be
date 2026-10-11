"""
Email/password flows against MemoryAccountStore + OutboxMailer.

Run from egx-api-be:  python -m pytest tests/test_auth_flows.py -v
"""

import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-flows-0123456789")

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from account_fakes import MemoryAccountStore, OutboxMailer
from app.core import auth_flows as flows
from app.core.auth import hash_password

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(minutes=2)


@pytest.fixture(autouse=True)
def fe_base(monkeypatch):
    monkeypatch.setenv("FE_BASE_URL", "https://app.test")


@pytest.fixture
def store():
    return MemoryAccountStore()


@pytest.fixture
def mailer():
    return OutboxMailer()


def _signup(store, mailer, email="Mark@Gmail.com ", now=NOW):
    flows.signup(store, mailer, name=" Mark ", email=email, password="correct-horse", now=now)


def test_signup_creates_unverified_and_emails_a_verify_link(store, mailer):
    _signup(store, mailer)
    acc = store.by_email("mark@gmail.com")
    assert acc and not acc.email_verified and acc.name == "Mark"
    assert mailer.sent[-1]["to"] == "mark@gmail.com"
    assert "/verify-email?token=" in mailer.sent[-1]["text"]


def test_login_before_verifying_is_403_email_not_verified(store, mailer):
    _signup(store, mailer)
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="mark@gmail.com", password="correct-horse")
    assert e.value.status_code == 403 and e.value.detail == "email_not_verified"


def test_verify_then_login(store, mailer):
    _signup(store, mailer)
    acc = flows.verify_email(store, token=mailer.last_link(), now=NOW)
    assert acc.email_verified
    assert flows.login(store, email="  MARK@gmail.com", password="correct-horse").id == acc.id


@pytest.mark.parametrize("email,password", [
    ("mark@gmail.com", "wrong-password"),
    ("nobody@gmail.com", "correct-horse"),
])
def test_bad_credentials_share_one_message(store, mailer, email, password):
    _signup(store, mailer)
    flows.verify_email(store, token=mailer.last_link(), now=NOW)
    with pytest.raises(HTTPException) as e:
        flows.login(store, email=email, password=password)
    assert e.value.status_code == 401 and e.value.detail == "Invalid email or password"


def test_account_without_password_cannot_password_login(store):
    store.create(email="g@x.com", name="", password_hash=None, email_verified=True, now=NOW, google_sub="g")
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="g@x.com", password="anything-long")
    assert e.value.status_code == 401


def test_disabled_account_is_403(store):
    a = store.create(email="d@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"), email_verified=True, now=NOW)
    store.update(a.id, is_active=False)
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="d@x.com", password="pw-pw-pw-pw")
    assert e.value.status_code == 403 and "disabled" in e.value.detail


def test_responses_do_not_reveal_whether_an_email_exists(store, mailer):
    """Review focus 2: identical return values for unknown / unverified / verified."""
    store.create(email="v@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"), email_verified=True, now=NOW)
    store.create(email="u@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"), email_verified=False, now=NOW)
    for email in ("new@x.com", "u@x.com", "v@x.com"):
        assert flows.signup(store, mailer, name="", email=email, password="another-pass", now=NOW) == flows.CHECK_EMAIL
        assert flows.resend_verification(store, mailer, email=email, now=LATER) == flows.CHECK_EMAIL
        assert flows.forgot_password(store, mailer, email=email, now=LATER + timedelta(minutes=2)) == flows.CHECK_EMAIL


def test_signup_for_a_verified_email_sends_already_registered_and_keeps_the_password(store, mailer):
    store.create(email="v@x.com", name="", password_hash=hash_password("original-pw"), email_verified=True, now=NOW)
    flows.signup(store, mailer, name="", email="v@x.com", password="attacker-pw", now=NOW)
    assert "already have" in mailer.sent[-1]["subject"]
    assert flows.login(store, email="v@x.com", password="original-pw")


def test_signup_again_while_unverified_replaces_password_and_resends(store, mailer):
    _signup(store, mailer)
    flows.signup(store, mailer, name="New", email="mark@gmail.com", password="second-pass", now=LATER)
    flows.verify_email(store, token=mailer.last_link(), now=LATER)
    assert flows.login(store, email="mark@gmail.com", password="second-pass")
    assert store.by_email("mark@gmail.com").name == "New"


def test_cooldown_sends_nothing_inside_60_seconds(store, mailer):
    _signup(store, mailer)
    flows.resend_verification(store, mailer, email="mark@gmail.com", now=NOW + timedelta(seconds=30))
    assert len(mailer.sent) == 1
    flows.resend_verification(store, mailer, email="mark@gmail.com", now=NOW + timedelta(seconds=61))
    assert len(mailer.sent) == 2


def test_forgot_password_then_set_password_signs_in_and_verifies(store, mailer):
    _signup(store, mailer)                     # unverified
    flows.forgot_password(store, mailer, email="mark@gmail.com", now=LATER)
    acc = flows.set_password(store, token=mailer.last_link(), password="brand-new-pw", now=LATER)
    assert acc.email_verified
    assert flows.login(store, email="mark@gmail.com", password="brand-new-pw").id == acc.id


@pytest.mark.parametrize("case", ["reused", "expired", "wrong_purpose", "superseded", "garbage"])
def test_bad_links_are_invalid_or_expired(store, mailer, case):
    """Review focus 3."""
    _signup(store, mailer)
    token = mailer.last_link()
    if case == "reused":
        flows.verify_email(store, token=token, now=NOW)
        call = lambda: flows.verify_email(store, token=token, now=NOW)
    elif case == "expired":
        call = lambda: flows.verify_email(store, token=token, now=NOW + timedelta(hours=25))
    elif case == "wrong_purpose":
        call = lambda: flows.set_password(store, token=token, password="whatever-pw", now=NOW)
    elif case == "superseded":
        flows.resend_verification(store, mailer, email="mark@gmail.com", now=LATER)
        call = lambda: flows.verify_email(store, token=token, now=LATER)
    else:
        call = lambda: flows.verify_email(store, token="not-a-token", now=NOW)
    with pytest.raises(HTTPException) as e:
        call()
    assert e.value.status_code == 400 and e.value.detail == "invalid_or_expired_link"


def test_send_failure_is_502_and_frees_the_token_and_cooldown(store, mailer):
    mailer.fail = True
    with pytest.raises(HTTPException) as e:
        _signup(store, mailer)
    assert e.value.status_code == 502
    assert store.tokens == {}
    assert store.by_email("mark@gmail.com").last_email_at is None


def test_missing_fe_base_url_is_503(store, mailer, monkeypatch):
    monkeypatch.delenv("FE_BASE_URL")
    with pytest.raises(HTTPException) as e:
        _signup(store, mailer)
    assert e.value.status_code == 503


def test_session_shape(store):
    a = store.create(email="a@x.com", name="A", password_hash=None, email_verified=True, now=NOW)
    s = flows.session(a)
    assert s["token_type"] == "bearer" and s["access_token"]
    assert s["user"] == {"id": a.id, "email": "a@x.com", "name": "A", "role": "user"}
