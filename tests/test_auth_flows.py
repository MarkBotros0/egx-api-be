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
from app.core.oauth import Identity

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


def test_resignup_inside_the_cooldown_changes_nothing_so_the_victims_link_wins(store, mailer):
    """Reviewer's takeover probe: attacker re-signs-up with their own password within 60 s."""
    _signup(store, mailer)                                   # victim, "correct-horse"
    victim_link = mailer.last_link()
    flows.signup(store, mailer, name="Evil", email="mark@gmail.com",
                 password="attacker-pass", now=NOW + timedelta(seconds=10))
    assert len(mailer.sent) == 1
    flows.verify_email(store, token=victim_link, now=NOW + timedelta(seconds=20))
    assert flows.login(store, email="mark@gmail.com", password="correct-horse")
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="mark@gmail.com", password="attacker-pass")
    assert e.value.status_code == 401
    assert store.by_email("mark@gmail.com").name == "Mark"


def test_resignup_after_the_cooldown_invalidates_the_old_link_and_carries_the_new_password(store, mailer):
    """Documented residual: after 60 s a re-signup replaces the pending credentials,
    but only under a NEW link sent to the inbox, and the old link dies."""
    _signup(store, mailer)
    old = mailer.last_link()
    flows.signup(store, mailer, name="New", email="mark@gmail.com", password="second-pass", now=LATER)
    assert store.by_email("mark@gmail.com").password_hash is None     # account row untouched
    with pytest.raises(HTTPException) as e:
        flows.verify_email(store, token=old, now=LATER)
    assert e.value.detail == "invalid_or_expired_link"
    flows.verify_email(store, token=mailer.last_link(), now=LATER)
    assert flows.login(store, email="mark@gmail.com", password="second-pass")
    assert store.by_email("mark@gmail.com").name == "New"


def test_resignup_of_a_disabled_unverified_account_sends_nothing_and_changes_nothing(store, mailer):
    _signup(store, mailer)
    acc = store.by_email("mark@gmail.com")
    store.update(acc.id, is_active=False)
    before = (dict(store.tokens), store.by_email("mark@gmail.com"))
    flows.signup(store, mailer, name="X", email="mark@gmail.com", password="whatever-pw", now=LATER)
    assert len(mailer.sent) == 1
    assert (dict(store.tokens), store.by_email("mark@gmail.com")) == before


def test_leftover_signup_link_cannot_overwrite_a_password_set_later(store, mailer):
    flows.signup(store, mailer, name="Evil", email="v@x.com", password="attacker-pass", now=NOW)
    link = mailer.last_link()
    flows.forgot_password(store, mailer, email="v@x.com", now=LATER)
    flows.set_password(store, token=mailer.last_link(), password="victim-secret", now=LATER)
    with pytest.raises(HTTPException) as e:
        flows.verify_email(store, token=link, now=LATER)
    assert e.value.detail == "invalid_or_expired_link"
    assert flows.login(store, email="v@x.com", password="victim-secret")
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="v@x.com", password="attacker-pass")
    assert e.value.status_code == 401


def test_verify_email_on_an_already_verified_account_changes_nothing(store):
    a = store.create(email="v@x.com", name="Vic", password_hash=hash_password("victim-secret"),
                     email_verified=True, now=NOW)
    token = store.issue_token(a.id, "verify", NOW, pending_password_hash=hash_password("attacker-pass"),
                              pending_name="Evil")
    acc = flows.verify_email(store, token=token, now=NOW)
    assert acc.id == a.id and acc.name == "Vic"
    assert acc.password_hash == a.password_hash


def test_resend_carries_the_original_pending_password(store, mailer):
    _signup(store, mailer)
    flows.resend_verification(store, mailer, email="mark@gmail.com", now=LATER)
    assert len(mailer.sent) == 2
    flows.verify_email(store, token=mailer.last_link(), now=LATER)
    assert flows.login(store, email="mark@gmail.com", password="correct-horse")
    assert store.by_email("mark@gmail.com").name == "Mark"


def test_resend_with_no_live_verify_token_sends_nothing(store, mailer):
    store.create(email="u@x.com", name="", password_hash=None, email_verified=False, now=NOW)
    flows.resend_verification(store, mailer, email="u@x.com", now=LATER)
    assert mailer.sent == []
    _signup(store, mailer)
    flows.resend_verification(store, mailer, email="mark@gmail.com", now=NOW + timedelta(hours=25))
    assert len(mailer.sent) == 1          # the only link has expired: nothing to carry forward


def test_login_before_verifying_distinguishes_right_and_wrong_pending_password(store, mailer):
    _signup(store, mailer)
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="mark@gmail.com", password="correct-horse", now=LATER)
    assert e.value.status_code == 403 and e.value.detail == "email_not_verified"
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="mark@gmail.com", password="attacker-pass", now=LATER)
    assert e.value.status_code == 401 and e.value.detail == "Invalid email or password"


def test_unknown_email_login_still_runs_a_password_check(store, monkeypatch):
    calls = []
    monkeypatch.setattr(flows, "verify_password", lambda p, h: calls.append(h) or False)
    with pytest.raises(HTTPException):
        flows.login(store, email="ghost@x.com", password="whatever-pw")
    assert calls == [flows._DUMMY_HASH]


def test_set_password_stamps_sessions_valid_after(store, mailer):
    _signup(store, mailer)
    flows.forgot_password(store, mailer, email="mark@gmail.com", now=LATER)
    acc = flows.set_password(store, token=mailer.last_link(), password="brand-new-pw", now=LATER)
    assert acc.sessions_valid_after == "2026-10-12T09:02:00Z"


def test_tokens_issued_before_sessions_valid_after_are_rejected(monkeypatch):
    from app.core import auth
    cutoff = "2026-10-12T09:02:00Z"
    ts = int(datetime(2026, 10, 12, 9, 2, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(auth, "_load_user", lambda uid: auth.CurrentUser(
        id=uid, email="a@x.com", sessions_valid_after=cutoff))
    monkeypatch.setattr(auth, "_decode_token", lambda t: {"sub": "u1", "iat": int(t)})
    with pytest.raises(HTTPException) as e:
        auth.get_current_user(f"Bearer {ts - 1}")
    assert e.value.status_code == 401 and "Session expired" in e.value.detail
    assert auth.get_current_user(f"Bearer {ts}").id == "u1"          # same second is fine
    assert auth.get_current_user(f"Bearer {ts + 5}").id == "u1"


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


def test_oauth_creates_a_verified_account(store):
    acc = flows.oauth_sign_in(store, provider="google",
                              identity=Identity(sub="g1", email="new@x.com", name="New"), now=NOW)
    assert acc.email_verified and acc.google_sub == "g1" and acc.name == "New" and acc.password_hash is None


def test_oauth_finds_by_sub_first(store):
    first = flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a1", email="r@privaterelay.appleid.com", name=""), now=NOW)
    again = flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a1", email=None, name=""), now=NOW)
    assert again.id == first.id


def test_oauth_links_a_verified_password_account_by_email_and_keeps_its_password(store):
    a = store.create(email="mark@gmail.com", name="Mark", password_hash=hash_password("keep-this-pw"), email_verified=True, now=NOW)
    acc = flows.oauth_sign_in(store, provider="google", identity=Identity(sub="g1", email="mark@gmail.com", name="Other"), now=NOW)
    assert acc.id == a.id and acc.google_sub == "g1" and acc.name == "Mark"
    assert flows.login(store, email="mark@gmail.com", password="keep-this-pw")


def test_oauth_takeover_of_an_unverified_account_clears_the_password(store, mailer):
    """Review focus 1."""
    flows.signup(store, mailer, name="", email="victim@x.com", password="attacker-pw", now=NOW)
    acc = flows.oauth_sign_in(store, provider="google", identity=Identity(sub="g9", email="victim@x.com", name="Victim"), now=NOW)
    assert acc.email_verified and acc.password_hash is None and acc.name == "Victim"
    with pytest.raises(HTTPException) as e:
        flows.login(store, email="victim@x.com", password="attacker-pw")
    assert e.value.status_code == 401
    # The pre-registered verify link (carrying the attacker's password) is dead too.
    with pytest.raises(HTTPException) as e:
        flows.verify_email(store, token=mailer.last_link(), now=NOW)
    assert e.value.detail == "invalid_or_expired_link"
    assert store.by_email("victim@x.com").password_hash is None


def test_oauth_disabled_account_is_403(store):
    a = store.create(email="d@x.com", name="", password_hash=None, email_verified=True, now=NOW, google_sub="g1")
    store.update(a.id, is_active=False)
    with pytest.raises(HTTPException) as e:
        flows.oauth_sign_in(store, provider="google", identity=Identity(sub="g1", email="d@x.com", name=""), now=NOW)
    assert e.value.status_code == 403


def test_apple_without_email_and_unknown_sub_is_400(store):
    with pytest.raises(HTTPException) as e:
        flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a-new", email=None, name=""), now=NOW)
    assert e.value.status_code == 400


def test_name_hint_fills_an_empty_name(store):
    acc = flows.oauth_sign_in(store, provider="apple", identity=Identity(sub="a1", email="a@x.com", name=""),
                              name_hint="  Sara  K ", now=NOW)
    assert acc.name == "Sara K"
