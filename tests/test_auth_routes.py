"""
HTTP layer for the auth routes: TestClient + dependency overrides (memory
store, outbox mailer). The auth gate middleware runs for real.

Run from egx-api-be:  python -m pytest tests/test_auth_routes.py -v
"""

import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-routes-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pytest
from fastapi.testclient import TestClient

from account_fakes import MemoryAccountStore, OutboxMailer
from app.core.email import get_mailer
from app.main import app
from app.routers.auth import get_store


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setenv("FE_BASE_URL", "https://app.test")
    store, mailer = MemoryAccountStore(), OutboxMailer()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_mailer] = lambda: mailer
    yield TestClient(app), store, mailer
    app.dependency_overrides.clear()


def test_signup_verify_login_round_trip(ctx):
    client, store, mailer = ctx
    r = client.post("/api/auth/signup", json={"name": "Mark", "email": "Mark@Gmail.com", "password": "correct-horse"})
    assert r.status_code == 200 and r.json() == {"status": "check_email"}

    r = client.post("/api/auth/login", json={"email": "mark@gmail.com", "password": "correct-horse"})
    assert r.status_code == 403 and r.json()["detail"] == "email_not_verified"

    r = client.post("/api/auth/verify-email", json={"token": mailer.last_link()})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["user"]["email"] == "mark@gmail.com"

    # /me goes through get_current_user, which reads the real DB — it is
    # covered in test_users_and_roles; here only the login response is checked.
    r = client.post("/api/auth/login", json={"email": " MARK@gmail.com ", "password": "correct-horse"})
    assert r.status_code == 200 and r.json()["user"]["id"] == store.by_email("mark@gmail.com").id


def test_public_routes_need_no_token(ctx):
    client, _, _ = ctx
    for path, body in [
        ("/api/auth/forgot-password", {"email": "x@y.com"}),
        ("/api/auth/resend-verification", {"email": "x@y.com"}),
    ]:
        assert client.post(path, json=body).status_code == 200


def test_bad_link_is_400(ctx):
    client, _, _ = ctx
    r = client.post("/api/auth/set-password", json={"token": "nope", "password": "long-enough-pw"})
    assert r.status_code == 400 and r.json()["detail"] == "invalid_or_expired_link"


def test_me_requires_a_token(ctx):
    client, _, _ = ctx
    assert client.get("/api/auth/me").status_code == 401


def test_forgot_and_resend_answer_identically_for_known_and_unknown_emails(ctx):
    client, store, mailer = ctx
    client.post("/api/auth/signup", json={"name": "M", "email": "known@x.com", "password": "correct-horse"})
    store.update(store.by_email("known@x.com").id, email_verified=True, last_email_at=None)
    for path in ("/api/auth/forgot-password", "/api/auth/resend-verification"):
        known = client.post(path, json={"email": "known@x.com"})
        unknown = client.post(path, json={"email": "nobody@x.com"})
        assert known.status_code == unknown.status_code == 200
        assert known.content == unknown.content
    # The reset email is delivered after the response, by the background task.
    assert any("set-password" in m["text"] for m in mailer.sent)
