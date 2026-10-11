"""
AccountStore against a REAL Postgres. Skipped unless TEST_DATABASE_URL is set.

    docker run -d --rm --name egx-auth-pg -e POSTGRES_PASSWORD=pw -p 55432:5432 postgres:16
    TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55432/postgres \
        python -m pytest tests/test_accounts_pg.py -v
    docker stop egx-auth-pg

Never point this at a real database: it drops tables.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    from app.core import db as db_mod
    from app.core.accounts import USER_OWNED_TABLES, AccountStore

    monkeypatch.setattr(db_mod, "_pool", None)
    monkeypatch.setattr(db_mod, "_initialized", True)
    d = db_mod._DB(db_mod._get_pool())
    for t in (*USER_OWNED_TABLES, "users"):
        d.execute(f"DROP TABLE IF EXISTS {t}")
    db_mod.init_db(d)
    return AccountStore(d)


def test_create_find_update(store):
    a = store.create(email="a@b.co", name="A", password_hash="h", email_verified=False, now=NOW)
    assert store.by_email("a@b.co").id == a.id
    store.update(a.id, email_verified=True, google_sub="g1")
    assert store.by_provider("google", "g1").email_verified is True
    assert store.users_table_is_legacy() is False


def test_tokens_expire_are_single_use_and_replaced(store):
    a = store.create(email="a@b.co", name="", password_hash=None, email_verified=False, now=NOW)
    old = store.issue_token(a.id, "verify", NOW)
    new = store.issue_token(a.id, "verify", NOW)
    assert store.consume_token(old, {"verify"}, NOW) is None
    assert store.consume_token(new, {"reset"}, NOW) is None
    assert store.consume_token(new, {"verify"}, NOW) == (a.id, None, None)
    assert store.consume_token(new, {"verify"}, NOW) is None
    late = store.issue_token(a.id, "reset", NOW)
    assert store.consume_token(late, {"reset"}, NOW + timedelta(hours=2)) is None


def test_delete_account_removes_tokens_and_row(store):
    a = store.create(email="a@b.co", name="", password_hash=None, email_verified=False, now=NOW)
    store.issue_token(a.id, "verify", NOW)
    store.delete_account(a.id)
    assert store.by_id(a.id) is None


def test_pending_credentials_ride_the_verify_token(store):
    a = store.create(email="p@b.co", name="", password_hash=None, email_verified=False, now=NOW)
    assert store.latest_pending(a.id, NOW) is None
    first = store.issue_token(a.id, "verify", NOW, pending_password_hash="H1", pending_name="One")
    assert store.latest_pending(a.id, NOW) == ("H1", "One")
    second = store.issue_token(a.id, "verify", NOW + timedelta(minutes=2),
                               pending_password_hash="H2", pending_name="Two")
    assert store.consume_token(first, {"verify"}, NOW) is None
    assert store.latest_pending(a.id, NOW + timedelta(minutes=3)) == ("H2", "Two")
    assert store.latest_pending(a.id, NOW + timedelta(hours=25)) is None
    store.delete_tokens(a.id, "verify")
    assert store.consume_token(second, {"verify"}, NOW + timedelta(minutes=3)) is None
    third = store.issue_token(a.id, "verify", NOW, pending_password_hash="H3", pending_name="")
    assert store.consume_token(third, {"verify"}, NOW) == (a.id, "H3", "")
    assert store.latest_pending(a.id, NOW) is None    # used


def test_sessions_valid_after_round_trips(store):
    a = store.create(email="s@b.co", name="", password_hash=None, email_verified=True, now=NOW)
    assert a.sessions_valid_after is None
    store.update(a.id, sessions_valid_after="2026-10-12T09:00:00Z")
    assert store.by_id(a.id).sessions_valid_after == "2026-10-12T09:00:00Z"
