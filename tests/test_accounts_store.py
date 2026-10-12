"""
AccountStore + validation. SQL is checked against a scriptable FakeDB (no
Postgres); real-Postgres behaviour is covered by test_accounts_pg.py.

Run from egx-api-be:  python -m pytest tests/test_accounts_store.py -v
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from alert_fakes import FakeDB
from app.core.accounts import (
    USER_OWNED_TABLES,
    AccountStore,
    clean_name,
    clean_password,
    hash_link_token,
    iso,
    normalize_email,
)

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("raw,expected", [
    ("  Mark@Gmail.COM ", "mark@gmail.com"),
    ("a.b+c@sub.example.org", "a.b+c@sub.example.org"),
])
def test_emails_are_trimmed_and_lowercased(raw, expected):
    assert normalize_email(raw) == expected


@pytest.mark.parametrize("bad", ["", "no-at-sign", "a@b", "a b@c.com", "@x.com", "x@" + "a" * 250 + ".com", None])
def test_invalid_emails_are_rejected(bad):
    with pytest.raises(HTTPException) as e:
        normalize_email(bad)
    assert e.value.status_code == 400


def test_passwords_are_length_checked_and_never_trimmed():
    assert clean_password("  spaced  ") == "  spaced  "
    for bad in ("short", "x" * 129, "", None):
        with pytest.raises(HTTPException):
            clean_password(bad)


def test_names_collapse_whitespace_and_cap_at_80():
    assert clean_name("  Mark   Botros ") == "Mark Botros"
    assert len(clean_name("x" * 200)) == 80
    assert clean_name(None) == ""


def test_iso_is_utc_with_z():
    assert iso(NOW) == "2026-10-12T09:00:00Z"


def test_by_email_queries_the_normalised_value():
    db = FakeDB()
    AccountStore(db).by_email("mark@gmail.com")
    sql, params = db.log[-1]
    assert "WHERE email = %s" in sql and params == ("mark@gmail.com",)


def test_by_provider_rejects_unknown_provider_columns():
    with pytest.raises(ValueError):
        AccountStore(FakeDB()).by_provider("facebook; DROP TABLE users", "x")


def test_update_refuses_role_and_unknown_fields():
    store = AccountStore(FakeDB())
    for bad in ({"role": "admin"}, {"email": "x@y.z"}, {"nope": 1}):
        with pytest.raises(ValueError):
            store.update("u1", **bad)


@pytest.mark.parametrize("bad", [{"role": "admin"}, {"email": "x@y.z"}, {"nope": 1}])
def test_the_memory_fake_refuses_the_same_fields_as_the_real_store(bad):
    # A flow writing a forbidden column must fail its tests, not production.
    from account_fakes import MemoryAccountStore
    store = MemoryAccountStore()
    acc = store.create(email="a@x.com", name="", password_hash=None, email_verified=False,
                       now=datetime(2026, 10, 12, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        store.update(acc.id, **bad)
    assert store.by_id(acc.id) == acc


def test_issue_token_stores_only_the_hash_and_replaces_older_unused():
    db = FakeDB()
    raw = AccountStore(db).issue_token("u1", "reset", NOW)
    stmts = [(s, p) for s, p in db.log]
    delete = next(p for s, p in stmts if s.startswith("DELETE FROM auth_tokens"))
    assert delete == ("u1", "reset")
    insert = next(p for s, p in stmts if "INSERT INTO auth_tokens" in s)
    assert insert[0] == hash_link_token(raw) and raw not in insert
    assert insert[3] == "2026-10-12T10:00:00Z"   # reset = 1 h
    assert insert[5:] == (None, None)


def test_issue_token_stores_pending_credentials():
    db = FakeDB()
    AccountStore(db).issue_token("u1", "verify", NOW, pending_password_hash="H", pending_name="Mark")
    insert = next(p for s, p in db.log if "INSERT INTO auth_tokens" in s)
    assert insert[5:] == ("H", "Mark")


def test_consume_token_is_one_atomic_update():
    db = FakeDB([("UPDATE auth_tokens", [("u1",)])])
    db.responses[0] = ("UPDATE auth_tokens", [("u1", "H", "Mark")])
    assert AccountStore(db).consume_token("raw", {"verify"}, NOW) == ("u1", "H", "Mark")
    sql, params = db.log[-1]
    assert "used_at IS NULL" in sql and "expires_at > %s" in sql
    assert "RETURNING user_id, pending_password_hash, pending_name" in sql
    assert params[1] == hash_link_token("raw") and params[3] == ["verify"]


def test_consume_token_returns_none_when_nothing_matched():
    assert AccountStore(FakeDB()).consume_token("raw", {"verify"}, NOW) is None


def test_latest_pending_reads_the_newest_live_verify_token():
    db = FakeDB([("FROM auth_tokens", [("H", None)])])
    assert AccountStore(db).latest_pending("u1", NOW) == ("H", "")
    sql, params = db.log[-1]
    assert "purpose = %s" in sql and "used_at IS NULL" in sql and "expires_at > %s" in sql
    assert "ORDER BY created_at DESC LIMIT 1" in sql
    assert params == ("u1", "verify", "2026-10-12T09:00:00Z")
    assert AccountStore(FakeDB()).latest_pending("u1", NOW) is None


def test_delete_tokens_only_removes_unused_tokens_of_that_purpose():
    db = FakeDB()
    AccountStore(db).delete_tokens("u1", "verify")
    sql, params = db.log[-1]
    assert sql.startswith("DELETE FROM auth_tokens") and "used_at IS NULL" in sql
    assert params == ("u1", "verify")


def test_sessions_valid_after_is_updatable():
    AccountStore(FakeDB()).update("u1", sessions_valid_after="2026-10-12T09:00:00Z")


def test_delete_account_clears_every_owned_table_then_the_user():
    db = FakeDB()
    AccountStore(db).delete_account("u1")
    tables = [re.match(r"DELETE FROM (\w+)", s).group(1) for s, _ in db.log]
    assert tables == list(USER_OWNED_TABLES) + ["users"]
    # Sales and dividends go before holdings, as the undo path depends on.
    assert tables.index("portfolio_sales") < tables.index("portfolio")


def test_user_owned_tables_matches_every_user_id_table_in_init_db():
    """A table added later with a user_id column must be added to USER_OWNED_TABLES."""
    src = (Path(__file__).resolve().parents[1] / "app" / "core" / "db.py").read_text(encoding="utf-8")
    bodies = re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\s*\)\s*\n?\s*\"\"\"", src, re.S)
    owned = {name for name, body in bodies if re.search(r"^\s*user_id\s+TEXT", body, re.M)}
    assert owned, "regex found no tables — fix the test, not the list"
    assert owned == set(USER_OWNED_TABLES)


def test_methods_lists_the_ways_an_account_can_sign_in():
    from account_fakes import MemoryAccountStore
    s = MemoryAccountStore()
    a = s.create(email="a@b.co", name="", password_hash="h", email_verified=True, now=NOW, google_sub="g")
    assert a.methods == ["password", "google"]
