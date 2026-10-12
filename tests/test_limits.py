"""
Abuse limits on the public auth routes (per-IP), the global daily email
budget, login throttling, and the daily pruning of auth leftovers.

Run from egx-api-be:  python -m pytest tests/test_limits.py -v
"""

import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-limits-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from account_fakes import MemoryAccountStore, OutboxMailer
from alert_fakes import FakeDB
from app.core import auth_flows as flows
from app.core import limits
from app.core.accounts import USER_OWNED_TABLES, AccountStore
from app.core.auth import hash_password
from app.core.email import get_mailer
from app.main import app
from app.routers.auth import get_store

NOW = datetime(2026, 10, 12, 9, 7, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frontend(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.test")


@pytest.fixture
def store():
    return MemoryAccountStore()


@pytest.fixture
def mailer():
    return OutboxMailer()


@pytest.fixture
def ctx(store, mailer):
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_mailer] = lambda: mailer
    yield TestClient(app), store, mailer
    app.dependency_overrides.clear()


# -- the counter --------------------------------------------------------------

def test_hit_is_one_atomic_upsert_on_the_floored_window():
    db = FakeDB([("INSERT INTO rate_limits", [(3,)])])
    assert AccountStore(db).hit("signup:1.2.3.4", timedelta(minutes=10), NOW) == 3
    sql, params = db.statements("rate_limits")[-1]
    assert "ON CONFLICT (key, window_start) DO UPDATE SET count = rate_limits.count + 1" in sql
    assert "RETURNING count" in sql
    assert params == ("signup:1.2.3.4", "2026-10-12T09:00:00Z")


def test_a_day_window_starts_at_utc_midnight():
    db = FakeDB([("INSERT INTO rate_limits", [(1,)])])
    AccountStore(db).hit("email:global", timedelta(days=1), NOW)
    assert db.statements("rate_limits")[-1][1] == ("email:global", "2026-10-12T00:00:00Z")


def test_count_reads_without_incrementing():
    db = FakeDB([("SELECT count FROM rate_limits", [(7,)])])
    assert AccountStore(db).count("login:1.2.3.4", timedelta(minutes=15), NOW) == 7
    sql, params = db.log[-1]
    assert sql.startswith("SELECT") and params == ("login:1.2.3.4", "2026-10-12T09:00:00Z")
    assert AccountStore(FakeDB()).count("x", timedelta(minutes=15), NOW) == 0


def test_init_db_creates_the_rate_limits_table():
    from app.core.db import init_db
    db = FakeDB()
    init_db(db)
    ddl = " ".join(s for s, _ in db.statements("CREATE TABLE IF NOT EXISTS rate_limits"))
    assert "count INTEGER NOT NULL" in ddl and "PRIMARY KEY (key, window_start)" in ddl


def test_memory_store_counts_per_window(store):
    w = timedelta(minutes=10)
    assert [store.hit("k", w, NOW) for _ in range(3)] == [1, 2, 3]
    assert store.count("k", w, NOW) == 3
    assert store.hit("k", w, NOW + timedelta(minutes=10)) == 1     # next window
    assert store.count("other", w, NOW) == 0


# -- client IP ----------------------------------------------------------------

def _request(headers=None, host="9.9.9.9"):
    return SimpleNamespace(headers=headers or {}, client=SimpleNamespace(host=host) if host else None)


def test_client_ip_is_the_first_forwarded_entry():
    assert limits.client_ip(_request({"x-forwarded-for": " 1.2.3.4 , 10.0.0.1"})) == "1.2.3.4"


def test_client_ip_falls_back_to_the_socket_peer():
    assert limits.client_ip(_request()) == "9.9.9.9"
    assert limits.client_ip(_request(host=None)) == "unknown"


# -- per-IP limits on the routes that send email ------------------------------

def _body(path, i):
    if path == "/api/auth/signup":
        return {"name": "", "email": f"u{i}@x.com", "password": "correct-horse"}
    return {"email": f"u{i}@x.com"}


@pytest.mark.parametrize("path", ["/api/auth/signup", "/api/auth/resend-verification",
                                  "/api/auth/forgot-password"])
def test_a_sixth_request_from_one_ip_in_ten_minutes_sends_nothing(ctx, path):
    client, store, mailer = ctx
    now = datetime.now(timezone.utc)
    if path != "/api/auth/signup":
        for i in range(6):
            u = store.create(email=f"u{i}@x.com", name="", password_hash=None,
                             email_verified=path == "/api/auth/forgot-password", now=now)
            if path == "/api/auth/resend-verification":
                store.issue_token(u.id, "verify", now, pending_password_hash="h")
    hdr = {"X-Forwarded-For": "1.2.3.4"}
    for i in range(5):
        r = client.post(path, json=_body(path, i), headers=hdr)
        assert r.status_code == 200 and r.json() == {"status": "check_email"}
    sent = len(mailer.sent)
    assert sent == 5
    r = client.post(path, json=_body(path, 5), headers=hdr)
    assert r.status_code == 200 and r.json() == {"status": "check_email"}
    assert len(mailer.sent) == sent
    if path == "/api/auth/signup":
        assert store.by_email("u5@x.com") is None
    # another address is unaffected
    r = client.post(path, json=_body(path, 5), headers={"X-Forwarded-For": "5.6.7.8"})
    assert r.status_code == 200 and len(mailer.sent) == sent + 1


# -- the global daily email budget --------------------------------------------

def _spend_budget(store, now=NOW):
    for _ in range(limits.EMAIL_DAILY_BUDGET):
        store.hit(limits.EMAIL_BUDGET_KEY, limits.EMAIL_BUDGET_WINDOW, now)


def test_every_send_attempt_counts_against_the_budget(store, mailer):
    flows.signup(store, mailer, name="", email="a@x.com", password="correct-horse", now=NOW)
    admin_target = flows.invite(store, mailer, email="i@x.com", name="", inviter="", now=NOW)
    flows.admin_send_reset(store, mailer, user_id=admin_target.id, now=NOW + timedelta(minutes=2))
    mailer.fail = True
    with pytest.raises(HTTPException):
        flows.invite(store, mailer, email="j@x.com", name="", inviter="", now=NOW)
    assert store.count(limits.EMAIL_BUDGET_KEY, limits.EMAIL_BUDGET_WINDOW, NOW) == 4


def test_over_budget_public_flows_send_nothing_and_still_say_check_email(store, mailer, capsys):
    store.create(email="v@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"),
                 email_verified=True, now=NOW)
    _spend_budget(store)
    assert flows.signup(store, mailer, name="", email="new@x.com", password="correct-horse",
                        now=NOW) == flows.CHECK_EMAIL
    assert flows.signup(store, mailer, name="", email="v@x.com", password="correct-horse",
                        now=NOW) == flows.CHECK_EMAIL
    assert flows.forgot_password(store, mailer, email="v@x.com", now=NOW) == flows.CHECK_EMAIL
    assert mailer.sent == [] and store.tokens == {}
    assert store.by_email("v@x.com").last_email_at is None
    assert "daily email limit" in capsys.readouterr().out.lower()


def test_over_budget_resend_sends_nothing_and_keeps_the_live_link(store, mailer):
    flows.signup(store, mailer, name="", email="new@x.com", password="correct-horse", now=NOW)
    link = mailer.last_link()
    _spend_budget(store)
    later = NOW + timedelta(minutes=2)
    assert flows.resend_verification(store, mailer, email="new@x.com", now=later) == flows.CHECK_EMAIL
    flows.signup(store, mailer, name="", email="new@x.com", password="other-pass", now=later)
    assert len(mailer.sent) == 1
    flows.verify_email(store, token=link, now=later)          # the first link still works
    assert flows.login(store, email="new@x.com", password="correct-horse", now=later)


def test_over_budget_admin_routes_answer_429(store, mailer):
    target = store.create(email="t@x.com", name="", password_hash=hash_password("pw-pw-pw-pw"),
                          email_verified=True, now=NOW)
    _spend_budget(store)
    for call in (lambda: flows.invite(store, mailer, email="i@x.com", name="", inviter="", now=NOW),
                 lambda: flows.admin_send_reset(store, mailer, user_id=target.id, now=NOW)):
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == 429
        assert e.value.detail == "Daily email limit reached — try again tomorrow."
    assert mailer.sent == [] and store.tokens == {}


def test_the_budget_resets_at_utc_midnight(store, mailer):
    _spend_budget(store)
    tomorrow = datetime(2026, 10, 13, 0, 0, 1, tzinfo=timezone.utc)
    flows.signup(store, mailer, name="", email="new@x.com", password="correct-horse", now=tomorrow)
    assert len(mailer.sent) == 1


# -- login throttling ---------------------------------------------------------

def _login(client, password, ip="1.2.3.4"):
    return client.post("/api/auth/login", json={"email": "v@x.com", "password": password},
                       headers={"X-Forwarded-For": ip})


def test_ten_failed_logins_from_one_ip_lock_it_out_before_the_password_is_checked(ctx):
    client, store, _ = ctx
    store.create(email="v@x.com", name="", password_hash=hash_password("right-password"),
                 email_verified=True, now=NOW)
    for _ in range(limits.LOGIN_FAIL_LIMIT):
        assert _login(client, "wrong-password").status_code == 401
    r = _login(client, "right-password")
    assert r.status_code == 429 and r.json()["detail"] == "Too many attempts — try again in a few minutes."
    assert _login(client, "right-password", ip="5.6.7.8").status_code == 200


def test_successful_logins_do_not_count(ctx):
    client, store, _ = ctx
    store.create(email="v@x.com", name="", password_hash=hash_password("right-password"),
                 email_verified=True, now=NOW)
    for _ in range(limits.LOGIN_FAIL_LIMIT + 2):
        assert _login(client, "right-password").status_code == 200


def test_login_limit_helpers(store):
    for _ in range(limits.LOGIN_FAIL_LIMIT - 1):
        limits.record_login_failure(store, "ip", NOW)
    limits.check_login_allowed(store, "ip", NOW)            # 9 failures: still allowed
    limits.record_login_failure(store, "ip", NOW)
    with pytest.raises(HTTPException) as e:
        limits.check_login_allowed(store, "ip", NOW)
    assert e.value.status_code == 429
    limits.check_login_allowed(store, "ip", NOW + limits.LOGIN_FAIL_WINDOW)   # next window


# -- pruning ------------------------------------------------------------------

def test_prune_tokens_sql_drops_used_or_expired_more_than_a_day_ago():
    db = FakeDB()
    AccountStore(db).prune_tokens(datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc))
    sql, params = db.statements("DELETE FROM auth_tokens")[-1]
    assert "used_at < %s" in sql and "expires_at < %s" in sql
    assert params == ("2026-10-11T09:00:00Z", "2026-10-11T09:00:00Z")


def test_prune_rate_limits_sql():
    db = FakeDB()
    AccountStore(db).prune_rate_limits(datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc))
    sql, params = db.statements("DELETE FROM rate_limits")[-1]
    assert "window_start < %s" in sql and params == ("2026-10-10T09:00:00Z",)


def test_prune_abandoned_accounts_sql_spares_anything_anyone_used():
    db = FakeDB([("DELETE FROM users", [("u1",), ("u2",)])])
    n = AccountStore(db).prune_abandoned_accounts(
        datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc), NOW)
    assert n == 2
    sql, params = db.statements("DELETE FROM users")[-1]
    for clause in ("email_verified = FALSE", "password_hash IS NULL", "google_sub IS NULL",
                   "apple_sub IS NULL", "role = 'user'", "created_at < %s"):
        assert clause in sql
    for table in USER_OWNED_TABLES:
        if table != "auth_tokens":
            assert f"FROM {table} " in sql, table
    assert "used_at IS NULL AND t.expires_at > %s" in sql      # a live link spares it
    assert params == ("2026-10-05T09:00:00Z", "2026-10-12T09:07:30Z")
    leftover = db.statements("DELETE FROM auth_tokens WHERE user_id = ANY")
    assert leftover and leftover[-1][1] == (["u1", "u2"],)


def test_prune_auth_uses_the_documented_ages(store):
    calls = []
    store.prune_tokens = lambda cutoff: calls.append(("tokens", cutoff)) or 0
    store.prune_rate_limits = lambda cutoff: calls.append(("rate_limits", cutoff)) or 0
    store.prune_abandoned_accounts = lambda cutoff, now: calls.append(("users", cutoff, now)) or 0
    out = limits.prune_auth(store, NOW)
    assert calls == [("tokens", NOW - timedelta(days=1)), ("rate_limits", NOW - timedelta(days=2)),
                     ("users", NOW - timedelta(days=7), NOW)]
    assert out == {"tokens": 0, "rate_limits": 0, "users": 0}


def test_memory_store_prunes_like_the_sql(store):
    old = NOW - timedelta(days=8)
    keep_verified = store.create(email="a@x.com", name="", password_hash=None, email_verified=True, now=old)
    keep_new = store.create(email="b@x.com", name="", password_hash=None, email_verified=False,
                            now=NOW - timedelta(days=2))
    keep_linked = store.create(email="c@x.com", name="", password_hash=None, email_verified=False,
                               now=old, google_sub="g")
    keep_owner = store.create(email="d@x.com", name="", password_hash=None, email_verified=False, now=old)
    store.owned_rows.add(("watchlist", keep_owner.id))
    keep_live_link = store.create(email="e@x.com", name="", password_hash=None, email_verified=False, now=old)
    store.issue_token(keep_live_link.id, "invite", NOW - timedelta(days=1))
    keep_admin = store.create(email="f@x.com", name="", password_hash=None, email_verified=False,
                              now=old, role="admin")
    gone = store.create(email="g@x.com", name="", password_hash=None, email_verified=False, now=old)
    store.issue_token(gone.id, "verify", old)                 # long expired
    assert store.prune_abandoned_accounts(NOW - timedelta(days=7), NOW) == 1
    assert store.by_id(gone.id) is None
    assert not [r for r in store.tokens.values() if r["user_id"] == gone.id]
    for kept in (keep_verified, keep_new, keep_linked, keep_owner, keep_live_link, keep_admin):
        assert store.by_id(kept.id) is not None


def test_the_nightly_refresh_prunes_best_effort(monkeypatch, capsys):
    from app.routers import pe as pe_router
    monkeypatch.delenv("PE_REFRESH_SECRET", raising=False)
    monkeypatch.setattr(pe_router, "get_db", lambda: FakeDB())
    monkeypatch.setattr(pe_router, "refresh_pe_data", lambda db: {"success": True})
    monkeypatch.setattr(pe_router, "refresh_annual_fundamentals", lambda db: {"success": True})
    monkeypatch.setattr(pe_router, "refresh_macro_series", lambda db: {"success": True})
    seen = []
    monkeypatch.setattr(pe_router, "prune_auth", lambda store, now: seen.append(store) or {"users": 1})
    out = pe_router.trigger_refresh(None)
    assert out["success"] is True and out["auth_prune"] == {"users": 1}
    assert isinstance(seen[0], AccountStore)

    def boom(store, now):
        raise RuntimeError("db down")
    monkeypatch.setattr(pe_router, "prune_auth", boom)
    out = pe_router.trigger_refresh(None)
    assert out["success"] is True and "auth_prune" not in out
    assert "prune failed" in capsys.readouterr().out
