"""
Tests for roles, admin user management, and per-user composite weights.

Run from egx-api-be:  python -m pytest tests/test_users_and_roles.py -v
"""

import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.auth import (
    ROLE_ADMIN,
    ROLE_USER,
    CurrentUser,
    require_admin,
    seed_users_from_env,
)
from account_fakes import MemoryAccountStore, OutboxMailer
import app.routers.users as users_mod


# ---------------------------------------------------------------------------
# A scriptable fake DB — matches on a substring of the SQL, records every
# statement. The real Postgres is never involved, matching test_sell_tracking.
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeDB:
    def __init__(self, responses=None):
        # responses: list of (sql_substring, rows) — first match wins
        self.responses = responses or []
        self.log = []          # (sql, params)
        self.tx_log = []       # statements issued inside transaction()
        self.committed = False

    def _answer(self, sql):
        for needle, rows in self.responses:
            if needle in sql:
                return rows
        return []

    def execute(self, sql, params=()):
        self.log.append((sql, params))
        return _Result(self._answer(sql))

    def commit(self):
        self.committed = True

    @contextmanager
    def transaction(self):
        outer = self

        class _Tx:
            def execute(self, sql, params=()):
                outer.tx_log.append((sql, params))
                outer.log.append((sql, params))
                return _Result(outer._answer(sql))

        yield _Tx()

    # helpers
    def sql_text(self):
        return " ".join(s for s, _ in self.log)


ADMIN = CurrentUser(id="admin-1", email="mark@example.com", role=ROLE_ADMIN)
PLAIN = CurrentUser(id="user-1", email="sara@example.com", role=ROLE_USER)


def _patch_db(monkeypatch, db):
    """Point every get_db() the routers use at the fake."""
    import app.routers.users as users_mod
    import app.core.auth as auth_mod
    import app.core.db as db_mod

    monkeypatch.setattr(users_mod, "get_db", lambda: db)
    monkeypatch.setattr(db_mod, "get_db", lambda: db)
    monkeypatch.setattr(auth_mod, "get_db", lambda: db, raising=False)
    return db


# ---------------------------------------------------------------------------
# Roles and the admin gate
# ---------------------------------------------------------------------------

def test_require_admin_rejects_a_plain_user():
    with pytest.raises(HTTPException) as e:
        require_admin(PLAIN)
    assert e.value.status_code == 403


def test_require_admin_lets_an_admin_through():
    assert require_admin(ADMIN) is ADMIN


def test_is_admin_is_derived_from_the_role_string():
    assert CurrentUser(id="x", email="x@example.com", role=ROLE_ADMIN).is_admin
    assert not CurrentUser(id="x", email="x@example.com").is_admin
    # The default must be the powerless role: a CurrentUser built without an
    # explicit role must never be an admin by accident.
    assert CurrentUser(id="x", email="x@example.com").role == ROLE_USER


def test_every_users_route_requires_admin_not_merely_a_login():
    """
    A route that used Depends(get_current_user) would let ANY signed-in user
    create accounts and reset passwords. Every route in this file must gate on
    require_admin.
    """
    src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "users.py").read_text(encoding="utf-8")
    decorators = re.findall(r"@router\.(get|post|patch|delete)\(", src)
    assert len(decorators) == 5, "expected 5 admin routes"
    assert src.count("Depends(require_admin)") == len(decorators)
    assert "Depends(get_current_user)" not in src


# ---------------------------------------------------------------------------
# A disabled account must lose access IMMEDIATELY, not in 30 days
# ---------------------------------------------------------------------------

def test_disabled_user_is_rejected_even_with_a_valid_token(monkeypatch):
    """
    Role and active-state are read from the row, not the JWT. If they were
    claims, a 30-day token would keep a disabled user working for a month and
    "disable" would be a lie.
    """
    import app.core.auth as auth_mod

    db = FakeDB([("FROM users WHERE id", [("user-1", "sara@example.com", "Sara", ROLE_USER, False, None)])])
    monkeypatch.setattr("app.core.db.get_db", lambda: db)

    assert auth_mod._load_user("user-1") is None


def test_active_user_loads_with_their_current_role(monkeypatch):
    import app.core.auth as auth_mod

    db = FakeDB([("FROM users WHERE id", [("admin-1", "mark@example.com", "Mark", ROLE_ADMIN, True, None)])])
    monkeypatch.setattr("app.core.db.get_db", lambda: db)

    user = auth_mod._load_user("admin-1")
    assert user.is_admin and user.email == "mark@example.com"


def test_deleted_user_holding_a_live_token_is_rejected(monkeypatch):
    db = FakeDB([("FROM users WHERE id", [])])
    monkeypatch.setattr("app.core.db.get_db", lambda: db)

    import app.core.auth as auth_mod
    assert auth_mod._load_user("ghost") is None


def test_get_current_user_does_not_trust_role_from_the_token():
    """The token must not carry role/is_active at all — see create_access_token."""
    src = (Path(__file__).resolve().parents[1] / "app" / "core" / "auth.py").read_text(encoding="utf-8")
    payload = src.split("payload = {", 1)[1].split("}", 1)[0]
    assert '"role"' not in payload
    assert '"is_active"' not in payload
    keys = set(re.findall(r'"(\w+)":', payload))
    assert keys == {"sub", "iat", "exp"}


# ---------------------------------------------------------------------------
# The admin API — route functions called directly over a MemoryAccountStore
# ---------------------------------------------------------------------------

T0 = datetime(2026, 10, 12, tzinfo=timezone.utc)


def _store_with(*roles):
    s = MemoryAccountStore()
    accs = []
    for i, role in enumerate(roles):
        a = s.create(email=f"u{i}@x.com", name="", password_hash=None, email_verified=True, now=T0)
        if role == "admin":
            s.set_role(a.id, "admin")
        accs.append(s.users[a.id])
    return s, accs


def _admin(acc):
    return CurrentUser(id=acc.id, email=acc.email, role=ROLE_ADMIN)


def test_cannot_disable_or_delete_yourself():
    s, (me, _) = _store_with("admin", "admin")
    for call in (lambda: users_mod.set_user_active(me.id, users_mod.UserPatch(is_active=False), _admin(me), s),
                 lambda: users_mod.delete_user(me.id, _admin(me), s)):
        with pytest.raises(HTTPException) as e:
            call()
        assert e.value.status_code == 400
        assert "your own account" in e.value.detail
    assert s.by_id(me.id) is not None and s.by_id(me.id).is_active


def test_cannot_disable_the_last_active_admin():
    s, (boss, other_admin) = _store_with("admin", "admin")
    s.update(other_admin.id, is_active=False)
    actor = CurrentUser(id="someone-else", email="x@x.com", role=ROLE_ADMIN)
    with pytest.raises(HTTPException) as e:
        users_mod.set_user_active(boss.id, users_mod.UserPatch(is_active=False), actor, s)
    assert e.value.status_code == 400 and "last active admin" in e.value.detail
    assert s.by_id(boss.id).is_active


def test_cannot_delete_the_last_active_admin_and_nothing_is_deleted():
    s, (boss, other_admin) = _store_with("admin", "admin")
    s.update(other_admin.id, is_active=False)
    actor = CurrentUser(id="someone-else", email="x@x.com", role=ROLE_ADMIN)
    with pytest.raises(HTTPException) as e:
        users_mod.delete_user(boss.id, actor, s)
    assert e.value.status_code == 400
    assert s.deleted_from == [] and s.by_id(boss.id) is not None


def test_a_second_active_admin_makes_deletion_allowed():
    s, (boss, other_admin) = _store_with("admin", "admin")
    out = users_mod.delete_user(other_admin.id, _admin(boss), s)
    assert out["deleted"] == other_admin.id and s.by_id(other_admin.id) is None


def test_re_enabling_a_user_is_never_blocked():
    """The guards exist to preserve access, so they must not block restoring it."""
    s, (boss, dormant) = _store_with("admin", "user")
    s.update(dormant.id, is_active=False)
    out = users_mod.set_user_active(dormant.id, users_mod.UserPatch(is_active=True), _admin(boss), s)
    assert out["is_active"] is True


def test_unknown_user_is_404():
    s, (boss,) = _store_with("admin")
    with pytest.raises(HTTPException) as e:
        users_mod.set_user_active("nope", users_mod.UserPatch(is_active=False), _admin(boss), s)
    assert e.value.status_code == 404


def test_delete_user_removes_every_owned_table():
    s, (boss, victim) = _store_with("admin", "user")
    users_mod.delete_user(victim.id, _admin(boss), s)
    from app.core.accounts import USER_OWNED_TABLES
    assert {t for t, uid in s.deleted_from if uid == victim.id} == set(USER_OWNED_TABLES)


def test_listing_shows_methods_and_never_a_password_hash():
    s, (boss,) = _store_with("admin")
    s.update(boss.id, password_hash="secret-hash", google_sub="g")
    out = users_mod.list_users(_admin(boss), s)["users"][0]
    assert out["methods"] == ["password", "google"]
    assert "secret-hash" not in str(out) and "password_hash" not in out
    assert set(out) == {"id", "email", "name", "role", "is_active", "email_verified",
                        "methods", "created_at", "holdings_count"}


def test_invite_route_uses_admin_name_as_inviter(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.test")
    s, (boss,) = _store_with("admin")
    mailer = OutboxMailer()
    actor = CurrentUser(id=boss.id, email=boss.email, name="Mark", role=ROLE_ADMIN)
    out = users_mod.invite_user(users_mod.InviteRequest(email="NEW@x.com", name="N"), actor, s, mailer)
    assert out["user"]["email"] == "new@x.com" and "Mark invited you" in mailer.sent[-1]["subject"]
    assert out["user"]["role"] == ROLE_USER


def test_send_reset_route_emails_a_link_and_writes_no_password(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.test")
    s, (boss, target) = _store_with("admin", "user")
    s.update(target.id, password_hash="old-hash")
    mailer = OutboxMailer()
    out = users_mod.send_reset(target.id, _admin(boss), s, mailer)
    assert out == {"id": target.id, "sent": "reset"}
    assert mailer.last_link() and s.by_id(target.id).password_hash == "old-hash"


def test_the_admin_api_cannot_grant_admin():
    """
    Admin status comes only from AUTH_ADMINS at boot. No file under
    app/routers may write a role, so privilege escalation through the API
    is structurally impossible.
    """
    routers = Path(__file__).resolve().parents[1] / "app" / "routers"
    files = sorted(routers.glob("*.py"))
    assert len(files) > 10, "router scan found too few files"
    for f in files:
        assert "SET role" not in f.read_text(encoding="utf-8"), f"{f.name} may not write a role"


def test_the_admin_api_writes_no_passwords():
    """set_password (auth_flows) is the only password writer, and it revokes sessions."""
    src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "users.py").read_text(encoding="utf-8")
    assert "password_hash" not in src
    assert "hash_password" not in src


def test_generate_password_is_gone():
    import app.core.auth as auth_mod
    assert not hasattr(auth_mod, "generate_password")


# ---------------------------------------------------------------------------
# Bootstrap — AUTH_ADMINS (emails)
# ---------------------------------------------------------------------------

def _seed_db(legacy=False):
    return FakeDB([("information_schema.columns", [(1,)] if legacy else [])])


def test_auth_admins_creates_missing_admins_verified_without_password(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", " Mark3essam@Gmail.com ")
    db = _seed_db()
    seed_users_from_env(db)
    insert = next(p for s, p in db.log if s.startswith("INSERT INTO users"))
    assert insert[1] == "mark3essam@gmail.com"
    assert insert[2] == ROLE_ADMIN
    sql = next(s for s, _ in db.log if s.startswith("INSERT INTO users"))
    assert "NULL, TRUE" in sql and "ON CONFLICT (email) DO NOTHING" in sql


def test_auth_admins_promotes_the_listed_emails(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", "mark@x.com,sara@x.com")
    db = _seed_db()
    seed_users_from_env(db)
    promote = next(p for s, p in db.log if "SET role" in s and "NOT (" not in s)
    assert promote[0] == ROLE_ADMIN and promote[1] == ["mark@x.com", "sara@x.com"]


def test_auth_admins_demotes_anyone_no_longer_listed(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", "mark@x.com")
    db = _seed_db()
    seed_users_from_env(db)
    demote = next(p for s, p in db.log if "NOT (email = ANY" in s)
    assert demote[0] == ROLE_USER and demote[1] == ["mark@x.com"]


def test_empty_auth_admins_touches_nothing(monkeypatch):
    monkeypatch.delenv("AUTH_ADMINS", raising=False)
    db = _seed_db()
    seed_users_from_env(db)
    assert db.log == []


def test_a_legacy_username_table_skips_seeding(monkeypatch):
    monkeypatch.setenv("AUTH_ADMINS", "mark@x.com")
    db = _seed_db(legacy=True)
    seed_users_from_env(db)
    assert not any(s.startswith(("INSERT", "UPDATE")) for s, _ in db.log)


# ---------------------------------------------------------------------------
# Per-user weights
# ---------------------------------------------------------------------------

from app.core.composite import DEFAULT_WEIGHTS, get_weights_from_db, weights_hash
from app.routers.analysis import composite_cache_key


def test_weights_fall_back_to_the_global_row_when_the_user_has_no_override():
    """
    Existing installs keep their saved weights as everyone's starting point,
    so nobody's scores jump on deploy.
    """
    db = FakeDB([
        ("FROM settings WHERE key LIKE", [("weight_trend", "40")]),
        ("FROM user_settings", []),
    ])
    assert get_weights_from_db(db, "user-1")["trend"] == 40


def test_a_user_override_wins_over_the_global_row():
    db = FakeDB([
        ("FROM settings WHERE key LIKE", [("weight_trend", "40")]),
        ("FROM user_settings", [("weight_trend", "5")]),
    ])
    assert get_weights_from_db(db, "user-1")["trend"] == 5


def test_override_is_per_key_not_all_or_nothing():
    """A user who moved one slider must keep the global value for the rest."""
    db = FakeDB([
        ("FROM settings WHERE key LIKE", [("weight_trend", "40"), ("weight_momentum", "30")]),
        ("FROM user_settings", [("weight_trend", "5")]),
    ])
    w = get_weights_from_db(db, "user-1")
    assert w["trend"] == 5 and w["momentum"] == 30


def test_anonymous_context_never_reads_user_settings():
    db = FakeDB([("FROM settings WHERE key LIKE", [("weight_trend", "40")])])
    get_weights_from_db(db, None)
    assert "user_settings" not in db.sql_text()


def test_no_sql_has_an_unescaped_percent():
    """
    _DB.execute always passes a params tuple, so psycopg parses every query for
    placeholders and a lone `%` raises ProgrammingError.

    This is not theoretical: `LIKE 'weight_%'` in get_weights_from_db raised on
    every call, and the `except Exception: return DEFAULT_WEIGHTS` around it
    swallowed the error. Saved composite weights were never read back — every
    score in the app was computed at Beginner Safe defaults regardless of what
    the user had set. A silent fallback makes this class of bug invisible, so
    it gets pinned rather than trusted.
    """
    import ast

    offenders = []
    for path in sorted((Path(__file__).resolve().parents[1] / "app").rglob("*.py")):
        if "vendor" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "execute"
                    and node.args):
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                sql = arg.value
            elif isinstance(arg, ast.JoinedStr):
                sql = "".join(
                    v.value if isinstance(v, ast.Constant) else "{}" for v in arg.values
                )
            else:
                continue
            stripped = sql.replace("%s", "").replace("%b", "").replace("%t", "").replace("%%", "")
            if "%" in stripped:
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        "unescaped % in SQL — psycopg will raise ProgrammingError: " + ", ".join(offenders)
    )


def test_missing_keys_still_fall_back_to_defaults():
    db = FakeDB([("FROM settings WHERE key LIKE", []), ("FROM user_settings", [])])
    assert get_weights_from_db(db, "user-1") == DEFAULT_WEIGHTS


def test_two_users_with_different_weights_get_different_cache_keys():
    """
    Otherwise one user's sliders would serve the other user's score. The key
    already folds in weights_hash, so this needs no user_id component.
    """
    tags_a = (weights_hash({**DEFAULT_WEIGHTS, "trend": 40}), "bullish", "rfr25")
    tags_b = (weights_hash({**DEFAULT_WEIGHTS, "trend": 5}), "bullish", "rfr25")
    assert composite_cache_key("COMI", "Daily", tags_a) != composite_cache_key("COMI", "Daily", tags_b)


def test_two_users_with_identical_weights_share_one_cache_entry():
    """Fragmenting the cache per user would multiply the scoring cost for nothing."""
    tags = (weights_hash(DEFAULT_WEIGHTS), "bullish", "rfr25")
    assert composite_cache_key("COMI", "Daily", tags) == composite_cache_key("COMI", "Daily", tags)
    assert "user" not in composite_cache_key("COMI", "Daily", tags)


def test_regime_reader_is_pinned_to_the_anonymous_context():
    """
    The regime bands were calibrated at DEFAULT weights. If the reader resolved
    one user's custom sliders it would miss every entry the public dashboard
    warmed and report "not enough data" forever.
    """
    src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "analysis.py").read_text(encoding="utf-8")
    body = src.split("def read_cached_scores", 1)[1].split("\ndef ", 1)[0]
    assert "scoring_cache_context(None)" in body


def test_risk_free_rate_is_never_user_scoped():
    """
    It is the Sharpe hurdle, the CBE rate on the macro card, AND the bar
    realized trades are graded against — a market fact, not a preference.
    """
    settings_src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "settings.py").read_text(encoding="utf-8")
    user_writes = [
        line for line in settings_src.splitlines()
        if "user_settings" in line and "INSERT" in line
    ]
    assert user_writes, "weights must be written per user"
    assert "risk_free_rate" not in " ".join(user_writes)


def test_settings_endpoint_requires_a_token():
    """It was reachable without one, and served every user the same weights."""
    src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "settings.py").read_text(encoding="utf-8")
    assert src.count("Depends(get_current_user)") >= 2


# ---------------------------------------------------------------------------
# Cross-user isolation of the sell ledger — pins today's correct behaviour so
# the admin work cannot regress it
# ---------------------------------------------------------------------------

_ROUTERS = Path(__file__).resolve().parents[1] / "app" / "routers"


def _sql_literals(path):
    """
    Every SQL string passed to .execute() in a module.

    Parsed with ast rather than regex because these queries are built from
    adjacent string literals and f-strings — `"SELECT ... " "WHERE user_id..."`
    is one statement, and a regex that stops at the first quote would read the
    scoping clause as missing.
    """
    import ast

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(
                p.value if isinstance(p, ast.Constant) else "{}" for p in node.values
            )
        return None

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
                and node.args):
            sql = literal(node.args[0])
            if sql:
                out.append(" ".join(sql.split()))
    return out


def test_every_portfolio_sales_statement_is_scoped_to_a_user():
    """
    One user must never read, delete or restore another's trades. The undo path
    is the sharp edge: it both deletes a sale AND adds shares back to a
    holding, so an unscoped statement there would let user B hand shares to
    user A's position.
    """
    checked = 0
    for sql in _sql_literals(_ROUTERS / "sales.py"):
        if "portfolio_sales" not in sql and "FROM portfolio" not in sql:
            continue
        if sql.startswith("INSERT"):
            assert "user_id" in sql, "the sale row must record its owner"
            checked += 1
            continue
        assert "user_id = %s" in sql, f"unscoped statement in sales.py: {sql!r}"
        checked += 1

    assert checked >= 4, "expected the select, insert, decrement and undo statements"


def test_every_holdings_read_is_scoped_to_a_user():
    """The same rule for the portfolio table itself, across both routers."""
    for name in ("portfolio.py", "sales.py"):
        for sql in _sql_literals(_ROUTERS / name):
            if "FROM portfolio" not in sql and "UPDATE portfolio " not in sql:
                continue
            assert "user_id = %s" in sql, f"unscoped statement in {name}: {sql!r}"


def test_sales_routes_all_demand_a_logged_in_user():
    src = (_ROUTERS / "sales.py").read_text(encoding="utf-8")
    routes = len(re.findall(r"@router\.(get|post|delete)\(", src))
    assert routes == 3
    assert src.count("Depends(get_current_user)") == routes
