"""
Tests for roles, admin user management, and per-user composite weights.

Run from egx-api-be:  python -m pytest tests/test_users_and_roles.py -v
"""

import re
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.auth import (
    ROLE_ADMIN,
    ROLE_USER,
    CurrentUser,
    generate_password,
    hash_password,
    require_admin,
    seed_users_from_env,
    verify_password,
)


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


ADMIN = CurrentUser(id="admin-1", username="mark", role=ROLE_ADMIN)
PLAIN = CurrentUser(id="user-1", username="sara", role=ROLE_USER)


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
    assert CurrentUser(id="x", username="x", role=ROLE_ADMIN).is_admin
    assert not CurrentUser(id="x", username="x").is_admin
    # The default must be the powerless role: a CurrentUser built without an
    # explicit role must never be an admin by accident.
    assert CurrentUser(id="x", username="x").role == ROLE_USER


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

    db = FakeDB([("FROM users WHERE id", [("user-1", "sara", ROLE_USER, False)])])
    monkeypatch.setattr("app.core.db.get_db", lambda: db)

    assert auth_mod._load_user("user-1") is None


def test_active_user_loads_with_their_current_role(monkeypatch):
    import app.core.auth as auth_mod

    db = FakeDB([("FROM users WHERE id", [("admin-1", "mark", ROLE_ADMIN, True)])])
    monkeypatch.setattr("app.core.db.get_db", lambda: db)

    user = auth_mod._load_user("admin-1")
    assert user.is_admin and user.username == "mark"


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


# ---------------------------------------------------------------------------
# Guards — you cannot lock everyone out of the app
# ---------------------------------------------------------------------------

def _users_mod():
    import app.routers.users as m
    return m


def test_cannot_disable_the_last_active_admin(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("admin-2", "other", ROLE_ADMIN, True)]),
        ("COUNT(*)", [(1,)]),
    ]))

    with pytest.raises(HTTPException) as e:
        m.set_user_active("admin-2", m.UserPatch(is_active=False), admin=ADMIN)
    assert e.value.status_code == 400
    assert "last active admin" in e.value.detail


def test_cannot_delete_the_last_active_admin(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("admin-2", "other", ROLE_ADMIN, True)]),
        ("COUNT(*)", [(1,)]),
    ]))

    with pytest.raises(HTTPException) as e:
        m.delete_user("admin-2", admin=ADMIN)
    assert e.value.status_code == 400
    assert db.tx_log == [], "nothing may be deleted when the guard trips"


def test_a_second_active_admin_makes_deletion_allowed(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("admin-2", "other", ROLE_ADMIN, True)]),
        ("COUNT(*)", [(2,)]),
    ]))

    out = m.delete_user("admin-2", admin=ADMIN)
    assert out["deleted"] == "admin-2"


def test_cannot_disable_yourself(monkeypatch):
    m = _users_mod()
    _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [(ADMIN.id, ADMIN.username, ROLE_ADMIN, True)]),
        ("COUNT(*)", [(5,)]),
    ]))

    with pytest.raises(HTTPException) as e:
        m.set_user_active(ADMIN.id, m.UserPatch(is_active=False), admin=ADMIN)
    assert "your own account" in e.value.detail


def test_cannot_delete_yourself(monkeypatch):
    m = _users_mod()
    _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [(ADMIN.id, ADMIN.username, ROLE_ADMIN, True)]),
        ("COUNT(*)", [(5,)]),
    ]))

    with pytest.raises(HTTPException) as e:
        m.delete_user(ADMIN.id, admin=ADMIN)
    assert "your own account" in e.value.detail


def test_re_enabling_a_user_is_never_blocked(monkeypatch):
    """The guards exist to preserve access, so they must not block restoring it."""
    m = _users_mod()
    _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("u9", "dormant", ROLE_USER, False)]),
    ]))
    out = m.set_user_active("u9", m.UserPatch(is_active=True), admin=ADMIN)
    assert out["is_active"] is True


# ---------------------------------------------------------------------------
# Deleting a user must not leave orphans — no table has an FK to `users`
# ---------------------------------------------------------------------------

def test_deleting_a_user_removes_everything_they_own(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("u2", "sara", ROLE_USER, True)]),
        ("COUNT(*)", [(2,)]),
    ]))

    m.delete_user("u2", admin=ADMIN)

    deleted_from = [sql for sql, _ in db.tx_log]
    for table in ("portfolio_sales", "portfolio", "watchlist", "user_settings", "users"):
        assert any(f"DELETE FROM {table} " in s for s in deleted_from), (
            f"deleting a user left {table} rows behind — nothing cascades, "
            "there is no FK to users"
        )
    assert all(p[-1] == "u2" for _, p in db.tx_log)


def test_sales_are_deleted_before_holdings(monkeypatch):
    """Matches the direction the sale-undo path depends on."""
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("u2", "sara", ROLE_USER, True)]),
        ("COUNT(*)", [(2,)]),
    ]))
    m.delete_user("u2", admin=ADMIN)

    order = [s for s, _ in db.tx_log]
    sales_at = next(i for i, s in enumerate(order) if "portfolio_sales" in s)
    holdings_at = next(i for i, s in enumerate(order) if "DELETE FROM portfolio " in s)
    users_at = next(i for i, s in enumerate(order) if "DELETE FROM users" in s)
    assert sales_at < holdings_at < users_at


def test_user_deletion_is_one_transaction(monkeypatch):
    """A half-deleted user is worse than an undeleted one."""
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("u2", "sara", ROLE_USER, True)]),
        ("COUNT(*)", [(2,)]),
    ]))
    m.delete_user("u2", admin=ADMIN)
    assert len(db.tx_log) == 6, "every delete must run inside transaction()"


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def test_generated_password_is_random_and_unambiguous():
    a, b = generate_password(), generate_password()
    assert a != b
    assert len(a) == 16
    # 0/O and 1/l/I get misread off a screen; they cost support time for no
    # entropy worth keeping.
    assert not set(a) & set("0O1lI")


def test_created_user_gets_a_generated_password_returned_once(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([("FROM users WHERE username", [])]))

    out = m.create_user(m.UserCreate(username="newbie"), admin=ADMIN)
    generated = out["generated_password"]
    assert generated and len(generated) == 16

    # Only the hash is stored, and it verifies against the returned password.
    insert = next(p for s, p in db.log if "INSERT INTO users" in s)
    stored_hash = insert[2]
    assert stored_hash != generated
    assert verify_password(generated, stored_hash)


def test_an_explicit_password_is_not_echoed_back(monkeypatch):
    m = _users_mod()
    _patch_db(monkeypatch, FakeDB([("FROM users WHERE username", [])]))

    out = m.create_user(
        m.UserCreate(username="newbie", password="hunter2-hunter2"), admin=ADMIN
    )
    assert out["generated_password"] is None


def test_short_explicit_password_is_rejected(monkeypatch):
    m = _users_mod()
    _patch_db(monkeypatch, FakeDB([("FROM users WHERE username", [])]))
    with pytest.raises(HTTPException) as e:
        m.create_user(m.UserCreate(username="newbie", password="short"), admin=ADMIN)
    assert e.value.status_code == 400


def test_reset_password_stores_a_new_hash(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users WHERE id", [("u2", "sara", ROLE_USER, True)]),
    ]))

    out = m.reset_password("u2", m.PasswordReset(), admin=ADMIN)
    update = next(p for s, p in db.log if "UPDATE users SET password_hash" in s)
    assert verify_password(out["generated_password"], update[0])


def test_listing_users_never_exposes_a_password(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([
        ("FROM users u", [("u1", "mark", ROLE_ADMIN, True, "2026-01-01", 3)]),
    ]))

    out = m.list_users(admin=ADMIN)
    assert out["users"][0]["username"] == "mark"
    assert "password" not in str(out).lower()
    assert "password_hash" not in db.sql_text()


def test_duplicate_username_is_rejected(monkeypatch):
    m = _users_mod()
    _patch_db(monkeypatch, FakeDB([("FROM users WHERE username", [("existing",)])]))
    with pytest.raises(HTTPException) as e:
        m.create_user(m.UserCreate(username="taken"), admin=ADMIN)
    assert e.value.status_code == 409


@pytest.mark.parametrize("bad", ["ab", "has space", "UPPER!", "x" * 33, ""])
def test_invalid_usernames_are_rejected(bad, monkeypatch):
    m = _users_mod()
    _patch_db(monkeypatch, FakeDB([("FROM users WHERE username", [])]))
    with pytest.raises(HTTPException) as e:
        m.create_user(m.UserCreate(username=bad), admin=ADMIN)
    assert e.value.status_code == 400


def test_username_is_normalised_to_lowercase(monkeypatch):
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([("FROM users WHERE username", [])]))
    out = m.create_user(m.UserCreate(username="  Sara.B  "), admin=ADMIN)
    assert out["user"]["username"] == "sara.b"


def test_the_admin_api_cannot_grant_admin(monkeypatch):
    """
    Admin status comes only from AUTH_ADMINS at boot. If create_user could set
    a role, privilege escalation through this API would be one request away.
    """
    m = _users_mod()
    db = _patch_db(monkeypatch, FakeDB([("FROM users WHERE username", [])]))
    out = m.create_user(m.UserCreate(username="newbie"), admin=ADMIN)
    assert out["user"]["role"] == ROLE_USER

    src = (Path(__file__).resolve().parents[1] / "app" / "routers" / "users.py").read_text(encoding="utf-8")
    assert "SET role" not in src, "no route may write a role"


# ---------------------------------------------------------------------------
# Bootstrap — AUTH_USERS / AUTH_ADMINS
# ---------------------------------------------------------------------------

def test_seeding_does_not_overwrite_an_existing_password(monkeypatch):
    """
    This used to rewrite the hash on every boot. Once an admin can reset
    passwords, that silently reverts their work on the next cold start.
    """
    monkeypatch.setenv("AUTH_USERS", "mark:from-the-env")
    monkeypatch.setenv("AUTH_ADMINS", "")
    db = FakeDB([("FROM users WHERE username", [("existing-id",)])])

    seed_users_from_env(db)

    assert not any("INSERT INTO users" in s for s, _ in db.log)
    assert not any("password_hash" in s and "UPDATE" in s for s, _ in db.log)


def test_seeding_creates_a_user_that_does_not_exist_yet(monkeypatch):
    monkeypatch.setenv("AUTH_USERS", "mark:secret-secret")
    monkeypatch.setenv("AUTH_ADMINS", "")
    db = FakeDB([("FROM users WHERE username", [])])

    seed_users_from_env(db)

    insert = next(p for s, p in db.log if "INSERT INTO users" in s)
    assert insert[1] == "mark"
    assert verify_password("secret-secret", insert[2])


def test_auth_admins_promotes_the_listed_usernames(monkeypatch):
    monkeypatch.setenv("AUTH_USERS", "")
    monkeypatch.setenv("AUTH_ADMINS", "mark, Sara")
    db = FakeDB()

    seed_users_from_env(db)

    promote = next(p for s, p in db.log if "SET role" in s and "NOT (" not in s)
    assert promote[0] == ROLE_ADMIN
    assert sorted(promote[1]) == ["mark", "sara"]


def test_a_new_user_named_in_auth_admins_is_created_as_admin(monkeypatch):
    monkeypatch.setenv("AUTH_USERS", "mark:secret-secret")
    monkeypatch.setenv("AUTH_ADMINS", "mark")
    db = FakeDB([("FROM users WHERE username", [])])

    seed_users_from_env(db)

    insert = next(p for s, p in db.log if "INSERT INTO users" in s)
    assert insert[4] == ROLE_ADMIN


def test_empty_auth_admins_demotes_nobody(monkeypatch):
    """
    A blank or unset var must never strip every admin — that leaves the app
    permanently unmanageable with no recovery path through the UI.
    """
    monkeypatch.setenv("AUTH_USERS", "")
    monkeypatch.setenv("AUTH_ADMINS", "")
    db = FakeDB()

    seed_users_from_env(db)

    assert not any("SET role" in s for s, _ in db.log)


def test_auth_admins_demotes_anyone_no_longer_listed(monkeypatch):
    monkeypatch.setenv("AUTH_USERS", "")
    monkeypatch.setenv("AUTH_ADMINS", "mark")
    db = FakeDB()

    seed_users_from_env(db)

    demote = next(p for s, p in db.log if "SET role" in s and "NOT (" in s)
    assert demote[0] == ROLE_USER
    assert demote[1] == ["mark"]


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
