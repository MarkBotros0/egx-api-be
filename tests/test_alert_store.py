"""Score alerts — storage glue. No Postgres (see alert_fakes.FakeDB)."""

from __future__ import annotations

from alert_fakes import FakeDB
from app.core.card_snapshot import CATEGORY_COLUMNS
from app.core.alert_store import (
    load_subscribers,
    prune_baselines,
    run_detection,
)
from app.core.card_snapshot import upsert_snapshot
from app.core.composite import CATEGORY_ORDER, PRESETS


def test_init_db_creates_the_four_alert_tables():
    from app.core.db import init_db

    db = FakeDB()
    init_db(db)
    sql = " ".join(s for s, _ in db.log)
    for table in ("push_devices", "alert_preferences", "score_baseline", "alert_log"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, table
    assert "UNIQUE (user_id, symbol, session_date, direction)" in sql
    assert "ALTER TABLE alert_log ADD COLUMN IF NOT EXISTS claimed_at TEXT" in sql
    assert "PRIMARY KEY (symbol, session_date)" in sql
    for column in CATEGORY_COLUMNS:
        assert f"{column} DOUBLE PRECISION" in sql


def _cats(v):
    return {name: v for name in CATEGORY_ORDER}


def _card(bar_date, v=60.0):
    return {"categories": _cats(v), "last_bar_date": bar_date}


def _full_card(bar_date):
    return {"categories": _cats(60.0), "last_bar_date": bar_date,
            "prev_close": 1.0, "sparkline_json": "[]"}


def _upsert(db, card):
    upsert_snapshot(db, "COMI", "now", None, False, True, card)


def test_upsert_with_a_card_captures_the_baseline_before_writing_the_row():
    """Both writers (cron and the dashboard's live upgrade) go through here."""
    db = FakeDB()
    _upsert(db, _full_card("2026-10-11"))
    sqls = [s for s, _ in db.log]
    assert len(sqls) == 2
    assert sqls[0].startswith("INSERT INTO score_baseline")
    assert sqls[1].startswith("INSERT INTO risk_snapshot")
    sql, params = db.log[0]
    cols = ", ".join(CATEGORY_COLUMNS)
    assert f"(market, symbol, session_date, {cols}, source_bar_date, captured_at)" in sql
    assert f"SELECT market, symbol, %s, {cols}, last_bar_date, %s FROM risk_snapshot" in sql
    assert "WHERE market = %s AND symbol = %s AND last_bar_date IS DISTINCT FROM %s" in sql
    # never guess a baseline from an unscored row
    assert " OR ".join(f"{c} IS NOT NULL" for c in CATEGORY_COLUMNS) in sql
    # first capture wins
    assert sql.rstrip().endswith("ON CONFLICT (market, symbol, session_date) DO NOTHING")
    assert params == ("2026-10-11", "now", "EG", "COMI", "2026-10-11")


def test_upsert_without_a_card_or_a_bar_date_captures_nothing():
    db = FakeDB()
    _upsert(db, None)
    _upsert(db, _full_card(None))
    _upsert(db, _full_card(""))
    assert db.statements("score_baseline") == []
    assert len(db.statements("INSERT INTO risk_snapshot")) == 3


def test_a_failing_baseline_statement_never_stops_the_snapshot_write():
    class _Flaky(FakeDB):
        def execute(self, sql, params=()):
            if "score_baseline" in sql:
                raise RuntimeError("baseline broke")
            return super().execute(sql, params)

    db = _Flaky()
    _upsert(db, _full_card("2026-10-11"))
    assert len(db.statements("INSERT INTO risk_snapshot")) == 1


def test_prune_deletes_baselines_older_than_retention():
    db = FakeDB()
    prune_baselines(db, "2026-10-11")
    [(sql, params)] = db.log
    assert sql.startswith("DELETE FROM score_baseline WHERE session_date <")
    assert params == ("2026-10-01",)


BAL = PRESETS["balanced"]


def _subs_db(prefs, watch=(), extra=()):
    """prefs rows: (user_id, scope, rise, fall)."""
    return FakeDB([
        ("FROM alert_preferences", list(prefs)),
        ("FROM watchlist", list(watch)),
        ("FROM settings", []),
        ("FROM user_settings", []),
        *extra,
    ])


def test_scope_all_subscribes_to_every_symbol_and_watchlist_only_to_followed():
    db = _subs_db(
        prefs=[("all-user", "all", 10, 10), ("watch-user", "watchlist", 5, None)],
        watch=[("watch-user", "COMI")],
    )
    out = load_subscribers(db, ["COMI", "FWRY"])
    assert sorted(s.user_id for s in out["COMI"]) == ["all-user", "watch-user"]
    assert [s.user_id for s in out["FWRY"]] == ["all-user"]
    watch = next(s for s in out["COMI"] if s.user_id == "watch-user")
    assert (watch.rise, watch.fall) == (5, None)
    pref_sql = db.statements("FROM alert_preferences")[0][0]
    assert "enabled" in pref_sql and "is_active" in pref_sql and "push_devices" in pref_sql
    follow_sql = db.statements("FROM watchlist")[0][0]
    assert "UNION" in follow_sql and "FROM portfolio" in follow_sql


def test_no_enabled_users_means_no_follow_query():
    db = _subs_db(prefs=[])
    assert load_subscribers(db, ["COMI"]) == {}
    assert db.statements("FROM watchlist") == []


def _baseline_row(symbol, session, v):
    return (symbol, session) + tuple([v] * len(CATEGORY_ORDER))


def test_run_detection_queues_a_pending_alert_for_a_qualifying_move():
    db = _subs_db(
        prefs=[("u1", "all", 10, 10)],
        extra=[
            ("FROM score_baseline", [_baseline_row("COMI", "2026-10-11", 50.0)]),
            ("INSERT INTO alert_log", [("id1",)]),
            ("FROM alert_log", []),
        ],
    )
    n = run_detection(db, {"COMI": {"session_date": "2026-10-11", "categories": _cats(70.0)}},
                      None, "now", new_id=lambda: "id1")
    assert n == 1
    [(sql, params)] = db.statements("INSERT INTO alert_log")
    assert "ON CONFLICT (user_id, market, symbol, session_date, direction) DO NOTHING" in sql
    assert params[:6] == ("id1", "u1", "EG", "COMI", "2026-10-11", "up")
    assert params[8] == "pending"


def test_run_detection_without_a_baseline_queues_nothing():
    db = _subs_db(prefs=[("u1", "all", 10, 10)], extra=[("FROM score_baseline", [])])
    assert run_detection(db, {"COMI": {"session_date": "2026-10-11", "categories": _cats(90.0)}},
                         None, "now") == 0
    assert db.statements("INSERT INTO alert_log") == []


def test_run_detection_skips_a_direction_already_logged_this_session():
    db = _subs_db(
        prefs=[("u1", "all", 10, 10)],
        extra=[
            ("FROM score_baseline", [_baseline_row("COMI", "2026-10-11", 50.0)]),
            ("SELECT user_id, symbol, direction FROM alert_log", [("u1", "COMI", "up")]),
            ("FROM alert_log", []),
        ],
    )
    assert run_detection(db, {"COMI": {"session_date": "2026-10-11", "categories": _cats(70.0)}},
                         None, "now") == 0


def test_run_detection_with_nothing_scored_touches_nothing():
    db = FakeDB()
    assert run_detection(db, {}, None, "now") == 0
    assert db.log == []


class _SessionDB(FakeDB):
    """FakeDB whose alert_log SELECTs answer per session (params[0])."""

    def __init__(self, responses, logged=None, counts=None):
        super().__init__(responses)
        self.logged, self.counts = logged or {}, counts or {}

    def execute(self, sql, params=()):
        if "FROM alert_log" in sql and sql.startswith("SELECT"):
            self.log.append((sql, params))
            src = self.counts if "COUNT(*)" in sql else self.logged
            from alert_fakes import _Result
            return _Result(src.get(params[0], []))
        return super().execute(sql, params)


def test_run_detection_scopes_dedup_to_each_symbols_own_session():
    db = _SessionDB(
        [
            ("FROM alert_preferences", [("u1", "all", 10, 10)]),
            ("FROM settings", []),
            ("FROM user_settings", []),
            ("FROM score_baseline", [_baseline_row("COMI", "2026-10-11", 50.0),
                                     _baseline_row("FWRY", "2026-10-07", 50.0)]),
            ("INSERT INTO alert_log", [("x",)]),
        ],
        logged={"2026-10-07": [("u1", "COMI", "up")]},
    )
    scored = {
        "COMI": {"session_date": "2026-10-11", "categories": _cats(70.0)},
        "FWRY": {"session_date": "2026-10-07", "categories": _cats(30.0)},
    }
    assert run_detection(db, scored, None, "now", new_id=lambda: "i") == 2
    inserts = {(p[3], p[4], p[5]) for _, p in db.statements("INSERT INTO alert_log")}
    assert inserts == {("COMI", "2026-10-11", "up"), ("FWRY", "2026-10-07", "down")}


def test_run_detection_applies_the_daily_cap_with_a_summary():
    db = _SessionDB(
        [
            ("FROM alert_preferences", [("u1", "all", 10, 10)]),
            ("FROM settings", []),
            ("FROM user_settings", []),
            ("FROM score_baseline", [_baseline_row("COMI", "2026-10-11", 50.0)]),
            ("INSERT INTO alert_log", [("x",)]),
        ],
        counts={"2026-10-11": [("u1", 10)]},
    )
    n = run_detection(db, {"COMI": {"session_date": "2026-10-11", "categories": _cats(70.0)}},
                      None, "now", new_id=lambda: "i")
    assert n == 2
    rows = [p for _, p in db.statements("INSERT INTO alert_log")]
    assert any(p[3] == "*" and p[5] == "summary" and p[8] == "pending" for p in rows)
    assert any(p[3] == "COMI" and p[5] == "up" and p[8] == "suppressed" for p in rows)
