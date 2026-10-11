"""Score alerts — storage glue. No Postgres (see alert_fakes.FakeDB)."""

from __future__ import annotations

from alert_fakes import FakeDB
from app.core.card_snapshot import CATEGORY_COLUMNS
from app.core.alert_store import (
    capture_baseline,
    load_subscribers,
    prune_baselines,
    read_previous_rows,
    run_detection,
)
from app.core.composite import CATEGORY_ORDER, PRESETS


def test_init_db_creates_the_four_alert_tables():
    from app.core.db import init_db

    db = FakeDB()
    init_db(db)
    sql = " ".join(s for s, _ in db.log)
    for table in ("push_devices", "alert_preferences", "score_baseline", "alert_log"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, table
    assert "UNIQUE (user_id, symbol, session_date, direction)" in sql
    assert "PRIMARY KEY (symbol, session_date)" in sql
    for column in CATEGORY_COLUMNS:
        assert f"{column} DOUBLE PRECISION" in sql


def _cats(v):
    return {name: v for name in CATEGORY_ORDER}


def _card(bar_date, v=60.0):
    return {"categories": _cats(v), "last_bar_date": bar_date}


def test_read_previous_rows_shapes_categories_by_name():
    row = ("COMI", "2026-10-07") + tuple(float(i) for i in range(len(CATEGORY_ORDER)))
    db = FakeDB([("FROM risk_snapshot", [row])])
    out = read_previous_rows(db, ["COMI"])
    assert out["COMI"]["last_bar_date"] == "2026-10-07"
    assert out["COMI"]["categories"][CATEGORY_ORDER[1]] == 1.0
    sql, params = db.log[0]
    assert "symbol = ANY(%s)" in sql and params == (["COMI"],)


def test_read_previous_rows_with_no_symbols_skips_the_query():
    db = FakeDB()
    assert read_previous_rows(db, []) == {}
    assert db.log == []


def test_a_new_session_captures_the_previous_close_as_baseline():
    db = FakeDB()
    prev = {"last_bar_date": "2026-10-07", "categories": _cats(50.0)}
    assert capture_baseline(db, "COMI", prev, _card("2026-10-11"), "now") is True
    [(sql, params)] = db.statements("INSERT INTO score_baseline")
    assert "ON CONFLICT (symbol, session_date) DO NOTHING" in sql
    assert params[:2] == ("COMI", "2026-10-11")
    assert params[2:2 + len(CATEGORY_ORDER)] == tuple([50.0] * len(CATEGORY_ORDER))
    assert params[-2:] == ("2026-10-07", "now")


def test_same_session_rescore_writes_no_baseline():
    db = FakeDB()
    prev = {"last_bar_date": "2026-10-11", "categories": _cats(50.0)}
    assert capture_baseline(db, "COMI", prev, _card("2026-10-11"), "now") is False
    assert db.log == []


def test_no_baseline_from_an_unscored_previous_row_or_without_a_card():
    db = FakeDB()
    empty = {"last_bar_date": "2026-10-07", "categories": _cats(None)}
    assert capture_baseline(db, "COMI", empty, _card("2026-10-11"), "now") is False
    assert capture_baseline(db, "COMI", None, _card("2026-10-11"), "now") is False
    full = {"last_bar_date": "2026-10-07", "categories": _cats(50.0)}
    assert capture_baseline(db, "COMI", full, None, "now") is False
    assert capture_baseline(db, "COMI", full, {"categories": _cats(1.0), "last_bar_date": None}, "now") is False
    assert db.log == []


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
            ("FROM alert_log", []),
        ],
    )
    n = run_detection(db, {"COMI": {"session_date": "2026-10-11", "categories": _cats(70.0)}},
                      None, "now", new_id=lambda: "id1")
    assert n == 1
    [(sql, params)] = db.statements("INSERT INTO alert_log")
    assert "ON CONFLICT (user_id, symbol, session_date, direction) DO NOTHING" in sql
    assert params[:5] == ("id1", "u1", "COMI", "2026-10-11", "up")
    assert params[7] == "pending"


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
