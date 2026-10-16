"""
US score alerts: detection and delivery per market.

Quiet hours and session expiry are in the market's own timezone and data
(America/New_York for US), baselines are keyed by market, a user's preferences
apply to the markets they hold, watch or have selected, and the copy names no
EGX/Egypt fact and gives no instruction. No network, no Postgres.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from alert_fakes import FakeDB
from app.core import alert_store
from app.core.expo_push import deliver_pending
from app.core.markets import EG, US
from app.core.score_alerts import BANNED_WORDS, Signal, alert_copy, in_quiet_hours

# 20:30 UTC on 2026-10-16: 23:30 in Cairo (quiet), 16:30 in New York (not).
CAIRO_NIGHT_NY_AFTERNOON = datetime(2026, 10, 16, 20, 30, tzinfo=timezone.utc)
# 03:00 UTC: 06:00 Cairo (quiet), 23:00 New York (quiet).
BOTH_NIGHT = datetime(2026, 10, 16, 3, 0, tzinfo=timezone.utc)
# 13:30 UTC: 16:30 Cairo, 09:30 New York — neither quiet.
BOTH_DAY = datetime(2026, 10, 16, 13, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Quiet hours
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("when, eg, us", [
    (CAIRO_NIGHT_NY_AFTERNOON, True, False),
    (BOTH_NIGHT, True, True),
    (BOTH_DAY, False, False),
    (datetime(2026, 10, 16, 11, 59, tzinfo=timezone.utc), False, True),  # 07:59 NY
    (datetime(2026, 10, 16, 1, 59, tzinfo=timezone.utc), True, False),  # 21:59 NY
])
def test_quiet_hours_are_each_markets_own(when, eg, us):
    assert in_quiet_hours(when, EG) is eg
    assert in_quiet_hours(when, US) is us
    assert in_quiet_hours(when) is eg, "the default stays Cairo"


def test_new_york_quiet_hours_follow_dst():
    # 22:00 New York is 02:00 UTC in summer (EDT) and 03:00 UTC in winter (EST).
    assert in_quiet_hours(datetime(2026, 7, 1, 2, 0, tzinfo=timezone.utc), US) is True
    assert in_quiet_hours(datetime(2026, 12, 1, 2, 0, tzinfo=timezone.utc), US) is False


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def _db(rows, tokens):
    return FakeDB([
        ("MAX(last_bar_date)", [("2026-10-16",)]),
        ("SET status = 'expired'", []),
        ("SET status = 'sending'", rows),
        ("FROM push_devices", tokens),
    ])


def test_us_delivery_runs_while_cairo_sleeps_and_is_scoped_to_us():
    sent = []
    db = _db([("a", "u1", "AAPL", "2026-10-16", "up", 50.0, 62.0, 0)],
             [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, CAIRO_NIGHT_NY_AFTERNOON, market=US,
                          sender=lambda m: sent.extend(m) or [{"status": "ok"}])
    assert out["held"] is False and out["sent"] == 1
    [msg] = sent
    assert msg["title"] == "AAPL score up 12 points"
    assert msg["data"] == {"type": "score_alert", "symbol": "AAPL",
                           "session": "2026-10-16", "market": "US"}
    # Every read and write of the queue names the market.
    [(sql, params)] = db.statements("MAX(last_bar_date)")
    assert "WHERE market = %s" in sql and params == ("US",)
    for needle in ("SET status = 'expired'", "SET status = 'sending'", "status = 'sending' AND claimed_at"):
        for sql, params in db.statements(needle):
            assert "market = %s" in sql and "US" in params, needle


def test_eg_delivery_is_held_at_the_same_moment():
    db = FakeDB()
    out = deliver_pending(db, CAIRO_NIGHT_NY_AFTERNOON, sender=lambda m: [])
    assert out["held"] is True and db.log == []


def test_us_delivery_is_held_during_new_york_night():
    db = FakeDB()
    out = deliver_pending(db, BOTH_NIGHT, market=US, sender=lambda m: [])
    assert out["held"] is True and db.log == []


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction, after", [("up", 85.0), ("down", 15.0), ("summary", 0.0)])
def test_us_copy_has_no_egyptian_fact_and_no_instruction(direction, after):
    title, body = alert_copy(Signal("u", "AAPL", "d", direction, 50.0, after), US)
    text = f"{title} {body}"
    for banned in ("EGX", "Egypt", "EGP"):
        assert banned not in text
    for word in BANNED_WORDS:
        assert not re.search(rf"\b{word}\b", text, re.IGNORECASE), word


def test_us_summary_names_the_market_and_eg_summary_is_unchanged():
    sig = Signal("u", "*", "d", "summary", 0.0, 0.0)
    assert alert_copy(sig) == ("More stocks moved today", "Open Borsa Compass to see your scores.")
    assert alert_copy(sig, US) == ("More US stocks moved today",
                                   "Open Borsa Compass to see your US scores.")


# ---------------------------------------------------------------------------
# Who hears about a US move
# ---------------------------------------------------------------------------

def _subs_db(prefs, in_market, watch=()):
    return FakeDB([
        ("s.key = 'market'", [(u,) for u in in_market]),
        ("FROM alert_preferences", list(prefs)),
        ("FROM watchlist", list(watch)),
        ("FROM settings", []),
        ("FROM user_settings", []),
    ])


def test_scope_all_applies_only_to_users_who_have_the_market():
    db = _subs_db(prefs=[("us-user", "all", 10, 10), ("eg-only", "all", 10, 10)],
                  in_market=["us-user"])
    subs = alert_store.load_subscribers(db, ["AAPL", "KO"], US.code)
    assert {s.user_id for s in subs["AAPL"]} == {"us-user"}
    [(sql, params)] = db.statements("s.key = 'market'")
    # holds or watches in US, or selected US in the app (absent = EG)
    assert "FROM watchlist WHERE market = %s" in sql
    assert "FROM portfolio WHERE market = %s" in sql and "quantity > 0" in sql
    assert params[0] == "US" and params[2] == "US"
    assert params[-2:] == ("EG", "US")


def test_watchlist_scope_matches_us_rows_only():
    db = _subs_db(prefs=[("w", "watchlist", 5, None)], in_market=["w"],
                  watch=[("w", "KO")])
    subs = alert_store.load_subscribers(db, ["AAPL", "KO"], US.code)
    assert list(subs) == ["KO"]
    [(sql, params)] = [x for x in db.statements("FROM watchlist")
                       if "symbol = ANY" in x[0]]
    assert params[0] == "US" and params[3] == "US"


def test_nobody_with_the_market_means_no_subscribers():
    db = _subs_db(prefs=[("eg-only", "all", 10, 10)], in_market=[])
    assert alert_store.load_subscribers(db, ["AAPL"], US.code) == {}


def test_the_daily_cap_counts_one_markets_session():
    """An EG day and a US day share a calendar date but are different sessions."""
    class _DB(FakeDB):
        def execute(self, sql, params=()):
            self.log.append((sql, params))
            from alert_fakes import _Result
            if "s.key = 'market'" in sql:
                return _Result([("u1",)])
            if "FROM alert_preferences" in sql:
                return _Result([("u1", "all", 10, 10)])
            if "FROM score_baseline" in sql:
                return _Result([])
            return _Result([])

    db = _DB()
    alert_store.run_detection(db, {"AAPL": {"session_date": "2026-10-16",
                                            "categories": {}}}, None, "now",
                              market_code=US.code)
    # No baseline -> no signal, but the baseline read is US-keyed.
    [(sql, params)] = db.statements("FROM score_baseline")
    assert "market = %s" in sql and params[0] == "US"


def test_run_detection_cap_query_is_market_scoped():
    import inspect
    src = inspect.getsource(alert_store.run_detection)
    assert "WHERE session_date = %s AND market = %s" in src


# ---------------------------------------------------------------------------
# The cron hook: no macro for US, the market threaded through
# ---------------------------------------------------------------------------

def test_us_run_alerts_blends_without_macro_and_delivers_us(monkeypatch):
    from app.routers import cron
    seen = {}
    monkeypatch.setattr(cron, "read_cached_macro",
                        lambda db: pytest.fail("US scores take no macro modulation"))
    monkeypatch.setattr(cron, "run_detection",
                        lambda db, scored, macro, now, market_code: seen.update(
                            macro=macro, code=market_code) or 1)
    monkeypatch.setattr(cron, "prune_baselines", lambda *a, **k: None)

    def deliver(db, now, deadline=None, market=None):
        seen["deliver"] = market.code
        return {"sent": 1, "held": False}

    out = cron.run_alerts(object(), {"AAPL": {}}, "2026-10-16T14:00:00+00:00", 5.0,
                          deliver=deliver, market=US)
    assert out == {"queued": 1, "sent": 1, "skipped": None, "error": None}
    assert seen == {"macro": None, "code": "US", "deliver": "US"}


def test_eg_run_alerts_still_reads_the_cached_regime(monkeypatch):
    from app.routers import cron
    seen = {}
    monkeypatch.setattr(cron, "read_cached_macro", lambda db: {"egx30": {"trend": "bearish"}})
    monkeypatch.setattr(cron, "run_detection",
                        lambda db, scored, macro, now, market_code: seen.update(
                            macro=macro, code=market_code) or 0)
    monkeypatch.setattr(cron, "prune_baselines", lambda *a, **k: None)
    cron.run_alerts(object(), {"COMI": {}}, "2026-10-16T09:00:00+00:00", 5.0,
                    deliver=lambda db, now, deadline=None, market=None: {"sent": 0})
    assert seen == {"macro": {"egx30": {"trend": "bearish"}}, "code": "EG"}
