"""
Dividend history + cadence, from Yahoo's chart API.

The parse and cadence helpers are pure, so the whole shape is tested without a
network call or Postgres — the seam is `parse_dividends(payload)`, the same
posture as `news_fetch` and `pe_fetch(rows=...)`.

WHY YAHOO: the EGX publishes no machine-readable dividend history, the
TradingView scanner carries only the ONE most-recent coupon per stock, and its
`fundamentals_annual.dps` has amounts but no ex-dates. Yahoo's
`chart/<SYM>.CA?events=div` is the one keyless source of DATED multi-year
history — verified live: COMI 15 dividends back to 2010, amounts matching EGX
filings (6.00 on ~7 Apr 2026).
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.dividend_history import (
    parse_dividends,
    summarize_cadence,
    yahoo_symbol,
)


def _ts(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())


_PAYLOAD = {
    "chart": {
        "result": [
            {
                "events": {
                    "dividends": {
                        str(_ts(2026, 4, 7)): {"amount": 6.0, "date": _ts(2026, 4, 7)},
                        str(_ts(2025, 4, 24)): {"amount": 0.5, "date": _ts(2025, 4, 24)},
                        str(_ts(2024, 4, 4)): {"amount": 0.488, "date": _ts(2024, 4, 4)},
                    }
                }
            }
        ]
    }
}


def test_yahoo_symbol_appends_the_ca_suffix():
    assert yahoo_symbol("COMI") == "COMI.CA"
    assert yahoo_symbol("comi") == "COMI.CA"
    assert yahoo_symbol(" SWDY ") == "SWDY.CA"


def test_parse_returns_dated_amounts_newest_first():
    out = parse_dividends(_PAYLOAD)
    assert [d["ex_date"] for d in out] == ["2026-04-07", "2025-04-24", "2024-04-04"]
    assert out[0]["amount"] == 6.0
    assert out[0]["year"] == 2026


def test_parse_is_empty_not_an_error_when_there_are_no_dividends():
    """
    A stock that has never paid is a fact, not a failure. Yahoo returns a result
    with no `events`, or `events` with no `dividends`. Either yields [].
    """
    assert parse_dividends({"chart": {"result": [{}]}}) == []
    assert parse_dividends({"chart": {"result": [{"events": {}}]}}) == []
    assert parse_dividends({"chart": {"result": []}}) == []
    assert parse_dividends({}) == []
    assert parse_dividends({"chart": {"error": "Not Found"}}) == []


def test_parse_drops_rows_with_no_usable_amount_or_timestamp():
    ts = _ts(2026, 4, 7)
    payload = {
        "chart": {"result": [{"events": {"dividends": {
            str(ts): {"amount": None, "date": ts},          # no amount
            "bad": {"amount": 1.0, "date": None},           # no ts
            str(_ts(2025, 4, 1)): {"amount": 0.3, "date": _ts(2025, 4, 1)},
        }}}]}
    }
    out = parse_dividends(payload)
    assert [d["ex_date"] for d in out] == ["2025-04-01"]


def test_cadence_reports_typical_month_and_last_date():
    """
    The estimate behind 'typically pays ~April'. Not a promise — the mode of
    past ex-date months, and the real last date.
    """
    c = summarize_cadence(parse_dividends(_PAYLOAD))
    assert c["last_ex_date"] == "2026-04-07"
    assert c["typical_month"] == 4
    assert c["typical_month_name"] == "Apr"
    assert c["count"] == 3
    assert c["payments_per_year"] == 1


def test_cadence_detects_semiannual_payers():
    """Two coupons in a single year -> payments_per_year 2 (the modal count)."""
    divs = [
        {"ex_date": "2026-05-20", "amount": 0.22, "year": 2026},
        {"ex_date": "2026-12-30", "amount": 0.22, "year": 2026},
        {"ex_date": "2025-05-20", "amount": 0.20, "year": 2025},
        {"ex_date": "2025-12-30", "amount": 0.20, "year": 2025},
    ]
    c = summarize_cadence(divs)
    assert c["payments_per_year"] == 2


def test_cadence_of_empty_history_is_null_shaped_not_a_crash():
    c = summarize_cadence([])
    assert c["last_ex_date"] is None
    assert c["typical_month"] is None
    assert c["typical_month_name"] is None
    assert c["payments_per_year"] is None
    assert c["count"] == 0


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
    def fetchall(self):
        return self._rows


class _FakeDB:
    """Minimal db stub — the read helpers are just row -> dict transforms."""
    def __init__(self, rows):
        self._rows = rows
    def execute(self, sql, params=None):
        return _FakeCursor(self._rows)


def test_read_dividends_maps_rows_to_dated_amounts():
    from app.core.dividend_history import read_dividends
    db = _FakeDB([("2026-04-07", 6.0), ("2025-04-24", 0.5)])
    out = read_dividends(db, "comi")
    assert out == [
        {"ex_date": "2026-04-07", "amount": 6.0, "year": 2026},
        {"ex_date": "2025-04-24", "amount": 0.5, "year": 2025},
    ]


def test_read_calendar_maps_and_sorts_newest_first():
    from app.core.dividend_history import read_calendar
    db = _FakeDB([("ABUK", "2026-04-20", 2.3), ("COMI", "2026-04-07", 6.0)])
    out = read_calendar(db)
    assert [d["symbol"] for d in out] == ["ABUK", "COMI"]  # 04-20 newer than 04-07
    assert out[0] == {"symbol": "ABUK", "ex_date": "2026-04-20", "amount": 2.3}


def test_both_dividend_routes_are_registered_and_behind_the_gate():
    """
    The app is closed, default-deny. Neither dividend route is public — they
    read the caller's context (history is user-scoped by auth; the calendar is
    still a signed-in-only surface). tests/test_auth_gate.py enforces this over
    the whole route table; this is the local canary.
    """
    from app.core.auth import PUBLIC_ENDPOINTS
    from app.main import app

    paths = {r.path for r in app.routes}
    assert "/api/dividend_history" in paths
    assert "/api/dividend_calendar" in paths
    for _, p in PUBLIC_ENDPOINTS:
        assert p not in ("/api/dividend_history", "/api/dividend_calendar")
