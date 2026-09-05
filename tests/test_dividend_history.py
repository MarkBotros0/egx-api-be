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
