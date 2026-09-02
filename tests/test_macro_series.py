"""
Dated macro series, and the release lag that makes them usable historically.

The lag is the whole point. A macro bar's timestamp is its REFERENCE period, so
August's inflation is stamped August and was not knowable until September. Every
historical read must filter on release, not on reference; without that, any
macro-conditioned backtest result can be manufactured.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.core.macro_series import (
    FX_SERIES,
    RISK_FREE_SERIES,
    SERIES,
    get_fx_at,
    get_macro_at,
    get_risk_free_at,
    released_at,
)


class _DB:
    """Stub standing in for Postgres — tests/ has no DB fixture by design."""

    def __init__(self, rows):
        self.rows = rows            # (code, period, value, released_at)
        self._result = None

    def execute(self, sql, params):
        assert "released_at <= %s" in sql, (
            "the release filter was removed — historical reads would see "
            "figures before they were published"
        )
        code, as_of = params
        hits = [r for r in self.rows if r[0] == code and r[3] <= as_of]
        hits.sort(key=lambda r: r[1], reverse=True)
        self._result = (hits[0][2],) if hits else None
        return self

    def fetchone(self):
        return self._result


def test_a_release_lag_pushes_a_figure_past_its_reference_period():
    """August's inflation is not knowable in August."""
    assert released_at("2026-08-01", 10) == "2026-08-11"


def test_a_price_series_has_no_lag():
    """An FX rate is knowable the day it prints; only releases are delayed."""
    assert SERIES[FX_SERIES][2] == 0
    assert released_at("2026-08-01", 0) == "2026-08-01"


def test_every_release_lag_is_non_negative():
    """A negative lag would publish a figure before its period ended."""
    for code, (_, _, lag, _) in SERIES.items():
        assert lag >= 0, f"{code} has a negative release lag"


def test_a_figure_is_invisible_before_it_was_published():
    """
    THE guard. The July reading is stamped 2026-07-01 but only becomes readable
    on 2026-07-11; a backtest standing on 2026-07-05 must not see it.
    """
    rows = [
        ("EGIRYY", "2026-06-01", 12.0, "2026-06-11"),
        ("EGIRYY", "2026-07-01", 14.9, "2026-07-11"),
    ]
    assert get_macro_at(_DB(rows), "EGIRYY", "2026-07-05") == 12.0
    assert get_macro_at(_DB(rows), "EGIRYY", "2026-07-15") == 14.9


def test_the_risk_free_rate_falls_back_rather_than_returning_none():
    """
    A missing history must not silently zero the Sharpe hurdle. The caller's
    default stands in, which is the behaviour every consumer already expects.
    """
    assert get_risk_free_at(_DB([]), "2020-01-01", default=19.0) == 19.0


def test_the_risk_free_rate_reads_the_policy_series():
    rows = [(RISK_FREE_SERIES, "2026-02-01", 19.0, "2026-02-01"),
            (RISK_FREE_SERIES, "2024-01-01", 27.0, "2024-01-01")]
    assert get_risk_free_at(_DB(rows), "2024-06-01", default=0.0) == 27.0
    assert get_risk_free_at(_DB(rows), "2026-06-01", default=0.0) == 19.0


def test_fx_reads_the_same_symbol_the_macro_card_uses():
    """
    FX_IDC:USDEGP, so the historical series and the live macro card cannot
    disagree about what the rate was.
    """
    assert FX_SERIES == "USDEGP"
    assert SERIES[FX_SERIES][1] == "FX_IDC"
    rows = [(FX_SERIES, "2026-09-02", 51.0, "2026-09-02")]
    assert get_fx_at(_DB(rows), "2026-09-02") == 51.0


@pytest.mark.parametrize("code", list(SERIES))
def test_every_series_declares_a_label_exchange_lag_and_unit(code):
    label, exchange, lag, unit = SERIES[code]
    assert label and exchange and unit
    assert isinstance(lag, int)
