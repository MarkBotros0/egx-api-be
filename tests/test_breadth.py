"""
Market breadth: an operational fix first, a weak signal second.

The regime card averages composite scores the DASHBOARD happens to have cached,
and since the app went closed that cache is mostly cold. Breadth aggregates
flags the nightly risk snapshot already stores, so it is always as fresh as last
night. What it must never become is a second forecast: its strongest leg reaches
t=-2.44, below this project's |t| > 3.0 bar.
"""

from __future__ import annotations

from app.core.breadth import (
    MIN_SYMBOLS_FOR_BREADTH,
    RSI_OVERSOLD,
    compute_breadth,
    describe,
)


def _rows(n, above=True, rsi=50.0, tradeable=True):
    return [{"tradeable": tradeable, "above_sma200": above, "rsi_14": rsi}
            for _ in range(n)]


def test_untradeable_names_do_not_count_toward_breadth():
    """
    A stock nobody can buy is not participating in anything, and this market has
    a large dead tail — including it would drag every reading toward whatever
    that tail happens to be doing.
    """
    rows = _rows(30, above=True) + _rows(60, above=False, tradeable=False)
    result = compute_breadth(rows)
    assert result["n_symbols"] == 30
    assert result["pct_above_sma200"] == 100.0


def test_thin_coverage_refuses_to_report_a_percentage():
    """
    A percentage over nine stocks is a handful of names wearing a percent sign.
    The regime card refuses below 15 for the same reason.
    """
    result = compute_breadth(_rows(MIN_SYMBOLS_FOR_BREADTH - 1))
    assert result["enough_data"] is False
    assert result["pct_above_sma200"] is None
    assert describe(result) is None


def test_oversold_counts_use_the_standard_threshold():
    rows = _rows(20, rsi=RSI_OVERSOLD - 5) + _rows(20, rsi=RSI_OVERSOLD + 5)
    result = compute_breadth(rows)
    assert result["pct_oversold"] == 50.0


def test_breadth_never_claims_significance():
    """
    The whole card is context. If someone flips this flag, the UI is entitled to
    present it as a forecast — which the measurement does not support.
    """
    result = compute_breadth(_rows(40))
    assert result["evidence"]["significant_at_project_bar"] is False
    assert abs(result["evidence"]["strongest_t"]) < 3.0


def test_the_description_is_about_today_not_tomorrow():
    result = compute_breadth(_rows(40, above=True))
    text = describe(result)
    assert "not what happens next" in text
    for banned in ("will", "expect", "forecast", "predict"):
        assert banned not in text.lower()


def test_missing_flags_do_not_crash_the_aggregate():
    """Symbols measured before the breadth columns existed carry None."""
    rows = _rows(20) + [{"tradeable": True, "above_sma200": None, "rsi_14": None}
                        for _ in range(10)]
    result = compute_breadth(rows)
    assert result["enough_data"] is True
    assert result["pct_above_sma200"] == 100.0
