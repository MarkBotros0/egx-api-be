"""
Market condition reading — the one forecast-shaped thing the backtest supported.

WHAT THIS IS
------------
The average composite score across the EGX30 + EGX70 constituents. Not a
per-stock signal: a reading of how broadly healthy the market looks right now.

WHY IT EXISTS WHEN THE PER-STOCK SCORE DOES NOT PREDICT
-------------------------------------------------------
The walk-forward backtest (scripts/backtest.py) found the composite cannot rank
one stock above another — cross-sectional IC ~0. But the same panel showed the
market-wide AVERAGE of those scores does carry information about the market
itself. Cross-sectional ranking and time-series level are different questions,
and the score answers the second one better than the first.

Measured on 221 monthly readings, 2007-2026, against EGX30 forward returns:

    horizon   sampling            rho     t     verdict
    21 days   every date        +0.084  +1.24  not significant
    63 days   every 3rd date    +0.318  +2.84  significant
    126 days  every 6th date    +0.115  +0.69  not significant

The 63-day row uses NON-OVERLAPPING windows. That matters: a time-series
correlation on overlapping windows shares data between consecutive
observations and inflates significance. De-overlapping here made the
correlation stronger (+0.170 -> +0.318), so it is not an overlap artifact —
but it also cut the sample to 74 independent observations, which is why this
is presented as an association and not a prediction.

WHAT THE BANDS ACTUALLY SAY
---------------------------
Terciles of the reading against the next three months of EGX30:

    band     reading      mean    median   3m positive
    weak     < 45.1      +0.6%    -0.1%       49%
    mixed    45.1-51.5   +6.3%    +5.5%       68%
    broad    >= 51.5     +5.4%    +6.7%       70%

Read it as a CAUTION signal rather than a go signal. The useful distinction is
weak-versus-not: when breadth has been in the bottom third the next quarter was
a coin flip averaging nothing, and otherwise it delivered the market's usual
drift about seven times in ten. The top two bands are not meaningfully
different from each other.

Note the base rate: the EGX rose substantially over this window in EGP terms,
helped by repeated devaluations, so "weak" means flat rather than negative.
"""

from __future__ import annotations

from typing import Optional

# Tercile cutoffs on the mean composite score, from 221 readings 2007-2026.
REGIME_WEAK_MAX = 45.1
REGIME_MIXED_MAX = 51.5

# Minimum stocks in the average before the reading means anything. The batch
# scorer returns partial results when it hits its deadline, so coverage varies.
MIN_SYMBOLS_FOR_REGIME = 15

# Horizon the association was measured at. Do not relabel this: 21 and 126 days
# were both tested and neither was significant.
REGIME_HORIZON_DAYS = 63

_BANDS = {
    "weak": {
        "label": "Broadly weak",
        "summary": (
            "Most of the market is in poor technical condition. Historically "
            "the next three months were a coin flip averaging about nothing."
        ),
        "hist_median_3m_pct": -0.1,
        "hist_positive_rate": 0.49,
        "observations": 74,
    },
    "mixed": {
        "label": "Mixed",
        "summary": (
            "The market is in middling condition. Historically the next three "
            "months were positive about two times in three."
        ),
        "hist_median_3m_pct": 5.5,
        "hist_positive_rate": 0.68,
        "observations": 73,
    },
    "broad": {
        "label": "Broadly healthy",
        "summary": (
            "Most of the market is in good technical condition. Historically "
            "the next three months were positive about seven times in ten — "
            "though not meaningfully better than the Mixed band."
        ),
        "hist_median_3m_pct": 6.7,
        "hist_positive_rate": 0.70,
        "observations": 74,
    },
}


def classify_regime(mean_score: Optional[float], n_symbols: int = 0) -> dict:
    """
    Turn the market-wide average composite into a condition reading.

    Returns a dict carrying the band AND the historical record behind it, so
    the UI never has to hardcode a claim — every number it shows comes from
    here, where the measurement is documented.

    `band` is None when coverage is too thin to average meaningfully; callers
    should render "not enough data" rather than a misleading middle reading.
    """
    if mean_score is None or n_symbols < MIN_SYMBOLS_FOR_REGIME:
        return {
            "mean_score": round(mean_score, 1) if mean_score is not None else None,
            "n_symbols": n_symbols,
            "band": None,
            "label": "Not enough data",
            "summary": (
                f"Needs at least {MIN_SYMBOLS_FOR_REGIME} scored stocks to average; "
                f"got {n_symbols}."
            ),
            "horizon_days": REGIME_HORIZON_DAYS,
        }

    if mean_score < REGIME_WEAK_MAX:
        key = "weak"
    elif mean_score < REGIME_MIXED_MAX:
        key = "mixed"
    else:
        key = "broad"

    band = _BANDS[key]
    return {
        "mean_score": round(mean_score, 1),
        "n_symbols": n_symbols,
        "band": key,
        "label": band["label"],
        "summary": band["summary"],
        "hist_median_3m_pct": band["hist_median_3m_pct"],
        "hist_positive_rate": band["hist_positive_rate"],
        "observations": band["observations"],
        "horizon_days": REGIME_HORIZON_DAYS,
        # Surfaced so the UI can state the strength of the association rather
        # than implying certainty.
        "association_rho": 0.318,
        "association_n": 74,
    }
