"""
Market condition reading — the one forecast-shaped thing the backtest supported.

WHAT THIS IS
------------
The average composite score across the EGX30 + EGX70 constituents. Not a
per-stock signal: a reading of how broadly healthy the market looks right now.

WHY IT EXISTS WHEN THE PER-STOCK SCORE DOES NOT PREDICT
-------------------------------------------------------
The walk-forward backtest (scripts/backtest.py) found the composite cannot rank
one stock above another — cross-sectional IC ~0, and slightly negative. But the
same panel showed the market-wide AVERAGE of those scores carries SOME
information about the market itself. Cross-sectional ranking and time-series
level are different questions, and the score answers the second one less badly.

    Measured on 221 monthly readings, 2007-2026, EGX30+EGX70 universe,
    against EGX30's next 63 trading days:

        overlapping rho          +0.162
        naive t                   +2.55   <- WRONG, see below
        Newey-West t              +1.74   <- the honest one; NOT significant

CORRECTION, 2026-09-02 — READ THIS BEFORE QUOTING A NUMBER FROM HERE
--------------------------------------------------------------------
This docstring used to claim "+0.318, t=2.84 across 74 NON-OVERLAPPING
periods", and defended it with: "de-overlapping made the correlation stronger
(+0.162 -> +0.309), so it is not an overlap artifact."

That reasoning was wrong. At a 63-day horizon on a 21-day rebalance grid there
are exactly THREE valid de-overlapped samplings, and they do not agree:

        offset 0:  rho=+0.309   t=+2.76   n=74     <- the number that shipped
        offset 1:  rho=+0.167   t=+1.44   n=74
        offset 2:  rho=+0.004   t=+0.03   n=73

De-overlapping did not VALIDATE the result, it RESAMPLED it. The best of three
draws was then read as a robustness check. Starting the same grid one month
later would have produced a card claiming nothing at all.

The correct treatment keeps all 221 observations and corrects the standard
error for the autocorrelation the overlap induces — Newey-West with lag equal
to the overlap depth. That gives **t = 1.74, which does not clear 1.96**, and
that is before any correction for having tested three horizons (21/63/126) and
reported the best one.

So: the association is real in sign and consistent with several independent
market-breadth measures that land in the same +0.15 to +0.29 range, but it is
WEAK and NOT statistically significant on its own. This card is CONTEXT, not a
forecast. Do not restore a single-phase statistic, and do not describe this as
predictive.

`scripts/calibrate.py` regenerates every number in this file and prints all
three phases side by side, so the cherry-pick cannot silently return.

WHAT THE BANDS ACTUALLY SAY
---------------------------
Terciles of the reading against the next three months of EGX30:

    band     reading      mean    median   3m positive
    weak     < 45.4      +1.0%    -0.0%       50%
    mixed    45.4-51.9   +6.1%    +5.4%       69%
    broad    >= 51.9     +5.2%    +6.7%       69%

Read it as a CAUTION signal rather than a go signal. The useful distinction is
weak-versus-not: when breadth has been in the bottom third the next quarter was
a literal coin flip (50.0% positive) averaging nothing, and otherwise it
delivered the market's usual drift about seven times in ten. The top two bands
are not meaningfully different from each other — 68.5% against 68.9%.

Note the base rate: the EGX rose substantially over this window in EGP terms,
helped by repeated devaluations, so "weak" means flat rather than negative.
"""

from __future__ import annotations

from typing import Optional

# Tercile cutoffs on the mean composite score, from 221 readings 2007-2026.
REGIME_WEAK_MAX = 45.4
REGIME_MIXED_MAX = 51.9

# Minimum stocks in the average before the reading means anything. The batch
# scorer returns partial results when it hits its deadline, so coverage varies.
MIN_SYMBOLS_FOR_REGIME = 15

# Horizon the association was measured at. Do not relabel this: 21 and 126 days
# were both tested and neither was significant — and on the honest statistic
# neither is 63, so this is the least-weak horizon rather than a good one.
REGIME_HORIZON_DAYS = 63

# The association behind the bands, stated the way it survives scrutiny.
# Regenerate with `python -m scripts.calibrate`; it prints all three
# de-overlapped phases so a single lucky one cannot be quoted again.
REGIME_ASSOCIATION_RHO = 0.162         # overlapping, all 221 readings
REGIME_ASSOCIATION_T = 1.74            # Newey-West, lag 3 (the overlap depth)
REGIME_ASSOCIATION_N = 221
REGIME_ASSOCIATION_SIGNIFICANT = False  # 1.74 does not clear 1.96
# What the three de-overlapped samplings actually give. Kept in the payload so
# the UI can never present one of them as "the" number.
REGIME_PHASE_RHOS = (0.309, 0.167, -0.006)

_BANDS = {
    "weak": {
        "label": "Broadly weak",
        "summary": (
            "Most of the market is in poor technical condition. Historically "
            "the next three months were a coin flip averaging about nothing."
        ),
        "hist_median_3m_pct": -0.03,
        "hist_positive_rate": 0.50,
        "observations": 74,
    },
    "mixed": {
        "label": "Mixed",
        "summary": (
            "The market is in middling condition. Historically the next three "
            "months were positive about two times in three."
        ),
        "hist_median_3m_pct": 5.42,
        "hist_positive_rate": 0.685,
        "observations": 73,
    },
    "broad": {
        "label": "Broadly healthy",
        "summary": (
            "Most of the market is in good technical condition. Historically "
            "the next three months were positive about seven times in ten — "
            "though not meaningfully better than the Mixed band."
        ),
        "hist_median_3m_pct": 6.72,
        "hist_positive_rate": 0.689,
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
        # Surfaced so the UI states the strength of the association rather than
        # implying certainty. `association_significant` is False and the UI must
        # honour it — this card is context, not a forecast. See the correction
        # note in this module's docstring for why the old 0.318 was wrong.
        "association_rho": REGIME_ASSOCIATION_RHO,
        "association_t": REGIME_ASSOCIATION_T,
        "association_n": REGIME_ASSOCIATION_N,
        "association_significant": REGIME_ASSOCIATION_SIGNIFICANT,
        "association_phase_rhos": list(REGIME_PHASE_RHOS),
    }
