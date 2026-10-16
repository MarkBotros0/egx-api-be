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

The fitted values now live in app/core/calibration/<market>.py (RegimeCal).
"""

from __future__ import annotations

from typing import Optional

from app.core.calibration import RegimeCal

# Minimum stocks in the average before the reading means anything. The batch
# scorer returns partial results when it hits its deadline, so coverage varies.
MIN_SYMBOLS_FOR_REGIME = 15

# Horizon the association was measured at. Do not relabel this: 21 and 126 days
# were both tested on EGX and neither was significant.
REGIME_HORIZON_DAYS = 63


def classify_regime(mean_score: Optional[float], n_symbols: int,
                    regime: RegimeCal) -> dict:
    """
    Turn the market-wide average composite into a condition reading, using
    the cutoffs and historical record of `regime` (the market's bundle).

    `band` is None when coverage is too thin to average meaningfully; callers
    render "not enough data" rather than a misleading middle reading.
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
            "universe_label": regime.universe_label,
        }

    if mean_score < regime.weak_max:
        key = "weak"
    elif mean_score < regime.mixed_max:
        key = "mixed"
    else:
        key = "broad"

    band = regime.bands[key]
    return {
        "mean_score": round(mean_score, 1),
        "n_symbols": n_symbols,
        "band": key,
        "label": band.label,
        "summary": band.summary,
        "hist_median_3m_pct": band.median_3m_pct,
        "hist_positive_rate": band.positive_rate,
        "observations": band.observations,
        "horizon_days": REGIME_HORIZON_DAYS,
        "universe_label": regime.universe_label,
        # The strength of the association, stated so the UI never implies
        # certainty. `association_significant` False must be honoured.
        "association_rho": regime.rho,
        "association_t": regime.t,
        "association_n": regime.n,
        "association_significant": regime.significant,
        "association_phase_rhos": list(regime.phase_rhos),
    }
