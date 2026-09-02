"""
Per-stock outcome ranges, calibrated against EGX's own return distribution.

WHAT CHANGED, AND WHY (2026-09-02)
----------------------------------
This module used to advertise coverage it did not deliver. Measured over 34,721
point-in-time observations from the cached panel (see scripts/calibrate.py):

    surface                       advertised     actually delivered
    +/-1 sigma daily band            68%              79.0%
    +/-1 sigma weekly band           68%              76.3%
    +/-1 sigma monthly band          68%              73.0%
    Monte Carlo p5-p95 cone          90%              85.8%

Note the two failures run in OPPOSITE directions, which is the part that is easy
to get wrong:

  * At DAILY scale EGX's distribution is THIN in the body, not fat. Price limits
    and flat, illiquid sessions pile mass near zero, so a +/-1-sigma band
    swallows 79% of next-day moves instead of 68%. That band is too WIDE for
    what it claims.
  * At 60-day scale, compounding plus volatility clustering fattens the
    aggregate, so a Gaussian 90% cone catches only 85.8%. That band is too
    NARROW.

A single "EGX has fat tails, widen everything" fix would have made the daily
band worse. The tails really are extreme -- P(|z|>3) is 2.00% against a normal
0.27%, and P(|z|>5) is 0.30% against 0.000057% -- but that coexists with a
squashed centre.

TWO FIXES THAT WERE TESTED AND FAILED. Do not retry them:
  * iid bootstrap of daily returns -> 84.0% coverage, no better than Gaussian,
    because resampling destroys the volatility clustering that creates the fat
    aggregate tail in the first place.
  * empirical distribution of trailing overlapping 60-day returns -> 78.9%,
    worse than what it replaced.

WHAT WORKS: EGX's own fitted |z| quantiles, applied to a zero-drift lognormal
band. Measured coverage lands on the nominal figure to a tenth of a point.

FIT WITH QUANTILES, NEVER A STANDARD DEVIATION. The z-distribution's moments are
meaningless here -- a handful of collapsed names give it a mean in the hundreds
against a median near zero. `numpy.percentile` is the only safe estimator.

NO DRIFT TERM, DELIBERATELY
---------------------------
The old cone's median line was P0*exp(days*(mu - sigma^2/2)) -- the trailing
400-day mean return, compounded forward. Rendered as a price and coloured green
when above spot, that is a price target with a direction attached, which this
app does not do (see "Removed: Max Buy Price"). The band here is symmetric in
log space and makes no directional claim at all.

Dropping the simulation also removed a live bug: monte_carlo_forecast drew from
`np.random.default_rng(None)`, so the displayed p5/p95 changed on every cache
miss. The band below is a closed-form function of the inputs and cannot jitter.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Fitted calibration. Regenerate with:  python -m scripts.calibrate
# ---------------------------------------------------------------------------

CALIBRATION = {
    "fitted_at": "2026-09-02",
    "n_observations": 34721,
    # ISIN-coded rows (EGS30AJ1C016, EGS48271C018-EGP, ...) are excluded because
    # they are not common stocks. The exclusion matches on LENGTH, not on the
    # "EGS" prefix — EGSA is a real four-character EGX ticker and a prefix rule
    # silently drops it.
    "universe": "EGX common stocks with >=460 daily bars, ISIN-coded "
                "instruments excluded",
    "sigma_window": 400,      # matches INTERNAL_BARS_MIN, the window every
                              # scoring path actually fetches
    "fit_horizon_days": 60,
}

# EGX's own |z| at each coverage level for a 60-day move, where
# z = (P_t/P_0 - 1) / (sigma_daily * sqrt(t)). Gaussian equivalents in the
# comment show how far this market is from the textbook.
EGX_Z_CONE_60D = {
    50: 0.600,   # gaussian 0.6745
    80: 1.333,   # gaussian 1.2816
    90: 1.999,   # gaussian 1.6449
    95: 2.791,   # gaussian 1.9600
}

# What the +/-1-sigma band this module draws ACTUALLY covers, per horizon.
# These replace the flat "68%" the UI used to print.
ONE_SIGMA_COVERAGE_PCT = {
    "daily": 79.0,
    "weekly": 76.3,
    "monthly": 73.0,
}

# Bands the outcome cone returns, inner first. Two is enough to read a shape on
# a phone; five was noise.
OUTCOME_BAND_COVERAGES = (80, 90)

# Below this many observations a per-stock sigma is not worth drawing a band
# from at all.
MIN_OBSERVATIONS = 20

_HORIZON_BARS = {"daily": 1, "weekly": 5, "monthly": 22}


def expected_move(returns: pd.Series) -> Optional[dict]:
    """
    The 1-sigma band at daily / weekly / monthly scale, with its REAL coverage.

    The percentages are unchanged from before; what changed is that each one now
    carries the coverage it was measured to deliver on EGX rather than the 68%
    Gaussian theory predicts. Callers must render `coverage_pct`, never a
    hardcoded figure -- tests/test_forecast_presentation.py enforces that.

    Returns None when there are too few observations or sigma is degenerate.
    """
    rets = returns.dropna()
    if len(rets) < MIN_OBSERVATIONS:
        return None
    sigma_daily = float(rets.std())
    if not math.isfinite(sigma_daily) or sigma_daily <= 0:
        return None

    out = {"method": "one_sigma_historical", "calibration": CALIBRATION}
    for name, bars in _HORIZON_BARS.items():
        out[f"{name}_pct"] = round(sigma_daily * math.sqrt(bars) * 100.0, 2)
        out[f"{name}_coverage_pct"] = ONE_SIGMA_COVERAGE_PCT[name]
    return out


def outcome_band(returns: pd.Series, current_price: float,
                 days: int = 60) -> Optional[dict]:
    """
    Calibrated range of where the price has historically ended up, in EGP.

    Replaces `monte_carlo_forecast`. Same job, but the quantiles come from
    EGX's measured return distribution instead of Gaussian draws, so the
    advertised coverage is the delivered coverage (90.0% measured against 90%
    nominal, versus 85.8% before).

    Shape, per band:
        lo(t) = P0 * exp(-z * sigma * sqrt(t))
        hi(t) = P0 * exp(+z * sigma * sqrt(t))

    Symmetric in log space and drift-free on purpose -- see the module
    docstring. There is no median series, because a median here would be a
    price target.

    CALIBRATION HORIZON: z was fitted at t = 60 trading days. The intermediate
    days are the sqrt-of-time interpolation needed to draw a shape, and are NOT
    separately validated; `endpoint` carries the day-`days` values, which are
    the ones the coverage claim actually applies to.
    """
    rets = returns.dropna()
    if len(rets) < MIN_OBSERVATIONS or current_price <= 0 or days < 1:
        return None
    sigma = float(rets.std())
    if not math.isfinite(sigma) or sigma <= 0:
        return None

    bands = []
    for coverage in OUTCOME_BAND_COVERAGES:
        z = EGX_Z_CONE_60D[coverage]
        lo, hi = [], []
        for t in range(1, days + 1):
            spread = z * sigma * math.sqrt(t)
            lo.append(round(current_price * math.exp(-spread), 2))
            hi.append(round(current_price * math.exp(spread), 2))
        bands.append({
            "coverage_pct": coverage,
            "z": z,
            "lo": lo,
            "hi": hi,
            # How often this band is expected to be WRONG, stated explicitly.
            # The complement is the single best-evidenced fix in the
            # forecast-communication literature -- readers systematically
            # misjudge which event a probability refers to without it.
            "outside_pct": 100 - coverage,
        })

    widest = bands[-1]
    return {
        "days": days,
        "current_price": round(current_price, 2),
        "method": "egx_calibrated_quantile",
        "calibration": CALIBRATION,
        "bands": bands,
        # The day-`days` values, i.e. the only ones the coverage claim covers.
        "endpoint": {
            "coverage_pct": widest["coverage_pct"],
            "lo": widest["lo"][-1],
            "hi": widest["hi"][-1],
        },
    }
