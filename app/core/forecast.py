"""
Per-stock forecast helpers.

Two descriptive views of a stock's historical volatility, neither of which is
a prediction:

1. `expected_move` — "How much does this stock typically move in a day / week /
   month?" A 1-σ band derived from the standard deviation of daily returns.
   Roughly 2 of every 3 daily moves fall inside the daily band; 1 in 3 is
   bigger. Useful to set expectations before interpreting a single day's move.

2. `monte_carlo_forecast` — "Given the stock's historical drift and vol, what
   range of price paths is plausible over the next N days?" Runs a vectorized
   geometric-Brownian-like simulation and returns percentile cones (p5..p95)
   in EGP. Not a directional forecast — fat-tail / regime-shift risk is
   explicitly NOT modelled.

Both return None when there's insufficient data; the caller renders the card
with an "unavailable" state.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def expected_move(returns: pd.Series) -> Optional[dict]:
    """
    Compute a 1-σ expected-move band at daily / weekly / monthly scales.
    Returns None when <20 observations or σ is zero / non-finite.
    """
    rets = returns.dropna()
    if len(rets) < 20:
        return None
    sigma_daily = float(rets.std())
    if not math.isfinite(sigma_daily) or sigma_daily <= 0:
        return None
    return {
        "daily_pct": round(sigma_daily * 100.0, 2),
        "weekly_pct": round(sigma_daily * math.sqrt(5) * 100.0, 2),
        "monthly_pct": round(sigma_daily * math.sqrt(22) * 100.0, 2),
        "method": "one_sigma_historical",
    }


def monte_carlo_forecast(returns: pd.Series, current_price: float,
                         days: int = 60, n_sims: int = 1000,
                         seed: Optional[int] = None) -> Optional[dict]:
    """
    Vectorized GBM-style simulation of future prices in EGP.

    Uses the stock's own historical mu/sigma (daily) to draw `n_sims` paths of
    length `days`, then returns the p5/p25/p50/p75/p95 percentile cones. Paths
    are price levels (EGP), not returns — the frontend charts them directly.

    Must be vectorized (`np.random.normal(mu, sigma, (n_sims, days))` +
    `np.cumprod`) — never loop. A 1000×60 draw is sub-millisecond; anything
    slower means something is wrong.
    """
    rets = returns.dropna()
    if len(rets) < 20 or current_price <= 0:
        return None
    mu = float(rets.mean())
    sigma = float(rets.std())
    if not (math.isfinite(mu) and math.isfinite(sigma)) or sigma <= 0:
        return None

    rng = np.random.default_rng(seed)
    sims = rng.normal(mu, sigma, (n_sims, days))
    paths = current_price * np.cumprod(1 + sims, axis=1)

    def _pct(p):
        return [round(float(v), 2) for v in np.percentile(paths, p, axis=0)]

    return {
        "days": days,
        "current_price": round(current_price, 2),
        "percentiles": {
            "p5":  _pct(5),
            "p25": _pct(25),
            "p50": _pct(50),
            "p75": _pct(75),
            "p95": _pct(95),
        },
    }
