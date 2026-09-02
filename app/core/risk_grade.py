"""
Per-stock risk forecasting — the one thing this app can genuinely predict.

WHY THIS EXISTS
---------------
The composite score was tested on 36,818 symbol-dates and cannot rank EGX
stocks: cross-sectional IC ~0 and slightly NEGATIVE (-0.029, t=-2.85 at 21
days). Nine of ten deciles had a median 21-day forward return of exactly 0.00%.
That is not a bug to fix by adding indicators; returns are close to a
martingale difference and the market is telling the truth.

Volatility is the opposite. It clusters, it mean-reverts, and it is strongly
persistent. Measured on this repo's own cache (`python -m scripts.calibrate`),
liquid universe, 16,220 observations across 202 symbols and 221 monthly dates:

    past 63d volatility ->            IC       t     non-overlapping IC / t
      next 126d realized volatility  +0.5631  +55.8      +0.5791  +24.0
      next 126d max drawdown         +0.4338  +40.5      +0.4707  +16.7

For scale, that is roughly TWENTY TIMES the magnitude of any return signal
found anywhere in this project, and it clears the pre-registered |t| > 3.0
evidence bar on non-overlapping data by a factor of eight.

WHAT THIS MODULE MUST NEVER CLAIM
---------------------------------
That calm stocks go UP. They do not, reliably. Low volatility does rank
positively against forward returns (IC +0.084, t=4.97 at 21 days), but the
realisable long/short spread is only t=1.70 over the full sample, the mean
forward return by quintile is flat-to-inverted, and no historical market cap
exists in the cache to neutralise a possible size effect. High-volatility EGX
names behave like lottery tickets: a few huge winners lift the MEAN while the
MEDIAN is clearly worse.

So this surface answers "how much will this move, and how deep a hole should I
expect" — never "will it go up". Attaching a return claim to a risk grade is
exactly the mistake that produced Buy/Sell labels and Max Buy Price.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

# Fitted by scripts/calibrate.py on 2026-09-02. Regenerate whenever the cache
# is refreshed; the script prints this table directly.
RISK_CALIBRATION = {
    "fitted_at": "2026-09-02",
    "n_observations": 16220,
    "n_symbols": 202,
    "n_dates": 221,
    "lookback_days": 63,
    "forward_days": 126,
    "vol_predicts_vol_ic": 0.5631,
    "vol_predicts_vol_t": 55.8,
    "vol_predicts_vol_ic_nonoverlapping": 0.5791,
    "vol_predicts_vol_t_nonoverlapping": 24.0,
    "vol_predicts_drawdown_ic": 0.4338,
    "vol_predicts_drawdown_t": 40.5,
    "vol_predicts_drawdown_ic_nonoverlapping": 0.4707,
    "vol_predicts_drawdown_t_nonoverlapping": 16.7,
}

# What each volatility quintile went on to do over the next 126 trading days.
# Medians, not means: EGX forward returns are heavily right-skewed and a mean
# describes a distribution nobody experiences.
RISK_QUINTILES = {
    1: {"future_vol_ann_pct": 32.0, "median_max_drawdown_pct": 20.0, "p90_max_drawdown_pct": 40.7},
    2: {"future_vol_ann_pct": 37.8, "median_max_drawdown_pct": 24.8, "p90_max_drawdown_pct": 46.3},
    3: {"future_vol_ann_pct": 42.4, "median_max_drawdown_pct": 27.6, "p90_max_drawdown_pct": 49.3},
    4: {"future_vol_ann_pct": 45.2, "median_max_drawdown_pct": 29.9, "p90_max_drawdown_pct": 53.0},
    5: {"future_vol_ann_pct": 51.2, "median_max_drawdown_pct": 34.2, "p90_max_drawdown_pct": 59.4},
}

RISK_BANDS = {
    1: {"key": "calm", "label": "Calm"},
    2: {"key": "steady", "label": "Steady"},
    3: {"key": "average", "label": "Average"},
    4: {"key": "jumpy", "label": "Jumpy"},
    5: {"key": "wild", "label": "Wild"},
}

# The tradeable universe the calibration was fitted on. Ranking a stock nobody
# can enter or exit against names that trade is meaningless, and the percentile
# it produces would be a fiction.
LIQUID_MIN_TURNOVER_EGP = 1_000_000
LIQUID_MIN_TRADED_SHARE = 0.80
LIQUID_MIN_PRICE_EGP = 0.5
LIQUIDITY_WINDOW_BARS = 60

VOL_LOOKBACK_BARS = 63
TRADING_DAYS_PER_YEAR = 252

# RiskMetrics (1996) derives lambda = 0.94 for a one-day horizon on developed
# markets. This app uses 0.97, and the reason is measured rather than inherited:
# `scripts/vol_backtest.py` over the full cached universe finds 0.97 beats a
# 400-bar trailing window on QLIKE at every horizon the app draws
# (3.1% at 5 days, 2.6% at 22), clearing the project's |t| > 3.0 bar at one day,
# while 0.94 is the better one-day forecaster but LOSES to the trailing window
# at 60 days. Longer memory suits a market where 19% of daily returns are
# exactly zero: 0.94 over-reacts to a run of flat sessions followed by one real
# move.
EWMA_LAMBDA = 0.97
EWMA_SEED_BARS = 60

# Without a floor, a run of flat or limit-locked EGX sessions drives the EWMA
# variance toward zero and the band collapses to a point. 19% of this market's
# daily returns are EXACTLY zero, so this is the common case, not the edge case.
EWMA_VARIANCE_FLOOR_SHARE = 0.10


def ewma_variance_series(returns: pd.Series,
                         lam: float = EWMA_LAMBDA) -> Optional[np.ndarray]:
    """
    The whole conditional-variance path, h_t, one value per input return.

    RiskMetrics: h_t = lam*h_{t-1} + (1-lam)*r_{t-1}^2, so h_t is a forecast for
    period t made from information up to t-1 — strictly one-step-ahead, which is
    what makes it usable in a walk-forward evaluation without look-ahead.

    Returned as a series rather than just the last value so scripts/vol_backtest.py
    can score the SHIPPED recursion rather than a re-implementation of it. A
    harness that grades a copy of the code grades nothing.

    The floor is applied INSIDE the recursion, not to the output, because it
    feeds forward: clamping after the fact would let a quiet stretch drive the
    path arbitrarily low and only lift the final reading.
    """
    rets = returns.dropna()
    if len(rets) < EWMA_SEED_BARS:
        return None
    values = rets.to_numpy(dtype=float)
    seed = float(np.var(values[:EWMA_SEED_BARS], ddof=1))
    if not math.isfinite(seed) or seed <= 0:
        return None

    floor = EWMA_VARIANCE_FLOOR_SHARE * float(np.var(values, ddof=1))
    out = np.empty(len(values), dtype=float)
    out[:EWMA_SEED_BARS] = seed
    h = seed
    for i in range(EWMA_SEED_BARS, len(values)):
        h = lam * h + (1.0 - lam) * values[i - 1] * values[i - 1]
        if floor > 0 and h < floor:
            h = floor
        out[i] = h
    return out


def ewma_volatility(returns: pd.Series, lam: float = EWMA_LAMBDA) -> Optional[float]:
    """
    RiskMetrics conditional volatility — the DAILY sigma, or None on thin data.

    The variance floor is not defensive decoration. EGX prints a lot of exactly
    zero returns; without it a quiet stretch produces sigma ~ 0 and every band
    built on it collapses to a point.
    """
    series = ewma_variance_series(returns, lam)
    if series is None:
        return None
    h = float(series[-1])
    if not math.isfinite(h) or h <= 0:
        return None
    return math.sqrt(h)


def annualized(sigma_daily: float) -> float:
    return sigma_daily * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0


def is_tradeable(close: pd.Series, volume: pd.Series) -> bool:
    """
    Would a retail investor actually be able to enter and exit this?

    Deliberately measured over the symbol's own recent bars for turnover, but
    note the known limitation this does NOT fix: a genuinely dead stock often
    has no rows at all rather than zero-volume rows, so a row-based test can
    under-detect it. The universe filter in the snapshot job pairs this with a
    freshness check against the benchmark calendar.
    """
    if close is None or volume is None or len(close) < LIQUIDITY_WINDOW_BARS:
        return False
    price = float(close.iloc[-1])
    if not math.isfinite(price) or price <= LIQUID_MIN_PRICE_EGP:
        return False
    window = slice(-LIQUIDITY_WINDOW_BARS, None)
    turnover = float((close * volume).iloc[window].mean())
    if not math.isfinite(turnover) or turnover < LIQUID_MIN_TURNOVER_EGP:
        return False
    traded = float((volume.iloc[window] > 0).mean())
    return traded >= LIQUID_MIN_TRADED_SHARE


def measure(close: pd.Series, volume: pd.Series,
            benchmark_close: Optional[pd.Series] = None) -> Optional[dict]:
    """
    The per-symbol inputs the snapshot stores. Cross-sectional ranking happens
    at READ time, not here — see grade_universe.
    """
    if close is None or len(close) < VOL_LOOKBACK_BARS + 1:
        return None
    rets = close.pct_change().dropna()
    if len(rets) < VOL_LOOKBACK_BARS:
        return None

    trailing = float(rets.iloc[-VOL_LOOKBACK_BARS:].std())
    if not math.isfinite(trailing) or trailing <= 0:
        return None
    ewma = ewma_volatility(rets)

    beta = None
    if benchmark_close is not None and len(benchmark_close) > VOL_LOOKBACK_BARS:
        try:
            from app.core.indicators import compute_beta
            b = compute_beta(rets, benchmark_close.pct_change().dropna())
            if b is not None and math.isfinite(b):
                beta = round(float(b), 2)
        except Exception:
            beta = None

    window = slice(-LIQUIDITY_WINDOW_BARS, None)
    return {
        # The calibrated input. EWMA is reported alongside because it forecasts
        # better, but the PERCENTILE is built on the trailing sigma the
        # calibration was actually fitted on — swapping the ranking input
        # without refitting would invalidate the quintile table below it.
        "sigma_63_ann_pct": round(annualized(trailing), 2),
        "sigma_ewma_ann_pct": round(annualized(ewma), 2) if ewma else None,
        "beta": beta,
        "turnover_egp": round(float((close * volume).iloc[window].mean()), 2),
        "traded_share": round(float((volume.iloc[window] > 0).mean()), 4),
        "last_price": round(float(close.iloc[-1]), 4),
    }


def grade_universe(rows: list) -> list:
    """
    Turn stored per-symbol sigmas into percentiles and bands, ACROSS the
    universe, at read time.

    Doing this on read rather than in the cron is what makes the chunked
    refresh safe: there is no run to finalize, no cursor state to corrupt, and
    a half-refreshed table still yields sensible ranks because every symbol
    carries its most recent measurement. The cost is one numpy sort over a few
    hundred rows.

    `rows` is [{symbol, sigma_63_ann_pct, ...}]. Returns the same dicts with
    pct_rank / quintile / band / expected-forward figures added.
    """
    usable = [r for r in rows
              if r.get("sigma_63_ann_pct") is not None
              and math.isfinite(float(r["sigma_63_ann_pct"]))]
    if len(usable) < 5:
        # Too thin to rank. Return the raw measurements ungraded rather than
        # inventing a percentile from a handful of names.
        for r in rows:
            r["pct_rank"] = None
            r["quintile"] = None
            r["band"] = None
        return rows

    sigmas = np.array([float(r["sigma_63_ann_pct"]) for r in usable])
    order = sigmas.argsort().argsort()          # 0 = calmest
    n = len(usable)

    for r, rank in zip(usable, order):
        pct = float(rank) / max(n - 1, 1)
        quintile = min(5, int(pct * 5) + 1)
        r["pct_rank"] = round(pct * 100, 1)
        r["quintile"] = quintile
        r["band"] = RISK_BANDS[quintile]["key"]
        r["band_label"] = RISK_BANDS[quintile]["label"]
        # What this quintile HISTORICALLY went on to do. Not a promise about
        # this stock — the wording on screen must stay in the past tense.
        r["historical"] = dict(RISK_QUINTILES[quintile])

    for r in rows:
        r.setdefault("pct_rank", None)
        r.setdefault("quintile", None)
        r.setdefault("band", None)
    return rows
