"""
Fit the constants the app ships as claims, from the cached panel.

WHY THIS EXISTS
---------------
Two numbers in this app were stated on screen as measured facts and were wrong:

1. `core/regime.py` claimed the market-condition reading correlates +0.318
   (t=2.84) with EGX30's next 63 days, and defended it with "de-overlapping made
   the correlation STRONGER (+0.170 -> +0.318), so it is not an overlap
   artifact." At a 63-day horizon on a ~21-day rebalance grid there are exactly
   THREE valid de-overlapped samplings. They give +0.318, +0.180 and +0.004.
   De-overlapping did not validate the number, it RESAMPLED it, and the best of
   three draws was read as a robustness check.

2. `core/forecast.py` advertised "68% of days" for its 1-sigma band and 90% for
   its p5-p95 Monte Carlo cone. Measured, they deliver ~79% and ~86%.

Both are fixed by fitting the constants here and importing them, so the numbers
on screen and the numbers in the data cannot drift apart again. Re-run this
whenever the cache is refreshed; the bundle lives in
`app/core/calibration/<market>.py`. `--market US --write` regenerates `us.py`;
EG's `eg.py` is the frozen record, so `--write` refuses EG.

THE CORRECT STATISTIC FOR OVERLAPPING WINDOWS
---------------------------------------------
Not "throw away two thirds of the data and hope you picked a good phase".
Keep every observation and correct the standard error for the autocorrelation
the overlap induces -- Newey-West with lag = the overlap depth. That is
reported here as `nw_t`, alongside all three de-overlapped phases so a future
reader can see the spread rather than one lucky draw.

Run:  ./.venv/bin/python -m scripts.calibrate [--market US] [--write]
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date  # noqa: E402

from app.core.calibration import (  # noqa: E402
    Calibration, ForecastCal, FundamentalsCal, RegimeBand, RegimeCal, RiskCal,
    band_index, for_market,
)
from app.core.index_membership import index_member_tier, symbols_in_index  # noqa: E402
from app.core.markets import EG, MARKETS  # noqa: E402
from scripts.market_cache import (  # noqa: E402
    REPO_ROOT, CacheSpec, read_universe_rows, regime_members, spec_for,
)

# The horizon regime.py reports, and the rebalance spacing of the panel. Their
# ratio is how many disjoint phases the de-overlapped grid has.
REGIME_HORIZON = 63
REBALANCE = 21
PHASES = REGIME_HORIZON // REBALANCE

# Matches INTERNAL_BARS_MIN -- the window every scoring path actually fetches,
# so the fitted band is the band the app can really compute at request time.
SIGMA_WINDOW = 400
CONE_DAYS = 60

# ---------------------------------------------------------------------------
# Shared statistics
# ---------------------------------------------------------------------------

def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation, computed as Pearson on ranks so no scipy is needed."""
    g = pd.concat([a, b], axis=1).dropna()
    return float(g.iloc[:, 0].rank().corr(g.iloc[:, 1].rank()))


def newey_west_t(a: pd.Series, b: pd.Series, lag: int) -> tuple:
    """
    HAC-corrected t on the rank association, keeping every observation.

    Regresses standardised rank(b) on standardised rank(a) and corrects the
    standard error with a Bartlett kernel out to `lag`. This is the honest
    alternative to de-overlapping: overlapping windows share data, which
    inflates a naive t-stat, and discarding data to avoid that both throws away
    power and makes the answer depend on which phase you happened to start on.
    """
    g = pd.concat([a, b], axis=1).dropna()
    x = g.iloc[:, 0].rank().values.astype(float)
    y = g.iloc[:, 1].rank().values.astype(float)
    x = (x - x.mean()) / x.std(ddof=1)
    y = (y - y.mean()) / y.std(ddof=1)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    u = resid[:, None] * X
    S = u.T @ u
    for L in range(1, lag + 1):
        G = u[L:].T @ u[:-L]
        S += (1.0 - L / (lag + 1)) * (G + G.T)
    se = float(np.sqrt((XtX_inv @ S @ XtX_inv)[1, 1]))
    return float(beta[1] / se), n


def naive_t(rho: float, n: int) -> float:
    return float(rho * np.sqrt((n - 2) / (1 - rho * rho)))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

MIN_FUNDAMENTALS_ROWS = 50
MIN_FLOOR_SYMBOLS = 10
TURNOVER_FLOOR_PCTILE = 5
LIQUIDITY_WINDOW = 60
# The usual US penny-stock line. EG's price floor comes from MarketConfig.
PRICE_FLOOR = {"US": 5.0}


class CalibrationRefused(RuntimeError):
    """The data cannot support a bundle; nothing is written."""


def load_benchmark(spec: CacheSpec) -> pd.Series:
    with open(os.path.join(spec.cache_dir, f"{spec.benchmark_symbol}.pkl"), "rb") as f:
        df = pickle.load(f)
    df.columns = [str(c).lower() for c in df.columns]
    return df["close"].astype(float)


def iter_symbol_frames(spec: CacheSpec | None = None, market=None):
    """Every cached frame that is a valid symbol of `market` with a close column.
    ISIN-coded instruments are dropped by MarketConfig.is_valid_symbol (LENGTH,
    not prefix: EGSA is a real EGX ticker). No arguments reads EG's cache, as
    scripts/vol_backtest.py and scripts/factor_backtest.py do."""
    spec = spec if spec is not None else spec_for(EG.code)
    market = market if market is not None else EG
    skip = {"panel", "breadth", *spec.frame_exclude}
    for path in sorted(glob.glob(os.path.join(spec.cache_dir, "*.pkl"))):
        name = os.path.basename(path)[:-4]
        if name in skip or not market.is_valid_symbol(name):
            continue
        with open(path, "rb") as f:
            df = pickle.load(f)
        if df is None or getattr(df, "empty", True):
            continue
        df.columns = [str(c).lower() for c in df.columns]
        if "close" not in df.columns:
            continue
        yield name, df


def load_frames(spec: CacheSpec, market) -> dict:
    return dict(iter_symbol_frames(spec, market))


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------

def regime_calibration(spec: CacheSpec) -> dict:
    """
    Tercile cutoffs, per-band forward record, and the association behind them.

    Averages the panel over the EGX30+EGX70 universe specifically, because that
    is what the live endpoint averages. Calibrating on the full panel gives
    materially different cutoffs (43.7 / 49.8 rather than 45.1 / 51.5), and a
    band boundary that does not match the universe it is applied to is a silent
    off-by-one on every reading.
    """
    panel = pd.read_pickle(os.path.join(spec.cache_dir, "panel.pkl"))
    bench = load_benchmark(spec)
    panel = panel[panel["symbol"].isin(regime_members(spec))]

    mean_score = panel.groupby("date")["score"].mean().sort_index()
    rows = []
    for date, score in mean_score.items():
        i = int(bench.index.searchsorted(date, side="right")) - 1
        if i < 0 or i + REGIME_HORIZON >= len(bench):
            continue
        fwd = (float(bench.iloc[i + REGIME_HORIZON]) / float(bench.iloc[i]) - 1) * 100
        rows.append((date, float(score), fwd))
    R = pd.DataFrame(rows, columns=["date", "mean_score", "fwd"]).set_index("date")

    lo, hi = R["mean_score"].quantile([1 / 3, 2 / 3]).values
    rho = spearman(R["mean_score"], R["fwd"])
    nw, n = newey_west_t(R["mean_score"], R["fwd"], PHASES)

    phases = []
    for offset in range(PHASES):
        sub = R.iloc[offset::PHASES]
        r = spearman(sub["mean_score"], sub["fwd"])
        phases.append({"offset": offset, "rho": round(r, 4),
                       "t": round(naive_t(r, len(sub)), 2), "n": len(sub)})

    bands = {}
    for key, a, b in (("weak", -np.inf, lo), ("mixed", lo, hi), ("broad", hi, np.inf)):
        g = R[(R["mean_score"] >= a) & (R["mean_score"] < b)]["fwd"]
        bands[key] = {"n": len(g), "median": round(float(g.median()), 2),
                      "mean": round(float(g.mean()), 2),
                      "positive_rate": round(float((g > 0).mean()), 3)}

    return {"n_readings": n, "cutoff_weak_max": round(float(lo), 1),
            "cutoff_mixed_max": round(float(hi), 1),
            "rho_overlapping": round(rho, 4), "naive_t": round(naive_t(rho, n), 2),
            "newey_west_t": round(nw, 2), "phases": phases, "bands": bands,
            "universe": spec.regime_universe_label, "horizon_days": REGIME_HORIZON}


# ---------------------------------------------------------------------------
# Forecast bands
# ---------------------------------------------------------------------------

def forecast_calibration(spec: CacheSpec, market, sigma_method: str = "trailing") -> dict:
    """
    Fit EGX's own |z| quantiles for the daily band and the 60-day cone.

    The two need DIFFERENT corrections, in OPPOSITE directions, which is the
    part that is easy to miss:

      - Daily: EGX's body is THIN, not fat. A +/-1-sigma band covers ~79% of
        next-day moves, not the Gaussian 68%, because price limits and flat
        illiquid sessions pile mass near zero. The band is too WIDE for its
        advertised coverage.
      - 60-day: compounding plus volatility clustering fattens the aggregate,
        so the Gaussian 90% cone covers only ~86%. That band is too NARROW.

    Fitted with QUANTILES, never a standard deviation: a handful of collapsed
    names give the z-distribution a mean and variance that are meaningless
    (mean in the hundreds against a median near zero).
    """
    HORIZONS = {"daily": 1, "weekly": 5, "monthly": 22}
    buckets = {k: [] for k in HORIZONS}
    cone, cone_sigma = [], []
    first_year, last_year = None, None
    use_ewma = sigma_method == "ewma"
    if use_ewma:
        from app.core.risk_grade import ewma_variance_series

    for _, df in iter_symbol_frames(spec, market):
        close = df["close"].astype(float)
        if len(close) < SIGMA_WINDOW + CONE_DAYS + REBALANCE:
            continue
        y0, y1 = close.index[0].year, close.index[-1].year
        first_year = y0 if first_year is None else min(first_year, y0)
        last_year = y1 if last_year is None else max(last_year, y1)
        rets = close.pct_change()
        ewma_path = None
        if use_ewma:
            series = ewma_variance_series(rets.dropna())
            if series is None:
                continue
            ewma_path = np.concatenate([[np.nan], np.sqrt(series)])
        for i in range(SIGMA_WINDOW, len(close) - CONE_DAYS - 1, REBALANCE):
            if use_ewma:
                sigma = float(ewma_path[i]) if i < len(ewma_path) else float("nan")
            else:
                sigma = float(rets.iloc[i - SIGMA_WINDOW:i].std())
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            p0 = float(close.iloc[i - 1])
            if p0 <= 0:
                continue
            for name, h in HORIZONS.items():
                pn = float(close.iloc[i - 1 + h])
                if np.isfinite(pn):
                    buckets[name].append((pn / p0 - 1.0) / (sigma * np.sqrt(h)))
            pn = float(close.iloc[i - 1 + CONE_DAYS])
            if np.isfinite(pn):
                cone.append((pn / p0 - 1.0) / (sigma * np.sqrt(CONE_DAYS)))
                cone_sigma.append(sigma)

    def _abs(x):
        a = np.abs(np.array(x, dtype=float))
        return a[np.isfinite(a)]

    c = _abs(cone)
    if c.size < 100:
        raise CalibrationRefused(f"forecast: only {c.size} cone observations in {spec.cache_dir}")
    out = {"n_cone": int(c.size), "cone_days": CONE_DAYS, "sigma_method": sigma_method,
           "sigma_window": SIGMA_WINDOW,
           "cone_gaussian_90_coverage": round(float((c <= 1.6449).mean() * 100), 1),
           "one_sigma_coverage": {}, "z_daily": {}, "z_cone": {},
           "first_year": first_year, "last_year": last_year}
    for name in HORIZONS:
        b = _abs(buckets[name])
        out["one_sigma_coverage"][name] = round(float((b <= 1.0).mean() * 100), 1)
        if name == "daily":
            out["n_daily"] = int(b.size)
            for pct in (50, 80, 90, 95):
                out["z_daily"][pct] = round(float(np.percentile(b, pct)), 3)
    for pct in (50, 80, 90, 95):
        out["z_cone"][pct] = round(float(np.percentile(c, pct)), 3)

    # Sharpness: how wide the fitted 90% band is, as a share of spot.
    z90 = out["z_cone"][90]
    s = np.array(cone_sigma, dtype=float) * np.sqrt(CONE_DAYS)
    width = (np.exp(z90 * s) - np.exp(-z90 * s)) * 100.0
    out["band_width_pct_of_spot"] = {
        "median": int(round(float(np.percentile(width, 50)))),
        "p25": int(round(float(np.percentile(width, 25)))),
        "p75": int(round(float(np.percentile(width, 75)))),
    }
    out["cone_90_coverage"] = round(float((c <= z90).mean() * 100), 1)
    return out


# ---------------------------------------------------------------------------
# Risk grade
# ---------------------------------------------------------------------------

# The tradeable cut. A result that only holds among names nobody can enter or
# exit is not a result a retail investor could have acted on.
LIQUID_MIN_TRADED_SHARE = 0.80     # of the last 60 sessions the symbol has rows for
RISK_FORWARD_DAYS = 126
RISK_LOOKBACK_DAYS = 63


def risk_calibration(spec: CacheSpec, market, min_turnover: float, min_price: float) -> dict:
    """
    Does past volatility predict FUTURE volatility and drawdown on the EGX?

    This is the one question where the answer is emphatically yes, and it is
    what the Risk Grade surface rests on. Volatility is a persistent,
    long-memory process; returns are close to a martingale difference. The app
    has spent its life trying to rank the second thing.

    Reported the same way as everything else here: per-date rank IC with a
    t-stat across dates (one observation per date, so overlapping forward
    windows cannot inflate it), plus a NON-OVERLAPPING variant that samples
    every RISK_FORWARD_DAYS/REBALANCE-th date, and the quintile mapping the UI
    will actually show.
    """
    panel = pd.read_pickle(os.path.join(spec.cache_dir, "panel.pkl"))
    dates = sorted(panel["date"].unique())

    frames = {}
    for name, df in iter_symbol_frames(spec, market):
        if len(df) >= 400:
            frames[name] = df

    rows = []
    for as_of in dates:
        for symbol, df in frames.items():
            close = df["close"].astype(float)
            volume = df["volume"].astype(float)
            i = int(close.index.searchsorted(as_of, side="right")) - 1
            if i < RISK_LOOKBACK_DAYS + 1 or i + RISK_FORWARD_DAYS >= len(close):
                continue
            price = float(close.iloc[i])
            if price <= min_price:
                continue

            window = slice(i - 59, i + 1)
            turnover = float((close * volume).iloc[window].mean())
            if not np.isfinite(turnover) or turnover < min_turnover:
                continue
            traded = float((volume.iloc[window] > 0).mean())
            if traded < LIQUID_MIN_TRADED_SHARE:
                continue

            rets = close.pct_change()
            past = float(rets.iloc[i - RISK_LOOKBACK_DAYS + 1:i + 1].std())
            fwd_rets = rets.iloc[i + 1:i + 1 + RISK_FORWARD_DAYS]
            fwd_vol = float(fwd_rets.std())
            if not (np.isfinite(past) and np.isfinite(fwd_vol)) or past <= 0:
                continue

            # Deepest peak-to-trough over the forward window, as a positive %.
            path = close.iloc[i:i + 1 + RISK_FORWARD_DAYS]
            drawdown = float((path / path.cummax() - 1.0).min() * -100.0)

            rows.append({
                "date": as_of, "symbol": symbol,
                "past_vol": past,
                "fwd_vol_ann_pct": fwd_vol * np.sqrt(252) * 100.0,
                "fwd_max_dd_pct": drawdown,
            })

    F = pd.DataFrame(rows)
    if F.empty:
        return {"error": "no rows"}

    def per_date_ic(target: str, every: int = 1) -> dict:
        keep = sorted(F["date"].unique())[::every]
        per = []
        for _, g in F[F["date"].isin(keep)].groupby("date"):
            g = g[["past_vol", target]].dropna()
            if len(g) < 10 or g["past_vol"].nunique() < 3:
                continue
            v = g["past_vol"].rank().corr(g[target].rank())
            if v == v:
                per.append(v)
        a = np.array(per, dtype=float)
        if len(a) < 5:
            return {"ic": None, "t": None, "n_dates": len(a)}
        return {"ic": round(float(a.mean()), 4),
                "t": round(float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), 2),
                "n_dates": len(a)}

    # Non-overlapping: forward windows are RISK_FORWARD_DAYS long and dates are
    # REBALANCE apart, so sampling every 6th date makes them disjoint.
    every = max(1, RISK_FORWARD_DAYS // REBALANCE)

    F["quintile"] = F.groupby("date")["past_vol"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    quintiles = []
    for q in sorted(F["quintile"].dropna().unique()):
        g = F[F["quintile"] == q]
        quintiles.append({
            "quintile": int(q) + 1,
            "n": int(len(g)),
            "future_vol_ann_pct": round(float(g["fwd_vol_ann_pct"].median()), 1),
            "median_max_drawdown_pct": round(float(g["fwd_max_dd_pct"].median()), 1),
            "p90_max_drawdown_pct": round(float(g["fwd_max_dd_pct"].quantile(0.90)), 1),
        })

    return {
        "n_observations": int(len(F)),
        "n_symbols": int(F["symbol"].nunique()),
        "n_dates": int(F["date"].nunique()),
        "median_names_per_date": int(F.groupby("date").size().median()),
        "lookback_days": RISK_LOOKBACK_DAYS,
        "forward_days": RISK_FORWARD_DAYS,
        "vol_predicts_vol": per_date_ic("fwd_vol_ann_pct"),
        "vol_predicts_vol_nonoverlapping": per_date_ic("fwd_vol_ann_pct", every),
        "vol_predicts_drawdown": per_date_ic("fwd_max_dd_pct"),
        "vol_predicts_drawdown_nonoverlapping": per_date_ic("fwd_max_dd_pct", every),
        "quintiles": quintiles,
    }


# ---------------------------------------------------------------------------

def round_down_nice(x: float) -> float:
    """Largest m*10^k <= x with m in {1, 2, 5}: 23.4M -> 20M, 9.8M -> 5M."""
    if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
        raise CalibrationRefused(f"cannot round {x!r} to a floor")
    k = math.floor(math.log10(x))
    for m in (5, 2, 1):
        if m * 10 ** k <= x:
            return float(m * 10 ** k)
    return float(10 ** k)


def read_pe_rows(db, market_code: str) -> tuple:
    """Current pe_data snapshot: (positive P/Es, positive yields). Fundamentals
    have no history, so this is a snapshot, not a backtest."""
    rows = db.execute("SELECT pe_ratio, dividend_yield FROM pe_data WHERE market = %s",
                      (market_code,)).fetchall()

    def clean(i):
        out = []
        for r in rows:
            v = r[i]
            if v is not None and math.isfinite(float(v)) and float(v) > 0:
                out.append(float(v))
        return out

    return clean(0), clean(1)


def _map_edges(ref_values, ref_edges, target_values, ndigits: int) -> list:
    """Each reference edge's quantile in the reference distribution, read off
    the target distribution: same position, other market."""
    ref = np.asarray(ref_values, dtype=float)
    tgt = np.asarray(target_values, dtype=float)
    out = [round(float(np.quantile(tgt, float((ref < e).mean()))), ndigits) for e in ref_edges]
    if any(b <= a for a, b in zip(out, out[1:])):
        raise CalibrationRefused(f"fitted band edges are not increasing: {out}")
    return out


def fit_fundamentals(eg_pe, eg_dy, pe, dy, market) -> FundamentalsCal:
    for name, values in (("EG P/E", eg_pe), ("EG yield", eg_dy),
                         (f"{market.code} P/E", pe), (f"{market.code} yield", dy)):
        if len(values) < MIN_FUNDAMENTALS_ROWS:
            raise CalibrationRefused(
                f"pe_data has {len(values)} usable {name} rows (< {MIN_FUNDAMENTALS_ROWS}); "
                f"refusing to write fundamentals")
    ref = for_market(EG).fundamentals
    pe_edges = _map_edges(eg_pe, [u for u, _ in ref.pe_bands[:-1]], pe, 1)
    dy_edges = _map_edges(eg_dy, [u for u, _ in ref.dy_bands[:-1]], dy, 2)
    fund = FundamentalsCal(
        label=market.short_name,
        pe_median=round(float(np.median(pe)), 1),
        pe_bands=tuple(zip(pe_edges + [math.inf], [p for _, p in ref.pe_bands])),
        dy_median=round(float(np.median(dy)), 2),
        dy_bands=tuple(zip(dy_edges + [math.inf], [p for _, p in ref.dy_bands])),
        dy_mid_under_risk_free=dy_edges[2] < float(market.risk_free_fallback_pct or 0.0),
    )
    # Why only the boundaries move: the median stock must land in the band the
    # EG median lands in, so "having data" is never itself worth points.
    if (band_index(fund.pe_bands, fund.pe_median) != band_index(ref.pe_bands, ref.pe_median)
            or band_index(fund.dy_bands, fund.dy_median) != band_index(ref.dy_bands, ref.dy_median)):
        raise CalibrationRefused(f"the {market.code} median falls in a different band than EG's: {fund}")
    return fund


def fit_turnover_floor(frames: dict) -> float:
    """5th percentile of 60-day mean dollar turnover, rounded down to a round number."""
    values = []
    for df in frames.values():
        t = float((df["close"].astype(float) * df["volume"].astype(float))
                  .iloc[-LIQUIDITY_WINDOW:].mean())
        if math.isfinite(t) and t > 0:
            values.append(t)
    if len(values) < MIN_FLOOR_SYMBOLS:
        raise CalibrationRefused(f"turnover floor: only {len(values)} symbols")
    return round_down_nice(float(np.percentile(values, TURNOVER_FLOOR_PCTILE)))


def volume_samples(frames: dict, symbols) -> np.ndarray:
    """20-day mean share volume, sampled every 21 bars over the trailing year."""
    out = []
    for s in symbols:
        df = frames.get(s)
        if df is None or len(df) < 40:
            continue
        avg = df["volume"].astype(float).iloc[-252:].rolling(20).mean()
        out.extend(v for v in avg.iloc[19::21] if math.isfinite(v))
    return np.asarray(out, dtype=float)


def other_listed_tier(thin: int, low: int, eg_tiers) -> tuple:
    """
    The other-listed floors: the index-member floors scaled by EG's
    NILEX/EGX30 ratios. Scaled EXACTLY, not rounded again — the inputs are
    already round, and re-rounding both down collapsed thin and low onto one
    number on the first US fit (500k/1M -> 20k/20k), which made "low"
    unreachable. A collapsed tier is refused rather than written.
    """
    thin_ref, low_ref = eg_tiers["EGX30"]
    n_thin, n_low = eg_tiers["NILEX"]
    other = (int(thin * n_thin / thin_ref), int(low * n_low / low_ref))
    if not 0 < other[0] < other[1]:
        raise CalibrationRefused(f"other-listed liquidity tier collapsed: {other}")
    return other


def fit_liquidity_tiers(eg_frames: dict, frames: dict, members, market) -> tuple:
    """
    Share-volume floors at the same percentiles of the index members'
    distribution that EG's EGX30 floors sit at in the EGX30's, scaled by EG's
    NILEX/EGX30 ratio for OTHER listings (the panel has no non-member history).

    Index members themselves get NO share-count floor, (0, 0). On the first US
    fit the member floors came out at 500k / 1M shares/day and labelled 18
    S&P members "thin, hard to exit" — AZO among them at ~$1B/day — because a
    share count means nothing across prices from $5 to $6,000. Index inclusion
    already screens for liquidity, and `liquidity_score`'s dead-session and
    calendar tests still catch a member that genuinely stops trading.
    """
    eg = for_market(EG).liquidity_tiers
    ref = volume_samples(eg_frames, symbols_in_index("EGX30"))
    tgt = volume_samples(frames, members)
    if len(ref) < 30 or len(tgt) < 30:
        raise CalibrationRefused(
            f"liquidity tiers: {len(ref)} EG / {len(tgt)} {market.code} volume samples "
            f"(run `python -m scripts.backtest --market EG --fetch-only` for the EG reference)")
    thin_ref, low_ref = eg["EGX30"]
    p_thin, p_low = float((ref < thin_ref).mean()), float((ref < low_ref).mean())
    thin = int(round_down_nice(float(np.quantile(tgt, max(p_thin, 1e-6)))))
    low = int(round_down_nice(float(np.quantile(tgt, p_low))))
    if low <= thin:
        raise CalibrationRefused(f"liquidity tiers collapsed: thin {thin} >= low {low}")
    tiers = {index_member_tier(market): (0, 0),
             market.code: other_listed_tier(thin, low, eg)}
    report = {"p_thin": round(p_thin, 3), "p_low": round(p_low, 3),
              "member_share_floors": (thin, low),
              "thin_share": round(float((tgt < thin).mean()), 3),
              "n_ref": int(len(ref)), "n_target": int(len(tgt))}
    return tiers, report


_REGIME_LEAD = {
    "weak": "Most of the market is in poor technical condition.",
    "mixed": "The market is in middling condition.",
    "broad": "Most of the market is in good technical condition.",
}
_REGIME_LABEL = {"weak": "Broadly weak", "mixed": "Mixed", "broad": "Broadly healthy"}


def regime_cal_from(reg: dict, spec: CacheSpec, fitted_at: str) -> RegimeCal:
    rho, t = float(reg["rho_overlapping"]), float(reg["newey_west_t"])
    bands = {
        k: RegimeBand(
            label=_REGIME_LABEL[k],
            summary=(f"{_REGIME_LEAD[k]} Historically the next three months were "
                     f"positive about {round(v['positive_rate'] * 10)} times in ten."),
            median_3m_pct=float(v["median"]), positive_rate=float(v["positive_rate"]),
            observations=int(v["n"]))
        for k, v in reg["bands"].items()
    }
    return RegimeCal(
        fitted_at=fitted_at, weak_max=float(reg["cutoff_weak_max"]),
        mixed_max=float(reg["cutoff_mixed_max"]), rho=round(rho, 3), t=round(t, 2),
        n=int(reg["n_readings"]), significant=abs(t) >= 1.96,
        phase_rhos=tuple(round(float(p["rho"]), 3) for p in reg["phases"]),
        bands=bands, universe_indexes=tuple(spec.regime_indexes),
        universe_label=spec.regime_universe_label,
        failure_outcome=("Shown as context, not a forecast." if rho > 0 else
                         f"Hidden on this market: the {spec.regime_universe_label} "
                         f"association ran the wrong way."),
    )


def forecast_cal_from(fc_t: dict, fc_e: dict, market, fitted_at: str) -> ForecastCal:
    who = f"{market.short_name} stocks'"
    g = fc_t["cone_gaussian_90_coverage"]
    daily = fc_e["one_sigma_coverage"]["daily"]
    indices = " / ".join(i.label for i in market.indices)
    return ForecastCal(
        fitted_at=fitted_at,
        n_observations=int(fc_t["n_cone"]),
        universe_label=f"current {indices} members, {fc_t['first_year']}–{fc_t['last_year']}",
        sigma_window=int(fc_t["sigma_window"]),
        fit_horizon_days=int(fc_t["cone_days"]),
        band_width_pct_of_spot=dict(fc_t["band_width_pct_of_spot"]),
        z_cone={int(k): float(v) for k, v in fc_t["z_cone"].items()},
        one_sigma_coverage={"ewma": dict(fc_e["one_sigma_coverage"]),
                            "trailing": dict(fc_t["one_sigma_coverage"])},
        gaussian_cone_90_coverage=float(g),
        cone_90_coverage=float(fc_t["cone_90_coverage"]),
        daily_band_note=(f"Gaussian theory says 68%. {who} centre is "
                         f"{'thinner' if daily > 68 else 'fatter'}."),
        cone_note=(f"Fitted to {who} own distribution. The textbook Gaussian band "
                   f"delivered {'only ' if g < 90 else ''}{g}%."),
    )


def risk_cal_from(risk: dict, fitted_at: str) -> RiskCal:
    if "error" in risk:
        raise CalibrationRefused(f"risk: {risk['error']}")
    keys = ("vol_predicts_vol", "vol_predicts_vol_nonoverlapping",
            "vol_predicts_drawdown", "vol_predicts_drawdown_nonoverlapping")
    for key in keys:
        if risk[key]["ic"] is None:
            raise CalibrationRefused(f"risk: {key} has too few dates for an IC")
    q = {int(r["quintile"]): {"future_vol_ann_pct": r["future_vol_ann_pct"],
                              "median_max_drawdown_pct": r["median_max_drawdown_pct"],
                              "p90_max_drawdown_pct": r["p90_max_drawdown_pct"]}
         for r in risk["quintiles"]}
    if sorted(q) != [1, 2, 3, 4, 5]:
        raise CalibrationRefused(f"risk: quintiles {sorted(q)}")
    v, vn, d, dn = (risk[k] for k in keys)
    return RiskCal(
        fitted_at=fitted_at, n_observations=int(risk["n_observations"]),
        n_symbols=int(risk["n_symbols"]), n_dates=int(risk["n_dates"]),
        lookback_days=int(risk["lookback_days"]), forward_days=int(risk["forward_days"]),
        vol_predicts_vol_ic=v["ic"], vol_predicts_vol_t=v["t"],
        vol_predicts_vol_ic_nonoverlapping=vn["ic"], vol_predicts_vol_t_nonoverlapping=vn["t"],
        vol_predicts_drawdown_ic=d["ic"], vol_predicts_drawdown_t=d["t"],
        vol_predicts_drawdown_ic_nonoverlapping=dn["ic"],
        vol_predicts_drawdown_t_nonoverlapping=dn["t"],
        quintiles=q,
    )


def composite_failures(spec: CacheSpec) -> tuple:
    """
    The composite-score test, run on THIS market's panel, for the Calibration
    screen's "what failed" list. Listed only when it failed the project's
    |t| > 3.0 bar — a market where the score did rank returns would need a
    different sentence, not this one.
    """
    from scripts.backtest import information_coefficient

    panel = pd.read_pickle(os.path.join(spec.cache_dir, "panel.pkl"))
    if "fwd_21" not in panel.columns:
        return ()
    ic = information_coefficient(panel, "score", 21)
    if ic.get("ic") is None or ic.get("t_stat") is None or abs(ic["t_stat"]) >= 3.0:
        return ()
    return ({
        "claim": "The composite score ranks stocks by future return",
        "measured": (f"IC ≈ 0 ({ic['ic']:+.3f} at 21 days, t {ic['t_stat']:.2f} "
                     f"over {ic['n_dates']} dates)"),
        "outcome": "Buy/Sell labels stay removed; the score describes condition.",
    },)


def build_bundle(market, spec: CacheSpec, *, eg_spec: CacheSpec, eg_pe, eg_dy, pe, dy,
                 today: date) -> tuple:
    fitted_at = today.isoformat()
    frames = load_frames(spec, market)
    if not frames:
        raise CalibrationRefused(f"no cached frames in {spec.cache_dir}")
    members = sorted({r["symbol"].upper() for r in read_universe_rows(spec) if r.get("indices")})
    floor = fit_turnover_floor({s: frames[s] for s in members if s in frames})
    price_floor = PRICE_FLOOR[market.code]
    tiers, tier_report = fit_liquidity_tiers(load_frames(eg_spec, EG), frames, members, market)
    fund = fit_fundamentals(eg_pe, eg_dy, pe, dy, market)
    fc_t = forecast_calibration(spec, market, "trailing")
    fc_e = forecast_calibration(spec, market, "ewma")
    reg = regime_calibration(spec)
    risk = risk_calibration(spec, market, floor, price_floor)
    bundle = Calibration(
        market=market.code,
        forecast=forecast_cal_from(fc_t, fc_e, market, fitted_at),
        regime=regime_cal_from(reg, spec, fitted_at),
        risk=risk_cal_from(risk, fitted_at),
        fundamentals=fund,
        liquidity_tiers=tiers,
        liquidity_default_tier=market.code,
        liquid_min_turnover=floor,
        liquid_min_price=price_floor,
        measured_failures=composite_failures(spec),
        breadth_measured=False,
        dividend_drag_measured=False,
        provisional=False,
    )
    return bundle, {"regime": reg, "forecast": fc_t, "forecast_ewma": fc_e,
                    "risk": risk, "liquidity": tier_report}


def _py(value, indent: int = 0) -> str:
    """A Python literal for a bundle value; dataclasses become constructor calls."""
    pad, end = "    " * (indent + 1), "    " * indent
    if dataclasses.is_dataclass(value):
        parts = [f"{pad}{f.name}={_py(getattr(value, f.name), indent + 1)},"
                 for f in dataclasses.fields(value)]
        return f"{type(value).__name__}(\n" + "\n".join(parts) + f"\n{end})"
    if isinstance(value, dict):
        if not value:
            return "{}"
        parts = [f"{pad}{_py(k, indent + 1)}: {_py(v, indent + 1)}," for k, v in value.items()]
        return "{\n" + "\n".join(parts) + f"\n{end}}}"
    if isinstance(value, tuple):
        if not value:
            return "()"
        return "(" + ", ".join(_py(v, indent + 1) for v in value) + ",)"
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "math.inf" if value > 0 else "-math.inf"
        return repr(value)
    if isinstance(value, (int, str)):
        return repr(value)
    raise TypeError(f"cannot render {type(value).__name__}")


_TEMPLATE = '''"""
{code} calibration bundle — GENERATED by `python -m scripts.calibrate --market {code} --write`
on {fitted_at}. Do not edit by hand; re-run the script.

Survivorship: the universe is today's index members ({label}), so names
that left the indices are missing from the history. The Calibration screen
shows this label.
"""
import math

from app.core.calibration.types import (
    Calibration, ForecastCal, FundamentalsCal, RegimeBand, RegimeCal, RiskCal,
)

BUNDLE = {body}
'''


def render_module(bundle: Calibration) -> str:
    return _TEMPLATE.format(code=bundle.market, fitted_at=bundle.forecast.fitted_at,
                            label=bundle.forecast.universe_label, body=_py(bundle))


def bundle_path(code: str) -> str:
    return os.path.join(REPO_ROOT, "app", "core", "calibration", f"{code.lower()}.py")


def report_eg(spec: CacheSpec, sections: str) -> int:
    """The EG report as before. eg.py is frozen, so compare by eye."""
    want = {"regime", "forecast", "risk"} if sections == "all" else set(sections.split(","))
    out = {}
    if "regime" in want or "risk" in want:
        if not os.path.exists(os.path.join(spec.cache_dir, "panel.pkl")):
            print(f"no panel at {spec.cache_dir} -- run `python -m scripts.backtest` first")
            return 1
    if "regime" in want:
        out["regime"] = regime_calibration(spec)
    if "forecast" in want:
        out["forecast"] = forecast_calibration(spec, EG, "trailing")
        out["forecast_ewma"] = forecast_calibration(spec, EG, "ewma")
    if "risk" in want:
        out["risk"] = risk_calibration(spec, EG, EG.liquid_min_turnover, EG.liquid_min_price)

    if "regime" in out:
        reg = out["regime"]
        print("=" * 74)
        print("REGIME  (universe: %s, horizon %dd, n=%d)"
              % (reg["universe"], reg["horizon_days"], reg["n_readings"]))
        print("=" * 74)
        print(f"  overlapping rho = {reg['rho_overlapping']:+.4f}   "
              f"naive t = {reg['naive_t']:+.2f}   NEWEY-WEST t = {reg['newey_west_t']:+.2f}")
        print("  de-overlapped phases (ALL of them -- never report just one):")
        for p in reg["phases"]:
            print(f"    offset {p['offset']}: rho={p['rho']:+.4f}  t={p['t']:+.2f}  n={p['n']}")
        print(f"  cutoffs: weak < {reg['cutoff_weak_max']} <= mixed < "
              f"{reg['cutoff_mixed_max']} <= broad")
        for k, v in reg["bands"].items():
            print(f"    {k:<6} n={v['n']:>3}  median={v['median']:+6.2f}%  "
                  f"mean={v['mean']:+6.2f}%  positive={v['positive_rate']:.3f}")

    if "forecast" in out:
        fc = out["forecast"]
        print()
        print("=" * 74)
        print(f"FORECAST BANDS  (n_daily={fc['n_daily']:,}, n_cone={fc['n_cone']:,}, "
              f"sigma window={fc['sigma_window']})")
        print("=" * 74)
        print("  +/-1 sigma band ACTUAL coverage (advertised as 68% since launch):")
        for name, pct in fc["one_sigma_coverage"].items():
            print(f"      {name:<8} {pct:>5}%")
        print(f"  Gaussian 90% cone at {fc['cone_days']}d covers "
              f"{fc['cone_gaussian_90_coverage']}%  (claimed 90%)")
        print(f"  {'coverage':>10}{'|z| daily':>12}{'|z| cone':>11}{'gaussian':>11}")
        for pct, g in ((50, 0.6745), (80, 1.2816), (90, 1.6449), (95, 1.9600)):
            print(f"  {pct:>9}%{fc['z_daily'][pct]:>12.3f}"
                  f"{fc['z_cone'][pct]:>11.3f}{g:>11.4f}")

    if "risk" in out:
        risk = out["risk"]
        print()
        print("=" * 74)
        print("RISK GRADE  (liquid universe: >%s EGP/day turnover, traded >=%d%% of "
              "the last 60 sessions)" % (f"{EG.liquid_min_turnover:,}",
                                         int(LIQUID_MIN_TRADED_SHARE * 100)))
        print("=" * 74)
        if "error" in risk:
            print("  " + risk["error"])
        else:
            print(f"  {risk['n_observations']:,} observations | {risk['n_symbols']} symbols "
                  f"| {risk['n_dates']} dates | median {risk['median_names_per_date']} names/date")
            print(f"  past {risk['lookback_days']}d vol -> next {risk['forward_days']}d:")
            for label, key in (("realized volatility", "vol_predicts_vol"),
                               ("max drawdown", "vol_predicts_drawdown")):
                a = risk[key]
                b = risk[key + "_nonoverlapping"]
                print(f"    {label:<22} IC={a['ic']:+.4f} t={a['t']:+6.1f} "
                      f"({a['n_dates']} dates)   non-overlapping IC={b['ic']:+.4f} "
                      f"t={b['t']:+5.1f} ({b['n_dates']})")
            print(f"\n  {'quintile':>9}{'n':>8}{'future vol':>13}{'median maxDD':>15}"
                  f"{'p90 maxDD':>12}")
            for q in risk["quintiles"]:
                print(f"  {q['quintile']:>9}{q['n']:>8,}"
                      f"{q['future_vol_ann_pct']:>12.1f}%"
                      f"{q['median_max_drawdown_pct']:>14.1f}%"
                      f"{q['p90_max_drawdown_pct']:>11.1f}%")

    print("\n" + "=" * 74)
    print("EG REPORT (eg.py is frozen; compare by eye)")
    print("=" * 74)
    print(json.dumps(out, indent=2, default=str))
    return 0


def print_report(bundle: Calibration, report: dict) -> None:
    f, r, k, q = bundle.forecast, bundle.regime, bundle.risk, bundle.fundamentals
    print("=" * 74)
    print(f"{bundle.market} BUNDLE  fitted {f.fitted_at}  universe: {f.universe_label}")
    print("=" * 74)
    print(f"  liquid_min_turnover = {bundle.liquid_min_turnover:,.0f}   "
          f"liquid_min_price = {bundle.liquid_min_price}")
    print(f"  liquidity tiers = {dict(bundle.liquidity_tiers)}   report = {report['liquidity']}")
    print(f"  P/E bands = {q.pe_bands}  median {q.pe_median}")
    print(f"  DY bands  = {q.dy_bands}  median {q.dy_median}  under cash: {q.dy_mid_under_risk_free}")
    print(f"  z_cone = {dict(f.z_cone)}  gaussian 90% delivered {f.gaussian_cone_90_coverage}%  "
          f"fitted 90% delivered {f.cone_90_coverage}%")
    print(f"  ±1σ coverage = {f.one_sigma_coverage}")
    print(f"  band width % of spot = {dict(f.band_width_pct_of_spot)}")
    print(f"  regime: weak<{r.weak_max}<=mixed<{r.mixed_max}<=broad  rho {r.rho}  NW t {r.t}  "
          f"n {r.n}  phases {r.phase_rhos}  shown {r.show}")
    print(f"  risk: vol IC {k.vol_predicts_vol_ic} (t_no {k.vol_predicts_vol_t_nonoverlapping})  "
          f"DD IC {k.vol_predicts_drawdown_ic} (t_no {k.vol_predicts_drawdown_t_nonoverlapping})  "
          f"{k.n_observations} obs / {k.n_symbols} symbols / {k.n_dates} dates")
    print(json.dumps(report, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="EG", choices=sorted(MARKETS))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--eg-cache-dir", default=None, help="EG cache (liquidity-tier reference)")
    ap.add_argument("--sections", default="all", help="EG report only: regime,forecast,risk")
    ap.add_argument("--write", action="store_true", help="write app/core/calibration/<market>.py")
    args = ap.parse_args(argv)
    market = MARKETS[args.market]
    spec = spec_for(market.code, args.cache_dir)
    if market.code == EG.code:
        if args.write:
            print("refusing to --write EG: eg.py is the frozen EG record (refitting EG is out of scope)")
            return 2
        return report_eg(spec, args.sections)

    from app.core.db import get_db
    db = get_db()
    eg_pe, eg_dy = read_pe_rows(db, EG.code)
    pe, dy = read_pe_rows(db, market.code)
    try:
        bundle, report = build_bundle(market, spec, eg_spec=spec_for(EG.code, args.eg_cache_dir),
                                      eg_pe=eg_pe, eg_dy=eg_dy, pe=pe, dy=dy, today=date.today())
    except CalibrationRefused as e:
        print(f"REFUSED: {e}")
        return 1
    print_report(bundle, report)
    if args.write:
        path = bundle_path(market.code)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_module(bundle))
        print(f"wrote {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
