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
whenever the cache is refreshed and paste the block it prints into
`app/core/constants.py`.

THE CORRECT STATISTIC FOR OVERLAPPING WINDOWS
---------------------------------------------
Not "throw away two thirds of the data and hope you picked a good phase".
Keep every observation and correct the standard error for the autocorrelation
the overlap induces -- Newey-West with lag = the overlap depth. That is
reported here as `nw_t`, alongside all three de-overlapped phases so a future
reader can see the spread rather than one lucky draw.

Run:  ./.venv/Scripts/python.exe -m scripts.calibrate
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
BENCHMARK = "EGX30"

# The horizon regime.py reports, and the rebalance spacing of the panel. Their
# ratio is how many disjoint phases the de-overlapped grid has.
REGIME_HORIZON = 63
REBALANCE = 21
PHASES = REGIME_HORIZON // REBALANCE

# Matches INTERNAL_BARS_MIN -- the window every scoring path actually fetches,
# so the fitted band is the band the app can really compute at request time.
SIGMA_WINDOW = 400
CONE_DAYS = 60

# Instruments whose symbol is an ISIN rather than a ticker (EGS30AJ1C016,
# EGS48271C018-EGP, ...). They are not common stocks and must not enter a
# market-wide statistic.
#
# Match on LENGTH, not on the "EGS" prefix alone: an Egyptian ISIN is 12
# characters (EG + 10), sometimes with a "-EGP" suffix, while **EGSA is a real
# four-character EGX ticker**. A bare startswith("EGS") silently drops it.
def _is_instrument(symbol: str) -> bool:
    s = symbol.upper()
    return len(s) >= 12 and s.startswith("EG")


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

def load_benchmark() -> pd.Series:
    with open(os.path.join(CACHE_DIR, f"{BENCHMARK}.pkl"), "rb") as f:
        df = pickle.load(f)
    df.columns = [str(c).lower() for c in df.columns]
    return df["close"].astype(float)


def regime_universe() -> set:
    """EGX30 + EGX70, the universe /api/market_regime actually averages over."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "data", "egx_tickers.json")
    with open(path, encoding="utf-8") as f:
        return {t["symbol"].upper() for t in json.load(f)
                if (t.get("index") or "").upper() in ("EGX30", "EGX70")}


def iter_symbol_frames():
    """Every cached OHLCV frame that is a common stock with usable history."""
    for path in glob.glob(os.path.join(CACHE_DIR, "*.pkl")):
        name = os.path.basename(path)[:-4]
        if name in ("panel", "breadth") or _is_instrument(name):
            continue
        with open(path, "rb") as f:
            df = pickle.load(f)
        if df is None or getattr(df, "empty", True):
            continue
        df.columns = [str(c).lower() for c in df.columns]
        if "close" not in df.columns:
            continue
        yield name, df


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------

def regime_calibration() -> dict:
    """
    Tercile cutoffs, per-band forward record, and the association behind them.

    Averages the panel over the EGX30+EGX70 universe specifically, because that
    is what the live endpoint averages. Calibrating on the full panel gives
    materially different cutoffs (43.7 / 49.8 rather than 45.1 / 51.5), and a
    band boundary that does not match the universe it is applied to is a silent
    off-by-one on every reading.
    """
    panel = pd.read_pickle(os.path.join(CACHE_DIR, "panel.pkl"))
    bench = load_benchmark()
    panel = panel[panel["symbol"].isin(regime_universe())]

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
            "universe": "EGX30+EGX70", "horizon_days": REGIME_HORIZON}


# ---------------------------------------------------------------------------
# Forecast bands
# ---------------------------------------------------------------------------

def forecast_calibration() -> dict:
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
    # The three horizons expected_move advertises, plus the cone's own.
    HORIZONS = {"daily": 1, "weekly": 5, "monthly": 22}
    buckets = {k: [] for k in HORIZONS}
    cone = []

    for _, df in iter_symbol_frames():
        close = df["close"].astype(float)
        if len(close) < SIGMA_WINDOW + CONE_DAYS + REBALANCE:
            continue
        rets = close.pct_change()
        for i in range(SIGMA_WINDOW, len(close) - CONE_DAYS - 1, REBALANCE):
            sigma = float(rets.iloc[i - SIGMA_WINDOW:i].std())
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            p0 = float(close.iloc[i - 1])
            if p0 <= 0:
                continue
            # expected_move scales one sigma by sqrt(h); test the band it
            # actually draws, at each horizon it actually labels.
            for name, h in HORIZONS.items():
                pn = float(close.iloc[i - 1 + h])
                if np.isfinite(pn):
                    buckets[name].append((pn / p0 - 1.0) / (sigma * np.sqrt(h)))
            pn = float(close.iloc[i - 1 + CONE_DAYS])
            if np.isfinite(pn):
                cone.append((pn / p0 - 1.0) / (sigma * np.sqrt(CONE_DAYS)))

    def _abs(x):
        a = np.abs(np.array(x, dtype=float))
        return a[np.isfinite(a)]

    c = _abs(cone)
    out = {"n_cone": int(c.size), "cone_days": CONE_DAYS,
           "sigma_window": SIGMA_WINDOW,
           "cone_gaussian_90_coverage": round(float((c <= 1.6449).mean() * 100), 1),
           "one_sigma_coverage": {}, "z_daily": {}, "z_cone": {}}

    for name in HORIZONS:
        b = _abs(buckets[name])
        # What the +/-1-sigma band this app draws ACTUALLY covers at this
        # horizon. Advertised as 68% since launch; it is not 68%.
        out["one_sigma_coverage"][name] = round(float((b <= 1.0).mean() * 100), 1)
        if name == "daily":
            out["n_daily"] = int(b.size)
            for pct in (50, 80, 90, 95):
                out["z_daily"][pct] = round(float(np.percentile(b, pct)), 3)

    for pct in (50, 80, 90, 95):
        out["z_cone"][pct] = round(float(np.percentile(c, pct)), 3)
    return out


# ---------------------------------------------------------------------------
# Risk grade
# ---------------------------------------------------------------------------

# The tradeable cut. A result that only holds among names nobody can enter or
# exit is not a result a retail investor could have acted on.
LIQUID_MIN_TURNOVER_EGP = 1_000_000
LIQUID_MIN_TRADED_SHARE = 0.80     # of the last 60 sessions the symbol has rows for
RISK_FORWARD_DAYS = 126
RISK_LOOKBACK_DAYS = 63


def risk_calibration() -> dict:
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
    panel = pd.read_pickle(os.path.join(CACHE_DIR, "panel.pkl"))
    dates = sorted(panel["date"].unique())

    frames = {}
    for name, df in iter_symbol_frames():
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
            if price <= 0.5:
                continue

            window = slice(i - 59, i + 1)
            turnover = float((close * volume).iloc[window].mean())
            if not np.isfinite(turnover) or turnover < LIQUID_MIN_TURNOVER_EGP:
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

def main() -> int:
    if not os.path.exists(os.path.join(CACHE_DIR, "panel.pkl")):
        print(f"no panel at {CACHE_DIR} -- run `python -m scripts.backtest` first")
        return 1

    reg = regime_calibration()
    fc = forecast_calibration()

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

    risk = risk_calibration()
    print()
    print("=" * 74)
    print("RISK GRADE  (liquid universe: >%s EGP/day turnover, traded >=%d%% of "
          "the last 60 sessions)" % (f"{LIQUID_MIN_TURNOVER_EGP:,}",
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
    print("PASTE INTO app/core/constants.py")
    print("=" * 74)
    print(json.dumps({"regime": reg, "forecast": fc, "risk": risk},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
