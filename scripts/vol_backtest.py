"""
The gate every volatility change must pass before it ships.

WHY THIS EXISTS
---------------
This project has twice shipped a number that sounded right and was not: a
market-condition correlation that was one lucky phase of three, and forecast
bands advertising 68% / 90% coverage while delivering 79% / 86%. Both survived
because nothing measured them. Volatility is the one thing on the EGX that is
genuinely forecastable, which makes it exactly the place where an unmeasured
"improvement" would do the most quiet damage.

So: no change to how this app estimates volatility lands without a run of this
script showing it wins.

WHAT IT MEASURES
----------------
One-step-ahead conditional VARIANCE forecasts, walk-forward, per symbol, using
only information available before the bar being predicted.

  QLIKE   log(h) + r^2/h, averaged. The standard loss for variance forecasts.
          It is robust to the fact that true variance is unobservable: using a
          noisy but UNBIASED proxy (the squared return) leaves the ranking of
          forecasts unchanged, which is not true of most other losses. Lower is
          better, and the level is not interpretable on its own -- only
          differences between methods are.

  MSE     mean (h - r^2)^2. Reported alongside because it is familiar, but it is
          dominated by a handful of huge days and ranks forecasts less reliably
          than QLIKE on fat-tailed data. Where the two disagree, believe QLIKE.

  DM      Diebold-Mariano on the QLIKE loss differential, with a Newey-West
          standard error. This is the part that turns "0.71 vs 0.83" into a
          claim: the losses are serially correlated, so a naive t on the
          difference overstates significance exactly the way the market-regime
          statistic once did.

  Coverage  What a +/-1-sigma band built on each forecast actually contains, and
            the |z| quantile each would need for a stated coverage level. This
            is what tells `core/forecast.py` whether its fitted z-table still
            applies after a change of sigma -- swapping the estimator without
            refitting would leave the table describing a different variable.

  Kupiec    Unconditional-coverage likelihood-ratio test on the 95% band. Chi-2
            with one degree of freedom; the 5% critical value is 3.841. Answers
            "is the miss rate distinguishable from the promised one".

Run:  ./.venv/Scripts/python.exe -m scripts.vol_backtest
      ./.venv/Scripts/python.exe -m scripts.vol_backtest --symbols 60
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.risk_grade import (  # noqa: E402
    EWMA_SEED_BARS,
    ewma_variance_series,
)
from scripts.calibrate import iter_symbol_frames  # noqa: E402

# Enough history that a 400-bar trailing window has something to say before the
# first scored bar.
MIN_BARS = 600

# Chi-squared(1) critical value at 5%. Hardcoded because this project keeps its
# runtime free of scipy, and one constant is cheaper than a dependency.
CHI2_1DF_95 = 3.841

# The project's pre-registered evidence bar, and deliberately stricter than the
# conventional 1.96. Roughly thirty candidate signals have now been tested
# against this one dataset, so a 5% threshold would be expected to hand back a
# false winner or two on noise alone. Anything that cannot clear this does not
# change what the app computes — the point of a gate is that it sometimes says
# no.
EVIDENCE_BAR = 3.0


# ---------------------------------------------------------------------------
# Forecasters. Each returns a one-step-ahead VARIANCE path aligned to `rets`,
# with NaN wherever the method cannot speak yet. None may use r_t to predict
# period t -- that is the whole discipline being tested.
# ---------------------------------------------------------------------------

def apply_floor(h: np.ndarray, rets: pd.Series) -> tuple:
    """
    Clamp a variance path at the same floor the shipped EWMA uses, and report
    how often the clamp binds.

    THIS IS NOT COSMETIC, and the first run of this harness proved it. Without a
    floor, trailing-window variance on EGX collapses toward zero whenever a
    window happens to contain a run of flat sessions -- and 19% of this market's
    daily returns are EXACTLY zero. The QLIKE term r^2/h then explodes, and the
    pooled mean came back as 1.1e13 for sd63, which measures the absence of a
    guard rather than the quality of the estimator.

    Every method is floored identically here, because that is the only form any
    of them could actually be deployed in. `floor_binds` is reported separately,
    since how often an estimator needs rescuing is itself a finding.
    """
    from app.core.risk_grade import EWMA_VARIANCE_FLOOR_SHARE

    values = rets.to_numpy(dtype=float)
    floor = EWMA_VARIANCE_FLOOR_SHARE * float(np.nanvar(values, ddof=1))
    if not np.isfinite(floor) or floor <= 0:
        return h, np.zeros(len(h), dtype=bool)
    binds = np.isfinite(h) & (h < floor)
    return np.where(binds, floor, h), binds


def trailing_variance(rets: pd.Series, window: int) -> np.ndarray:
    """Rolling sample variance, SHIFTED so period t is predicted from t-1 back."""
    return rets.rolling(window).var(ddof=1).shift(1).to_numpy(dtype=float)


def ewma_variance(rets: pd.Series, lam: float) -> np.ndarray:
    """The shipped recursion, scored as-is rather than re-implemented here."""
    series = ewma_variance_series(rets, lam)
    if series is None:
        return np.full(len(rets), np.nan)
    out = np.asarray(series, dtype=float).copy()
    out[:EWMA_SEED_BARS] = np.nan     # seeded, not forecast
    return out


METHODS = {
    "sd20": lambda r: trailing_variance(r, 20),
    "sd63": lambda r: trailing_variance(r, 63),
    "sd400": lambda r: trailing_variance(r, 400),      # the incumbent
    "ewma94": lambda r: ewma_variance(r, 0.94),        # RiskMetrics
    "ewma97": lambda r: ewma_variance(r, 0.97),
}

INCUMBENT = "sd400"
CHALLENGER = "ewma94"


# ---------------------------------------------------------------------------
# Losses and tests
# ---------------------------------------------------------------------------

def qlike(h: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """log(h) + r^2/h. NaN where h is not a usable positive variance."""
    out = np.full(h.shape, np.nan)
    ok = np.isfinite(h) & (h > 0) & np.isfinite(r2)
    out[ok] = np.log(h[ok]) + r2[ok] / h[ok]
    return out


def newey_west_se(d: np.ndarray, lag: int) -> float:
    """HAC standard error of a mean, Bartlett kernel."""
    n = len(d)
    dm = d - d.mean()
    gamma0 = float(dm @ dm) / n
    var = gamma0
    for L in range(1, lag + 1):
        cov = float(dm[L:] @ dm[:-L]) / n
        var += 2.0 * (1.0 - L / (lag + 1)) * cov
    if var <= 0:
        return float("nan")
    return math.sqrt(var / n)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 10) -> dict:
    """
    Is method A's loss lower than B's by more than sampling noise?

    Negative t means A wins. The Newey-West correction matters here: daily loss
    differentials are strongly serially correlated during volatile stretches, so
    a naive t-statistic would inflate the verdict.
    """
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    if len(d) < 100:
        return {"t": None, "n": len(d), "mean_diff": None}
    se = newey_west_se(d, lag)
    return {
        "t": round(float(d.mean() / se), 2) if se == se and se > 0 else None,
        "n": int(len(d)),
        "mean_diff": round(float(d.mean()), 6),
    }


def kupiec(exceptions: int, n: int, p: float) -> dict:
    """
    Unconditional-coverage LR test: does the observed miss rate match the
    promised one? LR ~ chi2(1); above 3.841 rejects at 5%.
    """
    if n == 0 or exceptions == 0 or exceptions == n:
        return {"rate": None, "lr": None, "rejects": None}
    rate = exceptions / n
    lr = -2.0 * (
        (n - exceptions) * math.log(1 - p) + exceptions * math.log(p)
        - (n - exceptions) * math.log(1 - rate) - exceptions * math.log(rate)
    )
    return {"rate": round(rate, 4), "lr": round(lr, 2),
            "rejects": bool(lr > CHI2_1DF_95)}


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=0,
                    help="cap the universe (0 = all cached symbols)")
    ap.add_argument("--challenger", default=CHALLENGER,
                    choices=list(METHODS),
                    help="which method to test against the incumbent")
    args = ap.parse_args()
    challenger = args.challenger

    losses = {name: [] for name in METHODS}
    z_values = {name: [] for name in METHODS}
    per_symbol_wins = {name: 0 for name in METHODS}
    floor_binds = {name: [0, 0] for name in METHODS}
    n_symbols = 0

    for symbol, df in iter_symbol_frames():
        if args.symbols and n_symbols >= args.symbols:
            break
        close = df["close"].astype(float)
        if len(close) < MIN_BARS:
            continue
        rets = close.pct_change().dropna()
        if len(rets) < MIN_BARS:
            continue
        r = rets.to_numpy(dtype=float)
        r2 = r * r

        # Score every method on the SAME bars. Each produces a path the length
        # of `rets`, but they warm up at different points (sd400 needs 400 bars,
        # ewma 60), so their finite masks differ. Pooling them independently and
        # truncating to a common length later pairs element i of one method with
        # a DIFFERENT (symbol, bar) of another — which silently invalidates the
        # Diebold-Mariano test, since it is a test on a paired difference. This
        # harness caught exactly that in its own first draft: QLIKE ranked sd400
        # ahead while the mis-paired DM reported the opposite.
        raw = {}
        for name, fn in METHODS.items():
            h = fn(rets)
            if h is None or len(h) != len(r):
                continue
            h, binds = apply_floor(h, rets)
            floor_binds[name][0] += int(binds.sum())
            floor_binds[name][1] += int(np.isfinite(h).sum())
            raw[name] = h
        if len(raw) != len(METHODS):
            continue

        usable = np.ones(len(r), dtype=bool)
        for h in raw.values():
            usable &= np.isfinite(h) & (h > 0)
        usable &= np.isfinite(r2)
        if not usable.any():
            continue

        sym_loss = {}
        for name, h in raw.items():
            l = qlike(h[usable], r2[usable])
            losses[name].append(l)
            sym_loss[name] = float(np.nanmean(l))
            with np.errstate(invalid="ignore", divide="ignore"):
                z = r[usable] / np.sqrt(h[usable])
            z_values[name].append(z[np.isfinite(z)])

        finite = {k: v for k, v in sym_loss.items() if v == v}
        if finite:
            per_symbol_wins[min(finite, key=finite.get)] += 1
        n_symbols += 1

    if not n_symbols:
        print("no usable symbols — run `python -m scripts.backtest` to warm the cache")
        return 1

    pooled = {}
    for name in METHODS:
        if not losses[name]:
            continue
        allo = np.concatenate(losses[name])
        pooled[name] = allo

    print("=" * 78)
    print(f"VOLATILITY FORECAST BENCH  ({n_symbols} symbols, one-step-ahead, "
          f"walk-forward)")
    print("=" * 78)
    print(f"  {'method':<9}{'QLIKE':>10}{'median':>10}{'floored':>9}"
          f"{'best on':>10}{'+/-1sig':>9}{'z90':>7}{'z95':>7}")
    for name in METHODS:
        if name not in pooled:
            continue
        l = pooled[name]
        l = l[np.isfinite(l)]
        z = np.abs(np.concatenate(z_values[name]))
        cov1 = (z <= 1.0).mean() * 100
        bound, total = floor_binds[name]
        pct_floored = (bound / total * 100) if total else 0.0
        print(f"  {name:<9}{l.mean():>10.4f}{np.median(l):>10.4f}"
              f"{pct_floored:>8.1f}%{per_symbol_wins[name]:>7} sym"
              f"{cov1:>8.1f}%"
              f"{np.percentile(z, 90):>7.3f}{np.percentile(z, 95):>7.3f}")

    print("\n  QLIKE is only meaningful as a DIFFERENCE between methods; the")
    print("  level carries no interpretation. Lower is better. `floored` is how")
    print("  often the variance floor had to rescue the estimator — on this")
    print("  market that is a property of the estimator, not an edge case.")

    # The claim that decides whether the app changes.
    if challenger in pooled and INCUMBENT in pooled:
        # Aligned by construction now — every method was scored on the same
        # bars, so these two arrays are element-wise paired.
        assert len(pooled[challenger]) == len(pooled[INCUMBENT]), (
            "loss arrays are not paired; the DM test would be meaningless"
        )
        dm = diebold_mariano(pooled[challenger], pooled[INCUMBENT])
        better = pooled[challenger][np.isfinite(pooled[challenger])].mean()
        worse = pooled[INCUMBENT][np.isfinite(pooled[INCUMBENT])].mean()
        delta = (better - worse) / abs(worse) * 100
        print("\n" + "=" * 78)
        print(f"DIEBOLD-MARIANO  {challenger} vs {INCUMBENT} (the incumbent in "
              f"core/forecast.py)")
        print("=" * 78)
        print(f"  QLIKE {better:.4f} vs {worse:.4f}   ({delta:+.1f}%)")
        print(f"  Newey-West t = {dm['t']}   over {dm['n']:,} paired observations")
        if dm["t"] is not None:
            verdict = ("CHALLENGER WINS" if dm["t"] < -EVIDENCE_BAR else
                       "INCUMBENT WINS" if dm["t"] > EVIDENCE_BAR else
                       "NO SIGNIFICANT DIFFERENCE — DO NOT SWITCH")
            print(f"  -> {verdict}   (bar is |t| > {EVIDENCE_BAR})")

    # Coverage honesty: a band is a promise, and this is the audit of it.
    print("\n" + "=" * 78)
    print("BAND COVERAGE  (Kupiec unconditional-coverage test on the 95% band)")
    print("=" * 78)
    print(f"  {'method':<10}{'miss rate':>12}{'promised':>11}{'LR':>9}{'verdict':>26}")
    for name in METHODS:
        if name not in z_values or not z_values[name]:
            continue
        z = np.abs(np.concatenate(z_values[name]))
        # A Gaussian 95% band is +/-1.96 sigma; count how often price left it.
        exceptions = int((z > 1.96).sum())
        k = kupiec(exceptions, len(z), 0.05)
        if k["lr"] is None:
            continue
        verdict = ("MISCALIBRATED" if k["rejects"] else "consistent with 5%")
        print(f"  {name:<10}{k['rate'] * 100:>11.2f}%{5.00:>10.2f}%"
              f"{k['lr']:>9.1f}{verdict:>26}")
    print("\n  A rejected test does NOT mean the estimator is bad — it means a")
    print("  GAUSSIAN band around it is. That is the finding core/forecast.py")
    print("  already encodes with its fitted EGX z-table, and the z90/z95")
    print("  columns above are what a refit would use.")

    multi_horizon(args)
    return 0


def multi_horizon(args) -> None:
    """
    The same contest at the horizons the app actually draws bands for.

    This exists because winning at one day ahead does NOT license a change to
    the 60-day cone. EWMA is an IGARCH: its multi-step forecast is FLAT at
    today's level, with no mean reversion to a long-run variance. After a
    violent week it projects that violence across the whole horizon, which a
    long trailing window — anchored much closer to the unconditional level —
    does not do. Whether that hurts is an empirical question, so it gets
    measured rather than argued.

    Forecast variance over H days is h_t * H (variance adds under iid), scored
    against the realized sum of squared returns over those same H days.
    """
    horizons = (5, 22, 60)
    acc = {h: {name: [] for name in METHODS} for h in horizons}
    n_symbols = 0

    for symbol, df in iter_symbol_frames():
        if args.symbols and n_symbols >= args.symbols:
            break
        close = df["close"].astype(float)
        if len(close) < MIN_BARS:
            continue
        rets = close.pct_change().dropna()
        if len(rets) < MIN_BARS:
            continue
        r = rets.to_numpy(dtype=float)
        r2 = r * r
        n_symbols += 1

        raw = {}
        for name, fn in METHODS.items():
            h = fn(rets)
            if h is None or len(h) != len(r):
                continue
            raw[name], _ = apply_floor(h, rets)
        if len(raw) != len(METHODS):
            continue

        # Realized variance over the NEXT H bars, aligned to the forecast made
        # at the start of that stretch.
        csum = np.concatenate([[0.0], np.cumsum(r2)])
        for H in horizons:
            realized = np.full(len(r), np.nan)
            valid = len(r) - H
            if valid <= 0:
                continue
            realized[:valid] = csum[H:H + valid] - csum[:valid]

            usable = np.isfinite(realized)
            for h in raw.values():
                usable &= np.isfinite(h) & (h > 0)
            if not usable.any():
                continue
            for name, h in raw.items():
                acc[H][name].append(qlike(h[usable] * H, realized[usable]))

    print("\n" + "=" * 78)
    print("MULTI-HORIZON  (does the one-day winner still win over a whole band?)")
    print("=" * 78)
    print(f"  {'horizon':<9}" + "".join(f"{n:>11}" for n in METHODS) + "   winner")
    for H in horizons:
        row = {}
        for name in METHODS:
            if not acc[H][name]:
                continue
            v = np.concatenate(acc[H][name])
            v = v[np.isfinite(v)]
            if len(v):
                row[name] = v.mean()
        if not row:
            continue
        best = min(row, key=row.get)
        cells = "".join(f"{row.get(n, float('nan')):>11.4f}" for n in METHODS)
        print(f"  {str(H) + 'd':<9}{cells}   {best}")

    print("\n  Read this before changing what sigma any BAND is built on. A")
    print("  one-day-ahead win is a claim about tomorrow, not about the 60-day")
    print("  outcome band, and core/forecast.py's fitted z-table is tied to the")
    print("  estimator it was fitted on — swapping one without refitting the")
    print("  other leaves the table describing a different variable.")


if __name__ == "__main__":
    raise SystemExit(main())
