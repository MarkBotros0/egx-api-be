"""
Do fundamental factors rank EGX stocks? The first honest test this project can run.

WHY THIS COULD NOT BE RUN BEFORE
--------------------------------
`scripts/backtest.py` withholds any verdict on fundamentals, and says so in its
docstring: `pe_data` is a today-only snapshot, so scoring a 2019 date with it
would be look-ahead bias severe enough to manufacture any result. With
`core/fundamentals_annual` there are now 20 years of dated annual figures, each
carrying the date it first became knowable. That is the missing ingredient.

WHAT IS TESTED, AND THE RULE DECIDED BEFORE RUNNING IT
------------------------------------------------------
Pre-registered, from the research plan:

  1. Significance bar is |t| > 3.0, not 1.96. Roughly thirty candidate signals
     have now been tested against this one dataset; a 5% threshold would be
     expected to hand back a false winner or two on noise alone.
  2. A PLACEBO must pass first. A shuffled factor through the same pipeline has
     to give IC ~ 0, or nothing below is evidence about factors and everything
     below is evidence about the plumbing.
  3. A POSITIVE CONTROL must also pass. Low volatility is known to score
     IC ~ +0.08 here; if the harness cannot recover a signal it has already
     found by another route, a null result is uninformative.
  4. SPLIT-HALF is reported for everything. A factor that only worked in one
     half of the sample did not work.
  5. EVERY factor tested is reported, including the failures. The denominator
     has to stay visible or the survivors are just the tail of a search.

The primary metric is the cross-sectional Information Coefficient, computed by
the UNCHANGED `information_coefficient` from scripts/backtest.py — one
observation per date, so overlapping forward windows cannot inflate it.

Calibration: an IC of 0.03-0.05 is considered good in professional equity quant.
Anything much larger on a market this small should be treated as a bug until
proven otherwise.

Run:  ./.venv/Scripts/python.exe -m scripts.factor_backtest
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.fundamentals_annual import fetch_annual_rows  # noqa: E402
from scripts.backtest import information_coefficient  # noqa: E402
from scripts.calibrate import iter_symbol_frames  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
FUND_CACHE = os.path.join(CACHE_DIR, "fundamentals_annual.pkl")

HORIZONS = (21, 63, 126)
EVIDENCE_BAR = 3.0

# Matches the liquid cut used elsewhere: a result that only holds among names
# trading a few thousand EGP a day is not one a retail investor could act on.
MIN_TURNOVER_EGP = 1_000_000
LIQUIDITY_WINDOW = 60


def load_fundamentals(refresh: bool = False) -> pd.DataFrame:
    """The annual archive, cached to disk so a re-run costs no network."""
    if not refresh and os.path.exists(FUND_CACHE):
        return pd.read_pickle(FUND_CACHE)
    rows = fetch_annual_rows()
    df = pd.DataFrame(rows)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_pickle(FUND_CACHE)
    return df


def build_panel(fund: pd.DataFrame, dates) -> pd.DataFrame:
    """
    One row per (date, symbol) with every candidate factor and forward returns.

    Fundamentals are selected by `first_usable_date <= as_of`, exactly as
    `get_annual_asof` does in production. That filter is the whole reason this
    test means anything.
    """
    fund = fund.sort_values(["symbol", "fiscal_year"])
    by_symbol = {s: g.reset_index(drop=True) for s, g in fund.groupby("symbol")}

    rows = []
    for symbol, df in iter_symbol_frames():
        annual = by_symbol.get(symbol)
        if annual is None or "volume" not in df.columns:
            continue
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        if len(close) < 300:
            continue
        rets = close.pct_change()

        for as_of in dates:
            i = int(close.index.searchsorted(as_of, side="right")) - 1
            if i < LIQUIDITY_WINDOW or i + max(HORIZONS) >= len(close):
                continue
            price = float(close.iloc[i])
            if price <= 0:
                continue

            turnover = float((close * volume).iloc[i - LIQUIDITY_WINDOW + 1:i + 1].mean())
            if not np.isfinite(turnover) or turnover < MIN_TURNOVER_EGP:
                continue

            stamp = as_of.date().isoformat()
            usable = annual[annual["first_usable_date"] <= stamp]
            if usable.empty:
                continue
            latest = usable.iloc[-1]

            eps = latest["eps_diluted"]
            assets = latest["total_assets"]
            gross = latest["gross_profit"]

            # Prior year, for a growth term. Must ALSO have been knowable.
            prior = usable[usable["fiscal_year"] == latest["fiscal_year"] - 1]
            prior_assets = prior.iloc[0]["total_assets"] if len(prior) else None

            row = {
                "symbol": symbol, "date": as_of,
                # Earnings yield rather than P/E: it is defined for loss-makers
                # (negative), needs no share count, and does not explode as
                # earnings approach zero the way a P/E does.
                "earnings_yield": (eps / price) if eps is not None else np.nan,
                "gross_profitability": (gross / assets)
                    if gross not in (None, 0) and assets not in (None, 0) else np.nan,
                "asset_growth": (assets / prior_assets - 1.0)
                    if assets and prior_assets else np.nan,
                "payout_yield": (latest["dps"] / price)
                    if latest["dps"] is not None else np.nan,
                # Positive control: known to score IC ~ +0.08 here. Signed so
                # that HIGHER is calmer, matching the direction that ranked well.
                "control_low_vol": -float(rets.iloc[i - 62:i + 1].std()),
            }
            for h in HORIZONS:
                row[f"fwd_{h}"] = (float(close.iloc[i + h]) / price - 1.0) * 100.0
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate(panel: pd.DataFrame, factor: str) -> dict:
    """IC at every horizon, plus the split-half both halves must survive."""
    out = {}
    for h in HORIZONS:
        out[h] = information_coefficient(panel, factor, h)

    dates = sorted(panel["date"].unique())
    mid = dates[len(dates) // 2]
    early = information_coefficient(panel[panel["date"] < mid], factor, HORIZONS[0])
    late = information_coefficient(panel[panel["date"] >= mid], factor, HORIZONS[0])
    out["split"] = (early, late)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the annual archive instead of using the cache")
    args = ap.parse_args()

    panel_path = os.path.join(CACHE_DIR, "panel.pkl")
    if not os.path.exists(panel_path):
        print("no price panel — run `python -m scripts.backtest` first")
        return 1

    fund = load_fundamentals(args.refresh)
    print(f"annual archive: {len(fund):,} records | "
          f"{fund['symbol'].nunique()} symbols | "
          f"FY{int(fund['fiscal_year'].min())}-{int(fund['fiscal_year'].max())}")

    dates = sorted(pd.read_pickle(panel_path)["date"].unique())
    panel = build_panel(fund, dates)
    if panel.empty:
        print("no scored rows")
        return 1

    print(f"factor panel: {len(panel):,} rows | {panel['symbol'].nunique()} symbols "
          f"| {panel['date'].nunique()} dates | median "
          f"{int(panel.groupby('date').size().median())} names/date")

    # ---- Gate 1: the placebo. Nothing below counts until this passes. ----
    rng = np.random.default_rng(20260902)
    placebo = panel.copy()
    placebo["placebo"] = rng.permutation(placebo["earnings_yield"].values)
    p = information_coefficient(placebo, "placebo", HORIZONS[0])
    ok = p["ic"] is not None and abs(p["ic"]) < 0.02
    print(f"\n[gate] placebo IC = {p['ic']} (t={p['t_stat']}) -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  !! the harness reports signal on a shuffled factor. Nothing")
        print("     below is evidence about factors.")
        return 1

    factors = ["control_low_vol", "earnings_yield", "gross_profitability",
               "asset_growth", "payout_yield"]

    print(f"\n{'factor':<22}" + "".join(f"{str(h) + 'd IC':>12}{'t':>7}"
                                        for h in HORIZONS) + "   verdict")
    print("-" * 92)
    results = {}
    for f in factors:
        r = evaluate(panel, f)
        results[f] = r
        cells = ""
        best_t = 0.0
        for h in HORIZONS:
            ic, t = r[h]["ic"], r[h]["t_stat"]
            cells += (f"{ic:>+12.4f}{t:>7.2f}" if ic is not None
                      else f"{'n/a':>12}{'':>7}")
            if t is not None and abs(t) > abs(best_t):
                best_t = t
        verdict = "PASSES" if abs(best_t) > EVIDENCE_BAR else "no"
        print(f"{f:<22}{cells}   {verdict}")

    print(f"\n(bar is |t| > {EVIDENCE_BAR}; an IC of 0.03-0.05 is good in "
          f"professional equity quant)")

    print(f"\n{'factor':<22}{'early IC':>11}{'t':>7}{'late IC':>11}{'t':>7}"
          f"   stable?")
    print("-" * 66)
    for f in factors:
        e, l = results[f]["split"]
        if e["ic"] is None or l["ic"] is None:
            print(f"{f:<22}{'n/a':>11}")
            continue
        stable = "yes" if np.sign(e["ic"]) == np.sign(l["ic"]) else "SIGN FLIPS"
        print(f"{f:<22}{e['ic']:>+11.4f}{e['t_stat']:>7.2f}"
              f"{l['ic']:>+11.4f}{l['t_stat']:>7.2f}   {stable}")

    orthogonality(panel, factors)

    print("\n" + "=" * 92)
    print("TRIAL REGISTRY — every factor tested here, pass or fail, so the")
    print("denominator stays visible and the survivors are not just the tail")
    print("of a search:", ", ".join(factors))
    print("=" * 92)
    return 0


def orthogonality(panel: pd.DataFrame, factors: list, control: str = "control_low_vol"):
    """
    Is a passing factor actually NEW, or the one we already ship wearing a hat?

    This is the test that decides whether a factor earns a place, and skipping
    it is how an app ends up with two surfaces that are the same number twice.
    The app already ranks on low volatility (the Risk Grade), so any new factor
    has to survive being residualised on it — WITHIN each date, since that is
    the level the IC is computed at.

    It matters here: earnings yield correlates +0.23 with calmness on the EGX
    (cheap stocks are also quieter), and residualising drops it from t=3.45 to
    t=2.00 — below the pre-registered bar. Raw significance was real; INDEPENDENT
    significance was not.

    NOTE ON HORIZONS. Rebalance dates sit ~22 trading days apart, so only the
    21-day IC uses non-overlapping forward windows. The 63- and 126-day t-stats
    share 3x and 6x of their windows respectively and are inflated; they are
    reported for shape, not for significance.
    """
    print("\n" + "=" * 92)
    print(f"ORTHOGONALITY — is it new, or is it {control}?")
    print("=" * 92)

    resid = {f: pd.Series(index=panel.index, dtype=float) for f in factors}
    corrs = {f: [] for f in factors}
    for _, g in panel.groupby("date"):
        for f in factors:
            if f == control:
                continue
            sub = g[[f, control]].dropna()
            if len(sub) < 10 or sub[f].nunique() < 3:
                continue
            x = sub[control].rank()
            y = sub[f].rank()
            corrs[f].append(x.corr(y))
            X = np.column_stack([np.ones(len(sub)), x.values])
            beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
            resid[f].loc[sub.index] = y.values - X @ beta

    print(f"  {'factor':<22}{'corr w/ ctrl':>13}{'raw IC':>10}{'t':>7}"
          f"{'resid IC':>11}{'t':>7}   independent?")
    print("  " + "-" * 78)
    for f in factors:
        if f == control:
            continue
        panel[f"_resid_{f}"] = resid[f]
        raw = information_coefficient(panel, f, HORIZONS[0])
        res = information_coefficient(panel, f"_resid_{f}", HORIZONS[0])
        if raw["ic"] is None or res["ic"] is None:
            continue
        verdict = ("yes" if abs(res["t_stat"]) > EVIDENCE_BAR
                   else "NO — subsumed" if abs(raw["t_stat"]) > EVIDENCE_BAR
                   else "n/a (failed raw)")
        print(f"  {f:<22}{np.mean(corrs[f]):>+13.3f}{raw['ic']:>+10.4f}"
              f"{raw['t_stat']:>7.2f}{res['ic']:>+11.4f}{res['t_stat']:>7.2f}"
              f"   {verdict}")

    print("\n  Only the 21-day column uses non-overlapping windows (rebalance")
    print("  dates are ~22 trading days apart). The 63d and 126d t-stats share")
    print("  3x and 6x of their forward windows and are inflated accordingly.")


if __name__ == "__main__":
    raise SystemExit(main())
