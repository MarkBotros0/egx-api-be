"""
Analyse the panel produced by scripts/backtest.py.

Separate from the backtest itself so the analysis can be re-run, and re-cut,
without re-scoring 40,000 symbol-dates.

Reports, in order of how much weight they should carry:

  1. SANITY CHECKS. A shuffled score must produce IC ~ 0. If the harness cannot
     detect the absence of signal, a positive result proves nothing about the
     score and everything about the plumbing. Nothing below is worth reading
     until this passes.
  2. INFORMATION COEFFICIENT for the composite, per horizon — the primary
     result, with a t-statistic and 95% interval.
  3. PER-CATEGORY IC — which of the eight actually carry signal.
  4. BUCKET RETURNS by signal band and score decile — the intuitive view, but
     the one most distorted by survivorship and by EGP devaluations, so it is
     reported last and read with suspicion.

Run:  python -m scripts.analyze_backtest            (from egx-api-be)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.composite import CATEGORY_ORDER  # noqa: E402
from scripts.backtest import (  # noqa: E402
    HORIZONS,
    bucket_returns,
    information_coefficient,
    read_run_meta,
    sanity_checks,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# Median EGX daily volume is ~1M shares; this keeps the "liquid" cut to names a
# retail investor could actually enter and exit at the quoted price.
LIQUID_MIN_AVG_VOLUME = 100_000


def describe(panel: pd.DataFrame) -> dict:
    return {
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "dates": int(panel["date"].nunique()),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
    }


def report(panel: pd.DataFrame, label: str, run_meta: dict | None = None) -> dict:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    meta = describe(panel)
    # How the RUN was scored, not how this machine would score it.
    run_meta = run_meta or read_run_meta("")
    confounded = list(run_meta.get("confounded") or [])
    print(f"{meta['rows']:,} symbol-dates | {meta['symbols']} symbols | "
          f"{meta['dates']} dates | {meta['first_date']} .. {meta['last_date']}")

    checks = sanity_checks(panel)
    verdict = "PASS" if checks["passes"] else "FAIL"
    print(f"\n[sanity] shuffled-score placebo IC = {checks['placebo_ic']} "
          f"(t={checks['placebo_t']})  -> {verdict}")
    if not checks["passes"]:
        print("  !! The harness reports signal on randomised scores. Nothing")
        print("     below can be trusted until this is understood.")

    print("\n[composite] Information Coefficient")
    print(f"  {'horizon':>8}  {'IC':>8}  {'t':>6}  {'95% CI':>18}  {'dates':>6}  {'IC>0':>6}")
    composite = {}
    for h in HORIZONS:
        r = information_coefficient(panel, "score", h)
        composite[h] = r
        if r["ic"] is None:
            print(f"  {h:>8}  {'n/a':>8}   (too few usable dates)")
            continue
        ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        print(f"  {h:>8}d {r['ic']:>+8.4f}  {r['t_stat']:>6.2f}  {ci:>18}  "
              f"{r['n_dates']:>6}  {r['hit_rate']:>6.1%}")

    print("\n[categories] IC at each horizon  (* = confounded, see note)")
    header = "  " + f"{'category':>20}" + "".join(f"{str(h) + 'd':>12}" for h in HORIZONS)
    print(header)
    cats = {}
    for name in CATEGORY_ORDER:
        col = f"cat_{name}"
        if col not in panel.columns:
            continue
        row = {}
        cells = ""
        for h in HORIZONS:
            r = information_coefficient(panel, col, h)
            row[h] = r
            cells += f"{r['ic']:>+12.4f}" if r["ic"] is not None else f"{'n/a':>12}"
        cats[name] = row
        mark = " *" if name in confounded else ""
        print(f"  {name:>20}{cells}{mark}")
    if confounded:
        print(f"  * {', '.join(confounded)}: scored against a FLAT "
              f"{run_meta.get('flat_rate_pct')}% risk-free rate.")
        print("    Egypt's policy rate ran 8.25% to 27.25% over this window, so this")
        print("    row is not evidence about whether the category earns its weight.")
        print("    Re-run scripts/backtest.py with macro_series populated to lift it.")
    else:
        print(f"  (risk-free rate is DATED — {run_meta.get('rate_steps')} steps from "
              f"macro_series; every row above is evidence)")

    print("\n[buckets] mean forward return by signal band")
    buckets = {}
    for h in HORIZONS:
        rows = bucket_returns(panel, h, by="signal")
        buckets[h] = rows
        cells = "  ".join(f"{r['bucket']}={r['mean_fwd_pct']:+.1f}%(n={r['n']})" for r in rows)
        print(f"  {h:>3}d: {cells}")

    deciles = {h: bucket_returns(panel, h, by="decile") for h in HORIZONS}
    top, bot = deciles[HORIZONS[0]][-1:], deciles[HORIZONS[0]][:1]
    if top and bot:
        print(f"\n  top vs bottom decile @ {HORIZONS[0]}d: "
              f"{top[0]['mean_fwd_pct']:+.2f}% vs {bot[0]['mean_fwd_pct']:+.2f}%")

    return {"meta": meta, "sanity": checks, "composite": composite,
            "categories": cats, "buckets": buckets, "deciles": deciles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(CACHE_DIR, "panel.pkl"))
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "results.json"))
    args = ap.parse_args()

    if not os.path.exists(args.panel):
        print(f"no panel at {args.panel} — run `python -m scripts.backtest` first")
        return 1

    panel = pd.read_pickle(args.panel)
    # Read the RUN's own stamp. An unstamped panel reads as flat-rate, so every
    # panel built before stamping existed keeps its caveat.
    run_meta = read_run_meta(args.panel)
    results = {
        "run": run_meta,
        "full_universe": report(panel, "FULL UNIVERSE", run_meta),
    }

    # The liquid cut matters because returns on names trading a few thousand
    # shares a day were never capturable — a strategy that "works" only there
    # is not a strategy.
    if "avg_volume" in panel.columns:
        liquid = panel[panel["avg_volume"] >= LIQUID_MIN_AVG_VOLUME]
        if len(liquid) > 500:
            results["liquid_only"] = report(
                liquid, f"LIQUID ONLY (>= {LIQUID_MIN_AVG_VOLUME:,} shares/day)",
                run_meta)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved -> {args.out}")

    print("\n" + "=" * 72)
    print("READ THIS BEFORE ACTING ON ANY NUMBER ABOVE")
    print("=" * 72)
    print("- Survivorship: the universe is TODAY's listings. Companies that")
    print("  delisted are absent, and nothing in the app records them, so the")
    print("  bias is upward and unmeasurable. IC ranks within each date's")
    print("  surviving cross-section, which is why it leads and bucket returns")
    print("  trail.")
    print("- Fundamentals are NOT tested: pe_data is a today-only snapshot and")
    print("  fundamentals_history only began collecting on 2026-08-25. Quality")
    print("  was scored on its technical inputs alone.")
    print("- Currency: the EGP was devalued sharply in 2016, 2022, 2023 and")
    print("  2024. Absolute returns are inflated in EGP terms; rank")
    print("  correlations largely absorb it.")
    print("- Costs: returns are GROSS. EGX commissions and spreads on thin")
    print("  names are not deducted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
