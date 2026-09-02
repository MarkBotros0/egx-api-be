"""
One-time backfill of Egyptian macro history into `macro_series`.

WHY THIS IS OFFLINE AND THE NIGHTLY REFRESH IS NOT
--------------------------------------------------
The nightly cron gets every series' CURRENT value from one keyless POST, which
is cheap. History is a different shape of work: each series comes through the
vendored tvDatafeed client one at a time, and USD/EGP alone is 5,000 daily bars.
That does not fit a 30-second serverless request, and it only needs doing once.

WHAT IT WRITES, AND THE PART THAT MATTERS
-----------------------------------------
Every bar's timestamp is its REFERENCE PERIOD. `macro_series.released_at` is
derived from it by adding that series' publication lag, and every historical
read filters on release rather than reference. Without that, a backtest asking
"what was inflation in August 2023" gets an answer nobody had until September,
and any macro-conditioned result becomes unfalsifiable.

Verified reachable before this was written:
  ECONOMICS:EGINTR   300 monthly bars, 2001-05 -> 2026-08
  FX_IDC:USDEGP    5,000 daily bars,  2006-09 -> 2026-09  (last 51.000)

Run:  ./.venv/Scripts/python.exe -m scripts.backfill_macro
      ./.venv/Scripts/python.exe -m scripts.backfill_macro --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.macro_series import SERIES, released_at, upsert_many  # noqa: E402

# Monthly for economic releases; daily for the FX rate, which is a price.
_INTERVAL = {"ECONOMICS": "Monthly", "FX_IDC": "Daily"}
_BARS = {"ECONOMICS": 300, "FX_IDC": 5000}

# Rows per multi-row INSERT, so the FX series is ten round trips rather than
# 5,000. Six bound parameters per row keeps a batch of 500 far inside
# Postgres' 65,535-parameter ceiling, and a batch that fails costs only itself.
WRITE_BATCH = 500


def fetch_history(code: str, exchange: str):
    from app.vendor.egxpy import get_OHLCV_data

    df = get_OHLCV_data(code, exchange,
                        _INTERVAL.get(exchange, "Monthly"),
                        _BARS.get(exchange, 300))
    if df is None or df.empty:
        return None
    df.columns = [str(c).lower() for c in df.columns]
    return df["close"].astype(float) if "close" in df.columns else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report without writing")
    ap.add_argument("--only", help="a single series code")
    args = ap.parse_args()

    db = None
    if not args.dry_run:
        from app.core.db import get_db, init_db
        db = get_db()
        init_db(db)          # idempotent; creates macro_series if absent

    total = 0
    incomplete = []
    for code, (label, exchange, lag, unit) in SERIES.items():
        if args.only and code != args.only.upper():
            continue
        try:
            series = fetch_history(code, exchange)
        except Exception as e:
            print(f"{code:<8} FAILED  {type(e).__name__}: {e}")
            continue
        if series is None or series.empty:
            print(f"{code:<8} no data")
            continue

        first, last = series.index[0], series.index[-1]
        print(f"{code:<8} {len(series):>5} bars  {first.date()} -> {last.date()}"
              f"  last={float(series.iloc[-1]):,.4f}  lag={lag}d  ({label})")
        print(f"         e.g. period {last.date()} becomes readable "
              f"{released_at(last.date().isoformat(), lag)}")

        if args.dry_run:
            continue

        # BATCHED multi-row INSERTs, and the batching is the whole fix.
        #
        # Three attempts, two of which lost data:
        #   1. One `upsert` per row on the pooled connection. 5,000 FX bars is
        #      5,000 round trips; the run exited 0 having stopped at 2022-10-31,
        #      because every row after the connection gave out fell into a bare
        #      `except: continue`.
        #   2. All 5,000 in ONE transaction. Neon's pooler closed the connection
        #      partway — "server closed the connection unexpectedly" — losing
        #      3,689 rows.
        #   3. 250-row transactions, still one statement per row. Slow enough
        #      that the run was killed before its first batch committed.
        #
        # `upsert_many` sends a batch as a single statement, so the FX series is
        # ~10 round trips. A batch that fails costs only itself, and a re-run
        # fills the gaps because every write is an idempotent upsert.
        written, failed, first_error = 0, 0, None
        rows = [(p.date().isoformat(), float(v))
                for p, v in series.items() if v == v]
        for start in range(0, len(rows), WRITE_BATCH):
            batch = rows[start:start + WRITE_BATCH]
            try:
                written += upsert_many(db, code, batch, source="backfill")
            except Exception as e:
                failed += len(batch)
                if first_error is None:
                    first_error = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            print(f"         ... {written:,}/{len(rows):,}", flush=True)

        total += written
        note = f"         wrote {written:,} rows"
        if failed:
            note += f"  ** {failed:,} NOT written — {first_error}"
        print(note)
        if failed:
            incomplete.append(code)

    print(f"\n{'would write' if args.dry_run else 'wrote'} {total:,} rows total")
    if args.dry_run:
        print("dry run — nothing was written")
        return 0
    if incomplete:
        # Non-zero exit, so a partial backfill can never be mistaken for a
        # finished one by whatever ran it. The first version of this script
        # returned 0 while writing two thirds of the data.
        print(f"INCOMPLETE: {', '.join(incomplete)}. Re-run — every write is "
              f"an idempotent upsert, so a second pass fills only the gaps.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
