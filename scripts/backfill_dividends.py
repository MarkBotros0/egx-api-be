"""
One-time / occasional backfill of DEEP dividend history into `dividend_events`.

WHY OFFLINE, AND WHY THE NIGHTLY REFRESH IS NOT
-----------------------------------------------
The nightly `/api/pe/refresh` appends the ONE latest coupon per stock from the
scanner POST it already makes — cheap, and it grows the store forward. But deep
back-history (COMI has 21 coupons to 2010) comes one Yahoo GET per symbol, ~166
requests, which does not fit a 30-second serverless request and only needs doing
once. So this seeds the whole store from Yahoo; the refresh keeps it current.

Idempotent: `dividend_events` is keyed on (symbol, ex_date), so re-running only
adds coupons that were not already stored.

Run:  ./.venv/Scripts/python.exe -m scripts.backfill_dividends
      ./.venv/Scripts/python.exe -m scripts.backfill_dividends --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db import get_db  # noqa: E402
from app.core.dividend_history import fetch_dividends, upsert_dividends  # noqa: E402

_TICKERS = os.path.join(
    os.path.dirname(__file__), "..", "data", "egx_tickers.json"
)


def _universe() -> list[str]:
    with open(_TICKERS, "r", encoding="utf-8") as f:
        return [(t.get("symbol") or "").strip().upper() for t in json.load(f) if t.get("symbol")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    args = ap.parse_args()

    symbols = _universe()
    db = None if args.dry_run else get_db()
    print(f"Backfilling dividends for {len(symbols)} symbols "
          f"({'DRY RUN' if args.dry_run else 'writing'})\n")

    total_fetched = total_written = with_history = 0
    for i, sym in enumerate(symbols, 1):
        try:
            divs = fetch_dividends(sym)
        except Exception as e:
            print(f"  {sym:8s} FAIL  {type(e).__name__}")
            continue
        total_fetched += len(divs)
        if divs:
            with_history += 1
        written = 0
        if divs and not args.dry_run:
            written = upsert_dividends(db, sym, divs, source="yahoo-backfill")
            total_written += written
        newest = divs[0]["ex_date"] if divs else "—"
        print(f"  [{i:3d}/{len(symbols)}] {sym:8s} {len(divs):3d} coupons  "
              f"newest {newest}  (+{written} new)")
        time.sleep(0.15)  # be a polite citizen to Yahoo

    print(f"\nDone. {with_history}/{len(symbols)} symbols have history; "
          f"{total_fetched} coupons fetched, {total_written} newly stored.")


if __name__ == "__main__":
    main()
