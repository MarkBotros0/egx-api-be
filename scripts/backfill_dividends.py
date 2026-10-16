"""
One-time / occasional backfill of DEEP dividend history into `dividend_events`.

WHY OFFLINE, AND WHY THE NIGHTLY REFRESH IS NOT
-----------------------------------------------
The nightly `/api/pe/refresh` appends the ONE latest coupon per stock from the
scanner POST it already makes — cheap, and it grows the store forward. But deep
back-history (COMI has 21 coupons to 2010) comes one Yahoo GET per symbol, ~166
requests, which does not fit a 30-second serverless request and only needs doing
once. So this seeds the whole store from Yahoo; the refresh keeps it current.

Idempotent: `dividend_events` is keyed on (market, symbol, ex_date), so
re-running only adds coupons that were not already stored.

Per market: EG walks data/egx_tickers.json and asks Yahoo for `<SYM>.CA`; US
walks the index members in `market_universe` (read-only — run the universe cron
first) and asks for the bare ticker, a class share spelled with a dash
(BRK.B -> BRK-B). Rows are stored under our own symbol and the market.

Run:  ./.venv/bin/python -m scripts.backfill_dividends
      ./.venv/bin/python -m scripts.backfill_dividends --dry-run
      ./.venv/bin/python -m scripts.backfill_dividends --market US
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
from app.core.markets import EG, get_registered_market  # noqa: E402
from app.core.universe import index_members  # noqa: E402

_TICKERS = os.path.join(
    os.path.dirname(__file__), "..", "data", "egx_tickers.json"
)


def _universe(market=EG, db=None) -> list[str]:
    if market.code != EG.code:
        return sorted(index_members(db, market) or ())
    with open(_TICKERS, "r", encoding="utf-8") as f:
        return [(t.get("symbol") or "").strip().upper() for t in json.load(f) if t.get("symbol")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    ap.add_argument("--market", default=EG.code, help="EG (default) or US")
    args = ap.parse_args()

    market = get_registered_market(args.market)
    # US reads its symbol list from market_universe, so it needs the DB even dry.
    db = None if args.dry_run and market.code == EG.code else get_db()
    symbols = _universe(market, db)
    print(f"Backfilling {market.code} dividends for {len(symbols)} symbols "
          f"({'DRY RUN' if args.dry_run else 'writing'})\n")

    total_fetched = total_written = with_history = 0
    for i, sym in enumerate(symbols, 1):
        try:
            divs = fetch_dividends(sym, market=market)
        except Exception as e:
            print(f"  {sym:8s} FAIL  {type(e).__name__}")
            continue
        total_fetched += len(divs)
        if divs:
            with_history += 1
        written = 0
        if divs and not args.dry_run:
            written = upsert_dividends(db, sym, divs, source="yahoo-backfill",
                                       market_code=market.code)
            total_written += written
        newest = divs[0]["ex_date"] if divs else "—"
        print(f"  [{i:3d}/{len(symbols)}] {sym:8s} {len(divs):3d} coupons  "
              f"newest {newest}  (+{written} new)")
        time.sleep(0.15)  # be a polite citizen to Yahoo

    print(f"\nDone. {with_history}/{len(symbols)} symbols have history; "
          f"{total_fetched} coupons fetched, {total_written} newly stored.")


if __name__ == "__main__":
    main()
