"""
Teach the risk snapshot which symbols the feed will never serve, from evidence
already on disk.

THE PROBLEM THIS SOLVES
-----------------------
`POST /api/cron/risk_snapshot` demotes a symbol after
FAILURE_DEMOTION_THRESHOLD consecutive refusals. That works, but on a cold table
it has to LEARN the bad half the expensive way: 84 of the 166 symbols in the
ticker file have never returned data, each refusal costs about six seconds
against a 15-second budget, and each needs three strikes. Left to discover this
itself the job spends the better part of a day failing before the symbols that
work get priority.

The disk cache already knows the answer. `scripts/.cache/` was built by an
exhaustive retry loop, and a cached `None` means the feed genuinely had nothing
for that symbol — the same signal the cron is slowly rediscovering.

Seeding those counts turns a day of grinding into one run.

IT IS A HINT, NOT A BLOCKLIST
-----------------------------
Every seeded symbol is still fetched, just after everything healthy is fresh,
and a single successful fetch resets its counter to zero. If TradingView starts
serving one of these tomorrow, the app picks it up without anyone editing a
list. That is why this writes a COUNTER rather than an exclusion.

`measured_at` is deliberately set to the epoch: the row exists so the counter
has somewhere to live, but the symbol still reads as maximally stale, so nothing
here can make an unmeasured stock look measured.

Run:  ./.venv/Scripts/python.exe -m scripts.seed_symbol_health --dry-run
      ./.venv/Scripts/python.exe -m scripts.seed_symbol_health
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
NOT_A_SYMBOL = {"panel", "breadth", "fundamentals_annual", "results"}

# So a seeded symbol sorts as never-measured while still being demoted.
EPOCH = "1970-01-01T00:00:00+00:00"


def cache_verdicts() -> tuple:
    """(symbols the feed served, symbols it had nothing for)."""
    served, empty = set(), set()
    for path in glob.glob(os.path.join(CACHE_DIR, "*.pkl")):
        name = os.path.basename(path)[:-4]
        if name in NOT_A_SYMBOL:
            continue
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
        except Exception:
            # Unreadable is not the same as empty — it says nothing about the
            # feed, only about this cache entry. Leave it out of both sets.
            continue
        (served if df is not None and not getattr(df, "empty", True)
         else empty).add(name.upper())
    return served, empty


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be seeded without writing")
    args = ap.parse_args()

    from app.routers.cron import FAILURE_DEMOTION_THRESHOLD, _universe

    universe = set(_universe())
    served, empty = cache_verdicts()

    known_bad = sorted(universe & empty)
    known_good = sorted(universe & served)
    unknown = sorted(universe - served - empty)

    print(f"universe                : {len(universe)}")
    print(f"  feed has served       : {len(known_good)}")
    print(f"  feed never served     : {len(known_bad)}  <- seeded as demoted")
    print(f"  no cache evidence     : {len(unknown)}  <- left alone")
    if known_bad:
        preview = ", ".join(known_bad[:12])
        more = f" (+{len(known_bad) - 12} more)" if len(known_bad) > 12 else ""
        print(f"\ndemoting: {preview}{more}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0
    if not known_bad:
        print("\nnothing to seed")
        return 0

    from app.core.db import get_db, init_db
    db = get_db()
    init_db(db)          # idempotent; ensures consecutive_failures exists

    written = 0
    for symbol in known_bad:
        try:
            db.execute(
                """
                INSERT INTO risk_snapshot
                    (symbol, measured_at, tradeable, consecutive_failures)
                VALUES (%s, %s, FALSE, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    consecutive_failures = GREATEST(
                        COALESCE(risk_snapshot.consecutive_failures, 0), %s
                    )
                """,
                (symbol, EPOCH, FAILURE_DEMOTION_THRESHOLD,
                 FAILURE_DEMOTION_THRESHOLD),
            )
            written += 1
        except Exception as e:
            print(f"  {symbol}: {type(e).__name__}: {e}")

    print(f"\nseeded {written} symbols as demoted.")
    print(f"The cron will now reach the {len(known_good)} working symbols first.")
    print("Any of them that starts working resets to 0 on its first good fetch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
