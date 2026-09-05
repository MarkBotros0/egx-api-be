"""
Twenty years of point-in-time annual fundamentals, from one scanner call.

WHAT THIS UNLOCKS
-----------------
Until now this project could not test a fundamental factor at all.
`scripts/backtest.py` says so in its own docstring and refuses to try: `pe_data`
is a current-value snapshot that every refresh destroys, and
`fundamentals_history` only began collecting on 2026-08-25. Scoring a 2019 date
with today's P/E is look-ahead bias severe enough to manufacture any result.

The TradingView scanner the nightly cron ALREADY calls also returns `*_fy_h`
history arrays — the whole annual series per company, aligned element-wise to
`fiscal_period_fy_h`. Verified live: 246 of 296 EGX rows (83%) carry them, 114
symbols with a full 20 years. That converts "fundamentals are untestable here"
into "fundamentals are testable back to 2012", for the cost of one extra POST
to a host we already talk to.

THE LOOK-AHEAD GUARD IS THE WHOLE POINT
---------------------------------------
A fiscal year's numbers are not knowable on the last day of that year. Every row
carries `first_usable_date` = fiscal year end + REPORTING_LAG_DAYS, and NOTHING
may read a row before that date. `get_annual_asof` enforces it; do not query the
table directly in a backtest.

TWO LIMITS THAT CANNOT BE ENGINEERED AWAY — state them wherever results are:

1. **The arrays are as-RESTATED, not as-first-reported.** There is no publish
   date at any spelling (report_publish_date_fy_h and friends return nothing for
   EGX), so a fixed lag is the only available defence and residual restatement
   bias survives it. A company that later revised a bad year looks better in
   this table than it did to an investor at the time.
2. **Pre-2012 is not usable and is refused at ingest.** Measured per fiscal
   year, symbols with a usable diluted EPS: 2018 → 225, 2015 → 207, 2012 → 160,
   then it falls off a cliff — 2011 → 63, 2009 → 52, 2007 → 23, 2006 → 15. The
   rows exist further back; the data does not. Including them would quietly
   weight a cross-sectional test toward whichever handful of large caps happened
   to report.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.core.tradingview import scan

# Positional column request; each `d` aligns to this order. The first is the
# key every other array is indexed against.
TV_HISTORY_COLUMNS = [
    "name",
    "fiscal_period_fy_h",
    "earnings_per_share_diluted_fy_h",
    "dps_common_stock_prim_issue_fy_h",
    "net_income_fy_h",
    "total_revenue_fy_h",
    "total_assets_fy_h",
    "gross_profit_fy_h",
    "total_debt_fy_h",
]

# Field order after `name` and `fiscal_period_fy_h`, matching the DB columns.
_VALUE_FIELDS = (
    "eps_diluted",
    "dps",
    "net_income",
    "revenue",
    "total_assets",
    "gross_profit",
    "total_debt",
)

# Days after fiscal year end before a figure is treated as knowable. EGX rules
# require annual financials within 90 days; 120 is deliberately conservative,
# because being a month late costs a backtest almost nothing while being a month
# early is look-ahead that invalidates it.
REPORTING_LAG_DAYS = 120

# Measured floor — see the module docstring. Below this, EPS coverage collapses.
MIN_FISCAL_YEAR = 2012

# A response far below the live 246 is truncated, not a market that shrank.
# Mirrors refresh_pe_data's guard, and for the same reason: a partial write that
# updates a third of the universe is worse than no write, because nothing on
# screen distinguishes the two.
MIN_EXPECTED_SYMBOLS = 150

# Rows per multi-row INSERT. See `_upsert_batch` for why this is batched at all,
# and why the number is 250 rather than "all of them".
WRITE_BATCH = 250


def first_usable_date(fiscal_year: int) -> str:
    """
    The earliest date this fiscal year's figures may be used.

    Assumes a December year end, which is the EGX norm. A company on a different
    fiscal calendar gets a date that is too EARLY by up to a year — the one
    direction that reintroduces look-ahead — so if a non-December cohort is ever
    identified, this needs a per-symbol year end rather than a constant.
    """
    return (date(fiscal_year, 12, 31)
            + timedelta(days=REPORTING_LAG_DAYS)).isoformat()


def _num(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None      # NaN -> None


def fetch_annual_rows(raw: Optional[list] = None) -> list:
    """
    One POST, exploded into one record per (symbol, fiscal year).

    Passing `raw` skips the HTTP call — the seam used by tests.
    """
    if raw is None:
        raw = scan(TV_HISTORY_COLUMNS)

    out = []
    for row in raw:
        d = row.get("d") or []
        if len(d) < len(TV_HISTORY_COLUMNS):
            continue
        symbol = (d[0] or "").strip().upper()
        years = d[1]
        if not symbol or not isinstance(years, list):
            continue

        arrays = d[2:2 + len(_VALUE_FIELDS)]
        for i, year in enumerate(years):
            try:
                fiscal_year = int(year)
            except (TypeError, ValueError):
                continue
            if fiscal_year < MIN_FISCAL_YEAR:
                continue

            record = {"symbol": symbol, "fiscal_year": fiscal_year,
                      "first_usable_date": first_usable_date(fiscal_year)}
            for field, arr in zip(_VALUE_FIELDS, arrays):
                record[field] = (_num(arr[i])
                                 if isinstance(arr, list) and i < len(arr)
                                 else None)
            # A row where every figure is missing carries no information and
            # would make a coverage count look better than it is.
            if all(record[f] is None for f in _VALUE_FIELDS):
                continue
            out.append(record)
    return out


def refresh_annual_fundamentals(db, rows: Optional[list] = None) -> dict:
    """
    Upsert the annual archive. NEVER WIPES, matching refresh_pe_data.

    Idempotent by (symbol, fiscal_year): a re-run with identical data rewrites
    the same values. Restatements land as updates, which is correct — but see
    the as-restated caveat in the module docstring before reading history off
    this table as if it were what the market knew at the time.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        if rows is None:
            rows = fetch_annual_rows()
    except Exception as e:
        return {"success": False, "written": 0, "error": f"{type(e).__name__}: {e}"}

    symbols = {r["symbol"] for r in rows}
    if len(symbols) < MIN_EXPECTED_SYMBOLS:
        return {"success": False, "written": 0,
                "error": (f"only {len(symbols)} symbols "
                          f"(expected >= {MIN_EXPECTED_SYMBOLS}) — refusing a "
                          f"partial write")}

    # Deduplicate on the PRIMARY KEY before batching, keeping the last value.
    # Postgres refuses an "ON CONFLICT DO UPDATE" that would touch the same row
    # twice in one statement, so a feed repeating a (symbol, fiscal_year) would
    # abort a whole batch — a failure per-row upserts never had to care about.
    # Same lesson as macro_series.upsert_many.
    deduped: dict = {}
    for r in rows:
        deduped[(r["symbol"], r["fiscal_year"])] = r
    unique = list(deduped.values())

    written = 0
    for start in range(0, len(unique), WRITE_BATCH):
        written += _upsert_batch(db, unique[start:start + WRITE_BATCH], now)

    return {"success": True, "written": written, "symbols": len(symbols),
            "years": len({r["fiscal_year"] for r in unique}),
            "batches": (len(unique) + WRITE_BATCH - 1) // WRITE_BATCH}


def _upsert_batch(db, batch: list, now: str) -> int:
    """
    One multi-row INSERT for up to WRITE_BATCH records.

    WHY THIS IS NOT A LOOP OF SINGLE-ROW UPSERTS. It was, and the live archive
    is ~2,555 records, so the nightly refresh issued ~2,555 sequential round
    trips to Neon from inside a request with a 30-second ceiling — on top of the
    ~130 `refresh_pe_data` already makes. That is the exact shape that cost the
    FX backfill two of its three attempts: one statement per row over a pooled
    connection either crawls or gets cut off partway, and a bare per-row write
    leaves no way to tell a finished run from a truncated one.

    Batched, the same archive lands in ~11 round trips.

    Deliberately NO wall-clock deadline here, unlike the risk-snapshot cron.
    That job measures independent symbols and stopping early costs nothing
    because selection is stalest-first. This one writes a coherent archive, and
    `refresh_annual_fundamentals` already refuses to write at all rather than
    write part of the universe — a deadline that truncated the write would
    reintroduce precisely the half-written state that guard exists to prevent.
    Batching removes the need for one.
    """
    if not batch:
        return 0

    row_sql = "(" + ", ".join(["%s"] * 11) + ")"
    values_sql = ", ".join([row_sql] * len(batch))

    params: list = []
    for r in batch:
        params.extend([
            r["symbol"], r["fiscal_year"], r["eps_diluted"], r["dps"],
            r["net_income"], r["revenue"], r["total_assets"],
            r["gross_profit"], r["total_debt"], r["first_usable_date"], now,
        ])

    db.execute(
        f"""
        INSERT INTO fundamentals_annual
            (symbol, fiscal_year, eps_diluted, dps, net_income, revenue,
             total_assets, gross_profit, total_debt, first_usable_date,
             updated_at)
        VALUES {values_sql}
        ON CONFLICT (symbol, fiscal_year) DO UPDATE SET
            eps_diluted       = EXCLUDED.eps_diluted,
            dps               = EXCLUDED.dps,
            net_income        = EXCLUDED.net_income,
            revenue           = EXCLUDED.revenue,
            total_assets      = EXCLUDED.total_assets,
            gross_profit      = EXCLUDED.gross_profit,
            total_debt        = EXCLUDED.total_debt,
            first_usable_date = EXCLUDED.first_usable_date,
            updated_at        = EXCLUDED.updated_at
        """,
        tuple(params),
    )
    return len(batch)


def get_annual_asof(db, as_of: str) -> dict:
    """
    The latest fundamentals KNOWABLE on `as_of`, per symbol: {symbol: row}.

    This is the only sanctioned way to read the table for a historical date.
    The `first_usable_date <= as_of` filter is what stands between a backtest
    and look-ahead bias — a fiscal year's figures are not knowable on 31
    December of that year, and using them there would let any factor appear to
    work.

    One query for a whole date, because a backtest asks this per rebalance date
    across the universe and per-symbol reads would dominate its runtime.
    """
    rows = db.execute(
        "SELECT DISTINCT ON (symbol) symbol, fiscal_year, eps_diluted, dps, "
        "net_income, revenue, total_assets, gross_profit, total_debt, "
        "first_usable_date "
        "FROM fundamentals_annual WHERE first_usable_date <= %s "
        "ORDER BY symbol, fiscal_year DESC",
        (as_of,),
    ).fetchall()
    return {
        r[0]: {"fiscal_year": r[1], "eps_diluted": r[2], "dps": r[3],
               "net_income": r[4], "revenue": r[5], "total_assets": r[6],
               "gross_profit": r[7], "total_debt": r[8],
               "first_usable_date": r[9]}
        for r in rows
    }
