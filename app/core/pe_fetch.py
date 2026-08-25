"""
EGX fundamentals feed — trailing P/E, dividend yield, and loss-making status.

WHY THIS IS NOT THE EGX WEBSITE ANY MORE
----------------------------------------
This module used to scrape https://www.egx.com.eg/en/MarketPECompanies.aspx.
That page is now behind a JavaScript bot challenge: the request returns HTTP
200 with ~6 KB reading "Please enable JavaScript to view the page content",
so the GridView parser matched zero rows. It had never once succeeded in
production — `pe_data` was empty and `pe_last_attempt_status` was blank, which
means `score_quality`'s P/E band had never executed against real data.

The source is now the TradingView scanner (`core/tradingview.py`), which the
ticker list already depends on. Three consequences worth knowing:

  1. It returns the BARE SYMBOL ("COMI"), so the whole company-name matching
     apparatus — overrides file, normalization, prefix and jaccard matching —
     is gone. Symbols are exact; there is nothing left to mis-resolve.
  2. It never reports a NEGATIVE P/E. Loss-making companies come back with a
     null P/E instead, so "is this company losing money" is derived from
     diluted EPS (TTM) < 0 and stored separately as `loss_making`.
  3. `dividend_yield == 0.0` is REAL DATA meaning "pays no dividend" — it is
     not the old EGX "0 means no data" sentinel. Only None means unknown.
     Conflating them would penalise growth companies for a missing value.

Coverage is partial and that is expected: of ~293 EGX stocks, roughly 64 have
a trailing P/E and 92 a dividend yield. A missing value skips its scoring band
rather than defaulting, so a stock is never punished for the feed's silence.

On any fetch failure the existing rows are left untouched and the read path
keeps returning last-known-good.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.core.tradingview import scan

# Positional column request. `d` in each returned row aligns to this order.
TV_COLUMNS = [
    "name",                            # bare symbol, e.g. "COMI"
    "description",                     # company name, for display only
    "price_earnings_ttm",
    "dividends_yield",                 # the only DY column that carries data
    "earnings_per_share_diluted_ttm",  # sign gives loss_making
]

# Above this, a P/E carries no information beyond "barely profitable" — and
# the reason string would render "P/E 2756.0", which reads as a broken app.
# The live EGX maximum is ~2756.
PE_SANITY_MAX = 300.0

# A dividend yield beyond this is a data error rather than a real payout.
DY_SANITY_MAX = 60.0

# EGX lists ~293 stocks. A response far below that is truncated, not a market
# that shrank — see refresh_pe_data for why partial writes are refused.
MIN_EXPECTED_ROWS = 100


# ---------------------------------------------------------------------------
# Fetch + normalize
# ---------------------------------------------------------------------------

def _clean_float(value, maximum: Optional[float] = None) -> Optional[float]:
    """Coerce a scanner cell to float. Non-numeric or out-of-range -> None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if maximum is not None and abs(f) > maximum:
        return None
    return f


def fetch_fundamentals_rows() -> list:
    """
    One POST for the whole EGX universe. Returns
        [{"symbol", "company_name", "pe_ratio", "dividend_yield", "loss_making"}]

    Raises on HTTP/JSON error — the caller owns the never-wipe guarantee.
    """
    raw = scan(TV_COLUMNS)

    out = []
    for row in raw:
        d = row.get("d") or []
        if len(d) < len(TV_COLUMNS):
            continue

        symbol = (d[0] or "").strip().upper()
        if not symbol:
            continue

        eps = _clean_float(d[4])

        out.append({
            "symbol": symbol,
            "company_name": (d[1] or "").strip() or None,
            # A zero or negative P/E from this source is meaningless: profitable
            # companies get a positive number and loss-makers get null.
            "pe_ratio": _clean_float(d[2], maximum=PE_SANITY_MAX),
            # 0.0 is preserved deliberately — it means "pays nothing".
            "dividend_yield": _clean_float(d[3], maximum=DY_SANITY_MAX),
            "loss_making": (eps < 0) if eps is not None else None,
        })
    return out


# ---------------------------------------------------------------------------
# DB writes / reads
# ---------------------------------------------------------------------------

def refresh_pe_data(db, rows: Optional[list] = None) -> dict:
    """
    Refresh the fundamentals feed and upsert into pe_data.

    Passing `rows` skips the HTTP call — the seam used by tests and local dev.

    NEVER WIPES. On failure the existing rows remain and only
    `pe_last_attempt_status` changes, so the read path keeps serving
    last-known-good rather than falling back to "no data".
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        if rows is None:
            rows = fetch_fundamentals_rows()
    except Exception as e:
        _write_setting(db, "pe_last_attempt_status", f"error: {type(e).__name__}: {e}")
        return {"success": False, "count": 0, "error": str(e)}

    if not rows:
        _write_setting(db, "pe_last_attempt_status", "error: no rows returned")
        return {"success": False, "count": 0, "error": "no rows returned"}

    # Refuse a truncated response WITHOUT writing anything. A partial refresh
    # that updates 50 symbols and silently leaves 240 stale is worse than
    # "everything is stale", because nothing on screen distinguishes the two.
    if len(rows) < MIN_EXPECTED_ROWS:
        msg = f"error: only {len(rows)} rows (expected >= {MIN_EXPECTED_ROWS})"
        _write_setting(db, "pe_last_attempt_status", msg)
        return {"success": False, "count": 0, "error": msg}

    written = 0
    skipped = 0
    for row in rows:
        # Nothing to say about this company — don't create a row that makes
        # get_pe_for_symbol return a truthy all-null dict.
        if (row.get("pe_ratio") is None
                and row.get("dividend_yield") is None
                and row.get("loss_making") is None):
            skipped += 1
            continue

        db.execute(
            """
            INSERT INTO pe_data
                (symbol, company_name, pe_ratio, dividend_yield, loss_making, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                company_name   = EXCLUDED.company_name,
                pe_ratio       = EXCLUDED.pe_ratio,
                dividend_yield = EXCLUDED.dividend_yield,
                loss_making    = EXCLUDED.loss_making,
                updated_at     = EXCLUDED.updated_at
            """,
            (row["symbol"], row.get("company_name"), row.get("pe_ratio"),
             row.get("dividend_yield"), row.get("loss_making"), now),
        )
        written += 1

    _write_setting(db, "pe_last_successful_fetch", now)
    _write_setting(db, "pe_last_attempt_status", "ok")

    return {
        "success": True,
        "count": written,
        "total_rows": len(rows),
        "skipped_empty": skipped,
    }


def get_pe_for_symbol(db, symbol: str) -> Optional[dict]:
    """Read-side helper. Returns the last stored fundamentals row, or None."""
    row = db.execute(
        "SELECT company_name, pe_ratio, dividend_yield, loss_making, updated_at "
        "FROM pe_data WHERE symbol = %s",
        (symbol.upper(),),
    ).fetchone()
    if not row:
        return None
    return {
        "company_name": row[0],
        "pe_ratio": row[1],
        "dividend_yield": row[2],
        "loss_making": row[3],
        "fetched_at": row[4],
    }


def _write_setting(db, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO settings (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )
