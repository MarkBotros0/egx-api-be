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
#
# The last four are PRICE-INDEPENDENT fundamentals kept for the history log.
# P/E, P/B and dividend yield all divide by price, so they move daily and a
# history of them would be mostly price noise; EPS / DPS / book value move
# quarterly and let any of the ratios be reconstructed at a past date.
TV_COLUMNS = [
    "name",                            # bare symbol, e.g. "COMI"
    "description",                     # company name, for display only
    "price_earnings_ttm",
    "dividends_yield",                 # the only DY column that carries data
    "earnings_per_share_diluted_ttm",  # sign gives loss_making
    "dps_common_stock_prim_issue_fy",  # annual DPS (the ttm variant is ~3% covered)
    "book_value_per_share_fq",         # ~60% covered — widest fundamental we get
    "close",
    # Added 2026-09-02, all verified live for EGX coverage before inclusion.
    # Same single POST, so they cost nothing.
    "market_cap_basic",                   # 82% — the missing size control
    "total_shares_outstanding_current",   # 80% — lets a share count be checked
    "beta_1_year",                        # 82%
    "Value.Traded",                       # 100% — EGP turnover, the honest
                                          #        liquidity primitive; the app
                                          #        currently proxies it with
                                          #        close * share volume
    #
    # DELIBERATELY NOT REQUESTED, coverage measured on 296 EGX rows:
    #   return_on_equity            25%  — too thin to score against a median
    #   price_book_ratio            33%
    #   earnings_release_next_date  10%  — 29 symbols. The research framed this
    #                                      as a forward-looking win; at 10% it
    #                                      cannot carry a feature, and a badge
    #                                      that appears on one stock in ten
    #                                      reads as a bug.
    #   float_shares_outstanding    55%  — revisit if free float is ever needed
]

# Fields whose change triggers a new history row. Deliberately excludes the
# ratios and the close: those move every day.
_HISTORY_FIELDS = ("eps_ttm", "dps_annual", "book_value_per_share", "loss_making")

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
            # Price-independent fundamentals, for the history log.
            "eps_ttm": eps,
            "dps_annual": _clean_float(d[5]),
            "book_value_per_share": _clean_float(d[6]),
            "close": _clean_float(d[7]),
            "market_cap": _clean_float(d[8]),
            "shares_outstanding": _clean_float(d[9]),
            "beta_1y": _clean_float(d[10]),
            "value_traded_egp": _clean_float(d[11]),
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
                (symbol, company_name, pe_ratio, dividend_yield, loss_making,
                 market_cap, shares_outstanding, beta_1y, value_traded_egp,
                 updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                company_name       = EXCLUDED.company_name,
                pe_ratio           = EXCLUDED.pe_ratio,
                dividend_yield     = EXCLUDED.dividend_yield,
                loss_making        = EXCLUDED.loss_making,
                market_cap         = EXCLUDED.market_cap,
                shares_outstanding = EXCLUDED.shares_outstanding,
                beta_1y            = EXCLUDED.beta_1y,
                value_traded_egp   = EXCLUDED.value_traded_egp,
                updated_at         = EXCLUDED.updated_at
            """,
            (row["symbol"], row.get("company_name"), row.get("pe_ratio"),
             row.get("dividend_yield"), row.get("loss_making"),
             row.get("market_cap"), row.get("shares_outstanding"),
             row.get("beta_1y"), row.get("value_traded_egp"), now),
        )
        written += 1

    appended = _append_fundamentals_history(db, rows, now)

    _write_setting(db, "pe_last_successful_fetch", now)
    _write_setting(db, "pe_last_attempt_status", "ok")

    return {
        "success": True,
        "count": written,
        "total_rows": len(rows),
        "skipped_empty": skipped,
        "history_rows_appended": appended,
    }


def _latest_history(db) -> dict:
    """
    Most recent history row per symbol: {symbol: {field: value}}.

    One query rather than one per symbol — the refresh handles ~293 symbols and
    per-symbol reads would dominate its runtime.
    """
    try:
        rows = db.execute(
            "SELECT DISTINCT ON (symbol) symbol, eps_ttm, dps_annual, "
            "book_value_per_share, loss_making "
            "FROM fundamentals_history ORDER BY symbol, observed_at DESC"
        ).fetchall()
    except Exception:
        return {}
    return {
        r[0]: {"eps_ttm": r[1], "dps_annual": r[2],
               "book_value_per_share": r[3], "loss_making": r[4]}
        for r in rows
    }


def _changed(previous: Optional[dict], row: dict) -> bool:
    """Has any price-independent fundamental moved since the last observation?"""
    if previous is None:
        return True
    for field in _HISTORY_FIELDS:
        old, new = previous.get(field), row.get(field)
        if old is None or new is None:
            if old is not new:      # one side gained or lost a value
                return True
            continue
        if isinstance(old, bool) or isinstance(new, bool):
            if bool(old) != bool(new):
                return True
            continue
        # Float compare with a relative tolerance: the feed re-derives these
        # and the last decimal jitters. Logging that jitter would defeat the
        # point of an append-on-change design.
        if abs(float(old) - float(new)) > max(1e-9, abs(float(old)) * 1e-6):
            return True
    return False


def _append_fundamentals_history(db, rows: list, now: str) -> int:
    """
    Append a row per symbol whose fundamentals changed since we last looked.

    Append-only and change-triggered: EPS, DPS and book value move quarterly,
    so this stays small, while `pe_data` keeps being overwritten in place for
    the read path. Failures here must NOT fail the refresh — the current-value
    feed is what the app serves; history is for later analysis.
    """
    previous = _latest_history(db)
    appended = 0
    for row in rows:
        if all(row.get(f) is None for f in _HISTORY_FIELDS):
            continue
        if not _changed(previous.get(row["symbol"]), row):
            continue
        try:
            db.execute(
                """
                INSERT INTO fundamentals_history
                    (symbol, observed_at, eps_ttm, dps_annual,
                     book_value_per_share, loss_making, close_at_observation)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (row["symbol"], now, row.get("eps_ttm"), row.get("dps_annual"),
                 row.get("book_value_per_share"), row.get("loss_making"),
                 row.get("close")),
            )
            appended += 1
        except Exception:
            continue
    return appended


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


def get_pe_for_symbols(db, symbols: list) -> dict:
    """
    The same read for a whole batch, in ONE query. {SYMBOL: row}.

    Callers that score several symbols per request — the dashboard batch path
    and the snapshot cron — were issuing one indexed lookup per symbol inside
    their loop. Cheap individually, but on Neon each is a network round trip
    from a serverless container, and both of those callers run under a
    wall-clock deadline they can spend better on the actual fetches.

    Symbols with no stored fundamentals are simply absent, matching
    `get_pe_for_symbol` returning None: a missing value SKIPS its valuation
    band rather than defaulting, so a stock is never punished for the feed's
    silence.
    """
    if not symbols:
        return {}
    wanted = [s.upper() for s in symbols]
    rows = db.execute(
        "SELECT symbol, company_name, pe_ratio, dividend_yield, loss_making, "
        "updated_at FROM pe_data WHERE symbol = ANY(%s)",
        (wanted,),
    ).fetchall()
    return {
        r[0]: {
            "company_name": r[1],
            "pe_ratio": r[2],
            "dividend_yield": r[3],
            "loss_making": r[4],
            "fetched_at": r[5],
        }
        for r in rows
    }


def get_fundamentals_at(db, symbol: str, as_of: str) -> Optional[dict]:
    """
    The fundamentals in force for `symbol` on date `as_of` (ISO string).

    This is the whole reason `fundamentals_history` exists. Scoring a stock in
    the past with TODAY's P/E is look-ahead bias severe enough to manufacture
    any backtest result you like, so anything evaluating a historical date must
    read through here rather than through `get_pe_for_symbol`.

    Returns the latest observation at or before `as_of`, or None if we had not
    yet observed the symbol then. Ratios are NOT stored (they divide by price
    and would be mostly price noise) — derive them from the close on the date
    you are evaluating:  pe = close_then / eps_ttm.
    """
    try:
        row = db.execute(
            "SELECT observed_at, eps_ttm, dps_annual, book_value_per_share, "
            "loss_making FROM fundamentals_history "
            "WHERE symbol = %s AND observed_at <= %s "
            "ORDER BY observed_at DESC LIMIT 1",
            (symbol.upper(), as_of),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return {
        "observed_at": row[0],
        "eps_ttm": row[1],
        "dps_annual": row[2],
        "book_value_per_share": row[3],
        "loss_making": row[4],
    }


def get_fundamentals_asof_all(db, as_of: str) -> dict:
    """
    `get_fundamentals_at` for every symbol at once: {symbol: {...}}.

    A backtest asks this per rebalance date across the whole universe; one
    query per symbol would dominate its runtime.
    """
    try:
        rows = db.execute(
            "SELECT DISTINCT ON (symbol) symbol, observed_at, eps_ttm, "
            "dps_annual, book_value_per_share, loss_making "
            "FROM fundamentals_history WHERE observed_at <= %s "
            "ORDER BY symbol, observed_at DESC",
            (as_of,),
        ).fetchall()
    except Exception:
        return {}
    return {
        r[0]: {"observed_at": r[1], "eps_ttm": r[2], "dps_annual": r[3],
               "book_value_per_share": r[4], "loss_making": r[5]}
        for r in rows
    }


def _write_setting(db, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO settings (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )
