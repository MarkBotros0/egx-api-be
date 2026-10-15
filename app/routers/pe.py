"""
/api/pe — fundamentals endpoints, backed by the nightly-refreshed pe_data table.

GET  /api/pe                  — All stored rows + freshness metadata
GET  /api/pe?symbol=XXX       — Single symbol (404 if no stored row)
POST /api/pe/refresh          — Trigger the refresh (cron-invoked; secret-guarded)

Rows carry trailing P/E, dividend yield and loss-making status. See
core/pe_fetch.py for the source and its null semantics.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.core.accounts import AccountStore
from app.core.db import get_db
from app.core.fundamentals_annual import refresh_annual_fundamentals
from app.core.limits import prune_auth
from app.core.macro_series import refresh_current as refresh_macro_series
from app.core.markets import EG, MarketConfig, market_param
from app.core.pe_fetch import _status_key, get_pe_for_symbol, refresh_pe_data
from app.core.universe import load_universe

router = APIRouter()


@router.get("/api/pe")
def get_pe(symbol: Optional[str] = Query(None),
           market: MarketConfig = Depends(market_param)):
    try:
        db = get_db()
        if symbol:
            data = get_pe_for_symbol(db, symbol, market.code)
            if not data:
                raise HTTPException(
                    status_code=404, detail=f"No P/E data for {symbol.upper()}"
                )
            return {"market": market.code, "currency": market.currency,
                    "symbol": symbol.upper(), **data}

        rows = db.execute(
            "SELECT symbol, company_name, pe_ratio, dividend_yield, loss_making, "
            "updated_at FROM pe_data WHERE market = %s",
            (market.code,),
        ).fetchall()
        last_row = db.execute(
            "SELECT value FROM settings WHERE key = %s",
            (_status_key(market, "pe_last_successful_fetch"),),
        ).fetchone()
        status_row = db.execute(
            "SELECT value FROM settings WHERE key = %s",
            (_status_key(market, "pe_last_attempt_status"),),
        ).fetchone()
        return {
            "market": market.code,
            "currency": market.currency,
            "data": [
                {
                    "symbol": r[0],
                    "company_name": r[1],
                    "pe_ratio": r[2],
                    "dividend_yield": r[3],
                    "loss_making": r[4],
                    "fetched_at": r[5],
                }
                for r in rows
            ],
            "last_successful_fetch": last_row[0] if last_row and last_row[0] else None,
            "last_attempt_status": status_row[0] if status_row and status_row[0] else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/pe/refresh")
def trigger_refresh(x_refresh_secret: Optional[str] = Header(default=None),
                    market: MarketConfig = Depends(market_param)):
    """
    Manual + cron-triggered refresh. When PE_REFRESH_SECRET env var is set,
    the X-Refresh-Secret header must match (used in production to prevent
    random callers from triggering EGX scrapes).

    `?market=` absent means EG, unchanged. Another market scans exactly its
    universe (`EXCHANGE:SYMBOL` from `load_universe`), writes `pe_last_*:<code>`
    status keys, and skips the EG-only macro and auth-prune steps.
    """
    expected = os.environ.get("PE_REFRESH_SECRET")
    if expected and x_refresh_secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    if market.code != EG.code:
        return _refresh_other_market(market)
    try:
        db = get_db()
        result = refresh_pe_data(db, EG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # The 20-year annual archive rides along on the same nightly slot: a second
    # POST to the same host, no new cron entry, no new vendor. It is what makes
    # a fundamental factor testable at all (see core/fundamentals_annual.py).
    #
    # Its failure must NOT fail this endpoint. `pe_data` is what the app
    # actually serves; the archive is for offline analysis, and taking the live
    # feed down with it would trade something the user sees for something only
    # a backtest reads.
    try:
        result["annual"] = refresh_annual_fundamentals(db, EG)
    except Exception as e:
        result["annual"] = {"success": False,
                            "error": f"{type(e).__name__}: {e}"}

    # Today's macro readings, one keyless POST. Same never-fail-the-feed rule:
    # pe_data is what the app serves. History is backfilled offline by
    # scripts/backfill_macro.py, because 5,000 FX bars do not fit in 30 seconds.
    try:
        result["macro"] = refresh_macro_series(db)
    except Exception as e:
        result["macro"] = {"success": False,
                           "error": f"{type(e).__name__}: {e}"}

    # Daily housekeeping for email accounts (spent links, old rate-limit
    # windows, sign-ups nobody ever used) rides the same nightly slot.
    # Best-effort: it must never fail the feed.
    try:
        result["auth_prune"] = prune_auth(AccountStore(db), datetime.now(timezone.utc))
    except Exception as e:
        print(f"[auth] prune failed: {type(e).__name__}: {e}")
    return result


def _refresh_other_market(market: MarketConfig) -> dict:
    """pe_data + the annual archive for a non-EG market, by its universe's tickers."""
    try:
        db = get_db()
        tickers = [f"{r['exchange']}:{r['symbol']}" for r in load_universe(db, market)]
        result = refresh_pe_data(db, market, tickers=tickers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Same never-fail-the-feed rule as EG.
    try:
        result["annual"] = refresh_annual_fundamentals(db, market, tickers=tickers)
    except Exception as e:
        result["annual"] = {"success": False,
                            "error": f"{type(e).__name__}: {e}"}
    return result
