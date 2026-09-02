"""
/api/pe — fundamentals endpoints, backed by the nightly-refreshed pe_data table.

GET  /api/pe                  — All stored rows + freshness metadata
GET  /api/pe?symbol=XXX       — Single symbol (404 if no stored row)
POST /api/pe/refresh          — Trigger the refresh (cron-invoked; secret-guarded)

Rows carry trailing P/E, dividend yield and loss-making status. See
core/pe_fetch.py for the source and its null semantics.
"""

import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from app.core.db import get_db
from app.core.fundamentals_annual import refresh_annual_fundamentals
from app.core.pe_fetch import get_pe_for_symbol, refresh_pe_data

router = APIRouter()


@router.get("/api/pe")
def get_pe(symbol: Optional[str] = Query(None)):
    try:
        db = get_db()
        if symbol:
            data = get_pe_for_symbol(db, symbol)
            if not data:
                raise HTTPException(
                    status_code=404, detail=f"No P/E data for {symbol.upper()}"
                )
            return {"symbol": symbol.upper(), **data}

        rows = db.execute(
            "SELECT symbol, company_name, pe_ratio, dividend_yield, loss_making, "
            "updated_at FROM pe_data"
        ).fetchall()
        last_row = db.execute(
            "SELECT value FROM settings WHERE key = 'pe_last_successful_fetch'"
        ).fetchone()
        status_row = db.execute(
            "SELECT value FROM settings WHERE key = 'pe_last_attempt_status'"
        ).fetchone()
        return {
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
def trigger_refresh(x_refresh_secret: Optional[str] = Header(default=None)):
    """
    Manual + cron-triggered refresh. When PE_REFRESH_SECRET env var is set,
    the X-Refresh-Secret header must match (used in production to prevent
    random callers from triggering EGX scrapes).
    """
    expected = os.environ.get("PE_REFRESH_SECRET")
    if expected and x_refresh_secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        db = get_db()
        result = refresh_pe_data(db)
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
        result["annual"] = refresh_annual_fundamentals(db)
    except Exception as e:
        result["annual"] = {"success": False,
                            "error": f"{type(e).__name__}: {e}"}
    return result
