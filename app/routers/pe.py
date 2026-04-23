"""
/api/pe — P/E ratio endpoints, backed by the nightly-scraped pe_data table.

GET  /api/pe                  — All stored P/E rows + freshness metadata
GET  /api/pe?symbol=XXX       — Single symbol (404 if no stored row)
POST /api/pe/refresh          — Trigger the scrape (cron-invoked; secret-guarded)
"""

import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from app.core.db import get_db
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
            "SELECT symbol, company_name, pe_ratio, dividend_yield, updated_at FROM pe_data"
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
                    "fetched_at": r[4],
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
        return refresh_pe_data(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
