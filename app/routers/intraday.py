"""
GET /api/intraday — Fetch intraday price data for EGX stocks.
"""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import universe
from app.core.cache import get, set, make_key
from app.core.db import get_db_or_none
from app.core.markets import EG, MarketConfig, market_param

router = APIRouter()

VALID_INTERVALS = {"1 minute", "5 minute", "30 minute"}


def _parse_date(s: str, default: date) -> date:
    try:
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return default


@router.get("/api/intraday")
def get_intraday(
    symbols: str = Query(...),
    interval: str = Query("5 Minute"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    market: MarketConfig = Depends(market_param),
):
    try:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not syms:
            raise HTTPException(status_code=400, detail="No valid symbols provided")

        if interval.lower() not in VALID_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval: {interval}. Use '1 Minute', '5 Minute', or '30 Minute'.",
            )

        today = date.today()
        start_date = _parse_date(start or "", today - timedelta(days=5))
        end_date = _parse_date(end or "", today)

        # Each symbol on its own listing exchange; unresolved ones are dropped.
        # EG resolves without a DB, and keeps passing the plain "EGX" string.
        db = None if market.code == EG.code else get_db_or_none()
        ex_map = universe.exchange_map(db, market, syms)
        if not ex_map:
            raise HTTPException(status_code=404, detail=f"Unknown symbol: {','.join(syms)}")
        syms = [s for s in syms if s in ex_map]
        exchange = market.exchanges[0] if market.code == EG.code else ex_map

        cache_key = make_key(market.code, "intraday", ",".join(syms), interval, str(start_date), str(end_date))
        cached = get(cache_key)
        if cached:
            return cached

        from app.vendor.egxpy import get_EGX_intraday_data

        df = get_EGX_intraday_data(syms, interval, start_date, end_date,
                                   exchange=exchange)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No intraday data found")

        result = {
            "symbols": syms,
            "market": market.code,
            "currency": market.currency,
            "dates": [str(idx) for idx in df.index],
        }
        for sym in syms:
            if sym in df.columns:
                result[sym] = df[sym].tolist()

        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching intraday data: {str(e)}")
