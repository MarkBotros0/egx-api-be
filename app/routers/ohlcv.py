"""
GET /api/ohlcv — Fetch OHLCV data for a single stock.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from app.core import universe
from app.core.cache import get, set, make_key
from app.core.db import get_db_or_none
from app.core.markets import EG, MarketConfig, market_param

router = APIRouter()

VALID_INTERVALS = {"daily", "weekly", "monthly"}


@router.get("/api/ohlcv")
def get_ohlcv(
    symbol: str = Query(...),
    # An explicit exchange beats the lookup. Absent, the symbol's own listing
    # exchange is resolved (a US stock may be on NYSE, NASDAQ or AMEX).
    exchange: Optional[str] = Query(None),
    interval: str = Query("Daily"),
    bars: int = Query(100),
    market: MarketConfig = Depends(market_param),
):
    try:
        symbol = symbol.upper()
        interval = interval.capitalize()
        bars = min(max(bars, 1), 5000)

        if interval.lower() not in VALID_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval: {interval}. Use Daily, Weekly, or Monthly.",
            )

        if exchange:
            exchange = exchange.upper()
        else:
            # EG resolves without a DB; never open one for it.
            db = None if market.code == EG.code else get_db_or_none()
            exchange = universe.resolve_exchange(db, market, symbol)
        if not exchange:
            raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")

        cache_key = make_key(market.code, "ohlcv", symbol, exchange, interval, bars)
        cached = get(cache_key)
        if cached:
            return cached

        from app.vendor.egxpy import get_OHLCV_data

        df = get_OHLCV_data(symbol, exchange, interval, bars)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        df.columns = [c.lower() for c in df.columns]

        records = []
        for idx, row in df.iterrows():
            records.append({
                "date": str(idx)[:10],
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            })

        result = {"symbol": symbol, "interval": interval, "data": records,
                  "market": market.code, "currency": market.currency}
        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching OHLCV data: {str(e)}")
