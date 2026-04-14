"""
GET /api/ohlcv — Fetch OHLCV data for a single stock.
"""

from fastapi import APIRouter, HTTPException, Query
from app.core.cache import get, set, make_key

router = APIRouter()

VALID_INTERVALS = {"daily", "weekly", "monthly"}


@router.get("/api/ohlcv")
def get_ohlcv(
    symbol: str = Query(...),
    exchange: str = Query("EGX"),
    interval: str = Query("Daily"),
    bars: int = Query(100),
):
    try:
        symbol = symbol.upper()
        exchange = exchange.upper()
        interval = interval.capitalize()
        bars = min(max(bars, 1), 5000)

        if interval.lower() not in VALID_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval: {interval}. Use Daily, Weekly, or Monthly.",
            )

        cache_key = make_key("ohlcv", symbol, exchange, interval, bars)
        cached = get(cache_key)
        if cached:
            return cached

        from egxpy.download import get_OHLCV_data

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

        result = {"symbol": symbol, "interval": interval, "data": records}
        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching OHLCV data: {str(e)}")
