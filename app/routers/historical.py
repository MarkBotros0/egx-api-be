"""
GET /api/historical — Fetch historical close prices for multiple EGX stocks.
"""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.cache import get, set, make_key

router = APIRouter()


def _parse_date(s: str, default: date) -> date:
    try:
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return default


@router.get("/api/historical")
def get_historical(
    symbols: str = Query(...),
    interval: str = Query("Daily"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    try:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not syms:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        if len(syms) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed")

        interval = interval.capitalize()
        today = date.today()
        start_date = _parse_date(start or "", today - timedelta(days=365))
        end_date = _parse_date(end or "", today)

        cache_key = make_key("historical", ",".join(syms), interval, str(start_date), str(end_date))
        cached = get(cache_key)
        if cached:
            return cached

        from egxpy.download import get_EGXdata

        df = get_EGXdata(syms, interval, start_date, end_date)

        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail="No data found for the given symbols and date range",
            )

        result = {
            "symbols": syms,
            "dates": [str(idx)[:10] for idx in df.index],
        }
        for sym in syms:
            if sym in df.columns:
                result[sym] = df[sym].tolist()

        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")
