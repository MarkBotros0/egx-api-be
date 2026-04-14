"""
GET /api/intraday — Fetch intraday price data for EGX stocks.
"""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.cache import get, set, make_key

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

        cache_key = make_key("intraday", ",".join(syms), interval, str(start_date), str(end_date))
        cached = get(cache_key)
        if cached:
            return cached

        from egxpy.download import get_EGX_intraday_data

        df = get_EGX_intraday_data(syms, interval, start_date, end_date)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No intraday data found")

        result = {
            "symbols": syms,
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
