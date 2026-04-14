"""
GET /api/compare — Compare performance of multiple EGX stocks.
"""

from datetime import date, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.core.cache import get, set, make_key

router = APIRouter()


def _parse_date(s: str, default: date) -> date:
    try:
        parts = s.split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return default


def _max_drawdown(prices: list) -> float:
    if not prices or len(prices) < 2:
        return 0.0
    arr = np.array(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    peak = arr[0]
    max_dd = 0.0
    for price in arr[1:]:
        if price > peak:
            peak = price
        dd = (price - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    return round(float(max_dd), 2)


@router.get("/api/compare")
def get_compare(
    symbols: str = Query(...),
    interval: str = Query("Daily"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    try:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if len(syms) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 symbols to compare")
        if len(syms) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols for comparison")

        interval = interval.capitalize()
        today = date.today()
        start_date = _parse_date(start or "", today - timedelta(days=180))
        end_date = _parse_date(end or "", today)

        cache_key = make_key("compare", ",".join(syms), interval, str(start_date), str(end_date))
        cached = get(cache_key)
        if cached:
            return cached

        from egxpy.download import get_EGXdata

        df = get_EGXdata(syms, interval, start_date, end_date)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data found for comparison")

        dates = [str(idx)[:10] for idx in df.index]
        result = {"symbols": syms, "dates": dates, "stats": {}}

        for sym in syms:
            if sym not in df.columns:
                continue

            prices = df[sym].tolist()
            first_valid = next((p for p in prices if p is not None and not np.isnan(p)), None)

            if first_valid and first_valid != 0:
                normalized = [
                    (p / first_valid - 1) * 100 if p is not None and not np.isnan(p) else None
                    for p in prices
                ]
            else:
                normalized = [0.0] * len(prices)

            result[sym] = normalized

            valid_prices = [p for p in prices if p is not None and not np.isnan(p)]
            if len(valid_prices) >= 2:
                total_return = (valid_prices[-1] / valid_prices[0] - 1) * 100
                returns = np.diff(valid_prices) / valid_prices[:-1]
                vol = float(np.std(returns)) if len(returns) > 0 else 0
                result["stats"][sym] = {
                    "total_return": round(total_return, 2),
                    "volatility": round(vol, 4),
                    "max_drawdown": _max_drawdown(valid_prices),
                }

        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing comparison: {str(e)}")
