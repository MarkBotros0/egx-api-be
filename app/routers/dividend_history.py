"""
GET /api/dividend_history?symbol=XXX — one stock's dated, multi-year dividend
history (from Yahoo, on demand), plus a cadence estimate.

GET /api/dividend_calendar — every EGX dividend payer's most-recent coupon
(from pe_data), for the all-stocks calendar view.

Named `dividend_history` to avoid the existing POST/DELETE /api/dividends
ledger routes. Both are behind the auth gate (NOT in PUBLIC_ENDPOINTS); the app
is closed and tests/test_auth_gate.py verifies it by enumeration.

Neither ever 500s on an upstream hiccup — an empty payload with a status, the
same degrade-don't-break posture as /api/macro and /api/news.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import cache
from app.core.auth import CurrentUser, get_current_user
from app.core.constants import DIVIDEND_HISTORY_TTL_SECONDS
from app.core.db import get_db
from app.core.dividend_history import fetch_dividends, summarize_cadence
from app.core.pe_fetch import get_dividend_payers

router = APIRouter()


@router.get("/api/dividend_history")
def get_dividend_history(
    symbol: str = Query(...),
    user: CurrentUser = Depends(get_current_user),
):
    sym = symbol.strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="Missing required query parameter: symbol")

    key = cache.make_key("divhist", sym)
    hit = cache.get(key)
    if hit is not None:
        return hit

    try:
        dividends = fetch_dividends(sym)
    except Exception:
        # Yahoo is a public web endpoint that can hiccup; a dividend card that
        # can't load must not take the page down. Not cached — the failure is
        # transient and we want the next visit to retry.
        return {
            "symbol": sym,
            "dividends": [],
            "cadence": summarize_cadence([]),
            "status": "unavailable",
        }

    payload = {
        "symbol": sym,
        "dividends": dividends,
        "cadence": summarize_cadence(dividends),
        "status": "ok",
    }
    cache.set(key, payload, ttl=DIVIDEND_HISTORY_TTL_SECONDS)
    return payload


@router.get("/api/dividend_calendar")
def get_dividend_calendar(user: CurrentUser = Depends(get_current_user)):
    key = cache.make_key("divcal")
    hit = cache.get(key)
    if hit is not None:
        return hit

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        stocks = get_dividend_payers(get_db())
    except Exception:
        return {"stocks": [], "count": 0, "as_of": now, "status": "unavailable"}

    payload = {"stocks": stocks, "count": len(stocks), "as_of": now, "status": "ok"}
    cache.set(key, payload, ttl=DIVIDEND_HISTORY_TTL_SECONDS)
    return payload
