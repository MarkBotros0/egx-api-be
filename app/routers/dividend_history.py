"""
GET /api/dividend_history?symbol=XXX — one stock's dated, multi-year dividend
history (from the `dividend_events` table, self-healing from Yahoo) + a cadence
estimate.

GET /api/dividend_calendar — every payer's most-recent coupon (from the table),
for the all-stocks calendar view.

Named `dividend_history` to avoid the existing POST/DELETE /api/dividends ledger
routes. Both behind the auth gate (NOT in PUBLIC_ENDPOINTS); tests/test_auth_gate
verifies by enumeration.

THE STORE. Dividends are persisted in `dividend_events`, seeded deep from Yahoo
(scripts/backfill_dividends) and appended to nightly by the refresh (the scanner
coupon it already fetches). Reads come from the table. When the table has nothing
for a symbol yet, the history endpoint fetches Yahoo on demand AND upserts what it
got — so the store fills in as stocks are viewed, without waiting on the backfill.
Neither route ever 500s on an upstream hiccup.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import cache
from app.core.auth import CurrentUser, get_current_user
from app.core.constants import DIVIDEND_HISTORY_TTL_SECONDS
from app.core.db import get_db
from app.core.dividend_history import (
    fetch_dividends,
    read_calendar,
    read_dividends,
    summarize_cadence,
    upsert_dividends,
)
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

    db = get_db()
    try:
        dividends = read_dividends(db, sym)
    except Exception:
        dividends = []

    # Table empty for this symbol — fetch Yahoo once, store what we got, and
    # serve it. This is how the store fills in ahead of the backfill script.
    if not dividends:
        try:
            fetched = fetch_dividends(sym)
        except Exception:
            fetched = None
        if fetched is None:
            return {
                "symbol": sym,
                "dividends": [],
                "cadence": summarize_cadence([]),
                "status": "unavailable",
            }
        if fetched:
            try:
                upsert_dividends(db, sym, fetched, source="yahoo")
            except Exception:
                pass
        dividends = fetched

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
    db = get_db()

    # Prefer the persisted table (deep, seeded from Yahoo). Before it is seeded,
    # fall back to pe_data's latest-coupon-per-symbol so the calendar still paints.
    stocks = []
    try:
        events = read_calendar(db)
    except Exception:
        events = []

    if events:
        try:
            payers = {p["symbol"]: p for p in get_dividend_payers(db)}
        except Exception:
            payers = {}
        for e in events:
            p = payers.get(e["symbol"], {})
            stocks.append({
                "symbol": e["symbol"],
                "name": p.get("name"),
                "dividend_yield": p.get("dividend_yield"),
                "ex_date": e["ex_date"],
                "amount": e["amount"],
            })
    else:
        try:
            stocks = get_dividend_payers(db)
        except Exception:
            return {"stocks": [], "count": 0, "as_of": now, "status": "unavailable"}

    payload = {"stocks": stocks, "count": len(stocks), "as_of": now, "status": "ok"}
    cache.set(key, payload, ttl=DIVIDEND_HISTORY_TTL_SECONDS)
    return payload
