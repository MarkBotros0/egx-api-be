"""
Watchlist CRUD — Persist user's watched tickers in Postgres.

GET    /api/watchlist?market=XX              — List the user's watched symbols in a market
POST   /api/watchlist                        — Add a symbol (body: {"symbol": "...", "market"?: "XX"})
DELETE /api/watchlist?symbol=XXX&market=XX   — Remove a symbol

Every route is scoped by the caller's user_id (from the JWT Bearer token) and
by market (absent means EG). A body `market` beats the query one.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import CurrentUser, get_current_user
from app.core.db import get_db
from app.core.markets import MarketConfig, get_market, market_param

router = APIRouter()


class WatchlistAdd(BaseModel):
    symbol: str
    market: Optional[str] = None


@router.get("/api/watchlist")
def get_watchlist(market: MarketConfig = Depends(market_param),
                  user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        rows = db.execute(
            "SELECT symbol FROM watchlist WHERE user_id = %s AND market = %s "
            "ORDER BY added_at ASC",
            (user.id, market.code),
        ).fetchall()
        return {"symbols": [r[0] for r in rows], "market": market.code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/watchlist", status_code=201)
def add_to_watchlist(
    body: WatchlistAdd,
    market: MarketConfig = Depends(market_param),
    user: CurrentUser = Depends(get_current_user),
):
    try:
        target = get_market(body.market) if body.market else market
        symbol = body.symbol.strip().upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing required field: symbol")

        now = datetime.utcnow().isoformat() + "Z"
        db = get_db()
        db.execute(
            "INSERT INTO watchlist (user_id, market, symbol, added_at) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (user_id, market, symbol) DO NOTHING",
            (user.id, target.code, symbol, now),
        )
        db.commit()
        return {"symbol": symbol, "added_at": now, "market": target.code}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/watchlist")
def remove_from_watchlist(
    symbol: str = Query(...),
    market: MarketConfig = Depends(market_param),
    user: CurrentUser = Depends(get_current_user),
):
    try:
        symbol = symbol.strip().upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing required query parameter: symbol")

        db = get_db()
        db.execute(
            "DELETE FROM watchlist WHERE user_id = %s AND market = %s AND symbol = %s",
            (user.id, market.code, symbol),
        )
        db.commit()
        return {"deleted": symbol}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
