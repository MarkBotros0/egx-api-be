"""
Watchlist CRUD — Persist user's watched tickers in Turso.

GET    /api/watchlist              — List the current user's watched symbols
POST   /api/watchlist              — Add a symbol (body: {"symbol": "..."})
DELETE /api/watchlist?symbol=XXX   — Remove a symbol

Every route is scoped by the caller's user_id (from the JWT Bearer token).
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import CurrentUser, get_current_user
from app.core.db import get_db

router = APIRouter()


class WatchlistAdd(BaseModel):
    symbol: str


@router.get("/api/watchlist")
def get_watchlist(user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        rows = db.execute(
            "SELECT symbol FROM watchlist WHERE user_id = ? ORDER BY added_at ASC",
            (user.id,),
        ).fetchall()
        return {"symbols": [r[0] for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/watchlist", status_code=201)
def add_to_watchlist(
    body: WatchlistAdd,
    user: CurrentUser = Depends(get_current_user),
):
    try:
        symbol = body.symbol.strip().upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing required field: symbol")

        now = datetime.utcnow().isoformat() + "Z"
        db = get_db()
        db.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, symbol, added_at) VALUES (?, ?, ?)",
            (user.id, symbol, now),
        )
        db.commit()
        return {"symbol": symbol, "added_at": now}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/watchlist")
def remove_from_watchlist(
    symbol: str = Query(...),
    user: CurrentUser = Depends(get_current_user),
):
    try:
        symbol = symbol.strip().upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing required query parameter: symbol")

        db = get_db()
        db.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?",
            (user.id, symbol),
        )
        db.commit()
        return {"deleted": symbol}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
