"""
Portfolio CRUD — Manage stock holdings via Turso.

GET    /api/portfolio          — List all holdings + settings
POST   /api/portfolio          — Add a new holding
PUT    /api/portfolio?id=xxx   — Update a holding
DELETE /api/portfolio?id=xxx   — Remove a holding
"""

import uuid
from datetime import datetime
from typing import Optional, Any

from fastapi import APIRouter, HTTPException, Query

from app.core.db import get_db

router = APIRouter()


def _row_to_dict(row):
    return {
        "id": row[0],
        "symbol": row[1],
        "name": row[2],
        "buy_price": row[3],
        "buy_date": row[4],
        "quantity": row[5],
        "notes": row[6],
        "sector": row[7],
        "target_price": row[8],
        "stop_loss": row[9],
        "created_at": row[10],
        "updated_at": row[11],
    }


@router.get("/api/portfolio")
def get_portfolio():
    try:
        db = get_db()
        rows = db.execute(
            "SELECT id, symbol, name, buy_price, buy_date, quantity, notes, sector, "
            "target_price, stop_loss, created_at, updated_at FROM portfolio"
        ).fetchall()
        holdings = [_row_to_dict(r) for r in rows]

        settings = db.execute(
            "SELECT key, value FROM settings WHERE key = 'currency'"
        ).fetchone()
        currency = settings[0] if settings else "EGP"

        return {"portfolio": holdings, "cash_available": 0, "currency": currency}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolio", status_code=201)
def add_holding(body: dict):
    try:
        symbol = body.get("symbol", "").upper()
        if not symbol:
            raise HTTPException(status_code=400, detail="Missing required field: symbol")

        buy_price = body.get("buy_price")
        if buy_price is None:
            raise HTTPException(status_code=400, detail="Missing required field: buy_price")

        quantity = body.get("quantity")
        if quantity is None:
            raise HTTPException(status_code=400, detail="Missing required field: quantity")

        now = datetime.utcnow().isoformat() + "Z"
        holding_id = str(uuid.uuid4())
        name = body.get("name", symbol)
        buy_date = body.get("buy_date", now[:10])
        notes = body.get("notes", "")
        sector = body.get("sector", "")
        target_price = float(body["target_price"]) if body.get("target_price") else None
        stop_loss = float(body["stop_loss"]) if body.get("stop_loss") else None

        db = get_db()
        db.execute(
            """INSERT INTO portfolio (id, symbol, name, buy_price, buy_date, quantity, notes,
               sector, target_price, stop_loss, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (holding_id, symbol, name, float(buy_price), buy_date, int(quantity),
             notes, sector, target_price, stop_loss, now, now),
        )
        db.commit()

        return {
            "id": holding_id,
            "symbol": symbol,
            "name": name,
            "buy_price": float(buy_price),
            "buy_date": buy_date,
            "quantity": int(quantity),
            "notes": notes,
            "sector": sector,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "created_at": now,
            "updated_at": now,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/portfolio")
def update_holding(id: str = Query(...), body: dict = None):
    if body is None:
        body = {}
    try:
        db = get_db()
        row = db.execute(
            "SELECT id, symbol, name, buy_price, buy_date, quantity, notes, sector, "
            "target_price, stop_loss, created_at, updated_at FROM portfolio WHERE id = ?",
            (id,),
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Holding not found: {id}")

        updatable = ["symbol", "name", "buy_price", "buy_date", "quantity", "notes",
                     "sector", "target_price", "stop_loss"]
        set_clauses = []
        values = []

        for key in updatable:
            if key in body:
                val = body[key]
                if key == "symbol" and val:
                    val = val.upper()
                if key in ("buy_price", "target_price", "stop_loss") and val is not None:
                    val = float(val)
                if key == "quantity" and val is not None:
                    val = int(val)
                set_clauses.append(f"{key} = ?")
                values.append(val)

        now = datetime.utcnow().isoformat() + "Z"
        set_clauses.append("updated_at = ?")
        values.append(now)
        values.append(id)

        db.execute(f"UPDATE portfolio SET {', '.join(set_clauses)} WHERE id = ?", values)
        db.commit()

        updated = db.execute(
            "SELECT id, symbol, name, buy_price, buy_date, quantity, notes, sector, "
            "target_price, stop_loss, created_at, updated_at FROM portfolio WHERE id = ?",
            (id,),
        ).fetchone()

        return _row_to_dict(updated)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/portfolio")
def delete_holding(id: str = Query(...)):
    try:
        db = get_db()
        row = db.execute(
            "SELECT id FROM portfolio WHERE id = ?", (id,)
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Holding not found: {id}")

        db.execute("DELETE FROM portfolio WHERE id = ?", (id,))
        db.commit()
        return {"deleted": id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
