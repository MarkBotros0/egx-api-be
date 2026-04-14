"""
GET /api/macro — Fetch Egyptian macro indicators (EGX30, USD/EGP, CBE rate).
"""

from fastapi import APIRouter, HTTPException
from app.core.db import get_db
from app.core.macro_fetch import fetch_macro

router = APIRouter()


@router.get("/api/macro")
def get_macro():
    try:
        db = get_db()
        data = fetch_macro(db)

        if data is None:
            data = {
                "egx30": {"value": None, "change_pct": None, "direction": None},
                "usd_egp": {"value": None, "change_pct": None, "direction": None},
                "interest_rate": {"value": 25.0, "direction": "stable"},
            }

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching macro data: {str(e)}")
