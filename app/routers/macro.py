"""
GET /api/macro — Fetch Egyptian macro indicators (EGX30, USD/EGP, CBE rate).
"""

from fastapi import APIRouter, HTTPException
from app.core.constants import DEFAULT_RISK_FREE_RATE_PCT
from app.core.currency import EGX30_TWENTY_YEAR
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
                # Was a hardcoded 25.0, which is the stale figure the rest of
                # the app was corrected away from. A fallback that contradicts
                # the constant is worse than no fallback.
                "interest_rate": {"value": float(DEFAULT_RISK_FREE_RATE_PCT),
                                  "direction": "stable"},
            }

        # Every figure this app shows is in EGP, and over the period it covers
        # the pound lost most of its value. Twenty years of "the market went up
        # 8x" is, in hard currency, twenty years of going nowhere. Shipped as
        # context beside the live rate so the user has a yardstick for their own
        # numbers. Both figures are true; they answer different questions.
        data["currency_context"] = EGX30_TWENTY_YEAR

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching macro data: {str(e)}")
