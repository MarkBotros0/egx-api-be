"""GET /api/markets — the enabled markets and their display fields."""
from datetime import date

from fastapi import APIRouter, Depends

from app.core.auth import CurrentUser, get_current_user
from app.core.markets import DEFAULT_CODE, MARKETS, enabled_codes, public_market

router = APIRouter()


@router.get("/api/markets")
def list_markets(user: CurrentUser = Depends(get_current_user)):
    today = date.today()
    return {
        "markets": [public_market(MARKETS[c], today) for c in enabled_codes()],
        "default": DEFAULT_CODE,
    }
