"""
GET /api/news — recent stories for the caller's holdings and watchlist, with
EGX30 market news below.

Fetched on demand, nothing persisted. See core/news_fetch.py for why this is
not a snapshot table and why it cannot be done in the browser.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.core import cache
from app.core.auth import CurrentUser, get_current_user
from app.core.constants import DEFAULT_CACHE_TTL_SECONDS
from app.core.db import get_db
from app.core.holdings import fetch_open_holdings
from app.core.index_membership import symbols_in_index
from app.core.news_fetch import build_feed, fetch_many, select_news_symbols

router = APIRouter()


@router.get("/api/news")
def get_news(user: CurrentUser = Depends(get_current_user)):
    now = datetime.now(timezone.utc)

    try:
        db = get_db()
        holdings = [h["symbol"] for h in fetch_open_holdings(db, user.id)]
        watch = [
            r[0]
            for r in db.execute(
                "SELECT symbol FROM watchlist WHERE user_id = %s ORDER BY added_at ASC",
                (user.id,),
            ).fetchall()
        ]
    except Exception:
        # The feed is worth showing even when the DB is unhappy — the market
        # half needs no user data at all.
        holdings, watch = [], []

    yours, market, dropped = select_news_symbols(holdings, watch, symbols_in_index("EGX30"))

    key = cache.make_key("news", ",".join(yours), ",".join(market))
    hit = cache.get(key)
    if hit is not None:
        return hit

    requested = yours + market
    fetched = fetch_many(requested)

    feed = build_feed(fetched, yours=yours, now=now, dropped=dropped)
    feed["fetched_at"] = now.isoformat().replace("+00:00", "Z")

    if not fetched:
        feed["status"] = "unavailable"
    elif len(fetched) < len(requested):
        feed["status"] = "partial"
    else:
        feed["status"] = "ok"

    # An unavailable feed is a transient upstream problem, not a fact worth
    # holding for fifteen minutes — the same reasoning behind
    # ERROR_CACHE_TTL_SECONDS on the dashboard's refused symbols.
    if feed["status"] != "unavailable":
        cache.set(key, feed, ttl=DEFAULT_CACHE_TTL_SECONDS)

    return feed
