"""
GET /api/news — recent stories for the caller's holdings and watchlist, with
market news below. Two sources: TradingView's wires (Reuters, Zawya, LSE),
tagged by the wire, and Arab Finance, tagged by headline.

Fetched on demand, nothing persisted. See core/news_fetch.py for why this is
not a snapshot table and why it cannot be done in the browser.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.core import cache
from app.core.markets import MarketConfig, market_param
from app.core.arabfinance import fetch_stories as fetch_arabfinance
from app.core.auth import CurrentUser, get_current_user
from app.core.constants import DEFAULT_CACHE_TTL_SECONDS
from app.core.db import get_db
from app.core.holdings import fetch_open_holdings
from app.core.index_membership import symbols_in_index
from app.core.news_fetch import (
    build_feed, feed_status, fetch_many, fetch_symbol_news, select_news_symbols,
)

router = APIRouter()


@router.get("/api/news")
def get_news(user: CurrentUser = Depends(get_current_user),
             market: MarketConfig = Depends(market_param)):
    now = datetime.now(timezone.utc)
    is_eg = market.code == "EG"

    try:
        db = get_db()
        holdings = [h["symbol"] for h in fetch_open_holdings(db, user.id, market.code)]
        watch = [
            r[0]
            for r in db.execute(
                "SELECT symbol FROM watchlist WHERE user_id = %s AND market = %s "
                "ORDER BY added_at ASC",
                (user.id, market.code),
            ).fetchall()
        ]
    except Exception:
        # The feed is worth showing even when the DB is unhappy — the market
        # half needs no user data at all.
        holdings, watch = [], []

    # The market half fans out over the EGX30 for EG. Other markets have no
    # index constituents wired yet (sub-project 4), so their half is empty.
    index_symbols = symbols_in_index("EGX30") if is_eg else []
    yours, market_half, dropped = select_news_symbols(holdings, watch, index_symbols)

    # `dropped` is baked into the cached payload (coverage.symbols_over_cap),
    # so it must be part of the key that addresses it — CLAUDE.md's "One Score
    # Per Stock" rule: any value in a cached response belongs in its cache
    # key. Two callers can share identical post-cap `yours`/`market` while
    # differing only in what the cap dropped (one holds exactly 40 symbols,
    # the other holds the same 40 plus a watchlist symbol that overflows) —
    # without `dropped` here they'd collide on one key and whoever populates
    # the cache first would silently decide the other's symbols_over_cap.
    key = cache.make_key(market.code, "news", ",".join(yours), ",".join(market_half), ",".join(dropped))
    hit = cache.get(key)
    if hit is not None:
        return hit

    requested = yours + market_half
    fetcher = partial(fetch_symbol_news, exchange=market.primary_exchange)
    # The two sources run side by side under the same deadline, so adding
    # Arab Finance costs no wall-clock time. Either one failing leaves the
    # other's stories standing. Arab Finance is Egyptian press: EG only.
    with ThreadPoolExecutor(max_workers=2) as pool:
        af_future = pool.submit(fetch_arabfinance) if is_eg else None
        fetched = fetch_many(requested, fetcher=fetcher)
        try:
            af_stories = af_future.result()[0] if af_future else []
        except Exception:
            af_stories = []

    feed = build_feed(fetched, yours=yours, now=now, dropped=dropped, extra=af_stories,
                      exchanges=market.exchanges)
    # "market" is already the market-news story list; the code goes beside it.
    feed["market_code"] = market.code
    feed["currency"] = market.currency
    feed["fetched_at"] = now.isoformat().replace("+00:00", "Z")
    feed["status"] = feed_status(fetched, requested, extra_count=len(af_stories))

    # An unavailable feed is a transient upstream problem, not a fact worth
    # holding for fifteen minutes — the same reasoning behind
    # ERROR_CACHE_TTL_SECONDS on the dashboard's refused symbols.
    if feed["status"] != "unavailable":
        cache.set(key, feed, ttl=DEFAULT_CACHE_TTL_SECONDS)

    return feed
