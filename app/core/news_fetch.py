"""
EGX news, from the one source a server-side fetch can actually reach.

WHY NOT egx.com.eg
------------------
The exchange sits behind an F5 bot challenge (APM_DO_NOT_TOUCH): a scripted GET
returns HTTP 200 with a JavaScript shell instead of content, and WebFetch gets
ECONNRESET. It renders only in a real browser. This is the same wall that made
the old MarketPECompanies.aspx P/E scraper never once succeed in production.

WHY NOT A SNAPSHOT TABLE
------------------------
pe_data and risk_snapshot are cached because their upstream refuses half the
universe at ~6s each. This source measured 24 symbols in 1.30s at 8 workers,
median 0.37s per symbol, 0 failures. A table, a cron, a secret and a staleness
story would all be machinery for a 1.3-second problem. If the deadline below
starts tripping routinely, that conclusion has expired — see the spec.

TWO TRAPS, both measured, both easy to fall into again:
  - `?market=egypt` is SILENTLY IGNORED. It returns 200 with the global stock
    feed: 200 items, zero EGX symbols, mostly Tesla and Santander. The market
    half is built by fanning out over EGX30, not by a market endpoint.
  - The news host sends NO Access-Control-Allow-Origin, so a browser cannot
    fetch it. (The scanner host does.) That is why this runs server-side.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from app.core.constants import (
    NEWS_DEADLINE_SECONDS,
    NEWS_FETCH_WORKERS,
    NEWS_MAX_ITEMS_PER_SYMBOL,
    NEWS_MAX_SYMBOLS,
    NEWS_RECENCY_DAYS,
    NEWS_REQUEST_TIMEOUT_SECONDS,
)
from app.core.tradingview import HEADERS

NEWS_URL = (
    "https://news-headlines.tradingview.com/v2/headlines"
    "?client=web&lang=en&symbol=EGX%3A{symbol}"
)

# The whole of a NewsItem. Deliberately excludes every body-ish field the
# upstream offers — see test_normalize_never_carries_article_body.
NEWS_ITEM_FIELDS = frozenset(
    {"id", "title", "provider", "published_at", "url", "symbols"}
)

STORY_BASE = "https://www.tradingview.com"


def _egx_symbols(raw: dict, fallback: str) -> list[str]:
    """EGX tickers a story is tagged with, bare and sorted. Never empty."""
    out = {fallback.upper()}
    for rel in raw.get("relatedSymbols") or []:
        if not isinstance(rel, dict):
            continue
        sym = (rel.get("symbol") or "")
        if sym.startswith("EGX:"):
            bare = sym.split(":", 1)[1].strip().upper()
            if bare:
                out.add(bare)
    return sorted(out)


def normalize_item(raw, symbol: str) -> Optional[dict]:
    """
    One upstream story -> a NewsItem, or None if it cannot be one.

    Returning None rather than a partial row is deliberate: a story with no
    timestamp cannot be placed in the recency window and one with no storyPath
    has nowhere to link, so either would render as a broken row.
    """
    if not isinstance(raw, dict):
        return None

    title = (raw.get("title") or "").strip()
    story_id = raw.get("id")
    path = raw.get("storyPath")
    if not title or not story_id or not path:
        return None

    try:
        ts = int(raw.get("published"))
    except (TypeError, ValueError):
        return None

    published_at = (
        datetime.fromtimestamp(ts, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    return {
        "id": str(story_id),
        "title": title,
        "provider": (raw.get("provider") or "").strip() or None,
        "published_at": published_at,
        "url": f"{STORY_BASE}{path}",
        "symbols": _egx_symbols(raw, symbol),
    }


def _parsed(item: dict) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError):
        return None


def is_recent(item: dict, now: datetime, days: int = NEWS_RECENCY_DAYS) -> bool:
    """
    Is this story inside the window that makes it news at all?

    `now` MUST be timezone-aware (UTC). `_parsed` always returns an aware
    datetime, so an aware/naive subtraction would raise TypeError — and that
    is the correct failure. Silently assuming UTC for a naive `now` would be
    guessing the caller's timezone, and a wrong guess is wrong by hours in a
    way nothing on screen would surface.
    """
    when = _parsed(item)
    if when is None:
        return False
    return (now - when) <= timedelta(days=days)


def dedupe_stories(items: list[dict]) -> list[dict]:
    """
    One row per story id, MERGING symbol tags, newest first.

    A story related to two of the user's holdings arrives twice, once per
    symbol fetched. Dropping the second copy would silently discard the fact
    that it concerns both.
    """
    merged: dict[str, dict] = {}
    for item in items:
        if not item:
            continue
        key = item["id"]
        if key in merged:
            both = set(merged[key]["symbols"]) | set(item["symbols"])
            merged[key]["symbols"] = sorted(both)
        else:
            merged[key] = dict(item, symbols=list(item["symbols"]))

    return sorted(
        merged.values(),
        key=lambda i: _parsed(i) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def _ordered_unique(symbols) -> list[str]:
    """Uppercase, blanks dropped, first occurrence wins, order preserved."""
    seen, out = set(), []
    for raw in symbols or []:
        sym = (raw or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def select_news_symbols(
    holdings, watchlist, market, cap: int = NEWS_MAX_SYMBOLS
) -> tuple[list[str], list[str], list[str]]:
    """
    Split the symbol budget into (yours, market_only, dropped).

    Priority is holdings, then watchlist, then the index. When the cap binds:
    - Index names (from market) are dropped first
    - Watchlist names are dropped second
    - Holdings are dropped only if holdings alone exceed the cap

    A symbol you own is removed from the market half so it is fetched once
    and rendered in one place. The third element (dropped) contains symbols
    from holdings+watchlist that were cut by the cap, in original order.
    """
    all_yours = _ordered_unique(list(holdings or []) + list(watchlist or []))
    yours = all_yours[:cap]
    dropped = all_yours[cap:]
    mine = set(yours)
    remaining = max(0, cap - len(yours))
    market_only = [s for s in _ordered_unique(market) if s not in mine][:remaining]
    return yours, market_only, dropped


def fetch_symbol_news(
    symbol: str, timeout: float = NEWS_REQUEST_TIMEOUT_SECONDS
) -> list[dict]:
    """One GET for one symbol. Returns the raw `items` array. Raises on error."""
    req = urllib.request.Request(
        NEWS_URL.format(symbol=symbol.upper()),
        headers={**HEADERS, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload.get("items") or []


def fetch_many(
    symbols: list[str],
    deadline_seconds: float = NEWS_DEADLINE_SECONDS,
    workers: int = NEWS_FETCH_WORKERS,
    fetcher: Optional[Callable] = None,
) -> dict[str, list[dict]]:
    """
    Fan out over `symbols`, returning {symbol: raw items} for whatever finished.

    A symbol that raises yields [] rather than taking the feed down. The
    deadline abandons stragglers — safe here precisely because nothing is
    persisted, so there is no half-written state to reconcile.

    `fetcher` is the test seam, the same shape as refresh_pe_data(db, rows=None).
    """
    fetch = fetcher or fetch_symbol_news
    out: dict[str, list[dict]] = {}
    if not symbols:
        return out

    deadline = time.monotonic() + deadline_seconds
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fetch, s): s for s in symbols}
        for future in as_completed(futures, timeout=max(0.01, deadline_seconds)):
            symbol = futures[future]
            try:
                out[symbol] = future.result(timeout=0) or []
            except Exception:
                out[symbol] = []
            if time.monotonic() >= deadline:
                break
    except Exception:
        # as_completed raises TimeoutError when the deadline expires with
        # futures outstanding. Whatever landed is the answer.
        pass
    finally:
        # wait=False + cancel_futures: the context-manager form would block on
        # shutdown and silently blow through the deadline we just enforced.
        pool.shutdown(wait=False, cancel_futures=True)

    return out


def build_feed(fetched: dict, yours: list[str], now: datetime, dropped=()) -> dict:
    """
    Raw per-symbol items -> {your_stocks, market, coverage}.

    `dropped` is the user's own symbols the symbol cap excluded (third element
    of select_news_symbols). Reported as coverage.symbols_over_cap so a
    dropped holding is visible rather than silently absent.

    A story is YOURS if any symbol it is tagged with is one you hold or watch,
    however it was fetched — a story reached via EGX:OCDI that also names COMI
    belongs in your section when you hold COMI, and in only one section.

    `coverage` describes YOUR symbols only. The index half is not counted: the
    user did not ask for those names and a "no news" tally against them would
    read as the app failing rather than as an absence of news.
    """
    mine = {s.upper() for s in yours}

    shaped = []
    with_news = set()
    for symbol, raw_items in (fetched or {}).items():
        kept = 0
        for raw in raw_items or []:
            item = normalize_item(raw, symbol)
            if item is None or not is_recent(item, now):
                continue
            shaped.append(item)
            kept += 1
            if kept >= NEWS_MAX_ITEMS_PER_SYMBOL:
                break
        if kept and symbol.upper() in mine:
            with_news.add(symbol.upper())

    stories = dedupe_stories(shaped)
    your_stories = [i for i in stories if mine & set(i["symbols"])]
    market_stories = [i for i in stories if not (mine & set(i["symbols"]))]

    return {
        "your_stocks": your_stories,
        "market": market_stories,
        "coverage": {
            "symbols_requested": len(mine),
            "symbols_with_news": len(with_news),
            "symbols_without_news": sorted(mine - with_news),
            "window_days": NEWS_RECENCY_DAYS,
            "symbols_over_cap": sorted(s.upper() for s in dropped),
        },
    }
