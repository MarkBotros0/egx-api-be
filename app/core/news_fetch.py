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

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.constants import NEWS_RECENCY_DAYS

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
