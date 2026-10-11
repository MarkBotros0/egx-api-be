"""
Arab Finance — the second news source, beside TradingView's wires.

WHAT IT IS
----------
arabfinance.com publishes ten free ENGLISH RSS feeds, one per sector
(`/en/rss/rssbycat/1..10`), each the latest 50 headlines. Measured 2026-10-11:
all ten 200 in well under a second, 495 unique stories, newest same day.

Two properties shape everything below:

  - **No stock tags.** Unlike TradingView's `relatedSymbols`, a story names
    no ticker, so `news_tagging.tag_symbols` matches the headline against
    company names. An untagged story is market news at best.
  - **The date is `a10:updated`, not `pubDate`** (Atom-namespaced, Cairo
    offset, e.g. `2026-10-10T12:16:00+03:00`). A story without it cannot be
    placed in the recency window and is dropped, as `normalize_item` drops a
    TradingView story with no timestamp.

WHY NOT AL BORSA NEWS (English)
-------------------------------
alborsanews.com sits behind a Cloudflare managed challenge (`cf-mitigated:
challenge`): every scripted request, feed and wp-json included, is a 403. The
same wall as egx.com.eg. Do not try to get around it.

Headline, provider, date, url, symbols — never the description. The feed's
<description> is the headline repeated today, but it is article copy by
contract, and this app links out rather than reproducing publishers' text.
"""

from __future__ import annotations

import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime, timezone
from html import unescape
from typing import Callable, Optional
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import fromstring as safe_fromstring

from app.core.constants import NEWS_DEADLINE_SECONDS, NEWS_REQUEST_TIMEOUT_SECONDS
from app.core.news_tagging import tag_symbols
from app.core.tradingview import HEADERS

PROVIDER = "arab-finance"
FEED_URL = "https://www.arabfinance.com/en/rss/rssbycat/{category}"
STORY_URL = "https://www.arabfinance.com/en/news/newdetails/{story_id}"

# The ten sector feeds, as the site names them.
CATEGORIES = {
    1: "Macroeconomy",
    2: "Finance and Banking",
    3: "Real Estate and Tourism",
    4: "Healthcare",
    5: "Energy",
    6: "Technology and Telecom",
    7: "Automotive and Transportation",
    8: "Industries",
    9: "Retail and Consumables",
    10: "Education",
}

# An untagged story still counts as market news when it is about the economy
# as a whole. Only the Macroeconomy feed qualifies by category; a Tourism or
# Education story that names no listed company is not stock-market news.
MARKET_CATEGORIES = frozenset({1})

_ATOM_UPDATED = "{http://www.w3.org/2005/Atom}updated"


def _text(el: Optional[Element]) -> str:
    return unescape((el.text or "") if el is not None else "").strip()


def _published(raw: str) -> Optional[str]:
    """`2026-10-10T12:16:00+03:00` -> `2026-10-10T09:16:00Z`, or None."""
    try:
        when = datetime.fromisoformat(raw.strip())
    except (ValueError, AttributeError):
        return None
    if when.tzinfo is None:
        # The feed has always carried an offset. A naive one would be a guess
        # about Cairo vs UTC — wrong by hours in a way nothing would surface.
        return None
    return when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_feed(xml_bytes: bytes, category: int) -> list[dict]:
    """
    One category's RSS -> stories shaped like NewsItem, plus `market_news`.

    A row with no id, title or usable date is dropped rather than half-kept.
    Raises on XML that does not parse — the caller treats that feed as empty.
    """
    # defusedxml: this is a third party's XML, so entity expansion and
    # external references are refused rather than trusted.
    root = safe_fromstring(xml_bytes)
    out = []
    for item in root.iter("item"):
        story_id = _text(item.find("guid"))
        title = " ".join(_text(item.find("title")).split())
        published_at = _published(_text(item.find(_ATOM_UPDATED)))
        if not story_id or not title or not published_at:
            continue
        symbols = tag_symbols(title)
        out.append(
            {
                "id": f"af:{story_id}",
                "title": title,
                "provider": PROVIDER,
                "published_at": published_at,
                # Always our own https URL built from the id, never the feed's
                # <link>, which is plain http.
                "url": STORY_URL.format(story_id=story_id),
                "symbols": symbols,
                # Tagged with a listed company, or from the Macroeconomy feed.
                # build_feed drops every other story.
                "market_news": bool(symbols) or category in MARKET_CATEGORIES,
            }
        )
    return out


def fetch_category(category: int, timeout: float = NEWS_REQUEST_TIMEOUT_SECONDS) -> bytes:
    req = urllib.request.Request(
        FEED_URL.format(category=category),
        headers={**HEADERS, "Accept": "application/rss+xml, application/xml"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_stories(
    deadline_seconds: float = NEWS_DEADLINE_SECONDS,
    fetcher: Optional[Callable[[int], bytes]] = None,
) -> tuple[list[dict], int]:
    """
    All ten feeds in parallel -> (stories, feeds_ok).

    A feed that raises or fails to parse contributes nothing; the deadline
    abandons stragglers. Same contract as news_fetch.fetch_many: nothing is
    persisted, so a partial answer is safe. `fetcher` is the test seam.
    """
    fetch = fetcher or fetch_category
    stories: list[dict] = []
    ok = 0
    deadline = time.monotonic() + deadline_seconds
    pool = ThreadPoolExecutor(max_workers=len(CATEGORIES))
    try:
        futures = {pool.submit(fetch, c): c for c in CATEGORIES}
        for future in as_completed(futures, timeout=max(0.01, deadline_seconds)):
            try:
                stories.extend(parse_feed(future.result(timeout=0), futures[future]))
                ok += 1
            except Exception:
                pass
            if time.monotonic() >= deadline:
                break
    except FuturesTimeoutError:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return merge_duplicates(stories), ok


def merge_duplicates(stories: list[dict]) -> list[dict]:
    """
    One row per story id. A story filed under two sector feeds arrives twice;
    it is market news if EITHER copy says so (the Macroeconomy copy of a
    banking story must not be undone by the Finance copy).
    """
    merged: dict[str, dict] = {}
    for story in stories:
        seen = merged.get(story["id"])
        if seen is None:
            merged[story["id"]] = dict(story)
        else:
            seen["market_news"] = seen["market_news"] or story["market_news"]
    return list(merged.values())

