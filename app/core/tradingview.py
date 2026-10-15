"""
Shared TradingView scanner client.

The scanner is the same public endpoint the ticker list already uses
(`routers/tickers.py::_fetch_live_tickers`). It is defined once here so the
two callers cannot drift on URL, headers or timeout — and so there is one
place to change if TradingView ever moves it.

It is a bulk endpoint: one POST returns every EGX-listed row, so a caller
needing data for 300 symbols makes one request, not 300.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional

from app.core.markets import EG, MarketConfig

SCAN_URL_TEMPLATE = "https://scanner.tradingview.com/{region}/scan"
SCAN_URL = SCAN_URL_TEMPLATE.format(region=EG.scanner_region)  # kept for existing importers

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}

DEFAULT_TIMEOUT_SECONDS = 20


def scan_url(market: MarketConfig) -> str:
    return SCAN_URL_TEMPLATE.format(region=market.scanner_region)


def _exchange_filter(market: MarketConfig) -> dict:
    """The one spelling of the scanner's exchange filter (also used by tickers.py)."""
    if len(market.exchanges) == 1:
        return {"left": "exchange", "operation": "equal", "right": market.exchanges[0]}
    return {"left": "exchange", "operation": "in_range", "right": list(market.exchanges)}


def scan(columns: list, market: MarketConfig = EG,
         timeout: int = DEFAULT_TIMEOUT_SECONDS, limit: int = 500, *,
         tickers: Optional[list] = None, symbolset: Optional[str] = None,
         extra_filter: Optional[list] = None, sort_by: Optional[str] = None) -> list:
    """
    Request `columns` from the market's scanner.

    Default: every common stock on the market's exchanges (unchanged).
    `symbolset`: an index's members (e.g. "SYML:SP;SPX"), no filter.
    `tickers`: exactly these "EXCHANGE:SYMBOL" listings, no filter.
    `extra_filter`: appended to the exchange filter; widens type to stock+fund
    (name search must find ETFs). `sort_by`: descending sort column.

    Returns the raw `data` array: [{"s": "EGX:COMI", "d": [...]}, ...] with
    `d` positionally aligned to `columns`. Raises on HTTP/JSON error — callers
    own their own failure policy.
    """
    body: dict = {"columns": columns, "range": [0, limit]}
    if tickers is not None:
        body["symbols"] = {"tickers": list(tickers)}
    elif symbolset is not None:
        body["symbols"] = {"symbolset": [symbolset]}
    else:
        type_filter = ({"left": "type", "operation": "in_range", "right": ["stock", "fund"]}
                       if extra_filter else
                       {"left": "type", "operation": "equal", "right": "stock"})
        body["filter"] = [_exchange_filter(market), type_filter, *(extra_filter or [])]
    if sort_by:
        body["sort"] = {"sortBy": sort_by, "sortOrder": "desc"}

    req = urllib.request.Request(scan_url(market), data=json.dumps(body).encode("utf-8"),
                                 headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())

    return payload.get("data") or []
