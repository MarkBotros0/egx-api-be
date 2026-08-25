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

SCAN_URL = "https://scanner.tradingview.com/egypt/scan"

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


def scan(columns: list, timeout: int = DEFAULT_TIMEOUT_SECONDS,
         limit: int = 500) -> list:
    """
    Request `columns` for every EGX common stock.

    Returns the raw `data` array: [{"s": "EGX:COMI", "d": [...]}, ...] with
    `d` positionally aligned to `columns`. Raises on HTTP/JSON error — callers
    own their own failure policy.
    """
    body = json.dumps({
        "columns": columns,
        "filter": [
            {"left": "exchange", "operation": "equal", "right": "EGX"},
            {"left": "type", "operation": "equal", "right": "stock"},
        ],
        "range": [0, limit],
    }).encode("utf-8")

    req = urllib.request.Request(SCAN_URL, data=body, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())

    return payload.get("data") or []
