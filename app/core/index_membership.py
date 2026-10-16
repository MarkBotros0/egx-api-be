"""
Static index-tier lookup (EGX30 / EGX70 / EGX100 / NILEX) for the scoring path.

WHY THIS IS NOT `tickers._load_tickers()`
-----------------------------------------
`liquidity_score` compares a stock's volume against floors that differ ~100x
between EGX30 and NILEX, so it needs to know which index a symbol belongs to.
That information lives in `data/egx_tickers.json`, which `routers/tickers.py`
already reads — but `_load_tickers()` also merges a live TradingView POST
(10 s timeout) and a DB query, and it does so on the first call in a cold
container. Calling it from the scoring path would put every dashboard card
behind a ticker-list fetch.

This module reads the static JSON directly instead: one file read per process,
no network, no DB. Index membership changes roughly twice a year and the JSON
is a deploy artifact, so there is nothing to refresh at runtime.

Unknown symbols return None, which is the SAFE answer — `liquidity_score`
already falls back to EGX100 floors for an unrecognised tier, and that is
what every symbol got before this module existed.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from app.core.markets import EG

_STATIC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "egx_tickers.json"
)

# Only tiers that liquidity_score has floors for. A symbol tagged with anything
# else (the tickers router tags TradingView-only discoveries as plain "EGX")
# is treated as unknown rather than silently mapped onto the wrong floors.
_KNOWN_TIERS = {"EGX30", "EGX70", "EGX100", "NILEX"}

_MEMBERSHIP: Optional[dict] = None


def _load() -> dict:
    """{SYMBOL: TIER} from the static ticker file. Empty dict if unreadable."""
    global _MEMBERSHIP
    if _MEMBERSHIP is not None:
        return _MEMBERSHIP

    out = {}
    try:
        with open(_STATIC_PATH, "r", encoding="utf-8") as f:
            for t in json.load(f):
                symbol = (t.get("symbol") or "").upper()
                tier = (t.get("index") or "").upper()
                if symbol and tier in _KNOWN_TIERS:
                    out[symbol] = tier
    except Exception:
        out = {}

    _MEMBERSHIP = out
    return _MEMBERSHIP


def get_index_membership(symbol: str, market=EG) -> Optional[str]:
    """
    Return "EGX30" | "EGX70" | "EGX100" | "NILEX" for an EG `symbol`, else None.

    None means "unknown tier", not "no index" — callers should pass it straight
    through to `liquidity_score`, which applies its EGX100 default.

    Any other market returns its own code ("US"), never an EGX tier: a US
    ticker that happens to share a symbol with an EGX listing must not borrow
    that listing's liquidity floors. `liquidity_score` has a floor row per
    market code.
    """
    if market.code != EG.code:
        return market.code
    if not symbol:
        return None
    return _load().get(symbol.upper())


def symbols_in_index(tier: str) -> list[str]:
    """
    Every symbol in `tier` ("EGX30" | "EGX70" | "EGX100" | "NILEX"), sorted.

    Unknown tiers return [] rather than everything: callers fan out over this
    list, and a typo that silently meant "the whole universe" would turn one
    bad string into 166 HTTP requests.

    Sorted so a cache key built from the result is stable across processes.
    """
    if not tier:
        return []
    want = tier.upper()
    return sorted(s for s, t in _load().items() if t == want)


def index_member_tier(market) -> str:
    """
    Liquidity tier for a member of one of `market`'s index symbolsets. For US
    that is any SPX / NDX / DJI member — all large caps — hence "US large-cap".
    Other listings of that market use the market code ("US") as their tier.
    """
    return f"{market.short_name} large-cap"
