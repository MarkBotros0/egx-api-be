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


def get_index_membership(symbol: str) -> Optional[str]:
    """
    Return "EGX30" | "EGX70" | "EGX100" | "NILEX" for `symbol`, else None.

    None means "unknown tier", not "no index" — callers should pass it straight
    through to `liquidity_score`, which applies its EGX100 default.
    """
    if not symbol:
        return None
    return _load().get(symbol.upper())
