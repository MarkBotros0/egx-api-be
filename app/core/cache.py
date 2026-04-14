"""
Simple in-memory cache with TTL for serverless functions.

Each warm container keeps its own cache. On cold start the cache is empty.
Default TTL is 5 minutes — prevents hitting TradingView too often for the
same data within a short window.
"""

import time
from typing import Any, Optional

_DEFAULT_TTL = 300  # 5 minutes in seconds

# Module-level dict: survives across requests in the same warm container
_store: dict[str, tuple[float, Any]] = {}


def make_key(*args) -> str:
    """Build a deterministic cache key from arguments."""
    return ":".join(str(a) for a in args)


def get(key: str) -> Optional[Any]:
    """Return cached value if it exists and hasn't expired, else None."""
    entry = _store.get(key)
    if entry is None:
        return None
    timestamp, value = entry
    if time.time() - timestamp > _DEFAULT_TTL:
        del _store[key]
        return None
    return value


def set(key: str, value: Any) -> None:
    """Store a value with the current timestamp."""
    _store[key] = (time.time(), value)
