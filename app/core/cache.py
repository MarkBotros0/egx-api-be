"""
Simple in-memory cache with TTL for serverless functions.

Each warm container keeps its own cache. On cold start the cache is empty.
Default TTL is 15 minutes — keeps dashboard/composite scores warm long
enough that repeat visits within a session skip the slow upstream fetch.
EGX trades Sun-Thu 10:00-14:30 Cairo time, so intraday data changes
slowly relative to this window.
"""

import time
from typing import Any, Optional

_DEFAULT_TTL = 900  # 15 minutes in seconds

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
