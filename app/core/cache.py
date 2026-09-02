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

from app.core.constants import DEFAULT_CACHE_TTL_SECONDS

_DEFAULT_TTL = DEFAULT_CACHE_TTL_SECONDS

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
    timestamp, value, ttl = entry
    if time.time() - timestamp > ttl:
        del _store[key]
        return None
    return value


def set(key: str, value: Any, ttl: Optional[float] = None) -> None:
    """
    Store a value with the current timestamp.

    `ttl` overrides the default for this entry. It exists so a FAILURE can be
    remembered on a shorter leash than a price: a symbol the upstream feed
    refuses costs ~6 seconds to learn about, and re-learning that on every
    dashboard load was a large part of why the grid was slow — but a refusal is
    a much more perishable fact than a close, so it must not sit for the full
    15 minutes a good result does.
    """
    _store[key] = (time.time(), value, _DEFAULT_TTL if ttl is None else ttl)
