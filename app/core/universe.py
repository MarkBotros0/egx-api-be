"""
The US universe and the one answer to "which exchange does this symbol trade on?".

WHY
---
tvDatafeed needs the listing exchange. Asked on the wrong one it waits ~5.5s
and returns nothing (KO on NASDAQ, measured 2026-10-15) — and ~70% of the
S&P 500 is on NYSE. So no symbol-level fetch may assume a market's primary
exchange: every one asks `resolve_exchange`.

EG needs none of this: one exchange, a hand-maintained ticker file.

The universe is the union of the market's index symbolsets (S&P 500,
NASDAQ-100, Dow 30), one scanner call each, plus every listing a user has
searched for and added (indices = {}). A refresh writes all or nothing.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from app.core.markets import EG, MarketConfig
from app.core.tradingview import scan

_COLUMNS = ["name", "description", "sector", "type"]
_KINDS = ("stock", "fund")
_UNIVERSE_TTL = 3600
_MISS_TTL = 3600

_universe_cache: dict[str, tuple[list, float]] = {}
_exchange_cache: dict[tuple[str, str], tuple[Optional[str], float]] = {}


def clear_caches() -> None:
    _universe_cache.clear()
    _exchange_cache.clear()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _parse(row: dict) -> Optional[dict]:
    d = row.get("d") or []
    s = row.get("s") or ""
    if len(d) < 4 or ":" not in s:
        return None
    kind = (d[3] or "").lower()
    if kind not in _KINDS:
        return None
    symbol = (d[0] or "").strip().upper()
    if not symbol:
        return None
    return {"symbol": symbol, "name": d[1] or symbol, "sector": d[2] or "Unknown",
            "exchange": s.split(":", 1)[0].upper(), "kind": kind}


def refresh_universe(db, market: MarketConfig) -> dict:
    sets = [i for i in market.indices if i.symbolset]
    if not sets:
        return {"success": False, "count": 0, "error": f"no symbolsets for {market.code}"}
    rows: dict[str, dict] = {}
    try:
        for idx in sets:
            for raw in scan(_COLUMNS, market=market, symbolset=idx.symbolset, limit=1000):
                r = _parse(raw)
                if r is None:
                    continue
                rows.setdefault(r["symbol"], {**r, "indices": []})["indices"].append(idx.id)
    except Exception as e:
        return {"success": False, "count": 0, "error": f"{type(e).__name__}: {e}"}
    if not rows:
        return {"success": False, "count": 0, "error": "no rows returned"}

    now = _now_iso()
    with db.transaction() as tx:
        tx.execute("UPDATE market_universe SET indices = '{}' WHERE market = %s", (market.code,))
        for r in rows.values():
            tx.execute(
                """INSERT INTO market_universe
                   (market, symbol, name, sector, exchange, kind, indices, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (market, symbol) DO UPDATE SET
                     name = EXCLUDED.name, sector = EXCLUDED.sector,
                     exchange = EXCLUDED.exchange, kind = EXCLUDED.kind,
                     indices = EXCLUDED.indices, updated_at = EXCLUDED.updated_at""",
                (market.code, r["symbol"], r["name"], r["sector"], r["exchange"],
                 r["kind"], r["indices"], now),
            )
    _universe_cache.pop(market.code, None)
    for r in rows.values():
        _exchange_cache[(market.code, r["symbol"])] = (r["exchange"], time.time())
    return {"success": True, "count": len(rows), "error": None}


def _read(db, market: MarketConfig) -> list:
    rows = db.execute(
        "SELECT symbol, name, sector, exchange, kind, indices FROM market_universe "
        "WHERE market = %s ORDER BY cardinality(indices) = 0, symbol",
        (market.code,),
    ).fetchall()
    return [{"symbol": r[0], "name": r[1] or r[0], "sector": r[2] or "Unknown",
             "exchange": r[3], "kind": r[4], "indices": list(r[5] or [])} for r in rows]


def load_universe(db, market: MarketConfig) -> list:
    if market.code == EG.code:
        return []
    hit = _universe_cache.get(market.code)
    if hit is not None and time.time() - hit[1] < _UNIVERSE_TTL:
        return hit[0]
    try:
        rows = _read(db, market)
        if not rows and refresh_universe(db, market)["success"]:
            rows = _read(db, market)
    except Exception:
        rows = []
    if rows:
        _universe_cache[market.code] = (rows, time.time())
    return rows


def lookup_listing(market: MarketConfig, symbol: str) -> Optional[dict]:
    """Exact listing across the market's exchanges, one scanner call. Raises on scanner error."""
    sym = symbol.strip().upper()
    raw = scan(_COLUMNS, market=market, tickers=[f"{ex}:{sym}" for ex in market.exchanges], limit=10)
    for r in raw:
        p = _parse(r)
        if p and p["symbol"] == sym:
            return p
    return None


def resolve_exchange(db, market: MarketConfig, symbol: str) -> Optional[str]:
    if market.code == EG.code:
        return market.primary_exchange
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    key = (market.code, sym)
    hit = _exchange_cache.get(key)
    if hit is not None and (hit[0] is not None or time.time() - hit[1] < _MISS_TTL):
        return hit[0]

    for sql in ("SELECT exchange FROM market_universe WHERE market = %s AND symbol = %s",
                "SELECT exchange FROM discovered_tickers WHERE market = %s AND symbol = %s"):
        try:
            row = db.execute(sql, (market.code, sym)).fetchone() if db is not None else None
        except Exception:
            row = None
        if row and row[0]:
            _exchange_cache[key] = (row[0].upper(), time.time())
            return row[0].upper()

    try:
        found = lookup_listing(market, sym)
    except Exception:
        return None            # scanner down: not a miss, do not cache
    if found is None:
        _exchange_cache[key] = (None, time.time())
        return None
    try:
        db.execute(
            """INSERT INTO market_universe
               (market, symbol, name, sector, exchange, kind, indices, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, '{}', %s)
               ON CONFLICT (market, symbol) DO NOTHING""",
            (market.code, found["symbol"], found["name"], found["sector"],
             found["exchange"], found["kind"], _now_iso()),
        )
        db.commit()
        _universe_cache.pop(market.code, None)
    except Exception:
        pass
    _exchange_cache[key] = (found["exchange"], time.time())
    return found["exchange"]


def search_symbols(market: MarketConfig, q: str, limit: int = 20) -> list:
    """Exact ticker first, then name matches by market cap. Raises on scanner error."""
    q = (q or "").strip()
    if not q:
        return []
    out: list = []
    seen: set = set()
    exact = lookup_listing(market, q) if " " not in q else None
    if exact:
        out.append(exact)
        seen.add(exact["symbol"])
    raw = scan(_COLUMNS, market=market, limit=limit, sort_by="market_cap_basic",
               extra_filter=[{"left": "description", "operation": "match", "right": q}])
    for r in raw:
        p = _parse(r)
        if p and p["symbol"] not in seen:
            out.append(p)
            seen.add(p["symbol"])
    return out[:limit]
