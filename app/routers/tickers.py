"""
GET /api/tickers — Returns the list of stocks listed on the requested market.
"""

import json
import os
import time
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core import universe
from app.core.cache import get as cache_get, make_key, set as cache_set
from app.core.constants import TICKERS_CACHE_TTL_SECONDS
from app.core.db import get_db
from app.core.markets import MarketConfig, market_param
from app.core.tradingview import HEADERS as _TV_HEADERS, _exchange_filter, scan_url

router = APIRouter()

# Per-market ticker lists: market code -> (list, fetched_at).
_tickers_cache: dict[str, tuple[list, float]] = {}
_TICKERS_TTL = TICKERS_CACHE_TTL_SECONDS

_TV_SEARCH = (
    "https://symbol-search.tradingview.com/symbol_search/"
    "?text={symbol}&exchange={exchange}&lang=en&type=stock&domain=production"
)


def exchange_of(scanner_symbol: str) -> Optional[str]:
    """`"NASDAQ:AAPL"` -> `"NASDAQ"`; no prefix -> None."""
    if not scanner_symbol or ":" not in scanner_symbol:
        return None
    return scanner_symbol.split(":", 1)[0].upper() or None


def _load_static_json():
    json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "egx_tickers.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_live_tickers(market: MarketConfig):
    try:
        import urllib.request
        body = json.dumps({
            "columns": ["name", "description", "sector", "type"],
            "filter": [_exchange_filter(market)],
            "range": [0, 500],
        }).encode("utf-8")
        req = urllib.request.Request(scan_url(market), data=body, headers=_TV_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        rows = data.get("data") or []
        if not rows:
            return None

        results = []
        for row in rows:
            d = row.get("d") or []
            if len(d) < 3:
                continue
            symbol, name, sector = d[0], d[1], d[2]
            row_type = d[3] if len(d) > 3 else None
            if row_type and row_type != "stock":
                continue
            if not symbol:
                continue
            results.append({
                "symbol": symbol.upper(),
                "name": name or symbol,
                "sector": sector or "Unknown",
            })
        return results or None
    except Exception:
        return None


def _merge_lists(live, static_list):
    live_by_sym = {t["symbol"].upper(): t for t in live}
    merged = []
    seen = set()
    for s in static_list:
        sym = s["symbol"].upper()
        live_t = live_by_sym.get(sym)
        merged.append({
            "symbol": sym,
            "name": (live_t or {}).get("name") or s["name"],
            "sector": s.get("sector") or (live_t or {}).get("sector") or "Unknown",
            "index": s.get("index") or "EGX",
            "indices": [s.get("index") or "EGX"],
        })
        seen.add(sym)
    for t in live:
        sym = t["symbol"].upper()
        if sym in seen:
            continue
        merged.append({
            "symbol": sym,
            "name": t["name"],
            "sector": t.get("sector") or "Unknown",
            "index": "EGX",
            "indices": ["EGX"],
        })
    return merged


def _load_tickers(market: MarketConfig):
    now = time.time()
    hit = _tickers_cache.get(market.code)
    if hit is not None and (now - hit[1]) < _TICKERS_TTL:
        return hit[0]

    if market.code == "EG":
        # data/egx_tickers.json is EGX's hand-maintained universe, merged with
        # the scanner.
        live = _fetch_live_tickers(market)
        static_list = _load_static_json()
        result = _merge_lists(live, static_list) if live else [
            {**t, "indices": [t.get("index") or "EGX"]} for t in static_list
        ]
        try:
            db = get_db()
            known = {t["symbol"].upper() for t in result}
            rows = db.execute(
                "SELECT symbol, name, sector, index_name, exchange FROM discovered_tickers "
                "WHERE market = %s",
                (market.code,),
            ).fetchall()
            for symbol, name, sector, index_name, exchange in rows:
                if symbol.upper() not in known:
                    idx = index_name or "EGX"
                    result.append({
                        "symbol": symbol,
                        "name": name,
                        "sector": sector or "Unknown",
                        "index": idx,
                        "indices": [idx],
                    })
        except Exception:
            pass
        _tickers_cache[market.code] = (result, now)
        return result

    # Any other market: the scanner-backed universe (index members first), plus
    # symbols users have validated that the universe does not list.
    db = get_db()
    result = [{
        "symbol": r["symbol"], "name": r["name"], "sector": r["sector"],
        "index": (r["indices"][0] if r["indices"] else r["exchange"]),
        "indices": r["indices"], "exchange": r["exchange"],
    } for r in universe.load_universe(db, market)]
    try:
        known = {t["symbol"].upper() for t in result}
        rows = db.execute(
            "SELECT symbol, name, sector, exchange FROM discovered_tickers "
            "WHERE market = %s",
            (market.code,),
        ).fetchall()
        for symbol, name, sector, exchange in rows:
            if exchange and symbol.upper() not in known:
                result.append({
                    "symbol": symbol, "name": name,
                    "sector": sector or "Unknown",
                    "index": exchange, "indices": [], "exchange": exchange,
                })
    except Exception:
        pass

    # An empty result is a scanner outage, not an answer: do not pin it.
    if result:
        _tickers_cache[market.code] = (result, now)
    return result


def _search_exchange(market: MarketConfig, symbol: str):
    """TradingView symbol-search over every exchange of `market`. The search
    endpoint takes one exchange per query, so a multi-exchange market asks each
    in turn and stops at the first exact match."""
    import urllib.request
    last_error: Optional[Exception] = None
    succeeded = False
    for exchange in market.exchanges:
        try:
            url = _TV_SEARCH.format(symbol=quote(symbol), exchange=exchange)
            req = urllib.request.Request(url, headers=_TV_HEADERS)
            with urllib.request.urlopen(req, timeout=6) as resp:
                results = json.loads(resp.read())
        except Exception as e:
            last_error = e
            continue
        succeeded = True
        for r in results:
            sym = (r.get("symbol") or "").replace("<em>", "").replace("</em>", "").upper()
            if sym == symbol:
                return r, (r.get("exchange") or exchange).upper()
    if not succeeded and last_error is not None:
        raise last_error
    return None, None


def _remember_discovered(market: MarketConfig, symbol: str, name: str,
                         sector: Optional[str], exchange: Optional[str]):
    """Persist a validated symbol so it joins the ticker list. Best effort."""
    try:
        from datetime import datetime, timezone
        db = get_db()
        # index_name keeps its 'EGX' default meaning only for EG rows; other
        # markets carry their index membership in the universe, not here.
        db.execute(
            """INSERT INTO discovered_tickers
               (market, symbol, name, sector, index_name, exchange, added_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (market, symbol) DO NOTHING""",
            (market.code, symbol, name, sector or "Unknown",
             "EGX" if market.code == "EG" else None,
             exchange, datetime.now(timezone.utc).replace(tzinfo=None).isoformat()),
        )
        db.commit()
    except Exception:
        pass


def _validate_listing(symbol: str, market: MarketConfig):
    """Non-EG validation through the scanner's exact lookup."""
    cache_key = make_key(market.code, "validate", symbol)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        listing = universe.lookup_listing(market, symbol)
    except Exception:
        result = {"valid": None, "name": None}
    else:
        if listing:
            result = {"valid": True, "name": listing["name"] or symbol}
            _remember_discovered(market, symbol, result["name"],
                                 listing.get("sector"), listing.get("exchange"))
        else:
            result = {"valid": False, "name": None}
    cache_set(cache_key, result)
    return result


def _validate_symbol(symbol: str, market: MarketConfig):
    cache_key = make_key(market.code, "validate", symbol)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    exchange = None
    try:
        match, exchange = _search_exchange(market, symbol)
        result = (
            {"valid": True, "name": match.get("description", symbol)}
            if match
            else {"valid": False, "name": None}
        )
    except Exception:
        result = {"valid": None, "name": None}

    cache_set(cache_key, result)

    if result["valid"] is True:
        _remember_discovered(market, symbol, result["name"], "Unknown", exchange)

    return result


@router.get("/api/tickers")
def get_tickers(
    index: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    validate: Optional[str] = Query(None),
    market: MarketConfig = Depends(market_param),
):
    try:
        if validate:
            sym = validate.strip().upper()
            tickers = _load_tickers(market)
            known = next((t for t in tickers if t["symbol"].upper() == sym), None)
            if known:
                return {"symbol": sym, "valid": True, "name": known["name"]}
            result = (_validate_symbol(sym, market) if market.code == "EG"
                      else _validate_listing(sym, market))
            return {"symbol": sym, **result}

        tickers = _load_tickers(market)

        if index:
            want = index.upper()
            tickers = [t for t in tickers
                       if want in [i.upper() for i in t.get("indices") or [t["index"]]]]
        if sector:
            tickers = [t for t in tickers if t["sector"].lower() == sector.lower()]
        if search:
            q = search.lower()
            tickers = [
                t for t in tickers
                if q in t["symbol"].lower() or q in t["name"].lower()
            ]

        return tickers

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/tickers/search")
def search_tickers(
    q: str = Query("", max_length=64),
    market: MarketConfig = Depends(market_param),
):
    q = q.strip()
    if not q:
        return []
    if market.code == "EG":
        ql = q.lower()
        return [{"symbol": t["symbol"], "name": t["name"], "sector": t["sector"],
                 "exchange": "EGX", "kind": "stock"}
                for t in _load_tickers(market)
                if ql in t["symbol"].lower() or ql in t["name"].lower()][:20]
    try:
        return universe.search_symbols(market, q, limit=20)
    except Exception:
        raise HTTPException(status_code=502, detail="Search unavailable")
