"""
GET /api/tickers — Returns the list of EGX-listed stocks.
"""

from urllib.parse import urlparse, parse_qs
import json
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.core.cache import get as cache_get, set as cache_set
from app.core.constants import TICKERS_CACHE_TTL_SECONDS
from app.core.db import get_db

router = APIRouter()

_tickers_cache = None
_tickers_cache_ts = 0
_TICKERS_TTL = TICKERS_CACHE_TTL_SECONDS

_TV_SCAN_URL = "https://scanner.tradingview.com/egypt/scan"
_TV_SEARCH = (
    "https://symbol-search.tradingview.com/symbol_search/"
    "?text={symbol}&exchange=EGX&lang=en&type=stock&domain=production"
)
_TV_HEADERS = {
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


def _load_static_json():
    json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "egx_tickers.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fetch_live_tickers():
    try:
        import urllib.request
        body = json.dumps({
            "columns": ["name", "description", "sector", "type"],
            "filter": [{"left": "exchange", "operation": "equal", "right": "EGX"}],
            "range": [0, 500],
        }).encode("utf-8")
        req = urllib.request.Request(_TV_SCAN_URL, data=body, headers=_TV_HEADERS)
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
        })
    return merged


def _load_tickers():
    global _tickers_cache, _tickers_cache_ts

    now = time.time()
    if _tickers_cache is not None and (now - _tickers_cache_ts) < _TICKERS_TTL:
        return _tickers_cache

    static_list = _load_static_json()
    live = _fetch_live_tickers()

    if live:
        result = _merge_lists(live, static_list)
    else:
        result = list(static_list)

    try:
        db = get_db()
        known = {t["symbol"].upper() for t in result}
        rows = db.execute(
            "SELECT symbol, name, sector, index_name FROM discovered_tickers"
        ).fetchall()
        for symbol, name, sector, index_name in rows:
            if symbol.upper() not in known:
                result.append({
                    "symbol": symbol,
                    "name": name,
                    "sector": sector or "Unknown",
                    "index": index_name or "EGX",
                })
    except Exception:
        pass

    _tickers_cache = result
    _tickers_cache_ts = now
    return result


def _validate_symbol(symbol: str):
    cache_key = f"validate:{symbol}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        import urllib.request
        url = _TV_SEARCH.format(symbol=symbol)
        req = urllib.request.Request(url, headers=_TV_HEADERS)
        with urllib.request.urlopen(req, timeout=6) as resp:
            results = json.loads(resp.read())
        match = None
        for r in results:
            sym = (r.get("symbol") or "").replace("<em>", "").replace("</em>", "").upper()
            if sym == symbol:
                match = r
                break
        result = (
            {"valid": True, "name": match.get("description", symbol)}
            if match
            else {"valid": False, "name": None}
        )
    except Exception:
        result = {"valid": None, "name": None}

    cache_set(cache_key, result)

    if result["valid"] is True:
        try:
            from datetime import datetime
            db = get_db()
            db.execute(
                """INSERT OR IGNORE INTO discovered_tickers
                   (symbol, name, sector, index_name, added_at)
                   VALUES (?, ?, 'Unknown', 'EGX', ?)""",
                (symbol, result["name"], datetime.utcnow().isoformat()),
            )
            db.commit()
        except Exception:
            pass

    return result


@router.get("/api/tickers")
def get_tickers(
    index: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    validate: Optional[str] = Query(None),
):
    try:
        if validate:
            sym = validate.strip().upper()
            tickers = _load_tickers()
            known = next((t for t in tickers if t["symbol"].upper() == sym), None)
            if known:
                return {"symbol": sym, "valid": True, "name": known["name"]}
            result = _validate_symbol(sym)
            return {"symbol": sym, **result}

        tickers = _load_tickers()

        if index:
            tickers = [t for t in tickers if t["index"].lower() == index.lower()]
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
