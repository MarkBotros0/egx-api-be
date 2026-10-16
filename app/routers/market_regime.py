"""
/api/market_regime — how broadly healthy the market looks right now.

This is deliberately the ONLY forecast-shaped surface in the app. The per-stock
composite was tested and cannot rank stocks; the market-wide AVERAGE of those
same scores does carry a measured association with the market's own next three
months. See core/regime.py for the numbers and their caveats.

It reads scores the dashboard has ALREADY computed and never fetches. Scoring
the 79-symbol universe on demand does not finish inside a serverless request —
measured at over 400 s against a cold cache, because each symbol pulls 400 bars
through a client that retries hard on socket timeouts. So coverage builds as
the user browses, `n_symbols` reports it honestly, and a persisted
last-known-good covers the gap rather than showing nothing.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core import universe as universe_mod
from app.core.cache import get as cache_get, make_key, set as cache_set
from app.core.markets import EG, MarketConfig, market_param
from app.core.db import get_db
from app.core.breadth import compute_breadth, describe as describe_breadth
from app.core.calibration import for_market
from app.core.regime import classify_regime
from app.routers.analysis import read_cached_scores

router = APIRouter()

def _store_reading(reading: dict, market_code: str = EG.code) -> None:
    """Append the reading so a history builds and a cold cache has a fallback."""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO market_regime (market, observed_at, mean_score, n_symbols, band) "
            "VALUES (%s, %s, %s, %s, %s)",
            (market_code, datetime.now(timezone.utc).isoformat(),
             reading["mean_score"], reading["n_symbols"], reading["band"]),
        )
        db.commit()
    except Exception:
        pass


def _last_stored_reading(market_code: str = EG.code) -> Optional[dict]:
    """Most recent stored reading, re-classified so wording stays in one place."""
    try:
        db = get_db()
        row = db.execute(
            "SELECT observed_at, mean_score, n_symbols, band FROM market_regime "
            "WHERE market = %s ORDER BY observed_at DESC LIMIT 1",
            (market_code,),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    reading = classify_regime(row[1], row[2], for_market(market_code).regime)
    reading["observed_at"] = row[0]
    return reading


def _eg_regime_universe(indexes) -> list:
    """EGX30 + EGX70 symbols from the static ticker file. No network."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "egx_tickers.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sorted(t["symbol"].upper() for t in json.load(f)
                          if (t.get("index") or "").upper() in indexes)
    except Exception:
        return []


def _regime_universe(market: MarketConfig, regime) -> list:
    """
    The constituents the market's bands were calibrated on: EG from the ticker
    file (EGX30 + EGX70, as before); other markets from `market_universe`
    (US: SPX members). An unreadable universe is [], which the route serves as
    a stale "not enough data" reading rather than a 500.
    """
    if market.code == EG.code:
        return _eg_regime_universe(regime.universe_indexes)
    try:
        rows = universe_mod.load_universe(get_db(), market)
    except Exception:
        return []
    want = set(regime.universe_indexes)
    return sorted(r["symbol"].upper() for r in rows if want & set(r.get("indices") or ()))


def _breadth(market_code: str = EG.code, measured: bool = True) -> dict:
    """Aggregate the snapshot's stored flags. Never fetches; never raises.
    Breadth's evidence numbers were measured on the EGX; a market whose bundle
    did not measure them gets the reading without the evidence."""
    try:
        db = get_db()
        rows = db.execute(
            "SELECT tradeable, above_sma200, rsi_14 FROM risk_snapshot "
            "WHERE market = %s",
            (market_code,),
        ).fetchall()
    except Exception:
        return {"enough_data": False, "n_symbols": 0}
    result = compute_breadth([
        {"tradeable": r[0], "above_sma200": r[1], "rsi_14": r[2]} for r in rows
    ])
    if not measured:
        result.pop("evidence", None)
    summary = describe_breadth(result)
    if summary:
        result["summary"] = summary
    return result


@router.get("/api/market_regime")
def get_market_regime(interval: str = "Daily",
                      market: MarketConfig = Depends(market_param)):
    stamp = {"market": market.code, "currency": market.currency}
    cal = for_market(market)
    if cal.provisional or not cal.regime.show:
        # Measured on this market and the association ran the wrong way: a card
        # saying "broad strength" that preceded WEAKER markets would mislead.
        # A provisional bundle is another market's fit standing in, so its
        # bands were never measured here at all — hidden for the same reason.
        return {**stamp, "band": None, "hidden": True, "n_symbols": 0}
    try:
        interval = interval.capitalize()
        symbols = _regime_universe(market, cal.regime)
        if not symbols:
            previous = _last_stored_reading(market.code)
            if previous is not None:
                previous.update(stamp)
                previous.update({"stale": True, "n_symbols_now": 0, "interval": interval})
                return previous
            return {**stamp, "band": None, "stale": True, "n_symbols": 0}

        ck = make_key(market.code, "regime", interval, len(symbols))
        hit = cache_get(ck)
        if hit is not None:
            return hit

        cached = read_cached_scores(symbols, interval, market)
        scores = [float(v["score"]) for v in cached.values()]
        mean_score = sum(scores) / len(scores) if scores else None

        reading = classify_regime(mean_score, len(scores), cal.regime)
        reading.update(stamp)
        reading["universe_size"] = len(symbols)
        reading["interval"] = interval
        reading["stale"] = False
        # Breadth, from flags the nightly risk snapshot already stores. It needs
        # no score cache, which is the point: the composite average above is
        # warm only while a signed-in user on default weights is browsing, and
        # the app has no anonymous traffic any more. Breadth is as fresh as last
        # night either way.
        #
        # It is NOT a second forecast. Its strongest leg reaches t=-2.44, below
        # this project's |t| > 3.0 bar, so it ships with the same weak-evidence
        # framing and carries its own numbers so the UI cannot invent a claim.
        reading["breadth"] = _breadth(market.code, cal.breadth_measured)

        if reading.get("band") is not None:
            cache_set(ck, reading)
            _store_reading(reading, market.code)
            return reading

        previous = _last_stored_reading(market.code)
        if previous is not None:
            previous.update(stamp)
            previous["stale"] = True
            previous["n_symbols_now"] = len(scores)
            previous["universe_size"] = len(symbols)
            previous["interval"] = interval
            return previous
        return reading
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
