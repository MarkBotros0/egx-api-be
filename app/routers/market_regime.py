"""
/api/market_regime — how broadly healthy the EGX looks right now.

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

from fastapi import APIRouter, HTTPException

from app.core.cache import get as cache_get, make_key, set as cache_set
from app.core.db import get_db
from app.core.breadth import compute_breadth, describe as describe_breadth
from app.core.regime import classify_regime
from app.routers.analysis import read_cached_scores

router = APIRouter()

# The universe the bands were calibrated on. EGX30 alone was measurably weaker
# (rho +0.137 vs +0.170 at 63 days), and the whole market cannot be scored
# inside a serverless request.
_REGIME_INDEXES = ("EGX30", "EGX70")


def _regime_universe() -> list:
    """EGX30 + EGX70 symbols from the static ticker file. No network."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "egx_tickers.json"
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sorted(
                t["symbol"].upper()
                for t in json.load(f)
                if (t.get("index") or "").upper() in _REGIME_INDEXES
            )
    except Exception:
        return []


def _store_reading(reading: dict) -> None:
    """Append the reading so a history builds and a cold cache has a fallback."""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO market_regime (observed_at, mean_score, n_symbols, band) "
            "VALUES (%s, %s, %s, %s)",
            (datetime.now(timezone.utc).isoformat(), reading["mean_score"],
             reading["n_symbols"], reading["band"]),
        )
        db.commit()
    except Exception:
        pass


def _last_stored_reading() -> Optional[dict]:
    """Most recent stored reading, re-classified so wording stays in one place."""
    try:
        db = get_db()
        row = db.execute(
            "SELECT observed_at, mean_score, n_symbols, band FROM market_regime "
            "ORDER BY observed_at DESC LIMIT 1"
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    reading = classify_regime(row[1], row[2])
    reading["observed_at"] = row[0]
    return reading


def _breadth() -> dict:
    """Aggregate the snapshot's stored flags. Never fetches; never raises."""
    try:
        db = get_db()
        rows = db.execute(
            "SELECT tradeable, above_sma200, rsi_14 FROM risk_snapshot"
        ).fetchall()
    except Exception:
        return {"enough_data": False, "n_symbols": 0}
    result = compute_breadth([
        {"tradeable": r[0], "above_sma200": r[1], "rsi_14": r[2]} for r in rows
    ])
    summary = describe_breadth(result)
    if summary:
        result["summary"] = summary
    return result


@router.get("/api/market_regime")
def get_market_regime(interval: str = "Daily"):
    try:
        universe = _regime_universe()
        if not universe:
            raise HTTPException(status_code=500, detail="No regime universe available")

        interval = interval.capitalize()
        ck = make_key("regime", interval, len(universe))
        hit = cache_get(ck)
        if hit is not None:
            return hit

        cached = read_cached_scores(universe, interval)
        scores = [float(v["score"]) for v in cached.values()]
        mean_score = sum(scores) / len(scores) if scores else None

        reading = classify_regime(mean_score, len(scores))
        reading["universe_size"] = len(universe)
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
        reading["breadth"] = _breadth()

        if reading.get("band") is not None:
            cache_set(ck, reading)
            _store_reading(reading)
            return reading

        # Coverage too thin right now. A reading from earlier today is still
        # informative; silence is not.
        previous = _last_stored_reading()
        if previous is not None:
            previous["stale"] = True
            previous["n_symbols_now"] = len(scores)
            previous["universe_size"] = len(universe)
            previous["interval"] = interval
            return previous
        return reading
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
