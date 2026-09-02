"""
GET /api/risk — the per-stock risk grade.

This is the app's strongest measured surface, and the only per-stock ranking
the evidence supports. See core/risk_grade.py for the numbers; in short, past
63-day volatility predicts the next 126 days' realized volatility at rank IC
+0.56 (t=+24.0 on non-overlapping data) and max drawdown at +0.43 (t=+16.7),
against a composite score that cannot rank returns at all.

Reads the chunked snapshot written by POST /api/cron/risk_snapshot and ranks it
cross-sectionally on the way out, so a partly-refreshed table is still coherent.

Authenticated like everything else — it is deliberately NOT in
PUBLIC_ENDPOINTS. The app is closed; only the scheduled writer is open, and
only behind a secret.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_current_user
from app.core.db import get_db
from app.core.risk_grade import (
    LIQUID_MIN_TURNOVER_EGP,
    RISK_CALIBRATION,
    grade_universe,
)

router = APIRouter()


def _rows(db) -> list:
    result = db.execute(
        "SELECT symbol, measured_at, sigma_63_ann_pct, sigma_ewma_ann_pct, "
        "beta, turnover_egp, traded_share, last_price, tradeable "
        "FROM risk_snapshot"
    ).fetchall()
    return [{
        "symbol": r[0], "measured_at": r[1], "sigma_63_ann_pct": r[2],
        "sigma_ewma_ann_pct": r[3], "beta": r[4], "turnover_egp": r[5],
        "traded_share": r[6], "last_price": r[7], "tradeable": r[8],
    } for r in result]


@router.get("/api/risk")
def get_risk(symbol: Optional[str] = Query(default=None),
             user=Depends(get_current_user)):
    """
    The whole graded universe, or one symbol's grade.

    Ranking is computed over the TRADEABLE subset only. Ranking a stock nobody
    can enter or exit against names that trade would produce a percentile that
    means nothing — and this market has plenty of them: 19% of its daily
    returns are exactly zero and a third of listed names barely trade.
    An untradeable symbol is still returned, with its raw sigma and a null
    band, rather than being given a rank it has not earned.
    """
    try:
        db = get_db()
        rows = _rows(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        return {"data": [], "n_symbols": 0, "coverage": 0,
                "calibration": RISK_CALIBRATION,
                "note": "No snapshot yet — the scheduled risk_snapshot job has "
                        "not run."}

    tradeable = [r for r in rows if r.get("tradeable")]
    graded = {r["symbol"]: r for r in grade_universe(tradeable)}
    for r in rows:
        if r["symbol"] not in graded:
            r["pct_rank"] = r["quintile"] = r["band"] = None

    data = [graded.get(r["symbol"], r) for r in rows]
    measured = [r["measured_at"] for r in rows if r.get("measured_at")]

    payload = {
        "data": sorted(data, key=lambda r: (r.get("pct_rank") is None,
                                            r.get("pct_rank") or 0)),
        "n_symbols": len(rows),
        "n_ranked": len(tradeable),
        # The thinnest corner, not the freshest: a snapshot is only as current
        # as its oldest row, and reporting the newest would flatter a refresh
        # that stalled halfway through the universe.
        "oldest_measurement": min(measured) if measured else None,
        "newest_measurement": max(measured) if measured else None,
        "liquidity_floor_egp": LIQUID_MIN_TURNOVER_EGP,
        "calibration": RISK_CALIBRATION,
    }

    if symbol:
        wanted = symbol.strip().upper()
        one = next((r for r in data if r["symbol"] == wanted), None)
        if one is None:
            raise HTTPException(status_code=404,
                                detail=f"No risk measurement for {wanted}")
        payload["data"] = [one]
        payload["symbol"] = wanted
    return payload
