"""
GET /api/dashboard — every stock card, from one query, with no upstream fetch.

This replaces the dozen concurrent `/api/analysis?mode=batch` requests the grid
used to fire. Those were slow for a structural reason, not a tunable one: each
card needed a live 400-bar pull through a client that retries hard on socket
timeouts, and the feed refuses 84 of the 166 symbols in the ticker file at about
six seconds a refusal. Fanning that across a dozen serverless containers meant a
dozen empty module-level caches, a dozen duplicate EGX30 benchmark fetches, and
results discarded whenever a container was recycled. Whether a card painted came
down to luck.

Here the work is already done. `POST /api/cron/risk_snapshot` scores every
symbol on bars it was fetching anyway, and this route reads the result and
blends it with THIS caller's weights. One DB round trip, no network, no
deadline, deterministic — the whole grid or nothing.

See `core/card_snapshot.py` for why the eight category scores are stored rather
than a finished composite; that is what keeps a card and its own detail page
from disagreeing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import CurrentUser, get_current_user
from app.core.card_snapshot import attach_risk_bands, read_rows, to_card
from app.core.composite import get_weights_from_db
from app.core.db import get_db
from app.core.macro_fetch import fetch_macro

router = APIRouter()


@router.get("/api/dashboard")
def get_dashboard(user: CurrentUser = Depends(get_current_user)):
    """
    Every symbol the snapshot holds, scored with the caller's own weights.

    Returns the WHOLE universe rather than a page. It costs one query and a few
    tens of kilobytes, and it is what lets the grid filter, search and sort
    across every stock instantly instead of only across the 24 cards that
    happened to have been fetched — sorting by score was structurally
    impossible before, because off-screen cards had no score to sort by.

    `oldest_measurement` is the STALEST row, not the freshest, matching
    `/api/risk`. A snapshot is only as current as its oldest corner, and a
    dashboard that reported its newest row would overstate its own freshness
    every single time.
    """
    try:
        db = get_db()
        rows = read_rows(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not rows:
        # An empty table is not an error, and it must not read as one: it means
        # the scheduled job has not run yet. The frontend falls back to the live
        # scoring path on this, which is what makes deploying the backend ahead
        # of the first cron pass safe.
        return {
            "rows": [], "n_symbols": 0, "n_available": 0,
            "oldest_measurement": None, "newest_measurement": None,
            "note": "No snapshot yet — the scheduled risk_snapshot job has "
                    "not run.",
        }

    weights = get_weights_from_db(db, user.id)
    try:
        macro = fetch_macro(db)
    except Exception:
        # The macro modulation is a post-hoc adjustment, not an input the score
        # cannot be computed without. Losing it costs the bearish damping, not
        # the grid.
        macro = None

    cards = [to_card(row, weights, macro) for row in rows]
    # The per-stock risk band is a cross-sectional rank over the whole universe,
    # so it is stamped here in the orchestrator through the SAME grade_universe
    # GET /api/risk uses — a card's dot and its detail page cannot disagree.
    attach_risk_bands(rows, cards)
    cards.sort(key=lambda c: c["symbol"])

    measured = [r["measured_at"] for r in rows if r.get("measured_at")]
    scored_at = [r["scored_at"] for r in rows if r.get("scored_at")]

    return {
        "rows": cards,
        "n_symbols": len(cards),
        "n_available": sum(1 for c in cards if c["available"]),
        "oldest_measurement": min(measured) if measured else None,
        "newest_measurement": max(measured) if measured else None,
        # When the most recently scored row was scored. Distinct from
        # measured_at because a symbol whose fetch failed keeps its previous
        # score rather than being blanked, so the two can legitimately differ.
        "newest_scored_at": max(scored_at) if scored_at else None,
        "macro_context": (
            ((macro or {}).get("egx30") or {}).get("trend")
        ),
    }
