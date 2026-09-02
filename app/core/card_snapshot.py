"""
The dashboard's read model: one pre-scored row per symbol.

WHY THIS EXISTS
---------------
Every dashboard card used to require a live 400-bar fetch through a client that
retries hard on socket timeouts — about 1.4s for a symbol the feed serves and
about 6s for one it refuses, and it refuses 84 of the 166 symbols in
`data/egx_tickers.json`. Twenty-four cards could not finish inside a serverless
request, so the page fanned the work out across a dozen concurrent requests.
Vercel answered those from a dozen separate containers, each with its own empty
module-level cache, each independently re-fetching the EGX30 benchmark, and each
discarding its results when it froze. Whether a card painted came down to which
container answered and whether it happened to be warm.

So the cards are served from a table instead. The nightly
`POST /api/cron/risk_snapshot` ALREADY fetches exactly the 400 daily bars a card
needs, for the whole universe, chunked stalest-first. Scoring on data already in
hand costs ~0.15s of CPU against that ~1.4s fetch, so the score rides along on a
fetch we already pay for. The same argument this codebase already made twice:
`fundamentals_annual` rides `/api/pe/refresh`, and the breadth flags ride
`measure()`.

THE EIGHT CATEGORY SCORES ARE STORED, NOT THE COMPOSITE
-------------------------------------------------------
This is what keeps *One Score Per Stock* true rather than trading it away.
Weighting, renormalisation and macro modulation are a pure function of the eight
category scores, the caller's weights and today's EGX30 regime — that function
is `composite.blend_categories`, and BOTH this module and `compute_composite`
call it. So the number on a card is this user's sliders applied to the same
inputs the detail page uses, and the two cannot disagree.

Storing a blended number instead would freeze one weight set into the card and
reintroduce exactly the divergence `extras_builder.py` exists to prevent
(measured once at 66 "Buy" on the card against 45 "Hold" on the detail page).

FRESHNESS IS STATED, NEVER IMPLIED
-----------------------------------
A row is as current as the cron pass that wrote it — post-close, so during
trading hours the price is intraday-stale. `measured_at` is per row and the read
path reports the STALEST one, matching what `/api/risk` already does. The
dashboard shows the date and upgrades visible cards to live prices in the
background; it never presents last night's close as this minute's.
"""

from __future__ import annotations

import json
from typing import Optional

from app.core.composite import CATEGORY_ORDER, blend_categories
from app.core.constants import FAILURE_DEMOTION_THRESHOLD

# Column name per category, DERIVED from CATEGORY_ORDER rather than typed out.
# Adding a ninth category then adds its column automatically in `init_db`, the
# cron's upsert and the read below — instead of failing silently in whichever
# of the three someone forgot. `tests/test_dashboard_snapshot.py` pins that the
# DDL and this mapping stay in step.
CATEGORY_COLUMNS = tuple(f"cat_{name}" for name in CATEGORY_ORDER)

# How many closes the card's sparkline draws. Matches the 30 the live batch
# path returns, so a card that upgrades to live data does not change shape.
SPARKLINE_BARS = 30

_READ_COLUMNS = (
    "symbol", "measured_at", "scored_at", "last_price", "prev_close",
    "sparkline_json", "tradeable", "sigma_63_ann_pct",
    "COALESCE(consecutive_failures, 0)",
) + CATEGORY_COLUMNS


def read_rows(db) -> list:
    """
    Every snapshot row, as dicts. The one spelling of this SELECT.

    Deliberately unfiltered: a symbol the feed refuses still has a row (the
    cron records EVERY attempt, which is what keeps stalest-first selection
    from re-picking it forever), and the dashboard needs those rows to show
    "no price feed" instead of a card that loads for ever.
    """
    result = db.execute(
        f"SELECT {', '.join(_READ_COLUMNS)} FROM risk_snapshot"
    ).fetchall()

    out = []
    for r in result:
        row = {
            "symbol": r[0],
            "measured_at": r[1],
            "scored_at": r[2],
            "last_price": r[3],
            "prev_close": r[4],
            "sparkline_json": r[5],
            "tradeable": r[6],
            "sigma_63_ann_pct": r[7],
            "consecutive_failures": r[8] or 0,
        }
        for i, name in enumerate(CATEGORY_ORDER):
            row[name] = r[9 + i]
        out.append(row)
    return out


def _sparkline(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(values, list):
        return []
    return [float(v) for v in values if isinstance(v, (int, float))]


def to_card(row: dict, weights: dict, macro: Optional[dict]) -> dict:
    """
    One snapshot row -> one dashboard card, scored with THIS caller's weights.

    `available` is false when the feed has refused this symbol enough times to
    be demoted, or when nothing was scorable. The distinction matters on screen:
    a card that says "no price feed" is information, while the "--" it used to
    show was indistinguishable from still-loading and sent the reader back to
    reload a page that was never going to fill.
    """
    category_scores = {name: row.get(name) for name in CATEGORY_ORDER}
    scorable = any(v is not None for v in category_scores.values())

    unavailable = (
        row.get("consecutive_failures", 0) >= FAILURE_DEMOTION_THRESHOLD
        or (not scorable and row.get("last_price") is None)
    )

    card = {
        "symbol": row["symbol"],
        "available": not unavailable,
        "score": None,
        "signal": None,
        "price": row.get("last_price"),
        "change": None,
        "change_pct": None,
        "sparkline": _sparkline(row.get("sparkline_json")),
        "measured_at": row.get("measured_at"),
        "scored_at": row.get("scored_at"),
        "tradeable": row.get("tradeable"),
        "sigma_63_ann_pct": row.get("sigma_63_ann_pct"),
    }

    if scorable:
        blended = blend_categories(category_scores, weights, macro)
        card["score"] = blended["score"]
        card["signal"] = blended["signal"]

    price, prev = row.get("last_price"), row.get("prev_close")
    if price is not None and prev:
        card["change"] = price - prev
        card["change_pct"] = (price - prev) / prev * 100

    return card
