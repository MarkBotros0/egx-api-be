"""
Market breadth — how much of the market is participating, not just the index.

WHY IT EXISTS, AND IT IS AN OPERATIONAL FIX FIRST
--------------------------------------------------
`/api/market_regime` averages composite scores that the DASHBOARD happens to
have cached. That worked while the app had anonymous traffic warming it. The app
is closed now, so the cache is warm only while a signed-in user on default
weights is browsing, and the card falls back to a stale reading most of the time.

Breadth needs no score cache. It aggregates flags the risk snapshot already
stores per symbol, so it is always as fresh as last night's cron and costs one
query. That is the main reason to have it.

WHAT THE EVIDENCE ACTUALLY SUPPORTS — READ BEFORE PRESENTING IT AS A FORECAST
-----------------------------------------------------------------------------
Measured on the cached panel against EGX30's next 63 trading days, with
Newey-West standard errors (the overlap correction the +0.318 regime claim
originally lacked):

    signal                          rho      NW t     verdict
    % of stocks oversold (RSI)    -0.188    -2.44     significant at 5%
    mean RSI                      +0.159    +1.95     not significant
    % above 50-day MA             +0.161    +1.90     not significant
    % above 200-day MA            +0.164    +1.66     not significant
    (composite mean, for scale)   +0.183    +1.98     not significant

A 4-leg blend scored r=+0.291 (t=+2.45) on non-overlapping samples, but a block
bootstrap over the full overlapping sample gives r=+0.138, p=0.093 — NOT
significant — and the four legs were chosen from about ten tested. None of this
clears the project's pre-registered |t| > 3.0 bar.

So breadth ships as CONTEXT with the same weak-evidence framing the regime card
already carries. It is not a new claim, and it must not be presented as one.
The one mildly interesting detail is the SIGN of the strongest leg: more stocks
oversold predicts WORSE forward returns, not a contrarian bounce.
"""

from __future__ import annotations

from typing import Optional

# Standard oversold threshold, and the leg that measured strongest.
RSI_OVERSOLD = 30

# Below this many measured symbols an aggregate is a handful of names wearing a
# percentage sign. Matches the regime card's refusal to classify thin coverage.
MIN_SYMBOLS_FOR_BREADTH = 15


def compute_breadth(rows: list) -> dict:
    """
    Aggregate stored per-symbol flags into a market reading.

    `rows` is what the risk snapshot holds: dicts with `tradeable`,
    `above_sma200` and `rsi_14`. Only TRADEABLE names count — a stock nobody can
    buy is not participating in anything, and including the market's dead tail
    would drag every reading toward whatever that tail happens to be doing.

    Returns None-valued fields rather than raising when coverage is too thin;
    the caller renders "not enough data" instead of a confident-looking number
    computed off nine stocks.
    """
    usable = [r for r in rows if r.get("tradeable")]
    above = [r["above_sma200"] for r in usable if r.get("above_sma200") is not None]
    rsis = [r["rsi_14"] for r in usable if r.get("rsi_14") is not None]

    if len(usable) < MIN_SYMBOLS_FOR_BREADTH or not above:
        return {
            "n_symbols": len(usable),
            "pct_above_sma200": None,
            "pct_oversold": None,
            "mean_rsi": None,
            "enough_data": False,
        }

    return {
        "n_symbols": len(usable),
        "pct_above_sma200": round(100.0 * sum(above) / len(above), 1),
        "pct_oversold": (round(100.0 * sum(1 for v in rsis if v < RSI_OVERSOLD)
                               / len(rsis), 1) if rsis else None),
        "mean_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
        "enough_data": True,
        # Carried so the UI never hardcodes a claim, exactly as classify_regime
        # does. Every one of these is below the project's evidence bar.
        "evidence": {
            "horizon_days": 63,
            "strongest_leg": "pct_oversold",
            "strongest_rho": -0.188,
            "strongest_t": -2.44,
            "significant_at_project_bar": False,
            "note": ("Breadth tracks how much of the market is participating. "
                     "Its link to the next three months is weak and does not "
                     "clear this project's evidence bar, so it is context "
                     "rather than a forecast."),
        },
    }


def describe(breadth: dict) -> Optional[str]:
    """One plain sentence, or None when there is not enough to say."""
    if not breadth.get("enough_data"):
        return None
    pct = breadth["pct_above_sma200"]
    if pct >= 60:
        shape = "Most shares are above their long-term average price"
    elif pct >= 40:
        shape = "Shares are split about evenly above and below their long-term average"
    else:
        shape = "Most shares are below their long-term average price"
    return (f"{shape} — {pct}% of {breadth['n_symbols']} tradeable stocks. "
            f"This describes today, not what happens next.")
