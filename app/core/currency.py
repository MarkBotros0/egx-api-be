"""
The same return, in the currency you actually spend.

WHY THIS EXISTS
---------------
Every figure in this app is in EGP, and over the period it covers the EGP lost
most of its value against the dollar. The scale of it is not a footnote:

    EGX30, 2006 -> 2026:   6,702 -> 55,277 in EGP   = 8.25x
                           1,168 ->  1,100 in USD   = 0.94x

Twenty years of "the market went up 8x" is, in hard-currency terms, twenty years
of going nowhere. A beginner reading "+66% in 2023" is reading a year in which
the pound was devalued twice. Nothing in the app has ever said so.

This module does not decide which number is "right" — both are true, they answer
different questions. EGP is what you spend at home; USD is what your savings are
worth if you ever need to buy anything priced in it, which in Egypt includes a
great deal. The app's job is to show both and let the user hold the tension.

WHAT THIS IS NOT
----------------
Not an inflation adjustment. Real (CPI-deflated) return is a third, different
number, and `macro_series` carries EGCPI for whoever builds it. USD conversion
is chosen here because the rate is observable daily and unambiguous, while a
CPI deflator needs a basket the user may not consume.
"""

from __future__ import annotations

from typing import Optional

# Measured on the cached panel, 2006-01-05 -> 2026-08-25, EGX30 close deflated
# by FX_IDC:USDEGP. Stated as context on the macro surface so the user has a
# yardstick for their own numbers rather than a bare percentage.
EGX30_TWENTY_YEAR = {
    "egp_multiple": 8.25,
    "usd_multiple": 0.94,
    "years": 20,
    "note": ("The EGX30 rose about 8x in Egyptian pounds over twenty years and "
             "went slightly backwards in dollars. Both are true; they answer "
             "different questions."),
}


def to_usd(amount_egp: Optional[float], fx: Optional[float]) -> Optional[float]:
    """EGP -> USD at a given USD/EGP rate. None in, None out."""
    if amount_egp is None or fx is None or fx <= 0:
        return None
    return amount_egp / fx


def usd_return_pct(price_start_egp: Optional[float], price_end_egp: Optional[float],
                   fx_start: Optional[float], fx_end: Optional[float]) -> Optional[float]:
    """
    The same holding period, priced in dollars.

    NOT the EGP return minus some devaluation figure — that shortcut is wrong
    whenever either move is large, and on this currency both usually are. The
    correct form converts each endpoint and compares:

        (P_end / FX_end) / (P_start / FX_start) - 1

    Worked: a stock that doubled in EGP (100 -> 200) while USD/EGP went 30 -> 50
    returned +100% in pounds and only +20% in dollars. Subtracting a "40%
    devaluation" from +100% would have said +60%, which is wrong by 40 points.
    """
    if None in (price_start_egp, price_end_egp, fx_start, fx_end):
        return None
    if price_start_egp <= 0 or fx_start <= 0 or fx_end <= 0:
        return None
    start_usd = price_start_egp / fx_start
    end_usd = price_end_egp / fx_end
    if start_usd <= 0:
        return None
    return (end_usd / start_usd - 1.0) * 100.0


def currency_drag_pct(egp_return_pct: Optional[float],
                      usd_return_pct_value: Optional[float]) -> Optional[float]:
    """
    How many percentage points of an EGP gain the currency took back.

    Reported as a separate figure rather than folded into either return,
    because it is the part a user is most likely to be surprised by and least
    likely to derive themselves.
    """
    if egp_return_pct is None or usd_return_pct_value is None:
        return None
    return egp_return_pct - usd_return_pct_value


def dual_return(price_start_egp: Optional[float], price_end_egp: Optional[float],
                fx_start: Optional[float], fx_end: Optional[float]) -> dict:
    """
    Both views of one holding period, plus the gap between them.

    Always returns the dict, with None where FX history is missing, so a caller
    never has to branch on availability — a portfolio bought before the FX
    backfill reaches simply shows the EGP figure alone.
    """
    egp = None
    if (price_start_egp not in (None, 0) and price_end_egp is not None
            and price_start_egp > 0):
        egp = (price_end_egp / price_start_egp - 1.0) * 100.0

    usd = usd_return_pct(price_start_egp, price_end_egp, fx_start, fx_end)
    return {
        "egp_pct": round(egp, 2) if egp is not None else None,
        "usd_pct": round(usd, 2) if usd is not None else None,
        "currency_drag_pct": (round(currency_drag_pct(egp, usd), 2)
                              if egp is not None and usd is not None else None),
        "fx_start": fx_start,
        "fx_end": fx_end,
    }
