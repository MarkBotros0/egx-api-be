"""
Total-return prices: what a holder actually earned, dividends included.

THE GAP THIS CLOSES
-------------------
The cached price panel is split-adjusted but DIVIDEND-unadjusted — TradingView
adjusts splits by default and treats dividend adjustment as an opt-in the
vendored client does not set. So every return this app computes is a PRICE
return, and `score_risk_adjusted` compares it against the CBE policy rate, which
is a TOTAL return. Cash pays you its yield; a stock's dividends were being
thrown away before the comparison.

On a market whose dividend yields run to several percent that is not a rounding
error, and it biases in one direction: every stock looks worse against cash than
it was.

WHY THE ADJUSTMENT IS A RATIO CHAIN, NOT A SUBTRACTION
-------------------------------------------------------
A dividend does not reduce your wealth — it moves part of it from the share
price into your pocket. The reinvested-total-return convention therefore scales
the whole pre-ex-date history by the fraction of value that stayed in the price:

    factor = 1 - dividend / close_on_the_day_before_ex

applied CUMULATIVELY to every bar before that ex-date. Subtracting the cash
amount instead would break the series at every ex-date and misstate percentage
returns on a stock whose price level has changed a lot — which, after five EGP
devaluations, is all of them.

WHAT THIS IS DELIBERATELY NOT USED FOR
--------------------------------------
Charts and support/resistance stay on RAW close. A dividend-adjusted chart moves
every historical level and would silently rescore trend, support, resistance and
every pivot the app draws. Total return belongs in return COMPARISONS only.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def total_return_factors(close: pd.Series, dividends: dict) -> Optional[pd.Series]:
    """
    A per-bar multiplier turning raw closes into a total-return series.

    `dividends` is {ISO date: amount per share}. Returns a Series aligned to
    `close`, or None when there is nothing to apply.

    Each factor is applied to every bar STRICTLY BEFORE its ex-date, so the
    present-day level is untouched and history is lifted to match it. That
    direction matters: adjusting forward instead would change today's price,
    which the user can see on their broker screen.
    """
    if close is None or close.empty or not dividends:
        return None

    factors = pd.Series(1.0, index=close.index)
    for iso, amount in sorted(dividends.items()):
        try:
            ex = pd.Timestamp(iso)
            cash = float(amount)
        except (TypeError, ValueError):
            continue
        if cash <= 0:
            continue

        before = close.index[close.index < ex]
        if len(before) == 0:
            continue
        prior_close = float(close.loc[before[-1]])
        if prior_close <= 0 or cash >= prior_close:
            # A "dividend" at or above the whole share price is a data error,
            # not a payout. Skipping it is right: applying it would zero or
            # invert the factor and silently corrupt the entire prior history.
            continue

        factors.loc[before] *= (1.0 - cash / prior_close)

    return factors


def total_return_series(close: pd.Series,
                        dividends: dict) -> Optional[pd.Series]:
    """Raw closes restated as total return. None when nothing applies."""
    factors = total_return_factors(close, dividends)
    if factors is None:
        return None
    return close * factors


def annualised_drag_pct(close: pd.Series, dividends: dict) -> Optional[float]:
    """
    Percentage points per year that ignoring dividends costs a return figure.

    This is the number worth stating on screen. It is the gap between the two
    CAGRs, not the dividend yield: reinvestment compounds, so over a long window
    the drag exceeds the average yield.
    """
    tr = total_return_series(close, dividends)
    if tr is None or len(close) < 2:
        return None
    years = (close.index[-1] - close.index[0]).days / 365.25
    if years <= 0:
        return None
    start_price, end_price = float(close.iloc[0]), float(close.iloc[-1])
    start_tr, end_tr = float(tr.iloc[0]), float(tr.iloc[-1])
    if start_price <= 0 or start_tr <= 0 or end_price <= 0 or end_tr <= 0:
        return None
    price_cagr = (end_price / start_price) ** (1 / years) - 1
    total_cagr = (end_tr / start_tr) ** (1 / years) - 1
    return round((total_cagr - price_cagr) * 100.0, 2)
