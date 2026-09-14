"""
Auto-drawn trendlines — the diagonal through the last few pivot lows (the
support trendline) and pivot highs (the resistance trendline).

In an uptrend that is the line under the higher lows and the line over the
higher highs; in a downtrend the lines over the lower highs and under the
lower lows. It is the line a chartist draws by hand, computed from the same
pivots `support_resistance` already finds for the horizontal levels.

THE GOVERNING RULE IS NO LINE BEATS A WRONG LINE. A side is drawn only when

  1. it has at least TRENDLINE_PIVOTS pivots and the newest of them line up
     in ONE direction, each at least TRENDLINE_MIN_STEP_PCT past the last —
     equal lows are a horizontal level, which the chart already draws;
  2. the line is fitted through those pivots and then shifted so it TOUCHES
     them (on or under every low, on or over every high). A least-squares
     line run through the middle of three lows would put a "support" above
     the very lows it claims;
  3. price has respected it: from the first anchor to today, no more than
     TRENDLINE_MAX_VIOLATION_SHARE of closes cross the line by more than
     TRENDLINE_TOLERANCE_PCT. A choppy stock fails this and gets nothing.

Anything else is None and the chart shows exactly what it showed before.

Purely presentation. Nothing here feeds the composite score or a signal, so
it cannot disturb One Score Per Stock; it is not built into `extras`.

Pure functions, no DB, no network.
"""

import numpy as np
import pandas as pd

from app.core.constants import (
    PIVOT_WINDOW_BARS,
    TRENDLINE_MAX_VIOLATION_SHARE,
    TRENDLINE_MIN_STEP_PCT,
    TRENDLINE_PIVOTS,
    TRENDLINE_TOLERANCE_PCT,
)
from app.core.indicators import find_pivots


def compute_trendlines(high: pd.Series, low: pd.Series, close: pd.Series,
                       n_out: int, window: int = PIVOT_WINDOW_BARS) -> dict:
    """
    Fit the support and resistance trendlines over the FULL frame and report
    them aligned to the last `n_out` bars — the ones the chart displays.

    Computed on the full frame like `support_resistance`, so the anchors do
    not move when the user changes the 60/100/200/500 bar selector; only the
    drawn portion does. A line whose anchors scrolled off the left edge is
    still drawn across the whole window, extended from them.

    Returns {"support": side | None, "resistance": side | None} where side is
      {
        "values":  [float | None] * n_out   — None before the first anchor,
                                              the line's price on every bar
                                              from it to the last, extended,
        "anchors": [{"date", "price"}, ...] — the pivots the line rests on,
        "direction": "up" | "down",
        "slope_pct_per_bar": float,         — rise per bar as a % of the
                                              line's value at the last bar
      }
    """
    pivots = find_pivots(high, low, window)
    closes = np.asarray(close.values, dtype=float)
    dates = [str(idx)[:10] for idx in close.index]

    return {
        "support": _fit_side(pivots["lows"], closes, dates, n_out, side="support"),
        "resistance": _fit_side(pivots["highs"], closes, dates, n_out, side="resistance"),
    }


def _fit_side(pivots, closes, dates, n_out, side):
    if len(pivots) < TRENDLINE_PIVOTS:
        return None

    anchors = pivots[-TRENDLINE_PIVOTS:]
    direction = _direction([price for _, price in anchors])
    if direction is None:
        return None

    xs = np.array([i for i, _ in anchors], dtype=float)
    ys = np.array([price for _, price in anchors], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)

    # Shift the fitted line so it TOUCHES the pivots rather than running
    # through their middle: under every low for support, over every high for
    # resistance. After this exactly one anchor sits on the line and the rest
    # sit on the correct side of it.
    resid = ys - (slope * xs + intercept)
    intercept += resid.min() if side == "support" else resid.max()

    n = len(closes)
    start = int(xs[0])
    idx = np.arange(start, n)
    line = slope * idx + intercept
    if np.any(line <= 0):
        return None

    # Has price respected it? A close through the line by more than the
    # tolerance is a violation; too many of them and the market has stopped
    # honouring this line, whatever the pivots say. NaN closes compare False
    # on both sides and so never count.
    tol = TRENDLINE_TOLERANCE_PCT / 100.0
    if side == "support":
        violations = closes[start:] < line * (1 - tol)
    else:
        violations = closes[start:] > line * (1 + tol)
    if violations.mean() > TRENDLINE_MAX_VIOLATION_SHARE:
        return None

    out_start = n - n_out
    values = [
        float(slope * i + intercept) if i >= start else None
        for i in range(out_start, n)
    ]
    last_value = slope * (n - 1) + intercept

    return {
        "values": values,
        "anchors": [{"date": dates[i], "price": float(price)} for i, price in anchors],
        "direction": direction,
        "slope_pct_per_bar": float(slope / last_value * 100),
    }


def _direction(prices):
    """'up' if every step rises by at least the minimum, 'down' if every step
    falls by at least it, None otherwise. Equal or mixed pivots are not a
    trend."""
    min_step = TRENDLINE_MIN_STEP_PCT / 100.0
    steps = [b / a - 1 for a, b in zip(prices[:-1], prices[1:]) if a > 0]
    if len(steps) != len(prices) - 1:
        return None
    if all(s >= min_step for s in steps):
        return "up"
    if all(s <= -min_step for s in steps):
        return "down"
    return None
