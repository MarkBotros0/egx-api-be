"""
Auto-drawn trendlines — the diagonal through the last few pivot lows (the
support trendline) and pivot highs (the resistance trendline).

The rule that shapes everything here: NO LINE IS BETTER THAN A WRONG LINE. A
side is drawn only when the last three pivots line up in one direction and
price has respected the line since the first of them. Everything else is None
and the chart shows exactly what it showed before.

Pure functions, no DB, no network — tests/ has no DB fixture by design.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.indicators import find_pivots, support_resistance
from app.core.trendlines import compute_trendlines


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------

def _frame(closes):
    """Build high/low/close series with a DatetimeIndex, like the analysis path.

    High sits 1% above the close and low 1% below, so pivot detection sees the
    same turning points the close series has.
    """
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2025-01-01", periods=len(closes))
    close = pd.Series(closes, index=idx)
    return close * 1.01, close * 0.99, close


def _zigzag(start, legs, seg_len):
    """Piecewise-linear path: each leg is a % move spread over seg_len bars.

    legs=[+12, -6, +12, -6] with seg_len=25 makes a rising zigzag whose lows
    are 50 bars apart — comfortably more than the 20-bar pivot window either
    side, so every turning point is a detectable pivot.
    """
    out = [float(start)]
    for pct in legs:
        base = out[-1]
        step = base * (pct / 100.0) / seg_len
        out.extend([base + step * (k + 1) for k in range(seg_len)])
    return out


# ---------------------------------------------------------------------------
# find_pivots — the shared scan, now keeping the bar index
# ---------------------------------------------------------------------------

def test_find_pivots_returns_indexed_turning_points_of_a_zigzag():
    high, low, close = _frame(_zigzag(100, [+12, -6, +12, -6, +12, -6], 25))
    pivots = find_pivots(high, low, window=20)

    # Highs land at the end of each up-leg (bars 25, 75, 125); lows at the end
    # of each down-leg (bars 50, 100, 150 — 150 has no 20 bars after it, so the
    # detectable lows are 50 and 100).
    assert [i for i, _ in pivots["highs"]] == [25, 75, 125]
    assert [i for i, _ in pivots["lows"]] == [50, 100]

    # And the price at each index is that bar's own extreme, not the close.
    for i, price in pivots["lows"]:
        assert price == pytest.approx(float(low.iloc[i]))
    for i, price in pivots["highs"]:
        assert price == pytest.approx(float(high.iloc[i]))


def test_support_resistance_is_unchanged_by_the_pivot_refactor():
    """Characterisation guard: the horizontal S/R levels the chart, the Key
    Levels card and the entry/exit zones all read must come out identical
    after `support_resistance` starts delegating to `find_pivots`."""
    high, low, close = _frame(_zigzag(100, [+12, -6, +12, -6, +12, -6], 25))
    sr = support_resistance(high, low, close)

    # Two detectable lows (bars 50 and 100) sit 12% apart, so they do not
    # cluster: two support levels of strength 1, sorted by strength then as
    # inserted. Three highs, likewise three resistances.
    assert [s["strength"] for s in sr["supports"]] == [1, 1]
    assert [r["strength"] for r in sr["resistances"]] == [1, 1, 1]
    assert sr["supports"][0]["price"] == pytest.approx(round(float(low.iloc[50]), 2))
    assert sr["resistances"][0]["price"] == pytest.approx(round(float(high.iloc[25]), 2))


# ---------------------------------------------------------------------------
# compute_trendlines — the diagonal through the last three pivots
# ---------------------------------------------------------------------------

RISING = _zigzag(100, [+12, -6] * 4, 25)      # 201 bars: lows 50/100/150, highs 25/75/125/175
FALLING = _zigzag(100, [-12, +6] * 4, 25)     # mirror: highs 50/100/150, lows 25/75/125/175


def _line_at(side, i):
    return side["values"][i]


def test_rising_lows_draw_a_support_trendline_touching_the_anchors():
    high, low, close = _frame(RISING)
    tl = compute_trendlines(high, low, close, n_out=len(close))
    sup = tl["support"]

    assert sup is not None
    assert sup["direction"] == "up"
    assert [a["date"] for a in sup["anchors"]] == [str(close.index[i])[:10] for i in (50, 100, 150)]
    assert len(sup["values"]) == len(close)

    # Nothing is drawn before the first anchor; the line runs from it to the
    # last bar, extended past the newest pivot.
    assert sup["values"][49] is None
    assert sup["values"][50] is not None
    assert sup["values"][-1] is not None

    # A trendline TOUCHES its pivots — every anchor sits on or above the line,
    # and at least one sits exactly on it. Running the fit through the middle
    # of the points would put a "support" above the very lows it claims.
    gaps = [float(low.iloc[i]) - _line_at(sup, i) for i in (50, 100, 150)]
    assert all(g >= -1e-9 for g in gaps)
    assert min(gaps) == pytest.approx(0.0, abs=1e-9)

    # Price respected it: no close since the first anchor sits below the line.
    for i in range(50, len(close)):
        assert float(close.iloc[i]) >= _line_at(sup, i) * 0.99


def test_rising_highs_draw_a_resistance_trendline_over_the_anchors():
    high, low, close = _frame(RISING)
    res = compute_trendlines(high, low, close, n_out=len(close))["resistance"]

    assert res is not None
    assert res["direction"] == "up"
    assert [a["date"] for a in res["anchors"]] == [str(close.index[i])[:10] for i in (75, 125, 175)]
    gaps = [_line_at(res, i) - float(high.iloc[i]) for i in (75, 125, 175)]
    assert all(g >= -1e-9 for g in gaps)
    assert min(gaps) == pytest.approx(0.0, abs=1e-9)


def test_a_downtrend_draws_both_lines_pointing_down():
    high, low, close = _frame(FALLING)
    tl = compute_trendlines(high, low, close, n_out=len(close))

    assert tl["resistance"]["direction"] == "down"
    assert tl["support"]["direction"] == "down"
    assert tl["resistance"]["slope_pct_per_bar"] < 0
    assert tl["support"]["slope_pct_per_bar"] < 0


def test_equal_lows_are_a_horizontal_level_not_a_trendline():
    # +10% then back to exactly the start: every low is 100, every high 110.
    # The chart already draws those as horizontal S/R; a "trendline" through
    # three identical lows would be the same line drawn twice.
    flat = _zigzag(100, [+10, -(1 - 1 / 1.1) * 100] * 4, 25)
    high, low, close = _frame(flat)
    tl = compute_trendlines(high, low, close, n_out=len(close))
    assert tl["support"] is None
    assert tl["resistance"] is None


def test_pivots_that_do_not_line_up_draw_nothing():
    # Lows go 105 -> 94 -> 99: down then up. No line beats a wrong line.
    high, low, close = _frame(_zigzag(100, [+12, -6, +12, -20, +12, -6, +12, -6], 25))
    tl = compute_trendlines(high, low, close, n_out=len(close))
    assert tl["support"] is None
    assert tl["resistance"] is None


def test_a_broken_trendline_is_not_drawn():
    # Three clean rising lows, then price collapses 30% and stays under the
    # line for forty bars. The pivots still line up; the market stopped
    # respecting them.
    broken = _zigzag(100, [+12, -6] * 4 + [-30], 40)
    high, low, close = _frame(broken)
    tl = compute_trendlines(high, low, close, n_out=len(close))
    assert tl["support"] is None
    # The highs are untouched by the collapse and price stayed under them, so
    # the resistance line is still a fair description.
    assert tl["resistance"] is not None


def test_fewer_than_three_pivots_draw_nothing():
    high, low, close = _frame(_zigzag(100, [+12, -6, +12, -6], 25))
    tl = compute_trendlines(high, low, close, n_out=len(close))
    assert tl["support"] is None
    assert tl["resistance"] is None


def test_values_are_aligned_to_the_displayed_window_and_extended_across_it():
    # The chart shows the last 60 bars; the anchors were found on the full
    # frame. The line still covers every displayed bar, because it is
    # extended from anchors that scrolled off the left edge.
    high, low, close = _frame(RISING)
    tl = compute_trendlines(high, low, close, n_out=60)
    sup = tl["support"]
    assert len(sup["values"]) == 60
    assert all(v is not None for v in sup["values"])
    # And the value at the last displayed bar is the same line evaluated at
    # the same bar as the full-window call — trimming moves nothing.
    full = compute_trendlines(high, low, close, n_out=len(close))["support"]
    assert sup["values"][-1] == pytest.approx(full["values"][-1])
    assert sup["anchors"] == full["anchors"]


# ---------------------------------------------------------------------------
# Wiring — /api/analysis serves it, and nothing scores on it
# ---------------------------------------------------------------------------

import ast
import re
import pathlib

_APP = pathlib.Path(__file__).resolve().parents[1] / "app"


def _calls_in(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }


def test_analysis_serves_trendlines_aligned_to_the_displayed_bars():
    src = (_APP / "routers" / "analysis.py").read_text(encoding="utf-8")
    assert "compute_trendlines" in _calls_in(_APP / "routers" / "analysis.py")
    assert '"trendlines":' in src, "the response dict must carry the trendlines field"


def test_trendlines_are_presentation_only_and_never_a_scoring_input():
    """A trendline is a drawing aid. Feeding it into the composite would make
    it a scoring input on one page and not the others — the exact divergence
    One Score Per Stock exists to prevent — and would hand the score a
    directional call the backtest gives it no right to make."""
    for rel in ("core/composite.py", "core/extras_builder.py", "routers/portfolio_analysis.py"):
        src = (_APP / rel).read_text(encoding="utf-8")
        assert "trendline" not in src.lower(), f"{rel} must not read trendlines"


# ---------------------------------------------------------------------------
# Frontend — the chart draws it, the page toggles it, the Learn page explains it
#
# The frontend is a SEPARATE repository; a backend-only checkout has no copy.
# Skip rather than fail in that case, as test_forecast_presentation does.
# ---------------------------------------------------------------------------

_FE = (pathlib.Path(__file__).resolve().parents[2] / "egx-api-fe" / "src" / "app")


def _fe(rel):
    path = _FE / rel
    if not path.exists():
        pytest.skip(f"frontend checkout not present: {path}")
    return path.read_text(encoding="utf-8")


def test_the_price_chart_draws_both_trendlines_behind_one_overlay_toggle():
    src = _fe("components/PriceChart.tsx")
    assert 'dataKey="trend_support"' in src
    assert 'dataKey="trend_resistance"' in src
    assert "overlays.trendlines" in src, "trendlines must be an overlay like SMA 20, not always-on"


def test_the_stock_page_feeds_the_chart_and_defaults_the_toggle_on():
    src = _fe("stock/[symbol]/page.tsx")
    assert "trendlines: true" in src, "the pill is on by default — the user asked to see it"
    assert "trend_support:" in src and "trend_resistance:" in src
    assert "data.trendlines" in src


def test_types_declare_the_field():
    src = _fe("lib/types.ts")
    assert re.search(r"trendlines\??:\s*TrendLines", src), "AnalysisResponse must carry trendlines: TrendLines"


def test_the_learn_page_has_the_anchor_the_tooltip_links_to():
    """Anchors are a public contract: the overlay tooltip deep-links
    /learn#trendlines, and a renamed id silently breaks an in-app link."""
    src = _fe("learn/curriculum.tsx")
    assert 'id: "trendlines"' in src


def test_the_pivot_window_is_one_number_everywhere():
    """indicators.py deliberately imports nothing from app, so it cannot read
    PIVOT_WINDOW_BARS and restates 20 as a default instead. The trendline fit
    passes the constant explicitly. Three spellings of one number is a drift
    waiting to happen — pin them."""
    import inspect
    from app.core.constants import PIVOT_WINDOW_BARS
    assert inspect.signature(find_pivots).parameters["window"].default == PIVOT_WINDOW_BARS
    assert inspect.signature(support_resistance).parameters["window"].default == PIVOT_WINDOW_BARS
    assert inspect.signature(compute_trendlines).parameters["window"].default == PIVOT_WINDOW_BARS
