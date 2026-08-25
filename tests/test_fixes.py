"""
Regression tests for the analysis-correctness audit.

Each test pins down a defect that produced a wrong or self-contradicting
number in the UI. The comments say what the user used to see, so a future
change that reintroduces the bug fails with an explanation rather than just
a red assertion.

Run from egx-api-be:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from app.core import composite as C
from app.core.composite import (
    apply_macro_modulation,
    classify_signal,
    compute_composite,
    score_quality,
    score_risk_adjusted,
    score_volume,
    weights_hash,
)
from app.core.constants import (
    DIVERGENCE_LOOKBACK_FULL,
    SCORE_BUY_MAX,
    SCORE_HOLD_MAX,
    SCORE_SELL_MAX,
    SCORE_STRONG_SELL_MAX,
    STOP_LOSS_ATR_MULTIPLIER,
)
from app.core.extras_builder import build_composite_extras
from app.core.index_membership import get_index_membership
from app.core.indicators import (
    _cluster_levels,
    atr,
    compute_all,
    fibonacci_levels,
    liquidity_score,
    rsi,
)
from app.core.levels import compute_entry_exit, compute_key_levels


# ---------------------------------------------------------------------------
# Fixtures — deterministic synthetic market data
# ---------------------------------------------------------------------------

def _make_ohlcv(n=400, seed=7, start=100.0, drift=0.03):
    """A reproducible OHLCV frame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 1.0, n)
    close = start + np.cumsum(steps)
    close = np.maximum(close, 1.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    high = close + np.abs(rng.normal(0, 0.6, n))
    low = close - np.abs(rng.normal(0, 0.6, n))
    return pd.DataFrame(
        {
            "open": close - rng.normal(0, 0.2, n),
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(50_000, 500_000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def stock_df():
    return _make_ohlcv()


@pytest.fixture(scope="module")
def bench_close():
    return _make_ohlcv(seed=21, start=20_000.0, drift=1.5)["close"]


# ---------------------------------------------------------------------------
# Signal bands — backend is canon, lower bound inclusive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "Very Weak"),
        (19.99, "Very Weak"),
        (20, "Weak"),          # exactly 20 is Weak, NOT Very Weak
        (39.99, "Weak"),
        (40, "Neutral"),
        (59.99, "Neutral"),
        (60, "Strong"),        # exactly 60 is Strong, NOT Neutral
        (79.99, "Strong"),
        (80, "Very Strong"),
        (100, "Very Strong"),
    ],
)
def test_classify_signal_boundaries(score, expected):
    assert classify_signal(score) == expected


def test_labels_describe_condition_not_action():
    """
    The labels used to read Strong Buy / Buy / Hold / Sell / Strong Sell. The
    backtest found the score cannot rank stocks — nine of ten deciles had a
    median 21-day forward return of 0.00%, and among liquid names the "Sell"
    bucket slightly beat the "Buy" bucket. An instruction the evidence
    contradicts is worse than no instruction, so the labels now describe
    condition. This test exists so nobody reinstates them casually.
    """
    labels = {classify_signal(s) for s in (0, 25, 50, 70, 90)}
    forbidden = {"Buy", "Sell", "Strong Buy", "Strong Sell", "Hold"}
    assert not (labels & forbidden), (
        f"composite labels give trading instructions again: {labels & forbidden}"
    )


def test_band_constants_are_ordered():
    """The frontend mirrors these; a reordering would silently skew bands."""
    assert SCORE_STRONG_SELL_MAX < SCORE_SELL_MAX < SCORE_HOLD_MAX < SCORE_BUY_MAX


# ---------------------------------------------------------------------------
# RSI / ATR — Wilder smoothing
# ---------------------------------------------------------------------------

def test_rsi_uses_wilder_smoothing(stock_df):
    """
    RSI must use alpha = 1/period. With span=period (alpha = 2/(period+1))
    it smooths ~1.9x too fast, deviating by up to 14 RSI points and
    misclassifying overbought/oversold against every charting platform.
    """
    close = stock_df["close"]
    period = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    expected = 100 - 100 / (1 + avg_gain / avg_loss)

    got = rsi(close, period)
    assert np.nanmax(np.abs(got - expected)) < 1e-9


def test_rsi_differs_from_old_span_ema(stock_df):
    """Guard against a silent revert to the span-based formula."""
    close = stock_df["close"]
    period = 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    old = 100 - 100 / (
        1
        + gain.ewm(span=period, adjust=False).mean()
        / loss.ewm(span=period, adjust=False).mean()
    )
    assert np.nanmax(np.abs(rsi(close, period) - old)) > 1.0


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(100, 140, dtype=float))
    assert rsi(close, 14).iloc[-1] == pytest.approx(100.0)


def test_atr_matches_wilder(stock_df):
    """
    ATR must be Wilder-smoothed so it agrees with the ATR computed inside
    adx(). A simple rolling mean drifted from it by up to ~40%, and ATR
    drives every suggested stop-loss.
    """
    high, low, close = stock_df["high"], stock_df["low"], stock_df["close"]
    period = 14
    prev_close = close.shift(1)
    tr = pd.DataFrame(
        {
            "hl": high - low,
            "hc": (high - prev_close).abs(),
            "lc": (low - prev_close).abs(),
        }
    ).max(axis=1)
    expected = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    assert np.nanmax(np.abs(atr(high, low, close, period) - expected)) < 1e-9


# ---------------------------------------------------------------------------
# Fibonacci direction
# ---------------------------------------------------------------------------

def test_fibonacci_downtrend_anchors_from_low():
    """
    In a down-swing the retracement is measured UP from the low. Always
    retracing down from the high mislabeled every level — what the UI called
    the 61.8% level was actually the 38.2%.
    """
    down = pd.Series(np.linspace(100, 50, 80))
    fib = fibonacci_levels(down + 0.5, down - 0.5, lookback=60)

    assert fib["direction"] == "down"
    lo, hi = fib["low"], fib["high"]
    assert fib["levels"]["61.8%"] == pytest.approx(lo + 0.618 * (hi - lo), abs=0.02)
    assert fib["levels"]["0%"] == pytest.approx(lo, abs=0.02)
    assert fib["levels"]["100%"] == pytest.approx(hi, abs=0.02)


def test_fibonacci_uptrend_anchors_from_high():
    up = pd.Series(np.linspace(50, 100, 80))
    fib = fibonacci_levels(up + 0.5, up - 0.5, lookback=60)

    assert fib["direction"] == "up"
    lo, hi = fib["low"], fib["high"]
    assert fib["levels"]["61.8%"] == pytest.approx(hi - 0.618 * (hi - lo), abs=0.02)
    assert fib["levels"]["0%"] == pytest.approx(hi, abs=0.02)


# ---------------------------------------------------------------------------
# Level clustering
# ---------------------------------------------------------------------------

def test_cluster_levels_does_not_chain():
    """
    Comparing each candidate to the LAST member let a ladder of levels each
    just under the threshold merge into one cluster of unbounded width —
    reporting a price nothing bounced at, with a fabricated strength that
    then unlocked "high confidence" entry signals.
    """
    ladder = [100.0, 101.5, 103.0, 104.5, 106.0, 107.5, 109.0]
    clusters = _cluster_levels(ladder, threshold=0.02)

    # The ladder spans 9%; it must not become a single "2% cluster".
    assert len(clusters) > 1
    # Strength is rendered as "Tested Nx" and gates high-confidence signals,
    # so an inflated count manufactures conviction that isn't there.
    assert max(c["strength"] for c in clusters) <= 2
    # Every reported level must be a price the stock actually visited,
    # not the centroid of a wide chain.
    for c in clusters:
        assert min(abs(c["price"] - lv) / lv for lv in ladder) < 0.02


def test_cluster_levels_still_groups_genuine_neighbours():
    clusters = _cluster_levels([100.0, 100.2, 100.4], threshold=0.02)
    assert len(clusters) == 1
    assert clusters[0]["strength"] == 3


# ---------------------------------------------------------------------------
# Macro modulation
# ---------------------------------------------------------------------------

def test_sideways_macro_is_a_noop():
    """A neutral market must leave scores alone; otherwise every stock
    carries a permanent penalty and no regime is ever neutral."""
    macro = {"egx30": {"trend": "sideways"}}
    for s in (85, 65, 50, 35, 15):
        adjusted, delta, _ = apply_macro_modulation(s, macro)
        assert adjusted == s
        assert delta == 0.0


def test_bearish_macro_shifts_scores_down():
    macro = {"egx30": {"trend": "bearish"}}
    assert apply_macro_modulation(85, macro)[0] < 85   # damped toward neutral
    assert apply_macro_modulation(15, macro)[0] < 15   # pushed further down
    assert apply_macro_modulation(50, macro)[0] == 50  # neutral is a fixed point


def test_bullish_macro_is_a_noop():
    macro = {"egx30": {"trend": "bullish"}}
    assert apply_macro_modulation(65, macro) == (65, 0.0, None)


# ---------------------------------------------------------------------------
# Risk-free rate / weights hash
# ---------------------------------------------------------------------------

def test_zero_risk_free_rate_is_not_swallowed():
    """`extras.get(...) or 25.0` treated a legitimate 0% rate as missing."""
    at_zero, _ = score_risk_adjusted(10.0, 0.0, 20.0, 3.0, 200)
    at_default, _ = score_risk_adjusted(10.0, 25.0, 20.0, 3.0, 200)
    assert at_zero != at_default
    assert at_zero > at_default  # beating a 0% rate should score better


def test_weights_hash_distinguishes_small_tweaks():
    """Rounding to int collided distinct weight sets onto one cache key, so
    nudging a slider served a stale score."""
    base = {
        "trend": 18.2, "momentum": 15, "volume": 12, "volatility": 10,
        "divergence": 8, "quality": 12, "risk_adjusted": 13,
        "relative_strength": 11.8,
    }
    tweaked = dict(base, trend=18.4, relative_strength=11.6)
    assert weights_hash(base) != weights_hash(tweaked)


# ---------------------------------------------------------------------------
# Quality drawdown units
# ---------------------------------------------------------------------------

def test_quality_drawdown_is_monotonic():
    """
    Sniffing the magnitude to guess percent-vs-fraction made a -0.9%
    drawdown read as -90% and score WORSE than a -1.5% one.
    """
    scores = [score_quality(None, None, dd)[0] for dd in (-0.005, -0.02, -0.10, -0.25, -0.40)]
    assert scores == sorted(scores, reverse=True)


def test_small_drawdown_reads_as_near_peak():
    score, reasons = score_quality(None, None, -0.009)  # -0.9%
    text = " ".join(reasons).lower()
    assert "severe" not in text
    assert score >= 50


# ---------------------------------------------------------------------------
# Stop-loss convention — ONE number app-wide
# ---------------------------------------------------------------------------

def test_entry_zone_stop_loss_uses_the_house_convention():
    """
    One stop-loss convention app-wide: STOP_LOSS_ATR_MULTIPLIER x ATR below
    the nearest support. This used to be asserted against the Max Buy Price
    card too, which showed a different multiplier on the same screen; that
    card has since been removed (it rejected breakouts as "wait for a
    pullback"), so the entry zone is now the single surface for this number.
    """
    support, atr_val, price = 100.0, 3.0, 102.0
    sr = {
        "supports": [{"price": support, "strength": 4}],
        "resistances": [{"price": 130.0, "strength": 3}],
    }
    zones = compute_entry_exit(price, sr, rsi_latest=45, stoch_k_latest=40, atr_latest=atr_val)

    expected = round(support - STOP_LOSS_ATR_MULTIPLIER * atr_val, 2)
    assert zones["entry_zone"]["suggested_stop_loss"] == expected


def test_max_buy_price_helper_is_gone():
    """
    The card systematically told the user to wait on breakouts: it computed
    reward as (nearest_resistance - price), which is negative for a stock
    making new highs, so the strongest setups were rejected. Removed rather
    than tuned, because computing a reward above the market would mean
    inventing a price target, which this app deliberately does not do.
    """
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.core.entry_price")


# ---------------------------------------------------------------------------
# Entry / exit zone bands must contain the price that activated them
# ---------------------------------------------------------------------------

def test_active_entry_band_contains_current_price():
    """
    The zone activated up to 5% above support but the band only reached
    2.5%, so the card could say "buy 100.00-102.50" while the live price
    ticker beside it read 104.00.
    """
    support = 100.0
    sr = {"supports": [{"price": support, "strength": 4}], "resistances": []}
    for price in (100.5, 102.0, 104.0, 104.8):
        zones = compute_entry_exit(price, sr, rsi_latest=45, stoch_k_latest=40, atr_latest=3.0)
        entry = zones["entry_zone"]
        if entry["active"]:
            band = entry["price_range"]
            assert band["low"] <= price <= band["high"], (
                f"price {price} outside active buy band {band}"
            )


def test_active_exit_band_contains_current_price_after_breakout():
    """Exit stays active up to 1% above resistance; the trim band must
    reach the price rather than sitting entirely below the market."""
    resistance = 100.0
    sr = {"supports": [], "resistances": [{"price": resistance, "strength": 4}]}
    price = 100.8
    zones = compute_entry_exit(price, sr, rsi_latest=78, stoch_k_latest=85, atr_latest=2.0)
    exit_zone = zones["exit_zone"]
    if exit_zone["active"]:
        band = exit_zone["price_range"]
        assert band["low"] <= price <= band["high"]


def test_zone_bands_are_low_to_high():
    sr = {
        "supports": [{"price": 95.0, "strength": 3}],
        "resistances": [{"price": 105.0, "strength": 3}],
    }
    zones = compute_entry_exit(97.0, sr, rsi_latest=45, stoch_k_latest=40, atr_latest=2.0)
    for z in zones.values():
        if z.get("price_range"):
            assert z["price_range"]["low"] <= z["price_range"]["high"]


# ---------------------------------------------------------------------------
# Key levels — nearest, not strongest
# ---------------------------------------------------------------------------

def test_key_levels_picks_nearest_not_strongest():
    """
    support_resistance() returns levels sorted by STRENGTH. Reading index 0
    as "nearest" produced a red "broke below support at 70.00" alert while
    the Key Levels card on the same row correctly showed support at 98.00.
    """
    sr = {
        "supports": [
            {"price": 70.0, "strength": 9},   # strongest, but far away
            {"price": 98.0, "strength": 3},   # actually nearest
            {"price": 92.0, "strength": 2},
        ],
        "resistances": [
            {"price": 150.0, "strength": 8},
            {"price": 104.0, "strength": 2},
        ],
    }
    levels = compute_key_levels(100.0, sr)
    assert levels["nearest_support"]["price"] == 98.0
    assert levels["nearest_resistance"]["price"] == 104.0
    assert levels["nearest_support"]["distance_pct"] < 0   # below price
    assert levels["nearest_resistance"]["distance_pct"] > 0  # above price


# ---------------------------------------------------------------------------
# THE headline test: one score per stock, whatever page you open
# ---------------------------------------------------------------------------

def test_all_three_paths_build_identical_extras(stock_df, bench_close):
    """
    The dashboard card, the stock detail page and the portfolio row must
    score a stock identically. They used to hand-roll their own `extras`:
    the batch path omitted the inputs for quality / risk-adjusted /
    relative-strength, so those three scorers returned None and the score
    renormalized over 5 of 8 categories — dropping exactly the punitive
    ones. Measured on real data: 66 "Buy" on the card, 45 "Hold" on the
    detail page for the same stock.
    """
    indicators = compute_all(stock_df)
    kwargs = dict(
        egx30_close=bench_close,
        include_multi_timeframe=True,
        risk_free_rate_pct=25.0,
        pe_ratio=14.0,
        dividend_yield=3.1,
        loss_making=False,
        index_membership="EGX30",
        divergence_lookback=DIVERGENCE_LOOKBACK_FULL,
    )

    detail = build_composite_extras(stock_df, indicators, **kwargs)["extras"]
    batch = build_composite_extras(stock_df, indicators, **kwargs)["extras"]
    portfolio = build_composite_extras(stock_df, indicators, **kwargs)["extras"]

    assert detail.keys() == batch.keys() == portfolio.keys()
    for key in detail:
        assert repr(detail[key]) == repr(batch[key]) == repr(portfolio[key]), (
            f"extras['{key}'] differs between paths"
        )


def test_the_old_lightweight_batch_extras_would_now_fail_loudly(stock_df):
    """
    Pins the actual regression. The batch path's old extras dict omitted the
    quality / risk-adjusted / relative-strength inputs; this reproduces it
    and shows the damage — categories dropped and the score materially
    different from the full-input score on identical market data.
    """
    indicators = compute_all(stock_df)
    close = stock_df["close"]

    old_batch_extras = {
        "current_price": float(close.iloc[-1]),
        "divergences": {"rsi": {}, "macd": {}},
        "volume_price": None,
        "bb_squeeze": False,
        "obv_rising": None,
        "price_rising_20d": None,
        "golden_cross_active": False,
        "history_days": len(close),
    }
    crippled = compute_composite(indicators, extras=old_batch_extras)
    dropped = [n for n, c in crippled["categories"].items() if c["score"] is None]

    # The three categories that used to vanish from dashboard scores.
    assert {"quality", "risk_adjusted", "relative_strength"} <= set(dropped)

    # And the remaining categories absorbed their weight, so trend carried
    # far more than the 18% the weights modal advertises.
    trend = crippled["categories"]["trend"]
    assert trend["effective_weight"] > trend["weight"] * 1.4


def test_no_router_hand_rolls_composite_extras():
    """
    Every composite call must route through build_composite_extras. A
    hand-built `extras={...}` literal in a router is how the three paths
    drifted apart in the first place.
    """
    import pathlib
    import re

    routers = pathlib.Path(__file__).resolve().parents[1] / "app" / "routers"
    offenders = []
    for path in routers.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        # `extras={` followed by anything other than a closing brace means a
        # dict literal is being constructed inline.
        if re.search(r"extras\s*=\s*\{\s*[^}]", src):
            offenders.append(path.name)
    assert not offenders, (
        f"routers building composite extras by hand: {offenders} — "
        "use core.extras_builder.build_composite_extras instead"
    )


def test_all_three_call_sites_pass_the_same_builder_kwargs():
    """
    The three scoring paths must hand build_composite_extras the SAME set of
    named inputs.

    test_all_three_paths_build_identical_extras only proves the builder is
    deterministic — it calls one function three times with identical kwargs.
    It cannot catch a call site that FORGETS a kwarg, which is exactly how the
    66-"Buy"-on-the-card / 45-"Hold"-on-the-detail-page divergence happened:
    the batch path omitted the inputs the punitive categories needed.

    Every new scoring input multiplies that risk, so compare the call sites at
    the source level.
    """
    import ast
    import pathlib

    routers = pathlib.Path(__file__).resolve().parents[1] / "app" / "routers"
    call_sites = []
    for path in (routers / "analysis.py", routers / "portfolio_analysis.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "build_composite_extras"):
                call_sites.append(
                    (path.name, node.lineno,
                     frozenset(kw.arg for kw in node.keywords if kw.arg))
                )

    assert len(call_sites) == 3, (
        f"expected 3 build_composite_extras call sites (detail, batch, "
        f"portfolio), found {len(call_sites)}: "
        f"{[(f, ln) for f, ln, _ in call_sites]}"
    )

    distinct = {kwargs for _, _, kwargs in call_sites}
    if len(distinct) > 1:
        shared = frozenset.intersection(*distinct)
        detail = "\n".join(
            f"  {f}:{ln} extra={sorted(kw - shared)}"
            for f, ln, kw in call_sites
        )
        raise AssertionError(
            "scoring paths disagree on which inputs they pass — the dropped "
            f"input's category will score differently per page:\n{detail}"
        )


def test_all_eight_categories_score_with_full_inputs(stock_df, bench_close):
    """
    Every category must be scorable from the shared builder's output. If one
    silently returns None, its weight is redistributed onto the others and
    the score drifts from what the weights modal says it should be.
    """
    indicators = compute_all(stock_df)
    built = build_composite_extras(
        stock_df, indicators,
        egx30_close=bench_close,
        risk_free_rate_pct=25.0,
        pe_ratio=12.0,
        dividend_yield=3.1,
        loss_making=False,
        index_membership="EGX30",
    )
    result = compute_composite(indicators, extras=built["extras"])

    unscored = [name for name, cat in result["categories"].items() if cat["score"] is None]
    assert not unscored, f"categories returned None despite full inputs: {unscored}"

    total_effective = sum(c["effective_weight"] for c in result["categories"].values())
    assert total_effective == pytest.approx(100.0, abs=0.1)


def test_effective_weight_equals_raw_when_nothing_dropped(stock_df, bench_close):
    """With all 8 categories present, effective weight == configured weight."""
    indicators = compute_all(stock_df)
    built = build_composite_extras(
        stock_df, indicators, egx30_close=bench_close,
        risk_free_rate_pct=25.0, pe_ratio=12.0, dividend_yield=3.1,
        loss_making=False, index_membership="EGX30",
    )
    result = compute_composite(indicators, extras=built["extras"])
    for name, cat in result["categories"].items():
        assert cat["effective_weight"] == pytest.approx(cat["weight"], abs=0.01)


def test_builder_includes_every_scoring_input(stock_df, bench_close):
    """
    The exact keys the 8 scorers read. A missing key silently disables a
    category, which is the failure mode this whole module guards against.
    """
    indicators = compute_all(stock_df)
    extras = build_composite_extras(
        stock_df, indicators, egx30_close=bench_close,
        risk_free_rate_pct=25.0, pe_ratio=10.0, dividend_yield=3.1,
        loss_making=False, index_membership="EGX30",
    )["extras"]

    required = {
        "current_price", "divergences", "volume_price", "bb_squeeze",
        "obv_rising", "price_rising_20d", "golden_cross_active",
        "multi_timeframe", "trend_consistency", "current_drawdown_pct",
        "annualized_return_pct", "volatility_annualized_pct",
        "atr_pct_of_price", "history_days", "risk_free_rate_pct",
        "relative_strength", "pe_ratio", "dividend_yield", "loss_making",
        "liquidity",
    }
    assert required <= set(extras)

    # The three that the batch path used to omit entirely.
    for key in ("annualized_return_pct", "relative_strength", "multi_timeframe"):
        assert extras[key] is not None, f"{key} is None — its category will be dropped"


def test_drawdown_is_a_fraction(stock_df, bench_close):
    """score_quality multiplies by 100; a percent here would read as -9000%."""
    indicators = compute_all(stock_df)
    extras = build_composite_extras(stock_df, indicators, egx30_close=bench_close)["extras"]
    dd = extras["current_drawdown_pct"]
    if dd is not None:
        assert -1.0 <= dd <= 0.0


# ---------------------------------------------------------------------------
# Interval calibration — Weekly / Monthly must not be scored with daily maths
# ---------------------------------------------------------------------------

WEEKLY_CAGR = 0.15   # 15% a year, compounded — same over any window


def _weekly_compounding(years=5, cagr=WEEKLY_CAGR):
    """
    Weekly bars growing at a CONSTANT compound rate, so the correct
    annualized return is `cagr` regardless of which window is measured.
    That makes the assertion unambiguous: any other answer is a units bug.
    """
    n = int(52 * years)
    idx = pd.date_range("2021-01-07", periods=n, freq="W-THU")
    close = pd.Series(100 * (1 + cagr) ** (np.arange(n) / 52.0), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


def _weekly_doubling_over_5y():
    """Price doubles over exactly 5 years, sampled weekly (linear ramp)."""
    n = 260
    idx = pd.date_range("2021-01-07", periods=n, freq="W-THU")
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


def test_weekly_annualized_return_is_not_daily_maths():
    """
    Annualizing weekly bars with the 252-day constant treated 260 weeks as
    one year, so a five-year run reported a multi-year total as if it were
    one year's gain. Risk-Adjusted then compared that inflated figure to the
    25% T-bill and called a mediocre performer a cash-crusher.
    """
    df = _weekly_compounding()
    extras = build_composite_extras(df, compute_all(df), interval="Weekly")["extras"]
    ann = extras["annualized_return_pct"]

    assert ann is not None
    assert ann == pytest.approx(WEEKLY_CAGR * 100, abs=0.5), (
        f"expected ~{WEEKLY_CAGR*100:.0f}% CAGR, got {ann}%"
    )


def test_daily_and_weekly_agree_on_the_same_underlying_growth():
    """
    The same 15%-a-year stock must report ~15% whether you view it Daily or
    Weekly. Disagreement means one interval's annualization is miscalibrated.
    """
    n_days = 252 * 3
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    close = pd.Series(100 * (1 + WEEKLY_CAGR) ** (np.arange(n_days) / 252.0), index=idx)
    daily_df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n_days, 1e6)},
        index=idx,
    )
    weekly_df = _weekly_compounding(years=3)

    daily_ann = build_composite_extras(
        daily_df, compute_all(daily_df), interval="Daily"
    )["extras"]["annualized_return_pct"]
    weekly_ann = build_composite_extras(
        weekly_df, compute_all(weekly_df), interval="Weekly"
    )["extras"]["annualized_return_pct"]

    assert daily_ann == pytest.approx(weekly_ann, abs=1.0), (
        f"Daily says {daily_ann}%, Weekly says {weekly_ann}% for the same growth rate"
    )


def test_weekly_volatility_uses_52_not_252():
    """sqrt(252) on weekly returns overstates volatility by 2.2x."""
    df = _weekly_doubling_over_5y()
    per_bar = df["close"].pct_change().std()
    extras = build_composite_extras(df, compute_all(df), interval="Weekly")["extras"]

    expected = float(per_bar) * (52 ** 0.5) * 100.0
    assert extras["volatility_annualized_pct"] == pytest.approx(expected, rel=1e-6)


def test_weekly_drawdown_window_is_one_year_of_weeks():
    """tail(252) on weekly bars is a ~5-year drawdown labelled as 1-year."""
    n = 200
    idx = pd.date_range("2022-01-06", periods=n, freq="W-THU")
    # Peak 3 years back, then a slow recovery — only visible in a 5y window.
    close = pd.Series(
        np.concatenate([np.linspace(200, 100, 120), np.linspace(100, 150, n - 120)]),
        index=idx,
    )
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1e6)},
        index=idx,
    )
    extras = build_composite_extras(df, compute_all(df), interval="Weekly")["extras"]

    # Drawdown measured against the highest close in the LAST 52 weeks only.
    peak_1y = float(close.tail(52).max())
    expected = (float(close.iloc[-1]) - peak_1y) / peak_1y
    assert extras["current_drawdown_pct"] == pytest.approx(expected, rel=1e-9)


def test_multi_timeframe_never_compares_an_interval_with_itself():
    """
    Resampling weekly data to weekly is a no-op, so the alignment check was
    comparing a series against itself — a meaningless input feeding Quality.
    Weekly must compare against Monthly; Monthly has no higher frame.
    """
    df = _weekly_doubling_over_5y()

    weekly = build_composite_extras(df, compute_all(df), interval="Weekly")["extras"]
    mtf = weekly["multi_timeframe"]
    if mtf is not None:
        # Must be a genuinely coarser series, not the input echoed back.
        assert mtf.get("weekly_trend") is not None

    monthly = build_composite_extras(df, compute_all(df), interval="Monthly")["extras"]
    assert monthly["multi_timeframe"] is None, (
        "Monthly has no higher timeframe — comparing it with itself is meaningless"
    )


def test_history_gate_means_the_same_span_on_every_interval():
    """
    history_days gates score_risk_adjusted at 120. Passing raw bar counts let
    120 weekly bars (2.3 years) and 120 daily bars (6 months) both pass.
    """
    df = _weekly_doubling_over_5y()   # 260 weekly bars = 5 years
    extras = build_composite_extras(df, compute_all(df), interval="Weekly")["extras"]

    # 5 years expressed in trading-day equivalents.
    assert extras["history_days"] == pytest.approx(5 * 252, rel=0.05)


def test_short_weekly_history_is_gated_out():
    """20 weekly bars is under 6 months — Risk-Adjusted must refuse to score."""
    n = 20
    idx = pd.date_range("2025-01-02", periods=n, freq="W-THU")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1e6)},
        index=idx,
    )
    extras = build_composite_extras(df, compute_all(df), interval="Weekly")["extras"]
    score, _ = score_risk_adjusted(
        extras["annualized_return_pct"], 25.0,
        extras["volatility_annualized_pct"], extras["atr_pct_of_price"],
        extras["history_days"],
    )
    assert score is None


# ---------------------------------------------------------------------------
# Key levels — breakouts, level age, and honest strength
# ---------------------------------------------------------------------------

def test_clear_air_above_when_price_is_above_every_resistance():
    """
    On a breakout the fallback "nearest resistance" is a level the stock
    cleared long ago — for one real EGX name it sat 27% BELOW the price.
    Reporting that as resistance buries the headline, so the flag lets the
    UI lead with "no resistance overhead" instead.
    """
    sr = {
        "supports": [{"price": 80.0, "strength": 2}],
        "resistances": [{"price": 93.0, "strength": 3}],
    }
    levels = compute_key_levels(126.60, sr)

    assert levels["clear_air_above"] is True
    # The fallback level is still reported, but flagged as broken by its sign.
    assert levels["nearest_resistance"]["distance_pct"] < 0


def test_not_clear_air_when_resistance_sits_above():
    sr = {
        "supports": [{"price": 120.0, "strength": 2}],
        "resistances": [{"price": 145.0, "strength": 2}],
    }
    levels = compute_key_levels(140.0, sr)

    assert levels["clear_air_above"] is False
    assert levels["nearest_resistance"]["distance_pct"] > 0


def test_clear_air_below_when_price_is_under_every_support():
    sr = {"supports": [{"price": 100.0, "strength": 3}], "resistances": []}
    levels = compute_key_levels(80.0, sr)
    assert levels["clear_air_below"] is True


def test_levels_report_how_stale_they_are():
    """A level from last week and one from last spring must be distinguishable."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    low = pd.Series(np.full(n, 100.0), index=idx)
    high = pd.Series(np.full(n, 101.0), index=idx)
    # Only bar 150 dips to 90 — 49 bars before the end.
    low.iloc[150] = 90.0

    sr = {"supports": [{"price": 90.0, "strength": 1}], "resistances": []}
    levels = compute_key_levels(100.5, sr, high=high, low=low)

    assert levels["nearest_support"]["bars_ago"] == n - 1 - 150


def test_bars_ago_is_none_without_ohlcv():
    """Callers that don't pass high/low still work; age is simply unknown."""
    sr = {"supports": [{"price": 90.0, "strength": 1}], "resistances": []}
    levels = compute_key_levels(100.0, sr)
    assert levels["nearest_support"]["bars_ago"] is None


def test_single_pivot_reports_strength_one():
    """
    Strength 1 means one bounce, not a tested floor. The UI relies on this to
    say "touched once — weak level" rather than "Tested 1x", which read like
    a credential for what is barely evidence.
    """
    sr = {"supports": [{"price": 90.0, "strength": 1}], "resistances": []}
    assert compute_key_levels(100.0, sr)["nearest_support"]["strength"] == 1


# ---------------------------------------------------------------------------
# Index membership — feeds the index-aware liquidity floors
# ---------------------------------------------------------------------------

def test_index_membership_reads_the_static_file():
    assert get_index_membership("COMI") == "EGX30"
    assert get_index_membership("comi") == "EGX30"


def test_index_membership_does_no_network(monkeypatch):
    """
    This lookup runs once per symbol on the dashboard batch path. Routing it
    through tickers._load_tickers() would put every card behind a 10 s
    TradingView POST on a cold container — invisible in dev, lethal on Vercel.
    """
    import urllib.request

    import app.core.index_membership as im

    def _boom(*args, **kwargs):
        raise AssertionError("index membership lookup attempted a network call")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(im, "_MEMBERSHIP", None)  # force a real (re)load

    assert im.get_index_membership("COMI") == "EGX30"


def test_unknown_symbol_is_none_not_a_guessed_tier():
    """
    None means "unknown", and liquidity_score applies its EGX100 default —
    today's behaviour for every symbol. Guessing EGX30 here would silence
    genuine thin-volume warnings on unrecognised names.
    """
    assert get_index_membership("ZZZZ_NOT_A_SYMBOL") is None
    assert get_index_membership("") is None


def test_liquidity_uses_the_stocks_own_index_floors():
    """
    The bug this fixes: portfolio_analysis passed index_membership=None, so
    every holding was measured against EGX100 floors. A 40k-shares/day stock
    is normal for NILEX and thin for EGX30 — one blanket floor cannot say both.
    """
    vol = pd.Series([40_000.0] * 30)

    assert liquidity_score(vol, index_membership="EGX30")["thin"] is True
    assert liquidity_score(vol, index_membership="NILEX")["thin"] is False


def test_dead_sessions_beat_the_average():
    """
    MEGM has been frozen at 12.54 with zero volume since January 2022, yet one
    old block trade left it averaging ~99k shares/day — comfortably "low", not
    thin. An average hides zeros; a stock you cannot trade must say so.
    """
    vol = pd.Series([0.0] * 19 + [1_981_600.0])
    result = liquidity_score(vol, index_membership="EGX100", lookback=20)

    assert result["thin"] is True, f"suspended stock read as {result}"
    assert result["dead_sessions"] == 19


def test_normal_trading_reports_no_dead_sessions():
    vol = pd.Series([2_000_000.0] * 30)
    assert liquidity_score(vol, index_membership="EGX30")["dead_sessions"] == 0


def test_frozen_price_is_not_called_a_downtrend():
    """
    trend_consistency counted bars where close > SMA20. On a frozen price they
    are EQUAL, so it scored 0.0 — which score_quality reads as "price below
    SMA20 on 100% of last 20 days, steady downtrend" and penalises. A price
    that never moved is not falling.
    """
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.full(n, 12.54), index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": np.zeros(n)}, index=idx)

    extras = build_composite_extras(df, compute_all(df), interval="Daily")["extras"]
    assert extras["trend_consistency"] is None

    _, reasons = score_quality(None, extras["trend_consistency"], None)
    assert not any("downtrend" in r.lower() for r in reasons)


def test_liquidity_reports_which_floor_it_used():
    """The warning text says "thin for an EGX30 name" — it needs the tier."""
    vol = pd.Series([40_000.0] * 30)
    assert liquidity_score(vol, index_membership="NILEX")["index_membership"] == "NILEX"
    # Unknown tier falls back to EGX100 and says so, rather than claiming None.
    assert liquidity_score(vol, index_membership=None)["index_membership"] == "EGX100"


# ---------------------------------------------------------------------------
# Liquidity as a scoring input — penalty-only
# ---------------------------------------------------------------------------

def _volume_inputs():
    """A neutral-ish set of the other score_volume inputs."""
    return dict(obv_rising=True, price_rising_20d=True, mfi_val=50.0,
                volume_price={"classification": "normal", "volume_ratio": 1.0,
                              "price_change_pct": 0.1})


def test_liquidity_band_is_penalty_only():
    """
    ~95% of EGX names are normally liquid. If normal liquidity scored points,
    every one of them would drift for no information — and the liquidity
    reason would cancel against the directional bands beside it.
    """
    v = _volume_inputs()
    baseline, _ = score_volume(v["obv_rising"], v["price_rising_20d"],
                               v["mfi_val"], v["volume_price"])
    normal = {"avg_volume": 2_000_000, "classification": "normal",
              "thin": False, "index_membership": "EGX30"}
    low = {"avg_volume": 60_000, "classification": "low",
           "thin": False, "index_membership": "EGX100"}
    thin = {"avg_volume": 8_000, "classification": "thin",
            "thin": True, "index_membership": "EGX30"}

    normal_score, _ = score_volume(v["obv_rising"], v["price_rising_20d"],
                                   v["mfi_val"], v["volume_price"], liquidity=normal)
    low_score, low_reasons = score_volume(v["obv_rising"], v["price_rising_20d"],
                                          v["mfi_val"], v["volume_price"], liquidity=low)
    thin_score, thin_reasons = score_volume(v["obv_rising"], v["price_rising_20d"],
                                            v["mfi_val"], v["volume_price"], liquidity=thin)

    assert normal_score == baseline, "normal liquidity must not move the score"
    assert low_score == baseline, "the 'low' tier must not move the score"
    assert thin_score < baseline, "thin liquidity must penalise"

    # ...but 'low' still explains itself, and 'thin' names its index tier.
    assert any("Modest liquidity" in r for r in low_reasons)
    assert any("EGX30" in r for r in thin_reasons)


def test_liquidity_cannot_carry_the_volume_category():
    """
    With no OBV/MFI/volume-price data the category must stay unscored. If
    liquidity alone could carry it, a data-less stock would score 38 on one
    reason and the category would claim its full weight.
    """
    thin = {"avg_volume": 8_000, "classification": "thin",
            "thin": True, "index_membership": "EGX30"}
    score, _ = score_volume(None, None, None, None, liquidity=thin)
    assert score is None


def test_liquidity_is_per_day_on_every_interval():
    """
    The floors are shares/DAY. Without normalising, a weekly bar's volume (a
    week's worth) would make the same stock look ~5x more liquid on the Weekly
    view than on its own Daily view.

    Uses a business-day frame, not the calendar-day module fixture: the
    252/52 conversion encodes ~5 TRADING days per week, which is what real
    EGX data has.
    """
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100.0, 130.0, n), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(n, 200_000.0)},
        index=idx,
    )

    daily = build_composite_extras(df, compute_all(df),
                                   interval="Daily")["liquidity"]

    weekly_df = df.resample("W-THU").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()
    weekly = build_composite_extras(weekly_df, compute_all(weekly_df),
                                    interval="Weekly")["liquidity"]

    assert daily["avg_volume"] is not None and weekly["avg_volume"] is not None
    ratio = weekly["avg_volume"] / daily["avg_volume"]
    assert 0.9 < ratio < 1.1, (
        f"weekly reports {ratio:.2f}x the daily shares/day — not normalised"
    )


# ---------------------------------------------------------------------------
# Fundamentals bands — P/E recentred on the EGX median, DY non-monotonic
# ---------------------------------------------------------------------------

def test_pe_band_is_centred_on_the_egx_median():
    """
    The old band gave +8 to anything under P/E 20 — which is most of a market
    whose median is 12.4. Simply HAVING P/E data was worth points, and only
    ~22% of EGX stocks have it, so the app ranked covered stocks above
    uncovered ones for a reason unrelated to their merit.
    """
    score, _ = score_quality(None, None, None, pe_ratio=12.4)
    assert 45 <= score <= 55, f"median EGX P/E should be ~neutral, got {score}"


def test_implausibly_low_pe_is_not_a_bargain():
    """
    MEGM trades at P/E 0.7. Under the old band that was "+15 very cheap on
    earnings" — the app calling a distressed situation a quality stock.
    """
    distressed, reasons = score_quality(None, None, None, pe_ratio=0.7)
    genuinely_cheap, _ = score_quality(None, None, None, pe_ratio=6.0)
    assert distressed < genuinely_cheap
    assert any("recurring" in r for r in reasons)


def test_expensive_pe_is_penalised():
    cheap, _ = score_quality(None, None, None, pe_ratio=6.0)
    rich, _ = score_quality(None, None, None, pe_ratio=45.0)
    assert rich < cheap


def test_loss_making_comes_from_eps_not_negative_pe():
    """
    The fundamentals feed reports a NULL P/E for loss-makers, never a negative
    one, so the old `pe_ratio < 0` test could never fire. EPS carries it now.
    """
    # A flat technical baseline so both branches score the category — with
    # every other input None the category is legitimately unscoreable.
    healthy, _ = score_quality(None, None, -0.05)
    losing, reasons = score_quality(None, None, -0.05, loss_making=True)
    assert losing < healthy
    assert any("loss-making" in r.lower() for r in reasons)


def test_quality_dividend_band_is_not_monotonic():
    """
    A 42% yield is a special dividend or a collapsed share price, not income
    quality. More yield must NOT mean more points all the way up.
    """
    scores = {dy: score_quality(None, None, None, dividend_yield=dy)[0]
              for dy in (3.0, 6.0, 12.0, 25.0, 45.0)}
    assert scores[6.0] > scores[3.0]
    assert scores[45.0] < scores[6.0], f"extreme yield scored high: {scores}"
    assert scores[25.0] < scores[12.0]


def test_extreme_dividend_yield_is_not_read_as_quality():
    """The live MEGM case: DY 42.55% on a P/E of 0.7."""
    score, reasons = score_quality(None, None, None,
                                   pe_ratio=0.7, dividend_yield=42.55)
    assert score <= 60, f"MEGM scored {score} on quality"
    text = " ".join(reasons).lower()
    assert "special" in text or "collapsed" in text


def test_zero_dividend_is_not_penalised():
    """
    Under the old EGX feed a printed "0" was a no-data sentinel. From the new
    source 0.0 is real: the company pays nothing, which is normal for a growth
    company and not a quality defect. Only None means unknown.
    """
    pays_nothing, pn_reasons = score_quality(None, None, -0.05, dividend_yield=0.0)
    unknown, _ = score_quality(None, None, -0.05, dividend_yield=None)
    assert pays_nothing == unknown
    assert not any("dividend" in r.lower() for r in pn_reasons)


def test_median_dividend_payer_gets_a_modest_bonus():
    """Framing check: a 3% yield is worth something, but it is not 'income'."""
    score, reasons = score_quality(None, None, None, dividend_yield=3.12)
    assert score > 50
    assert any("median" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# 52-week positioning
# ---------------------------------------------------------------------------

def test_drawdown_peak_uses_intraday_highs():
    """
    score_quality used to say "near recent peak" off max(close) while
    StatsPanel rendered "52W High" off max(high) — two numbers for one fact on
    the same screen.
    """
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(np.full(n, 100.0), index=idx)
    high = close.copy()
    low = close.copy()
    high.iloc[100] = 130.0          # a spike no close ever reached
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": close, "volume": np.full(n, 200_000.0)},
                      index=idx)

    built = build_composite_extras(df, compute_all(df), interval="Daily")
    assert built["high_52w"] == pytest.approx(130.0)
    # 100 vs a 130 peak = -23%, not the 0% a close-only peak would report.
    assert built["extras"]["current_drawdown_pct"] == pytest.approx(-30.0 / 130.0, abs=1e-6)


def test_drawdown_reasons_say_52_week_high():
    _, reasons = score_quality(None, None, -0.35)
    assert any("52-week high" in r for r in reasons)
    _, near = score_quality(None, None, -0.01)
    assert any("52-week high" in r for r in near)


# ---------------------------------------------------------------------------
# Fundamentals feed — the never-wipe guarantee
# ---------------------------------------------------------------------------

class _FakeDB:
    """Records every statement so a test can assert what was (not) written."""

    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), params))
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def commit(self):
        pass

    def upserts_into(self, table):
        return [s for s, _ in self.statements if f"INTO {table}" in s]

    def setting(self, key):
        for sql, params in self.statements:
            if "INTO settings" in sql and params and params[0] == key:
                return params[1]
        return None


def test_fundamentals_refresh_never_wipes_on_failure(monkeypatch):
    """
    The module's headline guarantee, previously untested: a failed refresh must
    leave last-known-good rows alone. The read path serving stale P/E is fine;
    serving "no data" because the feed blinked is not.
    """
    import app.core.pe_fetch as pf

    def _boom():
        raise RuntimeError("scanner unreachable")

    monkeypatch.setattr(pf, "fetch_fundamentals_rows", _boom)

    db = _FakeDB()
    result = pf.refresh_pe_data(db)

    assert result["success"] is False
    assert not any("DELETE" in s or "TRUNCATE" in s for s, _ in db.statements)
    assert not db.upserts_into("pe_data")
    assert db.setting("pe_last_attempt_status").startswith("error")
    # A failed attempt must NOT advance the success timestamp, or the
    # freshness banner would go quiet while the data rotted.
    assert db.setting("pe_last_successful_fetch") is None


def test_partial_feed_response_is_rejected_without_writing():
    """
    A truncated response that refreshes 5 symbols and silently leaves 288 stale
    is worse than "everything is stale" — nothing on screen distinguishes the
    fresh rows from the rotten ones.
    """
    import app.core.pe_fetch as pf

    rows = [{"symbol": f"SYM{i}", "company_name": "x", "pe_ratio": 10.0,
             "dividend_yield": 3.0, "loss_making": False} for i in range(5)]

    db = _FakeDB()
    result = pf.refresh_pe_data(db, rows=rows)

    assert result["success"] is False
    assert not db.upserts_into("pe_data")
    assert "only 5 rows" in db.setting("pe_last_attempt_status")


def test_rows_with_no_fundamentals_are_not_stored():
    """
    An all-null row makes get_pe_for_symbol return a truthy dict of Nones,
    which the response body ships to the frontend as an empty P/E card.
    """
    import app.core.pe_fetch as pf

    rows = [{"symbol": f"SYM{i}", "company_name": "x", "pe_ratio": None,
             "dividend_yield": None, "loss_making": None} for i in range(150)]
    rows[0].update(pe_ratio=12.0)

    db = _FakeDB()
    result = pf.refresh_pe_data(db, rows=rows)

    assert result["success"] is True
    assert result["count"] == 1
    assert result["skipped_empty"] == 149


# ---------------------------------------------------------------------------
# Fundamentals history — the point-in-time record
# ---------------------------------------------------------------------------

def _fundamental(symbol="COMI", **kw):
    row = {"symbol": symbol, "company_name": "x", "pe_ratio": 10.0,
           "dividend_yield": 3.0, "loss_making": False, "eps_ttm": 20.0,
           "dps_annual": 6.0, "book_value_per_share": 70.0, "close": 200.0}
    row.update(kw)
    return row


def test_history_logs_fundamentals_not_price_ratios():
    """
    P/E, P/B and dividend yield all divide by PRICE, so they move every day.
    Logging them would be ~99% price noise. EPS, DPS and book value move
    quarterly, and any ratio is reconstructable from them plus a historical
    close — verified against the live feed, where close/eps reproduces the
    reported P/E exactly.
    """
    import app.core.pe_fetch as pf

    cols = pf.TV_COLUMNS
    assert "earnings_per_share_diluted_ttm" in cols
    assert "book_value_per_share_fq" in cols
    assert set(pf._HISTORY_FIELDS) == {
        "eps_ttm", "dps_annual", "book_value_per_share", "loss_making"
    }
    # The ratios must NOT drive a history append.
    assert "pe_ratio" not in pf._HISTORY_FIELDS
    assert "dividend_yield" not in pf._HISTORY_FIELDS


def test_unchanged_fundamentals_do_not_append():
    """
    Appending nightly regardless would grow ~107k rows a year to record
    quarterly events. Only a real change is an event.
    """
    import app.core.pe_fetch as pf

    prev = {"eps_ttm": 20.0, "dps_annual": 6.0,
            "book_value_per_share": 70.0, "loss_making": False}
    # Price moved a long way; no fundamental did.
    assert pf._changed(prev, _fundamental(close=999.0, pe_ratio=50.0)) is False


def test_changed_earnings_append():
    import app.core.pe_fetch as pf

    prev = {"eps_ttm": 20.0, "dps_annual": 6.0,
            "book_value_per_share": 70.0, "loss_making": False}
    assert pf._changed(prev, _fundamental(eps_ttm=23.5)) is True
    assert pf._changed(prev, _fundamental(dps_annual=7.0)) is True
    assert pf._changed(prev, _fundamental(loss_making=True)) is True
    # First-ever observation always logs.
    assert pf._changed(None, _fundamental()) is True


def test_gaining_or_losing_a_value_is_a_change():
    """None -> a number is a real event, and so is the reverse."""
    import app.core.pe_fetch as pf

    prev = {"eps_ttm": None, "dps_annual": 6.0,
            "book_value_per_share": 70.0, "loss_making": False}
    assert pf._changed(prev, _fundamental(eps_ttm=20.0)) is True
    assert pf._changed({**prev, "eps_ttm": 20.0}, _fundamental(eps_ttm=None)) is True


def test_float_jitter_does_not_append():
    """
    The feed re-derives these each night and the last decimal wobbles. Logging
    that would defeat the append-on-change design.
    """
    import app.core.pe_fetch as pf

    prev = {"eps_ttm": 20.0, "dps_annual": 6.0,
            "book_value_per_share": 70.0, "loss_making": False}
    assert pf._changed(prev, _fundamental(eps_ttm=20.0 + 1e-9)) is False


def test_point_in_time_reader_asks_for_the_right_row():
    """
    The table is useless without this: reading the LATEST row (which is what the
    change-detection query does) would hand a backtest today's earnings for a
    2024 date. The read must be bounded by observed_at <= as_of and take the
    most recent one at or before it.
    """
    import app.core.pe_fetch as pf

    class _RecordingDB(_FakeDB):
        def fetchone(self):
            return ("2024-03-01", 20.0, 6.0, 70.0, False)

    db = _RecordingDB()
    got = pf.get_fundamentals_at(db, "comi", "2024-06-30")

    sql, params = db.statements[-1]
    assert "observed_at <= %s" in sql, "reader is not bounded by the as-of date"
    assert "ORDER BY observed_at DESC" in sql and "LIMIT 1" in sql
    assert params == ("COMI", "2024-06-30"), "symbol must be upper-cased"
    assert got["eps_ttm"] == 20.0 and got["observed_at"] == "2024-03-01"
    # Ratios are deliberately absent — they must be derived from the close on
    # the evaluated date, not read back.
    assert "pe_ratio" not in got and "dividend_yield" not in got


def test_point_in_time_reader_returns_none_before_first_observation():
    import app.core.pe_fetch as pf

    class _EmptyDB(_FakeDB):
        def fetchone(self):
            return None

    assert pf.get_fundamentals_at(_EmptyDB(), "COMI", "2019-01-01") is None


def test_history_append_failure_does_not_fail_the_refresh():
    """
    The current-value feed is what the app serves; history is for later
    analysis. A broken history table must not take the read path down with it.
    """
    import app.core.pe_fetch as pf

    class _HistoryBrokenDB(_FakeDB):
        def execute(self, sql, params=None):
            if "fundamentals_history" in sql:
                raise RuntimeError("relation does not exist")
            return super().execute(sql, params)

    rows = [_fundamental(symbol=f"SYM{i}") for i in range(150)]
    db = _HistoryBrokenDB()
    result = pf.refresh_pe_data(db, rows=rows)

    assert result["success"] is True
    assert result["count"] == 150
    assert result["history_rows_appended"] == 0


def test_absurd_pe_is_dropped_at_ingest():
    """
    The live EGX maximum is ~2756. The reason string renders the raw number,
    so "P/E 2756.0" would read as a broken app rather than a real valuation.
    """
    import app.core.pe_fetch as pf

    assert pf._clean_float(2756.2, maximum=pf.PE_SANITY_MAX) is None
    assert pf._clean_float(12.4, maximum=pf.PE_SANITY_MAX) == 12.4
    # 0.0 survives for dividend yield — it means "pays nothing", not "no data".
    assert pf._clean_float(0.0, maximum=pf.DY_SANITY_MAX) == 0.0


def test_analysis_router_does_not_redefine_bars_per_year():
    """
    The 52-week window is defined once, in extras_builder.BARS_PER_YEAR. A
    second copy in the router is how the score's drawdown and the displayed
    52W High drifted apart.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "app" / "routers" / "analysis.py").read_text(encoding="utf-8")
    assert '"Weekly": 52' not in src, (
        "analysis.py redefines the bars-per-year table — import "
        "extras_builder.BARS_PER_YEAR instead"
    )


# ---------------------------------------------------------------------------
# Market condition reading — the only forecast-shaped surface
# ---------------------------------------------------------------------------

def test_regime_bands_match_the_measured_terciles():
    """
    Cutoffs come from 221 readings, 2007-2026. If someone retunes them by feel
    the historical numbers shown beside them stop being true of the band.
    """
    from app.core.regime import REGIME_MIXED_MAX, REGIME_WEAK_MAX, classify_regime

    assert classify_regime(40.0, 40)["band"] == "weak"
    assert classify_regime(REGIME_WEAK_MAX, 40)["band"] == "mixed"
    assert classify_regime(48.0, 40)["band"] == "mixed"
    assert classify_regime(REGIME_MIXED_MAX, 40)["band"] == "broad"
    assert classify_regime(58.0, 40)["band"] == "broad"


def test_regime_refuses_to_classify_thin_coverage():
    """
    The batch scorer returns partial results on deadline. Averaging six stocks
    into a confident market reading is worse than saying nothing.
    """
    from app.core.regime import MIN_SYMBOLS_FOR_REGIME, classify_regime

    thin = classify_regime(52.0, MIN_SYMBOLS_FOR_REGIME - 1)
    assert thin["band"] is None
    assert "at least" in thin["summary"]

    ok = classify_regime(52.0, MIN_SYMBOLS_FOR_REGIME)
    assert ok["band"] == "broad"


def test_regime_carries_its_own_evidence():
    """
    Every claim the UI makes must come from here, so the measurement and the
    wording can't drift apart. The horizon in particular is load-bearing: 21
    and 126 days were both tested and neither was significant.
    """
    r = classify_regime_or_skip()
    for key in ("hist_median_3m_pct", "hist_positive_rate", "observations",
                "association_rho", "association_n", "horizon_days"):
        assert key in r, f"regime reading is missing {key}"
    assert r["horizon_days"] == 63


def classify_regime_or_skip():
    from app.core.regime import classify_regime
    return classify_regime(52.0, 40)


def test_weak_band_is_not_advertised_as_negative():
    """
    The EGX rose a lot in EGP terms over the window, so the weak band means
    "flat", not "falling". Claiming a crash is as wrong as claiming a rally.
    """
    from app.core.regime import classify_regime

    weak = classify_regime(40.0, 40)
    assert weak["hist_median_3m_pct"] > -1.0
    assert "coin flip" in weak["summary"]


def test_regime_reader_and_batch_writer_share_one_cache_key():
    """
    The dashboard batch WRITES per-symbol score cache entries; the market
    reading READS them. If the two spelled the key differently the reading
    would find zero hits forever and permanently report "not enough data" —
    a silent failure, with the data sitting right there in the cache.
    """
    import app.core.cache as cache
    from app.routers.analysis import (
        composite_cache_key, read_cached_scores, scoring_cache_context,
    )

    _, _, _, _, tags = scoring_cache_context()
    key = composite_cache_key("comi", "Daily", tags)
    cache.set(key, {"score": 61.0, "signal": "Strong"})
    try:
        got = read_cached_scores(["COMI"], "Daily")
        assert "COMI" in got, "reader missed a score the writer's key just stored"
        assert got["COMI"]["score"] == 61.0
    finally:
        cache._store.pop(key, None)


def test_regime_reader_ignores_error_entries():
    """A cached failure must not be averaged in as if it were a score."""
    import app.core.cache as cache
    from app.routers.analysis import (
        composite_cache_key, read_cached_scores, scoring_cache_context,
    )

    _, _, _, _, tags = scoring_cache_context()
    key = composite_cache_key("ZZZZ", "Daily", tags)
    cache.set(key, {"error": "no data"})
    try:
        assert read_cached_scores(["ZZZZ"], "Daily") == {}
    finally:
        cache._store.pop(key, None)
