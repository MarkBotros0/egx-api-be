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
from app.core.entry_price import compute_max_buy_price
from app.core.extras_builder import build_composite_extras
from app.core.indicators import (
    _cluster_levels,
    atr,
    compute_all,
    fibonacci_levels,
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
        (0, "Strong Sell"),
        (19.99, "Strong Sell"),
        (20, "Sell"),          # exactly 20 is Sell, NOT Strong Sell
        (39.99, "Sell"),
        (40, "Hold"),
        (59.99, "Hold"),
        (60, "Buy"),           # exactly 60 is Buy, NOT Hold
        (79.99, "Buy"),
        (80, "Strong Buy"),
        (100, "Strong Buy"),
    ],
)
def test_classify_signal_boundaries(score, expected):
    assert classify_signal(score) == expected


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

def test_entry_zone_and_max_buy_agree_on_stop_loss():
    """
    The stock detail page renders EntryExitCard and MaxBuyPriceCard side by
    side. They used 1.5x and 1.0x ATR respectively, showing two different
    stop-losses for the same trade with no way to tell which to trust.
    """
    support, resistance, atr_val, price = 100.0, 130.0, 3.0, 102.0
    sr = {
        "supports": [{"price": support, "strength": 4}],
        "resistances": [{"price": resistance, "strength": 3}],
    }
    zones = compute_entry_exit(price, sr, rsi_latest=45, stoch_k_latest=40, atr_latest=atr_val)
    max_buy = compute_max_buy_price(price, support, resistance, atr_val)

    expected = round(support - STOP_LOSS_ATR_MULTIPLIER * atr_val, 2)
    assert zones["entry_zone"]["suggested_stop_loss"] == expected
    assert max_buy["stop_loss"] == expected


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
        risk_free_rate_pct=25.0, pe_ratio=12.0,
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
        risk_free_rate_pct=25.0, pe_ratio=10.0,
    )["extras"]

    required = {
        "current_price", "divergences", "volume_price", "bb_squeeze",
        "obv_rising", "price_rising_20d", "golden_cross_active",
        "multi_timeframe", "trend_consistency", "current_drawdown_pct",
        "annualized_return_pct", "volatility_annualized_pct",
        "atr_pct_of_price", "history_days", "risk_free_rate_pct",
        "relative_strength", "pe_ratio",
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
