"""
Composite Score Engine
======================

Combines the individual technical indicators into a single 0-100 score that
answers: "Should I buy, hold, or sell this stock right now?"

The composite is a weighted average of 8 category sub-scores, each 0-100.
Defaults below are the "Beginner Safe" preset — tilted toward stability, market
leadership, and beating the Egyptian T-bill rate:

  1. Trend             (default 18%) — SMA crossovers, ADX strength, DI+/DI-
  2. Momentum          (default 15%) — RSI, MACD, Stochastic
  3. Volume            (default 12%) — OBV trend, MFI, volume-price confirmation
  4. Volatility        (default 10%) — Bollinger Band position, Bollinger squeeze
  5. Divergence        (default  8%) — RSI divergence, MACD divergence
  6. Quality           (default 12%) — trend consistency, multi-timeframe alignment
  7. Risk-Adjusted     (default 13%) — annualized return vs T-bill, ATR stop context
  8. Relative Strength (default 12%) — alpha vs EGX30 (30-day window)

After the weighted sum, an optional MACRO MODULATION is applied: when the EGX30
itself is in a bearish regime, bullish-leaning scores are dampened and bearish
ones are reinforced (and vice versa). This prevents confidently-buying into a
falling market. The modulation delta is returned as `macro_adjustment`.

Signal thresholds:
   0-20  Strong Sell
  20-40  Sell
  40-60  Hold (neutral)
  60-80  Buy
  80-100 Strong Buy

This module is pure — no DB access, no I/O. It consumes indicator values that
have already been computed (typically by `_indicators.compute_all`) and returns
a structured result. `get_weights_from_db` is the one exception: a thin helper
that reads weight rows from the settings table.

Important: the composite score is educational only. It mathematically combines
multiple signals; it does NOT predict the future. Always consider broader
context (fundamentals, news, macro conditions) before trading decisions.
"""

from __future__ import annotations

import math
from typing import Optional

from app.core.constants import (
    DIVIDEND_DRAG_PP_PER_YEAR,
    DEFAULT_RISK_FREE_RATE_PCT,
    SCORE_BUY_MAX,
    SCORE_HOLD_MAX,
    SCORE_SELL_MAX,
    SCORE_STRONG_SELL_MAX,
)


# "Beginner Safe" default — tilts toward stable, leading, cash-beating stocks.
DEFAULT_WEIGHTS = {
    "trend": 18,
    "momentum": 15,
    "volume": 12,
    "volatility": 10,
    "divergence": 8,
    "quality": 12,
    "risk_adjusted": 13,
    "relative_strength": 12,
}

CATEGORY_ORDER = [
    "trend", "momentum", "volume", "volatility", "divergence",
    "quality", "risk_adjusted", "relative_strength",
]

PRESETS = {
    # Beginner Safe == DEFAULT_WEIGHTS; kept here as an explicit preset too.
    "beginner_safe":    {"trend": 18, "momentum": 15, "volume": 12, "volatility": 10,
                         "divergence": 8, "quality": 12, "risk_adjusted": 13,
                         "relative_strength": 12},
    "balanced":         {"trend": 14, "momentum": 13, "volume": 12, "volatility": 12,
                         "divergence": 12, "quality": 12, "risk_adjusted": 12,
                         "relative_strength": 13},
    "trend_follower":   {"trend": 30, "momentum": 15, "volume": 10, "volatility": 8,
                         "divergence": 2, "quality": 15, "risk_adjusted": 5,
                         "relative_strength": 15},
    "reversal_hunter":  {"trend": 10, "momentum": 20, "volume": 15, "volatility": 15,
                         "divergence": 25, "quality": 5, "risk_adjusted": 5,
                         "relative_strength": 5},
    # New preset: for cash-equivalent-conscious investors who care most about
    # beating the 25% T-bill and preserving capital.
    "income_defensive": {"trend": 15, "momentum": 8, "volume": 10, "volatility": 15,
                         "divergence": 2, "quality": 20, "risk_adjusted": 25,
                         "relative_strength": 5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_valid(series_list):
    """Return the last non-NaN value of a list-like, or None if all NaN/empty."""
    if series_list is None:
        return None
    for v in reversed(series_list):
        if v is None:
            continue
        try:
            if isinstance(v, float) and math.isnan(v):
                continue
        except Exception:
            pass
        return v
    return None


def _prev_valid(series_list):
    """Return the second-to-last non-NaN value."""
    if series_list is None:
        return None
    found = 0
    for v in reversed(series_list):
        if v is None:
            continue
        try:
            if isinstance(v, float) and math.isnan(v):
                continue
        except Exception:
            pass
        found += 1
        if found == 2:
            return v
    return None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def normalize_weights(weights: dict) -> dict:
    """
    Return a copy of `weights` scaled so the values sum to 100.
    Non-positive values are clamped to 0. If all are zero, returns DEFAULT_WEIGHTS.
    """
    cleaned = {k: max(0.0, float(weights.get(k, 0))) for k in CATEGORY_ORDER}
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: round(cleaned[k] / total * 100, 2) for k in CATEGORY_ORDER}


def classify_signal(score: float) -> str:
    """
    Map a 0-100 composite score to a description of the stock's CONDITION.

    These used to read "Strong Buy" / "Buy" / "Hold" / "Sell" / "Strong Sell".
    A walk-forward backtest over 2007-2026 (scripts/backtest.py) showed the
    score has no ability to rank one stock above another: the cross-sectional
    information coefficient is ~0 and slightly negative, nine of ten score
    deciles have a median 21-day forward return of exactly 0.00%, and among
    liquid names the "Sell"-labelled stocks slightly OUTPERFORMED the "Buy"
    ones. The instruction the old labels gave was not supported by evidence,
    and on the sell side it pointed the wrong way.

    The bands and the score are unchanged — only the claim is. The score is a
    correct summary of a stock's present technical condition, so it now says
    what it measures instead of what to do.
    """
    if score >= SCORE_BUY_MAX:
        return "Very Strong"
    if score >= SCORE_HOLD_MAX:
        return "Strong"
    if score >= SCORE_SELL_MAX:
        return "Neutral"
    if score >= SCORE_STRONG_SELL_MAX:
        return "Weak"
    return "Very Weak"


# ---------------------------------------------------------------------------
# Category scorers
# ---------------------------------------------------------------------------
#
# Each scorer returns (score | None, reasons_list). A None score means the
# category can't be evaluated with the given inputs (e.g., <200 bars so no
# SMA200), and compute_composite will renormalize weights across the
# available categories.
# ---------------------------------------------------------------------------

def score_trend(current_price, sma_20, sma_50, sma_200,
                adx_val, plus_di_val, minus_di_val,
                golden_cross_active: bool = False) -> tuple:
    """Score the trend category (0-100)."""
    # Need at least one moving average to evaluate trend
    if current_price is None or (sma_20 is None and sma_50 is None and sma_200 is None):
        return None, []

    score = 50.0
    reasons = []

    if sma_20 is not None:
        if current_price > sma_20:
            score += 8
            reasons.append(f"Price above SMA20 ({sma_20:.2f})")
        else:
            score -= 8
            reasons.append(f"Price below SMA20 ({sma_20:.2f})")

    if sma_50 is not None:
        if current_price > sma_50:
            score += 10
            reasons.append(f"Price above SMA50 ({sma_50:.2f})")
        else:
            score -= 10
            reasons.append(f"Price below SMA50 ({sma_50:.2f})")

    if sma_200 is not None:
        if current_price > sma_200:
            score += 10
            reasons.append(f"Price above SMA200 ({sma_200:.2f}) — long-term uptrend")
        else:
            score -= 10
            reasons.append(f"Price below SMA200 ({sma_200:.2f}) — long-term downtrend")

    # Golden cross / death cross territory
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            score += 8
            if golden_cross_active:
                score += 4
                reasons.append("Recent golden cross (SMA50 > SMA200)")
            else:
                reasons.append("Golden cross territory (SMA50 > SMA200)")
        else:
            score -= 8
            reasons.append("Death cross territory (SMA50 < SMA200)")

    # ADX: trend strength
    if adx_val is not None and plus_di_val is not None and minus_di_val is not None:
        if adx_val > 25:
            if plus_di_val > minus_di_val:
                score += 10
                reasons.append(f"Strong uptrend (ADX {adx_val:.0f}, +DI > -DI)")
            else:
                score -= 10
                reasons.append(f"Strong downtrend (ADX {adx_val:.0f}, -DI > +DI)")
        elif adx_val < 20:
            # No trend — dampen toward neutral (signals unreliable)
            score = score * 0.7 + 50.0 * 0.3
            reasons.append(f"No clear trend (ADX {adx_val:.0f}) — trend signals unreliable")

    return _clamp(score), reasons


def score_momentum(rsi_val, macd_hist, macd_hist_prev,
                   stoch_k_val, stoch_d_val) -> tuple:
    """Score the momentum category (0-100)."""
    if rsi_val is None and macd_hist is None and stoch_k_val is None:
        return None, []

    score = 50.0
    reasons = []

    # RSI
    if rsi_val is not None:
        if rsi_val < 30:
            score += 20
            reasons.append(f"RSI {rsi_val:.0f} — oversold (potential buy)")
        elif rsi_val < 40:
            score += 10
            reasons.append(f"RSI {rsi_val:.0f} — approaching oversold")
        elif rsi_val > 70:
            score -= 20
            reasons.append(f"RSI {rsi_val:.0f} — overbought (potential sell)")
        elif rsi_val > 60:
            score -= 10
            reasons.append(f"RSI {rsi_val:.0f} — approaching overbought")
        else:
            reasons.append(f"RSI {rsi_val:.0f} — neutral")

    # MACD histogram
    if macd_hist is not None and macd_hist_prev is not None:
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            score += 15
            reasons.append("MACD histogram positive and rising — bullish acceleration")
        elif macd_hist > 0 and macd_hist < macd_hist_prev:
            score += 5
            reasons.append("MACD histogram positive but slowing")
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            score -= 15
            reasons.append("MACD histogram negative and falling — bearish acceleration")
        elif macd_hist < 0 and macd_hist > macd_hist_prev:
            score -= 5
            reasons.append("MACD histogram negative but recovering")

    # Stochastic
    if stoch_k_val is not None:
        if stoch_k_val < 20:
            score += 10
            reasons.append(f"Stochastic %K {stoch_k_val:.0f} — oversold")
        elif stoch_k_val > 80:
            score -= 10
            reasons.append(f"Stochastic %K {stoch_k_val:.0f} — overbought")
        if stoch_d_val is not None:
            if stoch_k_val < 20 and stoch_k_val > stoch_d_val:
                score += 5
                reasons.append("Stochastic %K crossing above %D from oversold — bullish")
            elif stoch_k_val > 80 and stoch_k_val < stoch_d_val:
                score -= 5
                reasons.append("Stochastic %K crossing below %D from overbought — bearish")

    return _clamp(score), reasons


def score_volume(obv_rising: Optional[bool], price_rising_20d: Optional[bool],
                 mfi_val: Optional[float], volume_price: Optional[dict],
                 *, liquidity: Optional[dict] = None) -> tuple:
    """
    Score the volume category (0-100).

    `liquidity` (from indicators.liquidity_score) is PENALTY-ONLY: thin volume
    subtracts, everything else adds nothing. The other bands here are
    directional — "is volume confirming this move?" — while liquidity is
    structural: "can you get out?". Awarding points for normal liquidity would
    let the two cancel into a number that means neither, and would move ~95%
    of stocks for no information. Only the genuinely untradeable tail moves.
    """
    # Note the guard deliberately ignores `liquidity`: it must not be able to
    # carry the category on its own, or a stock with no volume-confirmation
    # data at all would score 38 off a single liquidity reason.
    if obv_rising is None and mfi_val is None and volume_price is None:
        return None, []

    score = 50.0
    reasons = []

    # OBV trend vs price trend
    if obv_rising is not None and price_rising_20d is not None:
        if obv_rising and price_rising_20d:
            score += 15
            reasons.append("OBV rising with price — uptrend confirmed by volume")
        elif not obv_rising and not price_rising_20d:
            score -= 15
            reasons.append("OBV falling with price — downtrend confirmed by volume")
        elif price_rising_20d and not obv_rising:
            score -= 10
            reasons.append("Price rising but OBV falling — bearish volume divergence")
        elif not price_rising_20d and obv_rising:
            score += 10
            reasons.append("Price falling but OBV rising — bullish accumulation")

    # MFI (volume-weighted RSI)
    if mfi_val is not None:
        if mfi_val < 20:
            score += 15
            reasons.append(f"MFI {mfi_val:.0f} — oversold, money has fled")
        elif mfi_val < 30:
            score += 8
            reasons.append(f"MFI {mfi_val:.0f} — approaching oversold")
        elif mfi_val > 80:
            score -= 15
            reasons.append(f"MFI {mfi_val:.0f} — overbought, heavy buying may exhaust")
        elif mfi_val > 70:
            score -= 8
            reasons.append(f"MFI {mfi_val:.0f} — approaching overbought")

    # Volume-price confirmation
    if volume_price is not None:
        cls = volume_price.get("classification", "normal")
        vr = volume_price.get("volume_ratio", 0.0)
        chg = volume_price.get("price_change_pct", 0.0)
        if cls == "confirmed_up":
            score += 10
            reasons.append(f"Rose {chg:.1f}% on {vr:.1f}x volume — real move")
        elif cls == "confirmed_down":
            score -= 10
            reasons.append(f"Fell {abs(chg):.1f}% on {vr:.1f}x volume — strong selling")
        elif cls == "unconfirmed_up":
            score -= 5
            reasons.append(f"Rose {chg:.1f}% on low volume — may not hold")
        elif cls == "unconfirmed_down":
            score += 5
            reasons.append(f"Fell {abs(chg):.1f}% on low volume — sellers lack conviction")
        elif cls == "accumulation":
            score += 8
            reasons.append(f"Flat price on {vr:.1f}x volume — quiet accumulation")

    # Liquidity: penalty-only (see docstring). The "low" tier scores 0 but
    # still explains itself — with ~a quarter of the market in that tier,
    # penalizing it would shift the whole distribution for no signal.
    if liquidity is not None and liquidity.get("avg_volume") is not None:
        avg_vol = liquidity["avg_volume"]
        tier = liquidity.get("index_membership") or "EGX100"
        dead = liquidity.get("dead_sessions") or 0
        share = liquidity.get("traded_share")
        if share is not None and share < 0.70 and liquidity.get("thin"):
            # ABSENCE from the market's calendar, which the dead-session count
            # below cannot see: a stock that stops being quoted has no row on
            # the days it misses, so its recent rows can look perfectly healthy.
            # Say it in days-out-of-ten, which is what "you may not be able to
            # sell on the day you want to" actually means.
            score -= 12
            reasons.append(
                f"Only trades about {round(share * 10)} days in 10 that the "
                f"market is open — you may not be able to sell on the day you "
                f"want to."
            )
        elif dead >= 1 and liquidity.get("thin"):
            # Sessions with no trading at all. Naming them matters: the average
            # can look respectable off one old block trade while the stock is
            # in practice untradeable.
            score -= 12
            reasons.append(
                f"No trading on {dead} of the last 20 sessions — you may not be "
                "able to buy or sell this at the quoted price."
            )
        elif liquidity.get("thin"):
            score -= 12
            reasons.append(
                f"Thin liquidity — {avg_vol:,} shares/day, below the floor for "
                f"an {tier} name. Hard to exit; keep the position small."
            )
        elif liquidity.get("classification") == "low":
            reasons.append(
                f"Modest liquidity — {avg_vol:,} shares/day for an {tier} name."
            )

    return _clamp(score), reasons


def score_volatility(current_price, bb_upper, bb_lower, bb_middle,
                     bb_squeeze: bool = False) -> tuple:
    """Score the volatility category (0-100)."""
    if current_price is None or bb_upper is None or bb_lower is None or bb_middle is None:
        return None, []

    band_width = bb_upper - bb_lower
    if band_width <= 0:
        return None, []

    bb_position = (current_price - bb_lower) / band_width
    score = 50.0
    reasons = []

    if bb_position < 0.1:
        score += 20
        reasons.append(f"Price at lower Bollinger band — oversold (bb_pos {bb_position:.2f})")
    elif bb_position < 0.3:
        score += 10
        reasons.append(f"Price near lower band (bb_pos {bb_position:.2f})")
    elif bb_position > 0.9:
        score -= 20
        reasons.append(f"Price at upper Bollinger band — overbought (bb_pos {bb_position:.2f})")
    elif bb_position > 0.7:
        score -= 10
        reasons.append(f"Price near upper band (bb_pos {bb_position:.2f})")
    else:
        reasons.append(f"Price in Bollinger middle zone (bb_pos {bb_position:.2f})")

    if bb_squeeze:
        reasons.append("Bollinger squeeze detected — volatility contracting, breakout likely")

    return _clamp(score), reasons


def score_divergence(divergences: Optional[dict]) -> tuple:
    """Score the divergence category (0-100)."""
    if divergences is None:
        return None, []

    rsi_div = divergences.get("rsi") or {}
    macd_div = divergences.get("macd") or {}

    score = 50.0
    reasons = []

    for name, div in (("RSI", rsi_div), ("MACD", macd_div)):
        if div.get("bullish"):
            score += 15
            reasons.append(f"{name}: bullish divergence (price lower low, indicator higher low)")
        elif div.get("bearish"):
            score -= 15
            reasons.append(f"{name}: bearish divergence (price higher high, indicator lower high)")
        if div.get("hidden_bullish"):
            score += 5
            reasons.append(f"{name}: hidden bullish divergence — trend continuation up")
        if div.get("hidden_bearish"):
            score -= 5
            reasons.append(f"{name}: hidden bearish divergence — trend continuation down")

    # Double divergence bonus (both RSI and MACD agreeing)
    if rsi_div.get("bullish") and macd_div.get("bullish"):
        score += 10
        reasons.append("⚡ Double bullish divergence (RSI + MACD) — high-confidence reversal signal")
    if rsi_div.get("bearish") and macd_div.get("bearish"):
        score -= 10
        reasons.append("⚡ Double bearish divergence (RSI + MACD) — high-confidence reversal signal")

    return _clamp(score), reasons


def score_quality(multi_timeframe: Optional[dict],
                  trend_consistency: Optional[float],
                  current_drawdown_pct: Optional[float],
                  *,
                  pe_ratio: Optional[float] = None,
                  dividend_yield: Optional[float] = None,
                  loss_making: Optional[bool] = None) -> tuple:
    """
    Score the quality category (0-100).

    Rewards stocks that trend smoothly (not whipsaws) and recover well from
    drawdowns. A beginner benefits from holding "clean" trends — choppy stocks
    are where over-trading losses come from.

    The fundamentals inputs are keyword-only on purpose: callers pass them by
    name, so a new one can be added without silently shifting an existing
    positional argument into the wrong slot.

    Inputs:
      - multi_timeframe: output of indicators.multi_timeframe_alignment(daily, weekly).
                         Keys: daily_trend, weekly_trend, aligned, alignment_score.
      - trend_consistency: float 0-1; fraction of the last 20 bars where close
                           was above the 20-day SMA (higher = more consistent).
      - current_drawdown_pct: FRACTION, not a percent — negative, e.g. -0.15
                              for -15% below the 52-week high. All callers
                              build it via `(price / peak) - 1`.
      - pe_ratio: trailing P/E. Bands are centred on the EGX MEDIAN (~12.4),
                  not on a developed-market notion of "cheap" — see the band
                  comment below. None → skipped.
      - dividend_yield: PERCENT (e.g. 3.12). 0.0 means "pays nothing", which is
                        not a defect; None means unknown. Both skip the band.
      - loss_making: from diluted EPS < 0. Separate from pe_ratio because the
                     feed reports null, never a negative P/E, for loss-makers.
    """
    if (multi_timeframe is None and trend_consistency is None
            and current_drawdown_pct is None and pe_ratio is None
            and dividend_yield is None and loss_making is None):
        return None, []

    score = 50.0
    reasons = []

    if multi_timeframe is not None:
        daily = multi_timeframe.get("daily_trend", "sideways")
        weekly = multi_timeframe.get("weekly_trend", "sideways")
        aligned = multi_timeframe.get("aligned", False)
        if aligned and daily == "up":
            score += 20
            reasons.append("Daily and weekly trends both up — high-quality uptrend")
        elif aligned and daily == "down":
            score -= 20
            reasons.append("Daily and weekly trends both down — high-quality downtrend")
        elif daily == "up" and weekly == "down":
            score -= 10
            reasons.append("Daily up but weekly down — rally against the larger trend")
        elif daily == "down" and weekly == "up":
            score += 5
            reasons.append("Daily weak but weekly still up — pullback in an uptrend")

    if trend_consistency is not None:
        if trend_consistency >= 0.8:
            score += 10
            reasons.append(f"Price above SMA20 on {int(trend_consistency * 100)}% of last 20 days — steady uptrend")
        elif trend_consistency <= 0.2:
            score -= 10
            reasons.append(f"Price below SMA20 on {int((1 - trend_consistency) * 100)}% of last 20 days — steady downtrend")

    if current_drawdown_pct is not None:
        # Contract is a fraction (see docstring). Sniffing the magnitude to
        # guess the unit made a -0.9% drawdown read as -90% and scored it
        # worse than a -1.5% one.
        #
        # The peak is the 52-week high, so the reasons say so — "recent peak"
        # was vague enough that users read it as a short-term high.
        dd_pct = current_drawdown_pct * 100
        if dd_pct <= -30:
            score -= 15
            reasons.append(
                f"{abs(dd_pct):.0f}% below its 52-week high — quality impaired"
            )
        elif dd_pct <= -15:
            score -= 8
            reasons.append(f"{abs(dd_pct):.0f}% below its 52-week high")
        elif dd_pct >= -3:
            score += 5
            reasons.append(
                f"Within {abs(dd_pct):.1f}% of its 52-week high — trading at the "
                "top of its range"
            )

    # Loss-making comes from diluted EPS, not from a negative P/E: the
    # fundamentals feed reports NULL for loss-makers, so the old `pe_ratio < 0`
    # test could never fire.
    if loss_making:
        score -= 15
        reasons.append("Company is loss-making — earnings-based valuation doesn't apply")

    # P/E sub-component, centred on the EGX MEDIAN (~12.4), not on a
    # developed-market idea of cheap. The previous bands gave +8 to anything
    # under 20, which is most of this market — so simply HAVING P/E data was
    # worth points, and only ~22% of EGX stocks have it. Bands now aim to give
    # the median stock roughly nothing, making this a relative signal.
    if pe_ratio is not None:
        if pe_ratio < 3:
            # Not a bargain — a P/E under 3 in this market means the earnings
            # are non-recurring or the price has collapsed. MEGM trades at 0.7.
            score += 3
            reasons.append(
                f"P/E {pe_ratio:.1f} — implausibly low; check the earnings are recurring"
            )
        elif pe_ratio < 8:
            score += 12
            reasons.append(f"Cheap versus the EGX median of ~12 (P/E {pe_ratio:.1f})")
        elif pe_ratio < 15:
            score += 4
            reasons.append(f"Around the EGX median (P/E {pe_ratio:.1f})")
        elif pe_ratio < 25:
            score -= 2
            reasons.append(f"Above the EGX median (P/E {pe_ratio:.1f})")
        elif pe_ratio < 40:
            score -= 8
            reasons.append(f"Expensive on earnings (P/E {pe_ratio:.1f})")
        else:
            score -= 14
            reasons.append(
                f"Very expensive (P/E {pe_ratio:.1f}) — needs confirmed growth to justify"
            )

    # Dividend yield. Deliberately NON-monotonic: a very high yield on the EGX
    # is almost always a special dividend or a collapsed share price, not
    # income quality. Every reason frames the payout as evidence of
    # cash-generative discipline, never as "good income" — even 7% loses badly
    # to a ~25% T-bill, and saying otherwise would mislead.
    # 0.0 means "pays nothing", which is normal for a growth company and not a
    # defect, so it scores the same as unknown.
    if dividend_yield is not None and dividend_yield > 0:
        if dividend_yield >= 15:
            score -= 8
            reasons.append(
                f"Extreme yield (DY {dividend_yield:.1f}%) — typically a special "
                "dividend or a collapsed share price, not steady income"
            )
        elif dividend_yield >= 8:
            score += 4
            reasons.append(
                f"Very high dividend (DY {dividend_yield:.1f}%) — check it recurs"
            )
        elif dividend_yield >= 4:
            score += 8
            reasons.append(
                f"Above-median dividend (DY {dividend_yield:.1f}%) — real cash returned "
                "to shareholders, though still under the T-bill"
            )
        elif dividend_yield >= 2:
            score += 4
            reasons.append(
                f"Pays about the EGX median dividend (DY {dividend_yield:.1f}%)"
            )
        else:
            reasons.append(f"Token dividend (DY {dividend_yield:.1f}%)")

    return _clamp(score), reasons


def score_risk_adjusted(annualized_return_pct: Optional[float],
                        risk_free_rate_pct: float,
                        volatility_annualized_pct: Optional[float],
                        atr_pct_of_price: Optional[float],
                        history_days: Optional[int]) -> tuple:
    """
    Score the risk-adjusted category (0-100).

    This is the most important category for an Egyptian retail investor:
    with T-bills paying ~25% annualized risk-free, any stock returning less
    is LOSING real money vs cash. Also penalises stocks whose daily range
    (ATR) is so wide that a reasonable stop-loss would be instantly hit.

    Minimum-history gate: returns None if <120 trading days to avoid
    misleading annualization. Caller's renormalization handles that.
    """
    if history_days is not None and history_days < 120:
        return None, []
    if annualized_return_pct is None:
        return None, []

    score = 50.0
    reasons = []

    # The comparison below is biased AGAINST the stock and the size is known.
    # `annualized_return_pct` is a PRICE return, while the policy rate is a
    # TOTAL return — cash pays its yield, and the stock's dividends were dropped
    # before this subtraction. Measured across eight liquid EGX names, that is
    # worth a median 3.70 percentage points a year (see DIVIDEND_DRAG_PP_PER_YEAR).
    # It is disclosed in the reason string rather than added back, because
    # score_quality already credits dividend yield and paying a stock twice for
    # one fact produces a number that means neither.
    excess = annualized_return_pct - risk_free_rate_pct
    if excess >= 20:
        score += 25
        reasons.append(f"Ann. return {annualized_return_pct:.0f}% vs T-bill {risk_free_rate_pct:.0f}% — crushes cash")
    elif excess >= 10:
        score += 15
        reasons.append(f"Ann. return {annualized_return_pct:.0f}% — comfortably beats T-bill")
    elif excess >= 0:
        score += 5
        reasons.append(f"Ann. return {annualized_return_pct:.0f}% — marginally beats T-bill")
    elif excess >= -10:
        score -= 10
        reasons.append(
            f"Ann. return {annualized_return_pct:.0f}% — UNDERPERFORMS the "
            f"{risk_free_rate_pct:.0f}% policy rate. Note this compares price "
            f"only: dividends are not counted, and on the EGX that is worth "
            f"about {DIVIDEND_DRAG_PP_PER_YEAR:.1f} points a year, so the real "
            f"gap is narrower."
        )
    else:
        score -= 20
        reasons.append(f"Ann. return {annualized_return_pct:.0f}% — severely underperforms cash")

    # Volatility penalty: very high annualized vol makes the return unstable
    if volatility_annualized_pct is not None:
        if volatility_annualized_pct > 60:
            score -= 10
            reasons.append(f"Annualized volatility {volatility_annualized_pct:.0f}% — very swingy")
        elif volatility_annualized_pct < 20:
            score += 5
            reasons.append(f"Annualized volatility {volatility_annualized_pct:.0f}% — relatively calm")

    # ATR context: if ATR is >5% of price, any stop-loss gets whipsawed
    if atr_pct_of_price is not None:
        if atr_pct_of_price > 5:
            score -= 5
            reasons.append(f"ATR is {atr_pct_of_price:.1f}% of price — stop-losses easily triggered")

    return _clamp(score), reasons


def score_relative_strength(rs: Optional[dict]) -> tuple:
    """
    Score the relative-strength category (0-100).

    A stock outperforming EGX30 is a LEADER (institutional money preferring
    it). A stock lagging is a LAGGARD. For a beginner, avoiding laggards
    eliminates a huge class of losing trades.

    Input: rs dict from indicators.relative_strength()
      Keys: stock_return_pct, benchmark_return_pct, alpha_pct, leader, laggard.
    """
    if rs is None or rs.get("alpha_pct") is None:
        return None, []

    alpha = rs["alpha_pct"]
    score = 50.0
    reasons = []

    if alpha >= 15:
        score += 30
        reasons.append(f"Leading EGX30 by {alpha:.1f}% (30d) — clear market leader")
    elif alpha >= 5:
        score += 15
        reasons.append(f"Leading EGX30 by {alpha:.1f}% (30d)")
    elif alpha >= -2:
        reasons.append(f"Tracking EGX30 (alpha {alpha:+.1f}% over 30d)")
    elif alpha >= -10:
        score -= 15
        reasons.append(f"Lagging EGX30 by {abs(alpha):.1f}% (30d)")
    else:
        score -= 30
        reasons.append(f"Lagging EGX30 by {abs(alpha):.1f}% (30d) — significant laggard")

    # Absolute return context
    stock_ret = rs.get("stock_return_pct")
    if stock_ret is not None:
        if stock_ret < -10:
            score -= 5
            reasons.append(f"Also down {abs(stock_ret):.1f}% in absolute terms over 30d")
        elif stock_ret > 10:
            score += 5
            reasons.append(f"Up {stock_ret:.1f}% in absolute terms over 30d")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Macro modulation
# ---------------------------------------------------------------------------

def apply_macro_modulation(raw_score: float, macro: Optional[dict]) -> tuple:
    """
    Adjust the composite after weighting based on the broader EGX30 regime.

    Regime is read from macro["egx30"]["trend"] in {bullish, bearish, sideways}.

    Bullish:  no change (1.0x).
    Sideways: no change (1.0x) — a neutral market must be neutral, otherwise
              every stock carries a permanent penalty and no regime can ever
              leave a score alone.
    Bearish:  the whole distribution SHIFTS DOWN. Above 50, scores are damped
              15% toward neutral (a stock must be exceptional to still read
              "Buy" in a falling market). Below 50, scores are pushed a
              further 15% AWAY from neutral (a weak stock in a bear market is
              worse than the same stock in a calm one).

    Note this is deliberately NOT a symmetric "pull toward neutral" — only the
    bullish half is pulled in; the bearish half is pushed out.

    Returns (adjusted_score, delta, description).
    """
    if macro is None:
        return raw_score, 0.0, None

    egx30 = (macro.get("egx30") or {}) if isinstance(macro, dict) else {}
    trend = (egx30.get("trend") or "").lower()

    if trend in ("bullish", "sideways"):
        return raw_score, 0.0, None

    if trend == "bearish":
        dampen = 0.85
        reinforce = 1.15
        desc = "EGX30 bearish — bullish scores damped, bearish scores deepened"
    else:
        return raw_score, 0.0, None

    if raw_score > 50:
        adjusted = 50 + (raw_score - 50) * dampen
    else:
        adjusted = 50 - (50 - raw_score) * reinforce

    adjusted = _clamp(adjusted)
    delta = round(adjusted - raw_score, 1)
    return adjusted, delta, desc


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def score_categories(indicators: dict, extras: Optional[dict] = None) -> dict:
    """
    Run all eight category scorers. Returns {name: (score | None, reasons)}.

    WEIGHT-FREE AND MACRO-FREE BY CONSTRUCTION, and that is the whole point of
    it being its own function. These eight numbers depend only on the price
    history and the fundamentals — not on whose sliders are set how, and not on
    today's EGX30 regime. Both of those enter later, in `blend_categories`.

    That separation is what lets the nightly snapshot store a score once and
    have every user read their OWN composite out of it: the expensive part
    (400 bars, indicators, divergences) is computed once, and the cheap part
    (a weighted mean and a modulation) is redone per request. See
    `routers/cron.py` for the writer and `routers/dashboard.py` for the reader.

    See `compute_composite` for what `extras` must contain.
    """
    extras = extras or {}

    current_price = extras.get("current_price")
    sma_20 = _last_valid(indicators.get("sma_20"))
    sma_50 = _last_valid(indicators.get("sma_50"))
    sma_200 = _last_valid(indicators.get("sma_200"))
    adx_val = _last_valid(indicators.get("adx"))
    plus_di_val = _last_valid(indicators.get("plus_di"))
    minus_di_val = _last_valid(indicators.get("minus_di"))
    rsi_val = _last_valid(indicators.get("rsi"))
    macd_hist = _last_valid(indicators.get("macd_histogram"))
    macd_hist_prev = _prev_valid(indicators.get("macd_histogram"))
    stoch_k = _last_valid(indicators.get("stochastic_k"))
    stoch_d = _last_valid(indicators.get("stochastic_d"))
    mfi_val = _last_valid(indicators.get("mfi"))
    bb_upper = _last_valid(indicators.get("bollinger_upper"))
    bb_lower = _last_valid(indicators.get("bollinger_lower"))
    bb_middle = _last_valid(indicators.get("bollinger_middle"))

    trend_score, trend_reasons = score_trend(
        current_price, sma_20, sma_50, sma_200,
        adx_val, plus_di_val, minus_di_val,
        golden_cross_active=bool(extras.get("golden_cross_active", False)),
    )
    momentum_score, momentum_reasons = score_momentum(
        rsi_val, macd_hist, macd_hist_prev, stoch_k, stoch_d,
    )
    volume_score, volume_reasons = score_volume(
        extras.get("obv_rising"),
        extras.get("price_rising_20d"),
        mfi_val,
        extras.get("volume_price"),
        liquidity=extras.get("liquidity"),
    )
    volatility_score, volatility_reasons = score_volatility(
        current_price, bb_upper, bb_lower, bb_middle,
        bb_squeeze=bool(extras.get("bb_squeeze", False)),
    )
    divergence_score, divergence_reasons = score_divergence(
        extras.get("divergences"),
    )
    quality_score, quality_reasons = score_quality(
        extras.get("multi_timeframe"),
        extras.get("trend_consistency"),
        extras.get("current_drawdown_pct"),
        pe_ratio=extras.get("pe_ratio"),
        dividend_yield=extras.get("dividend_yield"),
        loss_making=extras.get("loss_making"),
    )
    # `or` would treat a legitimate 0% rate as missing — check for None.
    _rfr = extras.get("risk_free_rate_pct")
    risk_adjusted_score, risk_adjusted_reasons = score_risk_adjusted(
        extras.get("annualized_return_pct"),
        float(_rfr) if _rfr is not None else float(DEFAULT_RISK_FREE_RATE_PCT),
        extras.get("volatility_annualized_pct"),
        extras.get("atr_pct_of_price"),
        extras.get("history_days"),
    )
    relative_strength_score, relative_strength_reasons = score_relative_strength(
        extras.get("relative_strength"),
    )

    return {
        "trend": (trend_score, trend_reasons),
        "momentum": (momentum_score, momentum_reasons),
        "volume": (volume_score, volume_reasons),
        "volatility": (volatility_score, volatility_reasons),
        "divergence": (divergence_score, divergence_reasons),
        "quality": (quality_score, quality_reasons),
        "risk_adjusted": (risk_adjusted_score, risk_adjusted_reasons),
        "relative_strength": (relative_strength_score, relative_strength_reasons),
    }


def blend_categories(category_scores: dict,
                     weights: Optional[dict] = None,
                     macro: Optional[dict] = None) -> dict:
    """
    Turn eight category scores into ONE composite. Pure; the only place this
    arithmetic is spelled.

    THIS FUNCTION IS THE "ONE SCORE PER STOCK" GUARANTEE. `compute_composite`
    calls it, and so does the dashboard's snapshot reader — which holds the
    same eight numbers in Postgres rather than having just computed them. Two
    independent spellings of this weighting would let a card and its own detail
    page disagree, which is the exact failure `build_composite_extras` and
    `composite_cache_key` already exist to prevent.

    Arguments:
      category_scores: {name: float 0-100 | None}. None means the category was
                       not scorable on this stock's data, and its weight is
                       redistributed across the rest — dropped on EVERY page
                       for the same reason, never silently absorbed.
      weights:         {category: weight_percent}; normalized here.
      macro:           macro dict; the EGX30 regime modulation is applied AFTER
                       the weighted sum, so it is not a ninth category.

    Returns {score, raw_score, signal, weights, effective_weights,
             contributions, macro_adjustment, macro_context}.
    `score` is post-modulation and rounded; `raw_score` is the weighted mean
    before it.
    """
    weights = normalize_weights(weights or DEFAULT_WEIGHTS)
    scores = {name: category_scores.get(name) for name in CATEGORY_ORDER}

    available_weight_sum = sum(
        weights[name] for name in CATEGORY_ORDER if scores[name] is not None
    )

    if available_weight_sum == 0:
        # Nothing scorable at all. 50 is the neutral midpoint, and the label
        # comes from classify_signal rather than a literal: hardcoding one here
        # is how "Hold" — an instruction the backtest contradicts — survived
        # the 2026-08-26 relabelling in this one branch.
        return {
            "score": 50.0,
            "raw_score": 50.0,
            "signal": classify_signal(50.0),
            "weights": weights,
            "effective_weights": {name: 0.0 for name in CATEGORY_ORDER},
            "contributions": {name: 0.0 for name in CATEGORY_ORDER},
            "macro_adjustment": None,
            "macro_context": None,
        }

    composite = 0.0
    effective_weights = {}
    contributions = {}
    for name in CATEGORY_ORDER:
        s = scores[name]
        if s is None:
            effective_weights[name] = 0.0
            contributions[name] = 0.0
            continue
        effective_weight = weights[name] / available_weight_sum * 100
        contribution = s * effective_weight / 100.0
        effective_weights[name] = effective_weight
        contributions[name] = contribution
        composite += contribution

    composite = _clamp(composite)
    final_score, macro_delta, macro_ctx = apply_macro_modulation(composite, macro)

    return {
        "score": round(final_score, 1),
        "raw_score": composite,
        "signal": classify_signal(final_score),
        "weights": weights,
        "effective_weights": effective_weights,
        "contributions": contributions,
        "macro_adjustment": macro_delta if macro_ctx else None,
        "macro_context": macro_ctx,
    }


def compute_composite(indicators: dict, extras: Optional[dict] = None,
                      weights: Optional[dict] = None,
                      macro: Optional[dict] = None) -> dict:
    """
    Compute the composite score from pre-computed indicator arrays and extras.

    Arguments:
      indicators: dict produced by indicators.compute_all (indicator_name -> list)
                  Expected keys: sma_20, sma_50, sma_200, rsi, macd_histogram,
                  bollinger_upper/lower/middle, stochastic_k, stochastic_d,
                  adx, plus_di, minus_di, mfi, obv.
      extras:     dict with optional extra inputs:
                  - "current_price": float (latest close)
                  - "divergences": {"rsi": {...}, "macd": {...}}
                  - "volume_price": {...}
                  - "bb_squeeze": bool
                  - "golden_cross_active": bool
                  - "price_rising_20d": bool
                  - "obv_rising": bool
                  # New (8-category) inputs:
                  - "multi_timeframe": {"daily_trend", "weekly_trend", "aligned", ...}
                  - "trend_consistency": float 0-1 (fraction of last 20 days above SMA20)
                  - "current_drawdown_pct": float (negative, e.g. -0.12 for -12%)
                  - "annualized_return_pct": float
                  - "volatility_annualized_pct": float
                  - "atr_pct_of_price": float
                  - "history_days": int (for risk-adjusted min-history gate)
                  - "risk_free_rate_pct": float (usually passed through from settings)
                  - "relative_strength": output of indicators.relative_strength(...)
                  - "pe_ratio": float (trailing P/E) | None
                  - "dividend_yield": float PERCENT; 0.0 = pays nothing, None = unknown
                  - "loss_making": bool (from diluted EPS) | None
                  - "liquidity": output of indicators.liquidity_score(...)
      weights:    dict {category: weight_percent}, default DEFAULT_WEIGHTS.
      macro:      optional macro dict (from macro_fetch.get_macro()) — when
                  provided, applies a post-hoc modulation based on EGX30 trend.

    Returns:
      {
        "score": float 0-100,             (AFTER macro modulation)
        "signal": str,
        "categories": {name: {...}},
        "weights": {name: float},
        "macro_adjustment": float | None,   (signed delta from raw to final score)
        "macro_context": str | None,        (human-readable note, e.g. "EGX30 bearish")
      }
    """
    category_raw = score_categories(indicators, extras)
    blended = blend_categories(
        {name: s for name, (s, _) in category_raw.items()}, weights, macro,
    )

    weights_out = blended["weights"]
    categories_out = {}
    for name in CATEGORY_ORDER:
        s, reasons = category_raw[name]
        categories_out[name] = {
            "score": round(s, 1) if s is not None else None,
            "weight": round(weights_out[name], 2),
            "effective_weight": round(blended["effective_weights"][name], 2),
            "weighted_contribution": round(blended["contributions"][name], 2),
            "reasons": reasons,
        }

    return {
        "score": blended["score"],
        "signal": blended["signal"],
        "categories": categories_out,
        "weights": weights_out,
        "macro_adjustment": blended["macro_adjustment"],
        "macro_context": blended["macro_context"],
    }


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def get_weights_from_db(db, user_id: str = None) -> dict:
    """
    Resolve this user's composite weights.

    Read chain, PER KEY: user_settings (this user's override) -> settings (the
    global value) -> DEFAULT_WEIGHTS.

    The middle tier is what makes weights per-user without moving anyone's
    scores on deploy: an install's existing saved weights stay everyone's
    starting point until a user touches their own sliders. The per-key
    granularity also means extending CATEGORY_ORDER from 5 to 8 still
    gracefully inherits defaults for the new categories, as before.

    `user_id=None` means the anonymous/global context — used by the public
    dashboard path and by the market-regime reader, whose bands were
    calibrated at default weights.
    """
    try:
        # The %% is load-bearing. _DB.execute always passes a params tuple, so
        # psycopg parses the string for placeholders and a lone %' raises
        # ProgrammingError — which the except below swallowed into
        # DEFAULT_WEIGHTS. Saved weights were therefore NEVER read back: every
        # score in the app was computed at Beginner Safe defaults no matter
        # what the sliders said. tests pin this.
        rows = db.execute(
            "SELECT key, value FROM settings WHERE key LIKE 'weight_%%'"
        ).fetchall()
    except Exception:
        return dict(DEFAULT_WEIGHTS)

    lookup = {r[0]: r[1] for r in rows}

    if user_id:
        try:
            user_rows = db.execute(
                "SELECT key, value FROM user_settings "
                "WHERE user_id = %s AND key LIKE 'weight_%%'",
                (user_id,),
            ).fetchall()
            lookup.update({r[0]: r[1] for r in user_rows})
        except Exception:
            # A missing user_settings table (cold DB) must not take scoring
            # down — the global tier is a correct answer.
            pass

    out = {}
    for name in CATEGORY_ORDER:
        raw = lookup.get(f"weight_{name}")
        try:
            out[name] = float(raw) if raw is not None else DEFAULT_WEIGHTS[name]
        except (TypeError, ValueError):
            out[name] = DEFAULT_WEIGHTS[name]
    return out


def weights_hash(weights: dict) -> str:
    """A small stable hash of the weights for cache key invalidation."""
    # Two decimals, matching normalize_weights' precision. Rounding to int
    # would collide distinct weight sets onto one cache key, so a small
    # slider tweak would serve a stale score.
    parts = [f"{k}{float(weights.get(k, 0)):.2f}" for k in CATEGORY_ORDER]
    return "_".join(parts)
