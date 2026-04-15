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
    """Map a 0-100 composite score to a human-readable signal string."""
    if score >= SCORE_BUY_MAX:
        return "Strong Buy"
    if score >= SCORE_HOLD_MAX:
        return "Buy"
    if score >= SCORE_SELL_MAX:
        return "Hold"
    if score >= SCORE_STRONG_SELL_MAX:
        return "Sell"
    return "Strong Sell"


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
                 mfi_val: Optional[float], volume_price: Optional[dict]) -> tuple:
    """Score the volume category (0-100)."""
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
                  current_drawdown_pct: Optional[float]) -> tuple:
    """
    Score the quality category (0-100).

    Rewards stocks that trend smoothly (not whipsaws) and recover well from
    drawdowns. A beginner benefits from holding "clean" trends — choppy stocks
    are where over-trading losses come from.

    Inputs:
      - multi_timeframe: output of indicators.multi_timeframe_alignment(daily, weekly).
                         Keys: daily_trend, weekly_trend, aligned, alignment_score.
      - trend_consistency: float 0-1; fraction of the last 20 bars where close
                           was above the 20-day SMA (higher = more consistent).
      - current_drawdown_pct: float (negative number, e.g. -0.15 for -15%); how
                              far the stock is below its recent peak.
    """
    if multi_timeframe is None and trend_consistency is None and current_drawdown_pct is None:
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
        dd_pct = current_drawdown_pct * 100 if abs(current_drawdown_pct) <= 1 else current_drawdown_pct
        if dd_pct <= -30:
            score -= 15
            reasons.append(f"Severe drawdown ({dd_pct:.0f}% from peak) — quality impaired")
        elif dd_pct <= -15:
            score -= 8
            reasons.append(f"Moderate drawdown ({dd_pct:.0f}% from peak)")
        elif dd_pct >= -3:
            score += 5
            reasons.append("Near recent peak — strong quality")

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
        reasons.append(f"Ann. return {annualized_return_pct:.0f}% — UNDERPERFORMS T-bill {risk_free_rate_pct:.0f}%")
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

    Bullish: no change (1.0×).
    Sideways: dampen bullish-side scores by 5%, reinforce bearish-side by 5%.
    Bearish: dampen bullish-side scores by 15%, reinforce bearish-side by 15%.

    Scores are pulled toward neutral (50) in bear markets — a stock must be
    exceptional to still register a "Buy" in a falling market.

    Returns (adjusted_score, delta, description).
    """
    if macro is None:
        return raw_score, 0.0, None

    egx30 = (macro.get("egx30") or {}) if isinstance(macro, dict) else {}
    trend = (egx30.get("trend") or "").lower()

    if trend == "bullish":
        return raw_score, 0.0, None

    if trend == "sideways":
        dampen = 0.95
        reinforce = 1.05
        desc = "EGX30 sideways"
    elif trend == "bearish":
        dampen = 0.85
        reinforce = 1.15
        desc = "EGX30 bearish — scores pulled toward neutral"
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
    extras = extras or {}
    weights = normalize_weights(weights or DEFAULT_WEIGHTS)

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

    # Fall back to the last close if current_price not supplied explicitly
    if current_price is None:
        # Not in `indicators`, but extras typically carries it.
        current_price = None

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
    )
    risk_adjusted_score, risk_adjusted_reasons = score_risk_adjusted(
        extras.get("annualized_return_pct"),
        float(extras.get("risk_free_rate_pct") or 25.0),
        extras.get("volatility_annualized_pct"),
        extras.get("atr_pct_of_price"),
        extras.get("history_days"),
    )
    relative_strength_score, relative_strength_reasons = score_relative_strength(
        extras.get("relative_strength"),
    )

    category_raw = {
        "trend": (trend_score, trend_reasons),
        "momentum": (momentum_score, momentum_reasons),
        "volume": (volume_score, volume_reasons),
        "volatility": (volatility_score, volatility_reasons),
        "divergence": (divergence_score, divergence_reasons),
        "quality": (quality_score, quality_reasons),
        "risk_adjusted": (risk_adjusted_score, risk_adjusted_reasons),
        "relative_strength": (relative_strength_score, relative_strength_reasons),
    }

    # Renormalize weights across categories that could be scored
    available_weight_sum = sum(weights[k] for k, (s, _) in category_raw.items() if s is not None)
    if available_weight_sum == 0:
        # Nothing scorable at all — return a neutral hold
        return {
            "score": 50.0,
            "signal": "Hold",
            "categories": {
                name: {
                    "score": None,
                    "weight": weights[name],
                    "effective_weight": 0.0,
                    "weighted_contribution": 0.0,
                    "reasons": reasons,
                } for name, (s, reasons) in category_raw.items()
            },
            "weights": weights,
            "macro_adjustment": None,
            "macro_context": None,
        }

    composite = 0.0
    categories_out = {}
    for name in CATEGORY_ORDER:
        s, reasons = category_raw[name]
        w_raw = weights[name]
        if s is None:
            effective_weight = 0.0
            contribution = 0.0
        else:
            effective_weight = w_raw / available_weight_sum * 100
            contribution = s * effective_weight / 100.0
            composite += contribution
        categories_out[name] = {
            "score": round(s, 1) if s is not None else None,
            "weight": round(w_raw, 2),
            "effective_weight": round(effective_weight, 2),
            "weighted_contribution": round(contribution, 2),
            "reasons": reasons,
        }

    composite = _clamp(composite)

    # Apply macro modulation (post-hoc multiplier, not a category)
    final_score, macro_delta, macro_ctx = apply_macro_modulation(composite, macro)

    return {
        "score": round(final_score, 1),
        "signal": classify_signal(final_score),
        "categories": categories_out,
        "weights": weights,
        "macro_adjustment": macro_delta if macro_ctx else None,
        "macro_context": macro_ctx,
    }


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def get_weights_from_db(db) -> dict:
    """
    Read weight_* rows from the settings table. Missing rows fall back to
    DEFAULT_WEIGHTS per-key — so existing DBs with only the original 5 keys
    will gracefully inherit defaults for new categories. Non-numeric values
    also fall back to default.
    """
    try:
        rows = db.execute(
            "SELECT key, value FROM settings WHERE key LIKE 'weight_%'"
        ).fetchall()
    except Exception:
        return dict(DEFAULT_WEIGHTS)

    lookup = {r[0]: r[1] for r in rows}
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
    parts = [f"{k}{int(round(weights.get(k, 0)))}" for k in CATEGORY_ORDER]
    return "_".join(parts)
