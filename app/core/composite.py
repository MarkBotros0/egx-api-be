"""
Composite Score Engine
======================

Combines the individual technical indicators into a single 0-100 score that
answers: "Should I buy, hold, or sell this stock right now?"

The composite is a weighted average of 5 category sub-scores, each 0-100:

  1. Trend       (default 25%) — SMA crossovers, ADX strength, DI+/DI-
  2. Momentum    (default 25%) — RSI, MACD, Stochastic
  3. Volume      (default 20%) — OBV trend, MFI, volume-price confirmation
  4. Volatility  (default 15%) — Bollinger Band position, Bollinger squeeze
  5. Divergence  (default 15%) — RSI divergence, MACD divergence

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


DEFAULT_WEIGHTS = {
    "trend": 25,
    "momentum": 25,
    "volume": 20,
    "volatility": 15,
    "divergence": 15,
}

CATEGORY_ORDER = ["trend", "momentum", "volume", "volatility", "divergence"]

PRESETS = {
    "balanced": {"trend": 25, "momentum": 25, "volume": 20, "volatility": 15, "divergence": 15},
    "trend_follower": {"trend": 40, "momentum": 25, "volume": 15, "volatility": 15, "divergence": 5},
    "reversal_hunter": {"trend": 15, "momentum": 25, "volume": 15, "volatility": 15, "divergence": 30},
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
    if score >= 80:
        return "Strong Buy"
    if score >= 60:
        return "Buy"
    if score >= 40:
        return "Hold"
    if score >= 20:
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


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def compute_composite(indicators: dict, extras: Optional[dict] = None,
                      weights: Optional[dict] = None) -> dict:
    """
    Compute the composite score from pre-computed indicator arrays and extras.

    Arguments:
      indicators: dict produced by _indicators.compute_all (indicator_name -> list)
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
      weights:    dict {category: weight_percent}, default DEFAULT_WEIGHTS.

    Returns:
      {
        "score": float 0-100,
        "signal": str,
        "categories": {name: {"score": float|None, "weight": float,
                              "weighted_contribution": float, "reasons": [str]}},
        "weights": {name: float}   (the effective weights used, normalized)
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

    category_raw = {
        "trend": (trend_score, trend_reasons),
        "momentum": (momentum_score, momentum_reasons),
        "volume": (volume_score, volume_reasons),
        "volatility": (volatility_score, volatility_reasons),
        "divergence": (divergence_score, divergence_reasons),
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
                    "weighted_contribution": 0.0,
                    "reasons": reasons,
                } for name, (s, reasons) in category_raw.items()
            },
            "weights": weights,
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

    return {
        "score": round(composite, 1),
        "signal": classify_signal(composite),
        "categories": categories_out,
        "weights": weights,
    }


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def get_weights_from_db(db) -> dict:
    """
    Read the 5 weight_* rows from the settings table. Missing rows fall back
    to DEFAULT_WEIGHTS. Non-numeric values fall back to their default.
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
