"""
Key levels + entry/exit zones.

Turns raw support/resistance arrays into beginner-friendly decision helpers:

- `compute_key_levels` — nearest support/resistance with distance percentages,
  rendered in the UI as a standalone card next to the price chart.
- `compute_entry_exit` — tactical entry/exit zones with confidence tiers,
  confirming momentum filters (RSI, Stochastic), and an ATR-based stop-loss
  suggestion. Surfaces as signals in portfolio advice and as a card on the
  stock detail page.

Pure functions, no DB or network access. Scoring is intentionally NOT modified
by this module — the composite score stays focused on indicators, and the
entry/exit layer rides alongside it as independent guidance.
"""

from typing import Optional


# --- Thresholds (kept local — tuning knobs specific to this module) ---

# Entry: price within this % of nearest support = "near support"
ENTRY_NEAR_SUPPORT_PCT = 5.0
# Exit: price within this % of nearest resistance = "near resistance"
EXIT_NEAR_RESISTANCE_PCT = 3.0

# Confirmation filters. Used to bucket confidence, NOT to veto the zone —
# we still show low-confidence zones so the user sees them.
ENTRY_RSI_OVERSOLD = 40
ENTRY_RSI_NEUTRAL_MAX = 55
ENTRY_RSI_WARM_MAX = 65
EXIT_RSI_HOT_MIN = 75
EXIT_RSI_WARM_MIN = 65
STOCH_OVERBOUGHT = 80

# Stop-loss: multiple of ATR placed below support. CLAUDE.md calls out
# 1.5–2× ATR as the house convention — pick 1.5 for a tighter stop.
STOP_LOSS_ATR_MULTIPLIER = 1.5
# Fallback stop-loss when ATR is unavailable — 2% below support.
STOP_LOSS_FALLBACK_PCT = 0.02


def _distance_pct(current: float, target: float) -> float:
    """
    Signed % distance from current price to a target level, as fraction of
    current price. Positive means target is ABOVE current.
    """
    if current <= 0:
        return 0.0
    return (target - current) / current * 100.0


def compute_key_levels(current_price: float, support_resistance: dict) -> dict:
    """
    Summarize the nearest support/resistance for UI display.

    `support_resistance` is the dict returned by indicators.support_resistance:
    `{"supports": [{price, strength}, ...], "resistances": [...]}`.

    Returns:
        {
          "current_price": float,
          "nearest_support": {"price", "distance_pct", "strength"} | None,
          "nearest_resistance": {"price", "distance_pct", "strength"} | None,
          "room_to_support_pct": float | None,     # how far below current (negative)
          "room_to_resistance_pct": float | None,  # how far above current (positive)
        }

    `distance_pct` is signed: negative for supports (below price), positive
    for resistances (above price). When price has broken through a level,
    the sign flips and the caller can detect it.
    """
    supports = (support_resistance or {}).get("supports") or []
    resistances = (support_resistance or {}).get("resistances") or []

    # "Nearest support" = highest support price that is <= current_price.
    # (Falling back to the closest one if all are above current.)
    below = [s for s in supports if s.get("price") is not None and s["price"] <= current_price]
    nearest_support_raw = max(below, key=lambda s: s["price"]) if below else (
        min(supports, key=lambda s: abs(s["price"] - current_price)) if supports else None
    )

    # "Nearest resistance" = lowest resistance price that is >= current_price.
    above = [r for r in resistances if r.get("price") is not None and r["price"] >= current_price]
    nearest_resistance_raw = min(above, key=lambda r: r["price"]) if above else (
        min(resistances, key=lambda r: abs(r["price"] - current_price)) if resistances else None
    )

    ns = None
    room_to_support = None
    if nearest_support_raw is not None:
        ns_price = float(nearest_support_raw["price"])
        ns = {
            "price": round(ns_price, 2),
            "distance_pct": round(_distance_pct(current_price, ns_price), 2),
            "strength": int(nearest_support_raw.get("strength", 1)),
        }
        room_to_support = ns["distance_pct"]

    nr = None
    room_to_resistance = None
    if nearest_resistance_raw is not None:
        nr_price = float(nearest_resistance_raw["price"])
        nr = {
            "price": round(nr_price, 2),
            "distance_pct": round(_distance_pct(current_price, nr_price), 2),
            "strength": int(nearest_resistance_raw.get("strength", 1)),
        }
        room_to_resistance = nr["distance_pct"]

    return {
        "current_price": round(float(current_price), 2),
        "nearest_support": ns,
        "nearest_resistance": nr,
        "room_to_support_pct": room_to_support,
        "room_to_resistance_pct": room_to_resistance,
    }


def _entry_confidence(distance_pct: float, strength: int,
                      rsi: Optional[float], stoch_k: Optional[float]) -> Optional[str]:
    """
    Bucket entry confidence. Returns None when not close enough to qualify.
    Low-confidence zones are still returned so the user can see them.
    """
    if distance_pct > ENTRY_NEAR_SUPPORT_PCT:
        return None

    # Momentum veto: if clearly overbought, this is not an entry zone at all.
    if rsi is not None and rsi >= ENTRY_RSI_WARM_MAX:
        return None
    if stoch_k is not None and stoch_k >= STOCH_OVERBOUGHT:
        return None

    # High: oversold AND strong support
    if (rsi is not None and rsi < ENTRY_RSI_OVERSOLD) and strength >= 3:
        return "high"
    # Medium: neutral momentum AND decent support
    if (rsi is None or rsi < ENTRY_RSI_NEUTRAL_MAX) and strength >= 2:
        return "medium"
    # Low: anything remaining within the zone
    return "low"


def _exit_confidence(distance_pct: float, strength: int,
                     rsi: Optional[float], stoch_k: Optional[float]) -> Optional[str]:
    """
    Bucket exit confidence. Returns None when price isn't near (or just past)
    resistance with some overbought confirmation.
    """
    # Exit zone applies when price is within range BELOW resistance OR has just
    # broken above (distance_pct between -1% and +EXIT_NEAR_RESISTANCE_PCT).
    # Break-above without overbought RSI isn't an exit — it's a breakout.
    if distance_pct < -1.0 or distance_pct > EXIT_NEAR_RESISTANCE_PCT:
        return None

    # Need overbought confirmation for an exit call — otherwise it's just
    # "approaching resistance" which is already an info-level signal elsewhere.
    rsi_hot = rsi is not None and rsi >= EXIT_RSI_WARM_MIN
    stoch_hot = stoch_k is not None and stoch_k >= STOCH_OVERBOUGHT
    if not (rsi_hot or stoch_hot):
        return None

    # High: very overbought AND strong resistance
    if rsi is not None and rsi >= EXIT_RSI_HOT_MIN and strength >= 3:
        return "high"
    # Medium: warm RSI AND decent resistance
    if rsi is not None and rsi >= EXIT_RSI_WARM_MIN and strength >= 2:
        return "medium"
    return "low"


def compute_entry_exit(
    current_price: float,
    support_resistance: dict,
    rsi_latest: Optional[float] = None,
    stoch_k_latest: Optional[float] = None,
    atr_latest: Optional[float] = None,
) -> dict:
    """
    Tactical entry/exit zones.

    Entry zone: price is within 5% of a support AND not overbought. Surfaces a
    suggested buy-band and an ATR-based stop-loss.

    Exit zone: price is within 3% of (or just past) a resistance AND
    overbought. Surfaces a suggested trim-band.

    Both zones may be inactive simultaneously — the common case between
    levels. The confidence tier is the primary call-to-action signal for the
    UI; "active: true with low confidence" renders as a soft hint, "high"
    renders as an action-required style.

    Returns:
        {"entry_zone": {...}, "exit_zone": {...}}

    Zone shape:
        {
          "active": bool,
          "confidence": "low" | "medium" | "high" | None,
          "price_range": {"low": float, "high": float} | None,
          "suggested_stop_loss": float | None,   # entry only
          "reasons": [str],
        }
    """
    key = compute_key_levels(current_price, support_resistance)
    ns = key["nearest_support"]
    nr = key["nearest_resistance"]

    # --- Entry zone ---

    entry: dict = {
        "active": False, "confidence": None, "price_range": None,
        "suggested_stop_loss": None, "reasons": [],
    }
    if ns is not None:
        # Distance for supports is signed: negative = support is BELOW price
        # (the normal case). We want absolute proximity from above.
        dist = abs(ns["distance_pct"]) if ns["distance_pct"] <= 0 else None
        if dist is not None:
            conf = _entry_confidence(dist, ns["strength"], rsi_latest, stoch_k_latest)
            if conf is not None:
                sp = ns["price"]
                # A narrow band just above the support level.
                entry["price_range"] = {
                    "low": round(sp, 2),
                    "high": round(sp * (1 + ENTRY_NEAR_SUPPORT_PCT / 100 / 2), 2),
                }
                # Stop-loss: ATR-based when possible, else a 2% buffer.
                if atr_latest is not None and atr_latest > 0:
                    entry["suggested_stop_loss"] = round(
                        sp - STOP_LOSS_ATR_MULTIPLIER * atr_latest, 2
                    )
                else:
                    entry["suggested_stop_loss"] = round(sp * (1 - STOP_LOSS_FALLBACK_PCT), 2)
                entry["active"] = True
                entry["confidence"] = conf

                reasons = [
                    f"Price is {dist:.1f}% above support at {sp:.2f} "
                    f"(tested {ns['strength']}x)."
                ]
                if rsi_latest is not None:
                    if rsi_latest < ENTRY_RSI_OVERSOLD:
                        reasons.append(f"RSI at {rsi_latest:.0f} — oversold.")
                    elif rsi_latest < ENTRY_RSI_NEUTRAL_MAX:
                        reasons.append(f"RSI at {rsi_latest:.0f} — not overbought.")
                if stoch_k_latest is not None and stoch_k_latest < 30:
                    reasons.append(f"Stochastic %K at {stoch_k_latest:.0f} — oversold.")
                entry["reasons"] = reasons

    # --- Exit zone ---

    exit_z: dict = {
        "active": False, "confidence": None, "price_range": None, "reasons": [],
    }
    if nr is not None:
        dist = nr["distance_pct"]  # signed: positive = resistance above price
        conf = _exit_confidence(dist, nr["strength"], rsi_latest, stoch_k_latest)
        if conf is not None:
            rp = nr["price"]
            exit_z["price_range"] = {
                "low": round(rp * (1 - EXIT_NEAR_RESISTANCE_PCT / 100 / 2), 2),
                "high": round(rp, 2),
            }
            exit_z["active"] = True
            exit_z["confidence"] = conf

            reasons = [
                f"Price is {abs(dist):.1f}% "
                f"{'below' if dist >= 0 else 'above'} resistance at {rp:.2f} "
                f"(tested {nr['strength']}x)."
            ]
            if rsi_latest is not None and rsi_latest >= EXIT_RSI_HOT_MIN:
                reasons.append(f"RSI at {rsi_latest:.0f} — overbought.")
            elif rsi_latest is not None and rsi_latest >= EXIT_RSI_WARM_MIN:
                reasons.append(f"RSI at {rsi_latest:.0f} — running warm.")
            if stoch_k_latest is not None and stoch_k_latest >= STOCH_OVERBOUGHT:
                reasons.append(f"Stochastic %K at {stoch_k_latest:.0f} — overbought.")
            exit_z["reasons"] = reasons

    return {"entry_zone": entry, "exit_zone": exit_z}
