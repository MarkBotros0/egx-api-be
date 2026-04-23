"""
Max-buy-price helper.

Answers the beginner question "how high can I chase this stock before the
trade stops making sense?" with a single number derived from the two
guardrails a retail investor should internalize first:

    1. Support-proximity cap
       Don't pay more than 5% above the nearest support level. Past 5%, a
       stop-loss below support is either too far away (a big loss if wrong)
       or must be placed above support (where normal noise whips you out).

    2. Risk:reward cap (R:R >= 2:1)
       The distance from entry to the nearest resistance (the implicit
       target) must be at least twice the distance from entry to the
       stop-loss. If not, the trade's expected value is too low to justify
       the stock-market risk — cash (T-bills) is a better home for that
       capital.

The max buy price is the LOWER of the two caps. When one cap is missing
(e.g. no resistance identified), the other stands alone; if neither can be
computed the helper returns None and the UI hides the card.

Stop-loss suggestion is 1 × ATR below support (beginner-friendly: buffered,
not pinned, to absorb normal daily noise without getting whipsawed).

This module is pure: no DB, no I/O. Consumes values already in the analysis
response.
"""

from __future__ import annotations

import math
from typing import Optional


# Beginner rule: don't chase more than 5% above nearest support.
SUPPORT_CHASE_PCT = 0.05

# Minimum acceptable reward/risk for an entry in this app's guidance.
MIN_RISK_REWARD = 2.0

# Fallback ATR% of price when ATR isn't available (keeps math robust for
# short-history stocks). 2% is a reasonable noise-floor proxy for EGX liquids.
FALLBACK_ATR_PCT_OF_PRICE = 0.02

# How far above max_buy_price current_price can drift before we flip verdict
# from "near_limit" to "above_limit". 2% gives a small usability buffer so
# the card isn't red just because of intraday tick-up.
NEAR_LIMIT_BAND_PCT = 0.02


def compute_max_buy_price(
    current_price: Optional[float],
    nearest_support: Optional[float],
    nearest_resistance: Optional[float],
    atr_value: Optional[float],
) -> Optional[dict]:
    """
    Compute the maximum price a beginner should be willing to pay today,
    with the reasoning that got us there.

    Returns:
        {
          "price": float,                    # max EGP per share
          "verdict": "ok" | "near_limit" | "above_limit",
          "verdict_distance_pct": float,     # (current - max) / max * 100; negative = below max = good
          "stop_loss": float,                # suggested stop in EGP
          "target": float | None,            # implicit target (resistance)
          "risk_reward_at_max": float | None,# R:R if you entered at the max
          "caps": {
            "support": float | None,
            "risk_reward": float | None,
          },
          "reasons": list[str],
        }
        or None if we can't compute meaningfully.
    """
    if current_price is None or current_price <= 0:
        return None
    if nearest_support is None or nearest_support <= 0:
        return None

    # ATR fallback — keeps the helper usable on short-history stocks.
    if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
        atr_value = current_price * FALLBACK_ATR_PCT_OF_PRICE

    stop_loss = round(nearest_support - atr_value, 2)

    # Cap A: support-proximity
    cap_support = round(nearest_support * (1 + SUPPORT_CHASE_PCT), 2)

    # Cap B: risk:reward >= MIN_RISK_REWARD, if we have a target
    cap_rr: Optional[float] = None
    if nearest_resistance is not None and nearest_resistance > stop_loss:
        # Solve (target - entry) >= k * (entry - stop)
        # → entry <= (target + k*stop) / (1 + k)
        k = MIN_RISK_REWARD
        cap_rr = round((nearest_resistance + k * stop_loss) / (1 + k), 2)
        # Clamp: never recommend buying above the target itself.
        if cap_rr >= nearest_resistance:
            cap_rr = round(nearest_resistance, 2)

    candidate_caps = [c for c in (cap_support, cap_rr) if c is not None]
    if not candidate_caps:
        return None
    max_price = min(candidate_caps)

    # Degenerate: max came out below stop-loss. Means resistance is so close to
    # support that there's no viable entry. Better to suppress than mislead.
    if max_price <= stop_loss:
        return None

    # R:R at the max price itself
    risk = max_price - stop_loss
    rr_at_max: Optional[float] = None
    if nearest_resistance is not None and risk > 0:
        reward = nearest_resistance - max_price
        if reward > 0:
            rr_at_max = round(reward / risk, 2)

    distance_pct = (current_price - max_price) / max_price * 100

    if current_price <= max_price:
        verdict = "ok"
    elif current_price <= max_price * (1 + NEAR_LIMIT_BAND_PCT):
        verdict = "near_limit"
    else:
        verdict = "above_limit"

    reasons: list = []
    reasons.append(
        f"Stop-loss {stop_loss:.2f} = 1 ATR ({atr_value:.2f}) below support "
        f"{nearest_support:.2f}."
    )
    if nearest_resistance is not None:
        reasons.append(
            f"Target {nearest_resistance:.2f} = nearest resistance."
        )
    reasons.append(
        f"Support cap: 5% above support = {cap_support:.2f}."
    )
    if cap_rr is not None:
        reasons.append(
            f"R:R cap (>= 2:1): entry ≤ (target + 2 × stop) / 3 = {cap_rr:.2f}."
        )
    reasons.append(
        f"Max buy price = lower of the two caps = {max_price:.2f}."
    )
    if rr_at_max is not None:
        reasons.append(
            f"At the max, R:R would be ≈ {rr_at_max:.2f}:1."
        )

    return {
        "price": round(max_price, 2),
        "verdict": verdict,
        "verdict_distance_pct": round(distance_pct, 2),
        "stop_loss": stop_loss,
        "target": round(nearest_resistance, 2) if nearest_resistance is not None else None,
        "risk_reward_at_max": rr_at_max,
        "caps": {
            "support": cap_support,
            "risk_reward": cap_rr,
        },
        "reasons": reasons,
    }
