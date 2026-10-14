"""
Score alerts — the pure decision layer: who should hear that a stock's score
moved, and what the notification says. No DB, no network; `alert_store` feeds
it and `expo_push` delivers what it decides.

An alert is an INDICATOR, never an instruction. The walk-forward backtest
(see classify_signal) shows a single stock's score does not rank its future
returns, so a push that framed a jump as an "opportunity" would claim what the
app's own evidence contradicts. The copy states the move and the band, and a
test bans instruction words outright.

Both ends of a comparison are blended with the SAME macro regime: a regime
flip moves every score at once, and only the stock's own movement should
alert. Each subscriber is judged on their own weights, so the number in the
push is the number their stock page shows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from app.core.composite import CATEGORY_ORDER, blend_categories, classify_signal
from app.core.markets import EG
from app.core.constants import (
    ALERT_DAILY_CAP,
    ALERT_QUIET_END_HOUR,
    ALERT_QUIET_START_HOUR,
)

BANNED_WORDS = ("buy", "sell", "opportunity", "risk", "recommend")
SUMMARY_SYMBOL = "*"


@dataclass(frozen=True)
class Subscriber:
    user_id: str
    weights: dict
    rise: Optional[int]
    fall: Optional[int]


@dataclass(frozen=True)
class Signal:
    user_id: str
    symbol: str
    session_date: str
    direction: str          # "up" | "down" | "summary"
    before: float
    after: float


def _scorable(categories: dict) -> bool:
    return any(categories.get(name) is not None for name in CATEGORY_ORDER)


def detect_signals(symbol: str, session_date: str, baseline: dict, current: dict,
                   macro: Optional[dict], subscribers: list[Subscriber],
                   already: set[tuple[str, str, str]]) -> list[Signal]:
    """Signals for one symbol across its subscribers. `already` = {(user, symbol, direction)}."""
    if not _scorable(baseline) or not _scorable(current):
        return []
    out = []
    for s in subscribers:
        before = blend_categories(baseline, s.weights, macro)["score"]
        after = blend_categories(current, s.weights, macro)["score"]
        delta = round(after - before, 1)
        direction = None
        if s.rise is not None and delta >= s.rise:
            direction = "up"
        elif s.fall is not None and delta <= -s.fall:
            direction = "down"
        if direction is None or (s.user_id, symbol, direction) in already:
            continue
        out.append(Signal(s.user_id, symbol, session_date, direction, before, after))
    return out


def apply_daily_cap(signals: list[Signal], sent_today: dict[str, int], summary_sent: set[str],
                    cap: int = ALERT_DAILY_CAP) -> list[tuple[Signal, str]]:
    """
    [(signal, "pending" | "suppressed")]. The cap-th+1 signal of a user's day
    becomes ONE summary alert (unless one was already queued); the rest are
    recorded as suppressed so a later call does not re-detect them.
    """
    counts = dict(sent_today)
    summarized = set(summary_sent)
    out = []
    for sig in signals:
        n = counts.get(sig.user_id, 0)
        if n < cap:
            out.append((sig, "pending"))
            counts[sig.user_id] = n + 1
            continue
        if sig.user_id not in summarized:
            summarized.add(sig.user_id)
            out.append((replace(sig, symbol=SUMMARY_SYMBOL, direction="summary",
                                before=0.0, after=0.0), "pending"))
        out.append((sig, "suppressed"))
    return out


def alert_copy(signal: Signal) -> tuple[str, str]:
    """(title, body). Facts only: the move, the scores, the band."""
    if signal.direction == "summary":
        return ("More stocks moved today",
                "Open Borsa Compass to see your scores.")
    before = round(signal.before)
    after = round(signal.after)
    points = abs(after - before)
    word = "up" if signal.direction == "up" else "down"
    title = f"{signal.symbol} score {word} {points} points"
    body = f"{before} → {after} · now {classify_signal(signal.after)}"
    return title, body


def in_quiet_hours(now_utc: datetime) -> bool:
    hour = now_utc.astimezone(ZoneInfo(EG.timezone)).hour
    return hour >= ALERT_QUIET_START_HOUR or hour < ALERT_QUIET_END_HOUR
