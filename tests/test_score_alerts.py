"""Score alerts — the pure decision layer. No DB, no network."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from app.core.composite import CATEGORY_ORDER, PRESETS, blend_categories
from app.core.score_alerts import (
    BANNED_WORDS,
    Signal,
    Subscriber,
    alert_copy,
    apply_daily_cap,
    detect_signals,
    in_quiet_hours,
)

BAL = PRESETS["balanced"]


def cats(value):
    return {name: float(value) for name in CATEGORY_ORDER}


def sub(user="u1", rise=10, fall=10, weights=BAL):
    return Subscriber(user_id=user, weights=weights, rise=rise, fall=fall)


def test_a_rise_of_exactly_the_threshold_fires_up():
    [s] = detect_signals("COMI", "2026-10-11", cats(50), cats(60), None, [sub()], set())
    assert (s.direction, s.before, s.after) == ("up", 50.0, 60.0)


def test_a_fall_of_exactly_the_threshold_fires_down():
    [s] = detect_signals("COMI", "2026-10-11", cats(60), cats(50), None, [sub()], set())
    assert s.direction == "down"


def test_just_under_the_threshold_fires_nothing():
    assert detect_signals("COMI", "d", cats(50), cats(59.9), None, [sub()], set()) == []


def test_a_direction_set_to_none_is_off():
    assert detect_signals("COMI", "d", cats(50), cats(70), None, [sub(rise=None)], set()) == []
    assert detect_signals("COMI", "d", cats(70), cats(50), None, [sub(fall=None)], set()) == []


def test_already_logged_direction_is_skipped_but_the_other_is_not():
    already = {("u1", "COMI", "up")}
    assert detect_signals("COMI", "d", cats(50), cats(70), None, [sub()], already) == []
    [s] = detect_signals("COMI", "d", cats(70), cats(50), None, [sub()], already)
    assert s.direction == "down"


def test_unscorable_baseline_or_current_fires_nothing():
    empty = {name: None for name in CATEGORY_ORDER}
    assert detect_signals("COMI", "d", empty, cats(90), None, [sub()], set()) == []
    assert detect_signals("COMI", "d", cats(10), empty, None, [sub()], set()) == []
    assert detect_signals("COMI", "d", {}, cats(90), None, [sub()], set()) == []


def test_the_same_regime_on_both_ends_means_a_regime_alone_never_fires():
    bearish = {"egx30": {"trend": "bearish"}}
    out = detect_signals("COMI", "d", cats(60), cats(60), bearish, [sub()], set())
    assert out == []


def test_each_subscriber_is_judged_on_their_own_weights():
    before = cats(50)
    after = dict(before, trend=100.0)   # only trend moved
    trend_heavy = {name: (100 if name == "trend" else 0) for name in CATEGORY_ORDER}
    flat = {name: 1 for name in CATEGORY_ORDER}
    out = detect_signals("COMI", "d", before, after, None,
                         [sub("heavy", weights=trend_heavy), sub("flat", weights=flat)], set())
    assert [s.user_id for s in out] == ["heavy"]


def test_scores_match_blend_categories_exactly():
    before, after = cats(40), cats(80)
    [s] = detect_signals("COMI", "d", before, after, None, [sub()], set())
    assert s.before == blend_categories(before, BAL, None)["score"]
    assert s.after == blend_categories(after, BAL, None)["score"]


def _sig(user, i):
    return Signal(user, f"S{i}", "d", "up", 50.0, 61.0)


def test_cap_ten_then_one_summary_then_suppressed():
    signals = [_sig("u1", i) for i in range(13)]
    out = apply_daily_cap(signals, sent_today={}, summary_sent=set())
    statuses = [(s.symbol, st) for s, st in out]
    assert [st for _, st in statuses[:10]] == ["pending"] * 10
    assert statuses[10] == ("*", "pending")
    assert [st for _, st in statuses[11:]] == ["suppressed"] * 3


def test_cap_counts_what_was_already_sent_today_and_summary_is_sent_once():
    out = apply_daily_cap([_sig("u1", 1)], sent_today={"u1": 10}, summary_sent={"u1"})
    assert [(s.symbol, st) for s, st in out] == [("S1", "suppressed")]


def test_cap_is_per_user():
    out = apply_daily_cap([_sig("a", 1), _sig("b", 1)], sent_today={"a": 10}, summary_sent=set())
    assert [(s.user_id, s.symbol, st) for s, st in out] == [
        ("a", "*", "pending"), ("a", "S1", "suppressed"), ("b", "S1", "pending"),
    ]


def test_copy_names_the_move_and_the_band():
    title, body = alert_copy(Signal("u", "COMI", "d", "up", 58.0, 70.4))
    assert title == "COMI score up 12 points"
    assert body.startswith("58 → 70 · now ")


def test_copy_for_a_fall_and_for_the_summary():
    title, _ = alert_copy(Signal("u", "FWRY", "d", "down", 62.0, 50.0))
    assert title == "FWRY score down 12 points"
    title, body = alert_copy(Signal("u", "*", "d", "summary", 0.0, 0.0))
    assert title == "More stocks moved today"
    assert "Borsa Compass" in body


@pytest.mark.parametrize("direction", ["up", "down", "summary"])
@pytest.mark.parametrize("after", [5.0, 30.0, 50.0, 70.0, 95.0])
def test_no_copy_ever_gives_an_instruction(direction, after):
    title, body = alert_copy(Signal("u", "COMI", "d", direction, 50.0, after))
    for word in BANNED_WORDS:
        assert not re.search(rf"\b{word}\b", f"{title} {body}", re.IGNORECASE), word


@pytest.mark.parametrize("hour_utc, quiet", [
    (18, False),   # 21:00 Cairo (EEST, UTC+3)
    (19, True),    # 22:00 Cairo
    (2, True),     # 05:00 Cairo
    (5, False),    # 08:00 Cairo
])
def test_quiet_hours_are_cairo_local(hour_utc, quiet):
    assert in_quiet_hours(datetime(2026, 10, 11, hour_utc, 0, tzinfo=timezone.utc)) is quiet
