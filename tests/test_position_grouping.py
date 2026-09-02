"""
Buying one symbol twice is ONE position held in two lots.

The portfolio shows a single card per symbol, so the advice beside it has to
speak once too — and the two halves of that are different problems. Technical
signals are identical per lot and just need collapsing; cost-basis signals are
NOT identical, and collapsing them would report one lot's loss as the
position's. These tests pin both, and pin that a single-lot portfolio — the
common case — is completely unaffected.

Run from egx-api-be:  python -m pytest tests/test_position_grouping.py -v
"""

import os
from datetime import date

import pytest

# BEFORE the router import, which pulls in app.core.auth — and that module
# snapshots AUTH_SECRET into a constant at import time. test_auth_gate.py sets
# the same default at ITS module scope, so whichever file imports auth first
# decides whether the secret exists. Importing a router here without this line
# makes three tests over there fail, but only when this file is collected
# first. Same literal as test_auth_gate.py, so the order cannot matter.
os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")

from app.core.constants import BIG_LOSS_PCT, PROFIT_TARGET_PCT  # noqa: E402
from app.routers.portfolio_analysis import (  # noqa: E402
    build_position_signals,
    dedupe_symbol_signals,
)

TODAY = date(2026, 9, 1)
RFR = 19.0


def _position(symbol="COMI", quantity=300, invested=12520.0, current_value=13000.0,
              current_price=45.0, earliest_buy_date="2026-01-12",
              stop_loss=None, target_price=None):
    return {
        "symbol": symbol, "quantity": quantity, "invested": invested,
        "current_value": current_value, "current_price": current_price,
        "earliest_buy_date": earliest_buy_date,
        "stop_loss": stop_loss, "target_price": target_price,
    }


def _types(signals):
    return [s["type"] for s in signals]


# ---- deduping the technical signals ----

def test_an_identical_signal_from_a_second_lot_is_dropped():
    rsi = {"type": "rsi_oversold", "severity": "opportunity", "symbol": "COMI",
           "message": "COMI RSI is at 28 (oversold <30).", "explanation": "",
           "learn_concept": "rsi"}
    assert dedupe_symbol_signals([rsi, dict(rsi)]) == [rsi]


def test_the_same_signal_on_a_different_symbol_survives():
    out = dedupe_symbol_signals([
        {"type": "rsi_oversold", "symbol": "COMI"},
        {"type": "rsi_oversold", "symbol": "SWDY"},
    ])
    assert len(out) == 2


def test_different_signals_on_one_symbol_all_survive():
    out = dedupe_symbol_signals([
        {"type": "golden_cross", "symbol": "COMI"},
        {"type": "rsi_oversold", "symbol": "COMI"},
        {"type": "near_support", "symbol": "COMI"},
    ])
    assert len(out) == 3


def test_the_first_occurrence_is_the_one_kept():
    out = dedupe_symbol_signals([
        {"type": "mfi_extreme", "symbol": "COMI", "severity": "opportunity"},
        {"type": "mfi_extreme", "symbol": "COMI", "severity": "warning"},
    ])
    assert out[0]["severity"] == "opportunity"


def test_portfolio_wide_signals_carrying_no_symbol_are_not_collapsed_together():
    # sector_concentration has symbol=None on every entry. Keyed on
    # (type, symbol) alone, three sector warnings would become one.
    out = dedupe_symbol_signals([
        {"type": "sector_concentration", "symbol": None, "message": "Banks"},
        {"type": "sector_concentration", "symbol": None, "message": "Real Estate"},
    ])
    assert len(out) == 2, (
        "dedupe must run over per-holding signals only, before the "
        "portfolio-level ones are appended"
    )


# ---- cost-basis signals, computed once per position ----

def test_two_lots_produce_one_big_loss_signal_for_the_whole_position():
    # 200 @ 41 and 100 @ 45.20 = 12,720 invested, now worth 9,000.
    out = build_position_signals(
        [_position(invested=12720.0, current_value=9000.0, current_price=30.0)],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    losses = [s for s in out if s["type"] == "big_loss"]
    assert len(losses) == 1
    assert "29.2%" in losses[0]["message"]


def test_a_losing_lot_inside_a_winning_position_raises_no_loss_signal():
    # The failure this exists to prevent: the June lot is down 20%, but the
    # position is up, and the card says up. One lot's number must not be
    # reported as the position's.
    out = build_position_signals(
        [_position(invested=10000.0, current_value=13000.0)],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert "big_loss" not in _types(out)
    assert "profit_taking" in _types(out)


def test_the_percentage_is_cost_weighted_not_an_average_of_the_lots():
    # 1,000 invested up 50%, 10,000 invested down 10% -> the position is DOWN
    # 4.5%. Averaging the two lot percentages says up 20%.
    out = build_position_signals(
        [_position(invested=11000.0, current_value=10500.0)],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert "profit_taking" not in _types(out)


def test_the_stop_that_triggers_first_is_the_one_reported():
    # Two lots, two stops. Price 45 sits above both; the higher stop is the
    # one price reaches first, so it is the one worth warning about.
    out = build_position_signals(
        [_position(current_price=45.0, stop_loss=43.0)],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    stops = [s for s in out if s["type"] == "stop_loss"]
    assert len(stops) == 1
    assert "43.00" in stops[0]["message"]


def test_a_breached_stop_reads_as_breached_not_as_distance():
    out = build_position_signals(
        [_position(current_price=40.0, stop_loss=43.0)],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert _types(out).count("stop_breached") == 1
    assert "stop_loss" not in _types(out)


def test_the_target_reached_first_is_the_one_reported():
    out = build_position_signals(
        [_position(current_price=45.0, target_price=48.0)],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    targets = [s for s in out if s["type"] == "target_reached"]
    assert len(targets) == 1 and "48.00" in targets[0]["message"]


def test_days_held_runs_from_the_earliest_lot():
    # Bought Jan 2025, up 8% — under the T-bill once annualized over ~20
    # months. Dated from a later top-up it would still be over 90 days but
    # would annualize to a different, flattering figure.
    out = build_position_signals(
        [_position(invested=10000.0, current_value=10800.0,
                   earliest_buy_date="2025-01-02")],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert "cash_underperformer" in _types(out)


def test_a_position_held_under_ninety_days_is_not_graded_against_cash():
    out = build_position_signals(
        [_position(invested=10000.0, current_value=10100.0,
                   earliest_buy_date="2026-08-01")],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert "cash_underperformer" not in _types(out)


def test_a_position_with_no_price_is_skipped_rather_than_divided_by_zero():
    out = build_position_signals(
        [_position(current_price=0.0), _position(invested=0.0)],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert out == []


def test_each_symbol_gets_its_own_signals():
    out = build_position_signals(
        [
            _position(symbol="COMI", invested=10000.0, current_value=13000.0),
            _position(symbol="SWDY", invested=10000.0, current_value=7000.0),
        ],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert {(s["type"], s["symbol"]) for s in out} >= {
        ("profit_taking", "COMI"), ("big_loss", "SWDY"),
    }


# ---- the common case is untouched ----

@pytest.mark.parametrize("current_value,expected", [
    (10000.0 * (1 + (BIG_LOSS_PCT - 5) / 100), "big_loss"),
    (10000.0 * (1 + (PROFIT_TARGET_PCT + 5) / 100), "profit_taking"),
])
def test_a_one_lot_position_emits_exactly_what_the_per_lot_code_did(current_value, expected):
    # Held 31 days, so the cash comparison (>90 days) stays out of the way and
    # the assertion can be exact.
    out = build_position_signals(
        [_position(quantity=100, invested=10000.0, current_value=current_value,
                   earliest_buy_date="2026-08-01")],
        risk_free_rate_pct=RFR, today=TODAY,
    )
    assert _types(out) == [expected]
