"""
The risk surface: the one per-stock ranking this app's own evidence supports.

These tests pin the properties that make it defensible, not the arithmetic.
The numbers themselves come from scripts/calibrate.py and are re-derivable; what
must not drift is the SHAPE of the claim.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.core import risk_grade as rg


def _returns(n=400, sigma=0.02, seed=5):
    return pd.Series(np.random.default_rng(seed).normal(0.0, sigma, n))


def _prices(n=400, sigma=0.02, seed=5, start=100.0):
    r = _returns(n, sigma, seed)
    return pd.Series(start * (1 + r).cumprod())


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

def test_ewma_tracks_a_volatility_regime_change():
    """
    The point of EWMA over a flat window is that it REACTS. A calm stretch
    followed by a violent one must read as violent, not as the average of both.
    """
    calm = np.random.default_rng(1).normal(0, 0.005, 300)
    wild = np.random.default_rng(2).normal(0, 0.04, 100)
    series = pd.Series(np.concatenate([calm, wild]))

    ewma = rg.ewma_volatility(series)
    trailing_all = float(series.std())
    assert ewma > trailing_all, (
        "EWMA did not weight the recent violent regime above the full-sample "
        "standard deviation — the recursion or lambda is wrong"
    )


def test_ewma_variance_floor_survives_a_run_of_flat_sessions():
    """
    19% of EGX daily returns are EXACTLY zero. Without the floor a quiet
    stretch drives the conditional variance toward zero and every band built on
    it collapses to a point — a confident-looking forecast of no movement.
    """
    series = pd.Series(np.concatenate([
        np.random.default_rng(3).normal(0, 0.02, 300),
        np.zeros(80),
    ]))
    sigma = rg.ewma_volatility(series)
    assert sigma is not None and sigma > 0, "sigma collapsed on flat sessions"
    # The floor is a share of full-window variance, so it must land at or above
    # sqrt(share * var) rather than merely "not zero".
    floor = math.sqrt(rg.EWMA_VARIANCE_FLOOR_SHARE * float(np.var(series, ddof=1)))
    assert sigma >= floor * 0.999


def test_ewma_returns_none_rather_than_guessing_on_thin_data():
    assert rg.ewma_volatility(pd.Series([0.01, -0.02, 0.005])) is None


# ---------------------------------------------------------------------------
# Tradeability gate
# ---------------------------------------------------------------------------

def test_a_frozen_stock_is_not_tradeable():
    """
    MEGM has been frozen with zero volume since January 2022 and is the
    standing proof that this app can produce a confident number about a stock
    nobody can trade.
    """
    close = pd.Series([12.54] * 200)
    volume = pd.Series([0] * 200)
    assert rg.is_tradeable(close, volume) is False


def test_a_liquid_stock_is_tradeable():
    close = _prices()
    volume = pd.Series([500_000] * len(close))
    assert rg.is_tradeable(close, volume) is True


def test_thin_turnover_fails_the_gate_even_when_it_trades_daily():
    """Trading every day in tiny size is still untradeable at any real size."""
    close = pd.Series([1.0] * 200)
    volume = pd.Series([100] * 200)          # 100 EGP/day
    assert rg.is_tradeable(close, volume) is False


# ---------------------------------------------------------------------------
# Cross-sectional grading
# ---------------------------------------------------------------------------

def test_grading_is_monotonic_in_volatility():
    rows = [{"symbol": f"S{i}", "sigma_63_ann_pct": 10.0 + i} for i in range(50)]
    graded = rg.grade_universe(rows)
    by_symbol = {r["symbol"]: r for r in graded}
    assert by_symbol["S0"]["quintile"] == 1, "the calmest name is not quintile 1"
    assert by_symbol["S49"]["quintile"] == 5, "the wildest name is not quintile 5"
    ranks = [by_symbol[f"S{i}"]["pct_rank"] for i in range(50)]
    assert ranks == sorted(ranks), "percentile rank is not monotonic in sigma"


def test_a_thin_universe_is_left_ungraded_rather_than_ranked():
    """
    A percentile over four names is a fiction. The regime card already refuses
    to classify below 15 symbols for the same reason.
    """
    graded = rg.grade_universe(
        [{"symbol": "A", "sigma_63_ann_pct": 20.0},
         {"symbol": "B", "sigma_63_ann_pct": 40.0}])
    assert all(r["band"] is None for r in graded)


def test_every_quintile_carries_its_historical_record():
    rows = [{"symbol": f"S{i}", "sigma_63_ann_pct": float(i)} for i in range(50)]
    for r in rg.grade_universe(rows):
        hist = r["historical"]
        assert set(hist) == {"future_vol_ann_pct", "median_max_drawdown_pct",
                             "p90_max_drawdown_pct"}


def test_the_historical_table_is_monotonic():
    """
    The whole claim is that more past volatility means more future volatility
    and deeper drawdowns. If the shipped table is not monotonic, either the
    calibration broke or someone hand-edited it.
    """
    for field in ("future_vol_ann_pct", "median_max_drawdown_pct",
                  "p90_max_drawdown_pct"):
        values = [rg.RISK_QUINTILES[q][field] for q in sorted(rg.RISK_QUINTILES)]
        assert values == sorted(values), f"{field} is not monotonic: {values}"


def test_risk_grade_makes_no_return_claim():
    """
    Low volatility ranks positively against forward returns, but the realisable
    spread is weak (t=1.70) and the mean by quintile is flat-to-inverted. This
    surface must describe MOVEMENT and DRAWDOWN only — a return promise here is
    how Buy/Sell labels come back.
    """
    banned = ("return", "outperform", "gain", "profit", "beat")
    for quintile in rg.RISK_QUINTILES.values():
        for key in quintile:
            assert not any(b in key.lower() for b in banned), (
                f"a return claim entered the risk quintile table via '{key}'"
            )
    for band in rg.RISK_BANDS.values():
        assert not any(b in band["label"].lower() for b in banned)


# ---------------------------------------------------------------------------
# measure()
# ---------------------------------------------------------------------------

def test_measure_reports_both_sigmas_but_ranks_on_the_calibrated_one():
    """
    EWMA forecasts better, but the quintile table was fitted on the trailing
    63-day sigma. Swapping the ranking input without refitting would leave the
    historical mapping describing a different variable.
    """
    close = _prices()
    volume = pd.Series([500_000] * len(close))
    stats = rg.measure(close, volume)
    assert stats["sigma_63_ann_pct"] is not None
    assert stats["sigma_ewma_ann_pct"] is not None
    assert stats["sigma_63_ann_pct"] != stats["sigma_ewma_ann_pct"]


def test_measure_returns_none_on_short_history():
    close = _prices(n=20)
    assert rg.measure(close, pd.Series([1000] * 20)) is None


@pytest.mark.parametrize("sigma,expected_more", [(0.01, False), (0.05, True)])
def test_annualized_scales_with_sigma(sigma, expected_more):
    base = rg.annualized(0.02)
    assert (rg.annualized(sigma) > base) is expected_more


# ---------------------------------------------------------------------------
# The scheduled snapshot's cursor arithmetic
#
# This is what an external scheduler actually drives, so it is tested as a pure
# function — tests/ has no Postgres fixture, which is why cron.py keeps the
# slicing independent of its writes.
# ---------------------------------------------------------------------------

from app.routers.cron import is_isin, plan_chunk  # noqa: E402


def test_chunking_walks_the_whole_universe_exactly_once():
    universe = [f"S{i}" for i in range(53)]
    seen, cursor, calls = [], 0, 0
    while True:
        plan = plan_chunk(universe, cursor, 20)
        seen.extend(plan["symbols"])
        calls += 1
        cursor = plan["cursor"]
        if not plan["remaining"]:
            break
        assert calls < 20, "cursor is not advancing — the scheduler would loop"
    assert seen == universe, "chunking dropped or duplicated symbols"
    assert calls == 3


def test_cursor_wraps_to_zero_when_the_universe_is_exhausted():
    """A scheduler replaying the returned cursor must loop, not stall."""
    plan = plan_chunk([f"S{i}" for i in range(10)], 0, 20)
    assert plan["remaining"] == 0
    assert plan["cursor"] == 0


def test_a_stale_cursor_past_the_end_restarts_instead_of_erroring():
    """
    The universe shrinks when a stock delists. A scheduler still holding
    yesterday's cursor should quietly restart rather than begin alerting.
    """
    plan = plan_chunk([f"S{i}" for i in range(5)], 999, 20)
    assert plan["symbols"] == [f"S{i}" for i in range(5)]


def test_isin_rows_are_excluded_but_egsa_survives():
    """
    EGSA is a real four-character EGX ticker. A startswith("EGS") rule — the
    obvious implementation — silently deletes it from the universe.
    """
    assert is_isin("EGS30AJ1C016-EGP") is True
    assert is_isin("EGS370O1C013") is True
    assert is_isin("EGSA") is False
    assert is_isin("COMI") is False


# ---------------------------------------------------------------------------
# Stalest-first selection
#
# cron-job.org cannot read a response body and feed a cursor back, so the
# production mode is one fixed URL on an interval and the server choosing what
# to refresh. These are pure -- tests/ has no Postgres fixture by design.
# ---------------------------------------------------------------------------

from app.routers.cron import select_stalest  # noqa: E402


def test_never_measured_symbols_go_first():
    """An absent symbol has no timestamp at all and must outrank every stale one."""
    universe = ["A", "B", "C", "D"]
    measured = {"A": "2026-09-01T00:00:00", "B": "2026-08-01T00:00:00"}
    assert select_stalest(universe, measured, 2) == ["C", "D"]


def test_among_measured_symbols_the_oldest_wins():
    universe = ["A", "B", "C"]
    measured = {"A": "2026-09-02T00:00:00",
                "B": "2026-08-01T00:00:00",
                "C": "2026-09-01T00:00:00"}
    assert select_stalest(universe, measured, 2) == ["B", "C"]


def test_selection_is_deterministic_when_timestamps_tie():
    """
    A whole universe refreshed in one pass shares a timestamp. Without a
    tiebreak the order would depend on dict iteration and the job could
    re-measure the same names forever while never reaching others.
    """
    universe = ["D", "C", "B", "A"]
    measured = {s: "2026-09-02T00:00:00" for s in universe}
    assert select_stalest(universe, measured, 2) == ["A", "B"]


def test_a_failing_symbol_is_retried_on_the_very_next_call():
    """
    This is the property a cursor could not provide. A cursor must advance past
    a failed chunk or one broken symbol pins it forever -- which means the
    failure waits a full pass. Staleness retries it immediately.
    """
    universe = ["A", "B", "C"]
    measured = {}
    first = select_stalest(universe, measured, 2)
    assert first == ["A", "B"]
    # "A" succeeded, "B" failed so it was never written.
    measured["A"] = "2026-09-02T12:00:00"
    assert "B" in select_stalest(universe, measured, 2)


def test_selection_never_exceeds_the_universe():
    assert len(select_stalest(["A", "B"], {}, 50)) == 2


def test_unmeasurable_symbols_must_not_starve_the_universe():
    """
    THE bug that stalest-first selection can produce, caught in simulation.

    A symbol with no row is maximally stale, so if a failed or unmeasurable
    symbol wrote nothing it would be re-picked on every single call. The EGX has
    roughly 34 effectively dead names against a 20-symbol batch, so they would
    fill every batch forever and NOT ONE live symbol would ever be measured.

    The endpoint therefore records the attempt even when there is nothing to
    measure (null sigma, tradeable=False). This test pins that: with the attempt
    recorded the universe converges; without it, it never starts.
    """
    universe = [f"S{i:03d}" for i in range(166)]
    dead = set(universe[:34])

    def sweep(record_failures: bool, calls: int = 20) -> int:
        measured: dict = {}
        for call in range(1, calls + 1):
            for symbol in select_stalest(universe, measured, 20):
                if symbol not in dead or record_failures:
                    measured[symbol] = f"2026-09-02T{call:02d}:00:00"
        return sum(1 for s in universe if s not in dead and s in measured)

    live = len(universe) - len(dead)
    assert sweep(record_failures=False) == 0, (
        "the simulation no longer reproduces the starvation bug, so this test "
        "is not guarding anything"
    )
    assert sweep(record_failures=True) == live, (
        "recording the attempt no longer lets the universe converge"
    )


def test_the_endpoint_records_every_attempt():
    """
    The behavioural guard for the above: every path through the symbol loop
    must reach an upsert. If a `continue` is ever added that skips one, the
    starvation bug returns silently.
    """
    import inspect

    from app.routers import cron as cron_mod

    src = inspect.getsource(cron_mod.risk_snapshot)
    body = src.split("for symbol in slice_:", 1)[1]
    # Each early exit in the loop must be immediately preceded by an upsert.
    for chunk in body.split("continue")[:-1]:
        assert "_upsert(" in chunk, (
            "a path through the measurement loop skips the upsert — an "
            "unmeasurable symbol would stay maximally stale and starve the "
            "universe under stalest-first selection"
        )


# ---------------------------------------------------------------------------
# Feed refusals
#
# 84 of the 166 symbols in the ticker file have NEVER returned data from
# tvDatafeed, and the failures are PERSISTENT — the identical set fails on every
# pass. Each refusal costs ~6s against a 15s budget, so without demotion half
# the budget goes to symbols that will never work and the healthy half never
# finishes. This is what timed the production cron out.
# ---------------------------------------------------------------------------

from app.routers.cron import FAILURE_DEMOTION_THRESHOLD  # noqa: E402


def test_persistently_refused_symbols_stop_eating_the_budget():
    universe = [f"S{i}" for i in range(6)]
    measured = {s: "2026-09-01T00:00:00" for s in universe}
    failures = {"S0": FAILURE_DEMOTION_THRESHOLD, "S1": FAILURE_DEMOTION_THRESHOLD}

    picked = select_stalest(universe, measured, 4, failures)
    assert "S0" not in picked and "S1" not in picked, (
        "symbols the feed keeps refusing are still being picked ahead of "
        "healthy ones — this is what timed the cron out in production"
    )


def test_refused_symbols_are_demoted_not_excluded():
    """
    A blocklist rots. A symbol that starts working must be able to come back,
    so the ordering still reaches it once everything healthy is fresh.
    """
    universe = [f"S{i}" for i in range(4)]
    measured = {s: "2026-09-01T00:00:00" for s in universe}
    failures = {"S0": 99}
    assert "S0" in select_stalest(universe, measured, 4, failures)


def test_a_symbol_below_the_threshold_is_not_demoted():
    """One bad night is not a dead symbol."""
    universe = ["A", "B"]
    measured = {"A": "2020-01-01", "B": "2026-09-01"}
    failures = {"A": FAILURE_DEMOTION_THRESHOLD - 1}
    assert select_stalest(universe, measured, 1, failures) == ["A"]


def test_selection_without_failure_data_is_unchanged():
    """The argument is optional, so nothing that omits it changes behaviour."""
    universe = ["A", "B", "C"]
    measured = {"A": "2026-01-01", "B": "2020-01-01"}
    assert select_stalest(universe, measured, 2) == \
        select_stalest(universe, measured, 2, {})


def test_the_handler_bounds_its_own_runtime():
    """
    The cron must return inside the scheduler's timeout no matter how slow the
    feed is. Stopping early is safe because whatever went unmeasured is still
    the stalest thing in the table and is picked up first next call.
    """
    import inspect

    from app.routers import cron as cron_mod

    src = inspect.getsource(cron_mod.risk_snapshot)
    assert "DEADLINE_SECONDS" in src, "the wall-clock deadline was removed"
    assert cron_mod.DEADLINE_SECONDS <= 25, (
        "the deadline no longer leaves margin inside Vercel's 30s limit"
    )
    assert cron_mod.DEFAULT_CHUNK * 6 > cron_mod.DEADLINE_SECONDS, (
        "chunk and deadline are inconsistent: at ~6s per refused symbol this "
        "chunk cannot finish, so the deadline is doing the work and the chunk "
        "size is misleading"
    )
