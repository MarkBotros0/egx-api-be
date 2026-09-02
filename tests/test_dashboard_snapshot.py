"""
The dashboard's pre-scored snapshot, and the one rule it must never break.

The grid is served from `risk_snapshot` rather than from live fetches, and what
is stored is the EIGHT CATEGORY SCORES, not a finished composite. That choice is
what keeps *One Score Per Stock* true: the read path applies the caller's own
weights and today's macro regime and must land on exactly the number the stock
detail page computes from the same inputs.

If that equality ever breaks, a user taps a card reading 66 and lands on a page
reading 45 — which is a divergence this project has already shipped once, and is
why `extras_builder.py` exists. These tests are the guard.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.card_snapshot import (  # noqa: E402
    CATEGORY_COLUMNS,
    SPARKLINE_BARS,
    to_card,
)
from app.core.composite import (  # noqa: E402
    CATEGORY_ORDER,
    DEFAULT_WEIGHTS,
    PRESETS,
    blend_categories,
    compute_composite,
    score_categories,
)
from app.core.constants import FAILURE_DEMOTION_THRESHOLD  # noqa: E402


# A spread of realistic category scores, including the None that means "this
# category was not scorable on this stock" — the case where renormalization
# actually does something and where a second implementation would drift.
_CATEGORY_CASES = [
    {"trend": 72.0, "momentum": 48.0, "volume": 55.0, "volatility": 61.0,
     "divergence": 50.0, "quality": 64.0, "risk_adjusted": 39.0,
     "relative_strength": 70.0},
    # Risk-adjusted excluded: the <120-bar history gate, which is common.
    {"trend": 30.0, "momentum": 22.0, "volume": 41.0, "volatility": 35.0,
     "divergence": 50.0, "quality": 28.0, "risk_adjusted": None,
     "relative_strength": 18.0},
    # Only two categories scorable — a NILEX name with almost no history.
    {"trend": None, "momentum": 55.0, "volume": None, "volatility": 44.0,
     "divergence": None, "quality": None, "risk_adjusted": None,
     "relative_strength": None},
    {"trend": 0.0, "momentum": 0.0, "volume": 0.0, "volatility": 0.0,
     "divergence": 0.0, "quality": 0.0, "risk_adjusted": 0.0,
     "relative_strength": 0.0},
    {"trend": 100.0, "momentum": 100.0, "volume": 100.0, "volatility": 100.0,
     "divergence": 100.0, "quality": 100.0, "risk_adjusted": 100.0,
     "relative_strength": 100.0},
]

_WEIGHT_SETS = [DEFAULT_WEIGHTS] + list(PRESETS.values()) + [
    # A hand-set slider arrangement that is none of the presets, including a
    # zeroed category — the read path must handle a user who has turned one off.
    {"trend": 50, "momentum": 0, "volume": 5, "volatility": 5,
     "divergence": 0, "quality": 20, "risk_adjusted": 10,
     "relative_strength": 10},
]

_MACROS = [
    None,
    {"egx30": {"trend": "bullish"}},
    {"egx30": {"trend": "sideways"}},
    {"egx30": {"trend": "bearish"}},
]


# ---------------------------------------------------------------------------
# The equality that the whole design rests on
# ---------------------------------------------------------------------------

def test_snapshot_blend_matches_live_scoring_exactly():
    """
    Re-blending stored category scores == scoring live, for every combination
    of weights and macro regime.

    This is the card-vs-detail-page guarantee stated as arithmetic. The
    snapshot stores what `score_categories` produced; the card calls
    `blend_categories` on it; the detail page calls `compute_composite`, which
    calls the same `blend_categories`. Exact equality, not approximate — a
    tolerance here would be an admission that two implementations exist.
    """
    for cats in _CATEGORY_CASES:
        for weights in _WEIGHT_SETS:
            for macro in _MACROS:
                from_snapshot = blend_categories(cats, weights, macro)
                # What compute_composite would produce from the same category
                # scores, exercised through its real assembly path.
                live = _compose_from_categories(cats, weights, macro)
                assert from_snapshot["score"] == live["score"], (
                    f"card and detail page disagree: "
                    f"{from_snapshot['score']} vs {live['score']} "
                    f"(cats={cats}, weights={weights}, macro={macro})"
                )
                assert from_snapshot["signal"] == live["signal"]


def _compose_from_categories(cats, weights, macro):
    """
    Drive `compute_composite`'s own blending with fixed category scores by
    monkeypatching the scorer it calls, so this compares the REAL assembly path
    rather than a reimplementation of it.
    """
    import app.core.composite as composite_module

    original = composite_module.score_categories
    composite_module.score_categories = lambda _i, _e: {
        name: (cats.get(name), []) for name in CATEGORY_ORDER
    }
    try:
        return composite_module.compute_composite({}, {}, weights, macro)
    finally:
        composite_module.score_categories = original


def test_stored_categories_are_weight_and_macro_free():
    """
    `score_categories` must not accept weights or a macro regime.

    If it ever did, the snapshot would be freezing one user's sliders and one
    day's regime into a stored number, and re-blending it per caller would be a
    lie. The separation is the entire reason a nightly job can serve every
    user's own score.
    """
    params = set(inspect.signature(score_categories).parameters)
    assert not (params & {"weights", "macro"}), (
        f"score_categories took {params & {'weights', 'macro'}} — the stored "
        "snapshot would no longer be user-neutral"
    )


def test_only_one_place_spells_the_weighting():
    """
    No module outside composite.py may re-implement the renormalize-and-weight
    loop. Mirrors test_regime_reader_and_batch_writer_share_one_cache_key: the
    failure mode is silent divergence, so it is checked by enumeration.
    """
    root = os.path.join(os.path.dirname(__file__), "..", "app")
    offenders = []
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            if os.path.normpath(path).endswith(os.path.join("core", "composite.py")):
                continue
            source = open(path, encoding="utf-8").read()
            tree = ast.parse(source)
            # The tell-tale: dividing by a sum of the available weights.
            for node in ast.walk(tree):
                if not isinstance(node, ast.Name):
                    continue
                if node.id in ("available_weight_sum", "effective_weight"):
                    offenders.append(f"{path}:{node.lineno} ({node.id})")
    assert not offenders, (
        "the composite weighting is spelled outside composite.py — a card and "
        f"its detail page can now disagree: {offenders}"
    )


# ---------------------------------------------------------------------------
# The read model
# ---------------------------------------------------------------------------

def _row(**overrides):
    row = {
        "symbol": "COMI", "measured_at": "2026-09-02T15:00:00+00:00",
        "scored_at": "2026-09-02T15:00:00+00:00",
        "last_price": 100.0, "prev_close": 96.0,
        "sparkline_json": json.dumps([90.0, 95.0, 100.0]),
        "tradeable": True, "sigma_63_ann_pct": 33.0, "consecutive_failures": 0,
    }
    row.update({name: 55.0 for name in CATEGORY_ORDER})
    row.update(overrides)
    return row


def test_demoted_symbol_is_reported_unavailable_not_blank():
    """
    A symbol the feed keeps refusing must come back flagged, not empty.

    84 of the 166 symbols in the ticker file have NEVER returned data. They
    used to render an indefinite "--" that was indistinguishable from a card
    still loading, so the reader reloaded a page that was never going to fill.
    `available: false` is what lets the grid say "no price feed" instead.
    """
    card = to_card(
        _row(consecutive_failures=FAILURE_DEMOTION_THRESHOLD),
        DEFAULT_WEIGHTS, None,
    )
    assert card["available"] is False

    healthy = to_card(
        _row(consecutive_failures=FAILURE_DEMOTION_THRESHOLD - 1),
        DEFAULT_WEIGHTS, None,
    )
    assert healthy["available"] is True


def test_a_symbol_with_no_categories_still_returns_its_price():
    """
    Too little history to score is NOT the same as no data.

    Such a stock has a price, a sparkline and a place on the grid; refusing to
    card it would leave it permanently blank for a reason the reader cannot
    see. The score is null and the card is still available.
    """
    row = _row(**{name: None for name in CATEGORY_ORDER})
    card = to_card(row, DEFAULT_WEIGHTS, None)
    assert card["score"] is None
    assert card["price"] == 100.0
    assert card["available"] is True


def test_change_is_derived_from_the_stored_previous_close():
    card = to_card(_row(), DEFAULT_WEIGHTS, None)
    assert card["change"] == pytest.approx(4.0)
    assert card["change_pct"] == pytest.approx(4.0 / 96.0 * 100)


def test_a_missing_previous_close_yields_no_change_not_zero():
    """
    Zero would render as "0.00 (0.00%)" in neutral green — a claim the stock
    was flat, which is not what a missing previous close means.
    """
    card = to_card(_row(prev_close=None), DEFAULT_WEIGHTS, None)
    assert card["change"] is None
    assert card["change_pct"] is None


def test_a_corrupt_sparkline_degrades_to_empty():
    """The grid must paint even if one row's JSON is unparseable."""
    assert to_card(_row(sparkline_json="not json"), DEFAULT_WEIGHTS, None)["sparkline"] == []
    assert to_card(_row(sparkline_json=None), DEFAULT_WEIGHTS, None)["sparkline"] == []


def test_the_card_carries_its_own_freshness():
    """
    A snapshot row is post-close, so during trading hours its price is stale.
    The card must carry the timestamp that says so; a grid that cannot state
    its own age presents last night's close as this minute's.
    """
    card = to_card(_row(), DEFAULT_WEIGHTS, None)
    assert card["measured_at"]
    assert card["scored_at"]


# ---------------------------------------------------------------------------
# Schema / writer agreement
# ---------------------------------------------------------------------------

def test_every_category_has_a_column_and_init_db_creates_it():
    """
    The DDL derives its column list from CATEGORY_COLUMNS, which derives from
    CATEGORY_ORDER. Adding a ninth category must therefore add its column
    automatically — a hand-typed list is how one of three places gets forgotten
    and a category silently stops being stored.
    """
    assert len(CATEGORY_COLUMNS) == len(CATEGORY_ORDER)
    assert CATEGORY_COLUMNS == tuple(f"cat_{n}" for n in CATEGORY_ORDER)

    db_source = open(
        os.path.join(os.path.dirname(__file__), "..", "app", "core", "db.py"),
        encoding="utf-8",
    ).read()
    assert "CATEGORY_COLUMNS" in db_source, (
        "init_db no longer derives the snapshot's category columns from "
        "CATEGORY_ORDER; a ninth category would not get a column"
    )


def test_the_cron_writes_every_category_column():
    """The writer must iterate the shared list, not a copy of it."""
    source = open(
        os.path.join(os.path.dirname(__file__), "..", "app", "routers", "cron.py"),
        encoding="utf-8",
    ).read()
    assert "CATEGORY_COLUMNS" in source
    assert "score_categories" in source, (
        "the snapshot cron no longer scores; the dashboard would serve "
        "permanently null scores"
    )


def test_the_cron_does_not_hand_roll_extras():
    """
    Same rule the routers already live under. The snapshot must score with
    build_composite_extras or its numbers stop matching every other page.
    """
    source = open(
        os.path.join(os.path.dirname(__file__), "..", "app", "routers", "cron.py"),
        encoding="utf-8",
    ).read()
    assert "build_composite_extras" in source


def test_a_failed_fetch_does_not_blank_an_existing_score():
    """
    NEVER WIPE ON FAILURE, applied to the card columns.

    A symbol the feed refused today has not changed price, so last night's
    score is still the last thing anyone knew. Blanking it would let one
    transient refusal empty a card — the exact failure this surface was rebuilt
    to remove. The writer omits the card columns entirely when it has no card,
    so they are left alone rather than set to NULL.
    """
    source = open(
        os.path.join(os.path.dirname(__file__), "..", "app", "routers", "cron.py"),
        encoding="utf-8",
    ).read()
    tree = ast.parse(source)

    upsert = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_upsert":
            upsert = node
    assert upsert is not None, "_upsert disappeared from the snapshot cron"

    # The card columns must be assembled conditionally, guarded on `card`.
    guarded = any(
        isinstance(n, ast.If) and "card" in ast.dump(n.test)
        for n in ast.walk(upsert)
    )
    assert guarded, (
        "_upsert no longer guards the card columns on having a card — a failed "
        "fetch would overwrite a good score with NULL"
    )


def test_sparkline_length_matches_the_live_batch_path():
    """
    A card that upgrades from snapshot to live data must not change shape. The
    live path returns the last 30 closes; so must the snapshot.
    """
    analysis_source = open(
        os.path.join(os.path.dirname(__file__), "..", "app", "routers", "analysis.py"),
        encoding="utf-8",
    ).read()
    assert f"iloc[-{SPARKLINE_BARS}:]" in analysis_source, (
        "the live batch path and the snapshot draw different-length sparklines"
    )
