"""
The forecast surface must not make claims the measurement does not support.

This is a grep-and-shape test in the spirit of
`test_fixes.py::test_labels_describe_condition_not_action` and
`test_users_and_roles.py::test_no_sql_has_an_unescaped_percent` — it encodes a
rule that is easy to state, easy to breach by accident, and expensive to catch
by eye.

WHAT IT IS DEFENDING, AND WHY THE RULE EXISTS
---------------------------------------------
`ForecastCard.tsx` used to render three prices to two decimals
(`finalP5/finalP50/finalP95.toFixed(2)`) with the median tile coloured green
when it sat above spot. Two separate problems:

  * The median was `P0*exp(60*(mu - sigma^2/2))` — the trailing 400-day mean
    return compounded forward. A mechanical extrapolation. Colouring it by
    direction turned it into a price target, which is exactly the feature this
    project already deleted once (see "Removed: Max Buy Price" in CLAUDE.md).
  * A price to two decimals asserts a precision the app measured itself not to
    have: the p5-p95 cone that claimed 90% coverage was delivering 85.8%.

It also printed a hardcoded "68% of days" that was wrong by eleven points — the
+/-1-sigma band actually covers 79.0% of next-day moves on EGX. Coverage figures
must therefore come from the API payload, where the fitted constant lives, and
never be typed into JSX.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from app.core.forecast import (
    EGX_Z_CONE_60D,
    ONE_SIGMA_COVERAGE_PCT,
    OUTCOME_BAND_COVERAGES,
    expected_move,
    outcome_band,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_FE = os.path.join(_HERE, "..", "..", "egx-api-fe", "src", "app")
FORECAST_CARD = os.path.normpath(os.path.join(_FE, "components", "ForecastCard.tsx"))


def _strip_comments(src: str) -> str:
    """
    Drop /* ... */ and // ... so the checks below read CODE, not prose.

    This matters more than it looks. ForecastCard's header comment explains the
    removal by quoting the very thing that was removed — `finalP50.toFixed(2)`,
    the hardcoded "68%" — so a naive substring scan flags the documentation that
    exists to prevent the regression. Deleting the comment to satisfy the test
    would be exactly backwards.
    """
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", " ", src)


def _card_source() -> str | None:
    """
    The frontend lives in a SEPARATE git repository, so a backend-only checkout
    legitimately has no copy of it. Skip rather than fail in that case — a test
    that fails on a valid checkout gets deleted, and then it defends nothing.
    """
    if not os.path.exists(FORECAST_CARD):
        return None
    with open(FORECAST_CARD, encoding="utf-8") as f:
        return _strip_comments(f.read())


# ---------------------------------------------------------------------------
# Presentation rules
# ---------------------------------------------------------------------------

def test_forecast_card_shows_no_directional_colour():
    """
    The outcome band is symmetric in log space and drift-free: it makes NO
    directional claim. A gain/loss colour anywhere on the card reintroduces one,
    which is how the green median tile happened.
    """
    src = _card_source()
    if src is None:
        return
    offenders = [c for c in ("text-gain", "text-loss") if c in src]
    assert not offenders, (
        f"ForecastCard paints a direction again ({offenders}). The band is "
        f"deliberately symmetric and says nothing about which way price goes."
    )


def test_forecast_card_has_no_median_series():
    """The p50 was a compounded trailing mean return — a price target."""
    src = _card_source()
    if src is None:
        return
    banned = ("finalP50", "finalP5", "finalP95", "p50", "percentiles")
    found = [t for t in banned if t in src]
    assert not found, (
        f"ForecastCard references the removed median/percentile cone: {found}"
    )


def test_forecast_card_states_no_recommendation():
    """
    Recommendation-shaped language is both unsupported by the evidence and the
    thing Egypt's FRA regulates. Keep the surface descriptive.
    """
    src = _card_source()
    if src is None:
        return
    banned = ("price target", "we expect", "will reach", "should buy",
              "should sell", "win rate", "accuracy")
    lowered = src.lower()
    found = [p for p in banned if p in lowered]
    assert not found, f"ForecastCard uses recommendation language: {found}"


def test_coverage_percentages_are_not_hardcoded_in_the_card():
    """
    "68% of days" was hardcoded and wrong by eleven points. Every coverage
    figure must come from the payload, whose values are fitted by
    scripts/calibrate.py.
    """
    src = _card_source()
    if src is None:
        return
    # Only coverage-shaped figures. `width="100%"` is a layout attribute, not a
    # claim about the world, so the check targets the values a coverage band
    # could plausibly take rather than every "NN%" in the file.
    literals = re.findall(r"(?<!\d)(50|68|80|90|95)%", src)
    assert not literals, (
        f"ForecastCard hardcodes coverage percentage(s) {sorted(set(literals))}. "
        f"Render coverage_pct / outside_pct from the API instead — the fitted "
        f"values live in core/forecast.py and move when the cache is refitted."
    )


# ---------------------------------------------------------------------------
# Payload shape — the card cannot render what the API does not send
# ---------------------------------------------------------------------------

def _flat_returns(n: int = 400, sigma: float = 0.02) -> pd.Series:
    return pd.Series(np.random.default_rng(11).normal(0.0, sigma, n))


def test_expected_move_reports_measured_coverage_not_68():
    """
    Gaussian theory says 68%. EGX delivers 79.0 / 76.2 / 72.9 because price
    limits and flat illiquid sessions squash the centre of the distribution.
    """
    em = expected_move(_flat_returns())
    assert em is not None
    for horizon in ("daily", "weekly", "monthly"):
        key = f"{horizon}_coverage_pct"
        assert key in em, f"expected_move must report {key}"
        assert em[key] == ONE_SIGMA_COVERAGE_PCT[horizon]
        assert em[key] != 68, "the 68% Gaussian figure is measurably wrong here"


def test_outcome_band_is_deterministic():
    """
    The old cone drew from `np.random.default_rng(None)`, so the p5/p95 prices
    on screen changed every time the 15-minute cache missed. A band that moves
    without the data moving is not a measurement.
    """
    rets = _flat_returns()
    assert outcome_band(rets, 100.0) == outcome_band(rets, 100.0)


def test_outcome_band_is_symmetric_and_carries_no_drift():
    """
    Drift-free by construction: lo and hi must be reciprocal around spot in log
    space. A drift term would recreate the directional median.
    """
    band = outcome_band(_flat_returns(), 100.0)
    assert band is not None
    for level in band["bands"]:
        lo, hi = level["lo"][-1], level["hi"][-1]
        # Prices are rounded to the piastre before they leave the module, so
        # perfect log symmetry is not attainable; on a ~100 EGP stock that
        # rounding alone is ~1e-4 in log space. A real drift term compounded
        # over 60 days would be orders of magnitude larger than this bound.
        skew = abs(np.log(lo / 100.0) + np.log(hi / 100.0))
        assert skew < 1e-3, (
            f"outcome band is off-centre by {skew:.2e} in log space — a drift "
            f"term has crept back in"
        )


def test_outcome_band_states_its_complement():
    """
    Stating how often the band is wrong is the single best-replicated fix in the
    forecast-communication literature; readers otherwise misjudge which event
    the number refers to.
    """
    band = outcome_band(_flat_returns(), 100.0)
    assert band is not None
    for level in band["bands"]:
        assert level["outside_pct"] == 100 - level["coverage_pct"]


def test_egx_quantiles_are_not_gaussian():
    """
    Pins the correction itself. The 60-day cone needs |z| = 2.001 for 90%
    coverage, not the Gaussian 1.645 — using the textbook value delivered 85.8%.
    If someone "simplifies" this back to scipy's norm.ppf, this fails.
    """
    assert EGX_Z_CONE_60D[90] > 1.85, (
        "the 60-day 90% band has reverted to something near the Gaussian "
        "1.645, which measured only 85.8% coverage on EGX"
    )
    # And the bands the card draws must actually be fitted, not invented.
    for coverage in OUTCOME_BAND_COVERAGES:
        assert coverage in EGX_Z_CONE_60D, (
            f"band {coverage}% is rendered but has no fitted z"
        )
