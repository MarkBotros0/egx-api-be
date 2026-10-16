"""
EG output must not move when its numbers move into the bundle. Compared
against tests/golden/eg_outputs.json, captured from the pre-bundle code.
"""
import json
import os

os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")
os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.calibration import for_market  # noqa: E402
from app.core.markets import EG  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "golden", "eg_outputs.json"), encoding="utf-8") as _f:
    GOLDEN = json.load(_f)
EG_CAL = for_market("EG")


def _j(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def _rets(n):
    return pd.Series(np.random.default_rng(11).normal(0.0, 0.02, n))


def test_calibration_route_is_unchanged_for_eg():
    from app.routers.calibration import get_calibration
    assert _j(get_calibration(user=None, market=EG)) == GOLDEN["calibration_route"]


def test_regime_readings_are_unchanged_for_eg():
    from app.core.regime import classify_regime
    for mean, want in GOLDEN["regime"].items():
        assert _j(classify_regime(float(mean), 40, EG_CAL.regime)) == want, mean
    assert _j(classify_regime(52.0, 3, EG_CAL.regime)) == GOLDEN["regime_thin"]


def test_risk_payload_and_grades_are_unchanged_for_eg():
    from app.core.risk_grade import grade_universe
    assert _j(EG_CAL.risk.payload()) == GOLDEN["risk_payload"]
    rows = [{"symbol": f"S{i}", "sigma_63_ann_pct": 10.0 + 5 * i} for i in range(10)]
    assert _j(grade_universe(rows, EG_CAL.risk.quintiles)) == GOLDEN["graded"]


def test_forecast_is_unchanged_for_eg_except_the_method_name():
    from app.core.forecast import expected_move, outcome_band
    assert _j(expected_move(_rets(400), EG_CAL)) == GOLDEN["expected_move_long"]
    assert _j(expected_move(_rets(30), EG_CAL)) == GOLDEN["expected_move_short"]
    want = dict(GOLDEN["outcome_band"], method="calibrated_quantile")
    assert _j(outcome_band(_rets(400), 100.0, EG_CAL)) == want


def test_forecast_follows_the_bundle_it_is_given():
    """A different market's bundle changes the band: nothing is read from a global."""
    import dataclasses
    from app.core.forecast import outcome_band
    wide = dataclasses.replace(
        EG_CAL, forecast=dataclasses.replace(EG_CAL.forecast, z_cone={50: 1.0, 80: 2.0, 90: 3.0, 95: 4.0}))
    narrow_hi = outcome_band(_rets(400), 100.0, EG_CAL)["endpoint"]["hi"]
    wide_hi = outcome_band(_rets(400), 100.0, wide)["endpoint"]["hi"]
    assert wide_hi > narrow_hi


import dataclasses  # noqa: E402
import math  # noqa: E402


def test_score_quality_is_unchanged_for_eg():
    from app.core.composite import score_quality
    for pe, dy, calibrated, score, reasons in GOLDEN["score_quality"]:
        want = [score, reasons]
        got = score_quality(None, None, None, pe_ratio=pe, dividend_yield=dy,
                            calibrated=calibrated)
        assert _j(list(got)) == want, (pe, dy, calibrated)
        got = score_quality(None, None, None, pe_ratio=pe, dividend_yield=dy,
                            calibrated=calibrated, fundamentals=EG_CAL.fundamentals)
        assert _j(list(got)) == want, (pe, dy, calibrated)


def test_liquidity_is_unchanged_for_eg():
    from app.core.indicators import liquidity_score
    # The old global table carried a "US" row (NILEX floors); it now lives in
    # the US bundle. Replicated here so the golden stays independent of us.py.
    old_us = dataclasses.replace(EG_CAL, liquidity_tiers={"US": (5_000, 20_000)},
                                 liquidity_default_tier="US")
    for tier, volume, want in GOLDEN["liquidity"]:
        cal = old_us if tier == "US" else None
        got = liquidity_score(pd.Series([float(volume)] * 30), tier, cal=cal)
        assert _j(got) == want, (tier, volume)


def test_band_boundaries_are_half_open():
    """A value AT a boundary lands in the band above, exactly as before."""
    from app.core.composite import score_quality
    f = dataclasses.replace(EG_CAL.fundamentals, label="ZZ",
                            pe_bands=((9.0, 3), (18.0, 12), (27.0, 4), (36.0, -2),
                                      (45.0, -8), (math.inf, -14)))
    assert score_quality(None, None, None, pe_ratio=8.99, fundamentals=f)[0] == 53
    assert score_quality(None, None, None, pe_ratio=9.0, fundamentals=f)[0] == 62
    assert score_quality(None, None, None, pe_ratio=45.0, fundamentals=f)[0] == 36
    reasons = score_quality(None, None, None, pe_ratio=10.0, fundamentals=f)[1]
    assert reasons == ["Cheap versus the ZZ median of ~12 (P/E 10.0)"]


def test_score_categories_reads_the_bundle_of_the_extras_market(monkeypatch):
    from app.core import calibration as cal_pkg
    from app.core.composite import score_categories
    shifted = dataclasses.replace(
        cal_pkg.for_market("US"),
        fundamentals=dataclasses.replace(
            EG_CAL.fundamentals, label="US",
            pe_bands=((1.0, 3), (2.0, 12), (3.0, 4), (4.0, -2), (5.0, -8), (math.inf, -14))))
    monkeypatch.setitem(cal_pkg._BUNDLES, "US", shifted)
    extras = {"pe_ratio": 5.0, "score_calibrated": True}
    eg = score_categories({}, {**extras, "market": "EG"})["quality"]
    us = score_categories({}, {**extras, "market": "US"})["quality"]
    assert eg[0] == 62 and us[0] == 36
    assert eg[1] == ["Cheap versus the EGX median of ~12 (P/E 5.0)"]


def test_extras_carry_the_market_and_its_liquidity_tiers(monkeypatch):
    from app.core import calibration as cal_pkg
    from app.core.extras_builder import build_composite_extras
    from app.core.indicators import compute_all
    monkeypatch.setitem(cal_pkg._BUNDLES, "US", dataclasses.replace(
        cal_pkg.for_market("US"), liquidity_tiers={"US": (1, 2)}, liquidity_default_tier="US"))
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = pd.Series(np.linspace(10, 20, 300), index=idx)
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 10.0}, index=idx)
    built = build_composite_extras(df, compute_all(df), index_membership="US",
                                   market_code="US")
    assert built["extras"]["market"] == "US"
    assert built["extras"]["liquidity"]["classification"] == "normal"
    eg = build_composite_extras(df, compute_all(df), index_membership="NILEX")
    assert eg["extras"]["market"] == "EG"
    assert eg["extras"]["liquidity"]["classification"] == "thin"
