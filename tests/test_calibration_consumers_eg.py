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
