"""
US calibration bundle — PROVISIONAL.

Today's effective US numbers: the EG fits US has been silently using, a
lenient "US" liquidity tier equal to NILEX, and no liquidity floors (so no US
stock is tradeable and none is risk-graded). It exists only so the suite stays
green between "consumers read bundles" and "US is fitted".

Replaced by `python -m scripts.calibrate --market US --write`; /api/calibration
refuses a provisional bundle.
"""
import dataclasses

from app.core.calibration.eg import BUNDLE as _EG

BUNDLE = dataclasses.replace(
    _EG,
    market="US",
    fundamentals=dataclasses.replace(_EG.fundamentals, label="US"),
    regime=dataclasses.replace(_EG.regime, universe_indexes=("SPX",),
                               universe_label="S&P 500"),
    liquidity_tiers={"US": (5_000, 20_000)},
    liquidity_default_tier="US",
    liquid_min_turnover=None,
    liquid_min_price=None,
    measured_failures=(),
    breadth_measured=False,
    dividend_drag_measured=False,
    provisional=True,
)
