"""
One calibration bundle per market. `for_market` is the only door in, and it
NEVER falls back to EG: a market without a bundle raises, because quoting
another market's fit as this one's is the bug this package exists to end.
"""
from app.core.calibration.types import (  # noqa: F401
    Calibration, ForecastCal, FundamentalsCal, RegimeBand, RegimeCal, RiskCal,
    band_index,
)
from app.core.calibration import eg as _eg
from app.core.calibration import us as _us

_BUNDLES = {"EG": _eg.BUNDLE, "US": _us.BUNDLE}


def for_market(market) -> Calibration:
    """The bundle for a market code ("EG") or MarketConfig. KeyError if none."""
    code = market if isinstance(market, str) else getattr(market, "code", None)
    key = code.strip().upper() if isinstance(code, str) else None
    if key not in _BUNDLES:
        raise KeyError(f"No calibration bundle for market {code!r}")
    return _BUNDLES[key]
