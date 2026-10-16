"""
GET /api/calibration — the app's own accuracy record.

WHY THIS EXISTS
---------------
This app makes two kinds of quantitative claim: forecast bands that promise a
coverage level, and a risk grade that promises an ordering. Both were measured
before they shipped. Almost no retail finance product shows that measurement
back to the person relying on it — FiveThirtyEight's "Checking Our Work" is the
canonical public example and it no longer exists.

Everything here is read from the fitted constants the app actually uses, so the
page cannot drift from the code. Regenerate them with
`python -m scripts.calibrate`; there is nothing to keep in sync by hand.

COVERAGE ALONE IS GAMEABLE, so width ships beside it. A band from zero to
infinity covers 100% of outcomes and tells you nothing. The 90% band spans a
median 88% of spot over sixty days, and that is the honest headline: the range
is wide, and a range that admits it is worth more than a point estimate that
does not.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user
from app.core.calibration import Calibration, for_market
from app.core.markets import MarketConfig, market_param

router = APIRouter()


@router.get("/api/calibration")
def get_calibration(user=Depends(get_current_user),
                    market: MarketConfig = Depends(market_param)):
    """Every measured claim the app makes, with what it actually delivered."""
    if market.code != "EG":
        raise HTTPException(status_code=400,
                            detail="Calibration is EGX-only until US is calibrated")
    return build_record(market, for_market(market))


def build_record(market: MarketConfig, cal: Calibration) -> dict:
    """The accuracy record of ONE market, read entirely from its bundle."""
    f, r, k = cal.forecast, cal.regime, cal.risk
    cover = f.one_sigma_coverage["ewma"]
    regime_entry = {
        "claim": "Market condition predicts the next three months",
        "measured": f"rho {r.rho}, Newey-West t {r.t} over {r.n} readings",
        "outcome": r.failure_outcome,
        "significant": r.significant,
    }
    # Reported in the same place as the things that DID work, deliberately.
    # A record that only lists successes is marketing. Only failures measured
    # on THIS market are listed; the regime entry keeps its EG position.
    what_failed = [dict(x) for x in cal.measured_failures]
    what_failed.insert(min(1, len(what_failed)), regime_entry)
    return {
        "market": market.code,
        "currency": market.currency,
        "forecast": {
            "fitted_at": f.fitted_at,
            "n_observations": f.n_observations,
            "universe": f.universe_label,
            "bands": [
                {"claim": "Typical daily move (±1σ)", "promised_pct": 68,
                 "delivered_pct": cover["daily"], "note": f.daily_band_note},
                {"claim": "Typical weekly move (±1σ)", "promised_pct": 68,
                 "delivered_pct": cover["weekly"], "note": None},
                {"claim": "Typical monthly move (±1σ)", "promised_pct": 68,
                 "delivered_pct": cover["monthly"], "note": None},
                {"claim": "3-month outcome range (90% band)", "promised_pct": 90,
                 "delivered_pct": f.cone_90_coverage, "note": f.cone_note},
            ],
            # The number that stops coverage being gamed.
            "sharpness": {
                "median_width_pct_of_spot": f.band_width_pct_of_spot["median"],
                "p25": f.band_width_pct_of_spot["p25"],
                "p75": f.band_width_pct_of_spot["p75"],
                "note": ("How WIDE the 90% band is, as a share of today's "
                         "price. Coverage on its own proves nothing — a band "
                         "from zero to infinity contains every outcome."),
            },
            "z_table": {str(z): v for z, v in f.z_cone.items()},
        },
        "risk_grade": {
            "fitted_at": k.fitted_at,
            "n_observations": k.n_observations,
            "claims": [
                {"claim": "Past volatility ranks future volatility",
                 "ic": k.vol_predicts_vol_ic,
                 "t_non_overlapping": k.vol_predicts_vol_t_nonoverlapping,
                 "verdict": "strong"},
                {"claim": "Past volatility ranks future drawdown",
                 "ic": k.vol_predicts_drawdown_ic,
                 "t_non_overlapping": k.vol_predicts_drawdown_t_nonoverlapping,
                 "verdict": "strong"},
            ],
        },
        "what_failed": what_failed,
    }
