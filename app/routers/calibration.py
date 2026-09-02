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

from fastapi import APIRouter, Depends

from app.core.auth import get_current_user
from app.core.forecast import (
    CALIBRATION,
    EGX_Z_CONE_60D,
    ONE_SIGMA_COVERAGE_PCT,
)
from app.core.regime import (
    REGIME_ASSOCIATION_N,
    REGIME_ASSOCIATION_RHO,
    REGIME_ASSOCIATION_SIGNIFICANT,
    REGIME_ASSOCIATION_T,
)
from app.core.risk_grade import RISK_CALIBRATION

router = APIRouter()


@router.get("/api/calibration")
def get_calibration(user=Depends(get_current_user)):
    """Every measured claim the app makes, with what it actually delivered."""
    return {
        "forecast": {
            "fitted_at": CALIBRATION["fitted_at"],
            "n_observations": CALIBRATION["n_observations"],
            "universe": CALIBRATION["universe"],
            "bands": [
                {
                    "claim": "Typical daily move (±1σ)",
                    "promised_pct": 68,
                    "delivered_pct": ONE_SIGMA_COVERAGE_PCT["ewma"]["daily"],
                    "note": "Gaussian theory says 68%. EGX's centre is thinner.",
                },
                {
                    "claim": "Typical weekly move (±1σ)",
                    "promised_pct": 68,
                    "delivered_pct": ONE_SIGMA_COVERAGE_PCT["ewma"]["weekly"],
                    "note": None,
                },
                {
                    "claim": "Typical monthly move (±1σ)",
                    "promised_pct": 68,
                    "delivered_pct": ONE_SIGMA_COVERAGE_PCT["ewma"]["monthly"],
                    "note": None,
                },
                {
                    "claim": "3-month outcome range (90% band)",
                    "promised_pct": 90,
                    "delivered_pct": 90.0,
                    "note": ("Fitted to EGX's own distribution. The textbook "
                             "Gaussian band delivered only 85.8%."),
                },
            ],
            # The number that stops coverage being gamed.
            "sharpness": {
                "median_width_pct_of_spot":
                    CALIBRATION["band_width_pct_of_spot_median"],
                "p25": CALIBRATION["band_width_pct_of_spot_p25"],
                "p75": CALIBRATION["band_width_pct_of_spot_p75"],
                "note": ("How WIDE the 90% band is, as a share of today's "
                         "price. Coverage on its own proves nothing — a band "
                         "from zero to infinity contains every outcome."),
            },
            "z_table": {str(k): v for k, v in EGX_Z_CONE_60D.items()},
        },
        "risk_grade": {
            "fitted_at": RISK_CALIBRATION["fitted_at"],
            "n_observations": RISK_CALIBRATION["n_observations"],
            "claims": [
                {
                    "claim": "Past volatility ranks future volatility",
                    "ic": RISK_CALIBRATION["vol_predicts_vol_ic"],
                    "t_non_overlapping":
                        RISK_CALIBRATION["vol_predicts_vol_t_nonoverlapping"],
                    "verdict": "strong",
                },
                {
                    "claim": "Past volatility ranks future drawdown",
                    "ic": RISK_CALIBRATION["vol_predicts_drawdown_ic"],
                    "t_non_overlapping":
                        RISK_CALIBRATION["vol_predicts_drawdown_t_nonoverlapping"],
                    "verdict": "strong",
                },
            ],
        },
        # Reported in the same place as the things that DID work, deliberately.
        # A record that only lists successes is marketing.
        "what_failed": [
            {
                "claim": "The composite score ranks stocks by future return",
                "measured": "IC ≈ 0, slightly negative (−0.029)",
                "outcome": "Buy/Sell labels removed; they describe condition now.",
            },
            {
                "claim": "Market condition predicts the next three months",
                "measured": (f"rho {REGIME_ASSOCIATION_RHO}, Newey-West "
                             f"t {REGIME_ASSOCIATION_T} over "
                             f"{REGIME_ASSOCIATION_N} readings"),
                "outcome": ("Was published as +0.318 until it was found to be "
                            "one lucky phase of three. Now shown as context, "
                            "not a forecast."),
                "significant": REGIME_ASSOCIATION_SIGNIFICANT,
            },
            {
                "claim": "Cheap stocks (earnings yield) outperform",
                "measured": "IC +0.042, t 3.45 — but t 2.00 once low volatility "
                            "is controlled for",
                "outcome": "Not shipped: real, but not independent of the risk "
                           "grade already on screen.",
            },
            {
                "claim": "Profitable or slow-growing companies outperform",
                "measured": "gross profitability t 1.12, asset growth t 0.66; "
                            "both flip sign between halves of the sample",
                "outcome": "Not shipped.",
            },
            {
                # This verdict was WITHHELD until 2026-09-02, because the
                # backtest ran one flat risk-free rate across twenty years in
                # which the CBE ranged 8.25%-27.25%. Dated rates from
                # macro_series made it readable, and the answer is no.
                "claim": "Beating the T-bill rate (Risk-Adjusted) ranks stocks "
                         "by future return",
                "measured": "IC −0.006 at 21 days — the only horizon whose "
                            "significance is trustworthy, since 63d and 126d "
                            "windows overlap. Longer horizons drift positive "
                            "(+0.012, +0.024) but cannot be read as evidence.",
                "outcome": ("The verdict was withheld for months because the "
                            "backtest ran a flat 19% rate against a policy rate "
                            "whose true median over the window was 9.40%. Dated "
                            "rates lifted the withholding; the category still "
                            "carries 13% of the default weight on no measured "
                            "edge."),
            },
        ],
    }
