"""
EG calibration bundle — today's fitted EG numbers, moved VERBATIM from
core/forecast.py, core/regime.py, core/risk_grade.py, core/composite.py
(P/E and yield bands), core/indicators.py (_INDEX_LIQ_FLOORS) and
routers/calibration.py on 2026-10-16. No value was changed.

Forecast, regime and risk were fitted by scripts/calibrate.py on 2026-09-02
over the cached EGX panel; the derivations and caveats are documented in
CLAUDE.md (*Forecast calibration*, *GET /api/market_regime*, *GET /api/risk*,
*Fundamentals bands*, *Liquidity*). Refitting EG is out of scope:
`scripts.calibrate --market EG --write` refuses.
"""
import math

from app.core.calibration.types import (
    Calibration, ForecastCal, FundamentalsCal, RegimeBand, RegimeCal, RiskCal,
)

BUNDLE = Calibration(
    market="EG",
    forecast=ForecastCal(
        fitted_at="2026-09-02",
        n_observations=34749,
        # ISIN-coded rows are excluded by LENGTH, not by the "EGS" prefix —
        # EGSA is a real four-character EGX ticker.
        universe_label="EGX common stocks with >=460 daily bars, ISIN-coded "
                       "instruments excluded",
        sigma_window=400,
        fit_horizon_days=60,
        band_width_pct_of_spot={"median": 88, "p25": 72, "p75": 110},
        z_cone={50: 0.600, 80: 1.333, 90: 1.999, 95: 2.791},
        one_sigma_coverage={
            "ewma": {"daily": 78.6, "weekly": 75.5, "monthly": 71.8},
            "trailing": {"daily": 79.0, "weekly": 76.3, "monthly": 73.0},
        },
        gaussian_cone_90_coverage=85.8,
        cone_90_coverage=90.0,
        daily_band_note="Gaussian theory says 68%. EGX's centre is thinner.",
        cone_note="Fitted to EGX's own distribution. The textbook Gaussian "
                  "band delivered only 85.8%.",
    ),
    regime=RegimeCal(
        fitted_at="2026-09-02",
        weak_max=45.4,
        mixed_max=51.9,
        rho=0.162,
        t=1.74,
        n=221,
        significant=False,
        phase_rhos=(0.309, 0.167, -0.006),
        bands={
            "weak": RegimeBand(
                label="Broadly weak",
                summary=("Most of the market is in poor technical condition. "
                         "Historically the next three months were a coin flip "
                         "averaging about nothing."),
                median_3m_pct=-0.03, positive_rate=0.50, observations=74),
            "mixed": RegimeBand(
                label="Mixed",
                summary=("The market is in middling condition. Historically the "
                         "next three months were positive about two times in three."),
                median_3m_pct=5.42, positive_rate=0.685, observations=73),
            "broad": RegimeBand(
                label="Broadly healthy",
                summary=("Most of the market is in good technical condition. "
                         "Historically the next three months were positive about "
                         "seven times in ten — though not meaningfully better "
                         "than the Mixed band."),
                median_3m_pct=6.72, positive_rate=0.689, observations=74),
        },
        universe_indexes=("EGX30", "EGX70"),
        universe_label="EGX30 and EGX70",
        failure_outcome=("Was published as +0.318 until it was found to be "
                         "one lucky phase of three. Now shown as context, "
                         "not a forecast."),
    ),
    risk=RiskCal(
        fitted_at="2026-09-02",
        n_observations=16220,
        n_symbols=202,
        n_dates=221,
        lookback_days=63,
        forward_days=126,
        vol_predicts_vol_ic=0.5631,
        vol_predicts_vol_t=55.8,
        vol_predicts_vol_ic_nonoverlapping=0.5791,
        vol_predicts_vol_t_nonoverlapping=24.0,
        vol_predicts_drawdown_ic=0.4338,
        vol_predicts_drawdown_t=40.5,
        vol_predicts_drawdown_ic_nonoverlapping=0.4707,
        vol_predicts_drawdown_t_nonoverlapping=16.7,
        quintiles={
            1: {"future_vol_ann_pct": 32.0, "median_max_drawdown_pct": 20.0, "p90_max_drawdown_pct": 40.7},
            2: {"future_vol_ann_pct": 37.8, "median_max_drawdown_pct": 24.8, "p90_max_drawdown_pct": 46.3},
            3: {"future_vol_ann_pct": 42.4, "median_max_drawdown_pct": 27.6, "p90_max_drawdown_pct": 49.3},
            4: {"future_vol_ann_pct": 45.2, "median_max_drawdown_pct": 29.9, "p90_max_drawdown_pct": 53.0},
            5: {"future_vol_ann_pct": 51.2, "median_max_drawdown_pct": 34.2, "p90_max_drawdown_pct": 59.4},
        },
    ),
    fundamentals=FundamentalsCal(
        label="EGX",
        pe_median=12.4,
        # <3 is "implausibly low" (MEGM trades at 0.7), not a bargain.
        pe_bands=((3.0, 3), (8.0, 12), (15.0, 4), (25.0, -2), (40.0, -8), (math.inf, -14)),
        dy_median=3.12,
        # Deliberately non-monotonic: >=15% is a special dividend or a collapsed price.
        dy_bands=((2.0, 0), (4.0, 4), (8.0, 8), (15.0, 4), (math.inf, -8)),
        dy_mid_under_risk_free=True,
    ),
    liquidity_tiers={
        "EGX30": (100_000, 500_000),
        "EGX70": (50_000, 200_000),
        "EGX100": (30_000, 100_000),
        "NILEX": (5_000, 20_000),
    },
    liquidity_default_tier="EGX100",
    liquid_min_turnover=1_000_000,
    liquid_min_price=0.5,
    measured_failures=(
        {"claim": "The composite score ranks stocks by future return",
         "measured": "IC ≈ 0, slightly negative (−0.029)",
         "outcome": "Buy/Sell labels removed; they describe condition now."},
        {"claim": "Cheap stocks (earnings yield) outperform",
         "measured": "IC +0.042, t 3.45 — but t 2.00 once low volatility "
                     "is controlled for",
         "outcome": "Not shipped: real, but not independent of the risk "
                    "grade already on screen."},
        {"claim": "Profitable or slow-growing companies outperform",
         "measured": "gross profitability t 1.12, asset growth t 0.66; "
                     "both flip sign between halves of the sample",
         "outcome": "Not shipped."},
        {"claim": "Beating the T-bill rate (Risk-Adjusted) ranks stocks "
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
                     "edge.")},
    ),
    breadth_measured=True,
    dividend_drag_measured=True,
    provisional=False,
)
