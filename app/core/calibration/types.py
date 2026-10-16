"""
Calibration bundles: every number the app states as a MEASURED FACT about a
market, grouped per market so a US screen can never quote an EGX fit.

One frozen `Calibration` per market lives beside this file (eg.py, us.py) and
is read through `app.core.calibration.for_market`. Nothing outside this
package may hold a fitted constant — tests/test_calibration_guard.py fails if
one of the old module-level names comes back.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class ForecastCal:
    fitted_at: str
    n_observations: int
    universe_label: str
    sigma_window: int
    fit_horizon_days: int
    band_width_pct_of_spot: Mapping[str, float]          # {"median", "p25", "p75"}
    z_cone: Mapping[int, float]                          # {50, 80, 90, 95}
    one_sigma_coverage: Mapping[str, Mapping[str, float]]  # {"ewma"|"trailing": {daily, weekly, monthly}}
    gaussian_cone_90_coverage: float   # what the textbook 90% band delivered
    cone_90_coverage: float            # what the fitted 90% band delivered
    daily_band_note: Optional[str]
    cone_note: str

    def payload(self) -> dict:
        """The `calibration` dict expected_move / outcome_band ship. Shape and
        key order unchanged since 2026-09-02, so clients see no difference."""
        return {
            "fitted_at": self.fitted_at,
            "n_observations": self.n_observations,
            "universe": self.universe_label,
            "sigma_window": self.sigma_window,
            "fit_horizon_days": self.fit_horizon_days,
            "band_width_pct_of_spot_median": self.band_width_pct_of_spot["median"],
            "band_width_pct_of_spot_p25": self.band_width_pct_of_spot["p25"],
            "band_width_pct_of_spot_p75": self.band_width_pct_of_spot["p75"],
        }


@dataclass(frozen=True)
class RegimeBand:
    label: str
    summary: str
    median_3m_pct: float
    positive_rate: float
    observations: int


@dataclass(frozen=True)
class RegimeCal:
    fitted_at: str
    weak_max: float
    mixed_max: float
    rho: float              # overlapping rank correlation, every reading
    t: float                # Newey-West, lag = overlap depth
    n: int
    significant: bool       # |t| >= 1.96
    phase_rhos: tuple       # all de-overlapped phases, never just one
    bands: Mapping[str, RegimeBand]   # "weak" | "mixed" | "broad"
    universe_indexes: tuple # index ids the reading averages over
    universe_label: str     # "EGX30 and EGX70", "S&P 500"
    failure_outcome: str    # the /api/calibration "what failed" outcome line

    @property
    def show(self) -> bool:
        """A wrong-signed association is hidden: a card saying "broad
        strength" that preceded WEAKER markets would mislead. A weak but
        right-signed one ships as context (EG: t=1.74)."""
        return self.rho > 0


@dataclass(frozen=True)
class RiskCal:
    fitted_at: str
    n_observations: int
    n_symbols: int
    n_dates: int
    lookback_days: int
    forward_days: int
    vol_predicts_vol_ic: float
    vol_predicts_vol_t: float
    vol_predicts_vol_ic_nonoverlapping: float
    vol_predicts_vol_t_nonoverlapping: float
    vol_predicts_drawdown_ic: float
    vol_predicts_drawdown_t: float
    vol_predicts_drawdown_ic_nonoverlapping: float
    vol_predicts_drawdown_t_nonoverlapping: float
    quintiles: Mapping[int, Mapping[str, float]]

    def payload(self) -> dict:
        """The `calibration` dict /api/risk ships (every field but quintiles)."""
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self) if f.name != "quintiles"}


@dataclass(frozen=True)
class FundamentalsCal:
    label: str              # the word in "the EGX median" / "the US median"
    pe_median: float
    pe_bands: tuple         # ((upper, points), ...) half-open "< upper"; last upper is math.inf
    dy_median: float
    dy_bands: tuple
    dy_mid_under_risk_free: bool  # the +8 yield band's upper bound sits below the market's cash rate
    # True only where the CAUSES behind the extreme bands were checked against
    # the market's own names: bottom P/E band = one-off earnings or a collapsed
    # price, top yield band = a special dividend or a collapsed price. Measured
    # on the EGX (MEGM: P/E 0.7, 42.55% yield). False everywhere else — the US
    # bottom P/E band is Allstate, AT&T, Honeywell; its top yield band a REIT —
    # so copy states the distance from the median and the causes as possibilities.
    extreme_causes_measured: bool = False


@dataclass(frozen=True)
class Calibration:
    market: str
    forecast: ForecastCal
    regime: RegimeCal
    risk: RiskCal
    fundamentals: FundamentalsCal
    liquidity_tiers: Mapping[str, tuple]   # {tier: (thin_below, low_below)} shares/day
    liquidity_default_tier: str
    liquid_min_turnover: Optional[float]   # mirrors MarketConfig; a test pins they agree
    liquid_min_price: Optional[float]
    measured_failures: tuple = ()          # /api/calibration "what failed" entries measured on THIS market
    breadth_measured: bool = False         # breadth's evidence numbers were measured on this market
    dividend_drag_measured: bool = False   # the price-vs-total-return drag was measured on this market
    provisional: bool = False              # True only until the market's bundle is fitted


def band_index(bands, value) -> int:
    """Index of the half-open band (`value < upper`) that `value` falls in."""
    for i, (upper, _points) in enumerate(bands):
        if value < upper:
            return i
    return len(bands) - 1
