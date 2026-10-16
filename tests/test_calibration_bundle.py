"""
The calibration bundles: every fitted number lives in app/core/calibration/,
one bundle per market. The EG bundle must equal the constants it replaced —
those literals are copied here from the pre-move code (2026-09-02 fit).
"""
import math

import pytest

from app.core.calibration import _BUNDLES, band_index, for_market
from app.core.markets import EG, MARKETS, US


def test_eg_forecast_equals_the_constants_it_replaced():
    f = for_market("EG").forecast
    assert f.fitted_at == "2026-09-02"
    assert f.n_observations == 34749
    assert f.universe_label == ("EGX common stocks with >=460 daily bars, "
                                "ISIN-coded instruments excluded")
    assert (f.sigma_window, f.fit_horizon_days) == (400, 60)
    assert dict(f.band_width_pct_of_spot) == {"median": 88, "p25": 72, "p75": 110}
    assert dict(f.z_cone) == {50: 0.600, 80: 1.333, 90: 1.999, 95: 2.791}
    assert {k: dict(v) for k, v in f.one_sigma_coverage.items()} == {
        "ewma": {"daily": 78.6, "weekly": 75.5, "monthly": 71.8},
        "trailing": {"daily": 79.0, "weekly": 76.3, "monthly": 73.0},
    }
    assert (f.gaussian_cone_90_coverage, f.cone_90_coverage) == (85.8, 90.0)
    assert f.daily_band_note == "Gaussian theory says 68%. EGX's centre is thinner."
    assert f.cone_note == ("Fitted to EGX's own distribution. The textbook "
                           "Gaussian band delivered only 85.8%.")


def test_eg_forecast_payload_is_the_old_calibration_dict():
    assert for_market("EG").forecast.payload() == {
        "fitted_at": "2026-09-02",
        "n_observations": 34749,
        "universe": "EGX common stocks with >=460 daily bars, ISIN-coded "
                    "instruments excluded",
        "sigma_window": 400,
        "fit_horizon_days": 60,
        "band_width_pct_of_spot_median": 88,
        "band_width_pct_of_spot_p25": 72,
        "band_width_pct_of_spot_p75": 110,
    }


def test_eg_regime_equals_the_constants_it_replaced():
    r = for_market("EG").regime
    assert (r.weak_max, r.mixed_max) == (45.4, 51.9)
    assert (r.rho, r.t, r.n, r.significant) == (0.162, 1.74, 221, False)
    assert r.phase_rhos == (0.309, 0.167, -0.006)
    assert r.show is True
    assert r.universe_indexes == ("EGX30", "EGX70")
    assert r.universe_label == "EGX30 and EGX70"
    assert {k: (b.label, b.median_3m_pct, b.positive_rate, b.observations)
            for k, b in r.bands.items()} == {
        "weak": ("Broadly weak", -0.03, 0.50, 74),
        "mixed": ("Mixed", 5.42, 0.685, 73),
        "broad": ("Broadly healthy", 6.72, 0.689, 74),
    }
    assert "coin flip" in r.bands["weak"].summary


def test_eg_risk_equals_the_constants_it_replaced():
    k = for_market("EG").risk
    assert k.payload() == {
        "fitted_at": "2026-09-02", "n_observations": 16220, "n_symbols": 202,
        "n_dates": 221, "lookback_days": 63, "forward_days": 126,
        "vol_predicts_vol_ic": 0.5631, "vol_predicts_vol_t": 55.8,
        "vol_predicts_vol_ic_nonoverlapping": 0.5791,
        "vol_predicts_vol_t_nonoverlapping": 24.0,
        "vol_predicts_drawdown_ic": 0.4338, "vol_predicts_drawdown_t": 40.5,
        "vol_predicts_drawdown_ic_nonoverlapping": 0.4707,
        "vol_predicts_drawdown_t_nonoverlapping": 16.7,
    }
    assert {q: dict(v) for q, v in k.quintiles.items()} == {
        1: {"future_vol_ann_pct": 32.0, "median_max_drawdown_pct": 20.0, "p90_max_drawdown_pct": 40.7},
        2: {"future_vol_ann_pct": 37.8, "median_max_drawdown_pct": 24.8, "p90_max_drawdown_pct": 46.3},
        3: {"future_vol_ann_pct": 42.4, "median_max_drawdown_pct": 27.6, "p90_max_drawdown_pct": 49.3},
        4: {"future_vol_ann_pct": 45.2, "median_max_drawdown_pct": 29.9, "p90_max_drawdown_pct": 53.0},
        5: {"future_vol_ann_pct": 51.2, "median_max_drawdown_pct": 34.2, "p90_max_drawdown_pct": 59.4},
    }


def test_eg_fundamentals_and_liquidity_equal_the_constants_they_replace():
    b = for_market("EG")
    f = b.fundamentals
    assert (f.label, f.pe_median, f.dy_median, f.dy_mid_under_risk_free) == ("EGX", 12.4, 3.12, True)
    assert f.pe_bands == ((3.0, 3), (8.0, 12), (15.0, 4), (25.0, -2), (40.0, -8), (math.inf, -14))
    assert f.dy_bands == ((2.0, 0), (4.0, 4), (8.0, 8), (15.0, 4), (math.inf, -8))
    assert dict(b.liquidity_tiers) == {
        "EGX30": (100_000, 500_000), "EGX70": (50_000, 200_000),
        "EGX100": (30_000, 100_000), "NILEX": (5_000, 20_000),
    }
    assert b.liquidity_default_tier == "EGX100"
    assert (b.liquid_min_turnover, b.liquid_min_price) == (1_000_000, 0.5)
    assert (b.breadth_measured, b.dividend_drag_measured, b.provisional) == (True, True, False)
    assert [x["claim"] for x in b.measured_failures] == [
        "The composite score ranks stocks by future return",
        "Cheap stocks (earnings yield) outperform",
        "Profitable or slow-growing companies outperform",
        "Beating the T-bill rate (Risk-Adjusted) ranks stocks by future return",
    ]


def test_for_market_accepts_a_code_or_a_config_and_refuses_unknown():
    assert for_market("eg") is for_market(EG) is _BUNDLES["EG"]
    assert for_market(US).market == "US"
    with pytest.raises(KeyError):
        for_market("XX")
    with pytest.raises(KeyError):
        for_market(None)


def test_every_registered_market_has_a_bundle():
    for code in MARKETS:
        assert for_market(code).market == code


def test_us_bundle_is_fitted_on_us_data():
    us = for_market("US")
    assert us.provisional is False
    assert us.forecast.universe_label.startswith("current S&P 500 / NASDAQ-100 / Dow 30 members")
    assert set(us.liquidity_tiers) == {"US large-cap", "US"}
    assert us.liquidity_default_tier == "US"
    assert us.fundamentals.label == "US"
    assert us.regime.universe_indexes == ("SPX",)
    assert (us.breadth_measured, us.dividend_drag_measured) == (False, False)
    # Measured on the US backtest panel, not copied from EG.
    assert [x["claim"] for x in us.measured_failures] == ["The composite score ranks stocks by future return"]
    assert all("EGX" not in str(x) for x in us.measured_failures)


def test_registry_floors_agree_with_the_bundle():
    for code, market in MARKETS.items():
        cal = for_market(code)
        assert (market.liquid_min_turnover, market.liquid_min_price) == \
            (cal.liquid_min_turnover, cal.liquid_min_price), code


def test_every_bundle_keeps_the_eg_band_shape_and_points():
    eg = for_market("EG").fundamentals
    for code in MARKETS:
        f = for_market(code).fundamentals
        for bands, ref in ((f.pe_bands, eg.pe_bands), (f.dy_bands, eg.dy_bands)):
            assert [p for _, p in bands] == [p for _, p in ref]
            uppers = [u for u, _ in bands]
            assert uppers[-1] == math.inf
            assert all(a < b for a, b in zip(uppers, uppers[1:]))


def test_band_index_is_half_open():
    bands = ((3.0, 3), (8.0, 12), (math.inf, 0))
    assert band_index(bands, 2.99) == 0
    assert band_index(bands, 3.0) == 1
    assert band_index(bands, 8.0) == 2
    assert band_index(bands, 1e9) == 2
