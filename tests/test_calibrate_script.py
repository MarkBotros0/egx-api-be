"""scripts/calibrate.py: fit a market bundle from a fixture cache, render it, refuse bad input."""
import dataclasses
import importlib.util
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from calibration_fixtures import make_frame, write_frames, write_json
from app.core.calibration import band_index, for_market
from app.core.index_membership import symbols_in_index
from app.core.markets import EG, US

US_SYMS = [f"S{i:02d}" for i in range(12)]


@pytest.fixture
def caches(tmp_path):
    from scripts.market_cache import spec_for
    us_dir, eg_dir = tmp_path / "US", tmp_path / "EG"
    # Turnover spread 1x-6.5x so the p5 floor sits at the thinnest name and >=10
    # names pass it on every date (risk ICs need >=10 names per date).
    frames = {s: make_frame(900, i + 1, price=40.0 + i, volume=1_000_000.0 * (2 + i))
              for i, s in enumerate(US_SYMS)}
    frames["SPY"] = make_frame(900, 77, price=400.0)
    write_frames(str(us_dir), frames)
    write_json(us_dir / "universe.json",
               [{"symbol": s, "exchange": "NYSE", "indices": ["SPX"]} for s in US_SYMS])
    rng = np.random.default_rng(5)
    dates = frames["SPY"].index[100:760:21]
    pd.DataFrame([{"symbol": s, "date": d, "score": float(rng.uniform(30, 70)),
                   "fwd_21": float(rng.normal(0.0, 5.0))}
                  for d in dates for s in US_SYMS]).to_pickle(us_dir / "panel.pkl")
    eg30 = symbols_in_index("EGX30")[:6]
    write_frames(str(eg_dir), {s: make_frame(400, 40 + i, volume=300_000.0, vol_spread=0.8)
                               for i, s in enumerate(eg30)})
    return spec_for("US", str(us_dir)), spec_for("EG", str(eg_dir))


def _pe_rows():
    rng = np.random.default_rng(9)
    eg_pe = list(np.exp(rng.normal(math.log(12.4), 0.8, 200)))
    eg_dy = list(np.exp(rng.normal(math.log(3.12), 0.6, 200)))
    pe = list(np.exp(rng.normal(math.log(24.0), 0.6, 200)))
    dy = list(np.exp(rng.normal(math.log(1.9), 0.6, 200)))
    return eg_pe, eg_dy, pe, dy


def _build(caches):
    from scripts.calibrate import build_bundle
    us_spec, eg_spec = caches
    eg_pe, eg_dy, pe, dy = _pe_rows()
    return build_bundle(US, us_spec, eg_spec=eg_spec, eg_pe=eg_pe, eg_dy=eg_dy,
                        pe=pe, dy=dy, today=date(2026, 10, 16))


def test_build_bundle_on_a_fixture_cache(caches):
    bundle, report = _build(caches)
    assert bundle.market == "US" and bundle.provisional is False
    f = bundle.forecast
    assert f.fitted_at == "2026-10-16"
    assert f.universe_label.startswith("current S&P 500 / NASDAQ-100 / Dow 30 members, ")
    assert sorted(f.z_cone) == [50, 80, 90, 95]
    assert [f.z_cone[k] for k in (50, 80, 90, 95)] == sorted(f.z_cone[k] for k in (50, 80, 90, 95))
    assert set(f.one_sigma_coverage) == {"ewma", "trailing"}
    assert f.band_width_pct_of_spot["p25"] <= f.band_width_pct_of_spot["median"] <= f.band_width_pct_of_spot["p75"]
    assert sorted(bundle.risk.quintiles) == [1, 2, 3, 4, 5]
    assert bundle.regime.universe_indexes == ("SPX",) and bundle.regime.universe_label == "S&P 500"
    assert set(bundle.liquidity_tiers) == {"US large-cap", "US"}
    # Index members carry no share-count floor; other listings a real one.
    assert bundle.liquidity_tiers["US large-cap"] == (0, 0)
    thin, low = bundle.liquidity_tiers["US"]
    assert 0 < thin < low
    assert bundle.liquidity_default_tier == "US"
    assert bundle.liquid_min_price == 5.0
    m = bundle.liquid_min_turnover / 10 ** math.floor(math.log10(bundle.liquid_min_turnover))
    assert round(m, 9) in (1.0, 2.0, 5.0)
    fund = bundle.fundamentals
    eg_fund = for_market("EG").fundamentals
    assert fund.label == "US"
    assert band_index(fund.pe_bands, fund.pe_median) == band_index(eg_fund.pe_bands, eg_fund.pe_median)
    assert band_index(fund.dy_bands, fund.dy_median) == band_index(eg_fund.dy_bands, eg_fund.dy_median)
    assert (bundle.breadth_measured, bundle.dividend_drag_measured) == (False, False)
    # Random fixture scores rank nothing, so the composite test is listed as failed.
    assert [x["claim"] for x in bundle.measured_failures] == ["The composite score ranks stocks by future return"]
    assert "thin_share" in report["liquidity"]


def test_render_round_trips(tmp_path, caches):
    from scripts.calibrate import render_module
    for bundle in (for_market("EG"), _build(caches)[0]):
        path = tmp_path / f"gen_{bundle.market.lower()}.py"
        path.write_text(render_module(bundle), encoding="utf-8")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.BUNDLE == bundle


def test_round_down_nice():
    from scripts.calibrate import CalibrationRefused, round_down_nice
    assert round_down_nice(23.4e6) == 20e6
    assert round_down_nice(9.8e6) == 5e6
    assert round_down_nice(1.0) == 1.0
    assert round_down_nice(1000.0) == 1000.0
    with pytest.raises(CalibrationRefused):
        round_down_nice(0.0)
    with pytest.raises(CalibrationRefused):
        round_down_nice(float("nan"))


def test_mapping_a_distribution_onto_itself_reproduces_its_edges():
    from scripts.calibrate import _map_edges
    rng = np.random.default_rng(3)
    ref = list(np.exp(rng.normal(math.log(12.4), 0.8, 5000)))
    got = _map_edges(ref, [3, 8, 15, 25, 40], ref, 1)
    assert all(abs(g - e) / e < 0.05 for g, e in zip(got, [3, 8, 15, 25, 40]))


def test_regime_with_a_wrong_signed_association_is_hidden(caches):
    from scripts.calibrate import regime_cal_from
    reg = {"cutoff_weak_max": 45.0, "cutoff_mixed_max": 52.0, "rho_overlapping": -0.05,
           "newey_west_t": -0.4, "n_readings": 200,
           "phases": [{"rho": -0.1}, {"rho": 0.0}, {"rho": -0.05}],
           "bands": {k: {"n": 60, "median": 1.0, "positive_rate": 0.55}
                     for k in ("weak", "mixed", "broad")}}
    r = regime_cal_from(reg, caches[0], "2026-10-16")
    assert r.show is False and r.significant is False
    assert "wrong way" in r.failure_outcome
    assert "positive about 6 times in ten" in r.bands["mixed"].summary


def test_fundamentals_refuse_thin_or_empty_data():
    from scripts.calibrate import CalibrationRefused, fit_fundamentals
    eg_pe, eg_dy, pe, dy = _pe_rows()
    with pytest.raises(CalibrationRefused):
        fit_fundamentals(eg_pe, eg_dy, [], dy, US)
    with pytest.raises(CalibrationRefused):
        fit_fundamentals(eg_pe, eg_dy, pe[:10], dy, US)


def test_main_refuses_without_fundamentals_and_writes_nothing(caches, tmp_path, monkeypatch):
    import scripts.calibrate as cal
    from alert_fakes import FakeDB
    monkeypatch.setattr("app.core.db.get_db", lambda: FakeDB())   # pe_data: no rows
    target = tmp_path / "us_out.py"
    monkeypatch.setattr(cal, "bundle_path", lambda code: str(target))
    us_spec, eg_spec = caches
    rc = cal.main(["--market", "US", "--cache-dir", us_spec.cache_dir,
                   "--eg-cache-dir", eg_spec.cache_dir, "--write"])
    assert rc == 1 and not target.exists()


def test_write_refuses_eg():
    import scripts.calibrate as cal
    assert cal.main(["--market", "EG", "--write"]) == 2


def test_the_other_listed_tier_never_collapses():
    """First US fit: 500k/1M re-rounded to 20k/20k, so "low" was unreachable."""
    from scripts.calibrate import CalibrationRefused, other_listed_tier
    eg = for_market("EG").liquidity_tiers
    assert other_listed_tier(500_000, 1_000_000, eg) == (25_000, 40_000)
    with pytest.raises(CalibrationRefused):
        other_listed_tier(500_000, 500_000, eg)


def test_a_score_that_did_rank_returns_is_not_listed_as_a_failure(tmp_path):
    from scripts.calibrate import composite_failures
    from scripts.market_cache import spec_for
    dates = pd.bdate_range("2020-01-01", periods=30)
    rows = [{"symbol": f"S{i}", "date": d, "score": float(i), "fwd_21": float(i)}
            for d in dates for i in range(20)]
    (tmp_path / "US").mkdir()
    pd.DataFrame(rows).to_pickle(tmp_path / "US" / "panel.pkl")
    assert composite_failures(spec_for("US", str(tmp_path / "US"))) == ()
