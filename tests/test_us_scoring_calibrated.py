"""US scores are calibrated: US floors per tier, US-centred valuation copy, no EGX wording."""
import json
import os

os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pandas as pd  # noqa: E402

from calibration_fixtures import make_frame  # noqa: E402
from app.core.calibration import band_index, for_market  # noqa: E402
from app.core.markets import EG, US, public_market  # noqa: E402

US_CAL = for_market("US")


def _universe(monkeypatch, rows=None, boom=False):
    from app.core import universe as universe_mod

    def fake(db, market):
        if boom:
            raise RuntimeError("db down")
        return rows if rows is not None else [
            {"symbol": "AAPL", "indices": ["SPX", "NDX"]},
            {"symbol": "XYZQ", "indices": []},
        ]
    monkeypatch.setattr(universe_mod, "load_universe", fake)


def test_us_index_members_get_the_large_cap_tier(monkeypatch):
    from app.core.index_membership import get_index_membership
    _universe(monkeypatch)
    assert get_index_membership("AAPL", US, db=object()) == "US large-cap"
    assert get_index_membership("aapl", US, db=object()) == "US large-cap"
    assert get_index_membership("COMI", EG, db=object()) == "EGX30"


def test_us_membership_degrades_to_other_listed(monkeypatch):
    from app.core.index_membership import get_index_membership
    _universe(monkeypatch)
    assert get_index_membership("XYZQ", US, db=object()) == "US"
    assert get_index_membership("NOTLISTED", US, db=object()) == "US"
    assert get_index_membership("AAPL", US) == "US"                  # no db
    _universe(monkeypatch, boom=True)
    assert get_index_membership("AAPL", US, db=object()) == "US"     # unreadable universe


def test_us_large_cap_floors_apply_and_name_the_tier():
    from app.core.composite import score_volume
    from app.core.indicators import liquidity_score
    thin_below, low_below = US_CAL.liquidity_tiers["US large-cap"]
    out = liquidity_score(pd.Series([float(thin_below) / 2] * 30), "US large-cap", cal=US_CAL)
    assert out["index_membership"] == "US large-cap" and out["classification"] == "thin"
    # One real volume input: liquidity alone never carries the category.
    _, reasons = score_volume(True, True, None, None, liquidity=out)
    assert any("a US large-cap name" in r for r in reasons)


def test_us_is_calibrated():
    assert US.score_calibrated is True
    assert public_market(US, pd.Timestamp("2026-10-16").date())["score_calibrated"] is True


def test_the_us_median_lands_in_the_band_the_eg_median_does():
    from app.core.composite import score_quality
    eg = for_market("EG").fundamentals
    us = US_CAL.fundamentals
    assert band_index(us.pe_bands, us.pe_median) == band_index(eg.pe_bands, eg.pe_median)
    assert band_index(us.dy_bands, us.dy_median) == band_index(eg.dy_bands, eg.dy_median)
    us_pts = score_quality(None, None, None, pe_ratio=us.pe_median, dividend_yield=us.dy_median,
                           fundamentals=us)
    eg_pts = score_quality(None, None, None, pe_ratio=eg.pe_median, dividend_yield=eg.dy_median,
                           fundamentals=eg)
    assert us_pts[0] == eg_pts[0]
    assert any("the US median" in r for r in us_pts[1])


def test_no_egx_or_egypt_in_any_us_reason():
    from app.core.composite import compute_composite, score_categories
    from app.core.extras_builder import build_composite_extras
    from app.core.indicators import compute_all
    df = make_frame(500, 4, price=120.0)
    bench = make_frame(500, 8, price=5000.0)["close"]
    built = build_composite_extras(
        df, compute_all(df), egx30_close=bench, risk_free_rate_pct=4.0,
        pe_ratio=US_CAL.fundamentals.pe_median, dividend_yield=US_CAL.fundamentals.dy_median,
        loss_making=False, index_membership="US large-cap", benchmark_label="S&P 500",
        score_calibrated=True, market_code="US")
    text = json.dumps(compute_composite(compute_all(df), extras=built["extras"],
                                        weights=None, macro=None), default=str)
    assert "EGX" not in text and "Egypt" not in text
    for pe in (1.0, 9.0, 20.0, 30.0, 45.0, 200.0):
        for dy in (0.5, 1.5, 3.0, 5.0, 20.0):
            text = json.dumps(score_categories({}, {"market": "US", "score_calibrated": True,
                                                    "pe_ratio": pe, "dividend_yield": dy}))
            assert "EGX" not in text and "Egypt" not in text, (pe, dy)
    lagging = score_categories({}, {"market": "US", "score_calibrated": True,
                                    "annualized_return_pct": -5.0, "risk_free_rate_pct": 4.0,
                                    "history_days": 300})["risk_adjusted"][1]
    assert any("T-bill" in r for r in lagging)
    assert not any("EGX" in r or "policy rate" in r for r in lagging)
