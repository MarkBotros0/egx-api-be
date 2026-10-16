"""US surfaces read the US bundle: Market Condition, the Calibration record, the risk grade."""
import dataclasses
import json
import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from alert_fakes import FakeDB  # noqa: E402
from app.core import auth as auth_mod  # noqa: E402
from app.core import cache  # noqa: E402
from app.core import calibration as cal_pkg  # noqa: E402
from app.core.auth import CurrentUser, create_access_token  # noqa: E402
from app.core.calibration import for_market  # noqa: E402
from app.core.markets import US  # noqa: E402
from app.main import app  # noqa: E402

ME = CurrentUser(id="u1", email="sara@example.com", role="user")
SPX = [f"S{i:02d}" for i in range(20)]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth_mod, "_load_user", lambda uid: ME)
    monkeypatch.setattr(auth_mod, "AUTH_SECRET", "test-secret-for-us-calibration-0123456789")
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    cache._store.clear()
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {create_access_token(ME.id)}"})
    yield c
    cache._store.clear()


@pytest.fixture
def regime_env(monkeypatch):
    from app.core import universe as universe_mod
    from app.routers import market_regime as regime_mod
    seen = {"symbols": None}
    rows = ([{"symbol": s, "indices": ["SPX"]} for s in SPX]
            + [{"symbol": "NDXONLY", "indices": ["NDX"]}, {"symbol": "XYZQ", "indices": []}])

    def fake_scores(symbols, interval="Daily", market=None):
        seen["symbols"] = list(symbols)
        return {s: {"score": 50.0} for s in symbols}

    monkeypatch.setattr(universe_mod, "load_universe", lambda db, market: rows)
    monkeypatch.setattr(regime_mod, "read_cached_scores", fake_scores)
    monkeypatch.setattr(regime_mod, "get_db", lambda: FakeDB())
    return seen


def _shown_us(monkeypatch, rho=0.2):
    # provisional=False: a provisional bundle's regime is hidden whatever its rho.
    us = for_market("US")
    monkeypatch.setitem(cal_pkg._BUNDLES, "US",
                        dataclasses.replace(us, provisional=False,
                                            regime=dataclasses.replace(us.regime, rho=rho)))
    return for_market("US")


def test_us_regime_averages_spx_members_with_us_bands(client, regime_env, monkeypatch):
    us = _shown_us(monkeypatch)
    r = client.get("/api/market_regime", params={"market": "US"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(regime_env["symbols"]) == SPX
    assert (body["universe_size"], body["n_symbols"]) == (20, 20)
    want = "weak" if 50.0 < us.regime.weak_max else "mixed" if 50.0 < us.regime.mixed_max else "broad"
    assert body["band"] == want
    assert body["hist_positive_rate"] == us.regime.bands[want].positive_rate
    assert body["universe_label"] == "S&P 500"
    assert "evidence" not in body.get("breadth", {})
    assert "EGX" not in json.dumps(body) and "Egypt" not in json.dumps(body)


def test_us_regime_hidden_when_association_has_wrong_sign(client, regime_env, monkeypatch):
    _shown_us(monkeypatch, rho=-0.05)
    r = client.get("/api/market_regime", params={"market": "US"})
    assert r.status_code == 200
    assert r.json() == {"market": "US", "currency": "USD", "band": None,
                        "hidden": True, "n_symbols": 0}
    assert regime_env["symbols"] is None


def test_us_regime_empty_universe_is_stale_not_500(client, regime_env, monkeypatch):
    from app.core import universe as universe_mod
    _shown_us(monkeypatch)
    monkeypatch.setattr(universe_mod, "load_universe", lambda db, market: [])
    r = client.get("/api/market_regime", params={"market": "US"})
    assert r.status_code == 200
    assert r.json()["band"] is None and r.json()["stale"] is True


def test_eg_regime_reading_names_its_universe():
    from app.core.regime import classify_regime
    assert classify_regime(52.0, 40, for_market("EG").regime)["universe_label"] == "EGX30 and EGX70"
    assert classify_regime(52.0, 3, for_market("EG").regime)["universe_label"] == "EGX30 and EGX70"


def test_us_calibration_route_serves_the_us_record(client):
    r = client.get("/api/calibration", params={"market": "US"})
    assert r.status_code == 200, r.text
    body = r.json()
    us = for_market("US")
    assert body["market"] == "US" and body["currency"] == "USD"
    assert body["forecast"]["universe"] == us.forecast.universe_label
    assert body["forecast"]["z_table"] == {str(k): v for k, v in us.forecast.z_cone.items()}
    assert body["risk_grade"]["n_observations"] == us.risk.n_observations
    assert [x["claim"] for x in body["what_failed"]] == [
        "The composite score ranks stocks by future return",
        "Market condition predicts the next three months",
    ]
    assert body["what_failed"][1]["measured"] == (
        f"rho {us.regime.rho}, Newey-West t {us.regime.t} over {us.regime.n} readings")
    assert "EGX" not in json.dumps(body) and "Egypt" not in json.dumps(body)


def test_a_provisional_bundle_is_refused(client, monkeypatch):
    monkeypatch.setitem(cal_pkg._BUNDLES, "US",
                        dataclasses.replace(for_market("US"), provisional=True))
    assert client.get("/api/calibration", params={"market": "US"}).status_code == 400


def test_us_risk_grades_with_fitted_floors_and_us_quintiles(client, monkeypatch):
    from app.routers import risk as risk_mod
    rows = [(f"S{i}", "2026-10-16T21:00:00", 15.0 + 5 * i, 14.0, 1.0, 5e8, 1.0, 100.0, True)
            for i in range(10)]
    monkeypatch.setattr(risk_mod, "get_db", lambda: FakeDB([("FROM risk_snapshot", rows)]))
    body = client.get("/api/risk", params={"market": "US"}).json()
    us = for_market("US")
    assert body["calibration"] == us.risk.payload()
    assert body["liquidity_floor"] == US.liquid_min_turnover is not None
    graded = [r for r in body["data"] if r["band"]]
    assert len(graded) == 10
    for r in graded:
        assert r["historical"] == {k: v for k, v in us.risk.quintiles[r["quintile"]].items()}


def test_a_provisional_bundle_hides_the_us_regime(client, regime_env, monkeypatch):
    """Never a band claimed on EG numbers: a provisional bundle is EG's fit."""
    monkeypatch.setitem(cal_pkg._BUNDLES, "US",
                        dataclasses.replace(for_market("US"), provisional=True))
    r = client.get("/api/market_regime", params={"market": "US"})
    assert r.status_code == 200
    assert r.json()["hidden"] is True and r.json()["band"] is None
    assert regime_env["symbols"] is None
