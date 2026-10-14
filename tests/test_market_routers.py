"""
Market plumbing through the price and scoring routers: no `market` means EG,
an explicit market is respected, and an unknown or disabled one is a 400.

Run from egx-api-be:  python -m pytest tests/test_market_routers.py -v

The auth gate middleware runs for real: a token is minted and `_load_user` is
patched, the same mechanism tests/test_auth_gate.py uses. Routes that touch
the DB get a FakeDB, so every SQL statement is inspectable.
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from alert_fakes import FakeDB  # noqa: E402
from app.core import auth as auth_mod  # noqa: E402
from app.core import cache  # noqa: E402
from app.core.auth import CurrentUser, create_access_token  # noqa: E402
from app.core.markets import US  # noqa: E402
from app.main import app  # noqa: E402

ME = CurrentUser(id="u1", email="sara@example.com", role="user")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth_mod, "_load_user", lambda uid: ME)
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    cache._store.clear()
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {create_access_token(ME.id)}"})
    yield c
    cache._store.clear()


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=idx)


@pytest.fixture
def fake_feed(monkeypatch):
    calls = []

    def fake(symbol, exchange, interval, bars):
        calls.append((symbol, exchange))
        return _frame(bars)

    import app.vendor.egxpy as egxpy
    monkeypatch.setattr(egxpy, "get_OHLCV_data", fake)
    return calls


@pytest.fixture
def fake_multi(monkeypatch):
    """get_EGXdata / get_EGX_intraday_data, recording the exchange asked for."""
    calls = []

    def frame(syms):
        idx = pd.bdate_range("2025-01-01", periods=5)
        return pd.DataFrame({s: [1.0, 1.1, 1.2, 1.1, 1.3] for s in syms}, index=idx)

    def daily(syms, interval, start, end, exchange="EGX", holiday_country="EG"):
        calls.append(("daily", exchange, holiday_country))
        return frame(syms)

    def intraday(syms, interval, start, end, exchange="EGX"):
        calls.append(("intraday", exchange))
        return frame(syms)

    import app.vendor.egxpy as egxpy
    monkeypatch.setattr(egxpy, "get_EGXdata", daily)
    monkeypatch.setattr(egxpy, "get_EGX_intraday_data", intraday)
    return calls


def _with_market(sql_log, needle):
    """Statements touching `needle` that filter or write by market."""
    return [(s, p) for s, p in sql_log if needle in s]


# ---------------------------------------------------------------------------
# /api/ohlcv
# ---------------------------------------------------------------------------

def test_ohlcv_defaults_to_egx(client, fake_feed):
    r = client.get("/api/ohlcv", params={"symbol": "COMI", "bars": 5})
    assert r.status_code == 200
    assert fake_feed[0] == ("COMI", "EGX")
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"


def test_ohlcv_refuses_unknown_market(client, fake_feed):
    r = client.get("/api/ohlcv", params={"symbol": "COMI", "market": "XX"})
    assert r.status_code == 400
    assert fake_feed == []


def test_ohlcv_refuses_disabled_market(client, fake_feed):
    r = client.get("/api/ohlcv", params={"symbol": "AAPL", "market": "US"})
    assert r.status_code == 400


def test_ohlcv_us_uses_us_exchange(client, fake_feed, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    r = client.get("/api/ohlcv", params={"symbol": "AAPL", "market": "US", "bars": 5})
    assert r.status_code == 200
    assert fake_feed[0] == ("AAPL", "NASDAQ")
    assert r.json()["market"] == "US" and r.json()["currency"] == "USD"


def test_ohlcv_explicit_exchange_still_overrides(client, fake_feed, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    client.get("/api/ohlcv", params={"symbol": "IBM", "market": "US",
                                     "exchange": "nyse", "bars": 5})
    assert fake_feed[0] == ("IBM", "NYSE")


def test_ohlcv_cache_is_per_market(client, fake_feed, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    client.get("/api/ohlcv", params={"symbol": "X", "exchange": "EGX", "bars": 5})
    r = client.get("/api/ohlcv", params={"symbol": "X", "exchange": "EGX",
                                         "market": "US", "bars": 5})
    assert len(fake_feed) == 2, "a US request was served from the EG cache entry"
    assert r.json()["market"] == "US"


# ---------------------------------------------------------------------------
# /api/historical, /api/compare, /api/intraday
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path, params, kind", [
    ("/api/historical", {"symbols": "COMI,ETEL"}, "daily"),
    ("/api/compare", {"symbols": "COMI,ETEL"}, "daily"),
    ("/api/intraday", {"symbols": "COMI", "interval": "5 Minute"}, "intraday"),
])
def test_multi_symbol_routes_default_to_eg(client, fake_multi, path, params, kind):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"
    assert fake_multi[0][0] == kind
    assert fake_multi[0][1] == "EGX"
    if kind == "daily":
        assert fake_multi[0][2] == "EG"


@pytest.mark.parametrize("path, params", [
    ("/api/historical", {"symbols": "COMI,ETEL"}),
    ("/api/compare", {"symbols": "COMI,ETEL"}),
    ("/api/intraday", {"symbols": "COMI", "interval": "5 Minute"}),
])
def test_multi_symbol_routes_refuse_unknown_market(client, fake_multi, path, params):
    r = client.get(path, params={**params, "market": "XX"})
    assert r.status_code == 400
    assert fake_multi == []


def test_historical_us_asks_the_us_exchange(client, fake_multi, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    r = client.get("/api/historical", params={"symbols": "AAPL", "market": "US"})
    assert r.status_code == 200
    assert fake_multi[0][1] == US.primary_exchange
    assert r.json()["currency"] == "USD"


# ---------------------------------------------------------------------------
# /api/analysis
# ---------------------------------------------------------------------------

@pytest.fixture
def analysis_db(monkeypatch):
    from app.routers import analysis as analysis_mod
    db = FakeDB()
    monkeypatch.setattr(analysis_mod, "get_db", lambda: db)
    monkeypatch.setattr(analysis_mod, "fetch_macro", lambda db: None)
    return db


def test_analysis_defaults_to_eg(client, fake_feed, analysis_db):
    r = client.get("/api/analysis", params={"symbol": "COMI", "bars": 60})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["market"] == "EG" and body["currency"] == "EGP"
    assert ("COMI", "EGX") in fake_feed
    assert ("EGX30", "EGX") in fake_feed, "benchmark not fetched for the market"
    pe = analysis_db.statements("FROM pe_data")
    assert pe and all("market = %s" in s and "EG" in p for s, p in pe)


def test_analysis_refuses_unknown_market(client, fake_feed, analysis_db):
    r = client.get("/api/analysis", params={"symbol": "COMI", "market": "XX"})
    assert r.status_code == 400
    assert fake_feed == []


def test_analysis_us_benchmark_is_the_sp500(client, fake_feed, analysis_db, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    r = client.get("/api/analysis", params={"symbol": "AAPL", "market": "US", "bars": 60})
    assert r.status_code == 200, r.text
    assert ("AAPL", "NASDAQ") in fake_feed
    assert (US.benchmark_symbol, US.benchmark_exchange) in fake_feed
    assert r.json()["currency"] == "USD"


def test_analysis_batch_reads_demotions_for_its_market(client, fake_feed, analysis_db):
    analysis_db.rows_for("consecutive_failures", [("COMI",)])
    r = client.get("/api/analysis", params={"mode": "batch", "symbols": "COMI"})
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG"
    demoted = analysis_db.statements("FROM risk_snapshot")
    assert demoted and all("market = %s" in s and "EG" in p for s, p in demoted)


def test_composite_cache_key_is_scoped_by_market():
    from app.routers.analysis import composite_cache_key
    tags = ("w", "n/a", "rfr19")
    assert composite_cache_key("COMI", "Daily", tags) == \
        composite_cache_key("COMI", "Daily", tags, "EG")
    assert composite_cache_key("COMI", "Daily", tags, "US") != \
        composite_cache_key("COMI", "Daily", tags)


# ---------------------------------------------------------------------------
# /api/dashboard
# ---------------------------------------------------------------------------

def _snapshot_row(symbol):
    from app.core.composite import CATEGORY_ORDER
    return (symbol, "2026-10-14T15:00", "2026-10-14T15:00", 10.0, 9.5,
            "[9.5, 10.0]", True, 30.0, 0, "2026-10-14") + (50.0,) * len(CATEGORY_ORDER)


@pytest.fixture
def dashboard_db(monkeypatch):
    from app.routers import dashboard as dashboard_mod
    db = FakeDB()
    db.rows_for("FROM risk_snapshot", [_snapshot_row("COMI")])
    monkeypatch.setattr(dashboard_mod, "get_db", lambda: db)
    monkeypatch.setattr(dashboard_mod, "fetch_macro", lambda db: None)
    return db


def test_dashboard_defaults_to_eg(client, dashboard_db):
    r = client.get("/api/dashboard")
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"
    for needle in ("FROM risk_snapshot", "FROM pe_data"):
        stmts = dashboard_db.statements(needle)
        assert stmts, needle
        assert all("market = %s" in s and "EG" in p for s, p in stmts), needle


def test_dashboard_refuses_unknown_market(client, dashboard_db):
    assert client.get("/api/dashboard", params={"market": "XX"}).status_code == 400
    assert dashboard_db.log == []


# ---------------------------------------------------------------------------
# /api/risk
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_db(monkeypatch):
    from app.routers import risk as risk_mod
    db = FakeDB()
    monkeypatch.setattr(risk_mod, "get_db", lambda: db)
    return db


def test_risk_defaults_to_eg(client, risk_db):
    r = client.get("/api/risk")
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"
    stmts = risk_db.statements("FROM risk_snapshot")
    assert stmts and all("market = %s" in s and "EG" in p for s, p in stmts)


def test_risk_refuses_unknown_market(client, risk_db):
    assert client.get("/api/risk", params={"market": "XX"}).status_code == 400
    assert risk_db.log == []


# ---------------------------------------------------------------------------
# /api/market_regime
# ---------------------------------------------------------------------------

@pytest.fixture
def regime_db(monkeypatch):
    from app.routers import market_regime as regime_mod
    db = FakeDB()
    seen = []

    def fake_scores(symbols, interval="Daily", market=None):
        seen.append(market)
        return {}

    monkeypatch.setattr(regime_mod, "get_db", lambda: db)
    monkeypatch.setattr(regime_mod, "read_cached_scores", fake_scores)
    db.seen_markets = seen
    return db


def test_market_regime_defaults_to_eg(client, regime_db):
    r = client.get("/api/market_regime")
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"
    assert regime_db.seen_markets[0].code == "EG"
    for needle in ("FROM risk_snapshot", "FROM market_regime"):
        stmts = regime_db.statements(needle)
        assert stmts, needle
        assert all("market = %s" in s and "EG" in p for s, p in stmts), needle


def test_market_regime_refuses_unknown_market(client, regime_db):
    assert client.get("/api/market_regime", params={"market": "XX"}).status_code == 400
    assert regime_db.log == []


def test_market_regime_has_no_us_universe_yet(client, regime_db, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    r = client.get("/api/market_regime", params={"market": "US"})
    assert r.status_code == 200
    assert r.json() == {"market": "US", "currency": "USD", "band": None,
                        "stale": True, "n_symbols": 0}
    assert regime_db.seen_markets == []
