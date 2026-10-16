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
    # app.core.auth snapshots AUTH_SECRET at first import, and whichever test
    # module imports it first decides it for the whole run — often a short
    # 24-byte secret that makes PyJWT warn on every token. A 32+ byte secret
    # here keeps this module's ~50 mint/decode calls silent.
    monkeypatch.setattr(auth_mod, "AUTH_SECRET", "test-secret-for-market-routers-0123456789")
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    cache._store.clear()
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {create_access_token(ME.id)}"})
    yield c
    cache._store.clear()


@pytest.fixture(autouse=True)
def us_listings(monkeypatch):
    """Where each symbol trades, without the scanner or Postgres.

    Every symbol fetch asks `universe.resolve_exchange`; for a non-EG market
    that would read the DB and then call the scanner. Neither may happen in a
    test, so the resolver answers from this table (EG is always EGX) and the
    DB the routers open for it is a FakeDB.
    """
    from app.core import db as db_mod
    from app.core import universe
    table = {"AAPL": "NASDAQ", "IBM": "NYSE"}
    monkeypatch.setattr(universe, "resolve_exchange",
                        lambda db, market, symbol: "EGX" if market.code == "EG"
                        else table.get(symbol.upper()))
    monkeypatch.setattr(db_mod, "get_db", lambda: FakeDB())
    return table


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
    assert fake_multi[0][1] == {"AAPL": "NASDAQ"}, "a per-symbol exchange map"
    assert r.json()["currency"] == "USD"


# ---------------------------------------------------------------------------
# /api/analysis
# ---------------------------------------------------------------------------

@pytest.fixture
def analysis_db(monkeypatch):
    from app.routers import analysis as analysis_mod
    db = FakeDB()
    monkeypatch.setattr(analysis_mod, "get_db", lambda: db)
    monkeypatch.setattr(analysis_mod, "fetch_macro_for", lambda db, market: None)
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
    monkeypatch.setattr(dashboard_mod, "fetch_macro_for", lambda db, market: None)
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


# ===========================================================================
# Task 7: feed routers, cron and alerts
# ===========================================================================

import json  # noqa: E402

from app.core.markets import EG  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Recorded(list):
    """URLs requested, in order; `answers` holds (url substring, payload)."""
    answers: list


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Every urllib request, recorded by URL; answered from `.answers` by substring."""
    import urllib.request
    seen = _Recorded()
    seen.answers = []

    def fake(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        seen.append(url)
        for needle, payload in seen.answers:
            if needle in url:
                return _FakeResponse(payload)
        return _FakeResponse({})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


# ---------------------------------------------------------------------------
# /api/tickers
# ---------------------------------------------------------------------------

@pytest.fixture
def tickers_db(monkeypatch):
    from app.routers import tickers as tickers_mod
    db = FakeDB()
    monkeypatch.setattr(tickers_mod, "get_db", lambda: db)
    tickers_mod._tickers_cache.clear()
    yield db
    tickers_mod._tickers_cache.clear()


def test_tickers_default_to_eg(client, tickers_db, monkeypatch):
    import urllib.request
    urls = []

    def fake(req, timeout=None):
        urls.append(req.full_url)
        return _FakeResponse({"data": [{"s": "EGX:COMI", "d": ["COMI", "CIB", "Finance", "stock"]}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    r = client.get("/api/tickers")
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list), "the response shape stays a list"
    assert urls == ["https://scanner.tradingview.com/egypt/scan"]
    assert any(t["symbol"] == "COMI" for t in r.json())
    # The static EGX file is merged for EG: far more than the one live row.
    assert len(r.json()) > 50
    stmts = tickers_db.statements("FROM discovered_tickers")
    assert stmts and all("market = %s" in s and p == ("EG",) for s, p in stmts)


def test_tickers_refuse_unknown_market(client, tickers_db):
    assert client.get("/api/tickers", params={"market": "XX"}).status_code == 400
    assert tickers_db.log == []


def test_tickers_us_comes_from_the_universe_and_not_the_egx_file(client, tickers_db, monkeypatch):
    from app.core import universe
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    seen = []

    def fake_universe(db, market):
        seen.append(market.code)
        return [
            {"symbol": "AAPL", "name": "Apple", "sector": "Tech", "exchange": "NASDAQ",
             "kind": "stock", "indices": ["NDX"]},
            {"symbol": "IBM", "name": "IBM", "sector": "Tech", "exchange": "NYSE",
             "kind": "stock", "indices": ["DJI"]},
            # Resolver-remembered only (indices empty, not discovered): not listed.
            {"symbol": "VOO", "name": "Vanguard", "sector": None, "exchange": "AMEX",
             "kind": "fund", "indices": []},
        ]

    monkeypatch.setattr(universe, "load_universe", fake_universe)
    r = client.get("/api/tickers", params={"market": "US"})
    assert r.status_code == 200, r.text
    assert seen == ["US"]
    assert {t["symbol"]: t["index"] for t in r.json()} == {"AAPL": "NDX", "IBM": "DJI"}
    _, params = tickers_db.statements("FROM discovered_tickers")[0]
    assert params == ("US",)


def test_ticker_cache_is_per_market(client, tickers_db, monkeypatch):
    import urllib.request
    from app.core import universe
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    calls = []
    loads = []

    def fake(req, timeout=None):
        calls.append(req.full_url)
        return _FakeResponse({"data": []})

    def fake_universe(db, market):
        loads.append(market.code)
        return [{"symbol": "AAPL", "name": "Apple", "sector": "Tech", "exchange": "NASDAQ",
                 "kind": "stock", "indices": []}]

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr(universe, "load_universe", fake_universe)
    client.get("/api/tickers")
    client.get("/api/tickers")
    client.get("/api/tickers", params={"market": "US"})
    client.get("/api/tickers", params={"market": "US"})
    assert calls == ["https://scanner.tradingview.com/egypt/scan"]
    assert loads == ["US"]


def test_validate_looks_the_symbol_up_and_stores_the_exchange(
        client, tickers_db, monkeypatch):
    from app.core import universe
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    monkeypatch.setattr(universe, "load_universe", lambda db, m: [])
    monkeypatch.setattr(universe, "lookup_listing", lambda m, s: {
        "symbol": "IBM", "name": "IBM Corp", "sector": "Tech", "exchange": "NYSE",
        "kind": "stock"})
    r = client.get("/api/tickers", params={"market": "US", "validate": "ibm"})
    assert r.json() == {"symbol": "IBM", "valid": True, "name": "IBM Corp"}
    [(sql, params)] = tickers_db.statements("INSERT INTO discovered_tickers")
    assert "exchange" in sql and "ON CONFLICT (market, symbol)" in sql
    assert params[:2] == ("US", "IBM") and params[5] == "NYSE"


# ---------------------------------------------------------------------------
# /api/pe
# ---------------------------------------------------------------------------

@pytest.fixture
def pe_db(monkeypatch):
    from app.routers import pe as pe_mod
    db = FakeDB()
    monkeypatch.setattr(pe_mod, "get_db", lambda: db)
    return db


def test_pe_defaults_to_eg(client, pe_db):
    r = client.get("/api/pe")
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"
    stmts = pe_db.statements("FROM pe_data")
    assert stmts and all("market = %s" in s and "EG" in p for s, p in stmts)


def test_pe_refuses_unknown_market(client, pe_db):
    assert client.get("/api/pe", params={"market": "XX"}).status_code == 400
    assert pe_db.log == []


def test_pe_refresh_refreshes_eg_only(client, pe_db, monkeypatch):
    from app.routers import pe as pe_mod
    seen = []
    monkeypatch.delenv("PE_REFRESH_SECRET", raising=False)
    monkeypatch.setattr(pe_mod, "refresh_pe_data",
                        lambda db, market: seen.append(("pe", market.code)) or {})
    monkeypatch.setattr(pe_mod, "refresh_annual_fundamentals",
                        lambda db, market: seen.append(("annual", market.code)) or {})
    monkeypatch.setattr(pe_mod, "refresh_macro_series", lambda db: {})
    monkeypatch.setattr(pe_mod, "prune_auth", lambda *a: {})
    assert client.post("/api/pe/refresh").status_code == 200
    assert seen == [("pe", "EG"), ("annual", "EG")]


def test_pe_refresh_writes_rows_for_its_market():
    from app.core import pe_fetch
    db = FakeDB()
    rows = [{"symbol": f"S{i}", "pe_ratio": 10.0, "dividend_yield": 1.0,
             "loss_making": False} for i in range(pe_fetch.MIN_EXPECTED_ROWS)]
    assert pe_fetch.refresh_pe_data(db, US, rows=rows)["success"] is True
    inserts = db.statements("INSERT INTO pe_data")
    assert inserts and all(p[0] == "US" for _, p in inserts)
    assert db.statements("fundamentals_history") == [], \
        "fundamentals_history has no market column; only EG feeds it"


def test_annual_fundamentals_write_rows_for_their_market():
    from app.core import fundamentals_annual as fa
    db = FakeDB()
    rows = [{"symbol": f"S{i}", "fiscal_year": 2024, "eps_diluted": 1.0, "dps": None,
             "net_income": None, "revenue": None, "total_assets": None,
             "gross_profit": None, "total_debt": None, "first_usable_date": "2025-04-30"}
            for i in range(fa.MIN_EXPECTED_SYMBOLS)]
    assert fa.refresh_annual_fundamentals(db, US, rows=rows)["success"] is True
    [(sql, params)] = db.statements("INSERT INTO fundamentals_annual")
    assert params[0] == "US" and params[12] == "US"


# ---------------------------------------------------------------------------
# /api/news
# ---------------------------------------------------------------------------

@pytest.fixture
def news_env(monkeypatch):
    from app.core import news_fetch
    from app.routers import news as news_mod
    db = FakeDB()
    db.rows_for("FROM watchlist", [("COMI",)])
    urls, af_calls = [], []

    def fake_get_json(url, timeout):
        urls.append(url)
        return {"items": []}

    monkeypatch.setattr(news_fetch, "_get_json", fake_get_json)
    monkeypatch.setattr(news_mod, "get_db", lambda: db)
    monkeypatch.setattr(news_mod, "fetch_arabfinance",
                        lambda: af_calls.append(1) or ([], None))
    db.urls, db.af_calls = urls, af_calls
    return db


def test_news_defaults_to_eg(client, news_env):
    r = client.get("/api/news")
    assert r.status_code == 200, r.text
    body = r.json()
    # `market` is the market-news STORY LIST the web and mobile pages read;
    # the market code must not overwrite it.
    assert isinstance(body["market"], list)
    assert body["market_code"] == "EG" and body["currency"] == "EGP"
    assert any(u.endswith("symbol=EGX%3ACOMI") for u in news_env.urls)
    assert len(news_env.urls) > 1, "the EGX30 market half was not fetched"
    assert news_env.af_calls == [1]
    for needle in ("FROM watchlist", "FROM portfolio"):
        [(sql, params)] = news_env.statements(needle)
        assert "market = %s" in sql and params == ("u1", "EG"), needle


def test_news_refuses_unknown_market(client, news_env):
    assert client.get("/api/news", params={"market": "XX"}).status_code == 400
    assert news_env.urls == [] and news_env.log == []


def test_news_us_uses_the_us_exchange_and_skips_egyptian_sources(
        client, news_env, monkeypatch):
    from app.routers import news as news_mod
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    monkeypatch.setattr(news_mod.universe, "resolve_exchange",
                        lambda db, m, s: "NASDAQ" if s == "COMI" else None)
    r = client.get("/api/news", params={"market": "US"})
    assert r.status_code == 200, r.text
    assert news_env.urls == [
        "https://news-headlines.tradingview.com/v2/headlines"
        "?client=web&lang=en&symbol=NASDAQ%3ACOMI"
    ], "only the caller's own symbol: no EGX30 half for US"
    assert news_env.af_calls == [], "Arab Finance is Egyptian press"
    _, params = news_env.statements("FROM watchlist")[0]
    assert params == ("u1", "US")


# ---------------------------------------------------------------------------
# /api/dividend_history, /api/dividend_calendar
# ---------------------------------------------------------------------------

@pytest.fixture
def div_db(monkeypatch):
    from app.routers import dividend_history as div_mod
    db = FakeDB()
    monkeypatch.setattr(div_mod, "get_db", lambda: db)
    return db


def test_dividend_history_defaults_to_eg_and_yahoo_ca(client, div_db, fake_urlopen):
    r = client.get("/api/dividend_history", params={"symbol": "COMI"})
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"
    [url] = fake_urlopen
    assert "/v8/finance/chart/COMI.CA?" in url
    [(sql, params)] = div_db.statements("FROM dividend_events")
    assert "market = %s" in sql and params == ("EG", "COMI")


def test_dividend_history_us_has_no_yahoo_suffix(client, div_db, fake_urlopen, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    fake_urlopen.answers.append(("AAPL", {"chart": {"result": [{"events": {"dividends": {
        "1": {"amount": 0.25, "date": 1_700_000_000}}}}]}}))
    r = client.get("/api/dividend_history", params={"symbol": "AAPL", "market": "US"})
    assert r.status_code == 200, r.text
    assert r.json()["currency"] == "USD"
    [url] = fake_urlopen
    assert "/v8/finance/chart/AAPL?" in url
    [(sql, params)] = div_db.statements("INSERT INTO dividend_events")
    assert params[:2] == ("US", "AAPL")


def test_dividend_routes_refuse_unknown_market(client, div_db, fake_urlopen):
    for path, params in (("/api/dividend_history", {"symbol": "COMI"}),
                         ("/api/dividend_calendar", {})):
        r = client.get(path, params={**params, "market": "XX"})
        assert r.status_code == 400, path
    assert div_db.log == [] and list(fake_urlopen) == []


def test_dividend_calendar_defaults_to_eg(client, div_db):
    r = client.get("/api/dividend_calendar")
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG"
    for needle in ("FROM dividend_events", "FROM pe_data"):
        stmts = div_db.statements(needle)
        assert stmts and all("market = %s" in s and p[0] == "EG" for s, p in stmts), needle


# ---------------------------------------------------------------------------
# /api/macro, /api/calibration
# ---------------------------------------------------------------------------

@pytest.fixture
def macro_seen(monkeypatch):
    from app.routers import macro as macro_mod
    seen = []
    monkeypatch.setattr(macro_mod, "get_db", lambda: FakeDB())
    monkeypatch.setattr(macro_mod, "fetch_macro", lambda db: seen.append(1) or None)
    return seen


def test_macro_defaults_to_eg(client, macro_seen):
    r = client.get("/api/macro")
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"
    assert macro_seen == [1]


def test_macro_refuses_unknown_market(client, macro_seen):
    assert client.get("/api/macro", params={"market": "XX"}).status_code == 400


def test_macro_is_eg_only_for_now(client, macro_seen, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    r = client.get("/api/macro", params={"market": "US"})
    assert r.json() == {"market": "US", "currency": "USD",
                        "egx30": None, "usd_egp": None, "interest_rate": None}
    assert macro_seen == [], "Egyptian macro was fetched for a US request"


def test_calibration_defaults_to_eg(client):
    r = client.get("/api/calibration")
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG" and r.json()["currency"] == "EGP"


def test_calibration_refuses_unknown_and_non_eg_markets(client, monkeypatch):
    assert client.get("/api/calibration", params={"market": "XX"}).status_code == 400
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    r = client.get("/api/calibration", params={"market": "US"})
    assert r.status_code == 400
    assert r.json()["detail"] == "Calibration is EGX-only until US is calibrated"


# ---------------------------------------------------------------------------
# /api/risk response fields
# ---------------------------------------------------------------------------

def test_risk_reports_the_market_liquidity_floor(client, risk_db):
    risk_db.rows_for("FROM risk_snapshot", [
        ("COMI", "2026-10-14", 30.0, 29.0, 1.0, 5e6, 1.0, 80.0, True)])
    body = client.get("/api/risk").json()
    assert body["liquidity_floor"] == EG.liquid_min_turnover
    assert body["currency"] == "EGP"
    assert "liquidity_floor_egp" not in body


# ---------------------------------------------------------------------------
# POST /api/cron/risk_snapshot
# ---------------------------------------------------------------------------

@pytest.fixture
def cron_db(monkeypatch):
    from app.routers import cron as cron_mod
    db = FakeDB()
    alerts = []
    monkeypatch.delenv("CRON_SECRET", raising=False)
    monkeypatch.setattr(cron_mod, "get_db", lambda: db)
    monkeypatch.setattr(cron_mod, "read_risk_free_rate", lambda db, market: 19.0)
    monkeypatch.setattr(cron_mod, "run_alerts",
                        lambda *a, **k: alerts.append(k.get("market_code")) or {})
    db.alert_markets = alerts
    return db


def test_cron_defaults_to_eg(client, fake_feed, cron_db):
    r = client.post("/api/cron/risk_snapshot", params={"limit": 1})
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "EG"
    measured = [c for c in fake_feed if c[0] != EG.benchmark_symbol]
    assert measured and all(ex == "EGX" for _, ex in measured)
    for needle in ("risk_snapshot", "pe_data", "score_baseline"):
        stmts = cron_db.statements(needle)
        assert stmts, needle
        assert all("market" in s and "EG" in p for s, p in stmts), needle
    assert cron_db.alert_markets == ["EG"]


def test_cron_refuses_unknown_market(client, fake_feed, cron_db):
    r = client.post("/api/cron/risk_snapshot", params={"market": "XX"})
    assert r.status_code == 400
    assert fake_feed == [] and cron_db.log == []


def test_cron_universe_drops_isins_through_the_market():
    from app.routers.cron import _universe
    universe = _universe(EG)
    assert universe and all(EG.is_valid_symbol(s) for s in universe)


# ---------------------------------------------------------------------------
# Score alerts: subscribers and the log are per market
# ---------------------------------------------------------------------------

def test_alert_subscribers_follow_only_the_markets_own_lists():
    from app.core.alert_store import load_subscribers
    db = FakeDB([("FROM alert_preferences", [("u1", "watchlist", 10, 10)]),
                 ("FROM watchlist", [("u1", "AAPL")])])
    out = load_subscribers(db, ["AAPL"], "US")
    assert [s.user_id for s in out["AAPL"]] == ["u1"]
    [(sql, params)] = db.statements("FROM watchlist")
    halves = sql.split("UNION")
    assert len(halves) == 2 and all("market = %s" in h for h in halves)
    assert params[0] == "US" and params[3] == "US"


def test_alert_log_rows_carry_the_market():
    from app.core.alert_store import run_detection
    from app.core.composite import CATEGORY_ORDER
    before = tuple([40.0] * len(CATEGORY_ORDER))
    after = {c: 70.0 for c in CATEGORY_ORDER}
    db = FakeDB([
        ("FROM alert_preferences", [("u1", "all", 10, 10)]),
        ("FROM score_baseline", [("AAPL", "2026-10-14") + before]),
        ("INSERT INTO alert_log", [("id1",)]),
    ])
    n = run_detection(db, {"AAPL": {"session_date": "2026-10-14", "categories": after}},
                      None, "now", new_id=lambda: "id1", market_code="US")
    assert n == 1
    [(sql, params)] = db.statements("INSERT INTO alert_log")
    assert "ON CONFLICT (user_id, market, symbol, session_date, direction)" in sql
    assert params[2] == "US"
    for needle in ("FROM score_baseline", "SELECT user_id, symbol, direction FROM alert_log"):
        [(sql, params)] = db.statements(needle)
        assert "market = %s" in sql and "US" in params, needle


def test_alert_quiet_hours_use_the_eg_clock():
    from datetime import datetime, timezone
    from app.core import constants
    from app.core.score_alerts import in_quiet_hours
    assert constants.ALERT_TIMEZONE == EG.timezone
    # 20:30 UTC is 23:30 Cairo (summer) — quiet.
    assert in_quiet_hours(datetime(2026, 7, 1, 20, 30, tzinfo=timezone.utc))


def test_default_risk_free_rate_is_egs_registry_fallback():
    from app.core.constants import DEFAULT_RISK_FREE_RATE_PCT
    assert DEFAULT_RISK_FREE_RATE_PCT == EG.risk_free_fallback_pct == 19


# ---------------------------------------------------------------------------
# Per-user data is scoped by market: watchlist, portfolio, sales, dividends,
# portfolio_analysis. Handlers are called directly with a FakeDB.
# ---------------------------------------------------------------------------

import app.routers.watchlist as watchlist_mod  # noqa: E402
import app.routers.portfolio as portfolio_mod  # noqa: E402
import app.routers.sales as sales_mod  # noqa: E402
import app.routers.dividends as dividends_mod  # noqa: E402


def _patched_db(monkeypatch, mod, responses=None):
    db = FakeDB(responses)
    monkeypatch.setattr(mod, "get_db", lambda: db)
    return db


def test_watchlist_is_scoped_by_market(monkeypatch):
    db = _patched_db(monkeypatch, watchlist_mod)
    out = watchlist_mod.get_watchlist(market=EG, user=ME)
    [(sql, params)] = db.statements("SELECT symbol FROM watchlist")
    assert "market = %s" in sql and params == ("u1", "EG")
    assert out["market"] == "EG"


def test_watchlist_add_writes_market_and_new_conflict_target(monkeypatch):
    db = _patched_db(monkeypatch, watchlist_mod)
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    out = watchlist_mod.add_to_watchlist(
        watchlist_mod.WatchlistAdd(symbol="aapl", market="US"), market=EG, user=ME)
    [(sql, params)] = db.statements("INSERT INTO watchlist")
    assert "ON CONFLICT (user_id, market, symbol)" in sql
    assert params[:3] == ("u1", "US", "AAPL")
    assert out["market"] == "US"


def test_watchlist_add_without_body_market_uses_the_query_market(monkeypatch):
    db = _patched_db(monkeypatch, watchlist_mod)
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    out = watchlist_mod.add_to_watchlist(
        watchlist_mod.WatchlistAdd(symbol="aapl"), market=US, user=ME)
    [(_, params)] = db.statements("INSERT INTO watchlist")
    assert params[1] == "US" and out["market"] == "US"


def test_watchlist_add_refuses_a_disabled_body_market(monkeypatch):
    from fastapi import HTTPException
    db = _patched_db(monkeypatch, watchlist_mod)
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    with pytest.raises(HTTPException) as e:
        watchlist_mod.add_to_watchlist(
            watchlist_mod.WatchlistAdd(symbol="aapl", market="US"), market=EG, user=ME)
    assert e.value.status_code == 400
    assert db.statements("INSERT INTO watchlist") == []


def test_watchlist_delete_is_scoped(monkeypatch):
    db = _patched_db(monkeypatch, watchlist_mod)
    watchlist_mod.remove_from_watchlist(symbol="COMI", market=EG, user=ME)
    [(sql, params)] = db.statements("DELETE FROM watchlist")
    assert "market = %s" in sql and params == ("u1", "EG", "COMI")


def test_portfolio_list_is_scoped_by_market(monkeypatch):
    db = _patched_db(monkeypatch, portfolio_mod)
    portfolio_mod.get_portfolio(market=US, user=ME)
    [(sql, params)] = db.statements("FROM portfolio")
    assert "market = %s" in sql and params == ("u1", "US")


def test_portfolio_currency_comes_from_market_not_settings(monkeypatch):
    db = _patched_db(monkeypatch, portfolio_mod)
    out = portfolio_mod.get_portfolio(market=EG, user=ME)
    assert out["currency"] == "EGP" and out["market"] == "EG"
    assert db.statements("SELECT value FROM settings WHERE key = 'currency'") == []
    out = portfolio_mod.get_portfolio(market=US, user=ME)
    assert out["currency"] == "USD" and out["market"] == "US"


def test_portfolio_create_writes_the_body_market(monkeypatch):
    db = _patched_db(monkeypatch, portfolio_mod)
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    out = portfolio_mod.add_holding(
        {"symbol": "aapl", "buy_price": 10, "quantity": 2, "market": "US"},
        market=EG, user=ME)
    [(sql, params)] = db.statements("INSERT INTO portfolio")
    assert "market" in sql.split("VALUES")[0]
    assert params[1:4] == ("u1", "US", "AAPL")
    assert out["market"] == "US"


def test_portfolio_create_defaults_to_the_query_market(monkeypatch):
    db = _patched_db(monkeypatch, portfolio_mod)
    out = portfolio_mod.add_holding(
        {"symbol": "comi", "buy_price": 10, "quantity": 2}, market=EG, user=ME)
    [(_, params)] = db.statements("INSERT INTO portfolio")
    assert params[2] == "EG" and out["market"] == "EG"


_LOT = ("h1", "AAPL", "Apple", 100.0, "2026-01-05", 10, "", "Tech",
        None, None, "2026-01-05T00:00:00Z", "2026-01-05T00:00:00Z")


def test_sale_takes_the_holdings_market_not_the_bodys(monkeypatch):
    db = _patched_db(monkeypatch, sales_mod, [
        ("WHERE id = %s AND user_id = %s", [_LOT + ("US",)]),
        ("AND quantity > 0", [_LOT]),
        ("RETURNING quantity", [(6,)]),
    ])
    monkeypatch.setattr(sales_mod, "get_risk_free_steps", lambda db: [])
    out = sales_mod.record_sale(
        {"holding_id": "h1", "quantity": 4, "sell_price": 120, "sell_date": "2026-02-01",
         "market": "EG"},
        user=ME)
    [(sql, params)] = db.statements("AND quantity > 0")
    assert "market = %s" in sql and params == ("u1", "AAPL", "US")
    [(sql, params)] = db.statements("INSERT INTO portfolio_sales")
    assert "market" in sql.split("VALUES")[0] and params[-1] == "US"
    assert out["market"] == "US" and out["currency"] == "USD"


def test_sales_ledger_is_scoped_by_market_with_market_currency(monkeypatch):
    db = _patched_db(monkeypatch, sales_mod)
    monkeypatch.setattr(sales_mod, "get_risk_free_steps", lambda db: [])
    out = sales_mod.get_sales(market=US, user=ME)
    for needle in ("FROM portfolio_sales", "FROM portfolio_dividends"):
        [(sql, params)] = db.statements(needle)
        assert "market = %s" in sql and params == ("u1", "US"), needle
    assert out["market"] == "US" and out["currency"] == "USD"
    assert db.statements("WHERE key = 'currency'") == []


def test_dividend_create_writes_market_and_dedupes_within_it(monkeypatch):
    db = _patched_db(monkeypatch, dividends_mod)
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    out = dividends_mod.record_dividend(
        {"symbol": "aapl", "amount": 5, "pay_date": "2026-02-01", "market": "US"},
        market=EG, user=ME)
    [(sql, params)] = db.statements("FROM portfolio_dividends")
    assert "market = %s" in sql and params == ("u1", "US")
    [(sql, params)] = db.statements("INSERT INTO portfolio_dividends")
    assert "market" in sql.split("VALUES")[0] and params[1:3] == ("u1", "US")
    assert out["market"] == "US" and out["currency"] == "USD"


def test_dividend_duplicate_message_names_the_markets_currency(monkeypatch):
    from fastapi import HTTPException
    dup = ("d1", "AAPL", "Apple", "", 5.0, "2026-02-01", None, "", "2026-02-01T00:00:00Z")
    _patched_db(monkeypatch, dividends_mod, [("FROM portfolio_dividends", [dup])])
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    with pytest.raises(HTTPException) as e:
        dividends_mod.record_dividend(
            {"symbol": "AAPL", "amount": 5, "pay_date": "2026-02-01"}, market=US, user=ME)
    assert e.value.status_code == 409 and "USD" in e.value.detail


@pytest.fixture
def pa_env(monkeypatch, fake_feed):
    import app.routers.portfolio_analysis as pa_mod
    from app.core import benchmark as bench_mod
    db = FakeDB()
    benchmarks = []
    monkeypatch.setattr(pa_mod, "get_db", lambda: db)
    monkeypatch.setattr(pa_mod, "fetch_macro_for", lambda db, market: None)
    monkeypatch.setattr(bench_mod, "get_benchmark_close",
                        lambda m, interval, bars: benchmarks.append(m.code) or None)
    db.benchmarks = benchmarks
    db.feed = fake_feed
    return db


def test_portfolio_analysis_defaults_to_eg(client, pa_env):
    pa_env.rows_for("FROM portfolio WHERE", [_LOT[:1] + ("COMI",) + _LOT[2:]])
    r = client.get("/api/portfolio_analysis")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["market"] == "EG" and body["currency"] == "EGP"
    [(sql, params)] = pa_env.statements("FROM portfolio WHERE")
    assert "market = %s" in sql and params == ("u1", "EG")
    assert pa_env.benchmarks == ["EG"]
    assert pa_env.feed == [("COMI", "EGX")]
    for needle in ("FROM pe_data", "FROM portfolio_dividends"):
        stmts = pa_env.statements(needle)
        assert stmts and all("market = %s" in s and "EG" in p for s, p in stmts), needle


def test_portfolio_analysis_us_uses_us_exchange_benchmark_and_pe(client, pa_env, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    pa_env.rows_for("FROM portfolio WHERE", [_LOT])
    r = client.get("/api/portfolio_analysis", params={"market": "US"})
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "US" and r.json()["currency"] == "USD"
    assert pa_env.benchmarks == ["US"]
    assert pa_env.feed == [("AAPL", "NASDAQ")]
    [(_, params)] = pa_env.statements("FROM pe_data")
    assert "US" in params


def test_portfolio_analysis_refuses_unknown_market(client, pa_env):
    assert client.get("/api/portfolio_analysis", params={"market": "XX"}).status_code == 400
    assert client.post("/api/portfolio_analysis", params={"market": "XX"},
                       json={"portfolio": []}).status_code == 400


@pytest.mark.parametrize("bad", [5, ["US"], {"code": "US"}])
def test_portfolio_and_dividend_create_refuse_a_non_string_body_market(monkeypatch, bad):
    from fastapi import HTTPException
    for mod, fn, body in (
        (portfolio_mod, portfolio_mod.add_holding,
         {"symbol": "comi", "buy_price": 10, "quantity": 2, "market": bad}),
        (dividends_mod, dividends_mod.record_dividend,
         {"symbol": "comi", "amount": 5, "pay_date": "2026-01-05", "shares": 10,
          "market": bad}),
    ):
        db = _patched_db(monkeypatch, mod)
        with pytest.raises(HTTPException) as e:
            fn(body, market=EG, user=ME)
        assert e.value.status_code == 400, (mod.__name__, bad)
        assert db.statements("INSERT INTO") == []
