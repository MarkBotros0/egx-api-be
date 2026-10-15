"""
Every symbol-level price fetch asks `universe.resolve_exchange` which exchange
the symbol trades on, instead of assuming the market's first exchange. KO on
NASDAQ waits ~5.5s and returns nothing; KO on NYSE answers.

Run from egx-api-be:  python -m pytest tests/test_exchange_resolution.py -v

Fixtures are copied from tests/test_market_routers.py (not imported), per the
plan, so the two files can change independently.
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
from app.core import universe as U  # noqa: E402
from app.core.auth import CurrentUser, create_access_token  # noqa: E402
from app.main import app  # noqa: E402

ME = CurrentUser(id="u1", email="sara@example.com", role="user")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth_mod, "_load_user", lambda uid: ME)
    monkeypatch.setattr(auth_mod, "AUTH_SECRET", "test-secret-for-market-routers-0123456789")
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    cache._store.clear()
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {create_access_token(ME.id)}"})
    yield c
    cache._store.clear()


@pytest.fixture(autouse=True)
def fake_db(monkeypatch):
    """No route here may reach Postgres: the DB is a FakeDB everywhere."""
    from app.core import db as db_mod
    from app.routers import analysis as analysis_mod
    from app.routers import portfolio_analysis as pa_mod
    db = FakeDB()
    monkeypatch.setattr(db_mod, "get_db", lambda: db)
    monkeypatch.setattr(analysis_mod, "get_db", lambda: db)
    monkeypatch.setattr(analysis_mod, "fetch_macro_for", lambda db, market: None)
    monkeypatch.setattr(pa_mod, "get_db", lambda: db)
    monkeypatch.setattr(pa_mod, "fetch_macro_for", lambda db, market: None)
    return db


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
        calls.append(("daily", exchange, holiday_country, list(syms)))
        return frame(syms)

    def intraday(syms, interval, start, end, exchange="EGX"):
        calls.append(("intraday", exchange, list(syms)))
        return frame(syms)

    import app.vendor.egxpy as egxpy
    monkeypatch.setattr(egxpy, "get_EGXdata", daily)
    monkeypatch.setattr(egxpy, "get_EGX_intraday_data", intraday)
    return calls


@pytest.fixture
def resolver(monkeypatch):
    table = {"KO": "NYSE", "AAPL": "NASDAQ", "BRK.B": "NYSE"}
    asked = []

    def fake(db, market, symbol):
        asked.append((market.code, symbol))
        return "EGX" if market.code == "EG" else table.get(symbol.upper())

    monkeypatch.setattr(U, "resolve_exchange", fake)
    U.clear_caches()
    table_with_log = dict(table)
    table_with_log["_asked"] = asked
    return table_with_log


@pytest.fixture
def us(monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")


# ---------------------------------------------------------------------------
# single-symbol routes
# ---------------------------------------------------------------------------

def test_us_ohlcv_uses_resolved_exchange(client, fake_feed, resolver, us):
    r = client.get("/api/ohlcv?symbol=KO&market=US")
    assert r.status_code == 200
    assert fake_feed[-1] == ("KO", "NYSE")


def test_us_unknown_symbol_404_without_fetch(client, fake_feed, resolver, us):
    r = client.get("/api/analysis?symbol=ZZZZQ&market=US")
    assert r.status_code == 404 and r.json()["detail"] == "Unknown symbol: ZZZZQ"
    assert fake_feed == []


def test_us_ohlcv_unknown_symbol_404(client, fake_feed, resolver, us):
    r = client.get("/api/ohlcv?symbol=ZZZZQ&market=US")
    assert r.status_code == 404 and r.json() == {"detail": "Unknown symbol: ZZZZQ"}
    assert fake_feed == []


def test_us_analysis_uses_resolved_exchange(client, fake_feed, resolver, us):
    r = client.get("/api/analysis?symbol=KO&market=US&bars=60")
    assert r.status_code == 200, r.text
    assert ("KO", "NYSE") in fake_feed


def test_explicit_exchange_override_wins(client, fake_feed, resolver, us):
    client.get("/api/ohlcv?symbol=KO&market=US&exchange=nasdaq")
    assert fake_feed[-1] == ("KO", "NASDAQ")
    assert resolver["_asked"] == [], "an explicit exchange needs no lookup"


def test_eg_still_egx(client, fake_feed, resolver):
    client.get("/api/ohlcv?symbol=COMI")
    assert fake_feed[-1] == ("COMI", "EGX")


# ---------------------------------------------------------------------------
# multi-symbol routes
# ---------------------------------------------------------------------------

def test_historical_passes_per_symbol_map(client, fake_multi, resolver, us):
    r = client.get("/api/historical?symbols=KO,AAPL,ZZZZQ&market=US")
    assert r.status_code == 200
    assert fake_multi[-1][1] == {"KO": "NYSE", "AAPL": "NASDAQ"}
    assert fake_multi[-1][3] == ["KO", "AAPL"], "unresolved symbols are dropped"
    assert r.json()["symbols"] == ["KO", "AAPL"]


def test_compare_passes_per_symbol_map(client, fake_multi, resolver, us):
    r = client.get("/api/compare?symbols=KO,AAPL,ZZZZQ&market=US")
    assert r.status_code == 200
    assert fake_multi[-1][1] == {"KO": "NYSE", "AAPL": "NASDAQ"}


def test_intraday_passes_per_symbol_map(client, fake_multi, resolver, us):
    r = client.get("/api/intraday", params={"symbols": "KO,ZZZZQ", "market": "US",
                                            "interval": "5 Minute"})
    assert r.status_code == 200
    assert fake_multi[-1][1] == {"KO": "NYSE"}


@pytest.mark.parametrize("path, params", [
    ("/api/historical", {"symbols": "ZZZZQ,QQQQZ"}),
    ("/api/compare", {"symbols": "ZZZZQ,QQQQZ"}),
    ("/api/intraday", {"symbols": "ZZZZQ", "interval": "5 Minute"}),
])
def test_multi_symbol_404_only_when_none_resolve(client, fake_multi, resolver, us,
                                                  path, params):
    r = client.get(path, params={**params, "market": "US"})
    assert r.status_code == 404
    assert r.json()["detail"].startswith("Unknown symbol: ")
    assert fake_multi == []


@pytest.mark.parametrize("path, params", [
    ("/api/historical", {"symbols": "COMI,ETEL"}),
    ("/api/compare", {"symbols": "COMI,ETEL"}),
    ("/api/intraday", {"symbols": "COMI", "interval": "5 Minute"}),
])
def test_eg_multi_symbol_still_passes_the_plain_string(client, fake_multi, resolver,
                                                         path, params):
    r = client.get(path, params=params)
    assert r.status_code == 200
    assert fake_multi[-1][1] == "EGX"


# ---------------------------------------------------------------------------
# batch, portfolio, news
# ---------------------------------------------------------------------------

def test_batch_marks_unknown(client, fake_feed, resolver, us):
    r = client.get("/api/analysis?mode=batch&symbols=KO,ZZZZQ&market=US")
    assert r.status_code == 200, r.text
    body = r.json()
    assert ("KO", "NYSE") in fake_feed
    assert all(s != "ZZZZQ" for s, _ in fake_feed)
    assert {"symbol": "ZZZZQ", "error": "unknown symbol"} in body["errors"]
    assert "KO" in body["scores"]


_LOT = ("h1", "KO", "Coca-Cola", 50.0, "2025-01-01", 2, None, "Staples", None, None,
        "2025-01-01", "2025-01-01")


def test_portfolio_analysis_resolves_each_holding(client, fake_feed, resolver, us,
                                                   fake_db, monkeypatch):
    from app.core import benchmark as bench_mod
    monkeypatch.setattr(bench_mod, "get_benchmark_close", lambda m, interval, bars: None)
    unknown = ("h2",) + ("ZZZZQ",) + _LOT[2:]
    fake_db.rows_for("FROM portfolio WHERE", [_LOT, unknown])
    r = client.get("/api/portfolio_analysis", params={"market": "US"})
    assert r.status_code == 200, r.text
    assert fake_feed == [("KO", "NYSE")]
    errs = [h for h in r.json()["holdings"] if h.get("symbol") == "ZZZZQ"]
    assert errs and errs[0]["error"] == "Could not fetch market data"


def test_news_asks_each_symbols_own_exchange(client, resolver, us, monkeypatch):
    from app.core import news_fetch
    from app.routers import news as news_mod
    db = FakeDB()
    db.rows_for("FROM watchlist", [("KO",), ("ZZZZQ",)])
    urls = []
    monkeypatch.setattr(news_fetch, "_get_json",
                        lambda url, timeout: urls.append(url) or {"items": []})
    monkeypatch.setattr(news_mod, "get_db", lambda: db)
    monkeypatch.setattr(news_mod, "fetch_arabfinance", lambda: ([], None))
    r = client.get("/api/news", params={"market": "US"})
    assert r.status_code == 200, r.text
    assert sorted(u.rsplit("symbol=", 1)[1] for u in urls) == [
        "NASDAQ%3AZZZZQ",      # unresolved: the market's first exchange
        "NYSE%3AKO",
    ]
