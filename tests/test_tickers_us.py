"""
Tickers API for non-EG markets: universe-backed list, index membership filter,
scanner-backed validate and search. No network or Postgres is touched.
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from alert_fakes import FakeDB  # noqa: E402
from app.core import auth as auth_mod  # noqa: E402
from app.core import cache  # noqa: E402
from app.core import universe as U  # noqa: E402
from app.core.auth import CurrentUser, create_access_token  # noqa: E402
from app.main import app  # noqa: E402
import app.routers.tickers as T  # noqa: E402

ME = CurrentUser(id="u1", email="sara@example.com", role="user")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth_mod, "_load_user", lambda uid: ME)
    monkeypatch.setattr(auth_mod, "AUTH_SECRET", "test-secret-for-tickers-us-0123456789abcdef")
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    cache._store.clear()
    T._tickers_cache.clear()
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {create_access_token(ME.id)}"})
    yield c
    cache._store.clear()
    T._tickers_cache.clear()


UNIV = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Electronic Technology", "exchange": "NASDAQ", "kind": "stock", "indices": ["SPX", "NDX", "DJI"]},
    {"symbol": "KO", "name": "Coca-Cola Company", "sector": "Consumer Non-Durables", "exchange": "NYSE", "kind": "stock", "indices": ["SPX", "DJI"]},
    {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "Miscellaneous", "exchange": "AMEX", "kind": "fund", "indices": []},
]


@pytest.fixture
def us(monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    monkeypatch.setattr(U, "load_universe", lambda db, m: UNIV if m.code == "US" else [])
    monkeypatch.setattr(T, "get_db", lambda: FakeDB())
    T._tickers_cache.clear()


def test_us_list_has_indices(client, us):
    rows = client.get("/api/tickers?market=US").json()
    by = {r["symbol"]: r for r in rows}
    assert by["AAPL"]["indices"] == ["SPX", "NDX", "DJI"] and by["AAPL"]["index"] == "SPX"
    assert by["KO"]["exchange"] == "NYSE"
    # VOO is only in the universe because the resolver placed it once (someone
    # OPENED it): not an index member, not validated/added -> not listed.
    assert "VOO" not in by


def test_us_list_keeps_a_resolved_symbol_once_it_is_discovered(client, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    monkeypatch.setattr(U, "load_universe", lambda db, m: UNIV)
    db = FakeDB()
    db.rows_for("discovered_tickers", [("VOO", "Vanguard", None, "AMEX")])
    monkeypatch.setattr(T, "get_db", lambda: db)
    T._tickers_cache.clear()
    rows = client.get("/api/tickers?market=US").json()
    assert [r["symbol"] for r in rows] == ["AAPL", "KO", "VOO"]
    voo = rows[-1]
    assert voo["index"] == "AMEX" and voo["indices"] == [] and voo["name"] == "Vanguard S&P 500 ETF"


def test_us_index_filter_uses_membership(client, us):
    syms = [r["symbol"] for r in client.get("/api/tickers?market=US&index=dji").json()]
    assert syms == ["AAPL", "KO"]


def test_us_validate_uses_scanner_lookup(client, us, monkeypatch):
    monkeypatch.setattr(U, "lookup_listing", lambda m, s: {"symbol": "BRK.B", "name": "Berkshire", "sector": "Finance", "exchange": "NYSE", "kind": "stock"} if s == "BRK.B" else None)
    assert client.get("/api/tickers?validate=brk.b&market=US").json() == {"symbol": "BRK.B", "valid": True, "name": "Berkshire"}
    assert client.get("/api/tickers?validate=ZZZZQ&market=US").json()["valid"] is False


def test_us_validate_scanner_down_is_unknown(client, us, monkeypatch):
    def boom(m, s): raise OSError("down")
    monkeypatch.setattr(U, "lookup_listing", boom)
    assert client.get("/api/tickers?validate=XYZ&market=US").json()["valid"] is None


def test_search_us(client, us, monkeypatch):
    monkeypatch.setattr(U, "search_symbols", lambda m, q, limit=20: [{"symbol": "KO", "name": "Coca-Cola Company", "sector": "X", "exchange": "NYSE", "kind": "stock"}])
    r = client.get("/api/tickers/search?q=coca&market=US")
    assert r.status_code == 200 and r.json()[0]["symbol"] == "KO"


def test_search_us_scanner_down_502(client, us, monkeypatch):
    def boom(m, q, limit=20): raise OSError("down")
    monkeypatch.setattr(U, "search_symbols", boom)
    r = client.get("/api/tickers/search?q=coca&market=US")
    assert r.status_code == 502 and r.json() == {"detail": "Search unavailable"}


def test_search_eg_filters_local(client, monkeypatch):
    monkeypatch.setattr(T, "_load_tickers", lambda m: [{"symbol": "COMI", "name": "Commercial International Bank", "sector": "Banking", "index": "EGX30", "indices": ["EGX30"]}])
    rows = client.get("/api/tickers/search?q=commercial").json()
    assert [r["symbol"] for r in rows] == ["COMI"]


def test_eg_rows_carry_indices(client, monkeypatch):
    monkeypatch.setattr(T, "_fetch_live_tickers", lambda m: None)
    monkeypatch.setattr(T, "get_db", lambda: FakeDB())
    T._tickers_cache.clear()
    rows = client.get("/api/tickers").json()
    assert all(r["indices"] == [r["index"]] for r in rows)


def test_search_exchange_continues_after_error(monkeypatch):
    calls = []
    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "exchange=NASDAQ" in req.full_url:
            raise OSError("down")
        class R:
            def read(self): return b'[{"symbol":"KO","exchange":"NYSE","description":"Coca-Cola"}]'
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R()
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    from app.core.markets import US
    match, ex = T._search_exchange(US, "KO")
    assert ex == "NYSE" and len(calls) == 2


def test_empty_us_universe_is_not_cached(client, us, monkeypatch):
    monkeypatch.setattr(U, "load_universe", lambda db, m: [])
    assert client.get("/api/tickers?market=US").json() == []
    assert "US" not in T._tickers_cache


def test_us_discovered_rows_appended_without_duplicates(client, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    monkeypatch.setattr(U, "load_universe", lambda db, m: UNIV)
    db = FakeDB()
    db.rows_for("discovered_tickers", [("AAPL", "Apple", None, "NASDAQ"),
                                       ("BRK.B", "Berkshire", "Finance", "NYSE"),
                                       ("XXX", "NoExch", None, None)])
    monkeypatch.setattr(T, "get_db", lambda: db)
    T._tickers_cache.clear()
    by = {r["symbol"]: r for r in client.get("/api/tickers?market=US").json()}
    assert by["BRK.B"]["index"] == "NYSE" and by["BRK.B"]["indices"] == []
    assert "XXX" not in by and by["AAPL"]["indices"] == ["SPX", "NDX", "DJI"]


def test_universe_outage_with_discovered_rows_is_not_cached(client, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    db = FakeDB()
    db.rows_for("discovered_tickers", [("BRK.B", "Berkshire", "Finance", "NYSE")])
    monkeypatch.setattr(T, "get_db", lambda: db)
    monkeypatch.setattr(U, "load_universe", lambda d, m: [])
    T._tickers_cache.clear()
    assert [r["symbol"] for r in client.get("/api/tickers?market=US").json()] == ["BRK.B"]
    assert "US" not in T._tickers_cache
    monkeypatch.setattr(U, "load_universe", lambda d, m: UNIV)
    syms = [r["symbol"] for r in client.get("/api/tickers?market=US").json()]
    assert syms == ["AAPL", "KO", "BRK.B"]
    assert "US" in T._tickers_cache
