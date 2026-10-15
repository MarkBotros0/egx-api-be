import os
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")
import pytest
from alert_fakes import FakeDB, _Result
from app.core import universe as U
from app.core.markets import EG, US

SPX = [{"s": "NYSE:KO", "d": ["KO", "Coca-Cola Company", "Consumer Non-Durables", "stock"]},
       {"s": "NASDAQ:AAPL", "d": ["AAPL", "Apple Inc.", "Electronic Technology", "stock"]},
       {"s": "NYSE:BRK.B", "d": ["BRK.B", "Berkshire Hathaway Inc. Class B", "Finance", "stock"]}]
NDX = [{"s": "NASDAQ:AAPL", "d": ["AAPL", "Apple Inc.", "Electronic Technology", "stock"]}]
DJI = [{"s": "NYSE:KO", "d": ["KO", "Coca-Cola Company", "Consumer Non-Durables", "stock"]},
       {"s": "NASDAQ:AAPL", "d": ["AAPL", "Apple Inc.", "Electronic Technology", "stock"]}]
SETS = {"SYML:SP;SPX": SPX, "SYML:NASDAQ;NDX": NDX, "SYML:DJ;DJI": DJI}

@pytest.fixture(autouse=True)
def _clear():
    U.clear_caches()
    yield
    U.clear_caches()

def _fake_scan(fail_on=None, tickers_answer=None, calls=None):
    def scan(columns, market=EG, timeout=20, limit=500, *, tickers=None, symbolset=None,
             extra_filter=None, sort_by=None):
        if calls is not None:
            calls.append({"tickers": tickers, "symbolset": symbolset, "extra_filter": extra_filter})
        if fail_on and symbolset == fail_on:
            raise OSError("scanner down")
        if symbolset:
            return SETS[symbolset]
        if tickers is not None:
            return tickers_answer or []
        return []
    return scan

def test_refresh_builds_membership_sets(monkeypatch):
    monkeypatch.setattr(U, "scan", _fake_scan())
    db = FakeDB()
    out = U.refresh_universe(db, US)
    assert out == {"success": True, "count": 3, "error": None}
    ups = db.statements("INSERT INTO market_universe")
    by_sym = {p[1]: p for _, p in ups}
    assert set(by_sym) == {"KO", "AAPL", "BRK.B"}
    assert sorted(by_sym["AAPL"][6]) == ["DJI", "NDX", "SPX"]
    assert by_sym["BRK.B"][4] == "NYSE"
    # index leavers keep their row but lose membership
    assert db.statements("UPDATE market_universe SET indices = '{}'")

def test_refresh_is_atomic_on_failure(monkeypatch):
    monkeypatch.setattr(U, "scan", _fake_scan(fail_on="SYML:DJ;DJI"))
    db = FakeDB()
    out = U.refresh_universe(db, US)
    assert out["success"] is False and out["count"] == 0
    assert not db.statements("INSERT INTO market_universe")
    assert not db.statements("UPDATE market_universe")

def test_refresh_refuses_eg():
    assert U.refresh_universe(FakeDB(), EG)["success"] is False

def test_load_cold_start_refreshes_once(monkeypatch):
    calls = []
    monkeypatch.setattr(U, "scan", _fake_scan(calls=calls))
    row = ("KO", "Coca-Cola Company", "Consumer Non-Durables", "NYSE", "stock", ["SPX", "DJI"])
    db = FakeDB()
    state = {"n": 0}
    orig = db.execute
    def execute(sql, params=()):
        if "FROM market_universe" in sql and "SELECT" in sql:
            state["n"] += 1
            return _Result([] if state["n"] == 1 else [row])
        return orig(sql, params)
    db.execute = execute
    rows = U.load_universe(db, US)
    assert [r["symbol"] for r in rows] == ["KO"]
    assert rows[0]["indices"] == ["SPX", "DJI"]
    assert len([c for c in calls if c["symbolset"]]) == 3
    U.load_universe(db, US)            # cached: no further scans
    assert len([c for c in calls if c["symbolset"]]) == 3

def test_load_does_not_cache_empty(monkeypatch):
    monkeypatch.setattr(U, "scan", _fake_scan(fail_on="SYML:SP;SPX"))
    db = FakeDB()
    assert U.load_universe(db, US) == []
    monkeypatch.setattr(U, "scan", _fake_scan())
    db.rows_for("FROM market_universe", [("KO", "Coca", "X", "NYSE", "stock", ["SPX"])])
    assert [r["symbol"] for r in U.load_universe(db, US)] == ["KO"]

def test_resolve_eg_is_always_egx():
    assert U.resolve_exchange(FakeDB(), EG, "COMI") == "EGX"

def test_resolve_prefers_universe_then_discovered_then_scanner(monkeypatch):
    calls = []
    monkeypatch.setattr(U, "scan", _fake_scan(calls=calls,
        tickers_answer=[{"s": "AMEX:VOO", "d": ["VOO", "Vanguard S&P 500 ETF", "Miscellaneous", "fund"]}]))
    db = FakeDB()
    db.rows_for("SELECT exchange FROM market_universe", [("NYSE",)])
    assert U.resolve_exchange(db, US, "brk.b") == "NYSE"
    U.clear_caches()
    db2 = FakeDB()
    db2.rows_for("SELECT exchange FROM discovered_tickers", [("NASDAQ",)])
    assert U.resolve_exchange(db2, US, "XYZ") == "NASDAQ"
    U.clear_caches()
    db3 = FakeDB()
    assert U.resolve_exchange(db3, US, "VOO") == "AMEX"
    assert calls[-1]["tickers"] == ["NASDAQ:VOO", "NYSE:VOO", "AMEX:VOO"]
    ins = db3.statements("INSERT INTO market_universe")
    assert ins and ins[0][1][1] == "VOO" and ins[0][1][5] == "fund"

def test_resolve_miss_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(U, "scan", _fake_scan(calls=calls, tickers_answer=[]))
    db = FakeDB()
    assert U.resolve_exchange(db, US, "ZZZZQ") is None
    assert U.resolve_exchange(db, US, "ZZZZQ") is None
    assert len(calls) == 1

def test_resolve_scanner_error_is_not_cached_as_miss(monkeypatch):
    def boom(*a, **k): raise OSError("down")
    monkeypatch.setattr(U, "scan", boom)
    assert U.resolve_exchange(FakeDB(), US, "KO") is None
    monkeypatch.setattr(U, "scan", _fake_scan(tickers_answer=[{"s": "NYSE:KO", "d": ["KO", "Coca", "X", "stock"]}]))
    assert U.resolve_exchange(FakeDB(), US, "KO") == "NYSE"

def test_lookup_ignores_non_stock_fund(monkeypatch):
    monkeypatch.setattr(U, "scan", _fake_scan(tickers_answer=[{"s": "NYSE:XX", "d": ["XX", "Some bond", "X", "bond"]}]))
    assert U.lookup_listing(US, "XX") is None

def test_search_symbols_merges_exact_and_name(monkeypatch):
    calls = []
    def scan(columns, market=EG, timeout=20, limit=500, *, tickers=None, symbolset=None,
             extra_filter=None, sort_by=None):
        calls.append((tickers, extra_filter, sort_by))
        if tickers:
            return [{"s": "NYSE:KO", "d": ["KO", "Coca-Cola Company", "Consumer Non-Durables", "stock"]}]
        return [{"s": "NYSE:KO", "d": ["KO", "Coca-Cola Company", "Consumer Non-Durables", "stock"]},
                {"s": "NASDAQ:COKE", "d": ["COKE", "Coca-Cola Consolidated", "Consumer Non-Durables", "stock"]}]
    monkeypatch.setattr(U, "scan", scan)
    out = U.search_symbols(US, "ko")
    assert [r["symbol"] for r in out] == ["KO", "COKE"]
    assert calls[1][1] == [{"left": "description", "operation": "match", "right": "ko"}]
    assert calls[1][2] == "market_cap_basic"
    assert U.search_symbols(US, "  ") == []
