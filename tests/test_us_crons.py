"""
US crons: universe refresh + T-bill rate, the US snapshot universe, and the
US fundamentals refresh (ticker-chunked scans, per-market status keys).
No network or Postgres is touched.
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from alert_fakes import FakeDB  # noqa: E402
from app.core.markets import EG, US  # noqa: E402
from app.main import app  # noqa: E402
import app.routers.cron as C  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    monkeypatch.setenv("CRON_SECRET", "s")
    monkeypatch.setenv("PE_REFRESH_SECRET", "s")
    return TestClient(app)


# --- POST /api/cron/universe -------------------------------------------------

def test_cron_universe_us(client, monkeypatch):
    monkeypatch.setattr(C, "refresh_universe", lambda db, m: {"success": True, "count": 3, "error": None})
    monkeypatch.setattr(C, "refresh_us_risk_free", lambda db: 4.21)
    monkeypatch.setattr(C, "get_db", lambda: FakeDB())
    r = client.post("/api/cron/universe?market=US", headers={"X-Cron-Secret": "s"})
    assert r.status_code == 200
    assert r.json() == {"market": "US", "universe": {"success": True, "count": 3, "error": None}, "risk_free": 4.21}


def test_cron_universe_rejects_eg(client):
    assert client.post("/api/cron/universe", headers={"X-Cron-Secret": "s"}).status_code == 400


def test_cron_universe_requires_secret(client):
    assert client.post("/api/cron/universe?market=US").status_code in (401, 403)


# --- snapshot universe --------------------------------------------------------

def test_us_snapshot_universe_includes_holdings_and_watchlist(monkeypatch):
    db = FakeDB()
    db.rows_for("FROM portfolio", [("TSLA",)])
    db.rows_for("FROM watchlist", [("VOO",), ("KO",)])
    monkeypatch.setattr(C, "load_universe", lambda d, m: [
        {"symbol": "KO", "indices": ["SPX"]}, {"symbol": "OLD", "indices": []}])
    monkeypatch.setattr(C, "get_db", lambda: db)
    assert C._universe(US) == ["KO", "TSLA", "VOO"]
    # Fully-sold holdings (quantity 0) are not worth measuring.
    (sql, params), = db.statements("FROM portfolio")
    assert "quantity > 0" in sql and params == ("US",)


def test_us_snapshot_universe_survives_a_failing_table(monkeypatch):
    class Boom(FakeDB):
        def execute(self, sql, params=()):
            if "FROM portfolio" in sql:
                raise RuntimeError("db down")
            return super().execute(sql, params)

    db = Boom()
    db.rows_for("FROM watchlist", [("VOO",)])
    monkeypatch.setattr(C, "load_universe", lambda d, m: [{"symbol": "KO", "indices": ["SPX"]}])
    monkeypatch.setattr(C, "get_db", lambda: db)
    assert C._universe(US) == ["KO", "VOO"]


def test_eg_snapshot_universe_unchanged():
    syms = C._universe(EG)
    assert "COMI" in syms


def test_alerts_run_for_every_market():
    import inspect
    src = inspect.getsource(C.risk_snapshot)
    assert '{"skipped": "market"}' not in src
    assert "market=market" in src


def test_us_snapshot_runs_alerts_for_us(client, monkeypatch):
    import pandas as pd
    import app.vendor.egxpy as egxpy
    from app.core import universe as U

    monkeypatch.setattr(C, "_universe", lambda m: ["KO"])
    monkeypatch.setattr(C, "get_db", lambda: FakeDB())
    monkeypatch.setattr(C, "_benchmark_close", lambda m: None)
    monkeypatch.setattr(U, "resolve_exchange", lambda db, m, s: None)   # unresolved -> failed
    monkeypatch.setattr(egxpy, "get_OHLCV_data", lambda *a: pd.DataFrame())
    called = []
    monkeypatch.setattr(C, "run_alerts",
                        lambda *a, **k: called.append(k["market"].code) or {"queued": 0})
    r = client.post("/api/cron/risk_snapshot?market=US", headers={"X-Cron-Secret": "s"})
    assert r.status_code == 200
    body = r.json()
    assert body["alerts"] == {"queued": 0} and body["failed"] == ["KO"]
    assert called == ["US"]


# --- US T-bill rate -----------------------------------------------------------

def test_refresh_us_risk_free(monkeypatch):
    import pandas as pd
    from app.core import macro_fetch as M
    import app.vendor.egxpy as egxpy
    monkeypatch.setattr(egxpy, "get_OHLCV_data", lambda s, e, i, n: pd.DataFrame({"close": [4.1, 4.216]}) if (s, e) == ("US03MY", "TVC") else None)
    db = FakeDB()
    assert M.refresh_us_risk_free(db) == 4.216
    assert db.statements("risk_free_rate:US")


def test_refresh_us_risk_free_never_raises(monkeypatch):
    from app.core import macro_fetch as M
    import app.vendor.egxpy as egxpy

    def boom(*a):
        raise RuntimeError("feed down")
    monkeypatch.setattr(egxpy, "get_OHLCV_data", boom)
    db = FakeDB()
    assert M.refresh_us_risk_free(db) is None
    assert not db.statements("risk_free_rate:US")
    monkeypatch.setattr(egxpy, "get_OHLCV_data", lambda *a: None)
    assert M.refresh_us_risk_free(db) is None


# --- US fundamentals ----------------------------------------------------------

def test_pe_fetch_by_tickers_chunks(monkeypatch):
    from app.core import pe_fetch as P
    seen = []
    monkeypatch.setattr(P, "scan", lambda cols, market=EG, tickers=None, limit=500, **k: seen.append(tickers) or [])
    P.fetch_fundamentals_rows(US, tickers=[f"NYSE:S{i}" for i in range(1201)])
    assert [len(c) for c in seen] == [500, 500, 201]


def test_annual_fetch_by_tickers_chunks(monkeypatch):
    from app.core import fundamentals_annual as F
    seen = []
    monkeypatch.setattr(F, "scan", lambda cols, market=EG, tickers=None, limit=500, **k: seen.append((tickers, limit)) or [])
    F.fetch_annual_rows(market=US, tickers=[f"NYSE:S{i}" for i in range(501)])
    assert [(len(t), lim) for t, lim in seen] == [(500, 500), (1, 1)]


def _settings_writes(db):
    return [p[0] for s, p in db.log if "INTO settings" in s]


def _rows(n):
    return [{"symbol": f"S{i}", "company_name": "x", "pe_ratio": 10.0,
             "dividend_yield": 1.0, "loss_making": False} for i in range(n)]


def test_us_pe_refresh_writes_only_us_status_keys():
    from app.core import pe_fetch as P
    db = FakeDB()
    out = P.refresh_pe_data(db, US, rows=_rows(60), tickers=[f"NYSE:S{i}" for i in range(100)])
    assert out["success"] is True
    keys = _settings_writes(db)
    assert "pe_last_attempt_status:US" in keys and "pe_last_successful_fetch:US" in keys
    assert "pe_last_attempt_status" not in keys and "pe_last_successful_fetch" not in keys


def test_us_pe_refresh_floor_scales_with_universe():
    from app.core import pe_fetch as P
    tickers = [f"NYSE:S{i}" for i in range(500)]
    db = FakeDB()
    out = P.refresh_pe_data(db, US, rows=_rows(249), tickers=tickers)   # floor is 250
    assert out["success"] is False and "expected >= 250" in out["error"]
    assert not db.statements("INTO pe_data")
    assert _settings_writes(db) == ["pe_last_attempt_status:US"]
    # A small universe still demands 50.
    out = P.refresh_pe_data(FakeDB(), US, rows=_rows(49), tickers=tickers[:10])
    assert out["success"] is False and "expected >= 50" in out["error"]


def test_eg_pe_refresh_keeps_its_status_keys():
    from app.core import pe_fetch as P
    db = FakeDB()
    assert P.refresh_pe_data(db, EG, rows=_rows(100))["success"] is True
    assert set(_settings_writes(db)) == {"pe_last_attempt_status", "pe_last_successful_fetch"}


def test_pe_refresh_us_uses_universe_tickers_and_skips_eg_steps(client, monkeypatch):
    from app.routers import pe as pe_mod
    seen = []
    monkeypatch.setattr(pe_mod, "get_db", lambda: FakeDB())
    monkeypatch.setattr(pe_mod, "load_universe", lambda db, m: [
        {"symbol": "KO", "exchange": "NYSE"}, {"symbol": "AAPL", "exchange": "NASDAQ"}])
    monkeypatch.setattr(pe_mod, "refresh_pe_data",
                        lambda db, market, tickers=None: seen.append(("pe", market.code, tickers)) or {"success": True})
    monkeypatch.setattr(pe_mod, "refresh_annual_fundamentals",
                        lambda db, market, tickers=None: seen.append(("annual", market.code, tickers)) or {"success": True})

    def must_not_run(*a, **k):
        raise AssertionError("EG-only step ran for US")
    monkeypatch.setattr(pe_mod, "refresh_macro_series", must_not_run)
    monkeypatch.setattr(pe_mod, "prune_auth", must_not_run)
    r = client.post("/api/pe/refresh?market=US", headers={"X-Refresh-Secret": "s"})
    assert r.status_code == 200
    body = r.json()
    assert "macro" not in body and "auth_prune" not in body
    assert seen == [("pe", "US", ["NYSE:KO", "NASDAQ:AAPL"]),
                    ("annual", "US", ["NYSE:KO", "NASDAQ:AAPL"])]


def test_get_pe_us_reads_us_status_keys(monkeypatch):
    from app.core import auth as auth_mod
    from app.core.auth import CurrentUser, create_access_token
    from app.routers import pe as pe_mod
    me = CurrentUser(id="u1", email="sara@example.com", role="user")
    monkeypatch.setattr(auth_mod, "_load_user", lambda uid: me)
    monkeypatch.setattr(auth_mod, "AUTH_SECRET", "test-secret-for-us-crons-0123456789abcdef")
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    db = FakeDB()
    monkeypatch.setattr(pe_mod, "get_db", lambda: db)
    c = TestClient(app)
    r = c.get("/api/pe?market=US", headers={"Authorization": f"Bearer {create_access_token(me.id)}"})
    assert r.status_code == 200
    keys = [p for s, p in db.log if "FROM settings" in s]
    assert keys == [("pe_last_successful_fetch:US",), ("pe_last_attempt_status:US",)]


# --- batched writes -----------------------------------------------------------

def test_pe_data_is_written_in_BATCHES_not_one_row_at_a_time():
    from app.core import pe_fetch as P
    rows = [{**r, "dividend_ex_date_recent": "2026-05-01", "dividend_amount_recent": 0.5}
            for r in _rows(515)]
    db = FakeDB()
    out = P.refresh_pe_data(db, US, rows=rows, tickers=[f"NYSE:S{i}" for i in range(515)])
    assert out["success"] is True and out["count"] == 515
    inserts = db.statements("INSERT INTO pe_data")
    assert len(inserts) == 3
    assert [len(p) // 13 for _, p in inserts] == [250, 250, 15]
    assert sorted(p[i] for _, p in inserts for i in range(1, len(p), 13)) == \
        sorted(f"S{i}" for i in range(515))
    assert len(db.statements("INSERT INTO dividend_events")) == 3


def test_pe_batch_keeps_the_last_duplicate_and_skips_empty_rows():
    from app.core import pe_fetch as P
    rows = _rows(100) + [{"symbol": "S0", "pe_ratio": 99.0, "dividend_yield": None,
                          "loss_making": None},
                         {"symbol": "EMPTY", "pe_ratio": None, "dividend_yield": None,
                          "loss_making": None}]
    db = FakeDB()
    out = P.refresh_pe_data(db, EG, rows=rows)
    # 102 rows in: one empty (skipped) and one repeat of S0 (collapsed), so
    # the count is the 100 rows actually written, not 101.
    assert out["count"] == 100 and out["skipped_empty"] == 1
    [(sql, params)] = db.statements("INSERT INTO pe_data")
    assert "ON CONFLICT (market, symbol) DO UPDATE" in sql
    symbols = params[1::13]
    assert len(symbols) == 100 and "EMPTY" not in symbols
    s0 = params[symbols.index("S0") * 13: symbols.index("S0") * 13 + 13]
    assert s0[0] == "EG" and s0[3] == 99.0


def test_dividend_events_batch_keeps_first_duplicate_and_counts_inserts():
    from app.core.dividend_history import insert_dividend_events

    class Counting(FakeDB):
        def execute(self, sql, params=()):
            res = super().execute(sql, params)
            res.rowcount = len(params) // 6 - 1      # pretend one row conflicted
            return res

    db = Counting()
    n = insert_dividend_events(db, [("KO", "2026-01-01", 1.0), ("KO", "2026-01-01", 2.0),
                                    ("KO", None, 3.0), ("PEP", "2026-02-01", 4.0)],
                               source="scanner", market_code="US")
    [(sql, params)] = db.statements("INSERT INTO dividend_events")
    assert "ON CONFLICT (market, symbol, ex_date) DO NOTHING" in sql
    assert params[3] == 1.0 and len(params) == 12
    assert n == 1


# --- rollout order: the US crons run BEFORE US is enabled for users -----------

@pytest.fixture
def eg_only_client(monkeypatch):
    from app.core import auth as auth_mod
    from app.core.auth import CurrentUser
    monkeypatch.setenv("ENABLED_MARKETS", "EG")
    monkeypatch.setenv("CRON_SECRET", "s")
    monkeypatch.setenv("PE_REFRESH_SECRET", "s")
    me = CurrentUser(id="u1", email="sara@example.com", role="user")
    monkeypatch.setattr(auth_mod, "_load_user", lambda uid: me)
    monkeypatch.setattr(auth_mod, "AUTH_SECRET", "test-secret-for-us-crons-0123456789abcdef")
    return TestClient(app), me


def test_us_crons_run_while_us_is_not_enabled(eg_only_client, monkeypatch):
    import pandas as pd
    import app.vendor.egxpy as egxpy
    from app.core import universe as U
    from app.core.auth import create_access_token
    from app.routers import pe as pe_mod

    client, me = eg_only_client
    monkeypatch.setattr(C, "get_db", lambda: FakeDB())
    monkeypatch.setattr(C, "refresh_universe", lambda db, m: {"success": True, "count": 3, "error": None})
    monkeypatch.setattr(C, "refresh_us_risk_free", lambda db: 4.21)
    monkeypatch.setattr(C, "_universe", lambda m: ["KO"])
    monkeypatch.setattr(C, "_benchmark_close", lambda m: None)
    monkeypatch.setattr(U, "resolve_exchange", lambda db, m, s: None)
    monkeypatch.setattr(egxpy, "get_OHLCV_data", lambda *a: pd.DataFrame())
    monkeypatch.setattr(pe_mod, "get_db", lambda: FakeDB())
    monkeypatch.setattr(pe_mod, "load_universe", lambda db, m: [{"symbol": "KO", "exchange": "NYSE"}])
    monkeypatch.setattr(pe_mod, "refresh_pe_data", lambda db, market, tickers=None: {"success": True})
    monkeypatch.setattr(pe_mod, "refresh_annual_fundamentals", lambda db, market, tickers=None: {"success": True})

    r = client.post("/api/cron/universe?market=US", headers={"X-Cron-Secret": "s"})
    assert r.status_code == 200 and r.json()["market"] == "US"
    r = client.post("/api/cron/risk_snapshot?market=US&limit=30", headers={"X-Cron-Secret": "s"})
    assert r.status_code == 200 and r.json()["failed"] == ["KO"]
    r = client.post("/api/pe/refresh?market=US", headers={"X-Refresh-Secret": "s"})
    assert r.status_code == 200

    # ...while every user-facing route still refuses US.
    auth = {"Authorization": f"Bearer {create_access_token(me.id)}"}
    r = client.get("/api/tickers?market=US", headers=auth)
    assert r.status_code == 400 and "not enabled" in r.json()["detail"]


@pytest.mark.parametrize("path,header", [
    ("/api/cron/universe", "X-Cron-Secret"),
    ("/api/cron/risk_snapshot", "X-Cron-Secret"),
    ("/api/pe/refresh", "X-Refresh-Secret"),
])
def test_us_crons_still_refuse_an_unknown_market(eg_only_client, path, header):
    client, _ = eg_only_client
    r = client.post(f"{path}?market=XX", headers={header: "s"})
    assert r.status_code == 400 and "Unknown market" in r.json()["detail"]


def test_registered_market_param_ignores_enabled_markets(monkeypatch):
    from fastapi import HTTPException
    from app.core.markets import get_market, get_registered_market
    monkeypatch.setenv("ENABLED_MARKETS", "EG")
    assert get_registered_market("us") is US
    assert get_registered_market(None) is EG
    with pytest.raises(HTTPException) as e:
        get_market("US")
    assert e.value.status_code == 400
    for bad in ("XX", 5, ["US"]):
        with pytest.raises(HTTPException) as e:
            get_registered_market(bad)
        assert e.value.status_code == 400 and "Unknown market" in e.value.detail


def test_registered_market_param_is_only_on_the_secret_crons():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    users = [p for p in (root / "app").rglob("*.py")
             if "Depends(registered_market_param)" in p.read_text()]
    assert sorted(p.relative_to(root).as_posix() for p in users) == [
        "app/routers/cron.py", "app/routers/pe.py"]
    assert (root / "app/routers/cron.py").read_text().count("Depends(registered_market_param)") == 2
    assert (root / "app/routers/pe.py").read_text().count("Depends(registered_market_param)") == 1
