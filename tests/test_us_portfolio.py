"""
US portfolio and watchlist: symbols are validated on add, and the analysis
speaks in the market's currency and benchmark rather than in EGP and EGX30.

Run from egx-api-be:  python -m pytest tests/test_us_portfolio.py -v

EG stays lenient: adding an EG holding or watchlist symbol must never call the
resolver. A US realized sale is graded against the US rate, never Egypt's.
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
    monkeypatch.setattr(auth_mod, "AUTH_SECRET", "test-secret-for-us-portfolio-0123456789")
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    cache._store.clear()
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {create_access_token(ME.id)}"})
    yield c
    cache._store.clear()


@pytest.fixture(autouse=True)
def us_listings(monkeypatch):
    """No scanner and no Postgres: EG is EGX, the US answers from a table."""
    from app.core import db as db_mod
    table = {"KO": "NYSE", "AAPL": "NASDAQ"}
    monkeypatch.setattr(U, "resolve_exchange",
                        lambda db, market, symbol: "EGX" if market.code == "EG"
                        else table.get(symbol.upper()))
    monkeypatch.setattr(db_mod, "get_db", lambda: FakeDB())
    return table


def _patch_db(monkeypatch, db):
    import app.routers.portfolio as P
    import app.routers.portfolio_analysis as PA
    import app.routers.sales as S
    import app.routers.watchlist as W
    for mod in (P, W, PA, S):
        monkeypatch.setattr(mod, "get_db", lambda: db)
    return db


@pytest.fixture
def fake_db(monkeypatch):
    return _patch_db(monkeypatch, FakeDB())


def _frame(n: int, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=idx)


def _flat_frame(n: int) -> pd.DataFrame:
    """A price that goes nowhere over the year: its annualized return sits in
    the "underperforms the T-bill" band, so that reason is emitted."""
    i = np.arange(n)
    close = 20 * (1 + 0.01 * np.sin(i / 7.0))
    idx = pd.bdate_range("2024-01-01", periods=n)
    vol = 2e5 + 1e5 * (i % 5)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": vol.astype(float),
    }, index=idx)


@pytest.fixture
def fake_feed(monkeypatch):
    calls = []

    def fake(symbol, exchange, interval, bars):
        calls.append((symbol, exchange))
        return _flat_frame(bars)

    import app.vendor.egxpy as egxpy
    monkeypatch.setattr(egxpy, "get_OHLCV_data", fake)
    return calls


# get_pe_for_symbol's row: company_name, pe_ratio, dividend_yield, loss_making,
# updated_at, dividend_ex_date_recent, dividend_amount_recent. A P/E under 8
# and a 2-4% yield hit the bands whose EG copy names the EGX median.
KO_PE = ("Coca-Cola", 6.2, 3.1, False, "2026-10-14", None, None)

BANNED = ("EGP", "EGX30", "EGX", "policy rate")


def _reasons(breakdown):
    return [r for cat in (breakdown or {}).values() for r in (cat.get("reasons") or [])]


@pytest.fixture
def fake_db_with_us_holding(monkeypatch):
    """One open KO lot, with a stop and a target set so both messages render.

    The row follows HOLDING_COLUMNS: id, symbol, name, buy_price, buy_date,
    quantity, notes, sector, target_price, stop_loss, created_at, updated_at.
    A stop far above the price is breached and a target far below it is hit,
    so the stop and target messages (both quoting prices) are emitted.
    """
    db = FakeDB()
    db.rows_for("FROM pe_data WHERE", [KO_PE])
    db.rows_for("FROM portfolio WHERE", [(
        "h1", "KO", "Coca-Cola", 15.0, "2024-01-02", 10, "", "Beverages",
        1.0, 1000.0, "2024-01-02T00:00:00Z", "2024-01-02T00:00:00Z",
    )])
    return _patch_db(monkeypatch, db)


# ---------------------------------------------------------------------------
# Add holding / add watchlist
# ---------------------------------------------------------------------------

def test_add_us_holding_unknown_symbol_422(client, monkeypatch, fake_db):
    monkeypatch.setattr(U, "resolve_exchange", lambda db, m, s: None)
    r = client.post("/api/portfolio?market=US",
                    json={"symbol": "ZZZZQ", "buy_price": 10, "quantity": 1})
    assert r.status_code == 422 and r.json()["detail"] == "Unknown US symbol: ZZZZQ"
    assert fake_db.statements("INSERT") == []


def test_add_us_holding_dotted_symbol(client, monkeypatch, fake_db):
    monkeypatch.setattr(U, "resolve_exchange",
                        lambda db, m, s: "NYSE" if s == "BRK.B" else None)
    r = client.post("/api/portfolio", json={"symbol": "brk.b", "buy_price": 500,
                                            "quantity": 2, "market": "US"})
    assert r.status_code == 201
    assert r.json()["symbol"] == "BRK.B" and r.json()["market"] == "US"


def test_add_eg_holding_still_lenient(client, monkeypatch, fake_db):
    monkeypatch.setattr(U, "resolve_exchange",
                        lambda *a: (_ for _ in ()).throw(AssertionError("EG must not validate")))
    r = client.post("/api/portfolio", json={"symbol": "ANYTHING", "buy_price": 1, "quantity": 1})
    assert r.status_code == 201


def test_watch_us_unknown_422(client, monkeypatch, fake_db):
    monkeypatch.setattr(U, "resolve_exchange", lambda db, m, s: None)
    r = client.post("/api/watchlist?market=US", json={"symbol": "ZZZZQ"})
    assert r.status_code == 422 and r.json()["detail"] == "Unknown US symbol: ZZZZQ"
    assert fake_db.statements("INSERT") == []


def test_watch_us_known_symbol_is_added(client, fake_db):
    r = client.post("/api/watchlist", json={"symbol": "aapl", "market": "US"})
    assert r.status_code == 201 and r.json() == {**r.json(), "symbol": "AAPL", "market": "US"}


def test_watch_eg_still_lenient(client, monkeypatch, fake_db):
    monkeypatch.setattr(U, "resolve_exchange",
                        lambda *a: (_ for _ in ()).throw(AssertionError("EG must not validate")))
    assert client.post("/api/watchlist", json={"symbol": "ANYTHING"}).status_code == 201


# ---------------------------------------------------------------------------
# Portfolio analysis copy
# ---------------------------------------------------------------------------

def _bench(bars):
    # A benchmark that fell hard over the last month, so KO leads it and the
    # relative-strength signal names the benchmark.
    close = _frame(bars)["Close"].copy()
    close.iloc[-30:] = close.iloc[-30:] * np.linspace(1.0, 0.7, 30)
    return close.rename("close")


def test_us_portfolio_analysis_has_no_eg_wording(client, monkeypatch, fake_feed,
                                                 fake_db_with_us_holding):
    import app.core.benchmark as B
    monkeypatch.setattr(U, "resolve_exchange", lambda db, m, s: "NYSE")
    monkeypatch.setattr(B, "get_benchmark_close", lambda m, i, b: _bench(b))
    # Even a macro dict carrying an egx30 reading must not reach a US response.
    import app.routers.portfolio_analysis as PA
    monkeypatch.setattr(PA, "fetch_macro_for", lambda db, m: {
        "egx30": {"value": 30000.0, "trend": "bullish", "monthly_change_pct": 2.0}})
    r = client.get("/api/portfolio_analysis?market=US")
    assert r.status_code == 200
    body = r.json()
    assert body["currency"] == "USD"
    text = r.text
    for banned in BANNED:
        assert banned not in text, banned
    assert "USD" in text
    assert "S&P 500" in text
    types = {s["type"] for s in body["signals"]}
    assert {"stop_breached", "target_hit", "relative_strength_leader"} <= types
    assert "macro_egx30" not in types
    assert not types & {"pe_undervalued", "dividend_yield_solid"}
    # The branches whose EG copy names the EGX actually fired, neutrally.
    reasons = _reasons(body["holdings"][0]["composite_breakdown"])
    assert "P/E 6.2" in reasons and "Dividend yield 3.1%" in reasons
    assert any("UNDERPERFORMS" in x and "T-bill rate" in x for x in reasons), reasons


def test_eg_portfolio_analysis_keeps_egp_wording(client, monkeypatch, fake_feed,
                                                 fake_db_with_us_holding):
    import app.core.benchmark as B
    import app.routers.portfolio_analysis as PA
    monkeypatch.setattr(B, "get_benchmark_close", lambda m, i, b: _bench(b))
    monkeypatch.setattr(PA, "fetch_macro_for", lambda db, m: {
        "egx30": {"value": 30000.0, "trend": "bullish", "monthly_change_pct": 2.0}})
    r = client.get("/api/portfolio_analysis")
    assert r.status_code == 200
    signals = r.json()["signals"]
    msgs = " ".join(s["message"] for s in signals)
    assert "1000.00 EGP" in msgs and "outperforming EGX30" in msgs
    assert "macro_egx30" in {s["type"] for s in signals}


def test_eg_reasons_keep_their_egx_copy(client, monkeypatch, fake_feed, fake_db_with_us_holding):
    import app.core.benchmark as B
    import app.routers.portfolio_analysis as PA
    monkeypatch.setattr(B, "get_benchmark_close", lambda m, i, b: _bench(b))
    monkeypatch.setattr(PA, "fetch_macro_for", lambda db, m: None)
    # A low EG rate puts the flat price in the "underperforms" band.
    fake_db_with_us_holding.rows_for("key = 'risk_free_rate'", [("4",)])
    body = client.get("/api/portfolio_analysis").json()
    reasons = _reasons(body["holdings"][0]["composite_breakdown"])
    assert "Cheap versus the EGX median of ~12 (P/E 6.2)" in reasons
    assert "Pays about the EGX median dividend (DY 3.1%)" in reasons
    assert any("policy rate" in x and "on the EGX" in x for x in reasons), reasons


def test_us_stock_page_has_no_eg_wording(client, monkeypatch, fake_feed):
    from app.routers import analysis as analysis_mod
    db = FakeDB()
    db.rows_for("FROM pe_data WHERE", [KO_PE])
    monkeypatch.setattr(analysis_mod, "get_db", lambda: db)
    monkeypatch.setattr(analysis_mod, "fetch_macro_for", lambda db, market: None)
    r = client.get("/api/analysis", params={"symbol": "KO", "market": "US", "bars": 60})
    assert r.status_code == 200, r.text
    import json
    body = r.json()
    body.pop("forecast")
    text = json.dumps(body, ensure_ascii=False)
    for banned in BANNED:
        assert banned not in text, banned
    assert "P/E 6.2" in r.text and "T-bill rate" in r.text and "S&P 500" in r.text


def test_position_signals_default_to_egp():
    from datetime import date
    from app.routers.portfolio_analysis import build_position_signals
    pos = {"symbol": "COMI", "invested": 100.0, "current_value": 90.0,
           "current_price": 9.0, "earliest_buy_date": "2026-01-01",
           "stop_loss": 10.0, "target_price": None}
    eg = build_position_signals([pos], risk_free_rate_pct=19, today=date(2026, 2, 1))
    us = build_position_signals([pos], risk_free_rate_pct=4, today=date(2026, 2, 1),
                                currency="USD")
    assert "10.00 EGP" in eg[0]["message"]
    assert "10.00 USD" in us[0]["message"] and "EGP" not in us[0]["message"]


# ---------------------------------------------------------------------------
# Realized sales: graded against the sale's own market rate
# ---------------------------------------------------------------------------

def _sale_db(monkeypatch, market_code):
    lot = ("h1", "KO", "Coca-Cola", 50.0, "2025-01-02", 10, "", "Beverages",
           None, None, "2025-01-02T00:00:00Z", "2025-01-02T00:00:00Z")
    db = FakeDB()
    db.rows_for("market FROM portfolio WHERE id", [lot + (market_code,)])
    db.rows_for("FROM portfolio WHERE", [lot])
    db.rows_for("UPDATE portfolio SET quantity", [(0,)])
    db.rows_for("key = 'risk_free_rate:US'", [("4.5",)])
    db.rows_for("key = 'risk_free_rate'", [("19",)])
    # Egypt's dated policy-rate history covers the window. It must grade an EG
    # sale and must NOT grade a US one.
    db.rows_for("FROM macro_series", [("2020-01-01", 27.25)])
    return _patch_db(monkeypatch, db)


def test_us_sale_is_graded_against_the_us_rate(client, monkeypatch):
    _sale_db(monkeypatch, "US")
    r = client.post("/api/sales", json={"holding_id": "h1", "quantity": 10,
                                        "sell_price": 60, "sell_date": "2025-12-01"})
    assert r.status_code == 201, r.text
    assert r.json()["currency"] == "USD"
    assert r.json()["sales"][0]["t_bill_hurdle_pct"] == 4.5


def test_eg_sale_still_uses_the_dated_egyptian_rate(client, monkeypatch):
    _sale_db(monkeypatch, "EG")
    r = client.post("/api/sales", json={"holding_id": "h1", "quantity": 10,
                                        "sell_price": 60, "sell_date": "2025-12-01"})
    assert r.status_code == 201, r.text
    assert r.json()["sales"][0]["t_bill_hurdle_pct"] == 27.25


def test_us_sales_ledger_reports_the_us_rate(client, monkeypatch):
    _sale_db(monkeypatch, "US")
    r = client.get("/api/sales?market=US")
    assert r.status_code == 200
    assert r.json()["risk_free_rate_pct"] == 4.5


def test_eg_sales_ledger_reports_the_eg_rate(client, monkeypatch):
    _sale_db(monkeypatch, "EG")
    r = client.get("/api/sales")
    assert r.status_code == 200
    assert r.json()["risk_free_rate_pct"] == 19.0
