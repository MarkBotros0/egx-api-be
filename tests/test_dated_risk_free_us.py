"""
US realized trades graded against the T-bill that prevailed while they ran.

`routers/sales._rate_steps` used to return EGINTR for EG and [] for every other
market, so a US sale was graded against TODAY's single T-bill rate — the exact
bug tests/test_dated_risk_free.py documents for EG. The series a market is
graded against is now a market fact (`MarketConfig.risk_free_series`), and a
daily yield that stopped being stored falls back to the scalar
(`risk_free_max_gap_days`) rather than being read as flat.

Pure functions and AST walks only — no Postgres, no network.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.core.markets import EG, MARKETS, US
from app.core.macro_series import RISK_FREE_SERIES, SERIES, US_RISK_FREE_SERIES
from app.core.sales import compute_sale_metrics

ROOT = Path(__file__).resolve().parents[1]


def _sale(**over):
    base = {
        "symbol": "AAPL", "quantity": 10,
        "buy_price": 100.0, "sell_price": 104.5,
        "buy_date": "2023-01-01", "sell_date": "2024-01-01",
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The series is a market fact
# ---------------------------------------------------------------------------

def test_each_market_names_its_own_risk_free_series():
    assert EG.risk_free_series == RISK_FREE_SERIES == "EGINTR"
    assert US.risk_free_series == US_RISK_FREE_SERIES == "US03MY"


def test_every_named_series_is_a_known_macro_series():
    for m in MARKETS.values():
        if m.risk_free_series:
            assert m.risk_free_series in SERIES, m.code


def test_eg_keeps_the_step_function_reading():
    """A policy rate holds until the next decision; EG output must not change."""
    assert EG.risk_free_max_gap_days is None
    assert US.risk_free_max_gap_days is not None


# ---------------------------------------------------------------------------
# The router reads the market's series
# ---------------------------------------------------------------------------

class _StubDB:
    def __init__(self, rows):
        self.rows, self.params = rows, None

    def execute(self, sql, params=None):
        self.params = params
        return self

    def fetchall(self):
        return self.rows


def test_rate_steps_reads_the_us_tbill_for_a_us_sale():
    from app.routers.sales import _rate_steps

    db = _StubDB([("2023-01-03", 4.4), ("2023-07-03", 5.3)])
    assert _rate_steps(db, US) == [("2023-01-03", 4.4), ("2023-07-03", 5.3)]
    assert db.params == ("US03MY",)


def test_rate_steps_still_reads_egintr_for_eg():
    from app.routers.sales import _rate_steps

    db = _StubDB([])
    _rate_steps(db, EG)
    assert db.params == ("EGINTR",)


def test_the_router_hardcodes_no_series_or_market():
    """The per-market series lives on MarketConfig, not in the router."""
    src = (ROOT / "app" / "routers" / "sales.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_rate_steps")
    literals = {n.value for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    code = [n for n in ast.walk(fn) if not isinstance(n, ast.Expr)]
    assert not literals - {fn.body[0].value.value}, literals
    names = {n.attr for n in code if isinstance(n, ast.Attribute)}
    assert "risk_free_series" in names
    assert "code" not in names, "_rate_steps branches on a market code again"


def test_every_router_call_passes_the_markets_staleness_limit():
    """Without it a stale US series would be read as flat for a new sale."""
    tree = ast.parse((ROOT / "app" / "routers" / "sales.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "compute_sale_metrics"]
    assert calls
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "rate_steps" in kw, c.lineno
        assert "max_stale_days" in kw, c.lineno
        assert ast.unparse(kw["max_stale_days"]) == "market.risk_free_max_gap_days", c.lineno


# ---------------------------------------------------------------------------
# The grade itself
# ---------------------------------------------------------------------------

DAILY_2023 = [("2023-01-03", 4.4), ("2023-07-03", 5.3), ("2024-01-02", 5.2)]


def test_a_us_trade_is_graded_against_the_tbill_that_prevailed():
    """
    +4.5% over 2023 beats today's ~4.2% but loses to the ~4.8–5.3% the T-bill
    paid while the trade ran.
    """
    flat = compute_sale_metrics(_sale(), 4.2)
    dated = compute_sale_metrics(_sale(), 4.2, rate_steps=DAILY_2023,
                                 max_stale_days=US.risk_free_max_gap_days)
    assert flat["beat_t_bill"] is True
    assert dated["beat_t_bill"] is False
    assert 4.4 < dated["t_bill_hurdle_pct"] < 5.3


def test_a_sale_past_the_stored_history_falls_back_to_the_scalar():
    """A daily yield that stopped printing is stale, not flat."""
    sale = _sale(buy_date="2024-01-01", sell_date="2024-06-01")
    got = compute_sale_metrics(sale, 4.2, rate_steps=DAILY_2023,
                               max_stale_days=US.risk_free_max_gap_days)
    assert got["t_bill_hurdle_pct"] == 4.2


def test_a_sale_inside_the_gap_is_still_graded_on_history():
    sale = _sale(sell_date="2024-01-09")
    got = compute_sale_metrics(sale, 4.2, rate_steps=DAILY_2023,
                               max_stale_days=US.risk_free_max_gap_days)
    assert got["t_bill_hurdle_pct"] != 4.2


def test_no_history_falls_back_to_the_scalar():
    got = compute_sale_metrics(_sale(), 4.2, rate_steps=[],
                               max_stale_days=US.risk_free_max_gap_days)
    assert got["t_bill_hurdle_pct"] == 4.2


def test_eg_policy_rate_is_carried_forward_without_a_limit():
    """EG: max_stale_days None — an old EGINTR step still applies, as before."""
    sale = _sale(buy_date="2025-01-01", sell_date="2026-01-01")
    steps = [("2020-01-01", 27.25)]
    a = compute_sale_metrics(sale, 19.0, rate_steps=steps)
    b = compute_sale_metrics(sale, 19.0, rate_steps=steps,
                             max_stale_days=EG.risk_free_max_gap_days)
    assert a == b
    assert a["t_bill_hurdle_pct"] == pytest.approx(27.25, abs=0.01)


# ---------------------------------------------------------------------------
# The history keeps growing after the one-off backfill
# ---------------------------------------------------------------------------

def test_the_daily_tbill_refresh_also_appends_to_macro_series(monkeypatch):
    import pandas as pd
    from alert_fakes import FakeDB
    from app.core import macro_fetch as M
    import app.vendor.egxpy as egxpy

    df = pd.DataFrame({"close": [4.19, 4.21]},
                      index=pd.to_datetime(["2026-10-15", "2026-10-16"]))
    monkeypatch.setattr(egxpy, "get_OHLCV_data",
                        lambda s, e, i, n: df if (s, e) == ("US03MY", "TVC") else None)
    db = FakeDB()
    assert M.refresh_us_risk_free(db) == 4.21
    writes = db.statements("macro_series")
    assert writes, "the T-bill refresh no longer extends the dated history"
