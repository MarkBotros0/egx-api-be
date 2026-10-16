"""
The US macro card: DISPLAY ONLY.

US scores take no macro modulation — the US calibration backtest ran without
it, so modulating them would detach every US score from its calibration. The
card therefore has its own path (`fetch_macro_display`), separate from the
scoring one (`fetch_macro_for`, which stays None for US). No network, no
Postgres.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from alert_fakes import FakeDB  # noqa: E402
from app.core import cache  # noqa: E402
from app.core import macro_fetch as M  # noqa: E402
from app.core.markets import EG, US  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _close(n=30, start=5000.0, step=10.0):
    return pd.Series([start + i * step for i in range(n)], name="close")


@pytest.fixture(autouse=True)
def _clear_cache():
    cache._store.clear()
    yield
    cache._store.clear()


@pytest.fixture
def us_bench(monkeypatch):
    import app.core.benchmark as B
    seen = []
    monkeypatch.setattr(B, "get_benchmark_close",
                        lambda m, i, b: seen.append((m.code, i, b)) or _close())
    return seen


def test_scoring_macro_stays_none_for_us(us_bench):
    """The guard this whole design exists for."""
    assert M.fetch_macro_for(FakeDB(), US) is None


def test_display_is_none_for_eg():
    assert M.fetch_macro_display(FakeDB(), EG) is None


def test_us_display_reads_the_benchmark_and_the_tbill(us_bench):
    db = FakeDB()
    db.rows_for("risk_free_rate:US", [("4.21",)])
    got = M.fetch_macro_display(db, US)
    assert got["market"] == "US" and got["currency"] == "USD"
    assert got["scoring"] is False
    assert got["egx30"] is None and got["usd_egp"] is None
    b = got["benchmark"]
    assert b["label"] == "S&P 500"
    assert b["value"] == 5290.0
    # The ±2% up/down/stable buckets are EGX30's: not sent for US.
    assert b["direction"] is None and "trend" not in b
    assert b["monthly_change_pct"] is not None
    assert got["interest_rate"] == {"value": 4.21, "direction": None, "label": "3-month T-bill"}
    # Same scoring window as the analysis paths, so it shares their cache.
    from app.core.constants import INTERNAL_BARS_MIN
    assert us_bench == [("US", "Daily", INTERNAL_BARS_MIN)]


def test_us_display_never_writes_the_eg_macro_cache_table(us_bench):
    db = FakeDB()
    M.fetch_macro_display(db, US)
    assert not db.statements("macro_data")


def test_us_display_degrades_when_the_benchmark_is_down(monkeypatch):
    import app.core.benchmark as B
    monkeypatch.setattr(B, "get_benchmark_close", lambda m, i, b: None)
    got = M.fetch_macro_display(FakeDB(), US)
    assert got["benchmark"]["value"] is None
    assert got["interest_rate"]["value"] == US.risk_free_fallback_pct


def test_us_display_has_no_egyptian_wording(us_bench):
    text = repr(M.fetch_macro_display(FakeDB(), US))
    for banned in ("EGX", "Egypt", "EGP", "CBE", "policy rate"):
        assert banned not in text, banned


def test_eg_index_reading_is_unchanged(monkeypatch):
    """_fetch_egx30's maths moved into _index_reading verbatim. Pinned to the pre-move output."""
    import app.vendor.egxpy as egxpy
    df = pd.DataFrame({"Close": [30000.0 + i * 50 for i in range(30)]})
    monkeypatch.setattr(egxpy, "get_OHLCV_data", lambda *a: df)
    assert M._fetch_egx30() == {
        "value": 31450.0, "change_pct": 0.16, "monthly_change_pct": 3.11,
        "direction": "up", "trend": "bullish",
    }


def test_display_never_reaches_scoring():
    """
    AST: in every app module, a `macro=` argument (compute_composite,
    blend_categories, ...) is never the display reading, and only the two
    routes and macro_fetch itself call fetch_macro_display.
    """
    callers = set()
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name == "fetch_macro_display":
                callers.add(path.relative_to(ROOT).as_posix())
            for kw in node.keywords:
                if kw.arg == "macro":
                    assert "display" not in ast.unparse(kw.value), (path, node.lineno)
    assert callers == {"app/routers/macro.py", "app/routers/portfolio_analysis.py"}
