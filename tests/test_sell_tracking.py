"""
Tests for sell tracking and realized gains.

Run from egx-api-be:  python -m pytest tests/test_sell_tracking.py -v
"""

from contextlib import contextmanager
from datetime import date

import pytest

from app.core.db import _DB


# ---------------------------------------------------------------------------
# Fakes — the DB wrapper is tested without a real Postgres
# ---------------------------------------------------------------------------

class _FakeCursor:
    description = None

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, log):
        self.log = log
        self.committed = False
        self.rolled_back = False

    def execute(self, sql, params=()):
        self.log.append(sql)
        return _FakeCursor()

    @contextmanager
    def transaction(self):
        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        self.committed = True


class _FakePool:
    def __init__(self):
        self.conns = []
        self.log = []

    @contextmanager
    def connection(self):
        conn = _FakeConn(self.log)
        self.conns.append(conn)
        yield conn


def test_transaction_runs_every_statement_on_one_connection():
    # execute() takes a FRESH pooled connection per call, so a sale insert and
    # its quantity decrement issued through it could half-land.
    pool = _FakePool()
    db = _DB(pool)

    with db.transaction() as tx:
        tx.execute("UPDATE portfolio SET quantity = 1")
        tx.execute("INSERT INTO portfolio_sales VALUES (1)")

    assert len(pool.conns) == 1, "both statements must share one connection"
    assert pool.conns[0].committed
    assert len(pool.log) == 2


def test_transaction_rolls_back_on_error():
    pool = _FakePool()
    db = _DB(pool)

    with pytest.raises(RuntimeError):
        with db.transaction() as tx:
            tx.execute("UPDATE portfolio SET quantity = 1")
            raise RuntimeError("over-sold")

    assert pool.conns[0].rolled_back
    assert not pool.conns[0].committed


from app.core.returns import (
    MIN_DAYS_FOR_ANNUALIZATION,
    annualized_return,
    days_between,
)


def test_annualization_is_suppressed_below_thirty_days():
    # A +5% week annualizes to five figures. The UI must show nothing.
    assert annualized_return(5.0, 29) is None
    assert annualized_return(5.0, MIN_DAYS_FOR_ANNUALIZATION) is not None


def test_annualized_return_over_one_year_is_the_plain_return():
    assert annualized_return(10.0, 365) == pytest.approx(10.0, abs=0.01)


def test_two_year_gain_annualizes_below_the_t_bill():
    # The whole point of the T-bill line: +8% held two years LOST to cash.
    ann = annualized_return(8.0, 730)
    assert ann == pytest.approx(3.92, abs=0.05)
    assert ann < 25.0


def test_total_loss_floors_at_minus_one_hundred():
    assert annualized_return(-100.0, 365) == -100.0
    assert annualized_return(-150.0, 365) == -100.0


def test_days_between_counts_calendar_days():
    assert days_between("2026-01-01", date(2026, 1, 31)) == 30
    assert days_between("2026-01-01T00:00:00Z", date(2026, 1, 31)) == 30


def test_days_between_returns_zero_on_unparseable_input():
    assert days_between("not-a-date", date(2026, 1, 31)) == 0


from pathlib import Path

_ROUTERS = Path(__file__).resolve().parents[1] / "app" / "routers"


def test_open_holdings_filter_is_spelled_once():
    """
    portfolio.py and portfolio_analysis.py each issue their own SELECT against
    the portfolio table — portfolio_analysis does NOT call /api/portfolio. If
    either hand-writes the `quantity > 0` filter, the two can drift and a
    fully-sold holding reappears in the risk metrics as a phantom position.
    """
    for name in ("portfolio.py", "portfolio_analysis.py"):
        src = (_ROUTERS / name).read_text(encoding="utf-8")
        assert "quantity > 0" not in src, (
            f"{name} hand-writes the open-holdings filter; "
            "call core.holdings.fetch_open_holdings instead"
        )
        assert "fetch_open_holdings" in src, (
            f"{name} must read holdings through core.holdings.fetch_open_holdings"
        )
