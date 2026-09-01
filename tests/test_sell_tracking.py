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
