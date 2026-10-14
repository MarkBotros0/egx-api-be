"""
A scriptable fake DB for the score-alerts tests: matches on a substring of the
SQL (first match wins) and records every statement. Same pattern as
tests/test_users_and_roles.py; imported top-level like route_table.py.
"""

from __future__ import annotations

from contextlib import contextmanager


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeDB:
    def __init__(self, responses=None):
        self.responses = responses or []   # (sql_substring, rows); first match wins
        self.log = []

    def _answer(self, sql):
        for needle, rows in self.responses:
            if needle in sql:
                return rows
        return []

    def execute(self, sql, params=()):
        self.log.append((sql, params))
        return _Result(self._answer(sql))

    def commit(self):
        pass

    @contextmanager
    def transaction(self):
        outer = self

        class _Tx:
            def execute(self, sql, params=()):
                return outer.execute(sql, params)

        yield _Tx()

    def rows_for(self, needle, rows):
        """Canned rows for any SQL containing `needle`; added after, so earlier
        responses still win. With none registered, execute() returns no rows."""
        self.responses.append((needle, rows))

    def statements(self, needle):
        return [(s, p) for s, p in self.log if needle in s]
