"""
Neon Postgres database connection helper.

Exposes get_db() returning a thin wrapper over a psycopg connection pool.
The wrapper preserves the historical `db.execute(sql, params).fetchone()/fetchall()`
and `db.commit()` surface so routers did not need structural changes when moving
off Turso. Connections run with autocommit=True, so `commit()` is a no-op.
"""

import os
from psycopg_pool import ConnectionPool

from app.core.constants import DEFAULT_RISK_FREE_RATE_PCT

_pool = None
_initialized = False


class _Result:
    __slots__ = ("_rows",)

    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _DB:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def execute(self, sql: str, params=()):
        with self._pool.connection() as conn:
            cur = conn.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            return _Result(rows)

    def commit(self):
        # autocommit is enabled on pool connections; retained for API parity
        pass


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        url = os.environ.get("DATABASE_URL", "")
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = ConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True},
            open=True,
        )
    return _pool


def init_db(db: _DB) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            buy_price DOUBLE PRECISION NOT NULL,
            buy_date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            notes TEXT DEFAULT '',
            sector TEXT DEFAULT '',
            target_price DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id)")

    db.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            key TEXT PRIMARY KEY,
            value DOUBLE PRECISION,
            previous_value DOUBLE PRECISION,
            change_pct DOUBLE PRECISION,
            updated_at TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            added_at TEXT NOT NULL,
            PRIMARY KEY (user_id, symbol)
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)")

    db.execute("""
        CREATE TABLE IF NOT EXISTS discovered_tickers (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT DEFAULT 'Unknown',
            index_name TEXT DEFAULT 'EGX',
            added_at TEXT NOT NULL
        )
    """)

    db.execute(
        "INSERT INTO settings (key, value) VALUES ('currency', 'EGP') "
        "ON CONFLICT (key) DO NOTHING"
    )
    db.execute(
        "INSERT INTO settings (key, value) VALUES ('risk_free_rate', %s) "
        "ON CONFLICT (key) DO NOTHING",
        (str(DEFAULT_RISK_FREE_RATE_PCT),),
    )

    # "Beginner Safe" composite weight defaults — ON CONFLICT keeps existing
    # DBs untouched.
    for key, default in (
        ("weight_trend", "18"),
        ("weight_momentum", "15"),
        ("weight_volume", "12"),
        ("weight_volatility", "10"),
        ("weight_divergence", "8"),
        ("weight_quality", "12"),
        ("weight_risk_adjusted", "13"),
        ("weight_relative_strength", "12"),
    ):
        db.execute(
            "INSERT INTO settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING",
            (key, default),
        )


def get_db():
    """Return a shared DB wrapper; initialize schema exactly once per process."""
    global _initialized
    db = _DB(_get_pool())
    if not _initialized:
        init_db(db)
        _initialized = True
    return db
