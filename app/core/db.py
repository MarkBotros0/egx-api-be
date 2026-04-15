"""
Turso (libsql) database connection helper.

Provides a get_db() function that returns a connection to the Turso cloud
SQLite database and ensures the schema is initialized on first call.
"""

import os
import libsql_experimental as libsql

from app.core.constants import DEFAULT_RISK_FREE_RATE_PCT

_conn = None


def get_connection():
    """Create a connection to the Turso database."""
    url = os.environ.get("TURSO_DATABASE_URL", "")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN", "")

    db_path = os.path.join("/tmp", "egx-analytics.db")
    conn = libsql.connect(db_path, sync_url=url, auth_token=auth_token)
    conn.sync()
    return conn


def _portfolio_has_user_id(conn) -> bool:
    rows = conn.execute("PRAGMA table_info(portfolio)").fetchall()
    return any(r[1] == "user_id" for r in rows)


def _watchlist_has_user_id(conn) -> bool:
    rows = conn.execute("PRAGMA table_info(watchlist)").fetchall()
    return any(r[1] == "user_id" for r in rows)


def init_db(conn):
    """Create tables if they don't exist.

    When adding auth, any pre-auth `portfolio` / `watchlist` tables (which
    lacked a `user_id` column) are dropped and recreated — the app shipped
    as single-user, so there is no backfill path.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Drop legacy (pre-auth) portfolio / watchlist tables so we can recreate
    # them with a NOT NULL user_id column.
    try:
        has_portfolio = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio'"
        ).fetchone()
        if has_portfolio and not _portfolio_has_user_id(conn):
            conn.execute("DROP TABLE portfolio")
    except Exception:
        pass

    try:
        has_watchlist = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='watchlist'"
        ).fetchone()
        if has_watchlist and not _watchlist_has_user_id(conn):
            conn.execute("DROP TABLE watchlist")
    except Exception:
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            buy_price REAL NOT NULL,
            buy_date TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            notes TEXT DEFAULT '',
            sector TEXT DEFAULT '',
            target_price REAL,
            stop_loss REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_data (
            key TEXT PRIMARY KEY,
            value REAL,
            previous_value REAL,
            change_pct REAL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            added_at TEXT NOT NULL,
            PRIMARY KEY (user_id, symbol)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovered_tickers (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT DEFAULT 'Unknown',
            index_name TEXT DEFAULT 'EGX',
            added_at TEXT NOT NULL
        )
    """)
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('currency', 'EGP')")
    conn.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES ('risk_free_rate', ?)",
        (str(DEFAULT_RISK_FREE_RATE_PCT),),
    )
    # Composite score category weights (must sum to 100 after normalization).
    # Seeded with the "Beginner Safe" defaults — existing DBs with older seeds
    # keep their stored values (INSERT OR IGNORE is a no-op) while fresh DBs
    # and any newly-added category defaults to the Beginner Safe weight.
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_trend', '18')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_momentum', '15')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_volume', '12')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_volatility', '10')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_divergence', '8')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_quality', '12')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_risk_adjusted', '13')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_relative_strength', '12')")
    conn.commit()


def get_db():
    """Get a database connection with schema initialized."""
    global _conn
    if _conn is None:
        _conn = get_connection()
        init_db(_conn)
    return _conn
