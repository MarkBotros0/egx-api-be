"""
Turso (libsql) database connection helper.

Provides a get_db() function that returns a connection to the Turso cloud
SQLite database and ensures the schema is initialized on first call.
"""

import os
import libsql_experimental as libsql

_conn = None


def get_connection():
    """Create a connection to the Turso database."""
    url = os.environ.get("TURSO_DATABASE_URL", "")
    auth_token = os.environ.get("TURSO_AUTH_TOKEN", "")

    db_path = os.path.join("/tmp", "egx-analytics.db")
    conn = libsql.connect(db_path, sync_url=url, auth_token=auth_token)
    conn.sync()
    return conn


def init_db(conn):
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id TEXT PRIMARY KEY,
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
            symbol TEXT PRIMARY KEY,
            added_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovered_tickers (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT DEFAULT 'Unknown',
            index_name TEXT DEFAULT 'EGX',
            added_at TEXT NOT NULL
        )
    """)
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('cash_available', '50000')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('currency', 'EGP')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('risk_free_rate', '25')")
    # Composite score category weights (must sum to 100 after normalization)
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_trend', '25')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_momentum', '25')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_volume', '20')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_volatility', '15')")
    conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('weight_divergence', '15')")
    conn.commit()


def get_db():
    """Get a database connection with schema initialized."""
    global _conn
    if _conn is None:
        _conn = get_connection()
        init_db(_conn)
    return _conn
