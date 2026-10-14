"""
The SQL the multi-market migration helpers emit. Pure: FakeDB records the
statements and executes nothing, so the behaviour on a real database is pinned
by tests/test_market_schema_pg.py (opt-in).
"""
from app.core import db as db_mod
from alert_fakes import FakeDB


def test_ensure_key_is_guarded_by_catalog_lookup():
    fake = FakeDB()
    db_mod._ensure_key(fake, "watchlist", "PRIMARY KEY", "watchlist_pkey",
                       ("user_id", "market", "symbol"))
    [(sql, _)] = fake.statements("DO $$")
    assert "pg_constraint" in sql
    assert "ADD CONSTRAINT watchlist_pkey PRIMARY KEY (user_id, market, symbol)" in sql
    assert "DROP CONSTRAINT" in sql
    # Only a migration that actually has work to do takes the table lock.
    assert sql.index("RETURN") < sql.index("LOCK TABLE watchlist")


def test_ensure_unique_replaces_the_key_without_market():
    fake = FakeDB()
    db_mod._ensure_key(fake, "alert_log", "UNIQUE", "alert_log_market_unique",
                       ("user_id", "market", "symbol", "session_date", "direction"))
    [(sql, _)] = fake.statements("DO $$")
    assert "c.contype = 'u'" in sql
    # The old key is found by its column SET (the new one minus market), not by
    # a guessed name or a column count.
    assert "ARRAY['direction','session_date','symbol','user_id']" in sql
    assert ("ADD CONSTRAINT alert_log_market_unique UNIQUE "
            "(user_id, market, symbol, session_date, direction)") in sql


def test_rename_column_is_guarded():
    fake = FakeDB()
    db_mod._rename_column(fake, "risk_snapshot", "turnover_egp", "turnover")
    [(sql, _)] = fake.statements("DO $$")
    assert "information_schema.columns" in sql
    assert "RENAME COLUMN turnover_egp TO turnover" in sql


def test_every_symbol_table_gets_market():
    fake = FakeDB()
    db_mod._add_market_columns(fake)
    tables = {s.split()[2] for s, _ in fake.statements("ALTER TABLE")}
    assert tables >= {
        "watchlist", "discovered_tickers", "pe_data", "risk_snapshot",
        "dividend_events", "fundamentals_annual", "score_baseline", "alert_log",
        "portfolio", "portfolio_sales", "portfolio_dividends",
        "fundamentals_history", "market_regime",
    }
    for s, _ in fake.statements("ADD COLUMN IF NOT EXISTS market"):
        assert "TEXT NOT NULL DEFAULT 'EG'" in s


def test_renames_run_before_the_additive_column_loops():
    """
    `ADD COLUMN IF NOT EXISTS value_traded` before the rename would create an
    empty column on an old database, and the rename would then find its target
    taken. The rename must come first.
    """
    fake = FakeDB()
    db_mod.init_db(fake)
    sqls = [s for s, _ in fake.log]

    def first(needle):
        return next(i for i, s in enumerate(sqls) if needle in s)

    assert first("RENAME COLUMN value_traded_egp TO value_traded") < \
        first("ADD COLUMN IF NOT EXISTS value_traded ")
    assert first("RENAME COLUMN turnover_egp TO turnover") < \
        first("ALTER TABLE risk_snapshot ADD COLUMN IF NOT EXISTS")
    assert not any("_egp" in s for s in sqls
                   if "CREATE TABLE" in s or "ADD COLUMN" in s)
