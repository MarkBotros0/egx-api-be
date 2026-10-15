"""
init_db's market migration against a REAL Postgres. Skipped unless
TEST_DATABASE_URL is set:

    docker run -d --rm --name egx-mkt-pg -e POSTGRES_PASSWORD=pw -p 55433:5432 postgres:16
    TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55433/postgres \
        python -m pytest tests/test_market_schema_pg.py -v
    docker stop egx-mkt-pg

Never point this at a real database: it drops every table in the schema.
Skipped unless the URL host is local (see tests/pg_guard.py; PG_TESTS_ALLOW_REMOTE=1 overrides).
"""
import os

import psycopg
import pytest

from pg_guard import requires_local_pg

pytestmark = requires_local_pg


def _wipe(d):
    """An EMPTY database: drop every table init_db could have created."""
    for (t,) in d.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
    ).fetchall():
        d.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')


def _key(d, table, contype):
    """Column lists of the table's constraints of one type, in key order."""
    rows = d.execute(
        "SELECT c.conname, ARRAY(SELECT a.attname::text "
        "  FROM unnest(c.conkey) WITH ORDINALITY k(n, i) "
        "  JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.n "
        "  ORDER BY k.i) "
        "FROM pg_constraint c WHERE c.conrelid = %s::regclass AND c.contype = %s "
        "ORDER BY c.conname",
        (table, contype),
    ).fetchall()
    return {name: tuple(cols) for name, cols in rows}


@pytest.fixture
def pg(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    from app.core import db as db_mod
    monkeypatch.setattr(db_mod, "_pool", None)
    monkeypatch.setattr(db_mod, "_initialized", True)
    d = db_mod._DB(db_mod._get_pool())
    _wipe(d)
    return db_mod, d


@pytest.fixture
def d(pg):
    """The OLD shapes, exactly as production has them today."""
    db_mod, d = pg
    d.execute("CREATE TABLE watchlist (user_id TEXT NOT NULL, symbol TEXT NOT NULL, "
              "added_at TEXT NOT NULL, PRIMARY KEY (user_id, symbol))")
    d.execute("CREATE TABLE risk_snapshot (symbol TEXT PRIMARY KEY, measured_at TEXT NOT NULL, "
              "turnover_egp DOUBLE PRECISION)")
    d.execute("CREATE TABLE pe_data (symbol TEXT PRIMARY KEY, company_name TEXT, "
              "pe_ratio DOUBLE PRECISION, dividend_yield DOUBLE PRECISION, "
              "updated_at TEXT NOT NULL, value_traded_egp DOUBLE PRECISION)")
    d.execute("CREATE TABLE score_baseline (symbol TEXT NOT NULL, session_date TEXT NOT NULL, "
              "captured_at TEXT NOT NULL, PRIMARY KEY (symbol, session_date))")
    d.execute("CREATE TABLE alert_log (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
              "symbol TEXT NOT NULL, session_date TEXT NOT NULL, direction TEXT NOT NULL, "
              "status TEXT NOT NULL, created_at TEXT NOT NULL, "
              "UNIQUE (user_id, symbol, session_date, direction))")
    d.execute("INSERT INTO watchlist VALUES ('u1', 'COMI', '2026-01-01')")
    d.execute("INSERT INTO risk_snapshot VALUES ('COMI', '2026-01-01', 5e6)")
    d.execute("INSERT INTO pe_data (symbol, updated_at, value_traded_egp) "
              "VALUES ('COMI', '2026-01-01', 7e6)")
    d.execute("INSERT INTO score_baseline VALUES ('COMI', '2026-01-01', 'x')")
    d.execute("INSERT INTO alert_log VALUES ('a1', 'u1', 'COMI', '2026-01-01', 'up', "
              "'sent', 'x')")
    return db_mod, d


def test_old_unique_on_alert_log_has_the_name_postgres_gives_it(d):
    _, d = d
    assert _key(d, "alert_log", "u") == {
        "alert_log_user_id_symbol_session_date_direction_key":
            ("user_id", "symbol", "session_date", "direction")}


def test_existing_rows_survive_rekey(d):
    db_mod, d = d
    db_mod.init_db(d)
    db_mod.init_db(d)  # idempotent
    assert d.execute("SELECT user_id, market, symbol FROM watchlist").fetchall() == [("u1", "EG", "COMI")]
    assert d.execute("SELECT market, turnover FROM risk_snapshot").fetchall() == [("EG", 5e6)]
    assert d.execute("SELECT market, value_traded FROM pe_data").fetchall() == [("EG", 7e6)]
    assert d.execute("SELECT market, symbol FROM score_baseline").fetchall() == [("EG", "COMI")]
    assert d.execute("SELECT market, symbol FROM alert_log").fetchall() == [("EG", "COMI")]


def test_old_shape_is_rekeyed(d):
    db_mod, d = d
    db_mod.init_db(d)
    assert _key(d, "watchlist", "p") == {"watchlist_pkey": ("user_id", "market", "symbol")}
    assert _key(d, "risk_snapshot", "p") == {"risk_snapshot_pkey": ("market", "symbol")}
    assert _key(d, "pe_data", "p") == {"pe_data_pkey": ("market", "symbol")}
    assert _key(d, "score_baseline", "p") == {
        "score_baseline_pkey": ("market", "symbol", "session_date")}
    # The old 4-column UNIQUE is GONE, not left beside the new one: kept, it
    # would still forbid the same alert in a second market.
    assert _key(d, "alert_log", "u") == {
        "alert_log_market_unique": ("user_id", "market", "symbol", "session_date", "direction")}
    assert _key(d, "alert_log", "p") == {"alert_log_pkey": ("id",)}
    cols = {r[0] for r in d.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name IN ('risk_snapshot', 'pe_data')").fetchall()}
    assert "turnover_egp" not in cols and "value_traded_egp" not in cols


def test_same_symbol_in_two_markets(d):
    db_mod, d = d
    db_mod.init_db(d)
    d.execute("INSERT INTO watchlist (user_id, market, symbol, added_at) VALUES ('u1','US','COMI','x')")
    assert d.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0] == 2
    with pytest.raises(psycopg.errors.UniqueViolation):
        d.execute("INSERT INTO watchlist (user_id, market, symbol, added_at) VALUES ('u1','US','COMI','y')")
    d.execute("INSERT INTO alert_log (id, user_id, market, symbol, session_date, direction, "
              "status, created_at) VALUES ('a2','u1','US','COMI','2026-01-01','up','sent','x')")
    with pytest.raises(psycopg.errors.UniqueViolation):
        d.execute("INSERT INTO alert_log (id, user_id, market, symbol, session_date, direction, "
                  "status, created_at) VALUES ('a3','u1','US','COMI','2026-01-01','up','sent','x')")


def test_second_run_changes_nothing(d):
    """A second init_db must not drop and rebuild a key that is already right."""
    db_mod, d = d
    db_mod.init_db(d)

    def oids():
        return d.execute(
            "SELECT conrelid::regclass::text, conname, oid FROM pg_constraint "
            "WHERE contype IN ('p', 'u') AND connamespace = 'public'::regnamespace "
            "ORDER BY 1, 2").fetchall()

    before = oids()
    db_mod.init_db(d)
    assert oids() == before


def test_empty_database_twice(pg):
    db_mod, d = pg
    db_mod.init_db(d)
    db_mod.init_db(d)
    assert _key(d, "watchlist", "p") == {"watchlist_pkey": ("user_id", "market", "symbol")}
    assert _key(d, "discovered_tickers", "p") == {"discovered_tickers_pkey": ("market", "symbol")}
    assert _key(d, "dividend_events", "p") == {
        "dividend_events_pkey": ("market", "symbol", "ex_date")}
    assert _key(d, "fundamentals_annual", "p") == {
        "fundamentals_annual_pkey": ("market", "symbol", "fiscal_year")}
    assert _key(d, "alert_log", "u") == {
        "alert_log_market_unique": ("user_id", "market", "symbol", "session_date", "direction")}
    for t in ("portfolio", "portfolio_sales", "portfolio_dividends",
              "fundamentals_history", "market_regime"):
        assert d.execute(
            "SELECT column_default, is_nullable FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = 'market'", (t,)
        ).fetchone() == ("'EG'::text", "NO"), t
    cols = {r[0] for r in d.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name IN ('risk_snapshot', 'pe_data', 'discovered_tickers')").fetchall()}
    assert {"turnover", "value_traded", "exchange"} <= cols
    assert not any(c.endswith("_egp") for c in cols)


def test_writers_upsert_against_the_new_keys(pg):
    """Every ON CONFLICT target must name a real unique index after the rekey."""
    db_mod, d = pg
    db_mod.init_db(d)
    from app.core.card_snapshot import upsert_snapshot
    from app.core.dividend_history import upsert_dividends
    from app.core.fundamentals_annual import _upsert_batch

    card = {"categories": {}, "last_bar_date": "2026-01-02"}
    stats = {"turnover": 5e6, "last_price": 10.0}
    upsert_snapshot(d, "COMI", "2026-01-01T00:00", stats, True, True, card)
    upsert_snapshot(d, "COMI", "2026-01-02T00:00", stats, True, True, card)
    assert d.execute("SELECT market, symbol, turnover FROM risk_snapshot").fetchall() == [
        ("EG", "COMI", 5e6)]

    upsert_dividends(d, "COMI", [{"ex_date": "2026-01-01", "amount": 1.0}], "test")
    upsert_dividends(d, "COMI", [{"ex_date": "2026-01-01", "amount": 1.0}], "test")
    assert d.execute("SELECT COUNT(*) FROM dividend_events").fetchone()[0] == 1

    rec = {"symbol": "COMI", "fiscal_year": 2025, "eps_diluted": 1.0, "dps": None,
           "net_income": None, "revenue": None, "total_assets": None,
           "gross_profit": None, "total_debt": None, "first_usable_date": "2026-04-30"}
    _upsert_batch(d, [rec], "now")
    _upsert_batch(d, [rec], "now")
    assert d.execute("SELECT market FROM fundamentals_annual").fetchall() == [("EG",)]

    from app.core.pe_fetch import refresh_pe_data
    rows = [{"symbol": f"S{i}", "pe_ratio": 10.0, "value_traded": 1e6}
            for i in range(120)]
    assert refresh_pe_data(d, rows=rows)["success"] is True
    assert refresh_pe_data(d, rows=rows)["success"] is True
    assert d.execute(
        "SELECT COUNT(*), MIN(market), MAX(value_traded) FROM pe_data").fetchone() == (120, "EG", 1e6)


def test_every_on_conflict_target_in_the_code_is_a_real_unique_key(pg):
    """
    Postgres rejects an ON CONFLICT target that matches no unique index, so a
    writer still naming an OLD key would fail every write after the rekey.
    Walks the source rather than a hand-kept list, so a writer added later is
    checked too.
    """
    import re
    from pathlib import Path

    db_mod, d = pg
    db_mod.init_db(d)
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(
        r"INSERT INTO\s+(\w+)(?:(?!INSERT INTO).){0,1500}?ON CONFLICT\s*\(([^)]*)\)",
        re.S)
    found = []
    for path in [*(root / "app").rglob("*.py"), *(root / "scripts").rglob("*.py")]:
        if "vendor" in path.parts:
            continue
        for table, cols in pattern.findall(path.read_text(encoding="utf-8")):
            found.append((path.name, table, tuple(sorted(c.strip() for c in cols.split(",")))))
    assert len(found) >= 15

    unique = {
        (t, tuple(sorted(cols)))
        for t, cols in d.execute(
            "SELECT i.indrelid::regclass::text, ARRAY(SELECT a.attname::text "
            "  FROM unnest(i.indkey) k(n) JOIN pg_attribute a "
            "    ON a.attrelid = i.indrelid AND a.attnum = k.n) "
            "FROM pg_index i WHERE i.indisunique").fetchall()
    }
    missing = [f for f in found if (f[1], f[2]) not in unique]
    assert missing == []
