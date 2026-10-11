"""Score alerts — storage glue. No Postgres (see alert_fakes.FakeDB)."""

from __future__ import annotations

from alert_fakes import FakeDB
from app.core.card_snapshot import CATEGORY_COLUMNS


def test_init_db_creates_the_four_alert_tables():
    from app.core.db import init_db

    db = FakeDB()
    init_db(db)
    sql = " ".join(s for s, _ in db.log)
    for table in ("push_devices", "alert_preferences", "score_baseline", "alert_log"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, table
    assert "UNIQUE (user_id, symbol, session_date, direction)" in sql
    assert "PRIMARY KEY (symbol, session_date)" in sql
    for column in CATEGORY_COLUMNS:
        assert f"{column} DOUBLE PRECISION" in sql
