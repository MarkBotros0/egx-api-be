"""
Score alerts — the Postgres glue between the scoring cron and the pure
decision layer in `score_alerts`. Every function takes `db` so tests run on a
fake; nothing here talks to the network.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from app.core.card_snapshot import CATEGORY_COLUMNS
from app.core.composite import CATEGORY_ORDER
from app.core.constants import ALERT_BASELINE_RETENTION_DAYS


def _categories(values) -> dict:
    return {name: values[i] for i, name in enumerate(CATEGORY_ORDER)}


def read_previous_rows(db, symbols: list) -> dict:
    """The snapshot rows about to be overwritten: bar date + eight categories."""
    if not symbols:
        return {}
    rows = db.execute(
        f"SELECT symbol, last_bar_date, {', '.join(CATEGORY_COLUMNS)} "
        "FROM risk_snapshot WHERE symbol = ANY(%s)",
        (list(symbols),),
    ).fetchall()
    return {r[0]: {"last_bar_date": r[1], "categories": _categories(r[2:])} for r in rows}


def capture_baseline(db, symbol: str, previous: Optional[dict],
                     card: Optional[dict], now: str) -> bool:
    """
    Store the previous session's close scores as this session's baseline, the
    first time a rescore carries a new bar date. First capture wins; never
    guesses a baseline from an unscored row.
    """
    if card is None or previous is None:
        return False
    session = card.get("last_bar_date")
    if not session or previous.get("last_bar_date") == session:
        return False
    cats = previous.get("categories") or {}
    if not any(cats.get(name) is not None for name in CATEGORY_ORDER):
        return False
    db.execute(
        f"INSERT INTO score_baseline (symbol, session_date, {', '.join(CATEGORY_COLUMNS)}, "
        "source_bar_date, captured_at) "
        f"VALUES ({', '.join(['%s'] * (len(CATEGORY_COLUMNS) + 4))}) "
        "ON CONFLICT (symbol, session_date) DO NOTHING",
        (symbol, session) + tuple(cats.get(name) for name in CATEGORY_ORDER)
        + (previous.get("last_bar_date"), now),
    )
    return True


def prune_baselines(db, today: str) -> None:
    cutoff = date.fromisoformat(today) - timedelta(days=ALERT_BASELINE_RETENTION_DAYS)
    db.execute("DELETE FROM score_baseline WHERE session_date < %s", (cutoff.isoformat(),))
