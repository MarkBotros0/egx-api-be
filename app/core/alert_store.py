"""
Score alerts — the Postgres glue between the scoring cron and the pure
decision layer in `score_alerts`. Every function takes `db` so tests run on a
fake; nothing here talks to the network.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
from uuid import uuid4

from app.core.card_snapshot import CATEGORY_COLUMNS
from app.core.composite import CATEGORY_ORDER, get_weights_from_db
from app.core.constants import ALERT_BASELINE_RETENTION_DAYS
from app.core.score_alerts import SUMMARY_SYMBOL, Subscriber, apply_daily_cap, detect_signals


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


def load_subscribers(db, symbols: list) -> dict:
    """{symbol: [Subscriber]} for users who would hear about that symbol now."""
    if not symbols:
        return {}
    prefs = db.execute(
        "SELECT p.user_id, p.scope, p.rise_threshold, p.fall_threshold "
        "FROM alert_preferences p JOIN users u ON u.id = p.user_id "
        "WHERE p.enabled AND u.is_active "
        "AND EXISTS (SELECT 1 FROM push_devices d WHERE d.user_id = p.user_id)"
    ).fetchall()
    if not prefs:
        return {}

    watch_users = [r[0] for r in prefs if r[1] != "all"]
    follows: set = set()
    if watch_users:
        rows = db.execute(
            "SELECT user_id, symbol FROM watchlist "
            "WHERE user_id = ANY(%s) AND symbol = ANY(%s) "
            "UNION SELECT user_id, symbol FROM portfolio "
            "WHERE user_id = ANY(%s) AND symbol = ANY(%s)",
            (watch_users, list(symbols), watch_users, list(symbols)),
        ).fetchall()
        follows = {(r[0], r[1]) for r in rows}

    out: dict = {}
    for user_id, scope, rise, fall in prefs:
        sub = Subscriber(user_id, get_weights_from_db(db, user_id), rise, fall)
        for symbol in symbols:
            if scope == "all" or (user_id, symbol) in follows:
                out.setdefault(symbol, []).append(sub)
    return out


def _read_baselines(db, scored: dict) -> dict:
    rows = db.execute(
        f"SELECT symbol, session_date, {', '.join(CATEGORY_COLUMNS)} FROM score_baseline "
        "WHERE symbol = ANY(%s) AND session_date = ANY(%s)",
        (list(scored), sorted({v["session_date"] for v in scored.values()})),
    ).fetchall()
    return {(r[0], r[1]): _categories(r[2:]) for r in rows}


def run_detection(db, scored: dict, macro: Optional[dict], now: str,
                  new_id=lambda: uuid4().hex) -> int:
    """Queue alert_log rows for every qualifying move among `scored` symbols."""
    if not scored:
        return 0
    subscribers = load_subscribers(db, list(scored))
    if not subscribers:
        return 0
    baselines = _read_baselines(db, scored)

    sessions = sorted({v["session_date"] for v in scored.values()})
    logged = db.execute(
        "SELECT user_id, symbol, direction FROM alert_log "
        "WHERE session_date = ANY(%s) AND symbol = ANY(%s)",
        (sessions, list(scored) + [SUMMARY_SYMBOL]),
    ).fetchall()
    already = {(r[0], r[1], r[2]) for r in logged}
    counts = db.execute(
        "SELECT user_id, COUNT(*) FROM alert_log WHERE session_date = ANY(%s) "
        "AND status IN ('pending', 'sent') AND direction <> 'summary' GROUP BY user_id",
        (sessions,),
    ).fetchall()
    sent_today = {r[0]: r[1] for r in counts}
    summary_sent = {u for (u, s, d) in already if d == "summary"}

    signals = []
    for symbol, info in scored.items():
        baseline = baselines.get((symbol, info["session_date"]))
        if baseline is None:
            continue
        signals += detect_signals(symbol, info["session_date"], baseline,
                                  info["categories"], macro,
                                  subscribers.get(symbol, []), already)

    inserted = 0
    for sig, status in apply_daily_cap(signals, sent_today, summary_sent):
        db.execute(
            "INSERT INTO alert_log (id, user_id, symbol, session_date, direction, "
            "score_before, score_after, status, attempts, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, %s) "
            "ON CONFLICT (user_id, symbol, session_date, direction) DO NOTHING",
            (new_id(), sig.user_id, sig.symbol, sig.session_date, sig.direction,
             sig.before, sig.after, status, now),
        )
        inserted += 1
    return inserted
