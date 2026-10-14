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
from app.core.markets import EG
from app.core.score_alerts import SUMMARY_SYMBOL, Subscriber, apply_daily_cap, detect_signals


def _categories(values) -> dict:
    return {name: values[i] for i, name in enumerate(CATEGORY_ORDER)}


def prune_baselines(db, today: str) -> None:
    cutoff = date.fromisoformat(today) - timedelta(days=ALERT_BASELINE_RETENTION_DAYS)
    db.execute("DELETE FROM score_baseline WHERE session_date < %s", (cutoff.isoformat(),))


def load_subscribers(db, symbols: list, market_code: str = EG.code) -> dict:
    """{symbol: [Subscriber]} for users who would hear about that symbol now.
    Watchlist and holdings are matched in `market_code` only: the same ticker
    on another market is a different stock."""
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
            "WHERE market = %s AND user_id = ANY(%s) AND symbol = ANY(%s) "
            "UNION SELECT user_id, symbol FROM portfolio "
            "WHERE market = %s AND user_id = ANY(%s) AND symbol = ANY(%s)",
            (market_code, watch_users, list(symbols),
             market_code, watch_users, list(symbols)),
        ).fetchall()
        follows = {(r[0], r[1]) for r in rows}

    out: dict = {}
    for user_id, scope, rise, fall in prefs:
        sub = Subscriber(user_id, get_weights_from_db(db, user_id), rise, fall)
        for symbol in symbols:
            if scope == "all" or (user_id, symbol) in follows:
                out.setdefault(symbol, []).append(sub)
    return out


def _read_baselines(db, scored: dict, market_code: str = EG.code) -> dict:
    rows = db.execute(
        f"SELECT symbol, session_date, {', '.join(CATEGORY_COLUMNS)} FROM score_baseline "
        "WHERE market = %s AND symbol = ANY(%s) AND session_date = ANY(%s)",
        (market_code, list(scored), sorted({v["session_date"] for v in scored.values()})),
    ).fetchall()
    return {(r[0], r[1]): _categories(r[2:]) for r in rows}


def run_detection(db, scored: dict, macro: Optional[dict], now: str,
                  new_id=lambda: uuid4().hex, market_code: str = EG.code) -> int:
    """Queue alert_log rows for every qualifying move among `scored` symbols,
    all of which belong to `market_code`."""
    if not scored:
        return 0
    subscribers = load_subscribers(db, list(scored), market_code)
    if not subscribers:
        return 0
    baselines = _read_baselines(db, scored, market_code)

    by_session: dict = {}
    for symbol, info in scored.items():
        by_session.setdefault(info["session_date"], []).append(symbol)

    inserted = 0
    for session, symbols in sorted(by_session.items()):
        logged = db.execute(
            "SELECT user_id, symbol, direction FROM alert_log "
            "WHERE session_date = %s AND market = %s AND symbol = ANY(%s)",
            (session, market_code, symbols + [SUMMARY_SYMBOL]),
        ).fetchall()
        already = {(r[0], r[1], r[2]) for r in logged}
        counts = db.execute(
            "SELECT user_id, COUNT(*) FROM alert_log WHERE session_date = %s "
            "AND status IN ('pending', 'sent') AND direction <> 'summary' GROUP BY user_id",
            (session,),
        ).fetchall()
        sent_today = {r[0]: r[1] for r in counts}
        summary_sent = {u for (u, s, d) in already if d == "summary"}

        signals = []
        for symbol in symbols:
            baseline = baselines.get((symbol, session))
            if baseline is None:
                continue
            signals += detect_signals(symbol, session, baseline,
                                      scored[symbol]["categories"], macro,
                                      subscribers.get(symbol, []), already)

        for sig, status in apply_daily_cap(signals, sent_today, summary_sent):
            row = db.execute(
                "INSERT INTO alert_log (id, user_id, market, symbol, session_date, "
                "direction, score_before, score_after, status, attempts, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, %s) "
                "ON CONFLICT (user_id, market, symbol, session_date, direction) "
                "DO NOTHING RETURNING id",
                (new_id(), sig.user_id, market_code, sig.symbol, sig.session_date, sig.direction,
                 sig.before, sig.after, status, now),
            ).fetchone()
            if row:
                inserted += 1
    return inserted
