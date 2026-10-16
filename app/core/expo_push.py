"""
Score alerts — delivery through the Expo Push Service (FCM on Android, APNs on
iOS). Free, no SDK: one JSON POST per batch of <=100 messages.

Runs inside the scoring cron's time budget, so every failure is soft: a row
that could not be sent stays pending for the next call, and gives up after
ALERT_MAX_ATTEMPTS. Receipts are not polled; DeviceNotRegistered on the ticket
already catches the common dead-token case.

DELIVERY CLAIM. Overlapping cron calls (two scheduled jobs on the same minute,
a manual trigger) would otherwise both read the same pending rows and push
them twice. Rows are CLAIMED first — one UPDATE ... FOR UPDATE SKIP LOCKED that
moves them to 'sending' — so a concurrent call claims nothing it does not own.
Results are written per Expo batch, which narrows the window in which a killed
call has sent a batch but not recorded it; a row left in 'sending' by such a
call is reclaimed after CLAIM_STALE_SECONDS.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

from app.core.constants import (
    ALERT_CHANNEL_ID,
    ALERT_MAX_ATTEMPTS,
    EXPO_BATCH_SIZE,
    EXPO_PUSH_URL,
    EXPO_TIMEOUT_SECONDS,
)
from app.core.markets import EG, MarketConfig
from app.core.score_alerts import Signal, alert_copy, in_quiet_hours

_PENDING_LIMIT = 200

# A 'sending' row older than this belongs to a call that died mid-delivery
# (Vercel kills at 30s, so two minutes is far past any live call).
CLAIM_STALE_SECONDS = 120


def send_expo(messages: list, timeout: float = EXPO_TIMEOUT_SECONDS) -> list:
    req = urllib.request.Request(
        EXPO_PUSH_URL,
        data=json.dumps(messages).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("data") or []


def _mark(db, sql: str, ids: list, *params) -> None:
    if ids:
        db.execute(f"UPDATE alert_log SET {sql} WHERE id = ANY(%s)", params + (ids,))


def _reclaim(db, now_utc: datetime, market_code: str = EG.code) -> None:
    """
    Rows left in 'sending' by a call that died. The lost try counts as an
    attempt, so a row that kills every call still ends at 'failed'.
    """
    stale = (now_utc - timedelta(seconds=CLAIM_STALE_SECONDS)).isoformat()
    db.execute(
        "UPDATE alert_log SET attempts = attempts + 1, "
        "status = CASE WHEN attempts + 1 >= %s THEN 'failed' ELSE 'pending' END, "
        "claimed_at = NULL WHERE status = 'sending' AND claimed_at < %s AND market = %s",
        (ALERT_MAX_ATTEMPTS, stale, market_code),
    )


def _claim(db, now_utc: datetime, market_code: str = EG.code) -> list:
    now = now_utc.isoformat()
    rows = db.execute(
        "UPDATE alert_log SET status = 'sending', claimed_at = %s "
        "WHERE id IN (SELECT id FROM alert_log WHERE status = 'pending' AND market = %s "
        f"ORDER BY session_date DESC, created_at LIMIT {_PENDING_LIMIT} "
        "FOR UPDATE SKIP LOCKED) "
        "RETURNING id, user_id, symbol, session_date, direction, "
        "score_before, score_after, attempts",
        (now, market_code),
    ).fetchall()
    # RETURNING has no order; newest session first (stable for the rest).
    return sorted(rows, key=lambda r: r[3], reverse=True)


def deliver_pending(db, now_utc: datetime, sender=send_expo,
                    deadline: Optional[float] = None,
                    clock=time.monotonic, market: MarketConfig = EG) -> dict:
    """
    Send this call's share of `market`'s queue. `deadline` is a `clock()` value
    past which no new Expo batch starts; rows claimed but not started go
    back to 'pending' without an attempt counted.

    Everything is scoped to the market: quiet hours are in its timezone, its
    latest session comes from its own risk_snapshot rows, and only its rows are
    reclaimed, expired and claimed — so the US cron neither sends EG alerts
    during Cairo's night nor expires them against a New York session date.
    """
    result = {"sent": 0, "failed": 0, "expired": 0, "held": False}
    if in_quiet_hours(now_utc, market):
        result["held"] = True
        return result

    # Before expiry, so a reclaimed row from an older session expires now.
    _reclaim(db, now_utc, market.code)

    latest = (db.execute("SELECT MAX(last_bar_date) FROM risk_snapshot WHERE market = %s",
                         (market.code,)).fetchone() or [None])[0]
    if latest:
        expired = db.execute(
            "UPDATE alert_log SET status = 'expired' "
            "WHERE status = 'pending' AND market = %s AND session_date < %s RETURNING id",
            (market.code, latest),
        ).fetchall()
        result["expired"] = len(expired)

    rows = _claim(db, now_utc, market.code)
    if not rows:
        return result

    users = sorted({r[1] for r in rows})
    tokens: dict = {}
    for token, user_id in db.execute(
        "SELECT token, user_id FROM push_devices WHERE user_id = ANY(%s)", (users,)
    ).fetchall():
        tokens.setdefault(user_id, []).append(token)

    # Batches of whole rows (a row's messages never straddle two batches), so
    # each row's outcome is known when its batch returns.
    batches: list = []                 # [(messages, owners)]; owners[i] = (row, token)
    no_device = []
    for row in rows:
        row_id, user_id, symbol, session, direction, before, after, _ = row
        if not tokens.get(user_id):
            no_device.append(row_id)
            continue
        title, body = alert_copy(Signal(user_id, symbol, session, direction,
                                        before or 0.0, after or 0.0), market)
        msgs = [{
            "to": token, "title": title, "body": body,
            # `market` lets the app open the stock on ITS market (a deep link
            # never switches the app's selected market).
            "data": {"type": "score_alert", "symbol": symbol, "session": session,
                     "market": market.code},
            "channelId": ALERT_CHANNEL_ID, "sound": None,
        } for token in tokens[user_id]]
        if not batches or len(batches[-1][0]) + len(msgs) > EXPO_BATCH_SIZE:
            batches.append(([], []))
        batches[-1][0].extend(msgs)
        batches[-1][1].extend((row, t) for t in tokens[user_id])

    _mark(db, "status = 'failed'", no_device)
    result["failed"] += len(no_device)

    now = now_utc.isoformat()
    for index, (chunk, chunk_owners) in enumerate(batches):
        if deadline is not None and clock() >= deadline:
            unstarted = []
            for _, owners in batches[index:]:
                for row, _ in owners:
                    if row[0] not in unstarted:
                        unstarted.append(row[0])
            _mark(db, "status = 'pending', claimed_at = NULL", unstarted)
            break
        try:
            tickets = sender(chunk)
        except Exception as e:
            print(f"[alerts] expo send failed: {e}")
            tickets = None
        # Normalise to exactly one entry per message: anything short, long or
        # non-list is padded/truncated, so every owner gets an outcome and an
        # unticketed row is bumped rather than retried forever.
        if not isinstance(tickets, list):
            tickets = []
        tickets = (tickets[:len(chunk)] + [None] * len(chunk))[:len(chunk)]
        outcome: dict = {}             # row_id -> (row, any_ok), insertion-ordered
        for (row, token), ticket in zip(chunk_owners, tickets):
            ok = isinstance(ticket, dict) and ticket.get("status") == "ok"
            details = ticket.get("details") if isinstance(ticket, dict) else None
            if isinstance(details, dict) and details.get("error") == "DeviceNotRegistered":
                db.execute("DELETE FROM push_devices WHERE token = %s", (token,))
            prev = outcome.get(row[0], (row, False))
            outcome[row[0]] = (row, prev[1] or ok)

        sent = [rid for rid, (_, ok) in outcome.items() if ok]
        retry = [rid for rid, (row, ok) in outcome.items()
                 if not ok and row[7] + 1 < ALERT_MAX_ATTEMPTS]
        give_up = [rid for rid, (row, ok) in outcome.items()
                   if not ok and row[7] + 1 >= ALERT_MAX_ATTEMPTS]
        _mark(db, "status = 'sent', sent_at = %s", sent, now)
        _mark(db, "status = 'pending', attempts = attempts + 1, claimed_at = NULL", retry)
        _mark(db, "status = 'failed', attempts = attempts + 1", give_up)
        result["sent"] += len(sent)
        result["failed"] += len(give_up)
    return result
