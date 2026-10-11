"""
Score alerts — delivery through the Expo Push Service (FCM on Android, APNs on
iOS). Free, no SDK: one JSON POST per batch of <=100 messages.

Runs inside the scoring cron's time budget, so every failure is soft: a row
that could not be sent stays pending for the next call, and gives up after
ALERT_MAX_ATTEMPTS. Receipts are not polled; DeviceNotRegistered on the ticket
already catches the common dead-token case.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime

from app.core.constants import (
    ALERT_CHANNEL_ID,
    ALERT_MAX_ATTEMPTS,
    EXPO_BATCH_SIZE,
    EXPO_PUSH_URL,
    EXPO_TIMEOUT_SECONDS,
)
from app.core.score_alerts import Signal, alert_copy, in_quiet_hours

_PENDING_LIMIT = 200


def send_expo(messages: list, timeout: float = EXPO_TIMEOUT_SECONDS) -> list:
    req = urllib.request.Request(
        EXPO_PUSH_URL,
        data=json.dumps(messages).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("data") or []


def _bump(db, row_id: str, attempts: int) -> bool:
    """attempts+1; True when the row is now failed for good."""
    if attempts + 1 >= ALERT_MAX_ATTEMPTS:
        db.execute("UPDATE alert_log SET status = 'failed', attempts = attempts + 1 "
                   "WHERE id = %s", (row_id,))
        return True
    db.execute("UPDATE alert_log SET status = 'pending', attempts = attempts + 1 "
               "WHERE id = %s", (row_id,))
    return False


def deliver_pending(db, now_utc: datetime, sender=send_expo) -> dict:
    result = {"sent": 0, "failed": 0, "expired": 0, "held": False}
    if in_quiet_hours(now_utc):
        result["held"] = True
        return result

    latest = (db.execute("SELECT MAX(last_bar_date) FROM risk_snapshot").fetchone() or [None])[0]
    if latest:
        expired = db.execute(
            "UPDATE alert_log SET status = 'expired' "
            "WHERE status = 'pending' AND session_date < %s RETURNING id",
            (latest,),
        ).fetchall()
        result["expired"] = len(expired)

    rows = db.execute(
        "SELECT id, user_id, symbol, session_date, direction, score_before, "
        "score_after, attempts FROM alert_log "
        f"WHERE status = 'pending' ORDER BY created_at LIMIT {_PENDING_LIMIT}"
    ).fetchall()
    if not rows:
        return result

    users = sorted({r[1] for r in rows})
    tokens: dict = {}
    for token, user_id in db.execute(
        "SELECT token, user_id FROM push_devices WHERE user_id = ANY(%s)", (users,)
    ).fetchall():
        tokens.setdefault(user_id, []).append(token)

    messages, owners = [], []          # owners[i] = (row, token) for messages[i]
    for row in rows:
        row_id, user_id, symbol, session, direction, before, after, _ = row
        if not tokens.get(user_id):
            db.execute("UPDATE alert_log SET status = 'failed' WHERE id = %s", (row_id,))
            result["failed"] += 1
            continue
        title, body = alert_copy(Signal(user_id, symbol, session, direction,
                                        before or 0.0, after or 0.0))
        for token in tokens[user_id]:
            messages.append({
                "to": token, "title": title, "body": body,
                "data": {"type": "score_alert", "symbol": symbol, "session": session},
                "channelId": ALERT_CHANNEL_ID, "sound": None,
            })
            owners.append((row, token))

    outcome: dict = {}                 # row_id -> (row, any_ok)
    now = now_utc.isoformat()
    for start in range(0, len(messages), EXPO_BATCH_SIZE):
        chunk = messages[start:start + EXPO_BATCH_SIZE]
        chunk_owners = owners[start:start + EXPO_BATCH_SIZE]
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
        for (row, token), ticket in zip(chunk_owners, tickets):
            ok = isinstance(ticket, dict) and ticket.get("status") == "ok"
            details = ticket.get("details") if isinstance(ticket, dict) else None
            if isinstance(details, dict) and details.get("error") == "DeviceNotRegistered":
                db.execute("DELETE FROM push_devices WHERE token = %s", (token,))
            prev = outcome.get(row[0], (row, False))
            outcome[row[0]] = (row, prev[1] or ok)

    for row_id, (row, any_ok) in outcome.items():
        if any_ok:
            db.execute("UPDATE alert_log SET status = 'sent', sent_at = %s WHERE id = %s",
                       (now, row_id))
            result["sent"] += 1
        elif _bump(db, row_id, row[7]):
            result["failed"] += 1
    return result
