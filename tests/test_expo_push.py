"""Score alerts — Expo delivery. The sender is injected; no network."""

from __future__ import annotations

from datetime import datetime, timezone

from app.core import expo_push
from app.core.expo_push import deliver_pending
from alert_fakes import FakeDB

DAY = datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc)      # 12:00 Cairo
NIGHT = datetime(2026, 10, 11, 21, 0, tzinfo=timezone.utc)   # 00:00 Cairo


def _row(id_, user="u1", symbol="COMI", direction="up", attempts=0):
    return (id_, user, symbol, "2026-10-11", direction, 58.0, 70.0, attempts)


def _db(rows, tokens):
    return FakeDB([
        ("MAX(last_bar_date)", [("2026-10-11",)]),
        ("SET status = 'expired'", []),
        ("SET status = 'sending'", rows),        # the claim
        ("FROM push_devices", tokens),
    ])


def _status_updates(db):
    return [(s, p) for s, p in db.log if s.startswith("UPDATE alert_log SET")]


def _ids(db, needle):
    """Every alert_log id moved by an UPDATE whose SQL contains `needle`."""
    out = []
    for s, p in _status_updates(db):
        if needle in s and "id = ANY(%s)" in s:
            out += list(p[-1])
    return out


def test_quiet_hours_hold_everything_without_a_query():
    db = FakeDB()
    out = deliver_pending(db, NIGHT, sender=lambda m: [])
    assert out["held"] is True and db.log == []


def test_ok_ticket_marks_the_row_sent_and_message_is_shaped_for_expo():
    sent = []
    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: sent.extend(m) or [{"status": "ok", "id": "t"}])
    assert out["sent"] == 1
    [msg] = sent
    assert msg["to"] == "ExponentPushToken[x]"
    assert msg["title"] == "COMI score up 12 points"
    assert msg["data"] == {"type": "score_alert", "symbol": "COMI", "session": "2026-10-11"}
    assert msg["channelId"] == "score-alerts"
    assert _ids(db, "status = 'sent'") == ["a"]


def test_device_not_registered_deletes_the_token():
    db = _db([_row("a")], [("ExponentPushToken[dead]", "u1")])
    deliver_pending(db, DAY, sender=lambda m: [
        {"status": "error", "message": "x", "details": {"error": "DeviceNotRegistered"}}
    ])
    assert db.statements("DELETE FROM push_devices WHERE token = %s")[0][1] == ("ExponentPushToken[dead]",)


def test_a_failed_send_retries_then_gives_up_at_three_attempts():
    def boom(_):
        raise OSError("expo down")

    db = _db([_row("a", attempts=0), _row("b", symbol="FWRY", attempts=2)],
             [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=boom)
    assert _ids(db, "status = 'pending', attempts = attempts + 1") == ["a"]
    assert _ids(db, "status = 'failed', attempts = attempts + 1") == ["b"]
    assert out["failed"] == 1


def test_a_user_with_no_device_fails_the_row_without_sending():
    calls = []
    db = _db([_row("a")], [])
    deliver_pending(db, DAY, sender=lambda m: calls.append(m) or [])
    assert calls == []
    assert _ids(db, "status = 'failed'") == ["a"]


def test_stale_sessions_expire_before_sending():
    db = _db([], [])
    deliver_pending(db, DAY, sender=lambda m: [])
    [(sql, params)] = db.statements("SET status = 'expired'")
    assert "session_date < %s" in sql and params == ("2026-10-11",)


def test_messages_are_batched_at_one_hundred():
    batches = []
    rows = [_row(f"r{i}", symbol=f"S{i}") for i in range(150)]
    db = _db(rows, [("ExponentPushToken[x]", "u1")])
    deliver_pending(db, DAY, sender=lambda m: batches.append(len(m)) or [{"status": "ok"}] * len(m))
    assert batches == [100, 50]


def test_send_expo_posts_json_and_returns_tickets(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"data": [{"status": "ok", "id": "1"}]}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["timeout"] = timeout
        captured["ctype"] = req.get_header("Content-type")
        return _Resp()

    monkeypatch.setattr(expo_push.urllib.request, "urlopen", fake_urlopen)
    tickets = expo_push.send_expo([{"to": "ExponentPushToken[x]", "title": "t", "body": "b"}])
    assert tickets == [{"status": "ok", "id": "1"}]
    assert captured["url"] == "https://exp.host/--/api/v2/push/send"
    assert captured["ctype"] == "application/json"
    assert captured["timeout"] == 3.0


def test_empty_ticket_list_bumps_attempts_instead_of_stranding_the_row():
    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: [])
    assert out["sent"] == 0 and out["failed"] == 0
    assert _ids(db, "status = 'pending', attempts = attempts + 1") == ["a"]


def test_non_list_response_is_treated_as_no_tickets():
    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: {"errors": [{"code": "X"}]})
    assert out["sent"] == 0
    assert _ids(db, "attempts = attempts + 1") == ["a"]


def test_short_ticket_list_is_padded_and_the_unticketed_row_is_bumped():
    rows = [_row("a", symbol="COMI"), _row("b", symbol="FWRY")]
    db = _db(rows, [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: [{"status": "ok", "id": "t"}])
    assert out["sent"] == 1
    assert _ids(db, "status = 'sent'") == ["a"]
    assert _ids(db, "attempts = attempts + 1") == ["b"]


def test_non_dict_ticket_entry_counts_as_a_failed_attempt():
    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: ["oops"])
    assert out["sent"] == 0
    assert _ids(db, "attempts = attempts + 1") == ["a"]


def test_sender_exception_is_logged(capsys):
    def boom(_):
        raise OSError("expo down")

    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    deliver_pending(db, DAY, sender=boom)
    assert "[alerts] expo send failed: expo down" in capsys.readouterr().out


# --- delivery claim (two overlapping cron calls must not both send a row) ---

def test_rows_are_claimed_with_skip_locked_newest_session_first():
    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    deliver_pending(db, DAY, sender=lambda m: [{"status": "ok"}] * len(m))
    [(sql, params)] = db.statements("SET status = 'sending'")
    assert sql.startswith("UPDATE alert_log SET status = 'sending', claimed_at = %s")
    assert "WHERE status = 'pending'" in sql
    assert "ORDER BY session_date DESC, created_at LIMIT 200" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert sql.rstrip().endswith(
        "RETURNING id, user_id, symbol, session_date, direction, "
        "score_before, score_after, attempts")
    assert params == (DAY.isoformat(),)


def test_stale_sending_rows_are_reclaimed_before_expiry_and_the_claim():
    """Reclaim first, so a reclaimed row from an older session expires this call."""
    db = _db([], [])
    deliver_pending(db, DAY, sender=lambda m: [])
    sqls = [s for s, _ in db.log]
    reclaim = next(i for i, s in enumerate(sqls) if "WHERE status = 'sending'" in s)
    expire = next(i for i, s in enumerate(sqls) if "SET status = 'expired'" in s)
    claim = next(i for i, s in enumerate(sqls) if "SET status = 'sending'" in s)
    assert reclaim < expire < claim
    sql, params = db.log[reclaim]
    assert "claimed_at < %s" in sql
    assert params == (3, "2026-10-11T08:58:00+00:00")


def test_reclaim_counts_the_lost_attempt_and_fails_at_the_limit():
    db = _db([], [])
    deliver_pending(db, DAY, sender=lambda m: [])
    [(sql, _)] = [(s, p) for s, p in db.log if "WHERE status = 'sending'" in s]
    assert "attempts = attempts + 1" in sql
    assert ("status = CASE WHEN attempts + 1 >= %s THEN 'failed' ELSE 'pending' END"
            in sql)
    assert "claimed_at = NULL" in sql


def test_a_second_overlapping_call_that_claims_nothing_sends_nothing():
    calls = []
    db = _db([], [("ExponentPushToken[x]", "u1")])   # the first call holds the rows
    out = deliver_pending(db, DAY, sender=lambda m: calls.append(m) or [])
    assert calls == [] and out["sent"] == 0
    assert db.statements("FROM push_devices") == []


def test_results_are_marked_per_batch_not_after_all_batches():
    rows = [_row(f"r{i}", symbol=f"S{i}") for i in range(150)]
    db = _db(rows, [("ExponentPushToken[x]", "u1")])
    seen = []

    def sender(m):
        seen.append(list(_ids(db, "status = 'sent'")))
        return [{"status": "ok"}] * len(m)

    out = deliver_pending(db, DAY, sender=sender)
    assert seen[0] == []                          # nothing marked before batch 1
    assert seen[1] == [f"r{i}" for i in range(100)]   # batch 1 marked before batch 2
    assert out["sent"] == 150
    sent_updates = [s for s, _ in _status_updates(db) if "status = 'sent'" in s]
    assert len(sent_updates) == 2, "one UPDATE per batch, not one per row"


def test_no_batch_starts_past_the_deadline_and_unstarted_rows_are_released():
    rows = [_row(f"r{i}", symbol=f"S{i}") for i in range(150)]
    db = _db(rows, [("ExponentPushToken[x]", "u1")])
    ticks = iter([0.0, 30.0, 30.0, 30.0])
    batches = []
    out = deliver_pending(db, DAY,
                          sender=lambda m: batches.append(len(m)) or [{"status": "ok"}] * len(m),
                          deadline=10.0, clock=lambda: next(ticks))
    assert batches == [100]
    assert out["sent"] == 100
    released = [(s, p) for s, p in _status_updates(db)
                if "status = 'pending', claimed_at = NULL" in s]
    [(sql, params)] = released
    assert "attempts" not in sql, "an unstarted row was never attempted"
    assert list(params[-1]) == [f"r{i}" for i in range(100, 150)]
