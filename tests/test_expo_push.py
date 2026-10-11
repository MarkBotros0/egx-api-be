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
        ("WHERE status = 'pending' ORDER BY", rows),
        ("FROM push_devices", tokens),
    ])


def _status_updates(db):
    return [(s, p) for s, p in db.log if s.startswith("UPDATE alert_log SET")]


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
    assert any("status = 'sent'" in s and p[-1] == "a" for s, p in _status_updates(db))


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
    updates = _status_updates(db)
    assert any("attempts = attempts + 1" in s and "'pending'" in s and p[-1] == "a" for s, p in updates)
    assert any("'failed'" in s and p[-1] == "b" for s, p in updates)
    assert out["failed"] == 1


def test_a_user_with_no_device_fails_the_row_without_sending():
    calls = []
    db = _db([_row("a")], [])
    deliver_pending(db, DAY, sender=lambda m: calls.append(m) or [])
    assert calls == []
    assert any("'failed'" in s and p == ("a",) for s, p in _status_updates(db))


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
    assert any("status = 'pending', attempts = attempts + 1" in s and p == ("a",)
               for s, p in _status_updates(db))


def test_non_list_response_is_treated_as_no_tickets():
    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: {"errors": [{"code": "X"}]})
    assert out["sent"] == 0
    assert any("attempts = attempts + 1" in s and p == ("a",) for s, p in _status_updates(db))


def test_short_ticket_list_is_padded_and_the_unticketed_row_is_bumped():
    rows = [_row("a", symbol="COMI"), _row("b", symbol="FWRY")]
    db = _db(rows, [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: [{"status": "ok", "id": "t"}])
    assert out["sent"] == 1
    updates = _status_updates(db)
    assert any("status = 'sent'" in s and p[-1] == "a" for s, p in updates)
    assert any("attempts = attempts + 1" in s and p == ("b",) for s, p in updates)


def test_non_dict_ticket_entry_counts_as_a_failed_attempt():
    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    out = deliver_pending(db, DAY, sender=lambda m: ["oops"])
    assert out["sent"] == 0
    assert any("attempts = attempts + 1" in s and p == ("a",) for s, p in _status_updates(db))


def test_sender_exception_is_logged(capsys):
    def boom(_):
        raise OSError("expo down")

    db = _db([_row("a")], [("ExponentPushToken[x]", "u1")])
    deliver_pending(db, DAY, sender=boom)
    assert "[alerts] expo send failed: expo down" in capsys.readouterr().out
