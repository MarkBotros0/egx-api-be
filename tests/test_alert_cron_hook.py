"""Score alerts ride the scoring cron and can never break it."""

from __future__ import annotations

import inspect
from datetime import datetime

from alert_fakes import FakeDB
from app.core import card_snapshot, macro_fetch
from app.routers import cron


class _DB:
    def execute(self, *a, **k):
        raise AssertionError("no DB expected")


def test_alerts_are_skipped_past_the_start_limit():
    out = cron.run_alerts(_DB(), {"COMI": {}}, "2026-10-11T09:00:00+00:00", 22.5)
    assert out["skipped"] == "deadline" and out["queued"] == 0


def test_an_exception_in_alerts_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("detector broke")

    monkeypatch.setattr(cron, "read_cached_macro", lambda db: None)
    monkeypatch.setattr(cron, "run_detection", boom)
    out = cron.run_alerts(object(), {"COMI": {}}, "2026-10-11T09:00:00+00:00", 5.0)
    assert out["error"] == "RuntimeError"


def test_quiet_hours_are_reported(monkeypatch):
    monkeypatch.setattr(cron, "run_detection", lambda *a, **k: 0)
    monkeypatch.setattr(cron, "prune_baselines", lambda *a, **k: None)
    out = cron.run_alerts(object(), {}, "2026-10-11T21:00:00+00:00", 5.0,
                          deliver=lambda db, now, **kw: {"sent": 0, "held": True})
    assert out["skipped"] == "quiet-hours"


def test_baseline_capture_lives_in_the_shared_writer_not_the_cron():
    """
    Capture rides upsert_snapshot, so the dashboard's live upgrade cannot
    overwrite a row without first saving the previous session's scores.
    """
    src = inspect.getsource(cron.risk_snapshot)
    assert "capture_baseline" not in src and "read_previous_rows" not in src
    assert "_upsert(symbol, stats, is_tradeable" in src
    assert "run_alerts(" in src and '"alerts": alerts' in src
    writer = inspect.getsource(card_snapshot.upsert_snapshot)
    assert writer.index("_capture_baseline(") < writer.index("INSERT INTO risk_snapshot")


def test_read_cached_macro_never_touches_the_network(monkeypatch):
    def no_net(*a, **k):
        raise AssertionError("network called")

    monkeypatch.setattr(macro_fetch, "_fetch_egx30", no_net)
    monkeypatch.setattr(macro_fetch, "_fetch_usdegp", no_net)
    monkeypatch.setattr(macro_fetch, "_store_macro", no_net)
    stale = "2000-01-01T00:00:00"   # far older than any TTL
    db = FakeDB([("FROM macro_data", [
        ("egx30", 30000.0, 1.0, 0.5, stale),
        ("usd_egp", 50.0, None, 0.1, stale),
    ])])
    out = macro_fetch.read_cached_macro(db)
    assert out["egx30"]["value"] == 30000.0
    assert out["egx30"]["trend"] in ("bullish", "bearish", "sideways")
    assert out["usd_egp"]["value"] == 50.0
    assert "interest_rate" in out
    assert macro_fetch.read_cached_macro(FakeDB()) is None


def test_run_alerts_happy_path(monkeypatch):
    seen = {}
    monkeypatch.setattr(cron, "read_cached_macro", lambda db: None)
    monkeypatch.setattr(cron, "run_detection", lambda *a, **k: 2)
    monkeypatch.setattr(cron, "prune_baselines", lambda *a, **k: None)

    def deliver(db, now, deadline=None):
        seen["now"] = now
        seen["deadline"] = deadline
        return {"sent": 1, "held": False}

    out = cron.run_alerts(object(), {"COMI": {}}, "2026-10-11T09:00:00+00:00",
                          5.0, deliver=deliver)
    assert out == {"queued": 2, "sent": 1, "skipped": None, "error": None}
    assert isinstance(seen["now"], datetime) and seen["now"].tzinfo is not None
    assert seen["deadline"] is not None


def test_run_alerts_passes_a_send_deadline_of_26s_into_the_call(monkeypatch):
    seen = {}
    monkeypatch.setattr(cron, "read_cached_macro", lambda db: None)
    monkeypatch.setattr(cron, "run_detection", lambda *a, **k: 0)
    monkeypatch.setattr(cron, "prune_baselines", lambda *a, **k: None)
    monkeypatch.setattr(cron.time, "monotonic", lambda: 1000.0)

    def deliver(db, now, deadline=None):
        seen["deadline"] = deadline
        return {"sent": 0, "held": False}

    cron.run_alerts(object(), {}, "2026-10-11T09:00:00+00:00", 5.0, deliver=deliver)
    assert seen["deadline"] == 1000.0 - 5.0 + 26.0
    cron.run_alerts(object(), {}, "2026-10-11T09:00:00+00:00", 5.0, deliver=deliver,
                    deadline=1234.0)
    assert seen["deadline"] == 1234.0
    src = inspect.getsource(cron.risk_snapshot)
    assert "deadline=call_started + ALERTS_SEND_DEADLINE_SECONDS" in src
