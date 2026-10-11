"""Score alerts ride the scoring cron and can never break it."""

from __future__ import annotations

import inspect

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

    monkeypatch.setattr(cron, "fetch_macro", lambda db: None)
    monkeypatch.setattr(cron, "run_detection", boom)
    out = cron.run_alerts(object(), {"COMI": {}}, "2026-10-11T09:00:00+00:00", 5.0)
    assert out["error"] == "RuntimeError"


def test_quiet_hours_are_reported(monkeypatch):
    monkeypatch.setattr(cron, "run_detection", lambda *a, **k: 0)
    monkeypatch.setattr(cron, "prune_baselines", lambda *a, **k: None)
    out = cron.run_alerts(object(), {}, "2026-10-11T21:00:00+00:00", 5.0,
                          deliver=lambda db, now: {"sent": 0, "held": True})
    assert out["skipped"] == "quiet-hours"


def test_baseline_is_captured_before_the_row_is_overwritten():
    """Order matters: capture reads the row the upsert is about to replace."""
    src = inspect.getsource(cron.risk_snapshot)
    assert src.index("read_previous_rows(") < src.index("for symbol in slice_")
    capture = src.index("capture_baseline(")
    assert capture > src.index("card = _score_card(symbol, df)")
    assert capture < src.index("_upsert(symbol, stats, is_tradeable")
    assert "run_alerts(" in src and '"alerts": alerts' in src
