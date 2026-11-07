"""scripts/devbox_cron.py: which jobs are due when, on market-local clocks."""
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "devbox_cron", Path(__file__).resolve().parent.parent / "scripts" / "devbox_cron.py")
devbox_cron = importlib.util.module_from_spec(_spec)
sys.modules["devbox_cron"] = devbox_cron  # dataclasses look the module up
_spec.loader.exec_module(devbox_cron)


def due(iso: str) -> set:
    return {j.name for j in devbox_cron.due_jobs(datetime.fromisoformat(iso))}


@pytest.mark.parametrize("iso, expected", [
    # Sunday 2026-11-08, Cairo is UTC+2 (no DST in November).
    ("2026-11-08T07:59:00+00:00", set()),            # 09:59 Cairo
    ("2026-11-08T08:00:00+00:00", {"eg_risk"}),      # 10:00 Cairo, intraday starts
    ("2026-11-08T13:30:00+00:00", {"eg_risk"}),      # 15:30, last intraday minute
    ("2026-11-08T13:31:00+00:00", set()),            # 15:31, not a */5 minute
    ("2026-11-08T13:35:00+00:00", {"eg_risk"}),      # 15:35, post-close pass
    ("2026-11-08T14:55:00+00:00", {"eg_risk"}),      # 16:55, last post-close fire
    ("2026-11-08T15:00:00+00:00", set()),            # 17:00 Cairo
    # Friday: no EGX session; Saturday: neither market.
    ("2026-11-13T08:00:00+00:00", set()),
    ("2026-11-14T15:00:00+00:00", set()),
])
def test_eg_schedule(iso, expected):
    assert due(iso) == expected


@pytest.mark.parametrize("iso, expected", [
    # Monday 2026-11-09, New York is UTC-5 (EST).
    ("2026-11-09T14:29:00+00:00", set()),            # 09:29 NY
    ("2026-11-09T14:31:00+00:00", {"us_risk"}),      # 09:31 NY (16:31 Cairo: no EG fire)
    ("2026-11-09T21:30:00+00:00", {"us_risk"}),      # 16:30 NY, last fire
    ("2026-11-09T21:31:00+00:00", set()),
    # Summer (EDT, UTC-4): the same local window follows DST.
    ("2026-07-06T13:31:00+00:00", {"us_risk"}),      # 09:31 EDT
    ("2026-07-06T14:30:00+00:00", {"us_risk"}),
    ("2026-07-06T20:31:00+00:00", set()),
])
def test_us_schedule_follows_new_york_time(iso, expected):
    assert due(iso) == expected


def test_both_markets_can_be_due_in_the_same_minute():
    # Monday 2026-11-09 14:30 UTC = 16:30 Cairo (*/5 post-close) and 09:30 NY.
    assert due("2026-11-09T14:30:00+00:00") == {"eg_risk", "us_risk"}


def test_daily_jobs_fire_once_at_their_local_time():
    assert due("2026-11-09T02:00:00+00:00") == {"eg_pe"}        # 04:00 Cairo
    assert due("2026-11-09T22:15:00+00:00") == {"us_universe"}  # 17:15 NY
    assert due("2026-11-09T22:45:00+00:00") == {"us_pe"}        # 17:45 NY
    assert due("2026-11-09T02:01:00+00:00") == set()


def test_env_file_must_hold_a_database_url(tmp_path, monkeypatch):
    env = tmp_path / "egx.env"
    env.write_text("DATABASE_URL=\n")
    env.chmod(0o600)
    with pytest.raises(SystemExit):
        devbox_cron._load_env(env)


def test_env_file_readable_by_others_only_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env = tmp_path / "egx.env"
    env.write_text("DATABASE_URL=postgresql://example.invalid/db\n")
    env.chmod(0o644)
    devbox_cron._load_env(env)
    assert "chmod 600" in capsys.readouterr().out


def test_summary_keeps_counts_not_bodies():
    body = {"market": "US", "processed": 24, "written": 24,
            "failed": ["A", "B"], "alerts": {"queued": 1}, "rows": [1, 2, 3]}
    assert devbox_cron._summary(body) == {
        "market": "US", "processed": 24, "written": 24,
        "alerts": {"queued": 1}, "failed": 2}
