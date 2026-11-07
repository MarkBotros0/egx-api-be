"""
Run the scheduled jobs on the devbox instead of through cron-job.org -> Vercel.

Vercel Hobby paused the team for Fluid Active CPU: the every-minute
risk_snapshot calls (EG and US, ~20s each) were ~79% of it. This runner calls
the SAME routes in-process through FastAPI's TestClient, so the code path is
identical and Vercel is not involved at all.

Invoked by the user's crontab every minute:

    * * * * * /home/mark/projects/egx-api/egx-api-be/.venv/bin/python \
        /home/mark/projects/egx-api/egx-api-be/scripts/devbox_cron.py \
        >> /home/mark/logs/egx-crons.log 2>&1

It decides which jobs are due THIS minute from market-local clocks
(Africa/Cairo, America/New_York), so DST is followed without a cron TZ.

Environment: read from EGX_CRON_ENV_FILE (default: this repo's .env, whose
Neon DATABASE_URL is the production database). The route secrets are set to a
fresh random value per run and sent as headers, so the runner never depends
on the deployed CRON_SECRET / PE_REFRESH_SECRET.

Usage:
    devbox_cron.py                 # run every job due now
    devbox_cron.py --job eg_risk   # run one job now, ignoring the schedule
    devbox_cron.py --due "2026-11-08T08:00Z"   # print what would be due
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

CAIRO = ZoneInfo("Africa/Cairo")
NEW_YORK = ZoneInfo("America/New_York")

DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
LOCK_DIR = Path.home() / ".cache" / "egx-crons"


def _minutes(t: datetime) -> int:
    return t.hour * 60 + t.minute


def _eg_risk_due(now: datetime) -> bool:
    # Superset of the intraday alert pass (every minute, 10:00-15:30) and the
    # post-close pass (*/5 15-16). Sun-Thu = the EGX week.
    t = now.astimezone(CAIRO)
    if t.weekday() not in (6, 0, 1, 2, 3):
        return False
    m = _minutes(t)
    return 600 <= m <= 930 or (900 <= m < 1020 and t.minute % 5 == 0)


def _us_risk_due(now: datetime) -> bool:
    # Every minute 09:30-16:30 New York, Mon-Fri (intraday + post-close pass).
    t = now.astimezone(NEW_YORK)
    return t.weekday() < 5 and 570 <= _minutes(t) <= 990


def _at(tz: ZoneInfo, hour: int, minute: int) -> Callable[[datetime], bool]:
    def due(now: datetime) -> bool:
        t = now.astimezone(tz)
        return t.hour == hour and t.minute == minute
    return due


@dataclass(frozen=True)
class Job:
    name: str
    path: str
    header: str
    due: Callable[[datetime], bool]


JOBS = (
    Job("eg_risk", "/api/cron/risk_snapshot?limit=6", "X-Cron-Secret", _eg_risk_due),
    Job("us_risk", "/api/cron/risk_snapshot?market=US&limit=30", "X-Cron-Secret", _us_risk_due),
    Job("eg_pe", "/api/pe/refresh", "X-Refresh-Secret", _at(CAIRO, 4, 0)),
    # After the US close; the universe (and T-bill) first, fundamentals after.
    Job("us_universe", "/api/cron/universe?market=US", "X-Cron-Secret", _at(NEW_YORK, 17, 15)),
    Job("us_pe", "/api/pe/refresh?market=US", "X-Refresh-Secret", _at(NEW_YORK, 17, 45)),
)
JOBS_BY_NAME = {j.name: j for j in JOBS}


def due_jobs(now: datetime) -> list[Job]:
    return [j for j in JOBS if j.due(now)]


def _summary(body: object) -> object:
    """Counts only - the log should not grow by whole response bodies."""
    if not isinstance(body, dict):
        return str(body)[:200]
    keep = ("market", "processed", "written", "stale_remaining", "universe",
            "alerts", "note", "detail", "count", "status")
    out = {k: body[k] for k in keep if k in body}
    if isinstance(body.get("failed"), list):
        out["failed"] = len(body["failed"])
    return out


def _load_env(env_file: Path) -> None:
    from dotenv import dotenv_values

    if not env_file.is_file():
        sys.exit(f"[devbox-cron] env file missing: {env_file}")
    if env_file.stat().st_mode & 0o077:
        print(f"[devbox-cron] warning: {env_file} is group/world readable; chmod 600 it", flush=True)
    values = {k: v for k, v in dotenv_values(env_file).items() if v}
    if not values.get("DATABASE_URL"):
        # Fail closed: without it nothing would be written, and the app must
        # never fall back to some other database.
        sys.exit(f"[devbox-cron] DATABASE_URL is empty in {env_file}")
    os.environ.update(values)


def _run(client, job: Job, secret: str) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_DIR / f"{job.name}.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"at": _stamp(), "job": job.name, "skipped": "still running"}), flush=True)
            return
        started = time.monotonic()
        try:
            r = client.post(job.path, headers={job.header: secret})
            try:
                body = r.json()
            except ValueError:
                body = r.text
            line = {"status": r.status_code, "result": _summary(body)}
        except Exception as e:  # noqa: BLE001 - one job must not stop the next
            line = {"status": "crashed", "error": type(e).__name__}
        line = {"at": _stamp(), "job": job.name,
                "secs": round(time.monotonic() - started, 1), **line}
        print(json.dumps(line, default=str), flush=True)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job", choices=sorted(JOBS_BY_NAME))
    p.add_argument("--due", metavar="ISO_TIME")
    args = p.parse_args(argv)

    if args.due:
        now = datetime.fromisoformat(args.due.replace("Z", "+00:00"))
        print(" ".join(j.name for j in due_jobs(now)) or "(none)")
        return 0

    jobs = [JOBS_BY_NAME[args.job]] if args.job else due_jobs(datetime.now(timezone.utc))
    if not jobs:
        return 0

    _load_env(Path(os.environ.get("EGX_CRON_ENV_FILE", DEFAULT_ENV_FILE)))
    secret = secrets.token_urlsafe(32)
    os.environ["CRON_SECRET"] = secret
    os.environ["PE_REFRESH_SECRET"] = secret

    # tvDatafeed warns "nologin method" on every fetch; at ~750 runs a day that
    # would bury the one JSON line per job the log is for.
    import logging
    logging.getLogger("tvDatafeed").setLevel(logging.ERROR)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        for job in jobs:
            _run(client, job, secret)
    return 0


if __name__ == "__main__":
    sys.exit(main())
