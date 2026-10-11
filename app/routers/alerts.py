"""
Score alerts — device registration and per-user preferences.

POST   /api/push/devices            register this phone's Expo push token
DELETE /api/push/devices?token=…    forget it (logout); only the caller's own
GET    /api/alerts/preferences      the caller's settings, or the defaults
PUT    /api/alerts/preferences      save them
GET    /api/alerts                  the caller's sent alerts, last 30 days

Detection and delivery are NOT here: they ride the secret-guarded scoring
cron (routers/cron.py), so this file adds no public route.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import CurrentUser, get_current_user
from app.core.constants import (
    ALERT_DEFAULT_THRESHOLD,
    ALERT_SCOPES,
    ALERT_THRESHOLD_CHOICES,
)
from app.core.db import get_db

router = APIRouter()

_TOKEN = re.compile(r"^Expo(nent)?PushToken\[[A-Za-z0-9_\-]+\]$")
_PLATFORMS = ("ios", "android")
_DEFAULTS = {
    "enabled": False,
    "scope": "watchlist",
    "rise_threshold": ALERT_DEFAULT_THRESHOLD,
    "fall_threshold": ALERT_DEFAULT_THRESHOLD,
}


class DeviceIn(BaseModel):
    token: str
    platform: str


class PreferencesIn(BaseModel):
    enabled: bool
    scope: str
    rise_threshold: Optional[int] = None
    fall_threshold: Optional[int] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/api/push/devices", status_code=201)
def register_device(body: DeviceIn, user: CurrentUser = Depends(get_current_user)):
    token = body.token.strip()
    if not _TOKEN.match(token) or body.platform not in _PLATFORMS:
        raise HTTPException(status_code=400, detail="Invalid push token or platform")
    now = _now()
    get_db().execute(
        "INSERT INTO push_devices (token, user_id, platform, created_at, last_seen) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (token) DO UPDATE SET user_id = EXCLUDED.user_id, "
        "platform = EXCLUDED.platform, last_seen = EXCLUDED.last_seen",
        (token, user.id, body.platform, now, now),
    )
    return {"token": token}


@router.delete("/api/push/devices")
def unregister_device(token: str = Query(...), user: CurrentUser = Depends(get_current_user)):
    rows = get_db().execute(
        "DELETE FROM push_devices WHERE token = %s AND user_id = %s RETURNING token",
        (token.strip(), user.id),
    ).fetchall()
    return {"deleted": bool(rows)}


@router.get("/api/alerts/preferences")
def get_preferences(user: CurrentUser = Depends(get_current_user)):
    row = get_db().execute(
        "SELECT enabled, scope, rise_threshold, fall_threshold "
        "FROM alert_preferences WHERE user_id = %s",
        (user.id,),
    ).fetchone()
    if not row:
        return dict(_DEFAULTS)
    return {"enabled": bool(row[0]), "scope": row[1],
            "rise_threshold": row[2], "fall_threshold": row[3]}


@router.put("/api/alerts/preferences")
def put_preferences(body: PreferencesIn, user: CurrentUser = Depends(get_current_user)):
    allowed = (None,) + ALERT_THRESHOLD_CHOICES
    if (body.scope not in ALERT_SCOPES or body.rise_threshold not in allowed
            or body.fall_threshold not in allowed):
        raise HTTPException(status_code=400, detail="Invalid alert preferences")
    get_db().execute(
        "INSERT INTO alert_preferences (user_id, enabled, scope, rise_threshold, "
        "fall_threshold, updated_at) VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET enabled = EXCLUDED.enabled, "
        "scope = EXCLUDED.scope, rise_threshold = EXCLUDED.rise_threshold, "
        "fall_threshold = EXCLUDED.fall_threshold, updated_at = EXCLUDED.updated_at",
        (user.id, body.enabled, body.scope, body.rise_threshold, body.fall_threshold, _now()),
    )
    return {"enabled": body.enabled, "scope": body.scope,
            "rise_threshold": body.rise_threshold, "fall_threshold": body.fall_threshold}


@router.get("/api/alerts")
def list_alerts(user: CurrentUser = Depends(get_current_user)):
    since = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    rows = get_db().execute(
        "SELECT symbol, session_date, direction, score_before, score_after, sent_at "
        "FROM alert_log WHERE user_id = %s AND status = 'sent' AND session_date >= %s "
        "ORDER BY sent_at DESC LIMIT 200",
        (user.id, since),
    ).fetchall()
    keys = ("symbol", "session_date", "direction", "score_before", "score_after", "sent_at")
    return {"alerts": [dict(zip(keys, r)) for r in rows]}
