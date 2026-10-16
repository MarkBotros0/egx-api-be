"""Alerts API: devices and preferences, scoped to the caller."""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

# app.core.auth reads AUTH_SECRET at import; this file sorts before test_auth_gate.py,
# so set the same default it does before anything imports auth.
os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")

import app.routers.alerts as alerts_mod  # noqa: E402
from app.core.auth import CurrentUser  # noqa: E402
from alert_fakes import FakeDB  # noqa: E402

ME = CurrentUser(id="u1", email="sara@example.com")


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(alerts_mod, "get_db", lambda: fake)
    return fake


def test_register_device_upserts_and_rebinds(db):
    out = alerts_mod.register_device(
        alerts_mod.DeviceIn(token="ExponentPushToken[abc-123]", platform="android"), ME)
    assert out == {"token": "ExponentPushToken[abc-123]"}
    [(sql, params)] = db.statements("INSERT INTO push_devices")
    assert "ON CONFLICT (token) DO UPDATE SET user_id = EXCLUDED.user_id" in sql
    assert params[:3] == ("ExponentPushToken[abc-123]", "u1", "android")


@pytest.mark.parametrize("token, platform", [
    ("not-a-token", "android"),
    ("ExponentPushToken[]", "ios"),
    ("ExponentPushToken[abc]", "web"),
])
def test_register_device_rejects_bad_input(db, token, platform):
    with pytest.raises(HTTPException) as e:
        alerts_mod.register_device(alerts_mod.DeviceIn(token=token, platform=platform), ME)
    assert e.value.status_code == 400


def test_unregister_only_touches_the_callers_token(db):
    alerts_mod.unregister_device("ExponentPushToken[abc]", ME)
    [(sql, params)] = db.statements("DELETE FROM push_devices")
    assert "user_id = %s" in sql and params == ("ExponentPushToken[abc]", "u1")


def test_preferences_default_when_no_row(db):
    assert alerts_mod.get_preferences(ME) == {
        "enabled": False, "scope": "watchlist", "rise_threshold": 10, "fall_threshold": 10,
    }


def test_preferences_round_trip(db):
    body = alerts_mod.PreferencesIn(enabled=True, scope="all", rise_threshold=None, fall_threshold=5)
    out = alerts_mod.put_preferences(body, ME)
    assert out == {"enabled": True, "scope": "all", "rise_threshold": None, "fall_threshold": 5}
    [(sql, params)] = db.statements("INSERT INTO alert_preferences")
    assert "ON CONFLICT (user_id) DO UPDATE" in sql
    assert params[:5] == ("u1", True, "all", None, 5)


@pytest.mark.parametrize("field, value", [("scope", "everything"), ("rise_threshold", 7), ("fall_threshold", 0)])
def test_preferences_reject_values_outside_the_choices(db, field, value):
    data = {"enabled": True, "scope": "watchlist", "rise_threshold": 10, "fall_threshold": 10}
    data[field] = value
    with pytest.raises(HTTPException) as e:
        alerts_mod.put_preferences(alerts_mod.PreferencesIn(**data), ME)
    assert e.value.status_code == 400


def test_alert_history_is_the_callers_sent_alerts(db):
    db.responses.append(("FROM alert_log", [("COMI", "2026-10-11", "up", 58.0, 70.0, "t", "EG"),
                                            ("AAPL", "2026-10-11", "up", 50.0, 62.0, "t", "US")]))
    out = alerts_mod.list_alerts(ME)
    assert out["alerts"][0]["symbol"] == "COMI"
    assert [a["market"] for a in out["alerts"]] == ["EG", "US"]
    sql, params = db.statements("FROM alert_log")[0]
    assert "status = 'sent'" in sql and params[0] == "u1"
