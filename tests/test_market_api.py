import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")

import app.routers.markets as markets_router  # noqa: E402
import app.routers.settings as settings_mod  # noqa: E402
from app.core.auth import CurrentUser  # noqa: E402
from alert_fakes import FakeDB  # noqa: E402

ME = CurrentUser(id="u1", email="sara@example.com")


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(settings_mod, "get_db", lambda: fake)
    return fake


def test_list_markets_only_enabled(monkeypatch):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    out = markets_router.list_markets(ME)
    assert [x["code"] for x in out["markets"]] == ["EG"]
    assert out["default"] == "EG"


def test_list_markets_both_when_enabled(monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    out = markets_router.list_markets(ME)
    assert [x["code"] for x in out["markets"]] == ["EG", "US"]


def test_market_pref_defaults_to_eg(db):
    assert settings_mod.get_settings(section="market", user=ME) == {"market": "EG"}


def test_market_pref_put_writes_user_settings(db, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    out = settings_mod.put_settings({"market": "us"}, section="market", user=ME)
    assert out == {"market": "US"}
    [(sql, params)] = db.statements("INSERT INTO user_settings")
    assert params == ("u1", "market", "US")


def test_market_pref_rejects_disabled(db, monkeypatch):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    with pytest.raises(HTTPException) as e:
        settings_mod.put_settings({"market": "US"}, section="market", user=ME)
    assert e.value.status_code == 400


def test_stored_pref_for_a_since_disabled_market_reads_as_eg(db, monkeypatch):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    db.rows_for("SELECT value FROM user_settings", [("US",)])
    assert settings_mod.get_settings(section="market", user=ME) == {"market": "EG"}
