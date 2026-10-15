"""The market registry: every per-market fact, and the gate on which are enabled."""
from datetime import date, datetime, timezone

import pytest
from fastapi import HTTPException

from app.core import markets as m


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_eg_entry_reproduces_todays_hardcoded_values():
    eg = m.MARKETS["EG"]
    assert (eg.currency, eg.timezone, eg.primary_exchange) == ("EGP", "Africa/Cairo", "EGX")
    assert (eg.benchmark_symbol, eg.benchmark_exchange, eg.benchmark_label) == ("EGX30", "EGX", "EGX30")
    assert eg.scanner_region == "egypt"
    assert eg.risk_free_fallback_pct == 19
    assert (eg.liquid_min_turnover, eg.liquid_min_price) == (1_000_000, 0.5)


def test_us_entry_shape():
    us = m.MARKETS["US"]
    assert (us.currency, us.timezone, us.exchanges) == ("USD", "America/New_York", ("NASDAQ", "NYSE", "AMEX"))
    assert (us.benchmark_symbol, us.benchmark_exchange, us.benchmark_label) == ("SPX", "SP", "S&P 500")
    assert us.scanner_region == "america"


def test_enabled_defaults_to_eg_only(monkeypatch):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    assert m.enabled_codes() == ("EG",)


def test_enabled_parses_env_and_drops_unknown(monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "eg, US ,XX")
    assert m.enabled_codes() == ("EG", "US")


def test_get_market_none_means_eg(monkeypatch):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    assert m.get_market(None).code == "EG"


def test_get_market_normalises_case_and_space(monkeypatch):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    assert m.get_market(" eg ").code == "EG"


@pytest.mark.parametrize("code", ["XX", "US"])
def test_get_market_rejects_unknown_or_disabled(monkeypatch, code):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    with pytest.raises(HTTPException) as e:
        m.get_market(code)
    assert e.value.status_code == 400


def test_get_market_accepts_us_once_enabled(monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    assert m.get_market("US").code == "US"


@pytest.mark.parametrize("now, open_", [
    (utc(2026, 10, 11, 8, 0), True),    # Sun 11:00 Cairo (EEST, UTC+3)
    (utc(2026, 10, 11, 6, 59), False),  # Sun 09:59 Cairo
    (utc(2026, 10, 11, 11, 30), True),  # Sun 14:30 Cairo, inclusive close
    (utc(2026, 10, 11, 11, 31), False),
    (utc(2026, 10, 9, 9, 0), False),    # Friday
    (utc(2026, 10, 8, 9, 0), False),    # Armed Forces Day (observed), a Thursday
])
def test_eg_open(now, open_):
    assert m.is_market_open(m.EG, now) is open_


@pytest.mark.parametrize("now, open_", [
    (utc(2026, 10, 14, 13, 30), True),   # Wed 09:30 New York (EDT, UTC-4)
    (utc(2026, 10, 14, 13, 29), False),
    (utc(2026, 11, 4, 14, 30), True),    # Wed 09:30 New York after DST ends (EST, UTC-5)
    (utc(2026, 11, 4, 13, 30), False),   # 08:30 EST
    (utc(2026, 11, 26, 15, 0), False),   # Thanksgiving
    (utc(2026, 10, 17, 15, 0), False),   # Saturday
])
def test_us_open_tracks_new_york_dst_and_nyse_holidays(now, open_):
    assert m.is_market_open(m.US, now) is open_


def test_eg_symbol_filter_drops_isins_but_keeps_egsa():
    assert m.EG.is_valid_symbol("COMI")
    assert m.EG.is_valid_symbol("EGSA")
    assert not m.EG.is_valid_symbol("EGS30AJ1C016")
    assert not m.EG.is_valid_symbol("EGS48271C018-EGP")


def test_us_symbol_filter_accepts_class_shares():
    assert m.US.is_valid_symbol("BRK.B")


def test_holidays_window_is_sorted_and_bounded():
    hs = m.market_holidays(m.US, date(2026, 10, 15), days=60)
    dates = [h["date"] for h in hs]
    assert dates == sorted(dates)
    assert "2026-11-26" in dates and all("2026-10-15" <= d <= "2026-12-14" for d in dates)


def test_public_market_is_json_ready():
    out = m.public_market(m.EG, date(2026, 10, 15))
    assert out["code"] == "EG" and out["currency"] == "EGP"
    assert out["trading_days"] == [6, 0, 1, 2, 3]   # Sun..Thu as Python weekday()
    assert out["open"] == "10:00" and out["close"] == "14:30"
    assert isinstance(out["holidays"], list)


@pytest.mark.parametrize("code", [5, 1.5, ["EG"], {"c": "EG"}, True])
def test_get_market_rejects_a_non_string_code_with_400(monkeypatch, code):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    with pytest.raises(HTTPException) as e:
        m.get_market(code)
    assert e.value.status_code == 400


def test_public_market_publishes_no_eg_holidays_until_verified():
    # The holidays library estimates Islamic dates and the old client clock
    # ignored holidays, so EG sends none; US keeps its NYSE calendar.
    assert m.EG.publish_holidays is False
    assert m.public_market(m.EG, date(2026, 10, 15))["holidays"] == []
    us = m.public_market(m.US, date(2026, 10, 15))["holidays"]
    assert "2026-11-26" in [h["date"] for h in us]   # Thanksgiving


def test_eg_server_clock_still_uses_the_holiday_calendar():
    # Only the published list is withheld; is_market_open keeps its calendar.
    assert len(m.EG.holidays(2026)) > 0
