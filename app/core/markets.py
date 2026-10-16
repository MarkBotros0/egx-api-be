"""
The market registry: every fact that differs between the Egyptian Exchange and
the US market lives here and nowhere else.

A market is a CURRENCY, a CLOCK and a BENCHMARK, not an exchange: US stocks
trade on NASDAQ, NYSE and AMEX but share one currency, one session and the
S&P 500. So a market lists its exchanges and the TradingView calls pick the
right one per symbol.

`ENABLED_MARKETS` (default "EG") gates which codes the API accepts. A market
can be fully wired and still refused in production until its data works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

import holidays as _holidays
from fastapi import HTTPException, Query


@dataclass(frozen=True)
class IndexDef:
    id: str
    label: str
    symbolset: Optional[str] = None   # TradingView scanner symbolset; None = from data/egx_tickers.json


@dataclass(frozen=True)
class MarketConfig:
    code: str
    name: str
    flag: str
    currency: str
    locale: str
    timezone: str
    trading_days: tuple[int, ...]       # datetime.weekday(): Mon=0 .. Sun=6
    open: time
    close: time                         # inclusive: 14:30 is still open
    holiday_country: str
    holiday_kind: Literal["country", "financial"]
    exchanges: tuple[str, ...]
    scanner_region: str
    benchmark_symbol: str
    benchmark_exchange: str
    benchmark_label: str
    risk_free_fallback_pct: Optional[float]
    liquid_min_turnover: Optional[float]
    liquid_min_price: Optional[float]
    isin_prefix: Optional[str] = None
    # False: /api/markets sends `holidays: []`. The server clock
    # (is_market_open) still uses the calendar either way.
    publish_holidays: bool = True
    indices: tuple[IndexDef, ...] = ()
    short_name: str = ""
    score_calibrated: bool = True

    @property
    def primary_exchange(self) -> str:
        return self.exchanges[0]

    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Drop bonds and structured products whose "symbol" is an ISIN. Matched on
        LENGTH, not prefix: EGSA is a real 4-char EGX ticker. Moved verbatim from
        routers/cron.py::is_isin.
        """
        if self.isin_prefix is None:
            return True
        s = symbol.upper()
        return not (len(s) >= 12 and s.startswith(self.isin_prefix))

    def holidays(self, years) -> _holidays.HolidayBase:
        if self.holiday_kind == "financial":
            return _holidays.financial_holidays(self.holiday_country, years=years)
        return _holidays.country_holidays(self.holiday_country, years=years)


EG = MarketConfig(
    code="EG", name="Egypt", flag="🇪🇬", currency="EGP", locale="en-EG",
    timezone="Africa/Cairo", trading_days=(6, 0, 1, 2, 3),
    open=time(10, 0), close=time(14, 30),
    holiday_country="EG", holiday_kind="country",
    exchanges=("EGX",), scanner_region="egypt",
    benchmark_symbol="EGX30", benchmark_exchange="EGX", benchmark_label="EGX30",
    risk_free_fallback_pct=19,
    liquid_min_turnover=1_000_000, liquid_min_price=0.5,
    isin_prefix="EG",
    # Unverified: the `holidays` library ESTIMATES Islamic dates, and the old
    # client clock ignored holidays. Publish none until Egyptian dates are checked.
    publish_holidays=False,
    indices=(IndexDef("EGX30", "EGX30"), IndexDef("EGX70", "EGX70"),
             IndexDef("EGX100", "EGX100"), IndexDef("NILEX", "NILEX")),
    short_name="EGX",
)

US = MarketConfig(
    code="US", name="United States", flag="🇺🇸", currency="USD", locale="en-US",
    timezone="America/New_York", trading_days=(0, 1, 2, 3, 4),
    open=time(9, 30), close=time(16, 0),
    holiday_country="NYSE", holiday_kind="financial",
    exchanges=("NASDAQ", "NYSE", "AMEX"), scanner_region="america",
    benchmark_symbol="SPX", benchmark_exchange="SP", benchmark_label="S&P 500",
    # 3-month T-bill fallback; live value from TVC:US03MY.
    risk_free_fallback_pct=4.0,
    liquid_min_turnover=None, liquid_min_price=None,
    indices=(IndexDef("SPX", "S&P 500", "SYML:SP;SPX"),
             IndexDef("NDX", "NASDAQ-100", "SYML:NASDAQ;NDX"),
             IndexDef("DJI", "Dow 30", "SYML:DJ;DJI")),
    short_name="US", score_calibrated=False,
)

MARKETS: dict[str, MarketConfig] = {"EG": EG, "US": US}
DEFAULT_CODE = "EG"


def enabled_codes() -> tuple[str, ...]:
    raw = os.environ.get("ENABLED_MARKETS", DEFAULT_CODE)
    codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
    out = tuple(dict.fromkeys(c for c in codes if c in MARKETS))
    return out or (DEFAULT_CODE,)


def get_market(code: Optional[str]) -> MarketConfig:
    market = get_registered_market(code)
    if market.code not in enabled_codes():
        raise HTTPException(status_code=400, detail=f"Market not enabled: {market.code}")
    return market


def market_param(market: Optional[str] = Query(None)) -> MarketConfig:
    """FastAPI dependency: `?market=` -> MarketConfig. Absent means EG."""
    return get_market(market)


def get_registered_market(code: Optional[str]) -> MarketConfig:
    """Like get_market, but ignores ENABLED_MARKETS: any code in MARKETS."""
    if code is not None and not isinstance(code, str):
        raise HTTPException(status_code=400, detail=f"Unknown market: {code!r}")
    c = (code or DEFAULT_CODE).strip().upper()
    if c not in MARKETS:
        raise HTTPException(status_code=400, detail=f"Unknown market: {code}")
    return MARKETS[c]


def registered_market_param(market: Optional[str] = Query(None)) -> MarketConfig:
    """
    FastAPI dependency for the secret-guarded crons ONLY (/api/cron/universe,
    /api/cron/risk_snapshot, /api/pe/refresh): accepts any registered market
    whether or not ENABLED_MARKETS lists it, so a market's data can be built
    BEFORE it is enabled for users. Never use it on a user-facing route.
    """
    return get_registered_market(market)


def is_market_open(market: MarketConfig, now: datetime) -> bool:
    local = now.astimezone(ZoneInfo(market.timezone))
    if local.weekday() not in market.trading_days:
        return False
    if local.date() in market.holidays(local.year):
        return False
    return market.open <= local.time().replace(second=0, microsecond=0) <= market.close


def market_holidays(market: MarketConfig, start: date, days: int = 365) -> list[dict]:
    end = start + timedelta(days=days - 1)
    cal = market.holidays(range(start.year, end.year + 1))
    return [
        {"date": d.isoformat(), "name": name}
        for d, name in sorted(cal.items())
        if start <= d <= end
    ]


def public_market(market: MarketConfig, today: date) -> dict:
    """Display fields for clients. No internals (scanner region, floors)."""
    return {
        "code": market.code,
        "name": market.name,
        "flag": market.flag,
        "currency": market.currency,
        "locale": market.locale,
        "timezone": market.timezone,
        "trading_days": list(market.trading_days),
        "open": market.open.strftime("%H:%M"),
        "close": market.close.strftime("%H:%M"),
        "benchmark_label": market.benchmark_label,
        "short_name": market.short_name,
        "score_calibrated": market.score_calibrated,
        "indices": [{"id": i.id, "label": i.label} for i in market.indices],
        "holidays": market_holidays(market, today) if market.publish_holidays else [],
    }
