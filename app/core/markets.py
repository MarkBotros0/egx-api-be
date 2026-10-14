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
)

US = MarketConfig(
    code="US", name="United States", flag="🇺🇸", currency="USD", locale="en-US",
    timezone="America/New_York", trading_days=(0, 1, 2, 3, 4),
    open=time(9, 30), close=time(16, 0),
    holiday_country="NYSE", holiday_kind="financial",
    exchanges=("NASDAQ", "NYSE", "AMEX"), scanner_region="america",
    benchmark_symbol="SPX", benchmark_exchange="SP", benchmark_label="S&P 500",
    # Calibrated in sub-project 3; None means "no US rate yet".
    risk_free_fallback_pct=None,
    liquid_min_turnover=None, liquid_min_price=None,
)

MARKETS: dict[str, MarketConfig] = {"EG": EG, "US": US}
DEFAULT_CODE = "EG"


def enabled_codes() -> tuple[str, ...]:
    raw = os.environ.get("ENABLED_MARKETS", DEFAULT_CODE)
    codes = [c.strip().upper() for c in raw.split(",") if c.strip()]
    out = tuple(dict.fromkeys(c for c in codes if c in MARKETS))
    return out or (DEFAULT_CODE,)


def get_market(code: Optional[str]) -> MarketConfig:
    c = (code or DEFAULT_CODE).strip().upper()
    if c not in MARKETS:
        raise HTTPException(status_code=400, detail=f"Unknown market: {code}")
    if c not in enabled_codes():
        raise HTTPException(status_code=400, detail=f"Market not enabled: {c}")
    return MARKETS[c]


def market_param(market: Optional[str] = Query(None)) -> MarketConfig:
    """FastAPI dependency: `?market=` -> MarketConfig. Absent means EG."""
    return get_market(market)


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
        "holidays": market_holidays(market, today),
    }
