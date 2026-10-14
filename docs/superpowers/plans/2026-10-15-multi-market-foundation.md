# Multi-market Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `market` dimension (EG today, US later) through the database, backend, shared client library and mobile app. EGX behaviour stays exactly as it is now, and the mobile country changer stays hidden until a second market is enabled.

**Architecture:**
- A backend registry, `app/core/markets.py`, holds every per-market fact. A FastAPI dependency resolves `?market=` (default `EG`) to a `MarketConfig`, and routers pass it down in place of hardcoded `"EGX"`, `"EGX30"` and Cairo values.
- Symbol-keyed tables gain a `market` column and composite keys, added by idempotent DDL in `init_db`.
- The shared `lib/` (web is the source, mobile copies it) adds `markets.ts` and `money.ts`, plus a module-level market in `api.ts`.
- On mobile, a `MarketProvider` persists the choice and drives refetches.

**Tech Stack:** FastAPI + psycopg (Neon Postgres), pytest; Next.js 14 (shared `lib/` only); Expo 57 / React Native, jest-expo, AsyncStorage.

**Spec:** `egx-api-be/docs/superpowers/specs/2026-10-15-multi-market-foundation-design.md`

## Global Constraints

**Markets and the API**
- Market codes are exactly `"EG"` and `"US"`. A request with no `market` means `"EG"`, everywhere: query params, POST bodies, DB defaults and client fallbacks.
- `ENABLED_MARKETS` env var: comma-separated, default `"EG"`. A code that is unknown or not enabled gives HTTP 400.
- `/api/markets` and `/api/settings?section=market` require sign-in. Nothing is added to `PUBLIC_ENDPOINTS`.
- Per-market facts (currency, risk-free rate) are never per-user. Only the *selected* market is a per-user preference, stored in `user_settings` under key `market`.

**Database**
- DDL lives only in `init_db` and must be idempotent: running it twice on the same DB is a no-op.
- A literal `%` in SQL must be written `%%`. `_DB.execute` always passes params, and `tests/test_users_and_roles.py::test_no_sql_has_an_unescaped_percent` enforces this.

**Shared `lib/`**
- `egx-api-fe/src/app/lib/*` must stay platform-neutral: no `window`, `document`, `localStorage`, `next/*` or component imports.
- After changing any file in it, run `npm run sync:shared` in `egx-api-mobile` and commit the copies there. Never edit `egx-api-mobile/src/shared/` by hand.

**UI**
- The changer is visible only when more than one market is enabled.
- `gain`/`loss` colours are never decoration. The market chip uses neutral colours.
- Touch targets are at least 44px on mobile.

**Process**
- Each repo works on branch `feat/multi-market-foundation`.
- Commits end with the attribution lines:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XDkAB7UCK8WyT267W5ZxNV
  ```
- Push only when asked, running `gh auth switch --hostname github.com --user MarkBotros0 && git push`.
- Long runs (full test suites, builds) go in tmux with a log in `~/logs/egx-api-<task>.log`.

## Review Focus

These failure modes are implied by the spec but not exercised by any task's own happy-path tests. The owning task adds a test for each.

1. **An existing row after the migration.** A pre-migration watchlist, holding or `pe_data` row must read back unchanged under `market=EG`, with no duplicates and none lost. Owner: Task 3 (PG test `test_existing_rows_survive_rekey`).
2. **The same symbol in two markets.** Inserting `("u1","EG","ABC")` and `("u1","US","ABC")` into `watchlist` must keep both, and a watchlist GET for EG must not return the US row. Owner: Task 3 (PG) and Task 8 (`test_watchlist_is_scoped_by_market`).
3. **Lower-case or padded market codes.** `?market=eg` and `?market= EG ` should behave like `EG`, not return 400. Owner: Task 1 (`test_get_market_normalises_case_and_space`).
4. **Switching markets mid-request on mobile.** A slow EG response that lands after the user has switched to US must not overwrite US state. Owner: Task 14 (`discards a response from the previous market`).
5. **The `/api/markets` call failing or an old backend.** The mobile app must still boot in EG with the changer hidden. Owner: Task 13 (`falls back to EG when /markets fails`).

---

## File Structure

**egx-api-be**

| File | Status | Responsibility |
|---|---|---|
| `app/core/markets.py` | new | `MarketConfig`, the registry, `ENABLED_MARKETS`, `get_market`, `market_param`, `is_market_open`, `market_holidays`, `public_market` |
| `app/routers/markets.py` | new | `GET /api/markets` |
| `app/core/benchmark.py` | new | `get_benchmark_close(market, interval, bars)`, the one benchmark fetch, replacing three copies |
| `app/core/db.py` | modify | `market` columns, the `_ensure_key` helper, column renames |
| `app/core/cache.py` | modify | `make_key(market_code, *args)` validates its first argument |
| `app/core/tradingview.py` | modify | `scan(columns, market=EG, ...)` uses `market.scanner_region` and `market.exchanges` |
| `app/vendor/egxpy/download.py` | modify | `exchange` and `holiday_country` params on `get_EGXdata` and `get_EGX_intraday_data` |
| `app/core/constants.py`, `app/core/score_alerts.py`, `app/core/risk_grade.py` | modify | read Cairo, risk-free fallback and liquidity floors from the EG market |
| `app/routers/*.py` (listed per task) | modify | take `market: MarketConfig = Depends(market_param)` |
| `app/main.py` | modify | include the markets router |
| `tests/test_markets.py`, `tests/test_market_api.py`, `tests/test_market_schema.py`, `tests/test_market_schema_pg.py`, `tests/test_cache_keys.py`, `tests/test_market_routers.py` | new | tests |

**egx-api-fe (shared `lib/` plus the web formatter swap)**

| File | Status | Responsibility |
|---|---|---|
| `src/app/lib/markets.ts` | new | `MarketCode`, `Market`, `EG_FALLBACK`, `isMarketOpen` |
| `src/app/lib/money.ts` | new | `formatMoney`, `formatPrice` |
| `src/app/lib/api.ts` | modify | `setApiMarket`, `getApiMarket`, `withMarket`, `fetchMarkets`, `fetchMarketPref`, `putMarketPref`; `fetchJSON` adds `market=` |
| `src/app/lib/types.ts` | modify | `market` and `currency` on responses |
| `src/app/page.tsx` + 15 components | modify | `isMarketOpen(EG_FALLBACK)`, `formatMoney` |

> The shared formatter is named `money.ts`, not `format.ts` as the spec says, because mobile already has `src/features/*/logic/format.ts` and the same name would be confusing. Task 10 updates the spec to match.

**egx-api-mobile**

| File | Status | Responsibility |
|---|---|---|
| `scripts/sync-shared.mjs` | modify | add `markets.ts` and `money.ts` |
| `src/providers/market.tsx` | new | `MarketProvider`, `useMarket` |
| `src/providers/AppProviders.tsx` | modify | MarketProvider outermost |
| `src/features/shell/MarketChip.tsx`, `src/features/shell/MarketSheet.tsx` | new | the changer |
| `src/app/(tabs)/_layout.tsx`, `src/features/account/AccountScreen.tsx` | modify | mount the chip and the account row |
| providers and screens listed in Task 14 | modify | `market.code` in fetch-effect deps |
| `src/features/market/logic/marketClock.ts` + EGP-literal components (Task 16) | modify | registry-driven clock and money |

---

## Phase A: Backend (`egx-api-be`)

### Task 0: Branches, venv, baseline

**Files:** none (environment only)

- [ ] **Step 1: Create the branches**

```bash
for r in egx-api-be egx-api-fe egx-api-mobile; do
  (cd /home/mark/projects/egx-api/$r && git status --short && git switch -c feat/multi-market-foundation)
done
```
Expected: `git status --short` prints nothing for each repo before the switch. If a repo is dirty, stop and ask the user.

- [ ] **Step 2: Create the backend venv** (`.venv/` is already in `.gitignore`)

```bash
cd /home/mark/projects/egx-api/egx-api-be
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt pytest httpx
.venv/bin/python -c "import holidays; print(holidays.financial_holidays('NYSE', years=2026).get('2026-11-26'))"
```
Expected: `Thanksgiving Day`.

- [ ] **Step 3: Record the baseline test results**

```bash
mkdir -p ~/logs
tmux kill-session -t egx-api-baseline 2>/dev/null
tmux new -d -s egx-api-baseline "cd /home/mark/projects/egx-api/egx-api-be && .venv/bin/python -m pytest -q 2>&1 | tee ~/logs/egx-api-baseline.log; cd ../egx-api-mobile && npx jest 2>&1 | tee -a ~/logs/egx-api-baseline.log"
```
When it finishes, note the pass/fail counts from `tail -5 ~/logs/egx-api-baseline.log`. Every later task must leave the same tests passing. Tests that already fail at baseline are reported to the user, not fixed here.

---

### Task 1: Market registry

**Files:**
- Create: `app/core/markets.py`
- Test: `tests/test_markets.py`

**Interfaces:**
- Produces:
  - `MarketConfig` (frozen dataclass) with fields `code, name, flag, currency, locale, timezone, trading_days: tuple[int,...]` (Python `weekday()`, Mon=0), `open: time, close: time, holiday_country: str, holiday_kind: Literal["country","financial"], exchanges: tuple[str,...], scanner_region, benchmark_symbol, benchmark_exchange, benchmark_label, risk_free_fallback_pct: float|None, liquid_min_turnover: float|None, liquid_min_price: float|None`
  - Methods: `.primary_exchange` (property), `.is_valid_symbol(sym) -> bool`, `.holidays(years) -> holidays.HolidayBase`
  - `MARKETS: dict[str, MarketConfig]`
  - `EG: MarketConfig`, `US: MarketConfig`
  - `enabled_codes() -> tuple[str, ...]`
  - `get_market(code: str | None) -> MarketConfig` (raises `HTTPException(400)`)
  - `is_market_open(market, now: datetime) -> bool`
  - `market_holidays(market, start: date, days: int = 365) -> list[dict]` returning `[{"date": "YYYY-MM-DD", "name": str}]`
  - `public_market(market, today: date) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_markets.py
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
    (utc(2026, 10, 6, 9, 0), False),    # Armed Forces Day holiday (6 Oct), a Tuesday
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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_markets.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'app.core.markets'`.

- [ ] **Step 3: Implement**

```python
# app/core/markets.py
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_markets.py -q`
Expected: all pass.
- If the EG holiday case fails because the `holidays` package dates Armed Forces Day differently, look up the real 2026 date with `.venv/bin/python -c "import holidays; print(sorted(holidays.EG(years=2026).items()))"` and use another weekday holiday from that list in the test. Don't change the code.

- [ ] **Step 5: Commit**

```bash
git add app/core/markets.py tests/test_markets.py
git commit -m "feat(markets): market registry with EG and US entries"
```

---

### Task 2: `GET /api/markets` and the market preference

**Files:**
- Create: `app/routers/markets.py`, `tests/test_market_api.py`
- Modify: `app/main.py` (include the router next to the other `include_router` calls), `app/routers/settings.py`

**Interfaces:**
- Consumes: `markets.enabled_codes`, `get_market`, `public_market`, `MARKETS` (Task 1).
- Produces:
  - `GET /api/markets` → `{"markets": [public_market...], "default": "EG"}`.
  - `GET /api/settings?section=market` → `{"market": "EG"}`.
  - `PUT /api/settings?section=market`, body `{"market": "US"}` → `{"market": "US"}`, or 400.
  - `routers.settings.read_market_pref(db, user_id) -> str`, used by nothing else yet.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_market_api.py
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
```

Before running, check what `tests/alert_fakes.py::FakeDB` supports. The tests above use two helpers: `statements(prefix)` (already used by `test_alerts_router.py`) and `rows_for(sql_prefix, rows)`, which makes `execute(...).fetchone()/fetchall()` return canned rows for SQL starting with that prefix. If `rows_for` (or its equivalent) is missing, add it to `FakeDB` in this task. Keep its default behaviour of returning no rows.

- [ ] **Step 2: Run and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_market_api.py -q`
Expected: `ModuleNotFoundError: app.routers.markets`.

- [ ] **Step 3: Implement**

```python
# app/routers/markets.py
"""GET /api/markets — the enabled markets and their display fields."""
from datetime import date

from fastapi import APIRouter, Depends

from app.core.auth import CurrentUser, get_current_user
from app.core.markets import DEFAULT_CODE, MARKETS, enabled_codes, public_market

router = APIRouter()


@router.get("/api/markets")
def list_markets(user: CurrentUser = Depends(get_current_user)):
    today = date.today()
    return {
        "markets": [public_market(MARKETS[c], today) for c in enabled_codes()],
        "default": DEFAULT_CODE,
    }
```

In `app/main.py`, add `from app.routers import markets as markets_router` and `app.include_router(markets_router.router)`, following the existing include pattern.

In `app/routers/settings.py`:
- Add `from app.core.markets import DEFAULT_CODE, enabled_codes, get_market`.
- Add this helper:

```python
def read_market_pref(db, user_id: str) -> str:
    """The caller's selected market; EG when unset or no longer enabled."""
    row = db.execute(
        "SELECT value FROM user_settings WHERE user_id = %s AND key = 'market'",
        (user_id,),
    ).fetchone()
    code = (row[0] if row else DEFAULT_CODE).upper()
    return code if code in enabled_codes() else DEFAULT_CODE
```

- In `get_settings`, before the `section == "weights"` branch, add:

```python
        if section == "market":
            return {"market": read_market_pref(get_db(), user.id)}
```

- In `put_settings`, before the weights branch, add:

```python
        if section == "market":
            market = get_market(body.get("market"))   # 400 on unknown/disabled
            db = get_db()
            # A per-user PREFERENCE (what to view), unlike the market facts.
            db.execute(
                "INSERT INTO user_settings (user_id, key, value) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value",
                (user.id, "market", market.code),
            )
            db.commit()
            return {"market": market.code}
```

- Add the two section forms to the module docstring.

- [ ] **Step 4: Run the tests plus the auth gate**

Run: `.venv/bin/python -m pytest tests/test_market_api.py tests/test_auth_gate.py -q`
Expected: all pass. `test_auth_gate` must confirm `/api/markets` rejects anonymous callers.

- [ ] **Step 5: Commit**

```bash
git add app/routers/markets.py app/main.py app/routers/settings.py tests/test_market_api.py tests/alert_fakes.py
git commit -m "feat(markets): GET /api/markets and per-user market preference"
```

---

### Task 3: Schema — `market` columns, composite keys, renames

**Files:**
- Modify: `app/core/db.py` (the table blocks at lines ~151-231, 279-310, 324-330, 403-441, 590-617)
- Modify the readers of the renamed columns: `app/core/pe_fetch.py`, `app/core/card_snapshot.py`, `app/core/risk_grade.py`, `app/routers/risk.py`, `egx-api-fe/src/app/lib/api.ts` (that one is changed in Task 10)
- Create: `tests/test_market_schema.py` (pure), `tests/test_market_schema_pg.py` (opt-in)

**Interfaces:**
- Produces:
  - `db._ensure_key(db, table, kind, name, columns)`, where `kind` is `"PRIMARY KEY"` or `"UNIQUE"`. It is idempotent.
  - `db._rename_column(db, table, old, new)`, also idempotent.
  - Every listed table has `market TEXT NOT NULL DEFAULT 'EG'`.
  - `risk_snapshot.turnover` and `pe_data.value_traded` replace the `_egp` names.

- [ ] **Step 1: Write the pure test (the SQL the helpers emit)**

```python
# tests/test_market_schema.py
from app.core import db as db_mod
from alert_fakes import FakeDB


def test_ensure_key_is_guarded_by_catalog_lookup():
    fake = FakeDB()
    db_mod._ensure_key(fake, "watchlist", "PRIMARY KEY", "watchlist_pkey",
                       ("user_id", "market", "symbol"))
    [(sql, _)] = fake.statements("DO $$")
    assert "pg_constraint" in sql
    assert "ADD CONSTRAINT watchlist_pkey PRIMARY KEY (user_id, market, symbol)" in sql
    assert "DROP CONSTRAINT" in sql


def test_rename_column_is_guarded():
    fake = FakeDB()
    db_mod._rename_column(fake, "risk_snapshot", "turnover_egp", "turnover")
    [(sql, _)] = fake.statements("DO $$")
    assert "information_schema.columns" in sql
    assert "RENAME COLUMN turnover_egp TO turnover" in sql


def test_every_symbol_table_gets_market():
    fake = FakeDB()
    db_mod._add_market_columns(fake)
    tables = {s.split()[2] for s, _ in fake.statements("ALTER TABLE")}
    assert tables >= {
        "watchlist", "discovered_tickers", "pe_data", "risk_snapshot",
        "dividend_events", "fundamentals_annual", "score_baseline", "alert_log",
        "portfolio", "portfolio_sales", "portfolio_dividends",
        "fundamentals_history", "market_regime",
    }
```

- [ ] **Step 2: Write the opt-in Postgres test**

```python
# tests/test_market_schema_pg.py
"""
init_db's market migration against a REAL Postgres. Skipped unless
TEST_DATABASE_URL is set (see tests/test_accounts_pg.py for the docker line).
Never point this at a real database: it drops tables.
"""
import os

import psycopg
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")

TABLES = ("watchlist", "pe_data", "risk_snapshot", "dividend_events")


@pytest.fixture
def d(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    from app.core import db as db_mod
    monkeypatch.setattr(db_mod, "_pool", None)
    monkeypatch.setattr(db_mod, "_initialized", True)
    d = db_mod._DB(db_mod._get_pool())
    for t in TABLES:
        d.execute(f"DROP TABLE IF EXISTS {t}")
    # The OLD shapes, exactly as production has them today.
    d.execute("CREATE TABLE watchlist (user_id TEXT NOT NULL, symbol TEXT NOT NULL, "
              "added_at TEXT NOT NULL, PRIMARY KEY (user_id, symbol))")
    d.execute("CREATE TABLE risk_snapshot (symbol TEXT PRIMARY KEY, measured_at TEXT NOT NULL, "
              "turnover_egp DOUBLE PRECISION)")
    d.execute("INSERT INTO watchlist VALUES ('u1', 'COMI', '2026-01-01')")
    d.execute("INSERT INTO risk_snapshot VALUES ('COMI', '2026-01-01', 5e6)")
    return db_mod, d


def test_existing_rows_survive_rekey(d):
    db_mod, d = d
    db_mod.init_db(d)
    db_mod.init_db(d)  # idempotent
    assert d.execute("SELECT user_id, market, symbol FROM watchlist").fetchall() == [("u1", "EG", "COMI")]
    assert d.execute("SELECT market, turnover FROM risk_snapshot").fetchall() == [("EG", 5e6)]


def test_same_symbol_in_two_markets(d):
    db_mod, d = d
    db_mod.init_db(d)
    d.execute("INSERT INTO watchlist (user_id, market, symbol, added_at) VALUES ('u1','US','COMI','x')")
    assert d.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0] == 2
    with pytest.raises(psycopg.errors.UniqueViolation):
        d.execute("INSERT INTO watchlist (user_id, market, symbol, added_at) VALUES ('u1','US','COMI','y')")
```

- [ ] **Step 3: Run the pure tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_market_schema.py -q`
Expected: `AttributeError: module 'app.core.db' has no attribute '_ensure_key'`.

- [ ] **Step 4: Implement the helpers in `app/core/db.py`** (put them above `init_db`)

```python
# Tables keyed by symbol gain a market. Keys list the NEW constraint columns;
# None means the table keeps an id key and only gets the column + an index.
_MARKET_KEYS: dict[str, tuple[str, tuple[str, ...]] | None] = {
    "watchlist":           ("PRIMARY KEY", ("user_id", "market", "symbol")),
    "discovered_tickers":  ("PRIMARY KEY", ("market", "symbol")),
    "pe_data":             ("PRIMARY KEY", ("market", "symbol")),
    "risk_snapshot":       ("PRIMARY KEY", ("market", "symbol")),
    "dividend_events":     ("PRIMARY KEY", ("market", "symbol", "ex_date")),
    "fundamentals_annual": ("PRIMARY KEY", ("market", "symbol", "fiscal_year")),
    "score_baseline":      ("PRIMARY KEY", ("market", "symbol", "session_date")),
    "alert_log":           ("UNIQUE", ("user_id", "market", "symbol", "session_date", "direction")),
    "portfolio":           None,
    "portfolio_sales":     None,
    "portfolio_dividends": None,
    "fundamentals_history": None,
    "market_regime":       None,
}


def _ensure_key(db, table: str, kind: str, name: str, columns: tuple[str, ...]) -> None:
    """
    Make `name` the table's key over `columns`, idempotently.

    Postgres has no ADD CONSTRAINT IF NOT EXISTS for keys, and an existing key
    over the OLD columns must be dropped first. The catalog lookup compares the
    column list, so a second run finds the new key already in place and does
    nothing.
    """
    cols = ", ".join(columns)
    want = "{" + ",".join(columns) + "}"
    contype = "p" if kind == "PRIMARY KEY" else "u"
    db.execute(f"""
        DO $$
        DECLARE old_name text; have text[];
        BEGIN
            SELECT c.conname,
                   ARRAY(SELECT a.attname FROM unnest(c.conkey) WITH ORDINALITY k(n, i)
                         JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.n
                         ORDER BY k.i)
              INTO old_name, have
              FROM pg_constraint c
             WHERE c.conrelid = '{table}'::regclass AND c.contype = '{contype}'
               AND (c.contype = 'p' OR c.conname = '{name}'
                    OR array_length(c.conkey, 1) = {len(columns) - 1})
             LIMIT 1;
            IF have IS NOT NULL AND have::text = '{want}' THEN
                RETURN;
            END IF;
            IF old_name IS NOT NULL THEN
                EXECUTE format('ALTER TABLE {table} DROP CONSTRAINT %%I', old_name);
            END IF;
            ALTER TABLE {table} ADD CONSTRAINT {name} {kind} ({cols});
        END $$
    """)


def _rename_column(db, table: str, old: str, new: str) -> None:
    db.execute(f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{table}' AND column_name = '{old}') THEN
                ALTER TABLE {table} RENAME COLUMN {old} TO {new};
            END IF;
        END $$
    """)


def _add_market_columns(db) -> None:
    for table, key in _MARKET_KEYS.items():
        db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS market TEXT NOT NULL DEFAULT 'EG'")
        if key is None:
            lead = "user_id, market" if table.startswith("portfolio") else "market"
            db.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_market ON {table}({lead})")
        else:
            kind, cols = key
            name = f"{table}_pkey" if kind == "PRIMARY KEY" else f"{table}_market_unique"
            _ensure_key(db, table, kind, name, cols)
```

> The `alert_log` UNIQUE swap: the old constraint is an unnamed UNIQUE over 4 columns, so Postgres named it `alert_log_user_id_symbol_session_date_direction_key`. The lookup above finds it through `array_length(conkey) = len(columns) - 1`. Before committing, confirm this against a real DB with `\d alert_log` in the PG test. If it doesn't match, pass that exact old name explicitly instead.
>
> SQL rule: `_DB.execute` always passes params, so psycopg parses the whole string for placeholders. That is why the `format()` call above is written `%%I`. Keep it that way. The FakeDB tests don't execute SQL, so only the PG test and `test_no_sql_has_an_unescaped_percent` will catch a mistake here.

Then in `init_db`:
- After the last `CREATE TABLE` of the tables listed above and before the settings seeds, add:

```python
    # Multi-market (2026-10-15). Every symbol-keyed row belongs to a market;
    # existing rows become EG through the default. Idempotent like the rest.
    _rename_column(db, "risk_snapshot", "turnover_egp", "turnover")
    _rename_column(db, "pe_data", "value_traded_egp", "value_traded")
    _add_market_columns(db)
    db.execute("ALTER TABLE discovered_tickers ADD COLUMN IF NOT EXISTS exchange TEXT")
```

- Change `CREATE TABLE` bodies so fresh databases are born in the new shape:
  - Use `turnover` / `value_traded` in place of the `_egp` names.
  - Leave the PKs as they are: `_add_market_columns` rekeys a fresh DB exactly as it rekeys an old one, so there is a single code path.

- [ ] **Step 5: Update the readers of the renamed columns**

```bash
grep -rn "turnover_egp\|value_traded_egp" app tests scripts
```
Rename each SQL column reference to `turnover` / `value_traded`. Python-side names in `app/core/risk_grade.py`, `app/routers/risk.py`, `app/core/card_snapshot.py` and `app/core/pe_fetch.py` follow:
- The JSON field `liquidity_floor_egp` in `routers/risk.py:92` stays as it is for now. It's an EG-only response field, and Task 7 replaces it with `liquidity_floor` plus `currency`.
- The scanner column name in `pe_fetch.py`'s `TV_COLUMNS` (TradingView's own `Value.Traded`) is NOT renamed; only our DB column is.

- [ ] **Step 6: Run the pure tests, the full suite, and the PG test if a DB is available**

```bash
.venv/bin/python -m pytest tests/test_market_schema.py -q
.venv/bin/python -m pytest -q
docker run -d --rm --name egx-mkt-pg -e POSTGRES_PASSWORD=pw -p 55433:5432 postgres:16 && sleep 3
TEST_DATABASE_URL=postgresql://postgres:pw@localhost:55433/postgres .venv/bin/python -m pytest tests/test_market_schema_pg.py tests/test_accounts_pg.py -v
docker stop egx-mkt-pg
```
Expected:
- The pure tests and the full suite pass, matching the baseline.
- The PG tests pass. If Docker isn't available, report that the PG tests were skipped. **Don't** claim they passed.

- [ ] **Step 7: Commit**

```bash
git add app/core/db.py app/core/pe_fetch.py app/core/card_snapshot.py app/core/risk_grade.py app/routers/risk.py tests/test_market_schema.py tests/test_market_schema_pg.py
git commit -m "feat(db): market column and composite keys on symbol tables"
```

---

### Task 4: Market-scoped cache keys

**Files:**
- Modify: `app/core/cache.py:22`
- Modify all 14 `make_key(` call sites: `routers/cron.py:268`, `compare.py:66`, `dividend_history.py:50,94`, `intraday.py:47`, `ohlcv.py:32`, `analysis.py:79,229,504`, `news.py:56`, `historical.py:46`, `market_regime.py:112`, `portfolio_analysis.py:233`
- Create: `tests/test_cache_keys.py`

**Interfaces:**
- Produces: `make_key(market_code: str, *args) -> str`. It raises `ValueError` when `market_code` isn't a key of `MARKETS`. The key starts with `"<code>:"`.
- Call sites in routers that don't have a `market` yet (Tasks 6-8 add it) pass `EG.code` for now. They import `from app.core.markets import EG`. Tasks 6-8 replace `EG.code` with `market.code`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cache_keys.py
import ast
import pathlib

import pytest

from app.core.cache import make_key


def test_key_is_prefixed_by_market():
    assert make_key("EG", "ohlcv", "COMI") == "EG:ohlcv:COMI"


def test_unknown_market_is_refused():
    with pytest.raises(ValueError):
        make_key("ohlcv", "COMI")


def test_no_call_site_passes_a_literal_first_argument():
    """A literal first arg ('ohlcv', 'EG') means the market was hardcoded or forgotten."""
    bad = []
    for path in pathlib.Path("app").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "make_key":
                if node.args and isinstance(node.args[0], ast.Constant):
                    bad.append(f"{path}:{node.lineno}")
    assert bad == []
```

- [ ] **Step 2: Run and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_cache_keys.py -q`
Expected: the first two tests fail, and the AST test lists 14 sites.

- [ ] **Step 3: Implement**

```python
# app/core/cache.py — replace make_key
def make_key(market_code: str, *args) -> str:
    """
    Build a deterministic cache key, ALWAYS scoped by market.

    The same ticker can exist on two markets, and a benchmark key that forgot
    its market would hand EGX30 closes to a US beta. So the market is a
    required first argument, validated here.
    """
    from app.core.markets import MARKETS
    if market_code not in MARKETS:
        raise ValueError(f"make_key needs a market code first, got {market_code!r}")
    return ":".join(str(a) for a in (market_code, *args))
```

Update the module docstring to say trading hours live in `app/core/markets.py`.

Then at each call site, prepend `EG.code` (add `from app.core.markets import EG` where it's missing). Example: `make_key("ohlcv", symbol, exchange, interval, bars)` becomes `make_key(EG.code, "ohlcv", symbol, exchange, interval, bars)`.
- For `cache.make_key(...)` in `dividend_history.py` and `news.py`, do the same.
- The three EGX30 keys (`analysis.py:79`, `cron.py:268`, `portfolio_analysis.py:233`) become `make_key(EG.code, "benchmark", interval, bars)`, with the `"EGX"` argument dropped. Task 5 then moves all three into one helper.
- **Note:** `analysis.py:79` currently keys on its `exchange` argument while fetching `"EGX"` regardless. Keying on `"benchmark"` fixes that latent mismatch.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: the baseline results plus the new tests, all passing. `test_regime_reader_and_batch_writer_share_one_cache_key` must still pass.

- [ ] **Step 5: Commit**

```bash
git add app tests/test_cache_keys.py
git commit -m "feat(cache): every cache key is scoped by market"
```

---

### Task 5: Data layer takes a market (scanner, egxpy, benchmark)

**Files:**
- Create: `app/core/benchmark.py`
- Modify: `app/core/tradingview.py`; `app/vendor/egxpy/download.py:96,103,135` (record the change in `app/vendor/egxpy/__init__.py`); `app/core/macro_fetch.py:92`; `routers/analysis.py:70-91`; `routers/cron.py:264-275`; `routers/portfolio_analysis.py:230-240`; `routers/tickers.py:53` (scanner filter)
- Test: add to `tests/test_markets.py` → new file `tests/test_market_data_layer.py`

**Interfaces:**
- Consumes: `MarketConfig`, `EG` (Task 1); `make_key` (Task 4).
- Produces:
  - `benchmark.get_benchmark_close(market: MarketConfig, interval: str, bars: int) -> pandas.Series | None`, cached under `make_key(market.code, "benchmark", interval, bars)`.
  - `tradingview.scan(columns, market: MarketConfig = EG, timeout=..., limit=500) -> list`.
  - `tradingview.scan_url(market) -> str`.
  - `get_EGXdata(stock_list, interval, start, end, exchange="EGX", holiday_country="EG")`.
  - `get_EGX_intraday_data(stock_list, interval, start, end, exchange="EGX")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_market_data_layer.py
import json

import pandas as pd

from app.core import benchmark, tradingview
from app.core.markets import EG, US


def test_scan_url_per_market():
    assert tradingview.scan_url(EG) == "https://scanner.tradingview.com/egypt/scan"
    assert tradingview.scan_url(US) == "https://scanner.tradingview.com/america/scan"


def test_scan_filters_on_market_exchanges(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"data": []}'

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return Resp()

    monkeypatch.setattr(tradingview.urllib.request, "urlopen", fake_urlopen)
    tradingview.scan(["close"], market=US)
    assert seen["url"].endswith("/america/scan")
    assert {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]} in seen["body"]["filter"]


def test_scan_default_market_is_byte_identical_to_today(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"data": []}'

    monkeypatch.setattr(tradingview.urllib.request, "urlopen",
                        lambda req, timeout: (seen.setdefault("body", json.loads(req.data)), Resp())[1])
    tradingview.scan(["close"])
    assert {"left": "exchange", "operation": "equal", "right": "EGX"} in seen["body"]["filter"]


def test_benchmark_uses_market_symbol_and_caches(monkeypatch):
    calls = []
    df = pd.DataFrame({"Close": [1.0, 2.0]})

    def fake_ohlcv(symbol, exchange, interval, bars):
        calls.append((symbol, exchange, interval, bars))
        return df.copy()

    monkeypatch.setattr(benchmark, "_get_ohlcv", fake_ohlcv)
    benchmark.cache_clear_for_tests()
    s1 = benchmark.get_benchmark_close(US, "Daily", 400)
    s2 = benchmark.get_benchmark_close(US, "Daily", 400)
    assert list(s1) == [1.0, 2.0] and list(s2) == [1.0, 2.0]
    assert calls == [("SPX", "SP", "Daily", 400)]


def test_benchmark_returns_none_on_feed_failure(monkeypatch):
    def boom(*a):
        raise RuntimeError("feed down")
    monkeypatch.setattr(benchmark, "_get_ohlcv", boom)
    benchmark.cache_clear_for_tests()
    assert benchmark.get_benchmark_close(EG, "Daily", 400) is None
```

- [ ] **Step 2: Run and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_market_data_layer.py -q`
Expected: `ImportError: cannot import name 'benchmark'`.

- [ ] **Step 3: Implement `app/core/benchmark.py`**

```python
"""
The market's benchmark index closes (EGX30 for EG, S&P 500 for US).

Beta, relative strength and the macro regime all compare a stock against this
series, so it is fetched through ONE function under ONE cache key per market —
three hand-rolled copies (analysis, cron, portfolio_analysis) used to exist and
could drift. See CLAUDE.md "One Score Per Stock", rule 4.
"""
from __future__ import annotations

from app.core import cache
from app.core.markets import MarketConfig


def _get_ohlcv(symbol, exchange, interval, bars):
    from app.vendor.egxpy import get_OHLCV_data
    return get_OHLCV_data(symbol, exchange, interval, bars)


def cache_clear_for_tests() -> None:
    for k in [k for k in cache._store if ":benchmark:" in k]:
        del cache._store[k]


def get_benchmark_close(market: MarketConfig, interval: str, bars: int):
    ck = cache.make_key(market.code, "benchmark", interval, bars)
    df = cache.get(ck)
    if df is None:
        try:
            raw = _get_ohlcv(market.benchmark_symbol, market.benchmark_exchange, interval, bars)
        except Exception:
            return None
        if raw is None or raw.empty:
            return None
        raw.columns = [c.lower() for c in raw.columns]
        cache.set(ck, raw)
        df = raw
    return df["close"] if "close" in df.columns else None
```

Replace the three copies:
- `analysis._get_egx30_close(exchange, interval, bars)` body becomes `return get_benchmark_close(EG, interval, bars)`. Keep the function name for now, because Task 6 renames it when it gains a market.
- `cron.py:264-275`: the helper becomes `get_benchmark_close(EG, "Daily", INTERNAL_BARS_MIN)`.
- `portfolio_analysis.py:230-240`: `egx30_close = get_benchmark_close(EG, "Daily", INTERNAL_BARS_MIN)`. Keep any downstream use of the full DataFrame working. If the code there uses columns other than `close`, read the cached frame with `cache.get(make_key(EG.code, "benchmark", "Daily", INTERNAL_BARS_MIN))` after the call.
- `macro_fetch.py:92`: `get_OHLCV_data(EG.benchmark_symbol, EG.benchmark_exchange, "Daily", 30)`.

`app/core/tradingview.py`:

```python
from app.core.markets import EG, MarketConfig

SCAN_URL_TEMPLATE = "https://scanner.tradingview.com/{region}/scan"
SCAN_URL = SCAN_URL_TEMPLATE.format(region=EG.scanner_region)  # kept for existing importers


def scan_url(market: MarketConfig) -> str:
    return SCAN_URL_TEMPLATE.format(region=market.scanner_region)


def _exchange_filter(market: MarketConfig) -> dict:
    if len(market.exchanges) == 1:
        return {"left": "exchange", "operation": "equal", "right": market.exchanges[0]}
    return {"left": "exchange", "operation": "in_range", "right": list(market.exchanges)}


def scan(columns: list, market: MarketConfig = EG,
         timeout: int = DEFAULT_TIMEOUT_SECONDS, limit: int = 500) -> list:
    body = json.dumps({
        "columns": columns,
        "filter": [_exchange_filter(market), {"left": "type", "operation": "equal", "right": "stock"}],
        "range": [0, limit],
    }).encode("utf-8")
    req = urllib.request.Request(scan_url(market), data=body, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload.get("data") or []
```

`routers/tickers.py:53` builds its own scanner body. Change its filter to `_exchange_filter(EG)` imported from `tradingview`, so there's one spelling of the filter. Task 7 then threads the market through.

`app/vendor/egxpy/download.py`:
- Add `exchange: str = "EGX", holiday_country: str = "EG"` to `get_EGXdata`, and use them in place of `'EGX'` (line 103) and `'EG'` (line 96).
- Add `exchange: str = "EGX"` to `get_EGX_intraday_data` (line 135).
- Note the deviation in `app/vendor/egxpy/__init__.py`'s deviations list: "2026-10-15: exchange/holiday params on get_EGXdata/get_EGX_intraday_data (multi-market)".

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. `test_regime_reader_and_batch_writer_share_one_cache_key` is still green.

- [ ] **Step 5: Commit**

```bash
git add app tests/test_market_data_layer.py
git commit -m "feat(data): scanner, egxpy and benchmark take a market"
```

---

### Task 6: Price and scoring routers take `?market=`

**Files:**
- Modify: `routers/ohlcv.py`, `routers/analysis.py`, `routers/historical.py`, `routers/compare.py`, `routers/intraday.py`, `routers/dashboard.py`, `routers/risk.py`, `routers/market_regime.py`, `app/core/card_snapshot.py` (if it reads `risk_snapshot` or `pe_data` by symbol)
- Create: `tests/test_market_routers.py`

**Interfaces:**
- Consumes: `market_param`, `MarketConfig` (Task 1); `get_benchmark_close` (Task 5); `make_key` (Task 4).
- Produces:
  - Each endpoint's signature gains `market: MarketConfig = Depends(market_param)`.
  - Responses that are dicts gain `"market": market.code, "currency": market.currency`.
  - `analysis.composite_cache_key(symbol, interval, tags, market_code="EG")` and `analysis.read_cached_scores(symbols, interval="Daily", market=EG)` take an optional market; defaults keep existing callers working.

**The rule for every endpoint in this task:**
1. Add `market: MarketConfig = Depends(market_param)` (import `from fastapi import Depends` and `from app.core.markets import EG, MarketConfig, market_param`).
2. Replace `"EGX"` passed as an exchange with `market.primary_exchange`.
   - `ohlcv` and `analysis` keep `exchange: Optional[str] = Query(None)` as an override: `exchange = (exchange or market.primary_exchange).upper()`.
3. Replace `_get_egx30_close(...)` with `get_benchmark_close(market, interval, INTERNAL_BARS_MIN)`, and delete `_get_egx30_close`.
4. Replace `EG.code` in `make_key(...)` (added in Task 4) with `market.code`.
5. Wherever the response is a dict, add `"market": market.code, "currency": market.currency`. For list responses, leave the shape alone; `/api/tickers` in Task 7 is the only one, and it's handled there.
6. SQL that reads `risk_snapshot`, `pe_data`, `market_regime` or `score_baseline` gains `AND market = %s` (or `WHERE market = %s`) with `market.code`. SQL that writes them puts `market` in the column list, and an `ON CONFLICT (symbol)` clause becomes `ON CONFLICT (market, symbol)`.
   Find them with: `grep -n "risk_snapshot\|pe_data\|market_regime\|score_baseline" app/routers/{analysis,dashboard,risk,market_regime}.py app/core/card_snapshot.py app/core/breadth.py`.
7. `market_regime.py` reads EGX30/EGX70 constituents from the ticker file. For `market.code != "EG"` it returns `{"market": market.code, "currency": ..., "band": None, "stale": True, "n_symbols": 0}`; the US universe comes in sub-project 3. Its `make_key` call becomes `make_key(market.code, "regime", interval, len(universe))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_market_routers.py
"""Market plumbing through the routers: default EG, explicit market respected, US refused while disabled."""
import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")

from app.core.auth import CurrentUser, get_current_user  # noqa: E402
from app.main import app  # noqa: E402

ME = CurrentUser(id="u1", email="sara@example.com")


@pytest.fixture
def client(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: ME
    monkeypatch.setattr("app.main.require_authentication_enabled", False, raising=False)
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def fake_feed(monkeypatch):
    calls = []

    def fake(symbol, exchange, interval, bars):
        calls.append((symbol, exchange))
        idx = pd.date_range("2025-01-01", periods=5)
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)

    import app.vendor.egxpy as egxpy
    monkeypatch.setattr(egxpy, "get_OHLCV_data", fake)
    return calls


def test_ohlcv_defaults_to_egx(client, fake_feed):
    r = client.get("/api/ohlcv", params={"symbol": "COMI", "bars": 5})
    assert r.status_code == 200
    assert fake_feed[0] == ("COMI", "EGX")


def test_ohlcv_refuses_disabled_market(client, fake_feed, monkeypatch):
    monkeypatch.delenv("ENABLED_MARKETS", raising=False)
    r = client.get("/api/ohlcv", params={"symbol": "AAPL", "market": "US"})
    assert r.status_code == 400


def test_ohlcv_us_uses_us_exchange(client, fake_feed, monkeypatch):
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    client.get("/api/ohlcv", params={"symbol": "AAPL", "market": "US", "bars": 5})
    assert fake_feed[0] == ("AAPL", "NASDAQ")
```

Before writing more, read how `tests/test_auth_gate.py` builds its client. The middleware in `app/main.py` validates real tokens, so if a dependency override doesn't get past it, mint a token the way `test_auth_gate.py` / `test_auth_routes.py` do and send `Authorization: Bearer …`. Use that same mechanism here and delete the `require_authentication_enabled` line. Also patch `get_db` for any route that touches the DB, using `alert_fakes.FakeDB` as in `test_alerts_router.py`.

Add one test per remaining router in this task, each asserting:
1. With no `market`, the response has `"market": "EG"` (for dict responses) and the feed or SQL saw EG values.
2. `market=XX` returns 400.

For SQL routers, assert through `FakeDB.statements(...)` that the query carries `market = %s` with `"EG"` in its params.

- [ ] **Step 2: Run and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_market_routers.py -q`
Expected: the US tests fail, because `market` is still an unknown query param and is silently ignored, so the US request isn't refused.

- [ ] **Step 3: Implement**, applying the rule above router by router. Commit after each two or three routers, running the test file each time.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, including `test_dashboard_snapshot.py`, `test_risk_grade.py` and `test_fixes.py`. Those pin the one-score-per-stock invariants, which must hold within a market.

- [ ] **Step 5: Commit**

```bash
git add app tests/test_market_routers.py
git commit -m "feat(api): price and scoring routers take ?market="
```

---

### Task 7: Feed routers, cron, and remaining EG constants

**Files:**
- Modify: `routers/tickers.py`, `routers/pe.py` + `core/pe_fetch.py`, `routers/news.py` + `core/news_fetch.py:52,72`, `routers/dividend_history.py` + `core/dividend_history.py` (Yahoo suffix), `routers/calibration.py`, `routers/macro.py`, `routers/cron.py`, `core/constants.py:149,265-267`, `core/score_alerts.py:120`, `core/risk_grade.py:103-105,200-207`, `core/fundamentals_annual.py`, `core/alert_store.py:45-47`
- Test: extend `tests/test_market_routers.py`; update the existing tests that import `cron.is_isin` and `LIQUID_MIN_*_EGP`

**Interfaces:**
- Consumes: Tasks 1, 4, 5.
- Produces:
  - `pe_fetch.refresh_pe_data(db, market=EG)` and `fundamentals_annual.refresh_annual_fundamentals(db, market=EG)` write rows with `market=market.code`.
  - `risk_grade.is_tradeable(row, market=EG)` uses `market.liquid_min_turnover` / `liquid_min_price`. If they're `None`, it returns `False` (no US floors until sub-project 3).
  - `cron.is_isin` is deleted; callers use `market.is_valid_symbol`.
  - `score_alerts` uses `ZoneInfo(EG.timezone)`.
  - `constants.ALERT_TIMEZONE` stays, defined as `EG.timezone`, because tests import it.
  - `constants.DEFAULT_RISK_FREE_RATE_PCT` stays at 19, with a comment that it is EG's fallback and equals `EG.risk_free_fallback_pct`.

**Rules:**
1. Each router endpoint takes `market: MarketConfig = Depends(market_param)`. Apply rules 2, 4, 5 and 6 from Task 6.
2. **tickers:**
   - The live scanner list comes from `tradingview.scan(..., market=market)`, or from its own POST with `_exchange_filter(market)` and `scan_url(market)`.
   - The static `egx_tickers.json` merge applies only when `market.code == "EG"`.
   - `discovered_tickers` reads and writes carry `market`, and inserts store `exchange` taken from the scanner's `s` prefix (`"NASDAQ:AAPL"` → `"NASDAQ"`).
   - `index_name DEFAULT 'EGX'` stays (EG rows only).
   - `?validate=` uses `market.exchanges` for TradingView symbol-search.
   - The response stays a list, so its shape doesn't change for old clients. The market is implied by the request.
   - Module-level ticker caches are keyed by `market.code`: replace the single module global with a `dict[str, list]`.
3. **pe:** `GET /api/pe` filters `WHERE market = %s`. `POST /api/pe/refresh` refreshes EG only in this sub-project; it calls `refresh_pe_data(db, EG)`. The secret check is unchanged.
4. **news:**
   - Symbol prefixes `f"EGX:{sym}"` become `f"{market.primary_exchange}:{sym}"`.
   - The market half uses `symbols_in_index(...)` only for EG. For other markets it is empty until sub-project 4.
   - Arab Finance is fetched only for EG.
   - Holdings and watchlist reads filter by market (see Task 8's SQL).
5. **dividend_history:**
   - The Yahoo suffix `.CA` applies only for EG; US uses the bare symbol.
   - Add `YAHOO_SUFFIX = {"EG": ".CA", "US": ""}` in `core/dividend_history.py` and pass `market` through.
   - The `dividend_events` SQL carries `market`.
6. **macro:** EG only in this sub-project. For `market.code != "EG"` it returns `{"market": ..., "currency": ..., "egx30": None, "usd_egp": None, "interest_rate": None}`.
7. **calibration:** EG constants. It adds `market`/`currency` to the response and returns 400 for non-EG with detail `"Calibration is EGX-only until US is calibrated"`.
8. **cron (`/api/cron/risk_snapshot`):**
   - It takes `market` like the others (default EG; the cron-job.org URL is unchanged).
   - `is_isin(symbol)` becomes `not market.is_valid_symbol(symbol)`.
   - `get_OHLCV_data(symbol, "EGX", ...)` uses `market.primary_exchange`.
   - Every `risk_snapshot` / `score_baseline` read and write carries `market`.
   - `run_alerts` passes `market.code` into `alert_store`.
9. **alert_store subscribers (lines 45-47):** the `UNION` gains `WHERE market = %s` on both halves. `alert_log` inserts carry `market`, and `ON CONFLICT (user_id, symbol, session_date, direction)` becomes `ON CONFLICT (user_id, market, symbol, session_date, direction)`.
10. **risk.py response:** `liquidity_floor_egp` becomes two fields, `"liquidity_floor": market.liquid_min_turnover` and `"currency": market.currency`. Search mobile and web for `liquidity_floor_egp` (`grep -rn liquidity_floor_egp ../egx-api-fe/src ../egx-api-mobile/src`). If either reads it, keep `liquidity_floor_egp` as well for EG, so old clients don't break, and note it in the commit message.

- [ ] **Step 1: Update the tests that reference moved names**

```bash
grep -rn "is_isin\|LIQUID_MIN_TURNOVER_EGP\|LIQUID_MIN_PRICE_EGP\|liquidity_floor_egp" tests
```
- Tests that call `cron.is_isin(x)` switch to `not EG.is_valid_symbol(x)`. Keep the same assertions, including EGSA.
- Tests that read the `LIQUID_MIN_*_EGP` constants switch to `EG.liquid_min_turnover` / `EG.liquid_min_price`.

- [ ] **Step 2: Add router tests** to `tests/test_market_routers.py`, one per router in this task:
1. With the default market, the SQL carries `"EG"` and the response has `market: "EG"` where it's a dict.
2. `market=XX` returns 400.
3. For news, the TradingView headline URL uses `EGX:` by default. Monkeypatch `news_fetch`'s HTTP function and assert the URL it was asked for.
4. For dividend_history, the Yahoo URL ends `.CA` for EG. With `ENABLED_MARKETS=EG,US` and `market=US`, it has no suffix.

- [ ] **Step 3: Run and confirm the new tests fail**, then implement the rules above.

- [ ] **Step 4: Check that nothing EG-specific is still hardcoded outside the registry**

```bash
grep -rn "\"EGX\"\|'EGX'\|\"EGX30\"\|Africa/Cairo\|egypt/scan" app --include=*.py | grep -v "app/core/markets.py\|app/vendor/"
```
Expected: matches only in:
- docstrings and comments
- user-facing EG-only copy in `composite.py` / `portfolio_analysis.py` reason strings (sub-project 3 makes those market-aware)
- `index_membership.py` tier names
- `market_regime.py`'s `_REGIME_INDEXES`

List any other match in the commit message, or fix it.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add app tests
git commit -m "feat(api): feed routers, cron and alerts take a market; EG constants read from registry"
```

---

### Task 8: Per-user data is scoped by market

**Files:**
- Modify: `routers/watchlist.py`, `routers/portfolio.py`, `routers/sales.py`, `routers/dividends.py`, `routers/portfolio_analysis.py`, `core/holdings.py:40-70`, `core/dividends.py:235-265`
- Test: `tests/test_market_routers.py`, plus the existing `tests/test_sell_tracking.py`, `tests/test_dividends.py` and `tests/test_position_grouping.py` (all must stay green)

**Interfaces:**
- Consumes: `market_param`, `MarketConfig` (Task 1); `get_benchmark_close` (Task 5).
- Produces:
  - `holdings.fetch_open_holdings(db, user_id, market_code="EG")` and `holdings.fetch_open_lots(db, user_id, symbol, market_code="EG")` filter by market.
  - `dividends.fetch_dividends(db, user_id, market_code="EG")` filters by market. Use the real function names in `core/dividends.py:235-265`; the default keeps other callers working.
  - Responses carry `"market"`, and `"currency": market.currency`, which replaces the read of `settings.currency` in `portfolio.py:31-35` and `sales.py:203-216`.
- Request bodies:
  - `WatchlistAdd`, the portfolio create model, the sale model and the dividend model gain `market: Optional[str] = None`.
  - The handler resolves `get_market(body.market or market.code)`, so a body value beats the query value, and absent means the query value, which defaults to EG.
  - **Update and delete by `id` don't take a market.** They act on the row, which already has one.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_market_routers.py`)

```python
import app.routers.watchlist as watchlist_mod
from alert_fakes import FakeDB
from app.core.markets import EG, US


def test_watchlist_is_scoped_by_market(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(watchlist_mod, "get_db", lambda: db)
    watchlist_mod.get_watchlist(market=EG, user=ME)
    [(sql, params)] = db.statements("SELECT symbol FROM watchlist")
    assert "market = %s" in sql and params == ("u1", "EG")


def test_watchlist_add_writes_market_and_new_conflict_target(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(watchlist_mod, "get_db", lambda: db)
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    out = watchlist_mod.add_to_watchlist(watchlist_mod.WatchlistAdd(symbol="aapl", market="US"), market=EG, user=ME)
    [(sql, params)] = db.statements("INSERT INTO watchlist")
    assert "ON CONFLICT (user_id, market, symbol)" in sql
    assert params[:3] == ("u1", "US", "AAPL")
    assert out["market"] == "US"


def test_watchlist_delete_is_scoped(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(watchlist_mod, "get_db", lambda: db)
    watchlist_mod.remove_from_watchlist(symbol="COMI", market=EG, user=ME)
    [(sql, params)] = db.statements("DELETE FROM watchlist")
    assert params == ("u1", "EG", "COMI")
```

Add the same three-case pattern (list filtered, create writes market, response carries currency) for portfolio, sales and dividends, plus:

```python
def test_portfolio_currency_comes_from_market_not_settings(monkeypatch):
    import app.routers.portfolio as portfolio_mod
    db = FakeDB()
    monkeypatch.setattr(portfolio_mod, "get_db", lambda: db)
    out = portfolio_mod.get_portfolio(market=EG, user=ME)
    assert out["currency"] == "EGP" and out["market"] == "EG"
    assert db.statements("SELECT value FROM settings WHERE key = 'currency'") == []
```

Adjust the function names (`get_portfolio`, etc.) to the real ones in each router before running.

- [ ] **Step 2: Run and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_market_routers.py -q -k "watchlist or portfolio or sales or dividend"`
Expected: FAIL, because the handlers don't take `market` yet.

- [ ] **Step 3: Implement.** Watchlist as the reference:

```python
from app.core.markets import MarketConfig, get_market, market_param


class WatchlistAdd(BaseModel):
    symbol: str
    market: Optional[str] = None


@router.get("/api/watchlist")
def get_watchlist(market: MarketConfig = Depends(market_param),
                  user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        rows = db.execute(
            "SELECT symbol FROM watchlist WHERE user_id = %s AND market = %s ORDER BY added_at ASC",
            (user.id, market.code),
        ).fetchall()
        return {"symbols": [r[0] for r in rows], "market": market.code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- In `add_to_watchlist`, add `market: MarketConfig = Depends(market_param)` and set `target = get_market(body.market) if body.market else market`.
- The insert is `INSERT INTO watchlist (user_id, market, symbol, added_at) VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, market, symbol) DO NOTHING`, with params `(user.id, target.code, symbol, now)`.
- Return `{"symbol": symbol, "added_at": now, "market": target.code}`.
- `remove_from_watchlist` adds `AND market = %s`, with params `(user.id, market.code, symbol)`.

Apply the same pattern to portfolio, sales and dividends:
- **Portfolio insert:** add `market` to the column list and use `target.code`.
- **Sale:** a sale's market is the holding's market. Read it from the holding row by `id`; the body's market is ignored. The FIFO lots come from `fetch_open_lots(db, user.id, symbol, holding_market)`.
- **Dividend duplicate guard (409):** add `market` to the `symbol + pay_date + amount` match.
- **`portfolio_analysis`:**
  - Takes `market` and passes it to `fetch_open_holdings`.
  - Uses `market.primary_exchange` for each holding's fetch and `get_benchmark_close(market, "Daily", INTERNAL_BARS_MIN)` for the benchmark.
  - Adds `market` / `currency` to its response.
  - `pe_data` and `risk_snapshot` lookups carry `market.code`.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. The sell-tracking and position-grouping invariants still hold.

- [ ] **Step 5: Commit**

```bash
git add app tests
git commit -m "feat(api): portfolio, sales, dividends and watchlist are scoped by market"
```

---

### Task 9: Backend verification gate

- [ ] **Step 1: Run the full suite in tmux, then the PG suite**

```bash
tmux kill-session -t egx-api-be-tests 2>/dev/null
tmux new -d -s egx-api-be-tests "cd /home/mark/projects/egx-api/egx-api-be && .venv/bin/python -m pytest -q 2>&1 | tee ~/logs/egx-api-be-tests.log"
```
Then re-run the PG tests exactly as in Task 3, Step 6.
Expected: no failures beyond the baseline's; the same counts as baseline plus the new tests.

- [ ] **Step 2: Boot the app locally and smoke-test it.** Use a throwaway Neon branch URL or the docker PG from Task 3; never production.

```bash
tmux new -d -s egx-api-be-dev "cd /home/mark/projects/egx-api/egx-api-be && DATABASE_URL=postgresql://postgres:pw@localhost:55433/postgres AUTH_SECRET=dev .venv/bin/uvicorn app.main:app --port 8001 2>&1 | tee ~/logs/egx-api-be-dev.log"
```
Then, with a token minted for a seeded user (as `test_auth_routes.py` does), run:

```bash
curl -s -H "Authorization: Bearer $T" localhost:8001/api/markets
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $T" "localhost:8001/api/watchlist?market=US"
```
Expected: the first returns one market, `EG`. The second returns `400`.

- [ ] **Step 3: Stop the sessions**

```bash
tmux kill-session -t egx-api-be-dev
docker stop egx-mkt-pg
```

---

## Phase B: Shared `lib/` and web (`egx-api-fe`)

### Task 10: `markets.ts`, `money.ts`, and the market-aware API client

**Files:**
- Create: `src/app/lib/markets.ts`, `src/app/lib/money.ts`
- Modify: `src/app/lib/api.ts`, `src/app/lib/types.ts`
- Modify: `egx-api-be/docs/superpowers/specs/2026-10-15-multi-market-foundation-design.md` (rename `format.ts` → `money.ts`)
- Tests for these files run in the mobile repo (Task 12), which has jest. The web has none.

**Interfaces:**
- Produces (exact exports):

```ts
// markets.ts
export type MarketCode = "EG" | "US";
export interface Market {
  code: MarketCode; name: string; flag: string; currency: string; locale: string;
  timezone: string; trading_days: number[]; open: string; close: string;
  benchmark_label: string; holidays: { date: string; name: string }[];
}
export const DEFAULT_MARKET: MarketCode;          // "EG"
export const EG_FALLBACK: Market;
export function isMarketOpen(market: Market, nowMs: number): boolean;

// money.ts
export function formatMoney(value: number, currency: string, opts?: { digits?: number; withCode?: boolean }): string;
export function formatPrice(value: number, currency: string): string;

// api.ts (new exports)
export function setApiMarket(code: MarketCode): void;
export function getApiMarket(): MarketCode;
export function withMarket(url: string, market?: MarketCode | null): string;
export async function fetchMarkets(): Promise<{ markets: Market[]; default: MarketCode }>;
export async function fetchMarketPref(): Promise<{ market: MarketCode }>;
export async function putMarketPref(code: MarketCode): Promise<{ market: MarketCode }>;
```

- [ ] **Step 1: Write `markets.ts`**

```ts
/**
 * Markets the app can show. The BACKEND registry (egx-api-be/app/core/markets.py)
 * is the source of truth and is served by GET /api/markets; EG_FALLBACK mirrors
 * its EG entry so the app still boots offline or against an older backend.
 *
 * Shared with egx-api-mobile (npm run sync:shared) — keep platform-neutral.
 */
export type MarketCode = "EG" | "US";

export interface Market {
  code: MarketCode;
  name: string;
  flag: string;
  currency: string;
  locale: string;
  timezone: string;
  /** JS-agnostic: Python weekday() numbers, Mon=0 .. Sun=6, as the API sends them. */
  trading_days: number[];
  open: string;   // "HH:MM" local
  close: string;  // "HH:MM" local, inclusive
  benchmark_label: string;
  holidays: { date: string; name: string }[];
}

export const DEFAULT_MARKET: MarketCode = "EG";

export const EG_FALLBACK: Market = {
  code: "EG",
  name: "Egypt",
  flag: "🇪🇬",
  currency: "EGP",
  locale: "en-EG",
  timezone: "Africa/Cairo",
  trading_days: [6, 0, 1, 2, 3],
  open: "10:00",
  close: "14:30",
  benchmark_label: "EGX30",
  holidays: [],
};

const WEEKDAY: Record<string, number> = { Mon: 0, Tue: 1, Wed: 2, Thu: 3, Fri: 4, Sat: 5, Sun: 6 };

const toMinutes = (hhmm: string) => {
  const [h, m] = hhmm.split(":").map((n) => parseInt(n, 10));
  return h * 60 + m;
};

/**
 * Is `market` open at `nowMs`? Read through Intl in the market's own timezone,
 * so it is right from any device timezone and across DST.
 *
 * If this JS engine cannot resolve the timezone it answers OPEN: a card then
 * shows its real "...ago" age, always true, rather than a "Market closed" that
 * might be false. (Same rule the EGX-only version had.)
 */
export function isMarketOpen(market: Market, nowMs: number): boolean {
  let parts: Intl.DateTimeFormatPart[];
  try {
    parts = new Intl.DateTimeFormat("en-US", {
      timeZone: market.timezone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      weekday: "short",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).formatToParts(new Date(nowMs));
  } catch {
    return true;
  }
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "";
  const weekday = WEEKDAY[get("weekday")];
  if (weekday === undefined) return true;
  if (!market.trading_days.includes(weekday)) return false;
  const iso = `${get("year")}-${get("month")}-${get("day")}`;
  if (market.holidays.some((h) => h.date === iso)) return false;
  let hour = parseInt(get("hour"), 10);
  if (hour === 24) hour = 0; // some engines emit "24" for midnight
  const minutes = hour * 60 + parseInt(get("minute"), 10);
  if (Number.isNaN(minutes)) return true;
  return minutes >= toMinutes(market.open) && minutes <= toMinutes(market.close);
}
```

- [ ] **Step 2: Write `money.ts`**

```ts
/**
 * Money in a market's own currency. One spelling, so no screen hardcodes "EGP".
 * The code is APPENDED ("1,234.50 EGP"), matching how the app has always shown
 * money, rather than Intl's symbol placement — "E£" is not what EGX users read.
 *
 * Shared with egx-api-mobile — keep platform-neutral.
 */
export function formatMoney(
  value: number,
  currency: string,
  opts: { digits?: number; withCode?: boolean } = {}
): string {
  const { digits = 2, withCode = true } = opts;
  const n = value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return withCode ? `${n} ${currency}` : n;
}

/** A share price: two decimals, with the currency code. */
export function formatPrice(value: number, currency: string): string {
  return formatMoney(value, currency, { digits: 2 });
}
```

- [ ] **Step 3: Make `api.ts` market-aware**

Add these near `base()`:

```ts
import { DEFAULT_MARKET, type Market, type MarketCode } from "./markets";

let currentMarket: MarketCode = DEFAULT_MARKET;

/** Set by each app's market provider; every data request carries it. */
export function setApiMarket(code: MarketCode): void {
  currentMarket = code;
}

export function getApiMarket(): MarketCode {
  return currentMarket;
}

/** Routes that are not market data never get `market=`. */
const MARKETLESS = ["/auth/", "/markets", "/settings", "/users", "/push/", "/alerts/preferences"];

/**
 * Append `market=` to a URL. `market` overrides the current one (a deep link
 * into the other market); `null` means "send none". An explicit market= already
 * in the URL wins.
 */
export function withMarket(url: string, market?: MarketCode | null): string {
  if (market === null) return url;
  if (MARKETLESS.some((p) => url.includes(p))) return url;
  if (/[?&]market=/.test(url)) return url;
  return `${url}${url.includes("?") ? "&" : "?"}market=${market ?? currentMarket}`;
}
```

- Change `fetchJSON`'s signature to `options?: RequestInit & { timeoutMs?: number; market?: MarketCode | null }`.
- Destructure it with `const { timeoutMs, market, ...init } = options ?? {};`.
- Call `fetch(withMarket(url, market), {...})`.

Add these at the end:

```ts
// ---- Markets ----

export async function fetchMarkets(): Promise<{ markets: Market[]; default: MarketCode }> {
  return fetchJSON(`${base()}/markets`);
}

export async function fetchMarketPref(): Promise<{ market: MarketCode }> {
  return fetchJSON(`${base()}/settings?section=market`);
}

export async function putMarketPref(code: MarketCode): Promise<{ market: MarketCode }> {
  return fetchJSON(`${base()}/settings?section=market`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ market: code }),
  });
}
```

Find every place in `api.ts` that reads the renamed backend fields `turnover_egp` / `value_traded_egp` (`grep -n "_egp" src/app/lib/api.ts src/app/lib/types.ts`) and rename them to `turnover` / `value_traded`. Leave `var_95_egp` alone; that portfolio metric's rename is out of scope.

- [ ] **Step 4: Update `types.ts`**

Add `market?: MarketCode; currency?: string;` (optional, so an older backend still type-checks) to:
- `Portfolio`, which already has `currency`, so add `market` only
- `SalesResponse`, which has `currency` at :716, so add `market` only
- `AnalysisResponse`, `PortfolioAnalysisResponse`, `NewsResponse`, `DividendHistoryResponse`, `DividendCalendarResponse`, `MacroData`

Import `MarketCode` from `./markets`.

- [ ] **Step 5: Type-check and build**

```bash
cd /home/mark/projects/egx-api/egx-api-fe
npx tsc --noEmit
tmux kill-session -t egx-api-fe-build 2>/dev/null
tmux new -d -s egx-api-fe-build "npm run build 2>&1 | tee ~/logs/egx-api-fe-build.log"
```
Expected: `tsc` reports no errors, and the build log ends with route sizes and no error.

- [ ] **Step 6: Update the spec's filename and commit both repos**

In the backend spec, replace `` `format.ts` `` with `` `money.ts` `` (two places, section 4).

```bash
cd /home/mark/projects/egx-api/egx-api-fe && git add src/app/lib && git commit -m "feat(lib): markets, money formatting and market-aware API client"
cd ../egx-api-be && git add docs/superpowers/specs && git commit -m "docs(spec): shared money.ts name"
```

---

### Task 11: Web uses the shared clock and money formatter

**Files:**
- Modify: `src/app/page.tsx:134-156,577`
- Modify these components: `DividendHistoryCard`, `MonteCarloChart`, `ForecastCard`, `AddHoldingForm`, `StatsPanel`, `SellHoldingForm`, `RealizedSection`, `RiskDashboard`, `KeyLevelsCard`, `PortfolioSummary`, `EntryExitCard`, `AddDividendForm`, `HoldingsTable`, `RiskGradeCard`
- Modify: `src/app/dividends/page.tsx`
- **Not touched:** `MacroCard` (its "USD/EGP" is an FX pair name, not a currency amount), the Learn components, and `curriculum.ts` (EGX teaching content; out of scope per the spec).

**Interfaces:**
- Consumes: `isMarketOpen`, `EG_FALLBACK` (markets.ts); `formatMoney`, `formatPrice` (money.ts).

- [ ] **Step 1: Swap the market clock in `page.tsx`.** Delete the local `isEgxOpen` (lines ~134-156) and add `import { EG_FALLBACK, isMarketOpen } from "./lib/markets";`. Line 577 becomes `const marketClosed = !isMarketOpen(EG_FALLBACK, now);`.

- [ ] **Step 2: Swap the money literals.** For each file:

```bash
grep -n "EGP" src/app/components/<File>.tsx
```
Replace each amount rendered with a literal `EGP` suffix by `formatMoney(...)` or `formatPrice(...)`, taking the currency from the nearest response object (`portfolio.currency`, `sales.currency`, `analysis.currency`), with `?? "EGP"` as the fallback.
- Example: `` `${x.toFixed(2)} EGP` `` becomes `formatPrice(x, currency)`. Keep the existing digit count by passing `{ digits: 0 }` where the old code used `toLocaleString()` with no decimals.
- Labels like `"Buy Price (EGP) *"` become `` `Buy Price (${currency}) *` ``.
- Where a component has no response in scope, add a `currency: string` prop and pass it from the page that renders it.
- Prose strings that *explain* EGX facts (tooltips such as the P/E explanation) are left as they are; their market-awareness comes in sub-project 3.

- [ ] **Step 3: Confirm only the intended literals remain**

```bash
grep -rn "EGP" src/app --include=*.tsx | grep -v "learn\|MacroCard\|curriculum"
```
Expected: only `?? "EGP"` fallbacks and tooltip prose remain. List the prose ones in the commit body.

- [ ] **Step 4: Lint and build**

```bash
npm run lint
tmux kill-session -t egx-api-fe-build 2>/dev/null
tmux new -d -s egx-api-fe-build "npm run build 2>&1 | tee ~/logs/egx-api-fe-build.log"
```
Expected: no lint errors, and the build succeeds.

- [ ] **Step 5: Check visually against the local backend from Task 9.** Run the web in tmux and open the dashboard and portfolio. With EG data, amounts must read exactly as before, e.g. `42.40 EGP`.

```bash
tmux new -d -s egx-api-fe-dev "NEXT_PUBLIC_API_BASE_URL=http://localhost:8001/api npm run dev 2>&1 | tee ~/logs/egx-api-fe-dev.log"
```

- [ ] **Step 6: Commit**

```bash
git add src/app && git commit -m "refactor(web): shared market clock and money formatter"
```

---

## Phase C: Mobile (`egx-api-mobile`)

### Task 12: Sync the new shared files and test them

**Files:**
- Modify: `scripts/sync-shared.mjs` (`SHARED_FILES`)
- Generated: `src/shared/markets.ts`, `src/shared/money.ts`, `src/shared/api.ts`, `src/shared/types.ts`
- Test: `src/__tests__/shared.test.ts`

- [ ] **Step 1: Write the failing tests** (append to `src/__tests__/shared.test.ts`)

```ts
import { EG_FALLBACK, isMarketOpen, type Market } from "@/shared/markets";
import { formatMoney, formatPrice } from "@/shared/money";
import { setApiMarket, withMarket } from "@/shared/api";

const US: Market = {
  ...EG_FALLBACK,
  code: "US",
  currency: "USD",
  timezone: "America/New_York",
  trading_days: [0, 1, 2, 3, 4],
  open: "09:30",
  close: "16:00",
  holidays: [{ date: "2026-11-26", name: "Thanksgiving Day" }],
};

describe("isMarketOpen", () => {
  it.each([
    ["2026-10-11T08:00:00Z", true],   // Sun 11:00 Cairo
    ["2026-10-11T11:31:00Z", false],  // Sun 14:31 Cairo
    ["2026-10-09T09:00:00Z", false],  // Friday
  ])("EG at %s -> %s", (iso, open) => {
    expect(isMarketOpen(EG_FALLBACK, Date.parse(iso))).toBe(open);
  });

  it.each([
    ["2026-10-14T13:30:00Z", true],   // 09:30 EDT
    ["2026-11-04T13:30:00Z", false],  // 08:30 EST — DST ended
    ["2026-11-04T14:30:00Z", true],   // 09:30 EST
    ["2026-11-26T15:00:00Z", false],  // holiday from the list
  ])("US at %s -> %s", (iso, open) => {
    expect(isMarketOpen(US, Date.parse(iso))).toBe(open);
  });
});

describe("formatMoney", () => {
  it("appends the market's code", () => {
    expect(formatMoney(2540, "EGP")).toBe("2,540.00 EGP");
    expect(formatPrice(189.5, "USD")).toBe("189.50 USD");
    expect(formatMoney(1234.5, "USD", { digits: 0, withCode: false })).toBe("1,235");
  });
});

describe("withMarket", () => {
  afterEach(() => setApiMarket("EG"));

  it("adds the current market to data routes", () => {
    setApiMarket("US");
    expect(withMarket("/api/portfolio")).toBe("/api/portfolio?market=US");
    expect(withMarket("/api/analysis?symbol=COMI")).toBe("/api/analysis?symbol=COMI&market=US");
  });

  it("leaves auth, settings and markets routes alone", () => {
    expect(withMarket("/api/auth/me")).toBe("/api/auth/me");
    expect(withMarket("/api/settings?section=market")).toBe("/api/settings?section=market");
    expect(withMarket("/api/markets")).toBe("/api/markets");
  });

  it("honours an explicit override and null", () => {
    expect(withMarket("/api/analysis?symbol=COMI", "EG")).toBe("/api/analysis?symbol=COMI&market=EG");
    expect(withMarket("/api/analysis?symbol=COMI", null)).toBe("/api/analysis?symbol=COMI");
  });
});
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `npx jest src/__tests__/shared.test.ts`
Expected: FAIL, `Cannot find module '@/shared/markets'`.

- [ ] **Step 3: Add the files to `SHARED_FILES` and sync**

In `scripts/sync-shared.mjs`, add `"markets.ts",` and `"money.ts",` to the `SHARED_FILES` array. Then run:

```bash
npm run sync:shared && npm run check:shared
```
Expected: `check:shared` exits 0.

- [ ] **Step 4: Run them again and confirm they pass**

Run: `npx jest src/__tests__/shared.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/sync-shared.mjs src/shared src/__tests__/shared.test.ts
git commit -m "feat(shared): sync markets and money modules"
```

---

### Task 13: `MarketProvider`

**Files:**
- Create: `src/providers/market.tsx`, `src/providers/__tests__/market.test.tsx`
- Modify: `src/providers/AppProviders.tsx`

**Interfaces:**
- Consumes: `fetchMarkets`, `fetchMarketPref`, `putMarketPref`, `setApiMarket` (shared api); `EG_FALLBACK`, `Market`, `MarketCode` (shared markets); `useAuth` (`@/auth/AuthProvider`).
- Produces:

```ts
export interface MarketCtx {
  market: Market;          // the selected market's display fields
  markets: Market[];       // enabled markets; length 1 hides the changer
  setMarket: (code: MarketCode) => void;
  loading: boolean;
}
export function MarketProvider({ children }: { children: React.ReactNode }): JSX.Element;
export function useMarket(): MarketCtx;  // outside a provider: EG_FALLBACK, [EG_FALLBACK], no-op
export const MARKET_STORAGE_KEY = "egx.market";
```

- [ ] **Step 1: Write the failing tests**

Look at an existing provider test in `src/providers/__tests__/` to see how `useAuth` and the shared API are mocked, and copy that setup. The cases:

```tsx
// src/providers/__tests__/market.test.tsx
import AsyncStorage from "@react-native-async-storage/async-storage";
import { act, renderHook, waitFor } from "@testing-library/react-native";
import React from "react";

import * as api from "@/shared/api";
import { EG_FALLBACK } from "@/shared/markets";
import { MARKET_STORAGE_KEY, MarketProvider, useMarket } from "../market";

jest.mock("@/auth/AuthProvider", () => ({
  useAuth: () => ({ isAuthenticated: true, user: { id: "u1" } }),
}));

const US = { ...EG_FALLBACK, code: "US" as const, currency: "USD", name: "United States" };
const wrapper = ({ children }: { children: React.ReactNode }) => <MarketProvider>{children}</MarketProvider>;

beforeEach(async () => {
  await AsyncStorage.clear();
  jest.restoreAllMocks();
  api.setApiMarket("EG");
});

it("boots from storage, then the server wins", async () => {
  await AsyncStorage.setItem(MARKET_STORAGE_KEY, "US");
  jest.spyOn(api, "fetchMarkets").mockResolvedValue({ markets: [EG_FALLBACK, US], default: "EG" });
  jest.spyOn(api, "fetchMarketPref").mockResolvedValue({ market: "EG" });
  const { result } = renderHook(() => useMarket(), { wrapper });
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.market.code).toBe("EG");
  expect(api.getApiMarket()).toBe("EG");
  expect(await AsyncStorage.getItem(MARKET_STORAGE_KEY)).toBe("EG");
});

it("falls back to EG when /markets fails", async () => {
  jest.spyOn(api, "fetchMarkets").mockRejectedValue(new Error("offline"));
  jest.spyOn(api, "fetchMarketPref").mockRejectedValue(new Error("offline"));
  const { result } = renderHook(() => useMarket(), { wrapper });
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.market.code).toBe("EG");
  expect(result.current.markets).toHaveLength(1);
});

it("ignores a stored market that is not enabled", async () => {
  await AsyncStorage.setItem(MARKET_STORAGE_KEY, "US");
  jest.spyOn(api, "fetchMarkets").mockResolvedValue({ markets: [EG_FALLBACK], default: "EG" });
  jest.spyOn(api, "fetchMarketPref").mockResolvedValue({ market: "EG" });
  const { result } = renderHook(() => useMarket(), { wrapper });
  await waitFor(() => expect(result.current.loading).toBe(false));
  expect(result.current.market.code).toBe("EG");
});

it("setMarket persists locally and remotely", async () => {
  jest.spyOn(api, "fetchMarkets").mockResolvedValue({ markets: [EG_FALLBACK, US], default: "EG" });
  jest.spyOn(api, "fetchMarketPref").mockResolvedValue({ market: "EG" });
  const put = jest.spyOn(api, "putMarketPref").mockResolvedValue({ market: "US" });
  const { result } = renderHook(() => useMarket(), { wrapper });
  await waitFor(() => expect(result.current.loading).toBe(false));
  act(() => result.current.setMarket("US"));
  expect(result.current.market.code).toBe("US");
  expect(api.getApiMarket()).toBe("US");
  expect(put).toHaveBeenCalledWith("US");
  await waitFor(async () => expect(await AsyncStorage.getItem(MARKET_STORAGE_KEY)).toBe("US"));
});

it("keeps the local choice when the remote save fails", async () => {
  jest.spyOn(api, "fetchMarkets").mockResolvedValue({ markets: [EG_FALLBACK, US], default: "EG" });
  jest.spyOn(api, "fetchMarketPref").mockResolvedValue({ market: "EG" });
  jest.spyOn(api, "putMarketPref").mockRejectedValue(new Error("500"));
  const { result } = renderHook(() => useMarket(), { wrapper });
  await waitFor(() => expect(result.current.loading).toBe(false));
  act(() => result.current.setMarket("US"));
  await waitFor(() => expect(result.current.market.code).toBe("US"));
});
```

If `jest.setup.ts` doesn't already mock AsyncStorage, add `jest.mock("@react-native-async-storage/async-storage", () => require("@react-native-async-storage/async-storage/jest/async-storage-mock"));` there.

- [ ] **Step 2: Run them and confirm they fail**

Run: `npx jest src/providers/__tests__/market.test.tsx`
Expected: FAIL, `Cannot find module '../market'`.

- [ ] **Step 3: Implement**

```tsx
// src/providers/market.tsx
/**
 * Which market (EG / US) the app is showing. Persisted on the device for an
 * instant boot and on the server so it follows the user across devices; the
 * server wins once it answers. Every shared API call reads it via setApiMarket.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { useAuth } from "@/auth/AuthProvider";
import { fetchMarketPref, fetchMarkets, putMarketPref, setApiMarket } from "@/shared/api";
import { DEFAULT_MARKET, EG_FALLBACK, type Market, type MarketCode } from "@/shared/markets";

export const MARKET_STORAGE_KEY = "egx.market";

export interface MarketCtx {
  market: Market;
  markets: Market[];
  setMarket: (code: MarketCode) => void;
  loading: boolean;
}

const Ctx = createContext<MarketCtx | null>(null);

const pick = (markets: Market[], code: string | null | undefined): Market =>
  markets.find((m) => m.code === code) ?? markets.find((m) => m.code === DEFAULT_MARKET) ?? EG_FALLBACK;

export function MarketProvider({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, user } = useAuth();
  const [markets, setMarkets] = useState<Market[]>([EG_FALLBACK]);
  const [code, setCode] = useState<MarketCode>(DEFAULT_MARKET);
  const [loading, setLoading] = useState(true);

  const apply = useCallback((next: MarketCode) => {
    setApiMarket(next);
    setCode(next);
    AsyncStorage.setItem(MARKET_STORAGE_KEY, next).catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const stored = await AsyncStorage.getItem(MARKET_STORAGE_KEY).catch(() => null);
      if (!cancelled && (stored === "EG" || stored === "US")) {
        setApiMarket(stored);
        setCode(stored);
      }
      if (!isAuthenticated) {
        if (!cancelled) setLoading(false);
        return;
      }
      const [list, pref] = await Promise.all([
        fetchMarkets().catch(() => null),
        fetchMarketPref().catch(() => null),
      ]);
      if (cancelled) return;
      const enabled = list?.markets?.length ? list.markets : [EG_FALLBACK];
      setMarkets(enabled);
      apply(pick(enabled, pref?.market ?? stored).code);
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [isAuthenticated, user?.id, apply]);

  const setMarket = useCallback(
    (next: MarketCode) => {
      if (next === code) return;
      apply(next);
      putMarketPref(next).catch(() => {
        // The device keeps the choice; the next successful boot syncs it.
      });
    },
    [apply, code]
  );

  const value = useMemo<MarketCtx>(
    () => ({ market: pick(markets, code), markets, setMarket, loading }),
    [markets, code, setMarket, loading]
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

const OUTSIDE: MarketCtx = { market: EG_FALLBACK, markets: [EG_FALLBACK], setMarket: () => {}, loading: false };

export function useMarket(): MarketCtx {
  return useContext(Ctx) ?? OUTSIDE;
}
```

> The spec mentions a toast when the remote save fails. The app has no toast primitive (`src/ui` has none), and adding one is out of scope, so the failure is silent and the next boot re-syncs. Note this in the commit body.

In `AppProviders.tsx`, wrap the existing tree in `<MarketProvider>` as the outermost element, and update the comment to say it is the outermost because every other provider's fetches carry the market.

- [ ] **Step 4: Run them again and confirm they pass**

Run: `npx jest src/providers`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/providers jest.setup.ts
git commit -m "feat(mobile): MarketProvider persists and syncs the selected market"
```

---

### Task 14: Refetch when the market changes

**Files:**
- Modify: `src/providers/tickers.tsx`, `src/providers/watchlist.tsx`
- Modify: `src/features/market/{DividendCalendarScreen,DividendHistoryCard,RiskGradeCard,MarketRegimeCard,StockDetailScreen,CompareScreen,NewsScreen}.tsx`
- Modify: `src/features/market/hooks/{useLiveUpgrade,useDashboardSnapshot}.ts`
- Modify: `src/features/calibration/CalibrationScreen.tsx`, `src/features/portfolio/usePortfolioData.ts`, `src/features/common/PEFreshnessBanner.tsx`
- Modify: `src/features/learn/liveData.ts`, which is always EG
- Test: `src/providers/__tests__/tickers.market.test.tsx`

**Interfaces:**
- Consumes: `useMarket` (Task 13); the `market` option on `fetchJSON` (Task 10).

**Rule:** in each file, add `const { market } = useMarket();` and add `market.code` to the dependency array of every effect or callback that calls the shared API.
- The existing `let cancelled = false … return () => { cancelled = true; }` pattern already throws away a response that arrives after the dependencies change. That is the stale-response guard, so keep it. If an effect lacks it, add it.
- Hooks with their own caches (`useDashboardSnapshot`, `useLiveUpgrade`) must key those caches by `market.code`. Grep each for `useRef(new Map` and `useState<Record` and include the code in the key.
- `learn/liveData.ts` teaches with COMI, an EGX stock. Its fetch passes `{ market: "EG" }` explicitly so the Learn page works in US mode.

- [ ] **Step 1: Write the failing test**

```tsx
// src/providers/__tests__/tickers.market.test.tsx
import { act, renderHook, waitFor } from "@testing-library/react-native";
import React from "react";

import * as api from "@/shared/api";
import { EG_FALLBACK } from "@/shared/markets";
import { MarketProvider, useMarket } from "../market";
import { TickersProvider, useTickers } from "../tickers";

jest.mock("@/auth/AuthProvider", () => ({
  useAuth: () => ({ isAuthenticated: true, user: { id: "u1" } }),
}));

const US = { ...EG_FALLBACK, code: "US" as const, currency: "USD" };
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <MarketProvider>
    <TickersProvider>{children}</TickersProvider>
  </MarketProvider>
);

beforeEach(() => {
  jest.spyOn(api, "fetchMarkets").mockResolvedValue({ markets: [EG_FALLBACK, US], default: "EG" });
  jest.spyOn(api, "fetchMarketPref").mockResolvedValue({ market: "EG" });
  jest.spyOn(api, "putMarketPref").mockResolvedValue({ market: "US" });
});

it("refetches tickers when the market changes", async () => {
  const spy = jest.spyOn(api, "fetchTickers").mockImplementation(async () =>
    api.getApiMarket() === "US" ? [{ symbol: "AAPL" } as never] : [{ symbol: "COMI" } as never]
  );
  const { result } = renderHook(() => ({ t: useTickers(), m: useMarket() }), { wrapper });
  await waitFor(() => expect(result.current.t.tickers[0]?.symbol).toBe("COMI"));
  act(() => result.current.m.setMarket("US"));
  await waitFor(() => expect(result.current.t.tickers[0]?.symbol).toBe("AAPL"));
  expect(spy).toHaveBeenCalledTimes(2);
});

it("discards a response from the previous market", async () => {
  let resolveEg: (v: never[]) => void = () => {};
  jest.spyOn(api, "fetchTickers").mockImplementation(() =>
    api.getApiMarket() === "EG"
      ? new Promise((r) => { resolveEg = r as never; })
      : Promise.resolve([{ symbol: "AAPL" } as never])
  );
  const { result } = renderHook(() => ({ t: useTickers(), m: useMarket() }), { wrapper });
  await waitFor(() => expect(result.current.m.loading).toBe(false));
  act(() => result.current.m.setMarket("US"));
  await waitFor(() => expect(result.current.t.tickers[0]?.symbol).toBe("AAPL"));
  await act(async () => resolveEg([{ symbol: "COMI" } as never]));
  expect(result.current.t.tickers[0]?.symbol).toBe("AAPL");
});
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `npx jest src/providers/__tests__/tickers.market.test.tsx`
Expected: the first test FAILS (`fetchTickers` is called only once).

- [ ] **Step 3: Implement the rule** in `tickers.tsx` (`[isAuthenticated]` becomes `[isAuthenticated, market.code]`), then in `watchlist.tsx` and every listed file. Run `npx jest` after each group of three files.

- [ ] **Step 4: Run the full mobile suite and the type-check**

Run: `npx jest && npx tsc --noEmit`
Expected: everything passes, matching the baseline plus the new tests.

- [ ] **Step 5: Commit**

```bash
git add src && git commit -m "feat(mobile): data refetches when the market changes"
```

---

### Task 15: The country changer (chip, sheet, account row)

**Files:**
- Create: `src/features/shell/MarketChip.tsx`, `src/features/shell/MarketSheet.tsx`, `src/features/shell/__tests__/MarketChip.test.tsx`
- Modify: `src/app/(tabs)/_layout.tsx` (`headerRight`), `src/features/account/AccountScreen.tsx`

**Interfaces:**
- Consumes: `useMarket` (Task 13); `Sheet`, `AppText`, `Pill` from `@/ui`.
- Produces:
  - `MarketChip()`, which renders `null` when `markets.length < 2`.
  - `MarketSheet({ visible, onClose })`.
  - `MarketRow()` (in `MarketSheet.tsx`), the account-screen row, which also renders `null` when `markets.length < 2`.

- [ ] **Step 1: Write the failing tests**

```tsx
// src/features/shell/__tests__/MarketChip.test.tsx
import { fireEvent, render, screen } from "@testing-library/react-native";
import React from "react";

import { EG_FALLBACK } from "@/shared/markets";
import { MarketChip } from "../MarketChip";

const US = { ...EG_FALLBACK, code: "US" as const, currency: "USD", name: "United States", flag: "🇺🇸" };
const setMarket = jest.fn();
let markets = [EG_FALLBACK];

jest.mock("@/providers/market", () => ({
  useMarket: () => ({ market: EG_FALLBACK, markets, setMarket, loading: false }),
}));

beforeEach(() => {
  setMarket.mockClear();
});

it("is hidden with a single market", () => {
  markets = [EG_FALLBACK];
  render(<MarketChip />);
  expect(screen.queryByLabelText(/Market:/)).toBeNull();
});

it("shows the current market and switches from the sheet", () => {
  markets = [EG_FALLBACK, US];
  render(<MarketChip />);
  fireEvent.press(screen.getByLabelText("Market: Egypt, EGP. Change market"));
  fireEvent.press(screen.getByLabelText("United States, USD"));
  expect(setMarket).toHaveBeenCalledWith("US");
});
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `npx jest src/features/shell/__tests__/MarketChip.test.tsx`
Expected: FAIL, the module is not found.

- [ ] **Step 3: Implement**

```tsx
// src/features/shell/MarketSheet.tsx
import { useState } from "react";
import { Pressable, View } from "react-native";

import { useMarket } from "@/providers/market";
import { AppText, Sheet } from "@/ui";

export function MarketSheet({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const { market, markets, setMarket } = useMarket();
  return (
    <Sheet visible={visible} onClose={onClose} title="Market">
      <View className="gap-2 pb-2">
        {markets.map((m) => {
          const active = m.code === market.code;
          return (
            <Pressable
              key={m.code}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}
              accessibilityLabel={`${m.name}, ${m.currency}`}
              onPress={() => {
                setMarket(m.code);
                onClose();
              }}
              className={`min-h-[44px] flex-row items-center gap-3 rounded-xl border px-3 ${
                active ? "border-accent bg-white/5" : "border-white/10"
              }`}
            >
              <AppText className="text-xl">{m.flag}</AppText>
              <AppText className="flex-1 font-sans-semibold text-white">{m.name}</AppText>
              <AppText className="font-mono text-xs text-white/50">{m.currency}</AppText>
            </Pressable>
          );
        })}
      </View>
    </Sheet>
  );
}

/** The account screen's "Market" row; hidden while only one market is enabled. */
export function MarketRow() {
  const { market, markets } = useMarket();
  const [open, setOpen] = useState(false);
  if (markets.length < 2) return null;
  return (
    <>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`Market: ${market.name}. Change market`}
        onPress={() => setOpen(true)}
        className="min-h-[44px] flex-row items-center justify-between"
      >
        <AppText className="text-white/70">Market</AppText>
        <AppText className="text-white">{`${market.flag} ${market.name}`}</AppText>
      </Pressable>
      <MarketSheet visible={open} onClose={() => setOpen(false)} />
    </>
  );
}
```

```tsx
// src/features/shell/MarketChip.tsx
/** Header chip (flag + currency) that opens the market sheet. Hidden with one market. */
import { useState } from "react";
import { Pressable } from "react-native";

import { useMarket } from "@/providers/market";
import { AppText } from "@/ui";
import { MarketSheet } from "./MarketSheet";

export function MarketChip() {
  const { market, markets } = useMarket();
  const [open, setOpen] = useState(false);
  if (markets.length < 2) return null;
  return (
    <>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`Market: ${market.name}, ${market.currency}. Change market`}
        onPress={() => setOpen(true)}
        hitSlop={6}
        className="min-h-[44px] flex-row items-center gap-1 rounded-full border border-white/15 px-3"
      >
        <AppText className="text-base">{market.flag}</AppText>
        <AppText className="font-mono text-xs text-white/80">{market.currency}</AppText>
        <AppText className="text-xs text-white/50">▾</AppText>
      </Pressable>
      <MarketSheet visible={open} onClose={() => setOpen(false)} />
    </>
  );
}
```

Check that `border-accent` exists in the mobile Tailwind config. If it doesn't, use the class the codebase uses for a selected state (look at `SegmentedControl.tsx`).

In `_layout.tsx`, put `<MarketChip />` first inside the `headerRight` row, before `<AlertsHeaderButton />`. In `AccountScreen.tsx`, render `<MarketRow />` in the settings group, next to the existing rows.

- [ ] **Step 4: Run the tests and type-check**

Run: `npx jest src/features/shell && npx tsc --noEmit`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src && git commit -m "feat(mobile): country changer in the header and account screen"
```

---

### Task 16: Market clock, money and labels on mobile

**Files:**
- Modify: `src/features/market/logic/marketClock.ts` (`isEgxOpen` is removed in favour of the shared `isMarketOpen`), `src/features/market/logic/__tests__/marketClock.test.ts`, `src/features/market/DashboardScreen.tsx:29,98`
- Modify the money literals in: `PositionCard`, `PortfolioSummary`, `RealizedSection`, `LotList`, `RiskDashboard`, `forms/AddHoldingForm`, `forms/SellHoldingForm`, `forms/AddDividendForm`, `features/portfolio/logic/display.ts:99`, `market/{KeyLevelsCard,EntryExitCard,StatsPanel,DividendHistoryCard,DividendCalendarScreen,OutcomeCone}.tsx`, `market/logic/levels.ts:35`, `charts/MonteCarloChart.tsx:69`
- **Not touched:** `learn/*` (EGX teaching content) and `MacroCard` (an FX pair name).

**Interfaces:**
- Consumes: `isMarketOpen` (shared markets); `formatMoney`, `formatPrice` (shared money); `useMarket` (Task 13).

- [ ] **Step 1: Update the clock test to use the shared function**

In `marketClock.test.ts`:
- Replace the `isEgxOpen` import and its `describe` block by importing `isMarketOpen` and `EG_FALLBACK` from `@/shared/markets`.
- Assert the same cases as `isMarketOpen(EG_FALLBACK, Date.parse(iso))`.
- Keep the "unresolvable timezone answers OPEN" case: `isMarketOpen({ ...EG_FALLBACK, timezone: "Not/AZone" }, Date.parse("2026-10-10T09:00:00Z"))` must be `true`.

Run: `npx jest src/features/market/logic`
Expected: PASS. The shared function already exists, and this proves it matches the old behaviour case for case.

- [ ] **Step 2: Remove `isEgxOpen`** from `marketClock.ts`, together with its docstring. In `DashboardScreen.tsx`, import `isMarketOpen` from `@/shared/markets`, read `const { market } = useMarket();`, and line 98 becomes `const marketClosed = !isMarketOpen(market, now);`.

- [ ] **Step 3: Swap the money literals**, using the same rule as web Task 11:
- The currency comes from the nearest response (`portfolio.currency`, `analysis.currency`), else from `useMarket().market.currency`.
- `` `${x.toFixed(2)} EGP` `` becomes `formatPrice(x, currency)`.
- `` `${egp(v)} EGP` `` becomes `formatMoney(v, currency, { digits: 0 })`, matching `egp()`'s default of zero digits; check `features/portfolio/logic/format.ts:11` first.
- Form labels use `` `(${currency})` ``.
- A plain `EGP` label `<AppText>` (`PortfolioSummary.tsx:51`, `DividendHistoryCard.tsx:78`, `DividendCalendarScreen.tsx:140`) renders `{currency}`.

Update the test fixtures that assert rendered strings only if their expected text changes. For EG it must not change: `formatPrice(42.4, "EGP")` is `"42.40 EGP"`, the same as before.

- [ ] **Step 4: Confirm only the intended literals remain, then run the tests**

```bash
grep -rn "EGP" src --include=*.tsx --include=*.ts | grep -v "__tests__\|src/shared/\|learn\|MacroCard\|testing/fixtures\|display.ts:6\|display.ts:8"
npx jest && npx tsc --noEmit && npm run check:shared
```
Expected: the grep shows only tooltip prose, which you list in the commit body. All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src && git commit -m "refactor(mobile): market-driven clock and money formatting"
```

---

### Task 17: Docs and the end-to-end check

**Files:**
- Modify: `egx-api-be/CLAUDE.md` and `egx-api-fe/CLAUDE.md`, which must stay byte-identical

- [ ] **Step 1: Add a "Markets" section** after "Architecture" in `egx-api-be/CLAUDE.md`:

```markdown
## Markets (EG today, US coming)

The app shows one market at a time. Every per-market fact — currency, timezone,
trading days and hours, holiday calendar, TradingView exchanges and scanner
region, benchmark, risk-free fallback, liquidity floors, the ISIN filter — lives
in `app/core/markets.py` and NOWHERE else. Add a fact there, not as a literal.

- `?market=EG|US` on every market-data route, resolved by the `market_param`
  dependency. **Absent means EG**, so old clients are unaffected. Unknown or
  not-enabled codes are 400.
- `ENABLED_MARKETS` (default `EG`) gates which codes the API accepts. US is fully
  wired but refused until sub-project 2 enables it.
- Symbol-keyed tables carry `market` and composite keys `(market, symbol, …)`;
  per-user tables carry `market` and filter on it. Existing rows are `EG`.
- `make_key(market_code, …)` refuses a missing market (`tests/test_cache_keys.py`).
- The benchmark is fetched only through `core/benchmark.get_benchmark_close`.
- Selected market = per-user PREFERENCE in `user_settings` (key `market`), via
  `GET/PUT /api/settings?section=market`. The market's FACTS are never per-user.
- Clients: shared `lib/markets.ts` (`isMarketOpen`, `EG_FALLBACK`), `lib/money.ts`
  (`formatMoney`), and `setApiMarket` in `lib/api.ts`, which appends `market=` to
  every data request. Mobile's `MarketProvider` owns the choice; the header chip
  and account row appear only when more than one market is enabled.
- Still EGX-only (later sub-projects): scoring calibration (P/E and yield bands,
  z-cone, regime, breadth), macro, Arab Finance news, Learn content, the web
  changer. Spec: `docs/superpowers/specs/2026-10-15-multi-market-foundation-design.md`.
```

Also update the stale lines this work contradicts:
- "**All P&L in EGP.**" becomes "**P&L is in the market's currency** (EGP for EG)."
- In the Database Schema block, add `market` to the listed tables and rename `turnover_egp` / `value_traded_egp` to `turnover` / `value_traded`.

Then copy the file: `cp egx-api-be/CLAUDE.md egx-api-fe/CLAUDE.md`, and check with `cmp` that the two are identical.

- [ ] **Step 2: Run the end-to-end check on mobile**
- Run the backend locally as in Task 9 with `ENABLED_MARKETS=EG,US`.
- Point the Expo app at it via `EXPO_PUBLIC_API_BASE_URL` and start it in tmux (`egx-api-mobile-dev`, log `~/logs/egx-api-mobile-dev.log`).
- Check, writing each result down:
  1. The chip shows 🇪🇬 EGP.
  2. Switching to US clears the dashboard and portfolio. Empty or error states are fine; US data comes in sub-project 2.
  3. Switching back restores EG exactly.
  4. The choice survives an app restart.
  5. With `ENABLED_MARKETS=EG`, the chip and the account row are gone.

Report any check that couldn't be done (for example, no simulator available). **Don't** report it as passed.

- [ ] **Step 3: Run the final full suites in tmux** (backend pytest, mobile `jest` + `tsc` + `check:shared`, web `lint` + `build`). Paste the summary lines into the hand-off message.

- [ ] **Step 4: Commit the docs**

```bash
cd /home/mark/projects/egx-api/egx-api-be && git add CLAUDE.md && git commit -m "docs: Markets section"
cd ../egx-api-fe && git add CLAUDE.md && git commit -m "docs: Markets section (mirror)"
```

- [ ] **Step 5: Hand off.** Don't push or deploy. Tell the user:
- the branch name in each repo
- the test summaries
- anything skipped
- the rollout order from spec section 6: Neon snapshot → backend with `ENABLED_MARKETS=EG` → web → mobile build

Pushing needs `gh auth switch --hostname github.com --user MarkBotros0 && git push`.
