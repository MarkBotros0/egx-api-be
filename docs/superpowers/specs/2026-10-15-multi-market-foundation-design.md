# Multi-market foundation (sub-project 1 of US market support) — Design

**Date:** 2026-10-15
**Repos touched:** `egx-api-be`, `egx-api-fe` (shared `lib/` only), `egx-api-mobile`

## Intent

**What the user said**
- Add a country changer so the app supports US stocks alongside EGX.
- Purpose: track their own US holdings.
- Model: markets are separate. US mode shows only US stocks and a USD portfolio; EGX stays in EGP. There is no combined total.
- US universe: browse the S&P 500, and search for any other US ticker.
- Eventually every feature should work for the US market: portfolio and sales, the stock page, search, the watchlist, the composite score and analysis, dividends, news, macro and alerts.
- Mobile comes first.
- Identity model: option A, a `market` column with composite keys.

**Assumptions**
- Data stays on TradingView (tvDatafeed, the scanner and headlines), which covers NASDAQ, NYSE and AMEX. No new data provider is needed.
- A single user drives this, so there's no per-market admin, billing or permissions work.

### Decomposition

US support ships as four sub-projects. Each gets its own spec, plan and implementation:

1. **Market foundation (this spec).** The market dimension end to end, with EGX behaviour unchanged and the changer hidden.
2. **US core.** Enable `US`; the S&P 500 list and search; the stock page and fundamentals through the `america` scanner; the watchlist; portfolio and sales in USD, benchmarked to the S&P 500.
3. **US analysis.** Recalibrate the composite score and forecast for US: P/E and yield bands, the z-cone, regime and breadth on the S&P 500, and the US risk-free rate.
4. **US extras.** Dividends (no Egyptian withholding), US news, Fed and US macro, and alerts and crons on New York hours. The web changer comes in sub-project 2 or later.

### Success criteria for this sub-project

- With `ENABLED_MARKETS=EG`, the website and the mobile app behave exactly as before. Existing tests pass, and the changer is not visible.
- Every symbol-keyed row carries a `market`, and every market-specific endpoint accepts `?market=`.
- Nothing EGX-specific is hardcoded outside the market registry: hours, timezone, holidays, exchange, scanner region, benchmark, risk-free rate and currency all come from it.
- With `ENABLED_MARKETS=EG,US` on a preview deployment, the mobile changer appears. Switching persists per user and refetches the providers. US screens may be empty or error until sub-project 2.

## 1. Data model (`app/core/db.py`, `init_db`)

Changes are idempotent `ALTER … IF NOT EXISTS` statements, matching the current schema approach (there is no migration framework).

**Add `market TEXT NOT NULL DEFAULT 'EG'` to these tables and rekey them:**

| Table | Primary key after |
|---|---|
| `watchlist` | `(user_id, market, symbol)` |
| `discovered_tickers` | `(market, symbol)`, and add `exchange TEXT` |
| `pe_data` | `(market, symbol)` |
| `risk_snapshot` | `(market, symbol)` |
| `dividend_events` | `(market, symbol, ex_date)` |
| `fundamentals_annual` | `(market, symbol, fiscal_year)` |
| `score_baseline` | `(market, symbol, session_date)` |
| `alert_log` | unique key `(user_id, market, symbol, session_date, direction)` |

**Add the column and an index only:**
- `portfolio`, `portfolio_sales` and `portfolio_dividends` (index `(user_id, market)`).
- `fundamentals_history` and `market_regime` (index `(market, …)`).
- A position's currency comes from its market, so there is no currency column.

**Leave unchanged:**
- `macro_series` and `macro_data`: series codes are already country-unique.
- `users`, auth, `push_devices` and `alert_preferences`.

**Rename** `risk_snapshot.turnover_egp` to `turnover`, and `pe_data.value_traded_egp` to `value_traded`. Values are in the market's own currency. Both renames are guarded so they run only once.

**How the primary keys are swapped.** Each swap is a `DO $$ … $$` block that:
1. checks `pg_constraint` for the new key's column set;
2. if it's missing, drops the old primary key or unique constraint and adds the new one.

This makes the blocks safe on every cold start. Existing rows become `EG` through the column default.

**Selected market:** stored as `user_settings (user_id, 'market', 'EG'|'US')`.

## 2. Market registry (`app/core/markets.py`)

There is one frozen `MarketConfig` per market:

| Field | EG | US |
|---|---|---|
| `code`, `name`, `flag` | EG, Egypt, 🇪🇬 | US, United States, 🇺🇸 |
| `currency`, `locale` | EGP, `en-EG` | USD, `en-US` |
| `timezone` | `Africa/Cairo` | `America/New_York` |
| `trading_days`, `open`, `close` | Sun–Thu, 10:00, 14:30 | Mon–Fri, 09:30, 16:00 |
| `holiday_calendar` | `holidays.country_holidays('EG')` | `holidays.financial_holidays('NYSE')` |
| `exchanges` | `["EGX"]` | `["NASDAQ", "NYSE", "AMEX"]` |
| `scanner_region` | `egypt` | `america` |
| `benchmark`, `benchmark_label` | `EGX:EGX30`, "EGX30" | `SP:SPX`, "S&P 500" |
| `risk_free` | reads the admin-editable `settings.risk_free_rate` and the dated `EGINTR` steps, exactly as today | placeholder; source chosen in sub-project 3 |
| `is_valid_symbol(sym)` | the current ISIN filter from `cron.py:214` | accepts everything until sub-project 2 |

**Enabled markets.** These come from the env var `ENABLED_MARKETS` (default `EG`). `get_market(code)` raises `HTTPException(400)` for a code that is unknown or not enabled.

**Helpers:**
- `is_market_open(market, now)` replaces the separate checks in `core/cache.py`, the web `page.tsx` and mobile `marketClock.ts`.
- `market_holidays(market, days=365)`.

**Replacing hardcoded values with the EG entry** (values stay the same):
- `cache.py` hours
- `DEFAULT_RISK_FREE_RATE_PCT` in `constants.py` (it becomes EG's fallback when `settings.risk_free_rate` is unset; the admin-editable global setting stays the source)
- the alert timezone in `constants.py:265-267` and `score_alerts.py`
- the egxpy holiday calendar in `download.py:96`
- `exchange='EGX'` in egxpy (`download.py:103,135`), which becomes a parameter
- the EGX30 literals in `analysis.py`, `cron.py`, `portfolio_analysis.py`, `historical.py`, `compare.py`, `intraday.py` and `composite.py`
- the scanner URL and exchange filter in `core/tradingview.py`
- `LIQUID_MIN_*_EGP` in `risk_grade.py`: moves to per-market config, with only EG populated
- the `EGX:` prefix in news

**Out of scope here (left as EG-only, marked `# EG-calibrated` for sub-project 3):** the P/E and yield bands, `EGX_Z_CONE_60D`, and the regime and breadth thresholds.

**Endpoint:** `GET /api/markets`, which requires sign-in like every route (the app is closed). It returns the enabled markets with display fields: code, name, flag, currency, locale, timezone, days and hours, holidays for the next 365 days, and benchmark_label.

## 3. API contract

**`market_param` dependency.** It reads `?market=` (default `EG`) and returns a `MarketConfig` via `get_market`. Because the default is `EG`, old web and mobile builds are unaffected.

**Market-specific routers** take the dependency and pass `market` to core functions in place of the literals: tickers, ohlcv, analysis, dashboard, historical, compare, intraday, risk, market_regime, pe (and refresh), news, dividend_history, dividend_calendar, calibration, and `cron/risk_snapshot`. `ohlcv` and `analysis` currently take `exchange: Query("EGX")`. They keep that parameter as an override, with its default now coming from `market.exchanges[0]`.

**Per-user routers** (portfolio, sales, dividends, watchlist, portfolio_analysis):
- They filter by market.
- POST and PUT bodies take an optional `market` (default `EG`, validated).
- Updates and deletes by `id` don't change; the market comes from the row.
- `portfolio_analysis` benchmarks against `market.benchmark`.

**Responses** from all of the above include `"market"` and `"currency"` at the top level.

**Settings:** a new `GET/PUT /api/settings?section=market` reads and writes the caller's own `user_settings` row, mirroring `section=weights`. The generic `{key, value}` form must NOT be used: it writes the global `settings` table and is admin-only. PUT validates against the enabled markets, and GET defaults to `EG`. This is a preference about what to view; the per-market facts (currency, risk-free rate) stay market facts and never become per-user, as the guide requires.

**Cache:**
- All 14 `make_key(` call sites take the market code as their first argument.
- The benchmark key becomes `<market>:benchmark:…`.
- A test fails if any `make_key(` call in `app/` has no market argument. It works by an AST scan for `make_key` calls whose first argument isn't `market.code` or a `market`-named variable.

**Crons:** the URLs don't change, and the default market `EG` is implied. Per-market cron schedules come in sub-project 4.

## 4. Clients

### Shared `lib/` (source of truth is `egx-api-fe/src/app/lib`, copied by `sync:shared`)

- **New `markets.ts`:** the `MarketCode` and `Market` types, `EG_FALLBACK` (mirroring the registry's EG display fields), and `isMarketOpen(market, now)`, which is timezone-aware through `Intl`.
- **New `money.ts`:** `formatMoney(value, currency, opts?)` and `formatPrice(value, currency)`, both using `Intl.NumberFormat`. Add both new files to `SHARED_FILES` in `egx-api-mobile/scripts/sync-shared.mjs`.
- **`api.ts`:**
  - Module-level `setApiMarket(code)` and `getApiMarket()`. `fetchJSON` appends `market=<code>` to every request URL, except `/auth/*`, `/markets` and `/settings`.
  - New `fetchMarkets()`.
  - New `fetchMarketPref()` and `putMarketPref(code)` for `section=market`.
- **`types.ts`:** the affected response types gain `market: MarketCode` and `currency: string`.

### Mobile (`egx-api-mobile`)

**New `src/providers/market.tsx`, `MarketProvider`.** It sits first inside `AppProviders`, wrapping ScoreWeights, Tickers and Watchlist.

On boot it:
1. reads `market` from AsyncStorage and calls `setApiMarket`;
2. then fetches `/api/markets` and `/api/settings?section=market`. The server value wins. If it differs, it updates storage and context.

If `/api/markets` fails, it uses `EG_FALLBACK`.

The context exposes `{ market, markets, setMarket, loading }`. `setMarket(code)`:
1. calls `setApiMarket`;
2. writes AsyncStorage;
3. updates the context;
4. sends `PUT /api/settings?section=market`. If the PUT fails, the local choice stays and a toast is shown.

**Refetching on change:**
- `TickersProvider`, `WatchlistProvider` and the screen-level fetch effects (portfolio, stock, dividends, compare, alerts) add `market.code` to their dependencies.
- Responses whose `market` ≠ the current code are discarded.

**Changer UI:**
- `MarketChip` (flag, currency, ▾) in the tabs header opens a bottom sheet with the enabled markets.
- The account screen gets a "Market" row that opens the same sheet.
- Both render only when `markets.length > 1`.

**Removing EGX assumptions:**
- `features/market/logic/marketClock.ts` uses `isMarketOpen`.
- The 18 components with "EGP" literals use `formatMoney(…, currency)`.
- "EGX30" labels use `market.benchmark_label`.

**Deep links:** `stock/[symbol]` accepts `?market=`. If it differs from the current market, the screen fetches with that market without switching the global one.

### Web (`egx-api-fe`)

- Receives only the shared `lib/` changes and the `formatMoney` swap for its ~20 "EGP" literals, with the currency taken from responses.
- `page.tsx` uses `isMarketOpen(EG_FALLBACK)`.
- No changer or provider in this sub-project.

## 5. Testing

**Backend (pytest, existing `tests/`).** Pure logic is tested without a DB, per the repo convention. SQL tests are opt-in through `TEST_DATABASE_URL`, like `tests/test_accounts_pg.py`.
- **Registry:** `get_market` validation and the `ENABLED_MARKETS` parsing; `is_market_open` for both markets across weekends, holidays and DST boundaries, including New York DST.
- **Schema (PG, opt-in):** run `init_db` twice against the PG test DB and confirm it's idempotent. Then insert the same symbol under EG and US into `watchlist`, `pe_data` and `dividend_events`. Both rows should be accepted, and a duplicate within a market should be rejected.
- **Migration (PG, opt-in):** seed the old-shape tables, run `init_db`, and check that rows are preserved with `market='EG'` and the renamed columns hold the values.
- **API:**
  - Every market-specific route without `?market=` returns the same as before (snapshot against the current fakes).
  - `?market=XX` gives 400; `?market=US` gives 400 while it's disabled.
  - Portfolio and watchlist rows are isolated by market.
  - `PUT /api/settings?section=market` rejects a market that isn't enabled, and a non-admin can set their own market.
  - The `test_auth_gate` sweep still passes; `/api/markets` is NOT made public, since the app is closed by design and clients fetch it after sign-in.
- **Cache key guard test** (section 3).

**Shared/web (existing test setup):** `formatMoney` for EGP and USD; `isMarketOpen`.

**Mobile (jest):**
- `MarketProvider`: the boot order (storage, then server wins), the fallback when `/markets` fails, and `setMarket` persisting and calling the API.
- Providers refetch on a market change, and a stale response is discarded.
- `MarketChip` is hidden with one market and visible with two.
- `check:shared` passes.

**Manual:** on a preview deployment with `ENABLED_MARKETS=EG,US`, switch markets in the Expo app. Confirm the change persists across restart and across devices, and that EG data looks identical to production.

## 6. Rollout

1. Create a Neon branch snapshot before the first deploy.
2. **Backend first.** Deploy with `ENABLED_MARKETS=EG` and smoke-test the existing web and mobile builds; they must be unaffected.
3. **Web:** run `sync:shared` and ship the formatter swap.
4. **Mobile build** with the provider and changer, which stays hidden in production.
5. Sub-project 2 sets `ENABLED_MARKETS=EG,US` once US data works.

The `CLAUDE.md` in both `egx-api-be` and `egx-api-fe` is updated identically with a "Markets" section.

## Non-goals (this sub-project)

- Fetching any US data.
- US calibration.
- A web changer.
- Combined EGP/USD totals.
- Per-market crons.
- i18n.
- Rewriting the Learn curriculum's EGX-specific content.
- Fixing the known `egx_tickers.json` drift.
