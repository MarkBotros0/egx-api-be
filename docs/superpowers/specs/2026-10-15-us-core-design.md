# US core (sub-project 2 of US market support) — Design

**Date:** 2026-10-15
**Repos touched:** `egx-api-be`, `egx-api-fe` (shared `lib/` only), `egx-api-mobile`
**Builds on:** `2026-10-15-multi-market-foundation-design.md` (sub-project 1, merged)

## Intent

**What the user said**
- Purpose: track their own US holdings in a separate USD portfolio, benchmarked to the S&P 500.
- US dashboard tabs: **S&P 500 · NASDAQ-100 · Dow 30** (plus All).
- Composite score on US stocks: **show it, marked "Preliminary · EGX-calibrated"** until sub-project 3 recalibrates.
- Holdings and watchlist: **any NASDAQ / NYSE / AMEX stock or ETF**, not just index members.
- Mobile first. The web country changer is still out of scope.

**Assumptions**
- TradingView stays the only data source. Probes run on 2026-10-15 confirmed:
  - `america` scanner with `symbols.symbolset` returns index members, each with its exchange: `SYML:SP;SPX` gives 503 rows (345 NYSE, 157 NASDAQ, 1 CBOE), `SYML:NASDAQ;NDX` gives 101 and `SYML:DJ;DJI` gives 30.
  - `symbols.tickers: ["NASDAQ:X","NYSE:X","AMEX:X"]` returns only the listing that exists, in one call (BRK.B → NYSE), with `type` stock or fund (ETFs such as VOO and QQQ are `fund`).
  - A `description match` filter searches by company name ("coca" → KO, CCEP, COKE).
  - tvDatafeed daily bars load for NYSE (BRK.B, KO), NASDAQ, CBOE and `SP:SPX`.
  - **The wrong exchange fails slowly:** KO asked on NASDAQ waited 5.5s and returned nothing. Today every US fetch uses NASDAQ, so about 70% of the S&P 500 is broken.
  - `TVC:US03MY` (3-month T-bill yield) loads through tvDatafeed.

### Success criteria

1. With `ENABLED_MARKETS=EG,US`, a user can switch to US on mobile and see:
   - S&P 500, NASDAQ-100 and Dow 30 cards with prices and preliminary scores;
   - a working stock page for any NASDAQ/NYSE/AMEX stock or ETF;
   - a US watchlist;
   - a USD portfolio with live prices, a sale flow and the S&P 500 benchmark.
2. Every US symbol fetch uses that symbol's own exchange. No call site passes `market.primary_exchange` for a symbol.
3. US screens carry no EGP or EGX wording.
4. EG behaviour is unchanged, and the existing test suites pass.

## 1. US universe (`app/core/universe.py`, new)

**Table `market_universe`** (in `init_db`, idempotent):

| column | type | note |
|---|---|---|
| market | TEXT | PK part |
| symbol | TEXT | PK part, e.g. `BRK.B` |
| name | TEXT | scanner `description` |
| sector | TEXT | scanner `sector` |
| exchange | TEXT | `NASDAQ` / `NYSE` / `AMEX` / `CBOE` |
| kind | TEXT | `stock` or `fund` |
| indices | TEXT[] | e.g. `{SPX,NDX,DJI}`; `{}` for a searched-and-added symbol |
| updated_at | TIMESTAMP | |

**The `MarketConfig` registry gains:**
- `indices: tuple[IndexDef, ...]`, where an `IndexDef` holds `id`, `label` and `symbolset`.
  - EG lists EGX30, EGX70, EGX100 and NILEX, with no symbolset; they still come from `data/egx_tickers.json`.
  - US lists SPX "S&P 500", NDX "NASDAQ-100" and DJI "Dow 30".
- `short_name`: `"EGX"` or `"US"`, used for copy such as "US Market Overview".
- `score_calibrated: bool`: True for EG, False for US.

**Functions:**
- `refresh_universe(db, market)` (US only).
  - Makes one scanner call per index symbolset and upserts every row.
  - A symbol's `indices` is the set of symbolsets it appeared in.
  - Rows that dropped out of every index keep their row with `indices = {}`, because a holding may still reference them.
  - If any scanner call fails, nothing is written. A half-written universe must never replace a whole one.
- `load_universe(db, market) -> list[UniverseRow]`.
  - Reads from the table.
  - If the US table is empty, calls `refresh_universe` inline once (three calls, about 2s).
  - Results are cached in process for 1h.
- **Triggers:** a new cron endpoint `POST /api/cron/universe?market=US` (secret-guarded, daily), plus the inline cold-start above.

## 2. Exchange resolution (`app/core/universe.py`)

`resolve_exchange(db, market, symbol) -> str | None` is the single way to answer "where does this symbol trade?":
1. **EG:** always `"EGX"` (unchanged behaviour).
2. **US:**
   - in-process cache;
   - then `market_universe`;
   - then `discovered_tickers.exchange`;
   - then a scanner `tickers` lookup across `NASDAQ:S`, `NYSE:S` and `AMEX:S` in one call.
   - A hit from the scanner is persisted as a `market_universe` row with `indices = {}`.
   - A miss returns None and is cached for 1h, so a typo does not re-hit the scanner.

**Every symbol-level call site switches from `market.primary_exchange` to `resolve_exchange`:**
- `analysis` (single and batch); `ohlcv`; `historical`; `compare`; `intraday`
- `portfolio_analysis`; the cron snapshot; `news`
- An explicit `?exchange=` override on `ohlcv` and `analysis` still wins.
- If `resolve_exchange` returns None, these endpoints return 404 "Unknown symbol" instead of a slow empty fetch.

**Guard:** a test fails if `primary_exchange` is used anywhere except for benchmark or index symbols. It works like the AST guard on `make_key` from sub-project 1.

## 3. Search and validation (`routers/tickers.py`)

- **`GET /api/tickers?market=US`:**
  - Returns `load_universe` rows that are in at least one index, plus discovered rows.
  - Each row gets a new field `indices: string[]`.
  - The existing `index` field stays for EG compatibility. For US it is the first index, or the exchange.
  - `?index=SPX` filters on `indices`.
  - **This replaces the current US path** (unsorted 500-row scan, `[]` cached for 12h on failure).
- **`GET /api/tickers?validate=SYM&market=US`:**
  - Uses `resolve_exchange` and accepts kind stock or fund.
  - EG keeps the symbol-search path.
- **New `GET /api/tickers/search?q=…&market=US`** returns up to 20 rows, sorted by market cap. It combines:
  - an exact-ticker match;
  - a scanner `description match` on NASDAQ/NYSE/AMEX for stocks and funds.
  - EG answers it by filtering the loaded list, so clients can call it for either market.
- **`_search_exchange`** now tries every exchange even if one errors. This fixes the "stops at the first exchange error" follow-up.

## 4. Dashboard and snapshot cron

- **`cron._universe(US)`** returns the union of index members and every symbol in a US `portfolio` or `watchlist` row. That is about 520 symbols, measured stalest-first by the existing chunked `risk_snapshot?market=US`.
  - Each fetch uses `resolve_exchange`.
  - Benchmark: `SP:SPX`.
- **Fundamentals:**
  - `refresh_pe_data(db, US)` and `refresh_annual_fundamentals(db, US)` scan by `symbols.tickers` built from the universe, in chunks of 500, instead of the exchange filter.
  - The `/api/pe/refresh` route gains `?market=`.
  - `/api/pe` reports the feed status of the requested market. This fixes the follow-up where it reported EG's status for any market.
- **`/api/dashboard?market=US`** serves snapshot rows as EG does. The card shape is unchanged.
- **The dashboard `macro` block** (EGX30 trend, USDEGP) is `null` for non-EG markets.

## 5. Scoring inputs that must be per market now

US calibration is sub-project 3. These are the minimum changes that keep the preliminary US score from being plainly wrong:
- **Risk-free rate:** `read_risk_free_rate(db, market)`.
  - US reads `TVC:US03MY`, stored in `settings` as `risk_free_rate:US` and refreshed by the universe cron.
  - Fallback: `US.risk_free_fallback_pct = 4.0`.
  - EG is unchanged. The composite cache key already includes `rfr_tag`.
- **Index tier for liquidity floors:** a US symbol has no EGX tier. It uses the most lenient existing tier rather than defaulting to "EGX100" by accident, and this is pinned by a test. US floors arrive in sub-project 3.
- **Market regime:** `/api/market_regime?market=US` keeps its empty response, and mobile hides the Market Condition card when `band` is null for a non-EG market.

## 6. Portfolio, sales and watchlist

- **Add holding / add watchlist item (US):**
  - The symbol must resolve via `resolve_exchange`; otherwise the response is 422 "Unknown US symbol".
  - EG keeps its current lenient behaviour.
- **`portfolio_analysis` (market-aware throughout):**
  - Fetches use the resolved exchange.
  - The benchmark comes from `get_benchmark_close(market)` (already), and the `egx30_*` variables are renamed to `bench_*`.
  - Every user-facing message builds amounts with the market currency and names the benchmark with `benchmark_label`. That covers stop and target, support and resistance, ATR, SMA, price range, the P/E explanation and "cost basis excluded".
  - The `macro_egx30` signal is EG-only.
  - P/E prose that cites "EGX median ~12" is EG-only; US omits that sentence until sub-project 3.
- **Sales:** no change. A sale already takes its market from the holding row.

## 7. API contract additions

`GET /api/markets` gains these fields per market:
- `short_name`
- `score_calibrated`
- `indices: [{id, label}]`

Nothing else in the response shape changes, apart from:
- `indices` on ticker rows;
- the new `/api/tickers/search`;
- `macro: null` for US on the dashboard.

## 8. Clients

### Shared `lib/` (`egx-api-fe/src/app/lib`, synced to mobile)
- **Types:** `MarketInfo` gains `short_name`, `score_calibrated` and `indices`. `Ticker` gains `indices?: string[]`.
- **API:** `searchTickers(q, {market})`.
- **Filters:** `dashboardFilters` filters on `indices` when present, and on `index` otherwise.

### Mobile
- **Dashboard:**
  - Index tabs come from `market.indices` plus "All". The EG fallback stays `EGX30…NILEX`.
  - Title: `${short_name} Market Overview`.
- **Hidden for non-EG:** the Market Condition card when it has no band, the macro widgets, and NILEX/EGX copy.
- **Score display:** when `!market.score_calibrated`, every score badge (card, stock page, compare) gets a small "Preliminary" tag. Tapping it explains that the score is calibrated on EGX data.
- **Search:**
  - Dashboard filter: when the local filter finds fewer than 5 matches and the market is US, the list adds "More US stocks" results from `searchTickers` (debounced 300ms).
  - Add-holding form: the same remote suggestions. Validation already calls `?validate=`.
- **Stock page and watchlist:** no client changes beyond the above; the server resolves exchanges.

### Web
- Shared `lib/` changes only. The web app stays EG-only until the web changer.

## 9. Testing

**Backend (pytest):**
- Scanner calls are mocked with recorded payloads (SPX/NDX/DJI samples, the BRK.B lookup and a miss).
- `refresh_universe`: atomic on failure, membership sets, retention of index leavers.
- `resolve_exchange`: order of lookup, persisting a hit, caching a miss, and EG always "EGX".
- Exchange guard test.
- Tickers list, `?index=`, `?validate=` and `/search` for US; EG regression.
- Portfolio and watchlist add: 422 for an unknown US symbol; EG still lenient.
- `portfolio_analysis` for US: no "EGP"/"EGX" in any message (string scan of the response), and benchmark label "S&P 500".
- Risk-free rate per market; cron universe for US, including holdings and watchlist.

**Mobile (jest):**
- Tabs from the market.
- Preliminary tag shown or hidden.
- Remote search merge and debounce.
- Regime card hidden for US.

**Live smoke script:** `scripts/smoke_us.py` hits real TradingView for the universe refresh, resolves AAPL, BRK.B and VOO, and checks an SPX close. It is run manually and is not in CI.

**Emulator end to end**, against a local backend (`ENABLED_MARKETS=EG,US`):
- switch to US;
- the S&P 500 tab shows cards;
- open BRK.B;
- add AAPL, BRK.B and VOO holdings and confirm USD values and the S&P 500 benchmark;
- watch and unwatch a stock;
- sell a lot;
- switch back to EG and confirm it is unchanged.

## 10. Rollout

1. Deploy the backend as in sub-project 1. The new table is created by `init_db`.
2. Run `POST /api/cron/universe?market=US` once, then `/api/pe/refresh?market=US`.
3. Add cron-job.org jobs: universe daily, and `risk_snapshot?market=US` on the same cadence as EG.
4. Set `ENABLED_MARKETS=EG,US`.
5. Build mobile.

## Non-goals

- US calibration: score bands, forecast cone, US regime and breadth, liquidity floors (sub-project 3).
- US dividends, news, macro, alerts and New York-hours crons (sub-project 4).
- The web country changer.
- Mutual funds, OTC, options and crypto. ETFs on NASDAQ/NYSE/AMEX are in scope.
- Early closes and EG holiday verification.
