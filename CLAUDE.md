# EGX Analytics — Project Guide for Claude

This file orients future Claude sessions to the codebase. Read it first before exploring.

> **This file is mirrored in both repos** — `egx-api-be/CLAUDE.md` and
> `egx-api-fe/CLAUDE.md` — because they are separate git repositories and a
> session usually has only one of them open. It documents the whole app, both
> halves, so **the two copies must be kept identical**. If you change one,
> change the other in the same breath. The working copy at the workspace root
> (`D:\Projects\egx-api\CLAUDE.md`) is outside both repos and is not tracked;
> treat the in-repo copies as canonical.

## What This App Is

An educational stock market analysis and portfolio tracker for the **Egyptian Exchange (EGX)**. The target user is a mobile-first Egyptian retail investor learning to analyze stocks. Every feature has an educational component (LearnTooltip, Learn page, "Learn more" links on advice).

**Key constraint:** The user primarily uses the app on phone — always design mobile-first.

## Architecture

**Next.js frontend + Python FastAPI backend, deployed to Vercel:**
- **Frontend:** Next.js 14 App Router, React 18, TypeScript, Tailwind CSS, Recharts (in `egx-api-fe/`)
- **Backend:** Python FastAPI app in `egx-api-be/` (Vercel Python runtime, 30s timeout)
- **Database:** Neon Postgres (serverless, accessed via `psycopg` + `psycopg_pool`)
- **Data source:** `egxpy`, **vendored** at `egx-api-be/app/vendor/egxpy` — `from app.vendor.egxpy import get_OHLCV_data, get_EGXdata, get_EGX_intraday_data`

  It is vendored because its upstream repo (`github.com/egxlytics/egxpy`) was deleted or made private — GitHub returns "Repository not found" even to an authenticated account that previously installed from it, and there is no PyPI release. Vercel builds failed at `uv lock` with `could not read Username`; deploys had been surviving on a warm build cache holding an already-cloned copy. There is no upstream to pull from, so **treat `app/vendor/egxpy` as our code** — edit it directly and note any deviation from v1.1.0 in its `__init__.py`. It is a thin wrapper over `tvDatafeed`, which is still public and installs from git normally. MIT licensed, © 2025 EGXLytics, LICENSE retained alongside the source.

Environment variables: `DATABASE_URL` (Neon connection string, includes
`sslmode=require`), `AUTH_SECRET`, `AUTH_USERS`, `AUTH_ADMINS`,
`PE_REFRESH_SECRET`, and **`CRON_SECRET`** (guards
`POST /api/cron/risk_snapshot`; scheduling is external, see that endpoint).

**Frontend environment variables** live in `egx-api-fe/.env`:
`NEXT_PUBLIC_API_BASE_URL` (the backend's `/api` base) and **`FE_BASE_URL`** —
the app's own public address, which `PasswordRevealDialog` pastes into the
credentials block so a new user is handed somewhere to go along with who they
are.

`FE_BASE_URL` is read by a CLIENT component, and Next ships only `NEXT_PUBLIC_*`
names to the browser by default. It is therefore listed in **`next.config.js`'s
`env` map**, which inlines it at build time under its own name — that map is
the only reason the variable can stay spelled `FE_BASE_URL` in `.env` and in the
Vercel dashboard rather than being renamed `NEXT_PUBLIC_*`. It holds a public
URL, not a secret. Verified inlined into the built admin chunk.

Unset, the dialog falls back to `window.location.origin`, resolved in an effect
so server and client markup match on first paint. **The link is never blank**,
so set the variable only where the origin the admin happens to be browsing
would be the wrong address to send.

## Directory Layout

The repo is split into two top-level directories — **`egx-api-be/`** (Python backend) and **`egx-api-fe/`** (Next.js frontend). CLAUDE.md previously described a flat `api/` layout; that is obsolete.

```
egx-api-be/                     # Python FastAPI backend
  app/
    main.py                     # FastAPI app entry; wires routers
    core/
      db.py                     # Neon Postgres connection + schema init (seeds weight_* keys + pe_* settings)
      cache.py                  # In-memory 5-min TTL cache
      composite.py              # Composite score engine (8 categories, macro modulation)
      indicators.py             # All technical indicators (pandas/numpy only)
      levels.py                 # Key levels (nearest support/resistance) + entry/exit zone computation
      macro_fetch.py            # Macro data fetch helper (EGX30, USD/EGP, CBE rate)
      news_fetch.py             # TradingView news: fetch, normalise, dedupe, 30-day window
      dividend_history.py       # Yahoo dividend history: parse, cadence, + dividend_events table upsert/read
      pe_fetch.py               # Fundamentals feed (P/E, dividend yield, loss-making, last ex-date) via TradingView scanner
      tradingview.py            # Shared TradingView scanner client (URL/headers), also used by tickers.py
      index_membership.py       # Static EGX30/70/100/NILEX lookup — no network, for the scoring hot path
      regime.py                 # Market-condition reading: bands + the evidence behind them
      forecast.py               # Calibrated outcome ranges (EGX-fitted quantiles, no drift)
      risk_grade.py             # EWMA vol, tradeability gate, cross-sectional risk grade
      constants.py              # Shared constants (thresholds, lookbacks)
      json_encoding.py          # Float/NaN JSON safety helpers
      holdings.py               # The one spelling of the open-holdings query (all, and per-symbol lots)
      returns.py                # Position-level return maths, shared open/closed
      sales.py                  # Realized-gains maths (pure)
      dividends.py              # Dividend ledger maths + the one spelling of its queries
    routers/
      analysis.py               # GET /api/analysis (single stock + batch mode)
      portfolio.py              # CRUD /api/portfolio
      sales.py                  # GET/POST/DELETE /api/sales (realized gains)
      dividends.py              # POST/DELETE /api/dividends
      portfolio_analysis.py     # GET/POST /api/portfolio_analysis
      pe.py                     # GET /api/pe ; POST /api/pe/refresh (cron-triggered)
      market_regime.py          # GET /api/market_regime (market-wide condition reading)
      risk.py                   # GET /api/risk (per-stock risk grade)
      cron.py                   # POST /api/cron/risk_snapshot (chunked; external scheduler)
      ohlcv.py                  # GET /api/ohlcv
      compare.py                # GET /api/compare
      historical.py             # GET /api/historical
      intraday.py               # GET /api/intraday
      tickers.py                # GET /api/tickers
      settings.py               # GET/PUT /api/settings  (weights live here too, section=weights)
      macro.py                  # GET /api/macro
      news.py                   # GET /api/news (on-demand, no table)
      dividend_history.py       # GET /api/dividend_history (Yahoo) ; /api/dividend_calendar (pe_data)
      watchlist.py              # GET/POST/DELETE /api/watchlist
      auth.py                   # POST /api/auth/login ; GET /api/auth/me
      users.py                  # Admin-only user management (/api/users)
    schemas/                    # (empty — Pydantic response shapes are inline dicts)

egx-api-fe/                     # Next.js frontend
  src/app/
    page.tsx                    # Dashboard
    stock/[symbol]/page.tsx     # Stock detail
    portfolio/page.tsx          # Portfolio tracker
    compare/page.tsx            # Multi-stock comparison
    dividends/page.tsx          # Dividend calendar — month GRID with year/month selectors, day -> payers list
    learn/page.tsx              # Educational content (has id anchors for learn_concept deep links)
    components/                 # React components
    lib/
      api.ts                    # Typed fetch wrappers
      types.ts                  # TypeScript interfaces
      constants.ts              # Frontend constants + fallback weight presets
      positions.ts              # Lots -> one position per symbol; FIFO sale preview

public/                         # Static assets, PWA manifest
```

**Important file-naming note:** core backend modules are `composite.py`, `db.py`, `indicators.py`, `macro_fetch.py` — no leading underscore. Older CLAUDE.md referenced `_composite.py` / `_db.py` etc.; those names no longer exist.

**Weights endpoint:** weights are NOT a separate `weights.py`. They are served via `GET/PUT /api/settings?section=weights` in `app/routers/settings.py`.

## Pages & Features

### Dashboard (`src/app/page.tsx`)
- **The only mobile route to Compare** — a `⇄ Compare` link beside the refresh
  button in the header, icon-only at a 44×44 target on mobile and icon+label at
  `md:`. Compare was removed from `BottomTabBar` to make room; the desktop top
  nav still links to it as well, so on desktop there are two ways in.
- Sticky search bar (mobile) with live symbol/name filtering
- Index filter pills (EGX30, EGX70, EGX100, NILEX) + sector filter pills, horizontal scroll on mobile
- Stock count display showing filtered results
- **Paginated grid** of `StockCard` components (24 per page, "Load More" with remaining count)
- Each `StockCard` has a **30-day sparkline mini chart** colored by trend (gain/loss)
- Price data per card is **lazy-loaded on scroll** (avoids hitting the API for off-screen cards)
- `StockCard` accepts optional `compositeScore`/`compositeSignal` props — renders a mini `CompositeGauge` badge when provided
- `StockCard` also shows a small **risk-grade dot** (Calm..Wild) under the score — the SAME grade `/api/risk` and `RiskGradeCard` show, stamped onto the dashboard rows by `card_snapshot.attach_risk_bands` via the one `grade_universe`, so a card's dot and that stock's detail page cannot disagree. Colour is the shared cool→hot ramp in `lib/riskBands.ts`, **never gain/loss** — a risk band is intensity, not direction. Shown only for a symbol tradeable enough to earn a band; thin/no-feed names get none. Independent of the composite-score toggle (a different axis: how much it moves, not its condition).
- Watchlist sidebar panel (uses `useWatchlist()` hook)

### Stock Detail (`src/app/stock/[symbol]/page.tsx`)
- Sticky header on mobile (hides on desktop) with Watch/Watching toggle
- Interval selector: Daily / Weekly / Monthly
- Bar count selector: 60 / 100 / 200 / 500
- Chart overlay toggles with color indicators: SMA 20, SMA 50, SMA 200, EMA 12, EMA 26, Bollinger Bands
- **Composite Score card** — `CompositeGauge` (lg) + `ScoreBreakdown` side by side; divergence/multi-timeframe/BB squeeze pill callouts below
- Price chart with support/resistance/Fibonacci ReferenceLines
- Volume chart, RSI/MACD/Stochastic/OBV tabbed indicators
- StatsPanel with Golden/Death Cross badge, Beta, ATR, RSI, volatility, 52W range
- Every overlay/indicator/stat has a LearnTooltip
- Re-fetches analysis when composite weights change (listens to `weightsVersion` from `ScoreWeightsProvider`)

### Portfolio (`src/app/portfolio/page.tsx`)

**One card per STOCK, not per purchase.** Buying the same symbol twice leaves
two `portfolio` rows, but the user owns one position, so `lib/positions.ts`
groups the analysed rows by symbol: total quantity, **cost-weighted** average
buy price (200 at 41.00 and 100 at 45.20 is 42.40, never the 43.10 a mean of
the prices gives), summed cost/value/P&L, and `pnl_pct` off total cost. The
technical half — score, RSI, zones, key levels, P/E — is carried straight from
the first priced lot: one symbol means one price fetch and one scoring pass, so
every lot already agrees. A single-lot position renders exactly as it always
has.

**The symbol line carries no badges.** It briefly held an `N lots` pill and a
`+X div` one; both were removed 2026-09-03 as title clutter, and neither took a
fact with it — the price already reads **`42.40 avg`** when more than one
purchase is behind it, the date line already reads **"First bought …"**, and
the dividend total and the purchases themselves are in the expanded detail.
Two pills were a second spelling, not a second fact. (`RealizedSection` keeps
its own `N lots` pill, which is a different job: it explains why one line in
the ledger covers 300 shares.)

**Annualized return is NOT aggregated.** The lots have different holding
periods, and averaging returns over different windows describes neither — the
same refusal `summarize_realized` makes. The tile reads "per lot" and the
figure is stated against each lot in the expanded card.

**Sell acts on the position, Edit and Delete on a purchase.** One Sell button
per card, max = total shares, spanning lots (see `POST /api/sales`); the form
previews the FIFO split and prices it **over the parts**, because
`(price − average) × qty` is only right when the sale consumes every lot —
selling 250 of that 200+100 position realizes 2,540 EGP at 52.00, not the
2,400 the average gives. Edit/Delete stay per lot inside the expanded row once
there is more than one.

- Add/Edit/Delete holdings with live re-analysis after each change
- **Target Price**, **Stop Loss** and **Notes** are still stored per holding and
  still drive distance calculations and stop-loss/target-hit signals, but
  **`AddHoldingForm` no longer exposes them** (2026-09-01 — the add/edit form is
  deliberately down to symbol, quantity, buy price, buy date). Existing values
  are carried through the edit payload unchanged so an edit cannot wipe them;
  nothing in the UI can set a new one today.
- **Sell** and **Dividend** actions per card → both land in `RealizedSection`.
- Mobile: FAB (floating action button) to add; full-screen modal form
- Desktop: inline form, top-right "+ Add Stock" button
- Re-runs analysis when composite weights change (listens to `weightsVersion`)

**Section order — what you hold comes first, then what you banked.**

```
PE banner → PortfolioSummary → HoldingsTable → RealizedSection
          → AdvicePanel → RiskDashboard → CorrelationHeatmap
          → MonteCarloChart → MacroCard
```

The holdings table sits directly under the summary that totals it; everything
below is either already banked or is analysis ABOUT those holdings. (Advice
therefore sits below the realized section rather than pinned to the rows it
describes — a deliberate call, and the one to revisit first if the order feels
wrong.)

**Realized is ONE section, not three cards.** The Winnings headline, Closed
Positions and Dividends Received used to stack as three separate disclosures
answering one question. `RealizedSection` makes the combined headline the
section header — so the figure worth seeing at a glance stays visible — with
the two ledgers plus a By-stock breakdown behind a tap, as TABS. Tabs rather
than stacked lists because the shapes differ: a sale has a buy price, a holding
period and a verdict against cash; a dividend has none of those, and stacking
them invites reading one set of columns as the other. Capital gains and
dividends stay separate figures throughout, summed only in the one headline
that says it is summing them.

### Compare (`src/app/compare/page.tsx`)
- 2–5 stock selection with autocomplete dropdown
- Customizable date range (start/end date inputs)
- Normalized performance chart (multi-series line, one per stock)
- Stats per stock: Total Return, Volatility, Max Drawdown
- Mobile: stat cards. Desktop: stat table.

### Learn (`src/app/learn/`)

A guided **learning path**, redesigned 2026-09-02 from the flat list of
`<details>` sections it used to be. Three files:

- `page.tsx` — a thin server component; owns the page `<title>` only
- `LearnClient.tsx` — the shell: hero, search, module nav, progress, hash router
- `curriculum.tsx` — **all content as data**, so the shell stays about layout

Plus `src/app/components/learn/`: `visuals.tsx` (the SVG diagram library),
`widgets.tsx` (the five calculators), `LiveChart.tsx`, `ConceptCard.tsx`.

**69 concepts in 9 ordered modules.** (`TOTAL_CONCEPTS` is derived from
`ALL_CONCEPTS.length`, so the code cannot drift from itself — but this
sentence can, and had: it read 68 while the page served 69. Do not restate
the count anywhere it could go stale; read the constant.) Order teaches, rather than grouping by
indicator family: Foundations → How the EGX Works → Reading a Chart → Signals
& Levels → The Composite Score → Taking a Decision → Managing Risk → Portfolio
Metrics → The Egyptian Context.

#### Anchors are a public contract — do not rename or drop one

`AdvicePanel` deep-links `/learn#<signal.learn_concept>` with ids chosen by the
BACKEND (23 of them, greppable from `app/routers/portfolio_analysis.py`), and
`ForecastCard`, `EntryExitCard`, `HoldingsTable` and `MarketRegimeCard` link to
fixed ones. The **nine module ids are the nine OLD section anchors**
(`market-basics`, `egx`, `technical-analysis`, `advanced-technical-indicators`,
`composite-score`, `decision-framework`, `risk-management`,
`portfolio-risk-metrics`, `macro-context`), kept unprefixed so bookmarks to
them still resolve. Module ids are hyphenated and concept ids use underscores,
so the two namespaces cannot collide.

Moving a concept between modules is free — the hash router finds it wherever it
lives, opens its module, scrolls to it and flashes it. **Renaming an id
silently breaks an in-app link**, which is the failure this layout is arranged
to avoid. 48 anchors were verified present after the redesign.

#### The four visual layers

1. **Static SVG diagrams** (`visuals.tsx`) — no charting dependency, so `/learn`
   pulls in **no Recharts** (41.7 kB route, 132 kB first load). `MiniChart` is
   the workhorse: a price path plus overlay lines, a volatility band, horizontal
   rails, point markers and vertical highlight regions. Also `ZoneScale`,
   `BarCompare`, `StepFlow`, `AllocationDonut`, `CorrelationGrid`, `ConeChart`,
   `LedgerRows`.
2. **Interactive calculators** (`widgets.tsx`) — RSI playground, stop-loss
   calculator, position sizer, T-bill race, score-band explorer.
3. **Live COMI data** (`LiveChart.tsx`) on 6 concepts.
4. **Worked EGP examples** — the `example` field on a `Concept`.

**Illustrative series are seeded, never random.** `walk(n, seed, …)` is a
deterministic LCG. `Math.random()` would hydrate-mismatch and flicker.

**Colour rule: `gain` green and `loss` red mean a real direction in the data,
never decoration.** Module identity is carried by a separate hue per module, so
a green line on the Learn page means what it means on the stock page. (Caught
once already: "You paid 41,000 EGP" was rendering loss-red. Paying is not a
loss.)

**`MiniChart`'s gradient id comes from `useId()`, not from the series.** It was
hashed from the data, and several charts deliberately reuse the same
illustrative series with different colours — so `url(#id)` resolved to the
first gradient on the page. Eight collisions on the first render.

#### LiveChart — one fetch, lazy, and it must degrade to the diagram

All instances share ONE module-level `/api/analysis` promise for **COMI**
(daily, 180 bars), fired by an `IntersectionObserver` so opening the page stays
as instant as it was when it was pure text. On any failure the caller's static
SVG renders instead and **no error is surfaced** — the page is service-worker
cached and read offline, and a teaching page should keep teaching. Variants:
`trend`, `bollinger`, `rsi`, `levels`, `volume`, `score`.

#### Two things that must not regress

- **Cards rest VISIBLE.** The scroll-reveal animation is additive: `.learn-card`
  is opaque by default and `.is-revealed` only adds a keyframe. If the
  `IntersectionObserver` never fires — old browser, blocked script, a hidden
  tab — the page is still a complete readable document. Verified: 0 reveals, 0
  dimmed cards. Never invert this into "hidden until JS shows it".
- **Widgets import their formula, never restate it.** The stop-loss calculator
  uses `STOP_LOSS_ATR_MULTIPLIER` and the band explorer uses `scoreBand()` /
  `SCORE_BAND_LABEL` from `lib/constants.ts`. A widget that teaches a different
  number from the one the app computes is worse than no widget.

#### The search is shaped like a command bar, and folds punctuation

`/` focuses it, Escape clears and leaves, and the live match count sits in the
field where feedback belongs. The `/` hint is `hidden md:block` — there is no
such key on a phone. The module pills sit ABOVE it: the modules are the order
the page is meant to be walked, search is the shortcut for when you already
know the term.

**`normalizeSearch` folds both the query and the text to alphanumerics**, and
that is a correctness fix rather than a nicety. The terms this page teaches are
written with punctuation a reader has no reason to reproduce, so a plain
substring match meant **"stop loss" returned NOTHING while "stop-loss" returned
nine** — and the old placeholder quietly worked around it by spelling the
hyphen out. "t bill", "p e" and "risk adjusted" now all find their concepts.

**The pill strip follows the reader.** Nine pills with about three visible on a
phone meant the active highlight kept landing off-screen. It scrolls the
strip's own `scrollLeft`, NOT `scrollIntoView` — that walks every scrollable
ancestor and would yank the page vertically mid-gesture, fighting the scroll
that triggered it.

A back-to-top button appears after one viewport, clears the nav via
`--bottom-nav-clearance`, and honours `prefers-reduced-motion`.

#### Progress

Per-concept read state in `localStorage` under `egx.learn.progress`, keyed by
`conceptKey()` (the anchor id, or a slug of the title). Read in an effect, not
during render, so server and client markup match on first paint. No backend, no
sync, per-device — it is a reading aid, and losing it costs nothing.

## PWA / Offline

- **Service worker:** `public/sw.js`, registered by `src/app/components/ServiceWorkerRegistrar.tsx`
- **Strategy:** network-first for `/api/*` and navigation; cache-first for static assets
- **Manifest:** `public/manifest.json` with maskable icons (192, 512) and SVG logo
- Standalone display mode (app-like install experience)
- iOS meta tags: `apple-web-app-capable`, status bar style, splash screens
- Safe-area insets via `env(safe-area-inset-*)` CSS variables

## API Endpoints

### GET /api/tickers
Returns list of all EGX tickers. Query: `index`, `sector`, `search` (case-insensitive substring match on symbol + name). Ticker list loaded once per container (module-level cache).

### GET /api/ohlcv
Raw OHLCV data. Query: `symbol`, `interval`, `bars`.

### GET /api/analysis
**The most important single-stock endpoint.** Returns OHLCV + all indicators + beta + support/resistance + fibonacci + MA crossovers + composite score.
- Fetches `max(bars, 400)` internally for SMA 200 accuracy, trims output
- Fetches EGX30 alongside for Beta computation (cached)
- Cache key includes a weight hash so scores invalidate automatically when weights change
- If interval is Daily, fetches weekly data and computes `multi_timeframe` alignment

Response shape (see `AnalysisResponse` in types.ts):
```
{ symbol, interval, bars, ohlcv, indicators, stats, beta,
  support_resistance, fibonacci, crossovers,
  composite_score, divergences, volume_price, multi_timeframe, bb_squeeze,
  key_levels, entry_exit, forecast }
```

- **`key_levels`** — `{current_price, nearest_support, nearest_resistance, room_to_support_pct, room_to_resistance_pct, clear_air_above, clear_air_below}`. Nearest levels are `{price, distance_pct (signed), strength, bars_ago}` or null. Computed by `levels.compute_key_levels` from `support_resistance`; pass `high`/`low` so `bars_ago` is populated. Consumed by `KeyLevelsCard`.

  **`clear_air_above` matters.** When a stock breaks to new highs there is genuinely nothing overhead, and the "nearest resistance" fallback returns a level it cleared long ago — on SWDY that was 27% BELOW the price, rendered in loss-red. The card now leads with "No resistance above" instead, and `portfolio_analysis` emits a `clear_air_above` opportunity signal. This is the same blind spot that made Max Buy Price reject every breakout; see *Removed: Max Buy Price*.

  **Level quality is visible.** `strength` 1 is a single pivot, not a tested floor — the card says "Touched once — weak level" and reserves "Tested Nx" for 2+. On live EGX data most detected supports are strength 1 and about two months old, which the user can now see. `bars_ago` renders as "~8 weeks ago" / "~6 months ago".

  **Known limit:** pivot detection needs ~20 bars either side, so **levels formed in the last month cannot be detected**. The Key Levels tooltip says so.
- **`entry_exit`** — `{entry_zone, exit_zone}`. Each zone: `{active, confidence, price_range, suggested_stop_loss (entry only), reasons}`. Confidence is `low`/`medium`/`high` or null. Entry zone active when price ≤5% above support AND RSI/Stoch not overbought. Exit zone active when price ≤3% below resistance AND RSI/Stoch overbought. Computed by `levels.compute_entry_exit`. Consumed by `EntryExitCard`.

- **`forecast`** — `{expected_move, outcome_band}`. See *Forecast calibration*
  below. Both are drift-free RANGES; neither carries a direction.

### Forecast calibration — the app used to advertise coverage it did not deliver

`core/forecast.py` states ranges, and until 2026-09-02 it stated them wrongly.
Measured over **34,749 point-in-time observations** from `scripts/.cache`
(regenerate with `python -m scripts.calibrate`):

| surface | advertised | actually delivered |
|---|---|---|
| ±1σ daily band | 68% | **79.0%** |
| ±1σ weekly band | 68% | 76.3% |
| ±1σ monthly band | 68% | 73.0% |
| Monte Carlo p5–p95 cone | 90% | **85.8%** |

**The two failures run in OPPOSITE directions, and that is the part that is easy
to get wrong.** At daily scale EGX's distribution is THIN in the body — price
limits and flat illiquid sessions pile mass near zero — so a ±1σ band is too
WIDE for its claim. At 60-day scale, compounding plus volatility clustering
fattens the aggregate, so the Gaussian cone is too NARROW. A blanket "EGX has fat
tails, widen everything" fix makes the daily band worse.

**Two fixes were tested and FAILED. Do not retry them:** an iid bootstrap of
daily returns gives 84.0% (no better than Gaussian — resampling destroys the
volatility clustering that creates the fat aggregate tail), and the empirical
distribution of trailing overlapping 60-day returns gives 78.9% (worse).

What works is EGX's own fitted |z| quantiles — at 60 days, **1.999 for 90%
coverage, not the Gaussian 1.645**. Measured coverage then lands on nominal.
**Fit with QUANTILES, never a standard deviation:** the z-distribution's mean is
in the hundreds against a median near zero, because a few collapsed names
dominate its moments.

**`monte_carlo_forecast` is gone**, replaced by `outcome_band`:
- It drew from `np.random.default_rng(None)`, so the p5/p95 prices on screen
  changed on every 15-minute cache miss. The band is now closed-form and cannot
  jitter.
- Its median was `P0·exp(60·(mu − σ²/2))` — the trailing 400-day mean return
  compounded forward — and `ForecastCard` rendered it to two decimals coloured
  green when above spot. **That is a price target with a direction attached**,
  the exact thing *Removed: Max Buy Price* exists to prevent. There is no median
  series any more, deliberately.
- `tests/test_forecast_presentation.py` fails if a directional colour, a
  percentile median, recommendation language, or a hardcoded coverage figure
  returns to the card. It strips comments before scanning, so the header comment
  that documents the removal does not trip the guard.

The portfolio-level Monte Carlo in `portfolio_analysis.py` is a DIFFERENT thing
and stays — it simulates a whole portfolio and is genuinely a simulation.

### Which sigma each band uses — and why they differ

`scripts/vol_backtest.py` is the gate every volatility change must pass. It
scores one-step-ahead conditional VARIANCE forecasts walk-forward by QLIKE, with
a Diebold-Mariano test (Newey-West) and a Kupiec coverage test, over the full
cached universe. **No change to how this app estimates volatility ships without
a run of it showing a win.**

Measured, full universe, QLIKE (lower is better):

| horizon | 400-bar trailing | EWMA(0.94) | EWMA(0.97) |
|---|---|---|---|
| 5 days | −4.0911 | −4.1789 | **−4.2186** (+3.1%) |
| 22 days | −2.5848 | −2.5599 | **−2.6510** (+2.6%) |
| 60 days | **−1.5543** | −1.4503 *(loses)* | −1.5574 (+0.2%, a tie) |

So the two surfaces use DIFFERENT estimators, deliberately:

- **`expected_move`** (daily / weekly / monthly tiles) → **EWMA(0.97)**, which
  clears the project's |t| > 3.0 bar at one day and wins at 5 and 22.
- **`outcome_band`** (60-day cone) → **the 400-bar trailing window**. At 60 days
  the two tie, and refitting the cone on EWMA measured **83.3% coverage against
  the trailing window's 85.8% — worse**. EWMA is an IGARCH: its multi-step
  forecast is flat at today's level with no mean reversion, which is the wrong
  shape for a long band.

**Coverage figures belong to the estimator they were fitted on**, so
`ONE_SIGMA_COVERAGE_PCT` is TWO tables — EWMA 78.6 / 75.5 / 71.8, trailing
79.0 / 76.3 / 73.0. `expected_move` falls back to the trailing window on a
history too short to seed the recursion, and reports that table instead.
Quoting one table for both would be the same class of error as the 68% it
replaced. `tests/test_forecast_presentation.py` pins both paths, and an AST walk
fails the build if `outcome_band` ever starts calling an EWMA without the cone
table being refitted.

**RiskMetrics' own lambda of 0.94 was REJECTED here.** It is the better one-day
forecaster but LOSES to the trailing window at 60 days. On a market where 19% of
daily returns are exactly zero, the shorter memory over-reacts to a run of flat
sessions followed by one real move.

**Two harness bugs worth remembering, both caught by the harness itself:**
- Without a variance floor, trailing-window QLIKE came back as **1.1e13** — flat
  EGX sessions drive `h` toward zero and `r²/h` explodes. Every method is now
  floored identically, because that is the only form any of them could be
  deployed in, and how often the floor binds is reported (trailing sd400: 10.5%
  of bars; EWMA: 0%).
- Pooling each method's losses independently and truncating to a common length
  pairs element *i* of one method with a DIFFERENT (symbol, bar) of another,
  which silently invalidates Diebold-Mariano — a test on a paired difference.
  The first draft reported the challenger winning while QLIKE ranked the
  incumbent ahead. All methods are now scored on the same bars, with an assert.

**Sample size mattered and nearly produced the wrong answer.** On 60 symbols the
challenger scored t = −2.28 and the harness said do not switch; on the full
universe the same comparison clears |t| > 3.0. Run the full universe before
concluding.

### GET /api/settings?section=weights, PUT /api/settings?section=weights
Composite score weights (stored as `weight_*` keys in the `settings` table). Handled in `app/routers/settings.py` — there is no separate `weights.py` (older docs called this `/api/weights`; it is not a real endpoint).
- GET returns `{ weights: { trend, momentum, volume, volatility, divergence, quality, risk_adjusted, relative_strength }, presets, default }`
- PUT accepts raw weights (any subset of the 8 keys), merges with current, normalizes to sum to 100, saves and returns normalized values
- Extending `CATEGORY_ORDER` in `composite.py` automatically extends what this endpoint accepts — the validation loop iterates `CATEGORY_ORDER`.

### GET /api/portfolio, POST/PUT/DELETE
CRUD for holdings. Stored in `portfolio` table in Turso.

### GET /api/sales, POST, DELETE

Realized gains ledger. `POST` records a full or partial sell: it inserts a
`portfolio_sales` row snapshotting the cost basis and decrements
`portfolio.quantity`, **both inside one `db.transaction()`** — `db.commit()` is
a no-op and each `execute()` takes its own autocommit connection, so without
the transaction a failure between the two statements would invent or lose
shares. The `quantity >= %s` guard lives in the UPDATE's WHERE clause, so two
rapid submits cannot both succeed.

**A sell is against the POSITION, and may span several purchase lots.**
`holding_id` names the position — the router reads its symbol and pulls every
open lot through `holdings.fetch_open_lots` — so `quantity` may exceed what
that one row holds: someone who bought 200 then 100 owns 300 shares and can
sell any number up to it. `sales.plan_sale_allocation` splits the count across
the lots **oldest first (FIFO)**, and each lot consumed writes **its own
`portfolio_sales` row**. The response is `{sales: [...], holdings: [...]}`,
plural because one submit can close parts of two purchases.

**One blended row was the alternative and it is wrong.** `compute_sale_metrics`
annualizes each trade over its own holding window and grades it against the
policy rate that prevailed across it, so a January basis blended with a June
one invents a purchase that never happened — the same refusal `summarize_realized`
makes when it declines to average annualized figures. Per-lot rows also keep
`DELETE /api/sales` restoring shares to the lot they came from, unchanged.

The date check follows the allocation, not the position: selling 150 of a
200+100 position touches only the older lot, so a sell date before the newer
lot's purchase is legitimate. Reaching into that lot rejects it, naming the
date. `validate_position_sale` and `plan_sale_allocation` are pure, so the
whole surface is tested without Postgres (`tests/test_sell_tracking.py`).

**One SUBMIT is one ORDER, and the ledger shows it that way.** Storing a row
per lot is right and the display that followed from it was not — selling 300
shares appeared as two closed positions, which is not what the user did. Every
row written together carries a **`sale_group_id`**, and `GET /api/sales`
returns `orders` beside the flat `sales`: same rows, folded up by
`sales.group_sale_orders`.

- **`summary` is still built from `sales`, never from `orders`.** Each trade is
  graded against the rate that prevailed over ITS window, which is precisely
  what an order spanning two purchases cannot state.
- **Money adds up; rates do not.** Quantity, cost, proceeds and P&L are sums
  and the percentage is cost-weighted, so those are exact. `days_held`,
  `annualized_return_pct`, `beat_t_bill` and `t_bill_hurdle_pct` are reported
  **only when every part ran over the same window** — otherwise null, with each
  part keeping its own. A single-lot order copies the row's figures rather than
  recomputing them, so it is the sale to the last decimal.
- **Grouping is on the STORED id, never on a `created_at` coincidence.** Rows
  written before the column existed are NULL and read as their own order
  (`COALESCE(sale_group_id, id)`). Nothing backfills it — how a sale was
  recorded is a fact, not something to infer afterwards.
- **`DELETE /api/sales?id=` accepts an order id OR a single row id** and
  removes every row it names, restoring each lot inside one transaction. That
  is deliberate rather than lax: the ledger shows one line per submit, so
  undoing what is on screen has to undo the whole thing — a delete that removed
  one part of a two-lot order would hand back half the shares and leave the
  line still sitting there. The confirm dialog says how many purchases it
  reaches. Addressing one part by its own id still works from the expanded row.

A full sell sets `quantity = 0` rather than deleting the row: the holding stays
as the anchor that makes `DELETE /api/sales` restore the position exactly, with
its target price, stop loss and notes intact.

Deliberately separate from `/api/portfolio_analysis` — realized gains need no
price fetch, so the Winnings card paints even when the analysis times out.

`summary` reports `beat_t_bill_count` of `annualizable_count`, **not** a
portfolio-level annualized return: annualized figures over trades of different
lengths cannot honestly be averaged. `total_realized_pnl_pct` is cost-weighted.
Trades held under 30 days report no annualized figure at all
(`MIN_DAYS_FOR_ANNUALIZATION` in `core/returns.py`).

**Each trade is graded against the rate that prevailed while IT ran**, not
today's. `compute_sale_metrics` takes `rate_steps` from
`macro_series.get_risk_free_steps` and `returns.annualized_cash_rate_pct`
compounds that step function across the holding window, annualizing back out —
so a flat rate returns itself exactly, and every other case is a deviation from
that anchor. The hurdle applied is returned as **`t_bill_hurdle_pct`**, because
a card cannot explain a verdict it cannot see.

**One scalar was wrong in BOTH directions, which is what made it more than
cosmetic.** The CBE has ranged 8.25%–27.25% over the period the app covers.
Measured against the real EGINTR series:

| trade | true hurdle | verdict | flat 19% said |
|---|---|---|---|
| +25% held through 2024 | **26.07%** | lost to cash | won |
| +18% held through 2019 | **14.69%** | beat cash | lost |

So `beat_t_bill_count` was era-dependent noise, not a consistent bias a reader
could mentally correct for. `rate_steps` is OPTIONAL and the scalar remains the
fallback, so a window the history does not reach degrades to the old behaviour
rather than failing — realized gains must paint even when everything else is
down. Reads happen ONCE per request, not once per trade;
`tests/test_dated_risk_free.py` has AST guards for both that and for the router
continuing to pass `rate_steps` at all.

`GET /api/sales` now also returns `dividends`, and `summary` is built by
`summarize_realized` (which replaced `summarize_sales`) — it owns both
ledgers so the combined headline is computed once, in tested Python, rather
than re-derived in the browser. `total_realized_pnl_pct`, `beat_t_bill_count`
/ `annualizable_count` and `best_trade` / `worst_trade` all stay
**capital-gains-only**: a dividend has no cost basis and no holding period to
annualize against, so folding it into those figures would either invent a
basis or silently change what "return" means mid-metric.

### POST /api/dividends, DELETE

Records cash a company paid the user for holding it — "profit share".

**A dividend is not a sale.** It reduces no position and closes no cost basis,
so it lives in `portfolio_dividends` and is anchored to the **symbol**, with no
`holding_id`. A sale carries one because undo must restore shares to a specific
position; a dividend restores nothing, so the column would buy no behaviour —
and would cost correctness, since deleting a holding would then destroy the
record of money genuinely received. Symbol-anchoring means dividend history
survives selling out entirely.

A single INSERT, so unlike `POST /api/sales` there is nothing for
`db.transaction()` to keep atomic.

`amount` is **the total EGP that actually landed**, already net of Egypt's
5–10% dividend withholding tax. The app computes no tax and must never present
this as gross. `shares` is optional and used only to display a per-share figure.

**Duplicate guard:** an exact `symbol + pay_date + amount` repeat returns **409**.
The primary surface is a phone, and a double-tapped submit is the likeliest way
this ledger goes wrong — a duplicate sale at least leaves a wrong share count,
a duplicate dividend leaves no trace at all.

**Reads are on `GET /api/sales`**, which serves both ledgers so the combined
headline is computed in tested Python rather than in the browser.

**The symbol anchor is now what the UI shows anyway.** Two `portfolio` rows
sharing a symbol render as ONE card, so the per-card dividend figure IS that
symbol's total and needs no caveat. (It used to read "(all lots)" because the
row was one lot of several. `dividends_symbol_shared` still travels on each
holding for any consumer that shows lots individually.) Splitting the amount by
today's share count would be fiction either way — the counts differed when the
dividend was paid. No aggregate is affected: every total sums the ledger
directly and never reaches it through holdings.

### GET /api/portfolio_analysis
**The heaviest endpoint.** Per-holding analysis + portfolio-level risk metrics + Monte Carlo + macro + signals.
- Fetches OHLCV for each holding sequentially (cache helps)
- Computes composite score per holding (divergence lookback = 30 bars; multi-timeframe skipped to stay within timeout; Risk-Adjusted requires ≥120-day history, else excluded)
- Applies macro modulation (EGX30 regime) to every per-holding score
- Risk: with 10+ holdings, can approach 30s timeout
- Returns `PortfolioAnalysisResponse` (see types.ts)

### GET /api/macro
Returns `{ egx30, usd_egp, interest_rate }`. 1-hour cache in `macro_data` table. Graceful degradation (returns null values if source fails).

### GET /api/news

Stories for the caller's holdings and watchlist, with EGX30 market news below.
**Fetched on demand — no table, no cron, nothing persisted.**

**The source is TradingView's news endpoint**, not the exchange.
`news-headlines.tradingview.com/v2/headlines?symbol=EGX:<SYM>` returns
Reuters/Zawya/LSE stories with `relatedSymbols`, so a story maps onto holdings.
egx.com.eg is unreachable from a server: it serves an F5 bot-challenge shell
(`APM_DO_NOT_TOUCH`), the same wall that made the old P/E scraper never once
succeed. **Do not re-propose scraping EGX from a serverless function.**

**Two traps, both measured:**
- **`?market=egypt` is SILENTLY IGNORED** and serves the global stock feed —
  200 items, **zero EGX symbols**, mostly Tesla and Santander. The market half
  is built by fanning out over EGX30 constituents and deduping.
- **The news host sends no `Access-Control-Allow-Origin`**, so a browser cannot
  fetch it. (The *scanner* host does reflect Origin.) This is why the work is
  server-side rather than a preference.

**Why no snapshot table.** `pe_data` and `risk_snapshot` are cached because
their upstream refuses half the universe at ~6s each. Measured here: **24
symbols in 1.30s** at 8 workers, median 0.37s, 0 failures — roughly 4x faster
than egxpy's healthy path. A table, a cron, a secret and a staleness story
would be machinery for a 1.3-second problem. **The revisit threshold is
`NEWS_DEADLINE_SECONDS` (8.0):** if it starts tripping routinely, the
conclusion has expired and `pe_data` is the template.

**A story older than `NEWS_RECENCY_DAYS` (30) is not news.** Chosen from the
data: across a 24-symbol sample the newest story ranged from 0 days old (ETEL)
to **275 days** (ACGC). A 7-day window empties the feed for most holdings; 90
days lets that 275-day-old item render as news. Filtered items are not hidden —
`coverage` reports them, the same convention as the dashboard's *"82 stocks ·
84 without a price feed"*. The same object also reports
**`symbols_over_cap`** — the caller's own symbols the 40-symbol fetch cap
excluded — so a large portfolio's missing names are visible rather than
silently dropped.

**`coverage` describes YOUR stocks only, never EGX30.** The user did not ask
for index names, and a "no news" tally against them would read as the app
failing rather than as an absence of news.

**Dedupe MERGES symbol tags rather than dropping the duplicate.** A story
arrives once per symbol fetched, and "Sodic Signs Medium-Term Facility With
CIB" is tagged `EGX:OCDI` **and** `EGX:COMI`. Keeping the first copy would
discard the fact that it concerns two holdings. Coverage is computed from
these deduped, merged-tag stories — a symbol "has news" because a story is
TAGGED with it, not because that symbol's own query happened to return it —
the same distinction `dedupe_symbol_signals` vs `build_position_signals` draws
in `portfolio_analysis`.

**Headline, provider, date, url, symbols — six fields, never body text.**
Stories are Reuters/Zawya/LSE copy; the app links out to TradingView and must
not store or reproduce them. `tests/test_news.py` fails if a body-ish field
survives `normalize_item`.

**Nothing on the page is `gain` green or `loss` red.** Those colours mean a
real direction in the data. A headline carries none, and no sentiment is
computed — tinting one green would claim a reading the app has not made.

Coverage is genuinely thin for small caps: of 24 sampled symbols, 2 had no news
at all (ESRS, EKHO) and several nothing inside 30 days (QNBE 56d, ACGC 275d).
The feature must read as honest when empty, not broken.

### GET /api/dividend_history, GET /api/dividend_calendar

Two dividend surfaces, both behind the auth gate (NOT in `PUBLIC_ENDPOINTS`).
Named `dividend_history` to avoid the existing `POST/DELETE /api/dividends`
LEDGER routes — those record the USER's own payouts; these describe the MARKET.

**`GET /api/dividend_history?symbol=XXX`** — one stock's dated, multi-year
dividend history plus a cadence estimate. **Persisted in the `dividend_events`
table** (PK `(symbol, ex_date)`), which is:
- **seeded deep from Yahoo** by `scripts/backfill_dividends.py` (all years —
  `query1.finance.yahoo.com/v8/finance/chart/<SYM>.CA?events=div`, the ONLY
  keyless source of DATED history — the scanner carries just the latest coupon,
  `fundamentals_annual.dps` has amounts but no ex-dates, egx.com.eg is bot-blocked);
- **grown forward for free by the nightly refresh** — `refresh_pe_data` appends
  the scanner's one latest coupon per symbol (`_append_dividend_events`), and the
  PK makes it idempotent so only a NEW (symbol, ex_date) lands. No extra fetch.

The endpoint **reads the table and self-heals**: on a table miss for a symbol it
fetches Yahoo once, upserts, and serves — so the store fills in as stocks are
viewed, ahead of the backfill. Verified: COMI 21 dividends to 2010, newest 6.00
on 7 Apr 2026 matching EGX's filing. Cached 6h
(`DIVIDEND_HISTORY_TTL_SECONDS`); a Yahoo hiccup on a cold symbol returns
`{dividends: [], status: "unavailable"}`, never a 500. `core/dividend_history.py`:
`parse_dividends` + `summarize_cadence` + `read_dividends`/`read_calendar` mapping
are pure (tested without network); `fetch_dividends` is the thin Yahoo GET,
`upsert_dividends` the idempotent write.

**`cadence` is descriptive of the PAST only.** `typical_month` is the mode of
past ex-dates, `payments_per_year` the modal count — never a forward promise.
There is no machine-readable EGX forward calendar, and the whole feature says so
wherever it estimates a "next" date.

**`GET /api/dividend_calendar`** — every payer's most-recent coupon for the
`/dividends` calendar view (a month GRID with year/month selectors; tapping a
day filters the list beneath to that day). Reads the `dividend_events` table
(`read_calendar`, latest-per-symbol), falling back to `pe_data`'s
`get_dividend_payers` before the table is seeded so it always paints.

**`pe_data` gained `dividend_ex_date_recent` (TEXT ISO) + `dividend_amount_recent`.**
The scanner's one dated coupon per stock, added to `TV_COLUMNS` via the
documented append pattern. ~34% coverage — which IS the dividend-payer
population, not a gap (contrast `earnings_release_next_date` at 10%, rejected as
a forward field that should cover everyone). Feeds the dashboard join, the
calendar, and the portfolio line below.

**Consumers:**
- **Stock page** — `DividendHistoryCard` self-fetches `/api/dividend_history`
  (RiskGradeCard pattern, off `AnalysisResponse` so a slow Yahoo call can't hold
  up the page). Dated list **paginated through ALL years** (Show more / Show less,
  `PAGE`=8) + estimated-next line. Links to `/learn#dividend_dates`. The stock
  page header also shows the **company name beside the ticker** (from
  `pe.company_name`, dropped when it just echoes the symbol).
- **Dashboard** — `/api/dashboard` joins `pe_data` (`dividend_yield` +
  `dividend_ex_date_recent`) so the grid has a **"Pays a dividend"** filter
  (`dividend_yield > 0`; 0 means pays nothing, real) and a **"Recently paid"**
  sort (newest ex-date first, non-payers sink). A `📅 Dividends` button in the
  header (Compare-button shape) opens `/dividends`.
- **Portfolio** — `portfolio_analysis` attaches `last_dividend_ex_date`/`_amount`
  per holding (off the `pe_info_h` it already fetches); `HoldingsTable`'s
  expanded row shows "Last market dividend", distinct from the user's recorded
  `dividends_collected`.

**No `gain`/`loss` colour on any dividend surface** — an amount and a yield
carry no direction, and a dividend paid is not a "loss".

### GET /api/settings, PUT /api/settings

Both routes require a token. `?section=weights` reads and writes the CALLER'S
own weights (see *Auth, roles and per-user settings*); the generic
`{key, value}` form writes the global `settings` table and requires an **admin**,
because everything left in there is shared — `currency`, `risk_free_rate` and
the `pe_last_*` feed status. Pre-seeded: `currency=EGP`, `risk_free_rate=19` (the CBE overnight deposit
rate as of 2026-08-20; it was a stale 25 until 2026-09-02 — see *Egypt context*).

### The app is CLOSED — read this before adding any route

**Nothing is served to anyone who is not signed in.** There is no public
surface: no landing page, no public dashboard, no anonymous API.

The backend enforces it in ONE place — `require_authentication`, an HTTP
middleware in `app/main.py` that rejects every `/api/*` request without a
valid, active user. The policy itself lives in `core/auth.py`:

```python
PUBLIC_ENDPOINTS = frozenset({
    ("POST", "/api/auth/login"),          # the way in
    ("POST", "/api/pe/refresh"),          # scheduled; guarded by PE_REFRESH_SECRET
    ("POST", "/api/cron/risk_snapshot"),  # scheduled; guarded by CRON_SECRET
})
```

`OPTIONS` also passes, because a CORS preflight carries no `Authorization`
header by design and blocking it breaks every browser call including login.

**Default-deny is the whole point.** A router added later is locked until
someone deliberately opens it, so the failure mode is "it 401s and I notice"
rather than "it has been serving the dataset to the internet since it
shipped" — which is exactly what **nine** endpoints (`tickers`, `ohlcv`,
`analysis`, `compare`, `historical`, `intraday`, `macro`, `market_regime`,
`pe`) were doing until 2026-09-02, with nothing in the codebase noticing.
`tests/test_auth_gate.py` walks the app's REAL route table and fails if any
non-allowlisted route answers an anonymous caller, so the guarantee is
verified by enumeration rather than by memory.

Routers that need the caller's identity still declare
`Depends(get_current_user)`. The middleware is a second layer, not a
replacement — do not remove those.

**Frontend:** `src/middleware.ts` is deny-by-default too. Only `/login` and
static assets (`/_next`, `/icons`, `/manifest.json`, `/sw.js`, favicons) are
reachable signed out; everything else redirects to `/login?next=…`. `Navbar`
and `BottomTabBar` return `null` when unauthenticated so the login page stands
alone. That redirect is UX only — the `egx.auth.present` cookie is set by
client JS and carries no signature, so the backend gate is the real guard.

**Logout wipes Cache Storage** (`clearStoredAuth` → `clearCachedResponses`).
`sw.js` is network-first but FALLS BACK to cache, so without the wipe a
signed-out person on a shared phone could go offline and have the worker
re-serve the last dashboard and API responses it saw. Verified: a session
holding 15 cached entries including `/api/portfolio` drops to just the
re-cached `/login` page.

**Consequence for `/api/market_regime`:** its cached scores used to be warmed
by anonymous dashboard traffic. There is no anonymous traffic any more, so the
reading stays warm only while the signed-in user is on DEFAULT weights (same
`weights_hash`, same cache key). A user with customised sliders will see the
card fall back to its last stored reading flagged `stale`. It degrades, it does
not break.

### Auth, roles and user management

**The app is multi-user with JWT auth.** Login is `POST /api/auth/login`
(bcrypt over a SHA-256 pre-hash, HS256 token, 30-day life); `GET /api/auth/me`
returns `{id, username, role}`. There is no registration endpoint — accounts
come from an admin or from the env seed.

Two roles, `user` and `admin`, on `users.role`.

**Admin status comes ONLY from the `AUTH_ADMINS` env var**, applied at boot by
`core/auth.seed_users_from_env` and re-asserted on every boot, so it survives a
DB reset and cannot be fumbled in the database. No API route writes a role —
`tests/test_users_and_roles.py::test_the_admin_api_cannot_grant_admin` greps for
`SET role` and fails if one appears. That makes privilege escalation through
`/api/users` structurally impossible, and it is why the admin tab has no
role picker. Demotion of unlisted users happens **only when `AUTH_ADMINS` is
non-empty** — a blank var must never strip every admin and leave the app
unmanageable.

**`AUTH_USERS=alice:pw1,bob:pw2` only CREATES users that don't exist.** It no
longer refreshes password hashes on boot. It used to, and that would silently
revert every admin password reset on the next cold start.

**Role and active-state are read from the DB on every request, never from the
token** (`core/auth.get_current_user` → `_load_user`). A 30-day token means
claims would let a disabled user keep working for a month, so "disable" would
be a lie. Cost is one indexed PK lookup on a pooled connection.

Dependencies in `core/auth.py`:
- `get_current_user` — 401 unless a valid token maps to an **active** row
- `require_admin` — the above plus 403 for non-admins
- `get_optional_user` — returns `None` instead of raising. **Currently unused:**
  `/api/analysis` used it while the dashboard was public, and now requires a
  real user like everything else. Kept because it is the right tool if a route
  ever legitimately needs to serve both. Do not reach for it to sidestep the
  gate — the gate would reject the request before the dependency ran anyway.

Note the **anonymous WEIGHTS context is a different thing** and still exists:
`get_weights_from_db(db, user_id=None)` and `read_cached_scores` pass `None` so
the market-regime average stays on the default weights its bands were
calibrated at. That is about whose sliders apply, not about who may call.

### /api/users — admin only

`GET` list · `POST` create · `POST /{id}/password` reset · `PATCH /{id}`
enable/disable · `DELETE /{id}`.

- A generated password (`core/auth.generate_password`, 16 chars, no `0O1lI`) is
  returned **exactly once**, in the response to the call that created it. Only
  the bcrypt hash is stored, so it can never be read back — `PasswordRevealDialog`
  says so on screen.
- **`DELETE` removes `portfolio_sales`, `portfolio_dividends`, `portfolio`,
  `watchlist` and `user_settings` first, then the user, all in one
  `db.transaction()`.** No table has an FK to `users`, so nothing cascades and
  the rows would otherwise survive as invisible orphans. Sales and dividends go
  before holdings, matching the direction the sale-undo path depends on.
- Guards: you cannot disable or delete **yourself**, or the **last active
  admin** — either would leave the app with nobody able to administer it.

### GET /api/watchlist, POST, DELETE
User's watched symbols, stored in the `watchlist` table in Turso.
- GET returns `{ symbols: string[] }` ordered by `added_at`
- POST body `{ "symbol": "..." }` — idempotent (INSERT OR IGNORE)
- DELETE `?symbol=XXX`

### GET /api/pe, POST /api/pe/refresh
Fundamentals per stock (trailing P/E, dividend yield, loss-making), served from
`pe_data`. Populated by a nightly job on **cron-job.org** (`0 4 * * *`) that calls
`POST /api/pe/refresh` with an `X-Refresh-Secret` header. **`vercel.json` no
longer defines any cron** — both scheduled jobs are external now, which
removes the Hobby 2-cron limit entirely. Note Vercel's own cron never sent
that header, so in production the guard was either unset or silently 403ing.
- `GET /api/pe` returns `{ data: [...], last_successful_fetch, last_attempt_status }`
- `GET /api/pe?symbol=XXX` returns the single row or 404
- `POST /api/pe/refresh` makes ONE POST to the TradingView scanner via
  `core/tradingview.scan`, and upserts. **Never wipes existing rows on failure** —
  last-known-good is preserved.
- `PE_REFRESH_SECRET` env var guards manual invocation in production.

**The source is no longer egx.com.eg.** That page is behind a JavaScript bot
challenge: the request returns HTTP 200 with ~6 KB reading *"Please enable
JavaScript to view the page content"*, so the GridView parser matched zero rows.
It had **never once succeeded in production** — `pe_data` was empty and
`settings.pe_last_attempt_status` was blank from the day the feature shipped
until 2026-08-25, which means `score_quality`'s P/E band had never executed
against real data. The whole name→symbol matching apparatus
(`match_symbol`, the overrides JSON, jaccard scoring) is **deleted**: the
scanner returns bare symbols, so there is nothing to resolve.

`core/tradingview.py` holds the scanner URL/headers, shared with
`routers/tickers.py::_fetch_live_tickers` so the two cannot drift.

**Null semantics — these changed with the source, and two of them invert:**

| Field | Meaning |
|-------|---------|
| `pe_ratio IS NULL` | no trailing P/E — usually a loss-maker or newly listed |
| `dividend_yield = 0.0` | **real data: the company pays nothing.** Only NULL means unknown — do NOT treat 0 as a missing-data sentinel the way the old EGX feed did |
| negative `pe_ratio` | **never occurs.** The feed returns NULL for loss-makers, so `loss_making` (from diluted EPS < 0) carries that instead |

**Coverage is partial and that is expected:** of ~293 EGX stocks, ~64 have a
trailing P/E and ~92 a dividend yield. A missing value SKIPS its band rather
than defaulting, so a stock is never punished for the feed's silence — and the
bands are centred on the EGX median so having data isn't itself an advantage
(see *Fundamentals bands*).

Guards in `refresh_pe_data`: a response under `MIN_EXPECTED_ROWS` (100) is
rejected **without writing anything** — a partial refresh that updates 50
symbols and leaves 240 stale is worse than all-stale, because nothing on screen
distinguishes them. Rows with no P/E, DY or EPS are skipped so
`get_pe_for_symbol` can't return a truthy all-null dict. `pe_ratio > 300` is
dropped at ingest (the live max is ~2756, which would render as "P/E 2756.0").

### GET /api/market_regime

**The only forecast-shaped surface in the app, and the one thing the backtest
supported.** Returns the average composite across the EGX30+EGX70 constituents,
its band, and the historical record behind that band.

The per-stock score cannot rank stocks (IC ≈ 0). The market-wide AVERAGE of
those same scores carries a WEAK association with the market itself — rank
correlation **+0.162 with the EGX30's next 63 trading days, Newey-West t = 1.74
across 221 overlapping readings**. That does not clear 1.96, so the card is
CONTEXT, not a forecast.

**Re-derived 2026-09-02 on the dated-risk-free panel** (see *The backtest's
Risk-Adjusted verdict*), which moved it from +0.17 / t=1.86. The signal got
marginally WEAKER, and the conclusion is unchanged.

**CORRECTED 2026-09-02 — this used to claim +0.318, t=2.84, 74 non-overlapping
periods.** It defended that with "de-overlapping made the correlation stronger,
so it is not an overlap artifact." At a 63-day horizon on a 21-day rebalance
grid there are exactly THREE valid de-overlapped samplings, and they give
**+0.309, +0.167 and −0.006**. De-overlapping did not VALIDATE the number, it
RESAMPLED it, and the best of three draws was read as a robustness check.
Starting the same grid one month later would have produced a card claiming
nothing — the third phase is now slightly NEGATIVE.

The honest statistic keeps all 221 observations and corrects the standard error
for the overlap (Newey-West, lag = overlap depth). Regenerate every number here
with `python -m scripts.calibrate`, which prints all three phases side by side
so the cherry-pick cannot silently return. 21-day and 126-day horizons were also
tested and neither was significant — do not relabel the horizon.

Bands (terciles of 221 readings, 2007–2026), against the next three months of
EGX30:

| band | reading | median | 3m positive |
|---|---|---|---|
| weak | < 45.4 | −0.03% | 50.0% |
| mixed | 45.4–51.9 | +5.42% | 68.5% |
| broad | ≥ 51.9 | +6.72% | 68.9% |

Read as **weak-versus-not**, not a dial: the top two bands are not meaningfully
different — 68.5% against 68.9%, and the weak band is now a literal coin flip. And the EGX rose substantially in EGP terms over the window, so
"weak" means flat, not falling.

**It never fetches.** Scoring the 79-symbol universe on demand does not finish
inside a serverless request — measured at >400 s cold, because each symbol
pulls 400 bars through a client that retries hard on socket timeouts. Instead
it reads scores the dashboard batch path has already cached, via
`analysis.read_cached_scores`, and reports coverage in `n_symbols`. Below
`MIN_SYMBOLS_FOR_REGIME` (15) it refuses to classify and serves the last stored
reading flagged `stale` rather than averaging a handful into a confident number.

**The cache key is spelled once**, in `analysis.composite_cache_key` /
`scoring_cache_context`, because the batch path WRITES those entries and this
endpoint READS them. Two independent spellings would mean zero hits forever and
a permanent, silent "not enough data".
`tests/test_fixes.py::test_regime_reader_and_batch_writer_share_one_cache_key`
pins it.

### GET /api/dashboard — the whole grid, from one query

**The dashboard no longer computes anything on demand.** It reads the
pre-scored snapshot and blends it with the caller's weights: one Postgres
round trip, no upstream fetch, no deadline, the whole universe or nothing.

Measured live: **166 rows, 82 with a score, 9.8 KB gzipped** (63 KB raw),
query 358 ms from a laptop and 0.0 ms to blend all 166.

**What it replaced, and why tuning could never have fixed it.** Each card
needed a live 400-bar pull through a client wrapping every call in
`@retry(tries=20)`; the feed serves 82 of 166 symbols at ~1.4 s and refuses the
other 84 at ~6 s each. Twenty-four cards do not fit a serverless request, so
`page.tsx` fanned the work out at `DASHBOARD_FETCH_CHUNK_SIZE = 2` — **twelve
simultaneous requests**, which Vercel answered from twelve separate containers.
Twelve cold starts, twelve duplicate EGX30 benchmark fetches, twelve private
module-level caches none of which the next request could reuse. Whether a card
painted came down to which container answered and whether it was warm. That is
the whole of "sometimes it loads, sometimes it doesn't."

Three defects compounded it, all now fixed:

- **`_cache_on_done` cached successes only**, so each of the 84 refused symbols
  re-paid its ~6 s refusal on every load, forever. Failures now cache under
  `ERROR_CACHE_TTL_SECONDS` (120 s) — short, because a refusal is far more
  perishable than a close.
- **A card that missed twice stayed blank permanently.** The fetch effect's
  deps omitted the data maps, so a failed chunk never re-triggered it, and
  `retriedRef` allowed exactly one retry then blocked for ever. Now bounded by
  `COMPOSITE_MAX_ATTEMPTS`, and every request carries an `AbortController`
  timeout — an unsettled request used to pin its symbols in the in-flight set
  for good.
- **`--` meant three things at once** — loading, refused, and never-existed.
  `StockCard` now has five distinct states (`loading` / `live` / `stale` /
  `unavailable` / `failed`).

**The Refresh button did nothing, twice over, and both causes are worth
keeping.** It fired zero requests:

- **An effect-ordering hazard.** The reset effect only SCHEDULES the demotion
  of live cards back to stale, and the upgrade effect runs in the SAME commit
  right after it — so `upgrade` still saw every symbol as `live`, filtered them
  all out, and returned. Its deps had already changed, so it never ran again.
  "Already upgraded" now lives in `upgradedRef`, a ref, because refs mutate
  synchronously and state does not. Never decide *have I done this* from state
  a sibling effect has only just queued.
- **The service worker was answering the refresh from its own cache.**
  `/api/dashboard` is served stale-while-revalidate, which is right on a normal
  visit and exactly wrong when the user has just asked for new numbers. An
  explicit refresh now carries `?fresh=<ts>`, which `sw.js` refuses to
  SWR-serve *or* cache, so it reaches the network and leaves no entry behind.

Verified: one `\/api\/dashboard?fresh=…` plus 2 batch calls at concurrency 2,
against 0 requests before.

**The live upgrade is deliberately expensive.** It pulls the full 400 bars and
re-runs all eight categories, so a card that upgrades carries a score computed
on today's bar — the same number the detail page computes. Measured, 9 of 24
finish; the other 15 keep their snapshot value and their `as of` stamp. A
price-only upgrade would be far faster and was rejected: the card's score would
then be last night's while the detail page rescored on today's bar, which is
the card-vs-detail divergence this whole design exists to prevent.

### THE EIGHT CATEGORY SCORES ARE STORED, NOT THE COMPOSITE

This is what keeps *One Score Per Stock* true rather than trading it away for
speed. Weighting, renormalisation and macro modulation are a pure function of
the eight category scores, the caller's weights and today's regime. That
function is **`composite.blend_categories`**, and both `compute_composite` and
the snapshot reader call it — so a card shows this user's sliders applied to
the same inputs the detail page uses, and the two cannot disagree.

`composite.score_categories` is the other half of the split: it takes
**neither weights nor macro**, which is what makes a nightly job able to serve
every user their own number.
`tests/test_dashboard_snapshot.py::test_snapshot_blend_matches_live_scoring_exactly`
asserts exact equality across every preset, a hand-set slider arrangement and
all four macro regimes — exact, not approximate, because a tolerance there
would be an admission that two implementations exist. Verified live on ACGC:
69.2 beginner_safe, 67.5 balanced, 74.5 trend_follower, 58.9 reversal_hunter,
71.0 income_defensive, 66.3 under a bearish macro.

Storing a blended number instead would freeze one weight set into the card and
reintroduce the divergence measured once at **66 "Buy" on the card against 45
"Hold" on the detail page**.

**Presentation rules:**
- `available: false` (from `consecutive_failures >= FAILURE_DEMOTION_THRESHOLD`)
  moves a stock into a collapsed **"No feed data (N)"** section rather than
  hiding it. They are real listings, and the count line stays truthful:
  *"82 stocks · 84 without a price feed"*. Membership is a counter, not a
  blocklist, so a symbol that starts working returns to the grid on its own.
- `oldest_measurement` is the STALEST row, not the newest — same convention as
  `/api/risk`. A grid that reported its freshest row would overstate its own
  currency every time.
- The snapshot is written after the close, so during trading hours its price is
  a previous close and the card says **"as of 2 Sep"**. It then upgrades the
  VISIBLE page to live figures in the background at
  `COMPOSITE_BATCH_CONCURRENCY = 2` — bounded so the requests land on one warm
  container instead of a dozen cold ones. **A card that fails to upgrade keeps
  its snapshot value and can never fall back to `--`.**
- **When the EGX is shut, the age stamp reads "Market closed", not a climbing
  "...ago".** A price cannot get newer than the last close while the market is
  closed (the Fri/Sat weekend, or any time outside 10:00-14:30 Cairo), so a
  growing age reads as staleness that isn't real — over a weekend it would climb
  past "30h ago" on a number that is exactly as fresh as it can be. `isEgxOpen`
  in `page.tsx` is the Cairo-aware, DST-safe gate (via `Intl`), read off the
  same ticking clock as the age it replaces so every card flips together. A
  closed-market bar also always shows "close", even for today's session once
  the 14:30 bell has rung.
- **Sorting by score exists only because of this.** While cards were fetched a
  page at a time, off-screen stocks had no score, so the control was
  structurally impossible.

**It also warms the market-regime card for free.** `read_cached_scores` falls
back to the snapshot when the in-memory cache misses, which per this document
is most of the time now the app is closed and has no anonymous traffic.

### GET /api/risk — the per-stock risk grade

**The strongest measured surface in the app, and the only per-stock ranking the
evidence supports.** Reads the `risk_snapshot` table and ranks it
cross-sectionally on the way out.

Measured on the cached panel (`python -m scripts.calibrate`), liquid universe,
16,220 observations / 202 symbols / 221 monthly dates:

| past 63d volatility predicts | IC | t | non-overlapping IC / t |
|---|---|---|---|
| next 126d realized volatility | **+0.5631** | +55.8 | +0.5791 / **+24.0** |
| next 126d max drawdown | **+0.4338** | +40.5 | +0.4707 / **+16.7** |
| *(composite score → return, for scale)* | *−0.029* | *−2.85* | — |

That is ~20x any return signal found anywhere in this project and clears the
pre-registered |t| > 3.0 bar on non-overlapping data eightfold. What each
quintile went on to do (medians — EGX forward returns are heavily right-skewed
and a mean describes a distribution nobody experiences):

| quintile | future ann. vol | median max DD | p90 max DD |
|---|---|---|---|
| 1 Calm | 32.0% | 20.0% | 40.7% |
| 2 Steady | 37.8% | 24.8% | 46.3% |
| 3 Average | 42.4% | 27.6% | 49.3% |
| 4 Jumpy | 45.2% | 29.9% | 53.0% |
| 5 Wild | 51.2% | 34.2% | 59.4% |

**This surface must never carry a return claim.** Low volatility does rank
positively against forward returns (IC +0.084, t=4.97 at 21d), but the
realisable long/short spread is only t=1.70, the mean by quintile is
flat-to-inverted, and no historical market cap exists to neutralise a possible
size effect. High-vol EGX names are lottery tickets: a few huge winners lift the
MEAN while the MEDIAN is clearly worse. It answers "how much will this move and
how deep a hole should I expect", never "will it go up".
`tests/test_risk_grade.py::test_risk_grade_makes_no_return_claim` greps for it.

**Ranking happens at READ time**, over the TRADEABLE subset only (>1M EGP/day
turnover, traded on ≥80% of the last 60 sessions, price >0.5 EGP). An
untradeable symbol is returned with its raw sigma and a **null band** rather
than a rank it has not earned. `oldest_measurement` is reported, not the
newest — a snapshot is only as current as its stalest row.

**Note `measure()` reports two sigmas.** EWMA(0.94) forecasts better (QLIKE ~14%
better than a 20-day trailing SD) and is surfaced, but the PERCENTILE is built
on the trailing 63-day sigma the quintile table was fitted on. Swapping the
ranking input without refitting would leave the historical mapping describing a
different variable.

### POST /api/cron/risk_snapshot — chunked, externally scheduled

**Scheduling is external (cron-job.org), not `vercel.json`.** That removes the
Hobby 2-cron/daily limit entirely. It does NOT remove the 30-second Vercel
execution limit — measuring the whole universe was clocked at >400 s cold,
because each symbol pulls 400 bars through a client that retries hard on socket
timeouts.

So each call measures a SLICE. **Omit `cursor` and the endpoint picks the
STALEST symbols itself** — that is the production mode, because cron-job.org
cannot read a response body and feed a cursor back:

```
POST /api/cron/risk_snapshot?limit=6
Header: X-Cron-Secret: <CRON_SECRET>
-> {processed, written, failed[], mode, stale_remaining, universe, measured_at}
```

One fixed URL on an interval and the universe converges. Nothing is stored,
nothing can desynchronise, and a symbol that fails is retried on the very next
call because it stays stale. Watch `stale_remaining`: it should fall to 0 within
a pass and stay there. Passing `cursor` explicitly walks fixed slices instead —
useful for a controlled manual sweep, and it stores no state either.

Design rules, all load-bearing:
- **Stalest-first replaced an auto-advancing cursor, deliberately.** A cursor
  must advance past a chunk that FAILED, or one permanently broken symbol pins
  it and starves everything behind it — which on a daily job means the failure
  waits a full day. Staleness retries immediately and needs no state.
- **EVERY attempt is recorded, including failures and unmeasurable symbols**
  (null sigma, `tradeable=False`). This is the sharp edge of stalest-first: a
  symbol with no row is maximally stale, so a failure that wrote nothing would
  be re-picked on every call. The EGX has ~34 effectively dead names against a
  20-symbol batch — measured in simulation, they filled every batch forever and
  **not one live symbol was ever measured**. `tests/test_risk_grade.py` pins
  both the convergence and the "no `continue` skips the upsert" rule.
- **There is deliberately no "finalize" step.** Percentiles are computed at read
  time, so a partly-refreshed table is coherent rather than half-ranked against
  yesterday. This is what makes chunking safe.
- **Never wipes**, matching `refresh_pe_data`. Each row carries its own
  `measured_at` so the read path can report real freshness.
- ISIN-coded rows are dropped, matched on **LENGTH not the "EGS" prefix**:
  an Egyptian ISIN is 12 chars, while **EGSA is a real 4-char EGX ticker** a
  prefix rule silently deletes.

- **HALF THE UNIVERSE IS UNFETCHABLE, and that is what timed the cron out.**
  Measured: **84 of the 166** symbols in `data/egx_tickers.json` have NEVER
  returned data from tvDatafeed, confirmed independently by the disk cache (269
  of 373 pickles have data; the rest are cached `None`). The failures are
  **PERSISTENT, not transient** — the identical set fails on every pass, and
  pacing does not help (5/10 with no pause versus 4/10 with a 2-second pause).

  A successful fetch takes ~3.5s; a refusal takes ~6s, because the vendored
  client wraps every call in `@retry(Exception, tries=20, delay=0.5)` around
  `get_hist(timeout=-1)`. Twenty symbols took 67 seconds, so the first
  production runs returned 200 while the scheduler had already given up.

  Two fixes, both load-bearing:
  - **A wall-clock `DEADLINE_SECONDS` (15s)**, mirroring `BATCH_DEADLINE_SECONDS`
    in the batch scorer. The handler stops and returns what it has. Stopping
    early is safe *because* selection is stalest-first: whatever went unmeasured
    is still the stalest thing in the table and is picked up first next call.
    This is the case that design was chosen for.
  - **`consecutive_failures` on `risk_snapshot`**, incremented on refusal and
    reset to 0 on success. At `FAILURE_DEMOTION_THRESHOLD` (3) a symbol is
    demoted behind everything healthy. **Demoted, not blocklisted** — a static
    list rots, and a symbol that starts working un-demotes itself on its first
    good fetch. `demoted_symbols` is reported so a growing number is visible.

  `DEFAULT_CHUNK` is **4**, not 20: four symbols is what reliably fits the
  deadline even when all four are refused. The deadline reserves
  `PER_SYMBOL_BUDGET_SECONDS` (7s) before STARTING another symbol — checking
  elapsed time alone let a fetch begin at 14.9s and run past the budget, which
  is why the first fixed run still reported `elapsed_seconds` 15.4.

  **Seed the counters instead of paying to learn them.** Left alone the job
  discovers the bad half the expensive way: 84 symbols x 3 strikes x ~6s, which
  is most of a day before the working symbols get priority.
  `scripts/seed_symbol_health.py` reads the answer off the disk cache — a cached
  `None` means the feed genuinely had nothing — and writes those counts once.
  Run it after any DB reset. It is still a HINT: every seeded symbol is fetched
  after the healthy ones are fresh, and one success resets it to zero.

  Note the good half is FAST. Measured: ABUK, ACGC, ADIB, AFDI and AJWA each
  return 400 bars in ~1.4s; only the refusals cost ~6s. So the budget problem
  was never throughput, it was the dead half consuming all of it.

`select_stalest()`, `plan_chunk()` and `is_isin()` are pure, so the selection
logic a scheduler depends on is testable without Postgres — tests/ has no DB
fixture by design.

**Cadence — and the ONE-HOUR WINDOW THAT DOES NOT FIT.** The data is DAILY bars,
so a symbol's sigma only changes once per trading day; refreshing more often
re-measures identical bars and hammers the feed for nothing. Run it after the EGX
close (Sun–Thu, 14:30 Cairo).

**A call measures at most SIX symbols, whatever `limit` says.** The deadline stops
the loop once `elapsed > DEADLINE_SECONDS - PER_SYMBOL_BUDGET_SECONDS` (8.0s), and
a healthy fetch takes ~1.4s. So the DEADLINE, not `limit`, is the binding
constraint at any `limit` above 6:

| `limit` | all-healthy chunk | all-refusal chunk |
|---|---|---|
| 4 | 4 | 2 |
| **6** | **6** | 2 |
| 20 | 6 | 2 |

This is why the example above passes `?limit=6`. It used to read `limit=20`
alongside a claim that *"the 166-symbol universe needs 9 calls, so every 5 minutes
within one hour covers it with headroom"* — arithmetic from before the deadline
existed, and wrong by a factor of three. **One hour does not cover the universe.**

Simulated against the real `select_stalest` and the real universe (166 symbols,
84 known-dead, 82 healthy), counting DISTINCT healthy symbols measured per day:

| schedule | calls/day | `limit=4` | `limit=6` |
|---|---|---|---|
| `*/5 15 * * 0-4` | 12 | 48 / 82 | 72 / 82 |
| **`*/5 15-16 * * 0-4`** | 24 | 82 / 82 | **82 / 82** |
| `*/5 15-17 * * 0-4` | 36 | 82 / 82 | 82 / 82 |

**Two hours at `?limit=6`** is the supported setting: 24 x 6 = 144 slots against 82
healthy symbols, 1.8x headroom, so a cold start or a missed fire still completes
the pass. Under-provision it and `stale_remaining` never reaches 0, and `/api/risk`
permanently ranks a partly-yesterday universe.

`0-4` is Sun–Thu (0 = Sunday), matching the EGX week. Pin the job to
**Africa/Cairo, not UTC** — Egypt observes DST, and a local-time schedule shifts
with the market close instead of drifting an hour away from it twice a year.

Because selection is staleness-driven, extra calls are harmless and missed ones
simply catch up next run.

**A demoted symbol is never re-fetched, so it cannot un-demote itself.**
`select_stalest` ranks on `(demoted, measured_at, symbol)`, and `demoted` is a HARD
priority rather than a tiebreak — with 82 healthy symbols against a chunk of 6, the
first 6 are always healthy. Simulated over four days at every schedule above, the
84 demoted symbols were selected **0 times**. That is correct for the ~84 names the
feed genuinely never serves, and it is what keeps the budget on live stocks. But
the "un-demotes itself on its first good fetch" recovery described above CANNOT
FIRE while the healthy half outnumbers the chunk, so a symbol that starts working
is not picked up on its own. Recovering one today means clearing its
`consecutive_failures` by hand.

**Auth:** in `PUBLIC_ENDPOINTS` (an external scheduler carries no user token),
guarded by the `CRON_SECRET` env var exactly as `/api/pe/refresh` is guarded by
`PE_REFRESH_SECRET`. `tests/test_auth_gate.py::test_every_public_cron_checks_a_shared_secret`
walks each public cron's source and fails if one stops reading its env var —
without that check the allowlist entry alone would open the route to anyone.

### The backtest's Risk-Adjusted verdict — withheld for months, now readable

`scripts/backtest.py` ran ONE flat risk-free rate across twenty years in which
the CBE ranged **8.25%–27.25%**, and withheld Risk-Adjusted's verdict because of
it. `macro_series` now supplies dated rates, so each scoring date is graded
against the cash return over **its own trailing year**, and the withholding is
lifted.

**The correction was large.** Across the 221 rebalance dates the true hurdle
ranges 8.25%–27.25% with a **median of 9.40%**, against the flat **19%**
previously applied to all of them — roughly double the real bar for most of the
window.

**The verdict is no.** Risk-Adjusted measures **IC −0.006 at 21 days**, the only
horizon whose significance is trustworthy (63d and 126d forward windows overlap
3× and 6×). It drifts positive at those longer horizons (+0.012, +0.024) but
those figures are shape, not evidence. The category still carries **13% of the
default weight on no measured edge** — the same standing as `trend` (−0.029) and
`relative_strength` (−0.036), which are actively wrong-signed.

**Adopting the dated panel barely moved the headline**, which is the reassuring
part: composite IC at 21d went −0.0293 → **−0.0288** (t −2.85 → −2.83), and the
forecast bands and the entire risk-grade calibration are **untouched** because
they are built on returns and volatility, not on scores. Only the market-regime
figures moved, and they got slightly weaker.

**A panel records how it was scored.** `write_run_meta` stamps a
`<panel>.meta.json` beside it and `analyze_backtest` reads that rather than a
module constant — otherwise a flat-rate panel analysed on a machine that happens
to have rate history would silently lose its caveat. **An unstamped panel reads
as flat-rate**, so every panel built before this existed keeps its withholding.

Note the previous `CONFOUNDED_CATEGORIES` was a module-level constant in
`backtest.py` imported by `analyze_backtest.py`, so the caveat described the
machine running the ANALYSIS rather than the run that produced the panel.

### macro_series — dated macro, and the release lag that makes it usable

```sql
macro_series (series_code, observed_period, value, released_at, source,
              updated_at, PRIMARY KEY (series_code, observed_period))
```

Seven series, all verified reachable before the table was written: EGINTR
(policy rate), EGIRYY (inflation), EGCPI, EGFER (reserves), EGM2, EGUR via the
keyless **ECONOMICS scanner** (`scanner.tradingview.com/economics2/scan`), plus
**FX_IDC:USDEGP**. History comes through the already-vendored client — EGINTR
300 monthly bars to 2001-05, USDEGP 5,000 daily bars to 51.000. ~6,600 rows.

**`released_at` is the load-bearing column, not `observed_period`.** A macro
bar's timestamp is its REFERENCE period: August's inflation is not knowable in
August. Each series carries a publication lag (CPI 10 days, reserves 7, M2 60);
an FX rate is lag 0 because a price is knowable the day it prints. Read through
`get_macro_at`, which filters on release — same discipline as
`fundamentals_annual.first_usable_date`.

Current values ride the nightly `/api/pe/refresh` slot as one POST. History is
**offline** (`scripts/backfill_macro.py`) because 5,000 FX bars do not fit in 30
seconds and only need fetching once. **Seeded and verified 2026-09-02: 6,604
rows**, all seven series complete, EGUR reaching back to 1993-06.

**Who reads it.** `get_risk_free_steps` serves the whole EGINTR step function to
the sales ledger and to `scripts/backtest.py`; see *GET /api/sales* and *The
backtest's Risk-Adjusted verdict*. It filters on `observed_period` rather than
`released_at` deliberately — scoring an outcome that already happened has no
look-ahead to defend against — while the backtest, which simulates a DECISION at
a past date, bounds its own window instead. EGINTR carries a zero publication
lag either way, so for this series the two columns agree.

**A dated series is worth nothing until something reads it.** For a day after
the backfill landed, `get_risk_free_at`, `get_macro_at` and `get_fx_at` had
**zero callers** — 6,604 rows of correct history sitting beside code that still
used one scalar. `get_fx_at` and `get_macro_at` still have none.

**Write in BATCHES — `upsert_many`, not `upsert` in a loop.** Three attempts at
the FX series, two of which lost data: one statement per row exited 0 having
stopped at 2022-10-31 (a bare `except: continue` swallowed a dropped
connection); all 5,000 in one transaction lost 3,689 rows when Neon's pooler
closed the connection mid-write; 250-row transactions still doing one statement
per row were too slow to commit a single batch. One multi-row INSERT per batch
makes the whole series ~10 round trips and it finishes in seconds. `upsert_many`
deduplicates WITHIN a batch because Postgres refuses an `ON CONFLICT DO UPDATE`
that would touch the same row twice — a repeated date fails the whole batch,
which per-row upserts never had to care about.

The read path is what proves the backfill: `get_fx_at` now returns 8.858 on
2016-11-01 and **17.70** on 2016-11-30 (the float), 47.34 after the March 2024
devaluation and 51.0 today. While USDEGP stopped at 2022, every date after it
answered **24.15** — understating the devaluation by half.

### The USD lens — the largest distortion the app used to hide

**EGX30 rose 8.25x in EGP over twenty years and went to 0.94x in USD.** Twenty
years of "the market went up 8x" is, in hard currency, twenty years of going
nowhere, and nothing on screen said so. `core/currency.py` supplies the maths and
`MacroCard` states it.

**Convert, never subtract.** A stock that doubled in EGP while USD/EGP went
30 → 50 returned +100% in pounds and **+20%** in dollars; subtracting a "40%
devaluation" gives +60% — wrong by 40 points, and wrong in the flattering
direction. `tests/test_currency.py` pins that the two disagree.

### The dividend gap — measured, and disclosed rather than patched

The cached panel is split-adjusted but **dividend-unadjusted**, so every return
the app computes is a PRICE return while the policy rate it is compared against
is a TOTAL return. Measured on eight liquid names via Yahoo's dividend history
(COMI.CA gives 6,493 bars, 24 dividends, 10 splits): **median drag 3.70 pp/yr**
— COMI 3.55, SWDY 3.85, ETEL 6.44, **ABUK 10.22**, TMGH 0.88.

`score_risk_adjusted` now says so in its reason string. It is **not** added back:
`score_quality` already credits dividend yield, and paying a stock twice for one
fact yields a number meaning neither — the failure the liquidity band exists to
avoid. `core/corporate_actions.py` builds a proper total-return series for
offline validation; charts stay on RAW close, because a dividend-adjusted chart
moves every historical level and would silently rescore trend and support.

### GET /api/calibration — the app's own accuracy record

Serves every measured claim with what it actually delivered, read from the
fitted constants the app uses, so the page cannot drift from the code. Rendered
at `/calibration`, linked from `ForecastCard`.

**Coverage alone is gameable** — a band from zero to infinity contains every
outcome — so width ships beside it: the 90% band spans a **median 88% of spot**
over 60 days (p25 72%, p75 110%). Saying the range is wide is the point.

**The failures are on the same page as the successes**, deliberately: the
composite's IC ≈ 0, the retracted +0.318, earnings yield falling to t=2.00 once
low volatility is controlled for, and the profitability/growth factors that flip
sign between halves. A record that only lists what worked is marketing.

### Market breadth — an operational fix first

`core/breadth.py` aggregates flags the **risk snapshot already stores**
(`above_sma200`, `rsi_14`, computed inside `measure()` since it holds the window
anyway). Breadth otherwise needs its own pass over the universe, measured at
>400s.

The real reason it exists: `/api/market_regime` averages composite scores the
DASHBOARD happens to have cached, and since the app went closed that cache is
mostly cold. Breadth is always as fresh as last night's cron.

**It is not a second forecast.** Its strongest leg — % of stocks oversold —
reaches rho −0.188, Newey-West t −2.44, below the |t| > 3.0 bar; the others
(mean RSI +1.95, % above 50-day MA +1.90, % above 200-day MA +1.66) do not clear
even 2. It ships as context with the same framing as the regime card. The one
notable detail is the SIGN: more stocks oversold predicts WORSE forward returns,
not a contrarian bounce.

**It renders as `BreadthStrip`, INSIDE `MarketRegimeCard`** — not as a card of
its own. Breadth and the score average answer the same question from two
sources, and two competing market cards on one screen invite picking whichever
agrees with you. Nesting also makes the real relationship legible: breadth is
what stays fresh when the score cache goes cold, which is most of the time now
the app is closed. (It was computed, returned by the API, and rendered NOWHERE
from the day it shipped until 2026-09-02.)

**Two colour rules, and they are the whole design:**

- `pct_above_sma200` is a genuine direction, so `gain`/`loss` applies — but only
  outside a **neutral 45–55 band**. At 50% the market is evenly split, and
  painting that green (or 49.9% red) claims a direction the reading does not
  have. Same refusal the regime card makes for its top two bands.
- **`pct_oversold` must NEVER be coloured as an opportunity.** The measured sign
  is counterintuitive, and a green "22% oversold" would teach the exact opposite
  of what was measured. It renders in neutral text with the sign stated in
  words: *"periods with more oversold shares were followed by weaker returns,
  not a bounce"*, followed by rho, t and "still too weak to act on".

Below `MIN_SYMBOLS_FOR_BREADTH` (15) the strip says what is missing and that the
nightly job fills it in, rather than showing a percentage computed off nine
stocks. That is the state on screen today — 12 tradeable symbols measured.

### GET /api/historical, GET /api/compare
Multi-symbol historical data for comparison page.

### GET /api/intraday
Intraday (1/5/30 minute) price data.

## Technical Indicators (`egx-api-be/app/core/indicators.py`)

All implemented from first principles — **no external TA libraries**. Each function has a detailed docstring explaining what it measures and why it matters. This is intentional — the app is a learning tool.

**Basic:** `sma`, `ema`, `rsi`, `macd`, `bollinger_bands`, `daily_returns`, `volatility`, `cumulative_returns`

**Advanced:** `atr`, `obv`, `stochastic`, `support_resistance`, `fibonacci_levels`, `ma_crossovers`, `compute_beta`

**Composite engine inputs:**
- `adx(high, low, close, period=14)` → `(adx_series, plus_di, minus_di)` — trend strength via Wilder's smoothing
- `mfi(high, low, close, volume, period=14)` → Series — volume-weighted RSI (0-100)
- `detect_divergences(close, indicator, lookback=60, swing_window=5)` → dict with `{bullish, bearish, hidden_bullish, hidden_bearish, detail}` — swing-based regular + hidden divergences
- `volume_price_confirmation(close, volume, lookback=20)` → dict with `{classification, price_change_pct, volume_ratio}` — classifies as confirmed_up/down, unconfirmed_up, accumulation, quiet, normal
- `multi_timeframe_alignment(daily_close, weekly_close)` → dict with `{daily_trend, weekly_trend, aligned, alignment_score}` — compares SMA50 slope on each timeframe
- `relative_strength(stock_close, benchmark_close, lookback=30)` → dict `{stock_return_pct, benchmark_return_pct, alpha_pct, leader, laggard}` — alpha vs EGX30
- `annualized_return(close, lookback=252)` → float | None — annualized return compared against T-bill rate in Risk-Adjusted scorer
- `liquidity_score(volume, index_membership, lookback=20)` → dict `{avg_volume, classification, thin}` — index-aware (EGX30 vs NILEX volume baselines differ ~100x)

`compute_all(df)` returns all array-based indicators (including adx/plus_di/minus_di/mfi) aligned to input dates. Non-array results (support/resistance, fibonacci, crossovers, beta, divergences, volume_price, multi_timeframe) are computed separately in endpoints.

## Composite Score Engine (`egx-api-be/app/core/composite.py`)

Pure functions, no DB access. Called by `analysis.py` and `portfolio_analysis.py`.

**8-category model** (expanded from the original 5 in 2026). Default is the "Beginner Safe" preset — tilted toward stable, leading, cash-beating stocks:

```python
# "Beginner Safe" default — same as DEFAULT_WEIGHTS
DEFAULT_WEIGHTS = {
    "trend": 18, "momentum": 15, "volume": 12, "volatility": 10,
    "divergence": 8, "quality": 12, "risk_adjusted": 13, "relative_strength": 12,
}

PRESETS = {
    "beginner_safe":    DEFAULT_WEIGHTS,
    "balanced":         # evenly across 8
    "trend_follower":   # heavy on trend/quality/relative_strength
    "reversal_hunter":  # heavy on divergence/momentum
    "income_defensive": # heavy on risk_adjusted/quality/volatility (capital preservation)
}
```

**Category scorers (each returns `(score | None, reasons)`):**
- `score_trend(...)` — price vs SMA50/200, ADX strength, DI±, Golden/Death Cross
- `score_momentum(...)` — RSI zone, MACD histogram direction, Stochastic crossover
- `score_volume(..., liquidity=None)` — OBV trend, MFI bands, volume-price classification, plus a **penalty-only** liquidity band (see *Liquidity*)
- `score_volatility(...)` — Bollinger Band position + squeeze detection
- `score_divergence(...)` — regular ±15, hidden ±5, double-divergence bonus ±10, baseline 50
- **`score_quality(multi_timeframe, trend_consistency, current_drawdown_pct, *, pe_ratio=None, dividend_yield=None, loss_making=None)`** — rewards smooth trends + aligned timeframes + being near the 52-week high (penalises whipsaws), plus the valuation bands below. **The fundamentals args are keyword-only** so a new one can't shift an existing positional argument.
- **`score_risk_adjusted(annualized_return_pct, risk_free_rate_pct, vol_ann_pct, atr_pct_of_price, history_days)`** — compares per-stock return to the ~19% policy rate. **Min 120-day history gate**: returns `None` if insufficient data, and renormalization excludes the category
- **`score_relative_strength(rs_dict)`** — alpha vs EGX30 (leader/laggard classification over 30 days)

**Orchestrator:**
- `compute_composite(indicators, extras, weights, macro=None)` — calls all 8 scorers, renormalizes weights across categories that returned non-None, applies **macro modulation** post-hoc, and returns `{score, signal, categories, weights, macro_adjustment, macro_context}`
- **Macro modulation** (`apply_macro_modulation`): applied after the weighted sum, based on EGX30 trend. **Bullish and Sideways are both 1.0× no-ops** — a neutral market must leave scores alone, otherwise every stock carries a permanent penalty. **Bearish shifts the whole distribution down**: above 50 scores are damped 15% toward neutral, below 50 they are pushed a further 15% away from it. Note this is deliberately asymmetric — it is a shift-down, not a symmetric pull-to-neutral. The delta is surfaced to UI as `macro_adjustment`.
- `get_weights_from_db(db, user_id=None)` — **weights are PER-USER.** Read chain, per key: `user_settings` for this user → global `settings` → `DEFAULT_WEIGHTS`. The middle tier is what let weights become per-user without moving anyone's scores on deploy, and the per-key granularity is why extending CATEGORY_ORDER from 5 → 8 did not break existing DBs. `user_id=None` is the anonymous/global context, used by the public dashboard and by the market-regime reader.

  **Passing `user_id` is how a score becomes the caller's.** `/api/analysis` gets it from `get_optional_user`, `portfolio_analysis` from its existing `user`. Forget it in a new scoring path and that page silently serves default-weighted scores while every other page serves the user's — the same class of divergence as *One Score Per Stock*.
- `weights_hash(weights)` — short hash for cache key invalidation; extended automatically to the 8 keys.

Signal bands are **half-open with the lower bound inclusive**, applied AFTER macro modulation:

| Score | Label |
|-------|-------|
| < 20 | Very Weak |
| 20–39.99 | Weak |
| 40–59.99 | Neutral |
| 60–79.99 | Strong |
| ≥ 80 | Very Strong |

**These describe CONDITION, not action — do not reinstate Buy/Sell.** They read
Strong Buy / Buy / Hold / Sell / Strong Sell until 2026-08-26. A walk-forward
backtest (`scripts/backtest.py`, 2007–2026, 36,818 symbol-dates) found the score
cannot rank one stock above another: cross-sectional IC ≈ 0 and slightly
negative, nine of ten score deciles had a median 21-day forward return of
**exactly 0.00%**, and among liquid names the "Sell" bucket slightly
**outperformed** the "Buy" bucket. An instruction the evidence contradicts is
worse than none, and on the sell side it pointed the wrong way.
`tests/test_fixes.py::test_labels_describe_condition_not_action` fails if the
action words come back.

`classify_signal` in `composite.py` is the canonical implementation. The frontend mirrors it via `scoreBand()` in `egx-api-fe/src/app/lib/constants.ts` — **every** colour, label and badge derives from that one function, so they cannot disagree at a boundary. The constants are named `SCORE_*_MAX` for historical reasons but each is really the *minimum* of the band above it; comparing with `<=` shifts every band down by one.

## One Score Per Stock — READ BEFORE TOUCHING ANY SCORING PATH

The composite score is computed in three places:

| Path | Entry point |
|------|-------------|
| Stock detail page | `analysis.py` → `get_analysis` full path |
| Dashboard cards | `analysis.py` → `_compute_batch_one` |
| Portfolio rows | `portfolio_analysis.py` → `_analyze` per-holding loop |

**All three MUST produce the same number for the same symbol.** They each used to assemble their own `extras` dict, and the batch path omitted the inputs that `score_quality`, `score_risk_adjusted` and `score_relative_strength` need. Those scorers returned `None`, `compute_composite` renormalized over the remaining 5 of 8 categories, and the categories that vanished were exactly the punitive ones. Measured on identical data: **66 "Buy" on the dashboard card, 45 "Hold" on the detail page** for the same stock. A user tapped a Buy and landed on a Hold.

Rules that keep them aligned:

1. **Never build an `extras` dict by hand.** Call `core/extras_builder.py::build_composite_extras`. `tests/test_fixes.py::test_no_router_hand_rolls_composite_extras` greps the routers and fails if you do.
2. **One fetch window** — `INTERNAL_BARS_MIN` (400) everywhere. The score depends on the window (volatility, S/R, SMA200, beta), so a "lightweight" 220-bar batch fetch silently rescored every card.
3. **One divergence lookback** — `DIVERGENCE_LOOKBACK_FULL` (60) everywhere.
4. **One benchmark series** — fetch EGX30 through the shared `make_key("egx30", exchange, interval, INTERNAL_BARS_MIN)` cache key so relative strength and beta compare against identical data.
5. **Multi-timeframe is derived by resampling the daily frame** (`W-THU`), not by a separate weekly fetch. It costs no extra network call, which is why the batch and portfolio paths can now afford it, and it guarantees daily and weekly agree.
6. **Scoring inputs belong in the cache key.** `/api/analysis` keys on `w_hash + macro_tag + rfr_tag`; the batch path keys on the same. Omitting the macro regime meant a bullish→bearish flip updated the dashboard immediately while the detail page served a pre-flip score for the rest of the 15-minute TTL.

If a category legitimately can't be scored (a stock with 60 bars has no 1-year return), it is dropped on *every* page for the same reason — and `ScoreBreakdown` renders it as "excluded — not enough data" rather than advertising a weight it didn't carry.

### Fundamentals bands — centred on the EGX median, NOT on "low = cheap"

Both valuation bands live in `score_quality` and are calibrated against the
**actual EGX distribution** (measured 2026-08-25 over all 293 listed stocks):

| Metric | coverage | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|
| P/E (TTM) | 64 | 5.8 | **12.4** | 22.0 | — | 2756 |
| Dividend yield (>0) | 86 | 1.17% | **3.12%** | 5.11% | 7.21% | 42.55% |

**Why centring matters.** The old band gave `+8` to any P/E under 20 — which is
most of a market whose median is 12.4. Since only ~22% of EGX stocks have a P/E
at all, simply HAVING data was worth points, and the app would have ranked
covered stocks above uncovered ones for a reason unrelated to their merit. The
rule to preserve: **the median EGX stock should score ≈0 from these bands.**

P/E: `<3 → +3`, `3–8 → +12`, `8–15 → +4`, `15–25 → −2`, `25–40 → −8`, `≥40 → −14`.

**The `<3` floor is not cosmetic.** A P/E under 3 on the EGX means one-off
earnings or a price that already collapsed. MEGM trades at **P/E 0.7 with a
42.55% dividend yield** — under the old band that was `+15 "Very cheap on
earnings"`, i.e. the app calling a suspended, distressed stock a bargain.

Dividend yield is deliberately **NON-monotonic**: `0–2 → 0`, `2–4 → +4`,
`4–8 → +8`, `8–15 → +4`, `≥15 → −8`. A yield above 15% is a special dividend or
a collapsed price, not income quality.

**Framing rule for every DY string:** with the policy rate near 19%, *no* EGX yield is
competitive as income — even 8% loses. Reason strings must present a dividend as
**evidence the company generates real cash**, never as income. `dividend_yield`
of `0.0` means "pays nothing", which is normal for a growth company; it scores
identically to unknown.

### Liquidity — penalty-only, index-aware, zero-aware

`indicators.liquidity_score` now feeds the composite through `score_volume`.
Three deliberate properties:

1. **Penalty-only** (`thin → −12`, `low → 0`, `normal → 0`). `score_volume`'s
   other bands are *directional* ("is volume confirming this move?"); liquidity
   is *structural* ("can you get out?"). Awarding points for normal liquidity
   would move ~95% of stocks for no information and let the two ideas cancel
   into a number meaning neither. The `low` tier still emits its reason string.
2. **It cannot carry the category.** The all-None guard in `score_volume`
   ignores `liquidity` on purpose — otherwise a stock with no volume-confirmation
   data would score 38 off a single liquidity reason and claim the full weight.
3. **Dead sessions beat the average.** A mean hides zeros: MEGM has been frozen
   at 12.54 with zero volume since **January 2022**, yet one old block trade left
   it averaging ~99k shares/day — comfortably "low", not thin. If ≥30% of the
   lookback window (`_DEAD_SESSION_SHARE`) has zero volume the stock is thin
   regardless of its mean, and the reason names the count.

4. **Absence from the market's calendar is a fourth test, and the only one
   that can see days the symbol has no row for.** `liquidity_score` takes an
   optional `calendar` (the benchmark's index, already fetched for beta) and
   flags a stock that traded on under 70% of the market's recent sessions. The
   dead-session count above cannot do this by construction: a stock that stops
   being quoted has no row on the days it misses, so its last 20 rows look
   perfectly healthy. Measured on the cached panel, exactly **3 of 235 symbols**
   change — SUCE (32% of sessions, last print three months old), LKGP (41%),
   CPME (47%).

   Two guards make it safe, and both were found by measuring rather than
   reasoning:
   - **The window starts at the symbol's first bar.** GOUR listed in February
     and has traded every session since; the uncorrected ratio (52%) would have
     condemned it. Flagging every new listing would be a worse bug than the one
     being fixed.
   - **The window is bounded to ~244 days INSIDE the function**, not left to the
     caller. Handed a full 5,000-bar benchmark history the test measures
     LIFETIME tradeability, and 31 symbols flipped — including DSCW, which
     turns over 48 million shares a day but scored 0.47 across its whole life.
   - `calendar` is optional, so omitting it reproduces the old behaviour
     exactly; only the Daily path passes one, because on a Weekly view the
     benchmark index is weekly too and "a session" would mean something else.

Index membership comes from `core/index_membership.py`, which reads
`data/egx_tickers.json` **directly**. Do NOT route this through
`tickers._load_tickers()` — that can fire a 10 s TradingView POST on a cold
container, putting every dashboard card behind a ticker-list fetch. Unknown
symbols return `None`, which `liquidity_score` maps to the EGX100 floors (the
behaviour every symbol had before membership was plumbed through).

#### `data/egx_tickers.json` is hand-maintained and is DRIFTING — verify before trusting it

Nine entries carried a **wrong company name, a wrong sector, or both**, fixed
2026-09-05 against the live scanner's `description` + `industry`. They were not
obscure: `MASR` was filed as *"Banque Misr" / Banking* and is **Madinet Masr**, a
real-estate developer; `EXPA` and `SAUD` were filed as non-banks and are both
banks; `ECAP` was filed as a financial and makes **ceramics**; `CLHO` carried
another company's name entirely; `BINV` and `BTFH` shared one name, so the file
had a literal duplicate. Sector drives the dashboard's sector filter, so these
were wrong on screen.

Corrections stayed inside the file's own 10-value sector vocabulary — do not
introduce a new sector value casually, the filter pills enumerate them.

**Three known problems remain, all deliberately NOT fixed because each needs a
decision rather than a correction:**

- **`index` membership is badly drifted.** Measured against the scanner's own
  `indexes` column: 13 symbols the file calls EGX30 are not in the feed's EGX30,
  **6 real EGX30 constituents are absent from the file entirely** (EFID, EMFD,
  ORHD, RMDA, VLMR, VLMRA), and 8 more sit under the wrong index. The feed only
  knows **EGX 30** — it reports `indexes: None` for 265 of 296 symbols — so
  there is no machine-readable source for EGX70 / EGX100 / NILEX, and `index`
  sets the liquidity floors. Reconstituting it means choosing a source, not
  editing a label.
- **`QNBE` is missing entirely** — Qatar National Bank Alahli, ~126bn EGP and
  top-8 by market cap. Adding it requires an `index` value that cannot be
  verified (the feed does not list it in EGX30, plausibly on free float, since
  QNB Group holds the overwhelming majority). Guessing would feed the liquidity
  band a fabricated input.
- **78 of the 166 file symbols do not resolve in the live scanner at all**,
  including `EKHO`/`EKHOA`, `GLCM` and the 6-character `EIPICO` (EIPICO trades
  as `PHAR`, which the file also lists separately). Some are genuinely delisted
  or renamed; `EKHO` quotes in USD and may simply fall outside the scanner's
  EGX filter. Removing 78 dashboard rows is a behaviour change, not a fix.

### When to split "Valuation" into its own category

`score_quality` now carries 3 technical inputs and 3 fundamental ones. Sub-bands
were chosen over a 9th category because a new category costs: `CATEGORY_ORDER` +
`DEFAULT_WEIGHTS` + all 5 `PRESETS` + `compute_composite`'s hand-written
`category_raw` dict (**not** derived from `CATEGORY_ORDER` — a missing entry is a
`KeyError`) + `db.py` seeds, and on the frontend **six** hardcoded lists of which
**three fail silently** (no slider, wrong normalization percentages, bar never
renders).

**The pre-decided trigger:** when a *fourth* fundamental input is added (payout
ratio, EPS growth, debt/equity), "Quality" stops being coherent — split
**Valuation** out and pay the 9-category tax once, deliberately.

### Interval calibration (Daily / Weekly / Monthly)

The stock detail page can be viewed on Daily, Weekly or Monthly bars. **Anything that annualizes, spans "one year", or gates on history length MUST scale by `BARS_PER_YEAR[interval]`** (252 / 52 / 12 in `extras_builder.py`) — never the bare `TRADING_DAYS_PER_YEAR` constant.

Hardcoding the daily constants made the non-daily views quietly wrong:

| Input | Bug when weekly bars hit daily constants |
|-------|------------------------------------------|
| `annualized_return` | treated 252 *weeks* as one year — a multi-year run reported as a single year's gain, then compared to the 19% policy rate |
| annualized volatility | weekly σ × √252 instead of √52 — overstated 2.2× |
| `current_drawdown_pct` | `tail(252)` = a ~5-year drawdown labelled 1-year |
| `history_days` gate | 120 weekly bars (2.3 y) passed the same gate as 120 daily bars (6 mo) |
| `relative_strength` | 30 weekly bars compared against a "30-day" label |
| `multi_timeframe` | resampling weekly→weekly is a no-op, so it compared the timeframe **with itself** |

Together those feed Risk-Adjusted (13%) and Quality (12%) — a quarter of the score. `interval` is now a parameter of `build_composite_extras` and the higher timeframe comes from `_HIGHER_TIMEFRAME_RULE` (Daily→Weekly, Weekly→Monthly, Monthly→none). Tests in `tests/test_fixes.py` assert Daily and Weekly report the same annualized return for the same underlying growth rate.

**Monthly is inherently data-starved**: SMA50/SMA200 need 50/200 *months* (4/16 years) of history, and there is no higher timeframe to align against, so Quality and Trend are scored from fewer inputs there. Daily remains the best-supported view.

### Removed: Max Buy Price

`core/entry_price.py` and `MaxBuyPriceCard.tsx` used to compute "the highest price you should pay today" from two rules: within 5% of the nearest support, and reward-to-resistance at least 2x the risk to the stop.

**It was removed because it systematically rejected breakouts.** Reward was computed as `nearest_resistance - current_price`, which is negative for any stock making new highs — so the strongest setups produced no computable reward and the card said "wait for a pullback". Measured across real EGX stocks: it said wait on **7 of 8** and never once said "OK to buy", contradicting the composite score's Buy rating on 4 of them. SWDY was trading 36% above its nearest resistance and the card wanted a pullback to 89.91 on a 126.60 stock scored 66 "Buy".

It also anchored to pivot lows sitting a median ~15% below price against a 5% cap — fixing that anchor (using SMA50 / recent swing low) helped but could not fix the reward side, because computing reward above the market means inventing a price target, which this app deliberately does not do. And it duplicated the Entry Zone card's job with different maths, so the two could disagree.

Do not reintroduce a single-number entry cap without solving the breakout case. The Entry Zone card (`core/levels.py`) is the supported surface for "is now a good time to buy, in what band, with what stop".

## Decision Framework for Beginners

The Learn page's "How to Take a Decision" section and the in-app signals both follow the same 6-step flow. Keep any new guidance consistent with this flow — it is the single source of truth:

1. **Check the macro.** What is the EGX30 trend? In bearish regimes, demand higher scores (≥ 70 instead of 60).
2. **Read the composite BREAKDOWN, not the number.** The category reasons are checkable facts; the blended score is not predictive (see the signal-band note). A low score is **not** a sell signal — historically the lowest-scoring EGX stocks bounced about as often as the highest.
3. **Check Risk-Adjusted.** Is annualized return comfortably above the ~19% policy rate? If not, be sceptical.
4. **Check Relative Strength.** Is the stock a leader (outperforming EGX30) or a laggard (underperforming by >10%)?
5. **Set the stop-loss BEFORE buying.** The house convention is **1.5× ATR below the nearest support** (`STOP_LOSS_ATR_MULTIPLIER` in `core/constants.py`) — anchored to support, not to your entry price, so the number is objective and computable before you buy. `levels.compute_entry_exit`, `entry_price.compute_max_buy_price` and the `atr_stop` signal all use that one constant. (The Portfolio add form no longer has a stop-loss field — see the Portfolio section.)
6. **Size the position at 5–10% max** per stock (2–3% for thin-liquidity / NILEX names).

## Portfolio Risk Metrics (`egx-api-be/app/routers/portfolio_analysis.py`)

- **Sharpe Ratio** — annualized, uses `risk_free_rate` from settings (default 19%)
- **Sortino Ratio** — downside-only variant
- **Max Drawdown** — with peak/trough dates and current drawdown
- **VaR 95% / CVaR 95%** — historical method
- **Correlation Matrix** — pandas `.corr()` on aligned daily returns
- **Monte Carlo** — **vectorized numpy**: `np.random.normal(mu, sigma, (1000, 60))`. Never use Python loops for paths. Returns percentile bands (p5/p25/p50/p75/p95) per day.
- **Avg Composite Score** — mean of per-holding composite scores

**Egypt context:** the CBE overnight deposit rate is **19.00%** (held 2026-08-20,
after 825bp of cuts from April 2025). Still very high globally, so Sharpe ratios
here look poor next to developed markets — always say so when explaining one.

**It was hardcoded at 25% until 2026-09-02**, i.e. ~600bp stale. That one number
is the Sharpe hurdle, the Sortino hurdle, the whole input to `score_risk_adjusted`
(13% of the composite) and the bar realized trades are graded against via
`beat_t_bill_count` — too high, and the app understates every Sharpe ratio and
fails trades that genuinely beat cash. `init_db` upgrades rows still holding the
stale 25 and ONLY those, so a deliberate admin value survives.

**The TEACHING layer kept the stale 25 for three more days**, and that is the
part worth remembering: fixing the constant did not fix the app. Six places in
`learn/curriculum.tsx` still told the reader T-bills pay 25% — including one
whose arithmetic was derived from it (*"gained 10% … you're 15% behind"*) — while
`StatsPanel` correctly interpolated `T_BILL_RATE_PCT`. Fixed 2026-09-05 by
interpolating the constant everywhere, the derived figure included
(`${T_BILL_RATE_PCT - 10}%`).

**Rule: a curriculum string that states a number the app computes MUST
interpolate the constant**, never restate it — the same rule already written down
for the Learn *widgets* ("widgets import their formula, never restate it"),
which the prose had simply never been held to. Note two `25%` occurrences in that
file are unrelated (a drawdown-recovery example, a position-concentration limit)
and must stay literal, so this is not a blind find-and-replace.

The same pass removed an **absolute valuation claim** from the `pe_ratio`
concept — *"Under 8 is genuinely cheap for this market"* — which asserts exactly
what `score_quality`'s own reason string is careful not to (it says *"Cheap
versus the EGX median of ~12"*). With the policy rate near 19% a low P/E does not
by itself mean the stock beats cash; the bands are a RELATIVE ranking and the
words now say so.

Caveat to repeat wherever it matters: this is the POLICY rate, not a 91-day
T-bill auction yield. There is no free machine-readable Egyptian T-bill series
(cbe.org.eg rejects automated requests), so the bill rate is approximated.

## Signals / Advice System

Returned in `signals` array from portfolio_analysis. Sorted by priority:

| Severity | Icon | Examples |
|----------|------|----------|
| `action_required` | `!!` | Stop-loss about to trigger, Death Cross, negative Sharpe, max drawdown > 20%, support broken, — |
| `warning` | `!` | Sector/stock concentration, high correlation pairs, OBV bearish divergence, near resistance, big loss, **`cash_underperformer`** (held >90d, ann. return < T-bill), **`relative_strength_laggard`** (alpha < -10% vs EGX30), **`mfi_extreme`** at >80, **`low_liquidity_warning`** (thin volume relative to the stock's own index), **`exit_zone_active`** at medium/high confidence, **`pe_overvalued`** (P/E ≥ 25 — expensive vs the EGX median of ~12), **`pe_loss_making`** (diluted EPS < 0), **`pe_implausibly_low`** (P/E < 3 — one-off earnings or a collapsed price, NOT a bargain), **`dividend_yield_extreme`** (DY ≥ 15%) |
| `opportunity` | `$` | Golden Cross, RSI/Stochastic oversold+bullish crossover, near support, target approaching, divergence_bullish, **`relative_strength_leader`** (alpha > +5% vs EGX30), **`mfi_extreme`** at <20, **`entry_zone_active`** at medium/high confidence, **`pe_undervalued`** (P/E < 8) |
| `info` | `i` | Beta context, ATR-based stop-loss suggestion, macro context, profit-taking reminder, **`very_strong_composite` / `strong_composite` / `weak_composite` / `very_weak_composite`** (condition readings — all `info` severity, never action_required, because the score is not predictive), **`adx_strong_trend`** (ADX > 30, direction from DI±), **`entry_zone_active` / `exit_zone_active` at LOW confidence** |

**One symbol, one voice.** The portfolio shows one card per stock, so the panel
beside it speaks once too. Two halves, and they are different problems:

- **Technical signals are deduped** by `(type, symbol)` in
  `dedupe_symbol_signals`. Two lots of one symbol are scored from ONE price
  series, so those signals come out character-for-character identical and
  keeping the first is lossless. It runs while `signals` holds only
  per-holding entries; signals with **no** symbol (sector concentration,
  drawdown, Sharpe) pass through untouched, or three sector warnings would
  fold into one.
- **Cost-basis signals are REBUILT from the position** by
  `build_position_signals` — `stop_breached`/`stop_loss`, `target_reached`/
  `target_hit`, `big_loss`, `profit_taking`, `cash_underperformer`. These are
  the ones whose duplicates disagree: a January lot up 30% and a June lot down
  20% are both true, and reporting either as "your position in X" is not. The
  percentage is cost-weighted, days run from the earliest lot, and stop/target
  come from the **binding** lot — the highest stop is the first price falls
  through, the lowest target the first it reaches. Deduping these instead
  would report one lot's loss as the position's.

`portfolio_metrics.num_holdings` counts distinct SYMBOLS for the same reason.

**Entry and exit zones are graded by confidence, both of them.** `low` is the
leftover bucket in `levels._entry_confidence` — price within 5% of a support
and momentum not overbought, but the support untested and RSI unremarkable.
Neither is a reason to buy; they are the absence of two reasons not to.

`sev = "opportunity"` was assigned UNCONDITIONALLY for entry zones, directly
beneath a comment reading *"low-confidence zones are hints, not calls to
action"* — so a bare hint rendered at the loudest non-alert tier the panel has,
while the exit branch eighteen lines below had always graded correctly. The
colour compounded it: the pill, the holding row's band and the stock page's
zone box were all gain-green regardless of confidence, which breaks the rule
that gain/loss carry a real direction and are never decoration. All three now
go neutral at `low`. `tests/test_fixes.py` fails the build on a bare
`sev = "opportunity"`.

New signal types added with the 8-category engine have their `learn_concept` anchors wired into the Learn page: `risk_adjusted_return`, `relative_strength`, `mfi`, `adx`, `liquidity`, `multi_timeframe`, `cash_underperformer`. Entry/exit zone signals use `entry_exit_zones`.

Each signal has a `learn_concept` key that links to a Learn page anchor (`id` attribute on a Concept card in `egx-api-fe/src/app/learn/page.tsx`). Deep links take the form `/learn#<concept>`.

## Frontend Components

Components in `src/app/components/`:

**Charts:**
- `PriceChart` — Recharts ComposedChart with SMA/EMA/Bollinger overlays, support/resistance/fibonacci ReferenceLines
- `VolumeChart` — BarChart colored by up/down
- `IndicatorPanel` — Tabbed panel with RSI, MACD, Stochastic, OBV subcharts
- `CompareChart` — Multi-series normalized comparison
- `MonteCarloChart` — AreaChart fan with 5 percentile bands
- `CorrelationHeatmap` — Full grid desktop, simplified pairs list mobile

**Composite Score:**
- `CompositeGauge` — Hand-rolled SVG semicircle gauge. Props: `score`, `signal`, `size ("sm"|"md"|"lg")`. Exports `scoreColor(score)` for use elsewhere. "sm" (40px) used as badges; "md" (96px) mid-size; "lg" (160px) hero. Pulses when score ≤ 20.
- `ScoreBreakdown` — 8 tappable category bars (score + weight% + expandable reasons). Gear button opens `ScoreWeightsModal`.

**Levels & Zones:**
- `KeyLevelsCard` — Displays nearest support/resistance from `AnalysisResponse.key_levels` with distance %, strength, and a "broken through" visual state when price has crossed a level. Used on the stock detail page above the price chart.
- `EntryExitCard` — Displays active entry/exit zones from `AnalysisResponse.entry_exit`. Shows price range, confidence tier, suggested stop-loss (entry only), and supporting reasons. Renders a "no active zone" empty state when neither is active. Used on the stock detail page. In `HoldingsTable`, a compact `ZoneBadge` pill surfaces the same state inline next to each holding; the expanded row uses `ZoneDetail` for full zone info.
- `ScoreWeightsModal` — 8 range sliders (0–50, step 5; 4 core + 4 behind an "advanced" toggle), normalized preview row, preset buttons rendered dynamically from the API's `presets` field (5 today), mobile full-screen / desktop card. Saves via `ScoreWeightsProvider`.
- `ScoreWeightsProvider` — React Context. Fetches weights once on mount, shares across all score-aware components. Exports `useScoreWeights()` hook. Increments `version` on save to trigger re-fetches in pages. Wrapped around `{children}` in `layout.tsx`.

**Portfolio views:**
- `PortfolioSummary` — Totals + avg composite score tile + sector allocation pie/stacked bar
- `RiskDashboard` — Sharpe/Sortino/MaxDD/VaR/Current DD grid
- `HoldingsTable` — Full table desktop (Score column with gauge + signal), cards mobile (gauge in card header + expanded detail). **Rows are POSITIONS, from `lib/positions.ts::groupHoldings`** — one per symbol, keyed by symbol; the purchases behind an average appear as `LotList` inside the expanded row, which is also where per-lot Edit/Delete live once there is more than one. Error rows use `colSpan={10}` plus a dedicated Actions cell, coordinated against the expanded detail row's `colSpan={12}` — both must total the table's 12 columns, and a mismatch between the two is a bug this project has already shipped once. The lot list goes INSIDE the existing expanded cell, so it adds no column. The symbol cell carries no badges (see *Portfolio*); the dividend total lives in the expanded detail, which also keeps it out of the colSpan arithmetic. `onSell` takes a **symbol** (the page resolves the lots), `onAddDividend` opens `AddDividendForm`.
- `MacroCard` — EGX30/USD-EGP/CBE rate indicator row
- `AdvicePanel` — Signals rendered with severity styles + learn links, and
  **collapsed behind one tap** the way `RealizedSection` is (a `<details>`,
  so it renders closed on the server and cannot flash open before
  hydration). A ten-holding portfolio makes twenty-odd signals, which is
  several phone screens between the holdings above and the risk metrics
  below. **The count line is what makes collapsing safe:** the summary
  carries a pill per severity in that severity's own colour, so "2 urgent"
  is on screen with the panel shut — the same trade `RealizedSection`
  makes by keeping its headline figure in the summary.
- `AddHoldingForm` — Full-screen modal on mobile, inline on desktop
- `SellHoldingForm` — Records a sell against a POSITION (`position={symbol, name, lots}`), up to its total shares across every open lot. Shows the FIFO split and sums the realized figure over those parts; `min` on the date picker is the newest lot the sale reaches, so it relaxes as the quantity falls back inside the older lot.
- `AddDividendForm` — Records a dividend payment (symbol, amount received, pay date, optional shares/notes) against a holding
- `RealizedSection` — **everything banked, in one collapsed section**, replacing `RealizedGainsCard` / `ClosedPositionsTable` / `DividendsTable` (all three deleted). The header is the combined gains + dividends headline, stated as two separate figures. Inside: the record/proceeds/best/worst row, the T-bill count, then tabs — **Closed** (rendered from `orders`, so one submit is one line; a `N lots` pill and a Show-purchases toggle expose the parts, each with its own basis and annualized figure), **Dividends**, **By stock**. The tab opened first is the first non-empty one. Undo on a closed row removes the WHOLE order and says how many purchases that is. Rows in all three tabs name the COMPANY beside the ticker — four opaque letters are not what a reader recognises — truncated rather than wrapped, and dropped entirely when the stored name is just the symbol again (`_bucket` in `core/dividends.py` falls back to it), which would otherwise render as "COMI COMI".

**Admin (admin role only):**
- `AdminUsersTable` — desktop table / mobile cards. Actions per user: reset password, disable/enable, delete. Hides the destructive actions on your own row (the backend guards them anyway).
- `CreateUserModal` — full-screen on mobile, card on desktop. Leaving the password blank is the intended path; the backend generates one.
- `PasswordRevealDialog` — shows a generated password ONCE and says plainly that it cannot be shown again. Used by both create and reset. **Copy puts the LINK, the username AND the password in one block** (`Link: …\nUsername: x\nPassword: y`) because it gets pasted into a single message; copying the password alone left the admin retyping the username into the same chat, and credentials with no address are a login the recipient cannot act on. A secondary **Copy password only** button sits below it for the WhatsApp case — the combined block arrives as one message, so a recipient long-pressing it copies all three lines; sending the bare password as its OWN message lets them copy it in one tap. The combined block stays the PRIMARY action, for the reason just given. The on-screen block mirrors the copied text exactly, so what is sent is what was checked. The link comes from **`FE_BASE_URL`** — see *Frontend environment variables*.
- Page at `src/app/admin/page.tsx`, gated on `isAdmin` from `useAuth()`.

**Learn page (`src/app/components/learn/`):**
- `visuals.tsx` — the SVG diagram library. `MiniChart` (price path + overlay
  lines + volatility band + horizontal rails + markers + highlight regions),
  `ZoneScale`, `BarCompare`, `StepFlow`, `AllocationDonut`, `CorrelationGrid`,
  `ConeChart`, `LedgerRows`, plus `walk()`/`smaOf()` for seeded illustrative
  series. No charting dependency — `/learn` loads no Recharts.
- `widgets.tsx` — `RsiPlayground`, `StopLossCalculator`, `PositionSizer`,
  `TBillRace`, `ScoreBandExplorer`. Each imports its formula from
  `lib/constants.ts` rather than restating it.
- `LiveChart.tsx` — real COMI data in a concept card. One shared fetch, lazy via
  `IntersectionObserver`, silently falls back to the caller's static SVG.
- `ConceptCard.tsx` — anchor target, module-hued rail, visual slot, definition,
  why/how blocks, worked example, mark-as-read toggle.

**UI helpers:**
- `Navbar`, `BottomTabBar` — mobile bottom nav, desktop top nav. **Neither carries a "Users" destination any more.** Administering accounts is not a place in the app, it is an account action, so it renders as a button beside Log out at the right-hand end of `Navbar` — icon-only on mobile where it pairs with the log-out icon, icon + label at `md:`, the same shape the dashboard's Compare button uses. It is **36px tall, not the project's usual 44px target**, deliberately: the nav row is 61px and `--top-nav-clearance` IS that number, so a taller control here silently pushes every sticky element on every page down out of alignment. **The two navs no longer carry the same destinations:** Compare left the mobile pill for a button in the dashboard header (see *Dashboard*) and stayed in the desktop top nav, which has room for it. So the pill is Dashboard / Portfolio / Learn / News **for everyone, admin or not** — it holds the places every user has, and its tab count no longer changes with a role. On `/admin` it shows no highlight at all (`activeIndex` is -1, so the travelling span is not rendered), which is correct for a page it does not contain.
- `AuthProvider` — token + user in localStorage, an `egx.auth.present` cookie for middleware, `useAuth()` → `{user, isAuthenticated, isAdmin, login, logout}`. Re-reads the role from `/api/auth/me` on every load, so a role change lands on next refresh.
- `LearnTooltip` — dashed-underline hover tooltip used everywhere for inline education
- `LoadingSkeleton` — Card/Chart/Table skeletons
- `StockCard`, `Watchlist`, `IndexFilter`, `SectorFilter`, `StatsPanel`

## Mobile-First Conventions

- Breakpoint: `md:` (768px) is the main one
- Bottom nav bar (`BottomTabBar`) visible only on mobile with `md:hidden`

### `--bottom-nav-clearance` — the one spelling of the nav's footprint

Anything that must sit clear of the bottom nav reads the
`--bottom-nav-clearance` CSS variable from `globals.css`:
`calc(env(safe-area-inset-bottom) + 70px)` on mobile, `0px` at `md:`. That is
the pill's 52px height, the 8px it floats clear of the safe area, and 10px of
breathing room. Consumers today: the `layout.tsx` footer, the portfolio FAB and
the admin FAB.

**It exists because three hardcoded copies of that number drifted apart.** The
footer had `pb-[60px]`, the portfolio FAB had
`calc(env(safe-area-inset-bottom) + 76px)` and the admin FAB had a bare
`bottom-[76px]` — **no safe-area term at all**. Measured on a 34px home
indicator: the nav's top edge sat 94px up and the FAB's bottom edge at 76px, so
**18px of the + button rendered behind the nav** (which is `z-50` against the
FAB's `z-40`). The emulator reports a 0px inset, so this is invisible in a
desktop browser and only shows on a real phone. Change the pill's height or
offset and change the variable in the same breath.

### `--top-nav-clearance` — the same lesson, at the other end

`Navbar` is `sticky top-0 z-50` and carries **no `md:` visibility class**, so it
covers the top of the viewport at every screen size. Anything that sticks
beneath it reads `--top-nav-clearance`:
`calc(env(safe-area-inset-top) + 61px)`.

**Both sticky search bars were parking underneath it.** They were sticking
correctly at `top-0` the whole time — behind an opaque nav at a higher
z-index — so they simply vanished on scroll and read as broken. The dashboard's
was worse than Learn's: a hardcoded `top-[56px]`, 5px short of the real 61px
**and with no safe-area term at all**, so on a notched phone its top edge sat
under the nav exactly the way the admin FAB once sat under the bottom one.

Measured: nav bottom edge 61px, sticky top after scrolling 61px, clears.

**A THIRD consumer had the identical bug and was missed both times** — the
stock detail page's mobile sticky header (`stock/[symbol]/page.tsx`) carried the
same hardcoded `top-[56px]` with no safe-area term, on the route every dashboard
card links to. Fixed 2026-09-05. The lesson is that finding two instances of
this is not evidence you have found them all: **grep `sticky` across
`src/app` and check every offset**, which is how the third was caught. The sweep
is now clean — the only other sticky offsets are `LearnClient`'s desktop sidebar
(`top-24`) and the dashboard's desktop watchlist rail (`lg:top-[72px]`), both
`md:`/`lg:`-gated and both larger than 61px, so they clear the nav by
construction.

### The bottom nav is a floating pill, not a bar

A centred capsule (`h-[52px]`, `rounded-full`, `bg-charcoal/85`,
`backdrop-blur-xl`), detached 8px from the bottom safe area — the shape the
newer Instagram builds use. The `<nav>` spans the screen only so the pill can
centre itself and is `pointer-events-none`, with `pointer-events-auto` on the
pill: **the transparent gutters either side stay tappable, and content
genuinely scrolls past the pill there** rather than being hidden behind a solid
bar. **The pill is FLUID** — `w-full max-w-[320px]` with `flex-1` tabs, not
sized to its own content, so the tabs grow with the screen rather than the
gutters doing it. Each tab is a 44px-tall `rounded-full` target — the project
minimum exactly, and the floor on any further shrinking — carrying a 21px icon
and a 10px label.

**The size has been moved three times; the cap is what settles it.** Content-
sized at ~276px it was 63% of a 440px iPhone 16 Pro Max and read as undersized;
`max-w-[430px]` then made it 93% of a 375px screen and read as *very big*. At
`max-w-[320px]` it is 85% of a 375px phone and 73% of a 440px one — measured
320px wide, 52px tall, four 44px tabs. Note `w-full` beats the cap on small
screens: a cap above ~343px does nothing at all on a 375px phone, which is why
raising it had no effect there and lowering it does.

#### The active highlight SLIDES, and it is one element

`BottomTabBar` renders a single absolutely-positioned `<span>` that travels
between tabs (`transform 340ms cubic-bezier(0.34, 1.4, 0.5, 1)`), rather than
toggling a background on each tab. A per-tab background can only cross-fade,
which reads as two things blinking; one travelling pill reads as the selection
physically moving. The icon also scales to 1.08 and lifts 1px as its tab
becomes active, so the motion is not purely horizontal. Both honour
`motion-reduce`.

Its geometry is **measured, never computed from the tab count** — the tabs are
`flex-1` inside a fluid pill, so their width depends on the screen. Three
things about that measurement are load-bearing, and each was a bug first:

- **Query the anchors, not `rail.children`.** The highlight is itself a child,
  so indexing `children` shifts every tab by one the moment it mounts.
- **`isAuthenticated` and the tab count belong in the effect deps.** The bar
  returns `null` while auth loads, so the first pass measures a rail that does
  not exist; without those deps nothing re-runs when the real bar appears and
  the highlight never renders at all.
- **"Have we placed it yet" must be STATE, not a ref.** A ref mutation does not
  re-render, so the element kept the `transition: none` it was first painted
  with and the highlight *teleported* between tabs. It measured as
  `transitionDuration: 0s` while the transform updated correctly — that pair is
  the signature of this bug.

Verified through the Web Animations API rather than by sampling frames:
`getAnimations()` reports a running `transform` transition of 340ms on the
right easing, and the element sits mid-flight 100ms in. **`requestAnimationFrame`
does not tick in a hidden preview pane**, so frame-sampling silently returns
zero frames and proves nothing — use `getAnimations()`.
- Tables → cards on mobile (`space-y-3 md:hidden` + `hidden md:block` pattern)
- Forms → full-screen modal on mobile, inline on desktop
- Touch targets: `min-h-[44px]` minimum
- Safe areas: `env(safe-area-inset-top/bottom)` on navbar/footer
- Horizontal scrolling for filter pills: `overflow-x-auto no-scrollbar`
- Charts wrapped in Recharts `ResponsiveContainer`

### Every route segment has a `loading.tsx`, and must keep one

The app had **none**, anywhere. In the App Router a segment without one has no
Suspense boundary, so the router will not commit a navigation until the target
page's whole tree is ready: the OLD page stays on screen and a tap on the
bottom nav reads as dead. On a phone that is the difference between "thinking"
and "broken", and it is the one a user retaps.

Measured after adding them: a nav tap commits in **11-32 ms** on a compiled
route, with the skeleton streaming in behind it. (The first hit of a route in
`next dev` still pays ~1.2 s to compile it — that is the dev server, not the
boundary.)

Each mirrors its own page's chrome so the transition is a fill rather than a
flash of unrelated layout, and each reuses `CardSkeleton` / `ChartSkeleton` /
`TableSkeleton` rather than inventing a new one. `/stock/[symbol]` matters
most: every dashboard card links there and that page pulls 400 bars and scores
them.

**Add a route, add its `loading.tsx` in the same breath.**

### Money inputs use `step="any"`, never `step={0.01}`

`<input type="number">` validates its value against the step, so `step={0.01}`
makes any third decimal a `stepMismatch` — and **EGX quotes to three decimals**
(CIEB at 27.697). All three money fields shipped that way, and the browser
rejected real prices with its own "Enter a valid value", truncated on mobile to
give no reason at all. Confirmed: at `step=0.01` the browser reports *"the two
nearest valid values are 27.69 and 27.7"*.

The dividend `amount` is the most exposed of the three — it is the total that
actually landed after the 5-10% withholding tax, so it is routinely not a round
two-decimal figure.

Quantity and dividend `shares` deliberately carry NO step, so they default to
integers. EGX has no fractional shares.

## Styling

Tailwind with custom colors (`tailwind.config.ts`):
- `charcoal-dark: #0a0a0f` — page background
- `charcoal: #12121a` — card background
- `gain: #00ff88` — green
- `loss: #ff3355` — red
- `accent: #4488ff` — blue

Fonts: Outfit (sans), JetBrains Mono (mono).

## Database Schema (`app/core/db.py`)

```sql
users (id, username UNIQUE, password_hash, created_at, role, is_active)
                 -- role: 'user' | 'admin'. Stamped from the AUTH_ADMINS env
                 -- var at boot and NEVER written by an API route.
                 -- is_active: FALSE blocks login AND invalidates any token
                 -- already issued, because get_current_user re-reads this row.
                 -- Both added by ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
user_settings (user_id, key, value, PRIMARY KEY (user_id, key))
                 -- Per-user overrides of the `settings` keys that are a
                 -- PREFERENCE rather than a fact — today only weight_*.
                 -- A separate table, not a user_id column on `settings`:
                 -- changing that PK is not idempotent in Postgres, and keeping
                 -- `settings` as the global tier makes the migration free.
portfolio (id, symbol, name, buy_price, buy_date, quantity, notes, sector,
           target_price, stop_loss, created_at, updated_at)
portfolio_sales (id, user_id, holding_id, symbol, name, sector, quantity,
                 buy_price, buy_date, sell_price, sell_date, notes, created_at,
                 sale_group_id)
                 -- Cost basis is SNAPSHOTTED, not joined: a sale is a
                 -- historical fact and must not change when the holding it
                 -- came from is later edited or deleted.
                 -- sale_group_id -> shared by every row ONE submit wrote, so a
                 --   sale spanning two purchase lots reads as one order in the
                 --   ledger and undoes as one. NULL on rows written before the
                 --   column existed; those read as their own single-part
                 --   order. Added by ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
                 -- portfolio.quantity = 0 means fully sold; the row is kept
                 -- as the undo anchor and filtered out of every read by
                 -- core/holdings.fetch_open_holdings.
portfolio_dividends (id, user_id, symbol, name, sector, amount, pay_date,
                     shares, notes, created_at)
                 -- Cash the company paid for holding it. Anchored to the
                 -- SYMBOL, deliberately with NO holding_id: a dividend
                 -- restores nothing on undo, so the column would buy no
                 -- behaviour and would cost correctness — deleting a holding
                 -- would destroy the record of money genuinely received.
                 -- amount   -> total EGP that ACTUALLY LANDED, already net of
                 --             the 5-10% withholding tax. Never gross.
                 -- shares   -> optional, display only. amount is never
                 --             derived from it.
                 -- An exact symbol+pay_date+amount repeat is rejected 409.
settings  (key, value)   -- pre-seeded: currency, risk_free_rate,
                         --             weight_trend, weight_momentum, weight_volume,
                         --             weight_volatility, weight_divergence,
                         --             weight_quality, weight_risk_adjusted,
                         --             weight_relative_strength,
                         --             pe_last_successful_fetch, pe_last_attempt_status
watchlist (symbol, added_at)                                     -- user's watched tickers
macro_data (key, value, previous_value, change_pct, updated_at)  -- 1-hour cache
pe_data    (symbol PK, company_name, pe_ratio, dividend_yield, loss_making,
            market_cap, shares_outstanding, beta_1y, value_traded_egp,
            dividend_ex_date_recent, dividend_amount_recent, updated_at)
                         -- Nightly refresh from the TradingView scanner via
                         -- core/pe_fetch.py. `symbol` is the exact ticker from
                         -- the feed — no name matching involved.
                         -- pe_ratio IS NULL  -> no trailing P/E (usually a loss-maker)
                         -- dividend_yield 0  -> REAL: pays nothing. Only NULL is unknown.
                         -- loss_making       -> from diluted EPS; the feed never
                         --                      reports a negative P/E.
                         -- dividend_ex_date_recent -> TEXT ISO of the last coupon's
                         --   ex-date (~34% coverage = the payer population). Feeds
                         --   the dashboard "Recently paid" sort, /api/dividend_calendar,
                         --   and the portfolio "last market dividend" line. Per-stock
                         --   HISTORY is Yahoo (core/dividend_history), not here.
```

```sql
dividend_events (symbol, ex_date, amount, source, created_at,
                 PRIMARY KEY (symbol, ex_date))
                 -- Persisted dividend history, append-only and idempotent by PK.
                 -- Seeded deep from Yahoo (scripts/backfill_dividends), grown
                 -- forward nightly by refresh_pe_data appending the scanner's
                 -- latest coupon. Read by /api/dividend_history (self-heals from
                 -- Yahoo on a miss) and /api/dividend_calendar (latest per symbol).
```

```sql
market_regime (id, observed_at, mean_score, n_symbols, band)
risk_snapshot (symbol PK, measured_at, sigma_63_ann_pct, sigma_ewma_ann_pct,
               beta, turnover_egp, traded_share, last_price, tradeable)
               -- Current-value read model, refreshed in CHUNKS by
               -- POST /api/cron/risk_snapshot and ranked cross-sectionally
               -- at READ time. No run to finalize and no cursor state that
               -- can corrupt, so a half-finished refresh still yields sane
               -- percentiles. measured_at is PER ROW, which is what lets the
               -- read path report its stalest corner instead of passing a
               -- partly-yesterday snapshot off as today's.
```
Append-only log of market-condition readings, so the card can show this
morning's reading rather than "no data" when the score cache is cold.

`loss_making` is added by an `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in
`init_db`. There is no migration framework — every statement there is
idempotent, so new columns land on the next cold start of any process.

### Fundamental factors: TESTED, and none of them ship

`scripts/factor_backtest.py` is the first honest test of a fundamental factor
this project could run, because `fundamentals_annual` finally supplies dated
figures. **The answer was no, and that is the result — do not re-propose these
without new data.**

Two gates ran first, and both had to pass before any number below counted:
a **placebo** (shuffled factor) at IC −0.0037, t=−0.39, and a **positive
control** — low volatility, already known to score here — recovering
IC +0.0646, t=3.54. A harness that cannot find a signal it has already found
by another route makes a null result uninformative.

Panel: 13,087 rows, 193 symbols, 148 dates, median 76 names/date, liquid cut.

| factor | 21d IC | t | verdict |
|---|---|---|---|
| *control: low volatility* | *+0.0646* | *3.54* | *(control)* |
| earnings yield (E/P) | +0.0419 | **3.45** | passes raw — then see below |
| gross profitability (GP/A) | +0.0174 | 1.12 | no, and SIGN FLIPS across halves |
| asset growth | +0.0080 | 0.66 | no, and SIGN FLIPS |
| payout yield (DPS/P) | +0.0252 | 1.75 | no |

**Earnings yield cleared the bar and still does not ship, because it is not
NEW.** Cheap EGX stocks are also calmer ones — within-date rank correlation with
the control is **+0.228** — and residualising earnings yield on low volatility
drops it from t=3.45 to **t=2.00**, below the pre-registered |t| > 3.0.
Raw significance was real; INDEPENDENT significance was not. Payout yield is
worse: residualised it goes NEGATIVE (−0.0150), so it was a low-vol proxy
outright.

**The orthogonality test is why an app does not end up shipping the same number
twice.** Any new factor must survive being residualised on what is already
ranked on — today, the Risk Grade's volatility percentile.

**Only the 21-day column is trustworthy for significance.** Rebalance dates sit
~22 trading days apart, so 63d and 126d forward windows overlap 3x and 6x; their
t-stats (4.94 and 5.55 for earnings yield) are inflated and are reported for
shape only. This is the same overlap error that produced the +0.318 regime claim.

This matches the strongest external prior: Zaremba's frontier study (>4,500
stocks, 22 countries) found value and momentum but **no consistent profitability
or investment premia** — which is exactly the pattern above.

### fundamentals_annual — twenty years, from a call we already make

```sql
fundamentals_annual (symbol, fiscal_year, eps_diluted, dps, net_income,
                     revenue, total_assets, gross_profit, total_debt,
                     first_usable_date, updated_at,
                     PRIMARY KEY (symbol, fiscal_year))
```

**This is what makes a fundamental factor testable at all.** `scripts/backtest.py`
refuses to score fundamentals in its own docstring, and correctly: `pe_data` is a
current-value snapshot every refresh destroys, and `fundamentals_history` only
starts 2026-08-25.

The TradingView scanner the nightly cron already calls also returns `*_fy_h`
history arrays — the whole annual series per company, aligned element-wise to
`fiscal_period_fy_h`. **Verified live: 246 of 296 EGX rows (83%), 114 symbols
with a full 20 years, 2,555 records after filtering.** It rides along on the
existing `/api/pe/refresh` slot as a second POST to the same host: no new cron
entry, no new vendor. Its failure is swallowed — `pe_data` is what the app
serves and must not go down with an archive only a backtest reads.

**`first_usable_date` = fiscal year end + 120 days is the look-ahead guard, not
decoration.** A fiscal year's figures were not knowable on 31 December of that
year, and treating them as if they were lets any factor appear to work. Read
through `get_annual_asof(db, as_of)`, which enforces it; querying the table
directly in a backtest bypasses every guard in the module.
`tests/test_fundamentals_annual.py` pins the filter, the array alignment, and
the truncated-response refusal.

**Two limits that cannot be engineered away — state them wherever results are:**
- **The arrays are as-RESTATED, not as-first-reported.** No publish date exists
  at any spelling for EGX, so a fixed lag is the only available defence and
  residual restatement bias survives it. A company that later revised a bad year
  looks better here than it did to an investor at the time.
- **Pre-2012 is refused at ingest.** Measured symbols with a usable diluted EPS
  per fiscal year: 2018 → 225, 2015 → 207, 2012 → 160, then a cliff —
  2011 → 63, 2009 → 52, 2007 → 23, 2006 → 15. The rows reach further back
  than the data does, and including them would weight a cross-sectional test
  toward whichever handful of large caps happened to report.

Note the current fiscal year is thin by construction (2025 → 100 symbols) because
companies have not all reported; `first_usable_date` handles that without a
special case.

**The archive is written in BATCHES — `_upsert_batch` at `WRITE_BATCH` (250),
never a per-row loop.** `refresh_annual_fundamentals` did exactly that loop until
2026-09-05: the live archive is ~2,555 records, so the nightly refresh issued
~2,555 sequential Neon round trips from inside a request with a 30-second
ceiling, on top of the ~130 `refresh_pe_data` already makes. That is the same
shape that cost the FX backfill two of its three attempts (see *macro_series*) —
one statement per row over a pooled connection either crawls or gets cut off
partway. Batched it is ~11 round trips.
`tests/test_fundamentals_annual.py::test_the_archive_is_written_in_BATCHES_not_one_row_at_a_time`
counts round trips and fails if the loop returns.

Two properties of that write are load-bearing:
- **Dedupe on the PK before batching.** Postgres refuses an `ON CONFLICT DO
  UPDATE` that would touch the same row twice in one statement, so a feed
  repeating a `(symbol, fiscal_year)` aborts the whole batch — a failure per-row
  upserts never had to care about. Same lesson as `macro_series.upsert_many`.
- **There is deliberately NO wall-clock deadline here**, unlike
  `/api/cron/risk_snapshot`. That job measures independent symbols and stopping
  early is free because selection is stalest-first. This one writes a coherent
  archive, and the `MIN_EXPECTED_SYMBOLS` guard already refuses to write at all
  rather than write part of the universe — a deadline that truncated the write
  would reintroduce exactly the half-written state that guard exists to prevent.
  Batching removes the need for one.

### fundamentals_history — the point-in-time record

```sql
fundamentals_history (id, symbol, observed_at, eps_ttm, dps_annual,
                      book_value_per_share, loss_making, close_at_observation)
```

`pe_data` is a **current-value read model**: every refresh overwrites it, so
yesterday's numbers are destroyed nightly. This table is the append-only log
that makes historical questions answerable at all. Without it the valuation
bands can never be validated, because scoring a past date with today's P/E is
look-ahead bias severe enough to manufacture any result.

**It stores fundamentals, not ratios — deliberately.** P/E, P/B and dividend
yield all divide by PRICE, so they move every single day and a log of them would
be ~99% price noise. Verified against the live feed: `close / eps_ttm`
reproduces the reported P/E exactly. EPS, DPS and book value move quarterly, so
the log stays tiny, and any ratio is reconstructable at a past date as
`(historical close from egxpy) / (the fundamental in force then)`.

It also reaches further than `pe_data` does: **book value per share covers 60%
of EGX stocks against P/E's 22%**, because banks and real estate dominate this
market and both are conventionally valued on book. First seed: 182 symbols.

**Rules:**
- Rows append **only when a fundamental actually changes**
  (`pe_fetch._changed`, relative-tolerance float compare so the feed's nightly
  last-decimal jitter doesn't append). Verified live: 182 rows on the first run,
  **0** on an immediate second run with identical data.
- **Read historical values through `get_fundamentals_at(db, symbol, as_of)`** —
  latest row with `observed_at <= as_of` — or `get_fundamentals_asof_all(db,
  as_of)` for a whole date in one query. Never use `get_pe_for_symbol` to
  evaluate a past date; it returns today's snapshot. `_latest_history` is for
  change detection only.
- History failures are swallowed: the current-value feed is what the app serves
  and must not go down with the log.

**Two caveats:** `observed_at` is when *we* observed the change, not when the
company reported it — the nightly cron bounds that lag to a day. And there is no
history before 2026-08-25, so anything validating the valuation bands can only
run forward from then.

Connection singleton in `_db.py` with lazy init. `get_db()` returns the ready connection.

## Caching Strategy

- **Serverside:** `api/_cache.py` — module-level dict, 5-min TTL, keyed by `endpoint:params`
- **Survives warm container**, cleared on cold start
- **Turso cache:** `macro_data` table for 1-hour macro data cache
- **EGX30 data:** Cached under dedicated key (reused across many stock analyses)
- **Composite score cache invalidation:** `analysis.py` and `portfolio_analysis.py` include `weights_hash(weights)` in their cache keys — changing weights via PUT /api/weights automatically busts cached scores

## Common Tasks

**Adding a new indicator:**
1. Write function in `api/_indicators.py` with detailed docstring
2. Add to `compute_all()` return dict (for array-based indicators)
3. Extend `AnalysisIndicators` interface in `src/app/lib/types.ts`
4. Add to `priceChartData`/`indicatorData` useMemo in stock detail page
5. Add chart rendering in `IndicatorPanel.tsx` or `PriceChart.tsx`
6. Add a Concept card to `src/app/learn/page.tsx`

**Adding a new signal type:**
1. Add rule in `_analyze()` method in `portfolio_analysis.py`
2. Use `"action_required" | "warning" | "opportunity" | "info"` severity
3. Include `learn_concept` string pointing to a Learn page anchor
4. Add an explanation sentence — target audience is a beginner

**Adding a new portfolio metric:**
1. Compute in `portfolio_analysis.py` after per-holding loop
2. Add to `portfolio_metrics` dict in response
3. Extend `PortfolioMetrics` interface in types.ts
4. Render in `PortfolioSummary.tsx` or `RiskDashboard.tsx`
5. Add Learn page concept

**Changing composite score logic:**
- Category scorers are in `api/_composite.py` — pure functions, safe to edit in isolation
- Adding a new scoring input: add the value to the `extras` dict passed from `analysis.py` / `portfolio_analysis.py`, then consume it in the relevant `score_*` function
- Changing weight defaults: update `DEFAULT_WEIGHTS` in `_composite.py` and the `INSERT OR IGNORE` seeds in `_db.py`

## Watchlist

Synced to Turso via `/api/watchlist` and exposed through `WatchlistProvider` (wrapped in `layout.tsx`). The `useWatchlist()` hook returns `{ symbols, loading, add, remove, has }` — shared across all consumers, so add/remove on one page updates everywhere. Adds/removes are optimistic and roll back on API failure. On first mount the provider migrates any legacy `egx-watchlist` localStorage entries to the DB and clears the key.

## Things to Know

- **EGX30 symbol:** use `get_OHLCV_data("EGX30", "EGX", ...)`. Wrap in try/except — some exchanges/intervals may not have it.
- **SMA 200 needs 200+ bars:** `analysis.py` fetches `max(bars, 400)` internally and trims output
- **Monte Carlo MUST be vectorized:** `np.random.normal(mu, sigma, (n_sims, n_days))` — never loop
- **All P&L in EGP.** Egypt's currency is the Egyptian pound.
- **EGX trading hours:** Sun–Thu, 10:00 AM – 2:30 PM Cairo time. EGX30 only updates during these hours.
- **T+2 settlement:** mentioned on Learn page but not enforced in portfolio logic
- **Auth:** every router scopes its queries by `user.id` from a JWT Bearer token (`app/routers/portfolio.py` and the rest); `app/main.py` calls `seed_users_from_env` to provision users.
- **The app is closed — see *The app is CLOSED*.** A new router is denied by default; opening it means editing `PUBLIC_ENDPOINTS` in `core/auth.py`, and `tests/test_auth_gate.py` will tell you if you leave something open by accident.
- **Pushing: switch to the `MarkBotros0` GitHub account first.** Both repos are
  owned by `MarkBotros0`, but the machine has several `gh` accounts
  authenticated at once and the active one is often `mark-aigorithm`, which
  gets `remote: Permission to MarkBotros0/egx-api-be.git denied` and a 403.
  Run `gh auth switch --user MarkBotros0` before `git push`. Check with
  `gh auth status` — the account marked `Active account: true` is the one git
  will use.
- **Composite score is educational only:** always note the disclaimer when surfacing scores to users — it does not predict future price; no fundamentals or news are considered.
- **Vercel timeout budget:** all three scoring paths now use identical inputs (see *One score per stock* below), so the levers are `BATCH_WORKERS` and the `include_multi_timeframe=False` escape hatch on `build_composite_extras` — **not** shrinking a single path's window, which is what caused the scores to diverge in the first place.
- **Missing data / short history:** `compute_composite` renormalizes weights across only available categories when a scorer returns None — scores on NILEX tickers with <50 bars will have reduced category coverage.
- **A literal `%` in SQL MUST be written `%%`.** `_DB.execute` always passes a params tuple, so psycopg parses every query for placeholders and a lone `%` raises `ProgrammingError: only '%s', '%b', '%t' are allowed as placeholders`. This is not hypothetical: `LIKE 'weight_%'` in `get_weights_from_db` raised on **every** call and the `except Exception: return DEFAULT_WEIGHTS` around it swallowed the error, so saved composite weights were never read back — every score in the app was computed at Beginner Safe defaults no matter what the sliders said, from the day the weights modal shipped until 2026-09-01. `tests/test_users_and_roles.py::test_no_sql_has_an_unescaped_percent` walks the AST of every `execute()` call and fails on a bare `%`.
- **A bare `except` around a DB read hides exactly this.** When a fallback is silent, a query that never once succeeded looks identical to one returning defaults. Prefer letting it raise, or assert the happy path in a test.

## Not Present / Deliberately Missing

- **Any public/anonymous surface.** No landing page, no public dashboard, no demo mode, no read-only guest. Signed out, the only thing that exists is the login form.
- Real-time streaming quotes (egxpy is polling, not streaming)
- Order placement (this is analysis-only; user trades through Thndr app separately)
- External TA libraries (ta-lib, pandas-ta) — everything is from-scratch for learning
- Per-stock composite score on the dashboard (StockCard accepts the prop but dashboard doesn't batch-fetch scores — out of scope)
- A single "max buy price" number (removed — see *Removed: Max Buy Price*)
- **Self-service password change.** Only an admin can set or reset a password; there is no "change my password" screen and no forced change on first login.
- **Role editing in the UI.** Admin status is `AUTH_ADMINS` only — deliberate, see *Auth, roles and user management*.
- **Per-user `risk_free_rate` or `currency`.** The T-bill rate is the Sharpe hurdle, the CBE policy rate on the macro card, AND the bar realized trades are graded against — a market fact, not a preference. Per-user values would mean each user grading their own trades against a different bar.

## Starting Points for Common Questions

- "Why is the stock detail page slow?" → `api/analysis.py` fetches 400 bars + weekly data; check cache hit
- "Why is portfolio analysis timing out?" → sequential `get_OHLCV_data` calls per holding; 10+ holdings = tight
- "Where are user settings stored?" → `settings` table in Turso, accessed via `/api/settings`
- "Where are composite weights stored?" → same `settings` table, keys `weight_*` (now 8 keys); accessed via `GET/PUT /api/settings?section=weights` in `app/routers/settings.py`
- "How do I show a tooltip on a new metric?" → wrap label with `<LearnTooltip term=... explanation=...>`
- "How is mobile different from desktop?" → `md:` breakpoint, bottom tab bar, card layouts, full-screen modals
- "Why did composite scores change after I hit save on weights?" → `weights_hash()` in cache key causes old entries to miss; fresh fetch re-computes with new weights
