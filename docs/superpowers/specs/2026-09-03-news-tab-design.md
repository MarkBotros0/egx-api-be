# News Tab — Design Spec

**Date:** 2026-09-03
**Status:** Approved for planning
**Repos:** `egx-api-be` (Python/FastAPI), `egx-api-fe` (Next.js)

## Problem

The app knows a great deal about what a stock's *price* has done and nothing at
all about *why*. There is no news anywhere in either repo — a grep for "news"
across `egx-api-be/app` and `egx-api-fe/src` returns only incidental matches in
comment prose.

That gap bites hardest on corporate actions. The user records dividends by hand
into `portfolio_dividends`, and nothing in the app tells them a coupon was
declared, so the ledger depends on them noticing elsewhere. The composite score
carries an explicit disclaimer that "no fundamentals or news are considered";
this does not change that, but it does put the missing half on screen beside it.

## What This Is Not

**It is not a scraper for egx.com.eg.** The exchange sits behind an F5 bot
challenge (`APM_DO_NOT_TOUCH`): `WebFetch` returns `ECONNRESET`, and a scripted
GET of `/en/rss.aspx` returns HTTP 200 with a 70 KB JavaScript challenge shell
rather than content. The site renders only in a real browser. This is the same
wall that made the old `egx.com.eg/en/MarketPECompanies.aspx` P/E scraper never
once succeed in production — see *The source is no longer egx.com.eg* in
CLAUDE.md. **Do not re-propose fetching EGX from a serverless function.**

**It is not a snapshot table.** `pe_data`, `risk_snapshot` and the dashboard
snapshot all exist because their upstream is slow enough that on-demand fetching
cannot fit a request. This source is not. See *Why on-demand* below.

**It is not an article reader.** The app stores and renders headline metadata
only, never body text. See *Copyright*.

**It is not a signal.** Nothing here feeds the composite score, the risk grade,
or any ranking. The project's bar for a new ranked input is |t| > 3.0 after
residualising on what is already ranked on (see *Fundamental factors: TESTED*),
and no such work has been done for news. It is context, in the same standing as
`MarketRegimeCard` and `BreadthStrip`.

## Source — chosen by measurement

Seven candidate sources were probed with stdlib `urllib` only, which is what a
server-side fetch actually has:

| source | result |
|---|---|
| **TradingView news v2** | **200, JSON, 98 items for `EGX:COMI`** |
| TradingView news v3 | HTTP 405 |
| Mubasher EN / AR RSS | HTTP 404 / 404 |
| Arab Finance | HTTP 410 Gone |
| Ahram business RSS | HTTP 404 |
| Enterprise Press, Zawya | HTTP 200 but HTML, no feed |
| Daily News Egypt RSS | 200, valid RSS, 10 items |
| egx.com.eg | 200, bot-challenge shell |

The winner is `https://news-headlines.tradingview.com/v2/headlines?client=web&lang=en&symbol=EGX:<SYM>`
— keyless, JSON, and the **same vendor the app already depends on** for the
scanner (`core/tradingview.py`) and, through the vendored `egxpy`/`tvDatafeed`,
for prices. No new vendor relationship.

Content is genuinely EGX-specific. `EGX:COMI` returns 98 stories from Reuters
(50), Zawya (37) and LSE (11), spanning 2025-06-11 → 2026-08-31: block trades,
CBE approvals, condensed financial statements, "CIB Bonus Shares Distribution".
Each item carries `relatedSymbols`, so a story maps onto holdings.

### `market=egypt` does not work — do not use it

`?category=stock&market=egypt` returns HTTP 200 with 200 items and **zero EGX
symbols**: the sample was Tesla, Santander, BRP and easyJet. The market filter
is silently ignored and the global stock feed is served instead. The market half
of this feature is therefore built by fanning out over EGX30 constituents and
deduping, not by a market-wide endpoint. A future contributor will find this
parameter in the API and assume it works; it does not.

### The browser cannot fetch this — it must be server-side

```
news-headlines.tradingview.com   Access-Control-Allow-Origin: (absent)
scanner.tradingview.com          Access-Control-Allow-Origin: <reflects Origin>
```

The scanner allows cross-origin reads; the news host does not. A client-side
fetch of news is blocked by the browser outright. This is not a preference, it
is the reason the fetch lives on the server.

## Why on-demand, and not a cached table

Measured, 24 EGX symbols through a thread pool at 8 workers:

```
wall clock       1.30 s for 24 symbols
per-symbol       min 0.25 s   median 0.37 s   max 0.73 s
outcome          22/24 with news, 2 empty, 0 failed
```

Against egxpy's ~1.4 s for a healthy symbol and ~6 s for a refusal, this source
is roughly 4× faster and does not refuse. A snapshot table would cost a schema
change, a scheduled job, a shared secret, a `PUBLIC_ENDPOINTS` entry and a
staleness story — to solve a latency problem that measures 1.3 seconds.

**The generalisation to resist:** "the dashboard needed a snapshot, so this does
too." The dashboard needed one because its upstream refuses half the universe at
6 s each. The reasoning was never "fetching on demand is wrong"; it was "*that*
upstream cannot be fetched on demand". This one can.

Revisit if measurement changes: if p95 fan-out ever exceeds ~8 s, or the source
starts rate-limiting, the snapshot pattern is the documented escape hatch and
`pe_data` is the template.

## Scope

**In scope**

1. `core/news_fetch.py` — fetch, normalise, dedupe, filter, select.
2. `GET /api/news` in `routers/news.py`, behind the auth gate.
3. `symbols_in_index(tier)` added to `core/index_membership.py`.
4. `src/app/news/page.tsx` + `loading.tsx`, `NewsList` / `NewsItem`.
5. News as the 4th tab in `BottomTabBar`, and in the desktop `Navbar`.
6. `NewsItem` / `NewsResponse` in `lib/types.ts`, wrapper in `lib/api.ts`.
7. CLAUDE.md updated in **both** repo copies.

**Explicitly out of scope**

- Any EGX-sourced corporate-action record (end-of-right / ex-date / payment
  date). That data exists — verified live on the exchange's own filing, CIB
  coupon 49: `Dividend Date 06/04/2026, Ex-Dividend Date 07/04/2026, Payment
  Date 09/04/2026` — but it is reachable only through a real browser. It is a
  separate project with a separate ingestion host, and folding it in here would
  drag a headless browser into a feature that otherwise needs one HTTP GET.
- Prefilling `AddDividendForm` from a story. Depends on the above.
- Arabic-language stories. `lang=en` only for v1.
- Push notifications or unread state.
- Full-text search over stories.
- Storing stories. Nothing is persisted; see *Why on-demand*.
- Any use of news in scoring, ranking or signals.

## API contract

```
GET /api/news
Authorization: Bearer <token>
```

```jsonc
{
  "your_stocks": [ NewsItem, ... ],   // holdings ∪ watchlist, newest first
  "market":      [ NewsItem, ... ],   // EGX30, newest first, minus the above
  // coverage describes YOUR STOCKS ONLY — the set the user can act on and is
  // entitled to an explanation about. EGX30 coverage is not reported: the user
  // did not ask for those symbols and a count against them would read as a
  // failure of the app rather than an absence of news.
  "coverage": {
    "symbols_requested":      9,
    "symbols_with_news":      5,
    "symbols_without_news":   ["ESRS", "EKHO", "ACGC", "QNBE"],
    // Your own symbols the 40-symbol cap excluded. Normally []. A dropped
    // holding must be VISIBLE, never silently absent — the same rule as the
    // dashboard's "82 stocks · 84 without a price feed". Added 2026-09-03
    // after Task 4's review found the truncation was silent.
    "symbols_over_cap":       [],
    "window_days":            30
  },
  "fetched_at": "2026-09-03T12:40:00Z",
  "status": "ok"                      // "ok" | "partial" | "unavailable"
}
```

```jsonc
NewsItem = {
  "id":           "tag:reuters.com,2026:newsml_FWN44P224:0",
  "title":        "Sodic Signs Medium-Term Facility With CIB",
  "provider":     "reuters",
  "published_at": "2026-08-31T00:00:00Z",
  "url":          "https://www.tradingview.com/news/...",
  "symbols":      ["OCDI", "COMI"]
}
```

`status` is `partial` when the deadline cut the fan-out short and `unavailable`
when nothing was retrieved. A total failure returns an **empty feed with a
status**, never a 500 — the graceful-degradation posture `/api/macro` already
takes.

## Module design

`core/news_fetch.py`. Everything except the two fetchers is pure, so the whole
surface tests without network or Postgres — `tests/` has no DB fixture by
design.

| function | contract |
|---|---|
| `fetch_symbol_news(symbol, timeout)` | one GET → raw items; raises |
| `fetch_many(symbols, deadline, fetcher=None)` | bounded pool → `{sym: [raw]}`; `fetcher` is the test seam, matching `refresh_pe_data(db, rows=None)` |
| `normalize_item(raw, symbol)` | → `NewsItem` or `None` if unusable |
| `dedupe_stories(items)` | by `id`, **merging** `symbols` |
| `is_recent(item, now, days)` | the staleness gate |
| `select_news_symbols(holdings, watchlist, cap)` | holdings first, then watchlist, capped |

`NEWS_URL` lives here; `HEADERS` is **imported from `core/tradingview.py`** so
there is one spelling of the client identity. It is a different host and verb
(GET on `news-headlines` vs POST on `scanner`) so it is not the same client, but
the headers must not drift — the same reasoning that put the scanner URL in one
module shared with `routers/tickers.py`.

### Dedupe merges tags, it does not drop stories

"Sodic Signs Medium-Term Facility With CIB" is returned by both `EGX:OCDI` and
`EGX:COMI` with `relatedSymbols: ["EGX:OCDI", "EGX:COMI"]`. Keeping the first
and discarding the second would lose the fact that it concerns two of the user's
holdings. `dedupe_stories` keeps one row carrying both tags. This is the same
distinction `dedupe_symbol_signals` versus `build_position_signals` draws in
`portfolio_analysis`: dedupe only where the duplicate is genuinely redundant.

### Payload must be trimmed at the boundary

The 22 symbols with news returned **419 KB** of raw JSON. `normalize_item` keeps
six fields, dropping `permission`, `source`, `sourceLogoId`, `urgency` and the
full `relatedSymbols` objects. The response is capped per symbol as well as by
the recency window.

## The staleness rule

**A story older than 30 days is not news.** Items outside the window are
filtered out, and the absence is reported rather than hidden:

> Showing 14 stories · 4 of your 9 stocks have had no news in 30 days

This is the `/api/dashboard` convention — *"82 stocks · 84 without a price
feed"* — where the count line stays truthful and a missing thing is visible
rather than silently absent.

30 days is chosen from the data, not from taste. Measured newest-story age
across the sample:

| symbol | items | newest |
|---|---|---|
| ETEL | 64 | 0 d |
| COMI | 98 | 3 d |
| ABUK | 19 | 4 d |
| HDBK | 12 | 23 d |
| JUFO | 11 | 29 d |
| ADIB | 8 | 32 d |
| CIEB | 5 | 36 d |
| QNBE | 3 | 56 d |
| **ACGC** | 1 | **275 d** |
| ESRS, EKHO | 0 | — |

A 7-day window empties the feed for most holdings. A 90-day window lets ACGC's
275-day-old item render as news — the failure this rule exists to prevent, and
the same class as `--` meaning loading, refused and never-existed at once on
`StockCard` before it gained five states.

Every rendered item shows its real age ("3d ago", "21d ago"). A headline with no
time context is not acceptable on this surface.

## Copyright

Stories are Reuters, Zawya and LSE copy. The app renders **headline, provider,
date and link** and links out to TradingView; it never stores, renders or
paraphrases body text. `storyPath` is TradingView-relative, so items open
off-app with an explicit external-link marker and `rel="noopener noreferrer"`.

A guard test asserts no body/content field survives `normalize_item`, in the
grep/AST style the project already uses for `test_risk_grade.py::test_risk_grade_makes_no_return_claim`
and `test_fixes.py::test_labels_describe_condition_not_action`.

## Symbol scope

- **Your stocks** — `holdings.fetch_open_holdings(db, user.id)` ∪ the user's
  watchlist rows, deduped.
- **Market** — EGX30 constituents via a new `index_membership.symbols_in_index("EGX30")`.
  Deliberately **not** `tickers._load_tickers()`, which merges a live 10 s
  TradingView POST on a cold container; `index_membership` reads the static
  `data/egx_tickers.json` with no network and no DB, which is exactly why that
  module exists.
- Capped at 40 distinct symbols, holdings winning the budget ahead of watchlist
  ahead of EGX30. At the measured 1.3 s for 24, 40 stays well inside one request.

A wall-clock deadline mirroring `BATCH_DEADLINE_SECONDS` bounds the fan-out and
returns what completed, flagged `partial`. The source is fast, but an unbounded
fan-out is precisely how the dashboard broke.

## Caching

`core/cache.py`, 15-minute TTL, keyed on the caller's sorted symbol set. Two
users with the same holdings share an entry; a user who adds a holding misses
and refetches, which is correct.

The service worker treats `/api/news` as `/api/*` — network-first with cache
fallback — so the tab still paints offline with the last feed seen. No `sw.js`
change is needed.

## Auth

`/api/news` requires a token via `Depends(get_current_user)` and is **not**
added to `PUBLIC_ENDPOINTS`. The app is closed; a new router is denied by
default and `tests/test_auth_gate.py` walks the real route table, so this is
verified by enumeration rather than by assertion here.

The feed is user-scoped — it reads that user's holdings and watchlist — so it
could not be public even if the policy allowed it.

## Frontend

`src/app/news/page.tsx` with `loading.tsx` added in the same breath, per the
standing rule that every route segment carries a Suspense boundary or the nav
tap reads as dead.

Cards, not tables, at every width: a news item is a headline plus three small
facts, and the mobile/desktop table split earns nothing here. 44 px targets.
Each item shows provider badge, relative age, and symbol chips linking to
`/stock/[symbol]`. Your-stocks section first, market below, each with a count
line. Empty states name the symbols that had nothing rather than rendering
blank.

Colour rule holds: `gain`/`loss` mean a real direction in data. **News carries
no direction**, so nothing on this page is green or red — headlines render in
neutral text. Sentiment is not computed and must not be implied by colour.

## The nav change

News becomes the 4th tab in `BottomTabBar`, and joins the desktop `Navbar`.

Geometry is already validated: CLAUDE.md records the pill measured at "320px
wide, 52px tall, **four** 44px tabs", which is the arrangement this restores.
`BottomTabBar`'s measured-highlight effect already carries `tabs.length` in its
deps, so the travelling pill re-measures correctly when the count changes.

**This reverses a documented decision.** CLAUDE.md states "Compare was removed
from `BottomTabBar` to make room" and that the pill "holds the destinations
every user has". News is such a destination; Compare does **not** return to the
pill and keeps its dashboard-header button and desktop nav link. Both repo
copies of CLAUDE.md must be updated in the same change, since the two are
required to stay identical.

## Testing

TDD, tests before implementation:

- `dedupe_stories` — a story related to two held symbols appears once carrying
  both tags, not twice and not once with one tag.
- `is_recent` — the 30-day boundary, both sides, and a null/garbage timestamp.
- `select_news_symbols` — holdings beat watchlist beat EGX30; the cap binds;
  duplicates across the three sources collapse.
- `normalize_item` — a malformed payload (missing `published`, missing
  `storyPath`, non-EGX `relatedSymbols`) yields `None` rather than a partial row.
- `fetch_many` — the deadline returns partial results rather than raising.
- Guard: no body/content field survives normalisation.
- `symbols_in_index` — returns EGX30 members, `[]` for an unknown tier, and
  fires no network call.

## Risks

- **Coverage is genuinely thin for small caps.** 2 of 24 sampled symbols had no
  news at all and several had nothing inside 30 days. The coverage line is the
  mitigation; the feature must read as honest when empty, not broken.
- **The endpoint is undocumented.** It is a public web endpoint TradingView uses
  for its own site, not a published API, and it can change or start requiring
  auth without notice. Failure is contained: `status: "unavailable"` and an
  empty feed, no 500, no effect on any other page. Cost of loss is one tab.
- **`lang=en` on an Arabic-first market.** Reuters/Zawya English coverage skews
  to large caps, which is part of why small-cap coverage is thin. Arabic is
  deliberately deferred rather than half-done.
