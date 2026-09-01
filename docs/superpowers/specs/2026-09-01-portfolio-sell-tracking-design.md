# Portfolio Sell Tracking & Realized Gains — Design

**Date:** 2026-09-01
**Status:** Approved design, not yet implemented
**Spans both repos:** `egx-api-be` (schema, API) and `egx-api-fe` (UI)

## Goal

Let the user record what they sold and for how much, then show what they
actually won. Two user-visible outcomes:

1. A **Sell** action on any holding — full or partial — capturing quantity,
   sell price and sell date.
2. A **Winnings card** with total realized P&L and a per-stock breakdown, plus
   a **Closed Positions** section listing individual sales.

## Non-goals

- **No cash balance.** `cash_available` is documented in CLAUDE.md but does not
  exist in the code — not seeded in `db.py`, never read, no UI. Nothing in this
  feature writes a cash balance, and the stale doc lines get corrected (see
  *Documentation*).
- **No FIFO lot tracking, no averaging into a position.** One holding row is one
  cost basis. If the user buys the same symbol twice they already create two
  rows today; that stays true.
- **No dividends** in realized P&L. Capital gains only.
- **No tax or commission modelling.** The sell price is what the user types.
- **No order placement.** The app remains analysis-only; trades happen in Thndr.

## Data model

One new table, created idempotently in `db.py::init_db` alongside the others
(there is no migration framework — every statement there is idempotent and new
tables land on the next cold start of any process).

```sql
CREATE TABLE IF NOT EXISTS portfolio_sales (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    holding_id  TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    name        TEXT NOT NULL,
    sector      TEXT DEFAULT '',
    quantity    INTEGER NOT NULL,
    buy_price   DOUBLE PRECISION NOT NULL,
    buy_date    TEXT NOT NULL,
    sell_price  DOUBLE PRECISION NOT NULL,
    sell_date   TEXT NOT NULL,
    notes       TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_portfolio_sales_user ON portfolio_sales(user_id);
```

### Why a ledger and not columns on `portfolio`

Adding `sell_price`/`sell_date`/`sold_quantity` to `portfolio` was rejected. A
partial sell would have to split the row in two, so a "holding" row would mean
two different things, every existing query would need a `WHERE sell_date IS
NULL`, and missing one puts a phantom position into the risk metrics. The split
also severs the link between the two halves.

A full trade ledger (buys and sells both as events, positions derived) was also
rejected as YAGNI — it rewrites portfolio CRUD and every consumer to buy lot
tracking the user did not ask for. This design can grow into that later.

### Cost basis is snapshotted, deliberately

`buy_price`, `buy_date`, `name` and `sector` are copied into the sale row rather
than joined from `portfolio`. This is the same point-in-time principle
`fundamentals_history` already uses in this codebase: the sale is a historical
fact and must not silently change when the user later edits the holding it came
from. It also means win history survives deleting the original holding.

### The open position stays the anchor

Selling N shares decrements `portfolio.quantity` by N. **A full sell sets
`quantity = 0`; it does not delete the row.** Zero-quantity rows are filtered
out of every read, so they are invisible in the UI and never reach analysis, but
they remain as the anchor that makes undo exact — deleting a sale adds the
quantity back and the position reappears intact, with its target price, stop
loss and notes still attached. No orphaned shares, no resurrection guesswork.

## Backend

### The open-holdings query must be spelled once

`quantity > 0` has to be applied in two places that today issue their own
near-identical SELECT:

- `portfolio.py::get_portfolio`
- `portfolio_analysis.py::get_portfolio_analysis` (it does **not** call
  `/api/portfolio`; it queries the table directly)

Two independent spellings is exactly the drift this codebase has been bitten by
before (see *One Score Per Stock* in CLAUDE.md). Extract a single
`fetch_open_holdings(db, user_id) -> list[dict]` into `app/core/holdings.py` and
have both call it.

The helper covers **list reads only**. By-id lookups stay inline — `portfolio.py`
needs them for PUT and DELETE, and `sales.py` needs one that deliberately
ignores the `quantity > 0` filter so a sale can be undone against a
fully-closed holding.

Note `POST /api/portfolio_analysis` takes holdings from the request body, so no
filter applies there — the client sends what it has.

### Shared return helpers

Move `_days_between` and `_annualized_return` (plus
`MIN_DAYS_FOR_ANNUALIZATION = 30`) out of `portfolio_analysis.py` into
`app/core/returns.py`, and import them in both places.

The sales math needs position-level annualization and must not invent a second
formula. The existing `_annualized_return` already returns `None` below 30 days
held — "a +5% week annualizes to five figures" — and that guard is exactly right
for a quick flip. `_days_between(date_str, end: date)` already takes an
arbitrary end date, so it works unchanged for buy_date → sell_date.

### New router: `app/routers/sales.py`

Deliberately separate from `portfolio_analysis`, which is already the heaviest
endpoint and flirts with the 30 s Vercel timeout. Realized gains need **no price
fetch at all**, so the Winnings card renders instantly even on a run where the
analysis times out.

Every route is scoped by `user.id` from the JWT, matching `portfolio.py`.

#### `POST /api/sales`

Body: `{holding_id, quantity, sell_price, sell_date, notes?}`

1. Load the holding by `(id, user_id)` → 404 if absent.
2. Validate (400 with a specific message on each):
   - `quantity` is an integer, `> 0`, and `<= holding.quantity` — over-selling
     is rejected, naming the remaining quantity.
   - `sell_price > 0`.
   - `sell_date` parses as `YYYY-MM-DD`, is **not before** `buy_date`, and is
     **not in the future**. Defaults to today when omitted.
3. Insert the sale row, snapshotting cost basis.
4. `UPDATE portfolio SET quantity = quantity - N, updated_at = now`.
5. Both statements commit together; on any failure neither lands.

Returns `{sale, holding}` so the client can update both without a refetch.

#### `GET /api/sales`

Returns `{sales, summary, currency}`.

Each sale carries its computed fields — `proceeds`, `cost`, `realized_pnl`,
`realized_pnl_pct`, `days_held`, `annualized_return_pct` (null under 30 days),
`beat_t_bill` (null when annualized is null).

`summary`:

```
{ total_realized_pnl, total_realized_pnl_pct, total_proceeds, total_cost,
  win_count, loss_count, best_trade, worst_trade,
  by_symbol: [{symbol, name, sector, sales_count, quantity,
               cost, proceeds, realized_pnl, realized_pnl_pct}] }
```

- `total_realized_pnl_pct` is `total_realized_pnl / total_cost * 100` —
  cost-weighted, never a mean of percentages.
- `by_symbol` aggregates every sale of a symbol and sorts by `realized_pnl`
  descending, so the biggest win leads.
- Empty portfolio → zeroed summary and `sales: []`, never a 404. The card
  renders its own empty state.

#### `DELETE /api/sales?id=`

Undo. Deletes the sale and adds `quantity` back to the holding if it still
exists. If the user hard-deleted the holding, the sale is removed and the
quantity is not restored — there is nothing to restore it to. The confirmation
dialog says which of the two will happen.

`risk_free_rate` is read from `settings` exactly as `portfolio_analysis.py:132`
does, defaulting to `DEFAULT_RISK_FREE_RATE_PCT`.

## Realized-return math

Per sale:

```
proceeds          = sell_price * quantity
cost              = buy_price  * quantity
realized_pnl      = proceeds - cost
realized_pnl_pct  = (sell_price / buy_price - 1) * 100
days_held         = sell_date - buy_date          (calendar days)
annualized        = _annualized_return(realized_pnl_pct, days_held)   # None < 30d
beat_t_bill       = annualized > risk_free_rate_pct                   # None if above is None
```

### The T-bill line

Each closed position shows its annualized return against the ~25% T-bill, and
the card carries one summary line for the portfolio as a whole. This was flagged
for the user's call and is included by default.

It exists because a +8% win held for two years **lost to cash**, and a card that
only says "+8% ✓" teaches the opposite. This is the same lesson the existing
`cash_underperformer` signal already delivers for open positions
(`portfolio_analysis.py:709`), applied to closed ones — so the app does not
congratulate the user for a trade it would have warned them about while they
held it.

Framing rules, matching the house voice used for dividend yield:

- A win that lost to T-bills is still shown as a win in EGP. The T-bill line is
  context, not a demotion — no red on a positive number.
- Under 30 days held, no annualized figure is shown at all. Not a zero, not a
  dash with a tooltip — the row simply omits it, because the number would be
  nonsense.
- The comparison is labelled as against *risk-free cash over the same period*,
  never as "you should have bought T-bills".

## Frontend

### Types (`lib/types.ts`)

`Sale`, `SalesSummary`, `SymbolRealized`, `SalesResponse`. `PortfolioHolding`
is unchanged — `quantity` already means "shares held" and still does.

### API wrappers (`lib/api.ts`)

`fetchSales()`, `recordSale(body)`, `deleteSale(id)`, following the existing
`fetchJSON` pattern at `api.ts:114-147`.

### `SellHoldingForm`

Full-screen modal on mobile, inline card on desktop — the exact structure
`AddHoldingForm` uses, including the body-scroll lock the portfolio page already
applies at `page.tsx:41`.

Fields: quantity (**pre-filled to the full remaining position**, `max` set to
it, with "of N shares" beside the input), sell price, sell date (defaults
today), optional notes. Live preview of the resulting P&L in EGP and % as the
user types, so the number is confirmed before saving, and the submit button is
disabled while the quantity exceeds what is held.

### Entry points in `HoldingsTable`

- Mobile: a third button in the expanded card's action row alongside Edit and
  Delete (`HoldingsTable.tsx:280`), `min-h-[44px]`.
- Desktop: a "Sell" link in the actions cell next to Edit / Del
  (`HoldingsTable.tsx:449`).

A new `onSell: (id: string) => void` prop, wired the same way `onEdit` is.

### `RealizedGainsCard`

Placed directly under `PortfolioSummary` on the portfolio page.

- Headline: total realized P&L in EGP, gain/loss coloured, with total return %.
- Support row: win/loss record, best and worst trade, total proceeds.
- One T-bill line for the portfolio's realized trades as a whole.
- Per-stock breakdown from `by_symbol`, biggest win first: symbol, shares sold,
  realized EGP, realized %.
- Empty state when there are no sales, explaining that selling a holding records
  it here.
- `LearnTooltip` on "Realized" and on the T-bill comparison — every metric in
  this app carries one.

### `ClosedPositionsTable`

Collapsed `<details>` section below the Winnings card. Per sale: symbol, buy →
sell price, quantity, date sold, days held, realized P&L, annualized vs T-bill
where available, and a delete (undo) action with a confirmation dialog matching
the existing delete dialog at `HoldingsTable.tsx:588`.

Desktop table / mobile cards, per the `space-y-3 md:hidden` + `hidden md:block`
convention.

### Page wiring (`portfolio/page.tsx`)

Sales load in parallel with the portfolio and independently of the analysis, so
the Winnings card paints on first load without waiting on price fetches. After
recording or deleting a sale, refresh both the sales and the portfolio and clear
`analysis` to re-trigger the existing re-analysis effect at `page.tsx:92`.

The Winnings card and Closed Positions render whenever sales exist — including
when every position has been closed and the Holdings table is empty. The current
empty state at `page.tsx:310` must not swallow the page in that case; a user who
sold everything should still see what they won.

### Learn page

A `realized_gains` Concept anchor covering realized vs unrealized P&L and why a
win can still lose to T-bills, so the card's tooltips have somewhere to link.

## Edge cases

| Case | Behaviour |
|---|---|
| Sell more than held | 400, message names the remaining quantity |
| Sell 0 or negative | 400 |
| `sell_date` before `buy_date`, or in the future | 400 |
| Held < 30 days | Realized P&L shown; annualized omitted entirely |
| `buy_price` of 0 | `realized_pnl_pct` reported as null, EGP still exact |
| Full sell | `quantity = 0`, row hidden everywhere, undo restores it |
| Holding hard-deleted after a sale | Sale survives with its snapshot; undo cannot restore quantity and the dialog says so |
| Every position closed | Holdings empty state suppressed; Winnings card still renders |
| Sale of a symbol whose price feed is down | Unaffected — sales never fetch prices |

## Testing

Backend, in `egx-api-be/tests/` following `test_fixes.py` conventions:

- Partial sell decrements the holding and leaves it visible.
- Full sell zeroes the holding and hides it from both read paths.
- Over-sell, zero quantity, negative price, reversed dates and a future
  `sell_date` each rejected.
- Delete restores the quantity exactly; two partial sells then two deletes
  return the holding to its original quantity.
- Realized P&L, cost-weighted total %, and `by_symbol` aggregation are correct
  over multiple sales of one symbol.
- Annualized return is `None` below 30 days and matches `_annualized_return`
  above it.
- `_analyze` never receives a zero-quantity row.
- A grep test: neither `portfolio.py` nor `portfolio_analysis.py` hand-writes a
  `quantity > 0` filter — both list reads go through `fetch_open_holdings`.

## Documentation

CLAUDE.md is mirrored in both repos and both copies must change together:

- Add `portfolio_sales` to the schema section and `/api/sales` to the endpoint
  list.
- Document the `quantity = 0` convention and that both read paths filter it.
- Note the new `core/holdings.py` and `core/returns.py`.
- **Correct the stale `cash_available` claims** in three places ("Cash balance
  management via `updateCash()`", the pre-seeded settings list, the schema
  comment) — the setting does not exist in the code.
- Correct the "No auth" line under *Things to Know*; `portfolio.py` scopes every
  query by `user.id`.
