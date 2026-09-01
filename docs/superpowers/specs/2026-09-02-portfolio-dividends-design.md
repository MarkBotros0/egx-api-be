# Portfolio Dividends — Design Spec

**Date:** 2026-09-02
**Status:** Approved for planning
**Repos:** `egx-api-be` (Python/FastAPI), `egx-api-fe` (Next.js)

## Problem

The user receives cash distributions — "profit share" — on stocks they hold,
most often on banking names. Today the app has nowhere to record that money.
The Winnings card, shipped 2026-09-01, counts **capital gains only**: it reads
`portfolio_sales`, which exists solely to record a sell.

The gap is not cosmetic. A bank holding down 5% on price but up 12% in
collected dividends currently reads as a loss everywhere in the app. For a
market where banks and real estate dominate — the same reason
`fundamentals_history` logs book value per share — that is a systematic
understatement of exactly the holdings this user owns.

## What This Is Not

A dividend **is not a sale**. It reduces no position, closes no cost basis, and
has no undo semantics. Recording one in `portfolio_sales` would corrupt every
invariant that table holds: `quantity` would decrement against shares still
owned, `DELETE /api/sales` would restore shares that were never sold, and
`fetch_open_holdings`'s `quantity > 0` filter would start hiding live positions.

It therefore gets its own table, its own write routes, and its own pure-maths
module. This separation is the load-bearing decision in this spec.

## Scope

**In scope**

1. `portfolio_dividends` table, symbol-anchored.
2. `POST` / `DELETE /api/dividends`.
3. `GET /api/sales` extended to return dividends and a combined summary.
4. `RealizedGainsCard` headline becomes total money made, split underneath.
5. Per-symbol breakdown covers dividend-only symbols (stocks never sold).
6. `AddDividendForm`, `DividendsTable` on the portfolio page.
7. A dividends-collected figure per open holding in `HoldingsTable`.
8. `DELETE /api/users` extended to clean the new table.
9. A `dividends` Learn-page concept.

**Explicitly out of scope**

- Feeding dividends into Sharpe, Sortino, VaR, CVaR, Monte Carlo or the
  correlation matrix. That is the risk maths on the app's heaviest endpoint and
  a separate decision from recording the money.
- Automatic dividend detection from any feed. `pe_data.dividend_yield` is a
  market-wide *rate*, not a record of what this user was paid; the two must
  never be conflated.
- Editing a dividend. Delete and re-add. A dividend is four fields.
- Dividend reinvestment (DRIP) modelling.
- Withholding-tax computation. See *Amount semantics*.

## Data Model

```sql
CREATE TABLE IF NOT EXISTS portfolio_dividends (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    name       TEXT NOT NULL,
    sector     TEXT DEFAULT '',
    amount     DOUBLE PRECISION NOT NULL,
    pay_date   TEXT NOT NULL,
    shares     INTEGER,
    notes      TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_portfolio_dividends_user
    ON portfolio_dividends(user_id, symbol);
```

Added to `init_db` in `app/core/db.py` alongside the existing idempotent DDL.
There is no migration framework — every statement in `init_db` is idempotent, so
this lands on the next cold start of any process.

### Symbol-anchored, deliberately no `holding_id`

A sale carries `holding_id` because `DELETE /api/sales` must restore shares to a
specific position. A dividend restores nothing, so the column would buy no
behaviour — and would cost real correctness: deleting a holding would orphan or
destroy the record of money genuinely received.

Symbol-anchoring means dividend history survives selling out of a position
entirely and survives deleting the holding row.

**Accepted consequence — shared-symbol display.** Nothing in
`POST /api/portfolio` prevents two `portfolio` rows for the same symbol
(verified: `app/routers/portfolio.py::add_holding` inserts unconditionally). When
that happens, the dividend figure shown on each of those rows is the **symbol's**
total, labelled as such — see *Shared-symbol labelling*. Splitting it across rows
by today's share count would be fiction: the share counts differed when the
dividend was paid. Totals are never affected, because every aggregate sums the
dividend ledger directly and never reaches it through holdings.

### Amount semantics

`amount` is **the total EGP that actually landed in the account** — already net
of Egypt's 5–10% dividend withholding tax. It is the figure the user can verify
against a Thndr statement.

The app does not compute tax, and must never present `amount` as a gross figure.
`shares` is optional and stored only so the UI can display an approximate
per-share figure; it is never used to derive `amount`.

## Backend

### `app/core/dividends.py`

Pure maths **plus the one spelling of the dividend queries** — the same shape as
`app/core/holdings.py`, which pairs pure `row_to_holding` with
`fetch_open_holdings(db, user_id)`. The read queries take `db` as a parameter,
so they stay fakeable, and the pure functions remain independently testable
against a `tests/` directory that has no Postgres fixture.

Putting the queries here rather than in the router is what stops three callers
(`routers/dividends.py`, `routers/sales.py`, `routers/portfolio_analysis.py`)
from growing three spellings of "this user's dividends".

```python
class DividendValidationError(ValueError):
    """A dividend that is not internally consistent. Maps to HTTP 400."""


# --- pure ---------------------------------------------------------------
def validate_dividend(*, symbol, amount, pay_date, shares, today: date) -> dict:
    """Returns normalized {symbol, amount, pay_date, shares}."""


def enrich_dividend(row: dict) -> dict:
    """Adds amount_per_share (None when shares is missing or 0)."""


def is_duplicate(existing: list, candidate: dict) -> bool:
    """True when an identical symbol + pay_date + amount is already recorded."""


def summarize_realized(priced_sales: list, dividends: list) -> dict:
    """The Winnings card's numbers, gains and dividends together."""


# --- the one spelling of the queries ------------------------------------
DIVIDEND_COLUMNS = "id, symbol, name, sector, amount, pay_date, shares, notes, created_at"


def fetch_dividends(db, user_id: str) -> list[dict]:
    """This user's dividends, enriched, newest pay_date first."""


def fetch_dividend_totals(db, user_id: str) -> dict:
    """{symbol: total_amount}. Returns {} when user_id is None."""
```

`summarize_realized` **replaces** `sales.summarize_sales`. It is not a second
summariser: two functions producing overlapping Winnings figures is precisely
the divergence class documented in *One Score Per Stock*.

The existing `summarize_sales` tests move to the new name and gain an `[]`
dividends argument. **Every assertion in them stays byte-identical** — that is
the regression gate proving the capital-gains numbers did not move.

Validation rules:

| Rule | Behaviour |
|---|---|
| `symbol` missing/blank | 400 `"Pick a stock."` |
| `symbol` casing | Upper-cased, matching `POST /api/portfolio` |
| `amount` not a number | 400 `"Amount must be a number."` |
| `amount <= 0` | 400 `"Amount must be greater than 0."` |
| `pay_date` blank | Defaults to today |
| `pay_date` unparseable | 400 `"Pay date must be a date like 2026-09-01."` |
| `pay_date` in the future | 400 `"Pay date cannot be in the future."` |
| `shares` absent/blank | Stored as NULL — optional field |
| `shares` present, not a positive whole number | 400 `"Shares must be a whole number of shares."` |

There is **no buy-date lower bound**. A dividend is symbol-anchored, so there is
no single holding whose buy date could bound it, and the user may legitimately
record a dividend on a position they have already sold.

**Duplicate guard.** An exact `symbol + pay_date + amount` match against the
user's existing dividends returns **409** with
`"You already recorded a dividend of X EGP for SYMBOL on DATE."` This exists
because the primary surface is a phone: a double-tapped submit is the most
likely way this ledger gets silently wrong, and unlike a sale, a duplicate
dividend corrupts no share count and so leaves no other trace.

### Summary shape

`summarize_realized` returns every field `summarize_sales` returned, unchanged,
plus:

| Field | Meaning |
|---|---|
| `total_dividends` | Sum of `amount` across all dividends |
| `dividend_count` | Number of dividend records |
| `total_winnings` | `total_realized_pnl + total_dividends` |

`total_realized_pnl_pct` **stays capital-gains-only and cost-weighted.**
Dividends have no matching cost in this ledger — the shares producing them may
still be held — so adding them to a numerator whose denominator is closed-trade
cost would produce a percentage of nothing. The headline percentage continues to
describe closed trades.

`beat_t_bill_count` / `annualizable_count` are **untouched**. They are facts
about individual closed trades; a dividend maps onto no single trade, so folding
it in would make the line unverifiable.

`best_trade` / `worst_trade` also stay **sales-only**. They are labelled "trade"
and each returns a whole `Sale` object the card reads fields off; making them
dividend-aware would mean returning two different shapes from one field.

`by_symbol` entries gain `dividends` and `total_winnings`, and the list becomes
a **union**: a symbol with dividends but no sales appears with `sales_count: 0`,
`cost: 0`, `proceeds: 0`, `realized_pnl: 0`, `realized_pnl_pct: null`, and its
dividend total. This is required — a stock you still hold and collect on would
otherwise be missing from the breakdown entirely. Sorted by `total_winnings`
descending.

### Routes

**`POST /api/dividends`** → 201

Body: `{symbol, name?, sector?, amount, pay_date?, shares?, notes?}`.
Single INSERT — no `db.transaction()` needed, because unlike a sale there is no
second statement to keep atomic with it. Returns the enriched dividend.

`name` and `sector` default to the symbol and `""` respectively when the client
does not supply them, matching `POST /api/portfolio`.

**`DELETE /api/dividends?id=`** → `{"deleted": id}`

`DELETE ... RETURNING id`; 404 when it matches nothing. No share restoration —
nothing was ever decremented.

**`GET /api/sales`** — extended, same URL

Response gains a `dividends` array — from `fetch_dividends`, so every entry
arrives already through `enrich_dividend` and carries `amount_per_share`.
`summary` comes from `summarize_realized`.

The URL does not change and no third endpoint is added, so that the combined
headline is computed in **tested Python** rather than in TSX. If the frontend
added gains and dividends together itself, that sum would be the one number on
the page with no test behind it. The endpoint's docstring is updated to say it
serves the realized ledger — sales and dividends — not sales alone.

Both routes are scoped by `user.id` from the JWT, like every other router.

### `/api/portfolio_analysis` — dividends per holding

`_analyze(holdings, user_id)` fetches dividend totals once, keyed by symbol:

```python
dividends_by_symbol = fetch_dividend_totals(db, user_id)  # {} when user_id is None
```

**It must key off `user_id`, not off `holdings`.** The POST path
(`post_portfolio_analysis`) takes holdings from the request body, so anything
derived from the caller's stored data has to come from `user_id`.

Each holding row gains:

| Field | Meaning |
|---|---|
| `dividends_collected` | EGP collected against this symbol, `0` when none |
| `dividends_symbol_shared` | `true` when the user has >1 open holding of this symbol |

`dividends_symbol_shared` is computed by counting symbol occurrences across the
`holdings` list passed to `_analyze`.

**The error-row append path must carry both fields too.** A holding whose price
feed is down already keeps its Actions cell so it stays sellable; a dividend
figure that vanishes on a feed error would look like the money disappeared.

This adds one indexed query with no price fetch, so it costs nothing against the
30 s Vercel budget.

### `DELETE /api/users` — must clean the new table

`app/routers/users.py` deletes `portfolio_sales`, `portfolio`, `watchlist` and
`user_settings` inside one `db.transaction()`. No table has an FK to `users`, so
nothing cascades. `portfolio_dividends` **must** be added to that transaction or
deleted users leave invisible orphan rows.

A test greps `users.py` for `portfolio_dividends` and fails if it is absent.

## Frontend

### Types (`src/app/lib/types.ts`)

```ts
export interface Dividend {
  id: string;
  symbol: string;
  name: string;
  sector: string;
  /** Total EGP received, already net of withholding tax. */
  amount: number;
  pay_date: string;
  /** Optional — shares held when paid. Null when not recorded. */
  shares: number | null;
  notes: string;
  created_at: string;
  /** Computed server-side; null when shares is null or 0. */
  amount_per_share: number | null;
}
```

`SalesSummary` gains `total_dividends`, `dividend_count`, `total_winnings`.
`SymbolRealized` gains `dividends`, `total_winnings`.
`SalesResponse` gains `dividends: Dividend[]`.
`HoldingAnalysis` — the per-holding interface, **not** `StockAnalysis` — gains
`dividends_collected: number` and `dividends_symbol_shared: boolean`, both
optional so a stale cached analysis response still type-checks.

Every nullable field is typed `| null`, matching the existing sales types.

### `src/app/lib/api.ts`

`recordDividend(body)` and `deleteDividend(id)`, via the existing `fetchJSON`
wrapper. No new fetch helper for reading — dividends arrive on `fetchSales()`.

### `RealizedGainsCard`

- Headline becomes `total_winnings`, coloured by its own sign.
- Directly beneath, a split line: `4,200 from sales · 1,150 in dividends`.
  Rendered only when `total_dividends > 0`, so a user who has never recorded a
  dividend sees today's card unchanged.
- The `total_realized_pnl_pct` line keeps its existing wording and continues to
  describe **closed trades only**. Its copy is made explicit about that, since
  the headline above it now includes more than closed trades.
- Per-stock rows show the dividend component when non-zero.
- The T-bill line is unchanged.
- Its `LearnTooltip` currently reads *"It counts capital gains only; dividends
  are not included."* — that sentence must be replaced, not left to contradict
  the card.
- Empty state now keys off `by_symbol.length`, which the union makes correct for
  a dividends-only user with no sales.

### `AddDividendForm`

Full-screen modal on mobile, inline card on desktop — the `AddHoldingForm` and
`SellHoldingForm` pattern.

Fields: symbol, amount received (EGP), pay date, shares (optional), notes
(optional).

- Reached from a holding's row action, which pre-fills symbol, name, sector and
  the current share count; or from a general add button, where symbol is chosen
  from the user's holdings **and** past sales, so a dividend can be logged
  against a position already closed.
- `max` on the date input is today; there is no `min`.
- Numeric inputs use `text-[16px] md:text-sm` to prevent iOS zoom.
- Touch targets `min-h-[44px]`.
- The `error` prop renders **inside** the modal, above the buttons — the
  sell-form fix: an error rendered on the page behind a full-screen modal is
  invisible on a phone.
- The 409 duplicate response surfaces as its message text, not a generic failure.
- Body scroll lock is derived from the same boolean that gates the modal's
  render, and is skipped above `md` — both fixes carried from `SellHoldingForm`.

### `DividendsTable`

Collapsed `<details>`, mobile cards / desktop table, delete per row with
confirmation. The `ClosedPositionsTable` pattern. Columns: symbol, pay date,
amount, per-share (when known), notes.

### `HoldingsTable`

A dividends-collected figure per holding, shown two ways and **without adding a
table column**:

1. A small pill beside the symbol — mobile card header and desktop Symbol cell —
   rendered only when `dividends_collected > 0`.
2. A full labelled line in the expanded detail, both breakpoints.

**No new `<th>`.** The desktop table already coordinates `colSpan={12}` on the
expanded row against `colSpan={10}` on the error row; a thirteenth column means
editing both in lockstep, and CLAUDE.md already records a colSpan mismatch in
this exact file as a shipped bug. A pill carries the same information at no
structural risk. Dividends are a per-holding fact worth seeing, not a column
worth scanning and sorting.

Suppressed entirely when `dividends_collected` is 0, so a user who records none
sees today's table unchanged.

**Shared-symbol labelling.** When `dividends_symbol_shared` is true, the figure
is labelled as the symbol's total (e.g. `COMI total`) rather than presented as
that row's own. A `LearnTooltip` explains that dividends are tracked per stock,
not per purchase lot.

### `src/app/portfolio/page.tsx`

Dividends arrive with the existing `loadSales` call, so no new independent
loader and no new failure mode. New state for the add-dividend modal only.
`DividendsTable` renders beside `ClosedPositionsTable`.

**Three existing render gates key off `sales.sales.length` and must widen**, or
a user whose only record is a dividend sees nothing:

| Location | Today | Must become |
|---|---|---|
| Never-traded empty state | `!sales?.sales.length` | also requires no dividends |
| `RealizedGainsCard` (full page) | `sales.sales.length > 0` | `\|\| sales.dividends.length > 0` |
| Sold-out branch | renders when `sales` | unchanged, but must render `DividendsTable` too |

The sold-out branch matters: a user who sold everything but still collects on a
past position must keep both histories on screen.

### Learn page

A `dividends` Concept anchor, following the existing title / definition /
whyItMatters / howToUse shape.

**Framing rule, carried from CLAUDE.md and binding on every string in this
feature:** with T-bills near 25%, no EGX dividend yield is competitive as
income. A dividend must be presented as **evidence the company generates real
cash**, never as income and never as beating anything. No copy in this feature
compares a dividend to the T-bill rate.

## Testing

`egx-api-be/tests/test_dividends.py`, pure-function tests throughout — no DB.

**Test command is `./.venv/Scripts/python.exe -m pytest`.** A bare `python` on
this machine has no pytest.

1. `validate_dividend` — each row of the validation table above.
2. `validate_dividend` upper-cases the symbol.
3. `enrich_dividend` computes `amount_per_share`; null when `shares` is null or 0.
4. `is_duplicate` — exact triple matches; differing amount/date/symbol does not.
5. `summarize_realized([], [])` returns the documented zero shape.
6. `summarize_realized(sales, [])` reproduces the pre-existing sales figures
   **exactly** — the regression gate.
7. `total_winnings == total_realized_pnl + total_dividends`.
8. `total_realized_pnl_pct` does **not** move when dividends are added.
9. `by_symbol` includes a dividend-only symbol with `sales_count: 0` and
   `realized_pnl_pct: null`.
10. `by_symbol` is sorted by `total_winnings` descending.
11. `beat_t_bill_count` / `annualizable_count` do not move when dividends are
    added.
12. Grep guard: `app/routers/users.py` mentions `portfolio_dividends`.
13. Grep guard: `summarize_sales` no longer exists as a second summariser.

The existing `tests/test_sell_tracking.py` must pass unchanged except for the
`summarize_sales` → `summarize_realized` rename.

## Global Constraints

- **psycopg3 placeholders are `%s`**, never `?`.
- **A literal `%` in SQL must be written `%%`.** `_DB.execute` always passes a
  params tuple, so psycopg parses every query for placeholders. There is a test
  walking the AST of every `execute()` call that fails on a bare `%`.
- **Mobile-first.** `md:` (768px) breakpoint, `min-h-[44px]` touch targets,
  `space-y-3 md:hidden` + `hidden md:block` for table/card switching,
  full-screen modals on mobile, `text-[16px] md:text-sm` on inputs.
- **All money in EGP.** Tailwind `gain` #00ff88 / `loss` #ff3355.
- **Every route scoped by `user.id`** from the JWT.
- **CLAUDE.md is mirrored** in `egx-api-be/` and `egx-api-fe/` and the two
  copies must stay byte-identical.
- **Pushing requires `gh auth switch --user MarkBotros0` first.**

## Manual Verification

Nothing in this branch can run against Postgres or a browser from the test
suite. Before relying on it:

1. `portfolio_dividends` lands on a cold start with the documented columns and
   index.
2. Record a dividend; confirm it appears in `GET /api/sales` and that
   `total_winnings` equals gains plus dividends.
3. Submit the identical dividend again; confirm **409** and that no second row
   was written.
4. Record a dividend on a symbol with no sales; confirm it appears in
   `by_symbol` with `sales_count: 0`.
5. Delete a holding that has dividends; confirm the dividends survive and the
   Winnings total is unchanged.
6. Delete a user via `/api/users`; confirm zero rows remain in
   `portfolio_dividends` for that user.
7. Six browser checks at 375px: add-dividend modal opens and scroll-locks; an
   invalid amount shows its error inside the modal; the duplicate 409 message is
   readable; `DividendsTable` expands; the holdings dividend figure appears; a
   shared-symbol holding shows the "total" label.
