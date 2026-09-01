# Portfolio Sell Tracking & Realized Gains — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user record full or partial sells of a holding and see total realized winnings with a per-stock breakdown.

**Architecture:** A `portfolio_sales` ledger table snapshots each sale's cost basis; selling decrements `portfolio.quantity`, and a full sell leaves the row at `quantity = 0` as an undo anchor that every read filters out. All sales maths lives in pure functions in `app/core/sales.py`; `app/routers/sales.py` is a thin IO layer. The Winnings card reads a dedicated `/api/sales` endpoint that never fetches prices, so it paints even when portfolio analysis is slow.

**Tech Stack:** FastAPI + psycopg3 (Neon Postgres) backend; Next.js 14 App Router + React 18 + TypeScript + Tailwind frontend. Backend tests are pytest over pure functions.

**Spec:** `egx-api-be/docs/superpowers/specs/2026-09-01-portfolio-sell-tracking-design.md`

## Global Constraints

- **Mobile-first.** `md:` (768px) is the breakpoint. Tables become cards on mobile via `space-y-3 md:hidden` + `hidden md:block`. Touch targets `min-h-[44px]`. Forms are full-screen modals on mobile, inline on desktop.
- **Colours:** `gain` (#00ff88), `loss` (#ff3355), `accent` (#4488ff), `charcoal` (#12121a), `charcoal-dark` (#0a0a0f). Money is always EGP.
- **Backend tests:** run from `egx-api-be` with `python -m pytest tests/ -v`. New tests go in `tests/test_sell_tracking.py`.
- **Frontend has NO test runner** (no jest/vitest in `package.json`). Frontend verification is `npx tsc --noEmit` then `npm run build`, plus the stated manual check.
- **psycopg3 uses `%s` placeholders**, never `?`.
- **Commits:** commit locally after each task. **Never push** — both repos are live and the user has not authorized pushes.
- **T-bill framing (from the spec, verbatim rules):** a win that lost to T-bills is still shown as a win in EGP — no red on a positive number; under 30 days held no annualized figure is shown at all; the comparison is labelled as against *risk-free cash over the same period*, never as "you should have bought T-bills".
- **Every user-facing metric gets a `LearnTooltip`.** This app is a teaching tool.
- **CLAUDE.md is mirrored** in `egx-api-be/` and `egx-api-fe/` and the two copies must stay identical.

## Deviations from the spec

Three, all discovered by reading the code and all behaviour-preserving or safety-improving:

1. **`_DB.transaction()` is new (Task 1).** The spec assumed the sale insert and the quantity decrement could "commit together". They cannot today: `db.commit()` is a documented no-op and each `execute()` takes its own pooled autocommit connection, so a failure between the two statements would lose or duplicate shares.
2. **Sales maths lives in `core/sales.py`, not in the router.** `tests/test_fixes.py` has no DB or TestClient fixtures — every test is a pure function over synthetic data. Pure-core/thin-router is also the split the rest of this codebase uses.
3. **No portfolio-level annualized return.** The spec asked for "one T-bill line for the portfolio as a whole", but annualized returns over trades of different lengths cannot be averaged into an honest single number. Instead the summary reports `beat_t_bill_count` of `annualizable_count`, rendered as "3 of 7 closed trades beat the 25% T-bill" — a fact rather than a fabricated aggregate.

## File Structure

**Backend — create:**
- `app/core/returns.py` — position-level return maths shared by open and closed positions.
- `app/core/holdings.py` — the single spelling of the open-holdings query.
- `app/core/sales.py` — pure sale validation, per-sale metrics, portfolio summary.
- `app/routers/sales.py` — `POST/GET/DELETE /api/sales`.
- `tests/test_sell_tracking.py` — all backend tests for this feature.

**Backend — modify:**
- `app/core/db.py` — add `_Tx` and `_DB.transaction()`; add `portfolio_sales` to `init_db`.
- `app/routers/portfolio.py` — use `fetch_open_holdings` / `row_to_holding`.
- `app/routers/portfolio_analysis.py` — use `fetch_open_holdings`; import return helpers from `core/returns.py`.
- `app/main.py` — register the sales router.

**Frontend — create:**
- `src/app/components/SellHoldingForm.tsx`
- `src/app/components/RealizedGainsCard.tsx`
- `src/app/components/ClosedPositionsTable.tsx`

**Frontend — modify:**
- `src/app/lib/types.ts`, `src/app/lib/api.ts`
- `src/app/components/HoldingsTable.tsx` — `onSell` prop and two entry points.
- `src/app/portfolio/page.tsx` — wiring, sales state, empty-state fix.
- `src/app/learn/page.tsx` — `realized_gains` concept anchor.

**Docs — modify:** `egx-api-be/CLAUDE.md` and `egx-api-fe/CLAUDE.md` (identical edits).

---

### Task 1: Atomic multi-statement writes

**Files:**
- Modify: `egx-api-be/app/core/db.py:19-44`
- Test: `egx-api-be/tests/test_sell_tracking.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `db.transaction()` — a context manager yielding an object with `.execute(sql, params) -> _Result`. Every statement runs on one connection inside a real transaction; raising inside the block rolls back.

- [ ] **Step 1: Write the failing tests**

Create `egx-api-be/tests/test_sell_tracking.py`:

```python
"""
Tests for sell tracking and realized gains.

Run from egx-api-be:  python -m pytest tests/test_sell_tracking.py -v
"""

from contextlib import contextmanager
from datetime import date

import pytest

from app.core.db import _DB


# ---------------------------------------------------------------------------
# Fakes — the DB wrapper is tested without a real Postgres
# ---------------------------------------------------------------------------

class _FakeCursor:
    description = None

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, log):
        self.log = log
        self.committed = False
        self.rolled_back = False

    def execute(self, sql, params=()):
        self.log.append(sql)
        return _FakeCursor()

    @contextmanager
    def transaction(self):
        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        self.committed = True


class _FakePool:
    def __init__(self):
        self.conns = []
        self.log = []

    @contextmanager
    def connection(self):
        conn = _FakeConn(self.log)
        self.conns.append(conn)
        yield conn


def test_transaction_runs_every_statement_on_one_connection():
    # execute() takes a FRESH pooled connection per call, so a sale insert and
    # its quantity decrement issued through it could half-land.
    pool = _FakePool()
    db = _DB(pool)

    with db.transaction() as tx:
        tx.execute("UPDATE portfolio SET quantity = 1")
        tx.execute("INSERT INTO portfolio_sales VALUES (1)")

    assert len(pool.conns) == 1, "both statements must share one connection"
    assert pool.conns[0].committed
    assert len(pool.log) == 2


def test_transaction_rolls_back_on_error():
    pool = _FakePool()
    db = _DB(pool)

    with pytest.raises(RuntimeError):
        with db.transaction() as tx:
            tx.execute("UPDATE portfolio SET quantity = 1")
            raise RuntimeError("over-sold")

    assert pool.conns[0].rolled_back
    assert not pool.conns[0].committed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sell_tracking.py -v`
Expected: FAIL with `AttributeError: '_DB' object has no attribute 'transaction'`

- [ ] **Step 3: Implement**

In `app/core/db.py`, add the import at the top:

```python
from contextlib import contextmanager
```

Add `_Tx` directly after the `_Result` class:

```python
class _Tx:
    """Statement executor bound to one connection inside a transaction."""

    __slots__ = ("_conn",)

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=()):
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall() if cur.description else []
        return _Result(rows)
```

Add this method to `_DB`, after `commit()`:

```python
    @contextmanager
    def transaction(self):
        """
        Run several statements on ONE connection inside a real transaction.

        execute() takes a fresh pooled connection per call and the pool runs
        with autocommit, so two related writes issued through it can half-land
        — recording a sale without decrementing the holding would invent
        shares. Anything that must be all-or-nothing goes through here.
        """
        with self._pool.connection() as conn:
            with conn.transaction():
                yield _Tx(conn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sell_tracking.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/core/db.py tests/test_sell_tracking.py
git commit -m "feat(db): add transaction() for all-or-nothing writes"
```

---

### Task 2: Shared position-return helpers

**Files:**
- Create: `egx-api-be/app/core/returns.py`
- Modify: `egx-api-be/app/routers/portfolio_analysis.py:54-84`
- Test: `egx-api-be/tests/test_sell_tracking.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MIN_DAYS_FOR_ANNUALIZATION = 30`; `days_between(start_date_str: str, end: date) -> int`; `annualized_return(total_return_pct: float, days_held: int) -> Optional[float]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sell_tracking.py`:

```python
from app.core.returns import (
    MIN_DAYS_FOR_ANNUALIZATION,
    annualized_return,
    days_between,
)


def test_annualization_is_suppressed_below_thirty_days():
    # A +5% week annualizes to five figures. The UI must show nothing.
    assert annualized_return(5.0, 29) is None
    assert annualized_return(5.0, MIN_DAYS_FOR_ANNUALIZATION) is not None


def test_annualized_return_over_one_year_is_the_plain_return():
    assert annualized_return(10.0, 365) == pytest.approx(10.0, abs=0.01)


def test_two_year_gain_annualizes_below_the_t_bill():
    # The whole point of the T-bill line: +8% held two years LOST to cash.
    ann = annualized_return(8.0, 730)
    assert ann == pytest.approx(3.92, abs=0.05)
    assert ann < 25.0


def test_total_loss_floors_at_minus_one_hundred():
    assert annualized_return(-100.0, 365) == -100.0
    assert annualized_return(-150.0, 365) == -100.0


def test_days_between_counts_calendar_days():
    assert days_between("2026-01-01", date(2026, 1, 31)) == 30
    assert days_between("2026-01-01T00:00:00Z", date(2026, 1, 31)) == 30


def test_days_between_returns_zero_on_unparseable_input():
    assert days_between("not-a-date", date(2026, 1, 31)) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sell_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.returns'`

- [ ] **Step 3: Create `app/core/returns.py`**

```python
"""
Position-level return maths.

Shared by portfolio analysis (open positions) and the sales ledger (closed
ones) so the two can never annualize differently — a realized win and an
unrealized one must be judged against the T-bill by the same formula.
"""

from datetime import date, datetime
from typing import Optional

# Below this many days held, annualizing a position's return produces
# nonsense (a +5% week annualizes to five figures). The signal layer and the
# UI both suppress the number instead of showing it.
MIN_DAYS_FOR_ANNUALIZATION = 30


def days_between(start_date_str: str, end: date) -> int:
    """Calendar days from an ISO date string to `end`. 0 if unparseable."""
    try:
        d = datetime.strptime(start_date_str[:10], "%Y-%m-%d").date()
        return (end - d).days
    except Exception:
        return 0


def annualized_return(total_return_pct: float, days_held: int) -> Optional[float]:
    """
    Annualize a POSITION's return over calendar days held.

    Note this is a different quantity from indicators.annualized_return(),
    which annualizes the STOCK's market return over trading bars. The two
    answer different questions ("how did my purchase do" vs "how did the
    stock do") and must not be compared to each other.

    Returns None when the position was held too briefly to annualize
    meaningfully.
    """
    if days_held < MIN_DAYS_FOR_ANNUALIZATION:
        return None
    base = 1 + total_return_pct / 100
    if base <= 0:
        return -100.0
    return (base ** (365 / days_held) - 1) * 100
```

- [ ] **Step 4: Point `portfolio_analysis.py` at the shared module**

Delete lines 54-84 of `app/routers/portfolio_analysis.py` (the `_days_between` function, the `MIN_DAYS_FOR_ANNUALIZATION` constant with its comment, and the `_annualized_return` function). Replace them with aliases so the ~40 existing call sites in that file need no edit:

```python
from app.core.returns import (  # noqa: F401  (MIN_DAYS_FOR_ANNUALIZATION re-exported)
    MIN_DAYS_FOR_ANNUALIZATION,
    annualized_return as _annualized_return,
    days_between as _days_between,
)
```

Put this with the other `app.core` imports at the top of the file, not at line 54.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all pass, including the pre-existing `test_fixes.py`. If `portfolio_analysis.py` fails to import, check that `date` and `datetime` are still imported there for their other uses.

- [ ] **Step 6: Commit**

```bash
git add app/core/returns.py app/routers/portfolio_analysis.py tests/test_sell_tracking.py
git commit -m "refactor: move position-return helpers to core/returns.py"
```

---

### Task 3: One spelling of the open-holdings query

**Files:**
- Create: `egx-api-be/app/core/holdings.py`
- Modify: `egx-api-be/app/routers/portfolio.py:23-57,126-173`
- Modify: `egx-api-be/app/routers/portfolio_analysis.py:1088-1106`
- Test: `egx-api-be/tests/test_sell_tracking.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `HOLDING_COLUMNS: str`; `row_to_holding(row) -> dict`; `fetch_open_holdings(db, user_id: str) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sell_tracking.py`:

```python
from pathlib import Path

_ROUTERS = Path(__file__).resolve().parents[1] / "app" / "routers"


def test_open_holdings_filter_is_spelled_once():
    """
    portfolio.py and portfolio_analysis.py each issue their own SELECT against
    the portfolio table — portfolio_analysis does NOT call /api/portfolio. If
    either hand-writes the `quantity > 0` filter, the two can drift and a
    fully-sold holding reappears in the risk metrics as a phantom position.
    """
    for name in ("portfolio.py", "portfolio_analysis.py"):
        src = (_ROUTERS / name).read_text(encoding="utf-8")
        assert "quantity > 0" not in src, (
            f"{name} hand-writes the open-holdings filter; "
            "call core.holdings.fetch_open_holdings instead"
        )
        assert "fetch_open_holdings" in src, (
            f"{name} must read holdings through core.holdings.fetch_open_holdings"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sell_tracking.py::test_open_holdings_filter_is_spelled_once -v`
Expected: FAIL — `portfolio.py must read holdings through core.holdings.fetch_open_holdings`

- [ ] **Step 3: Create `app/core/holdings.py`**

```python
"""
The one place the "open holdings" query is spelled.

A holding with quantity 0 is fully sold. The row is RETAINED so a sale can be
undone against it — restoring the position with its target price, stop loss
and notes intact — but it must never appear in the portfolio list or reach
portfolio analysis, where it would show as a phantom position holding zero
shares.

portfolio.py and portfolio_analysis.py both read the table directly
(portfolio_analysis does not call /api/portfolio), so the filter lives here
and both call it. tests/test_sell_tracking.py fails if either grows its own.

By-id lookups stay inline in their routers: PUT and DELETE need them, and
sales.py needs one that deliberately IGNORES this filter so a sale can be
undone against a fully-closed holding.
"""

HOLDING_COLUMNS = (
    "id, symbol, name, buy_price, buy_date, quantity, notes, sector, "
    "target_price, stop_loss, created_at, updated_at"
)


def row_to_holding(row) -> dict:
    """Map a HOLDING_COLUMNS row tuple to the holding dict the API returns."""
    return {
        "id": row[0],
        "symbol": row[1],
        "name": row[2],
        "buy_price": row[3],
        "buy_date": row[4],
        "quantity": row[5],
        "notes": row[6],
        "sector": row[7],
        "target_price": row[8],
        "stop_loss": row[9],
        "created_at": row[10],
        "updated_at": row[11],
    }


def fetch_open_holdings(db, user_id: str) -> list[dict]:
    """Every holding the user still owns at least one share of."""
    rows = db.execute(
        f"SELECT {HOLDING_COLUMNS} FROM portfolio "
        "WHERE user_id = %s AND quantity > 0 ORDER BY created_at",
        (user_id,),
    ).fetchall()
    return [row_to_holding(r) for r in rows]
```

- [ ] **Step 4: Rewire `portfolio.py`**

Add to its imports:

```python
from app.core.holdings import HOLDING_COLUMNS, fetch_open_holdings, row_to_holding
```

Delete the local `_row_to_dict` function (lines 23-37). Replace its three call sites with `row_to_holding`.

Replace the body of `get_portfolio` (lines 42-57) with:

```python
        db = get_db()
        holdings = fetch_open_holdings(db, user.id)

        settings = db.execute(
            "SELECT value FROM settings WHERE key = 'currency'"
        ).fetchone()
        currency = settings[0] if settings else "EGP"

        return {"portfolio": holdings, "currency": currency}
```

Note the SELECT is corrected to `SELECT value` — the original selected `key, value` then read `settings[0]`, which returned the string `"currency"` rather than `"EGP"`. Fixing it here is in scope because this is the query being rewritten.

In `update_holding`, replace both inline `SELECT id, symbol, ... FROM portfolio` column lists with `f"SELECT {HOLDING_COLUMNS} FROM portfolio WHERE id = %s AND user_id = %s"`.

- [ ] **Step 5: Rewire `portfolio_analysis.py`**

Add to its imports:

```python
from app.core.holdings import fetch_open_holdings
```

Replace the body of `get_portfolio_analysis` (lines 1091-1106) with:

```python
        db = get_db()
        holdings = fetch_open_holdings(db, user.id)
        return _analyze(holdings)
```

`_analyze` reads fields with `h.get(...)`, so the extra `created_at` / `updated_at` keys are harmless.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 7: Manual verification**

Start the API (`uvicorn app.main:app --reload --port 8000`), then confirm `GET /api/portfolio` still returns your holdings and that `currency` now reads `"EGP"` rather than `"currency"`.

- [ ] **Step 8: Commit**

```bash
git add app/core/holdings.py app/routers/portfolio.py app/routers/portfolio_analysis.py tests/test_sell_tracking.py
git commit -m "refactor: single source for the open-holdings query"
```

---

### Task 4: Sales maths and validation (pure)

**Files:**
- Create: `egx-api-be/app/core/sales.py`
- Test: `egx-api-be/tests/test_sell_tracking.py`

**Interfaces:**
- Consumes: `app.core.returns.annualized_return`, `days_between`.
- Produces:
  - `class SaleValidationError(ValueError)`
  - `validate_sale(*, holding: dict, quantity, sell_price, sell_date, today: date) -> dict` returning `{"quantity": int, "sell_price": float, "sell_date": str}`
  - `compute_sale_metrics(sale: dict, risk_free_rate_pct: float) -> dict`
  - `summarize_sales(priced_sales: list[dict]) -> dict` — takes sales already priced by `compute_sale_metrics`, so it needs no rate of its own

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sell_tracking.py`:

```python
from app.core.sales import (
    SaleValidationError,
    compute_sale_metrics,
    summarize_sales,
    validate_sale,
)

TODAY = date(2026, 9, 1)


def _holding(quantity=100, buy_price=50.0, buy_date="2026-01-01"):
    return {
        "id": "h1", "symbol": "COMI", "name": "Commercial International Bank",
        "sector": "Banks", "quantity": quantity, "buy_price": buy_price,
        "buy_date": buy_date,
    }


def _sale(quantity=100, buy_price=50.0, sell_price=60.0,
          buy_date="2026-01-01", sell_date="2026-09-01", symbol="COMI"):
    return {
        "id": "s1", "holding_id": "h1", "symbol": symbol,
        "name": "Commercial International Bank", "sector": "Banks",
        "quantity": quantity, "buy_price": buy_price, "buy_date": buy_date,
        "sell_price": sell_price, "sell_date": sell_date, "notes": "",
        "created_at": "2026-09-01T00:00:00Z",
    }


# ---- validation ----

def test_partial_sell_is_accepted():
    out = validate_sale(holding=_holding(quantity=100), quantity=40,
                        sell_price=60.0, sell_date="2026-09-01", today=TODAY)
    assert out == {"quantity": 40, "sell_price": 60.0, "sell_date": "2026-09-01"}


def test_selling_the_whole_position_is_accepted():
    out = validate_sale(holding=_holding(quantity=100), quantity=100,
                        sell_price=60.0, sell_date="2026-09-01", today=TODAY)
    assert out["quantity"] == 100


def test_over_selling_is_rejected_and_names_the_remaining_quantity():
    with pytest.raises(SaleValidationError) as exc:
        validate_sale(holding=_holding(quantity=100), quantity=101,
                      sell_price=60.0, sell_date="2026-09-01", today=TODAY)
    assert "100" in str(exc.value)


@pytest.mark.parametrize("bad_quantity", [0, -5, "abc", None, 1.5])
def test_non_positive_or_non_integer_quantity_is_rejected(bad_quantity):
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=bad_quantity,
                      sell_price=60.0, sell_date="2026-09-01", today=TODAY)


@pytest.mark.parametrize("bad_price", [0, -1.0, "abc", None])
def test_non_positive_sell_price_is_rejected(bad_price):
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=10,
                      sell_price=bad_price, sell_date="2026-09-01", today=TODAY)


def test_sell_date_before_buy_date_is_rejected():
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(buy_date="2026-06-01"), quantity=10,
                      sell_price=60.0, sell_date="2026-05-31", today=TODAY)


def test_future_sell_date_is_rejected():
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=10, sell_price=60.0,
                      sell_date="2026-09-02", today=TODAY)


def test_sell_date_defaults_to_today():
    out = validate_sale(holding=_holding(), quantity=10, sell_price=60.0,
                        sell_date=None, today=TODAY)
    assert out["sell_date"] == "2026-09-01"


def test_unparseable_sell_date_is_rejected():
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=10, sell_price=60.0,
                      sell_date="tomorrow", today=TODAY)


# ---- per-sale metrics ----

def test_realized_pnl_and_pct():
    m = compute_sale_metrics(_sale(quantity=100, buy_price=50.0, sell_price=60.0),
                             risk_free_rate_pct=25.0)
    assert m["cost"] == 5000.0
    assert m["proceeds"] == 6000.0
    assert m["realized_pnl"] == 1000.0
    assert m["realized_pnl_pct"] == pytest.approx(20.0)


def test_a_loss_is_negative_not_absolute():
    m = compute_sale_metrics(_sale(buy_price=60.0, sell_price=50.0),
                             risk_free_rate_pct=25.0)
    assert m["realized_pnl"] == -1000.0
    assert m["realized_pnl_pct"] == pytest.approx(-16.67, abs=0.01)


def test_a_two_year_eight_percent_win_did_not_beat_the_t_bill():
    # The reason the T-bill line exists at all.
    m = compute_sale_metrics(
        _sale(buy_price=100.0, sell_price=108.0,
              buy_date="2024-09-01", sell_date="2026-09-01"),
        risk_free_rate_pct=25.0,
    )
    assert m["realized_pnl"] > 0
    assert m["beat_t_bill"] is False


def test_a_quick_flip_reports_no_annualized_figure():
    m = compute_sale_metrics(
        _sale(buy_date="2026-08-20", sell_date="2026-09-01"),
        risk_free_rate_pct=25.0,
    )
    assert m["days_held"] == 12
    assert m["annualized_return_pct"] is None
    assert m["beat_t_bill"] is None


def test_zero_buy_price_reports_null_pct_but_exact_egp():
    m = compute_sale_metrics(_sale(buy_price=0.0, sell_price=60.0),
                             risk_free_rate_pct=25.0)
    assert m["realized_pnl_pct"] is None
    assert m["realized_pnl"] == 6000.0


def test_metrics_preserve_the_original_sale_fields():
    m = compute_sale_metrics(_sale(), risk_free_rate_pct=25.0)
    assert m["id"] == "s1" and m["symbol"] == "COMI"


# ---- summary ----

def test_empty_summary_is_zeroed_not_null():
    s = summarize_sales([])
    assert s["total_realized_pnl"] == 0
    assert s["total_realized_pnl_pct"] is None
    assert s["win_count"] == 0 and s["loss_count"] == 0
    assert s["by_symbol"] == []
    assert s["best_trade"] is None and s["worst_trade"] is None


def test_total_pct_is_cost_weighted_not_a_mean_of_percentages():
    # 10000 cost -> +1000 (+10%), 1000 cost -> +500 (+50%).
    # Mean of percentages says +30%. Cost-weighted truth is +13.64%.
    priced = [
        compute_sale_metrics(_sale(quantity=100, buy_price=100.0, sell_price=110.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=100.0, sell_price=150.0,
                                   symbol="SWDY"), 25.0),
    ]
    s = summarize_sales(priced)
    assert s["total_realized_pnl"] == 1500.0
    assert s["total_realized_pnl_pct"] == pytest.approx(13.64, abs=0.01)


def test_wins_and_losses_are_counted_and_extremes_identified():
    priced = [
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=60.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=40.0,
                                   symbol="SWDY"), 25.0),
    ]
    s = summarize_sales(priced)
    assert s["win_count"] == 1 and s["loss_count"] == 1
    assert s["best_trade"]["symbol"] == "COMI"
    assert s["worst_trade"]["symbol"] == "SWDY"


def test_by_symbol_aggregates_multiple_sales_and_sorts_by_pnl():
    priced = [
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=60.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=70.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=55.0,
                                   symbol="SWDY"), 25.0),
    ]
    s = summarize_sales(priced)
    assert [b["symbol"] for b in s["by_symbol"]] == ["COMI", "SWDY"]
    comi = s["by_symbol"][0]
    assert comi["sales_count"] == 2 and comi["quantity"] == 20
    assert comi["realized_pnl"] == 300.0
    assert comi["realized_pnl_pct"] == pytest.approx(30.0)


def test_t_bill_counts_ignore_trades_too_short_to_annualize():
    priced = [
        # Two years, +8% — annualizable, loses to the T-bill.
        compute_sale_metrics(_sale(buy_price=100.0, sell_price=108.0,
                                   buy_date="2024-09-01", sell_date="2026-09-01"), 25.0),
        # One year, +40% — annualizable, beats it.
        compute_sale_metrics(_sale(buy_price=100.0, sell_price=140.0,
                                   buy_date="2025-09-01", sell_date="2026-09-01"), 25.0),
        # Twelve days — not annualizable, must not count either way.
        compute_sale_metrics(_sale(buy_date="2026-08-20", sell_date="2026-09-01"), 25.0),
    ]
    s = summarize_sales(priced)
    assert s["annualizable_count"] == 2
    assert s["beat_t_bill_count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sell_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.sales'`

- [ ] **Step 3: Create `app/core/sales.py`**

```python
"""
Realized-gains maths for the sales ledger.

Pure functions, no DB access — routers/sales.py does the IO. Kept pure so the
whole surface is unit-testable, since tests/ has no Postgres fixture.

A note on the T-bill comparison: with Egyptian T-bills near 25%, a modest gain
held for a long time is a real-terms LOSS against risk-free cash. Every closed
position therefore reports its annualized return, and the frontend shows it
next to the T-bill rate — the same lesson the cash_underperformer signal
delivers for open positions, applied to closed ones.
"""

from datetime import date, datetime
from typing import Optional

from app.core.returns import annualized_return, days_between


class SaleValidationError(ValueError):
    """A sell request that is not internally consistent. Maps to HTTP 400."""


def _parse_date(value: str) -> Optional[date]:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def validate_sale(*, holding: dict, quantity, sell_price, sell_date, today: date) -> dict:
    """
    Check a sell request against the holding it is against.

    Returns normalized {quantity, sell_price, sell_date}. Raises
    SaleValidationError with a message written for the user, not the log.
    """
    held = int(holding.get("quantity") or 0)

    if isinstance(quantity, bool) or not isinstance(quantity, int):
        try:
            if float(quantity) != int(float(quantity)):
                raise ValueError
            quantity = int(float(quantity))
        except (TypeError, ValueError):
            raise SaleValidationError("Quantity must be a whole number of shares.")
    if quantity <= 0:
        raise SaleValidationError("Quantity must be at least 1 share.")
    if quantity > held:
        raise SaleValidationError(
            f"You hold {held} shares — you cannot sell {quantity}."
        )

    try:
        sell_price = float(sell_price)
    except (TypeError, ValueError):
        raise SaleValidationError("Sell price must be a number.")
    if sell_price <= 0:
        raise SaleValidationError("Sell price must be greater than 0.")

    if sell_date in (None, ""):
        parsed = today
    else:
        parsed = _parse_date(sell_date)
        if parsed is None:
            raise SaleValidationError("Sell date must be a date like 2026-09-01.")
    if parsed > today:
        raise SaleValidationError("Sell date cannot be in the future.")

    buy = _parse_date(holding.get("buy_date") or "")
    if buy is not None and parsed < buy:
        raise SaleValidationError(
            f"Sell date cannot be before the buy date ({buy.isoformat()})."
        )

    return {
        "quantity": quantity,
        "sell_price": sell_price,
        "sell_date": parsed.isoformat(),
    }


def compute_sale_metrics(sale: dict, risk_free_rate_pct: float) -> dict:
    """Add realized P&L, holding period and the T-bill verdict to one sale."""
    quantity = int(sale["quantity"])
    buy_price = float(sale["buy_price"])
    sell_price = float(sale["sell_price"])

    cost = buy_price * quantity
    proceeds = sell_price * quantity
    realized_pnl = proceeds - cost

    # A zero cost basis has no meaningful percentage, but the EGP figure is
    # still exact — report the number we have and null the one we don't.
    realized_pnl_pct = (sell_price / buy_price - 1) * 100 if buy_price > 0 else None

    sold_on = _parse_date(sale["sell_date"])
    days_held = days_between(sale["buy_date"], sold_on) if sold_on else 0

    ann = (
        annualized_return(realized_pnl_pct, days_held)
        if realized_pnl_pct is not None
        else None
    )

    return {
        **sale,
        "cost": round(cost, 2),
        "proceeds": round(proceeds, 2),
        "realized_pnl": round(realized_pnl, 2),
        "realized_pnl_pct": round(realized_pnl_pct, 2) if realized_pnl_pct is not None else None,
        "days_held": days_held,
        "annualized_return_pct": round(ann, 1) if ann is not None else None,
        "beat_t_bill": (ann > risk_free_rate_pct) if ann is not None else None,
    }


def summarize_sales(priced_sales: list) -> dict:
    """
    Roll priced sales up into the Winnings card's numbers.

    Takes sales already through compute_sale_metrics, which is where the
    T-bill comparison happened — so this needs no rate of its own.

    total_realized_pnl_pct is cost-weighted (total P&L over total cost), never
    a mean of percentages — a +50% gain on 1,000 EGP and a +10% gain on
    10,000 EGP is +13.6% overall, not +30%.

    There is deliberately NO portfolio-level annualized return: annualized
    figures over trades of different lengths cannot be averaged into an honest
    single number. beat_t_bill_count / annualizable_count reports the fact
    instead.
    """
    if not priced_sales:
        return {
            "total_realized_pnl": 0.0,
            "total_realized_pnl_pct": None,
            "total_proceeds": 0.0,
            "total_cost": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "beat_t_bill_count": 0,
            "annualizable_count": 0,
            "best_trade": None,
            "worst_trade": None,
            "by_symbol": [],
        }

    total_cost = sum(s["cost"] for s in priced_sales)
    total_proceeds = sum(s["proceeds"] for s in priced_sales)
    total_pnl = total_proceeds - total_cost

    annualizable = [s for s in priced_sales if s["beat_t_bill"] is not None]

    by_symbol: dict = {}
    for s in priced_sales:
        agg = by_symbol.setdefault(
            s["symbol"],
            {
                "symbol": s["symbol"], "name": s.get("name") or s["symbol"],
                "sector": s.get("sector") or "", "sales_count": 0,
                "quantity": 0, "cost": 0.0, "proceeds": 0.0,
            },
        )
        agg["sales_count"] += 1
        agg["quantity"] += int(s["quantity"])
        agg["cost"] += s["cost"]
        agg["proceeds"] += s["proceeds"]

    rollup = []
    for agg in by_symbol.values():
        pnl = agg["proceeds"] - agg["cost"]
        rollup.append({
            **agg,
            "cost": round(agg["cost"], 2),
            "proceeds": round(agg["proceeds"], 2),
            "realized_pnl": round(pnl, 2),
            "realized_pnl_pct": round(pnl / agg["cost"] * 100, 2) if agg["cost"] > 0 else None,
        })
    rollup.sort(key=lambda r: r["realized_pnl"], reverse=True)

    return {
        "total_realized_pnl": round(total_pnl, 2),
        "total_realized_pnl_pct": round(total_pnl / total_cost * 100, 2) if total_cost > 0 else None,
        "total_proceeds": round(total_proceeds, 2),
        "total_cost": round(total_cost, 2),
        "win_count": sum(1 for s in priced_sales if s["realized_pnl"] > 0),
        "loss_count": sum(1 for s in priced_sales if s["realized_pnl"] < 0),
        "beat_t_bill_count": sum(1 for s in annualizable if s["beat_t_bill"]),
        "annualizable_count": len(annualizable),
        "best_trade": max(priced_sales, key=lambda s: s["realized_pnl"]),
        "worst_trade": min(priced_sales, key=lambda s: s["realized_pnl"]),
        "by_symbol": rollup,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sell_tracking.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/core/sales.py tests/test_sell_tracking.py
git commit -m "feat: realized-gains maths and sell validation"
```

---

### Task 5: `portfolio_sales` schema

**Files:**
- Modify: `egx-api-be/app/core/db.py` (inside `init_db`, after the `portfolio` table block at line 74-90)

**Interfaces:**
- Consumes: nothing.
- Produces: the `portfolio_sales` table and its `user_id` index.

- [ ] **Step 1: Add the table**

There is no migration framework — every statement in `init_db` is idempotent and new tables land on the next cold start of any process. Insert directly after the `idx_portfolio_user` index line:

```python
    # Append-only ledger of sales. Cost basis is SNAPSHOTTED (buy_price,
    # buy_date, name, sector) rather than joined from portfolio: a sale is a
    # historical fact and must not change when the user later edits or deletes
    # the holding it came from. Same principle as fundamentals_history.
    db.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_sales (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            holding_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT DEFAULT '',
            quantity INTEGER NOT NULL,
            buy_price DOUBLE PRECISION NOT NULL,
            buy_date TEXT NOT NULL,
            sell_price DOUBLE PRECISION NOT NULL,
            sell_date TEXT NOT NULL,
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_sales_user "
        "ON portfolio_sales(user_id)"
    )
```

- [ ] **Step 2: Verify the schema lands**

Start the API (`uvicorn app.main:app --reload --port 8000`) so `init_db` runs, then confirm against your Neon database:

```sql
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'portfolio_sales' ORDER BY ordinal_position;
```

Expected: 13 columns in the order above.

- [ ] **Step 3: Commit**

```bash
git add app/core/db.py
git commit -m "feat(db): add portfolio_sales ledger table"
```

---

### Task 6: `/api/sales` endpoints

**Files:**
- Create: `egx-api-be/app/routers/sales.py`
- Modify: `egx-api-be/app/main.py:15-30,56-72`

**Interfaces:**
- Consumes: `db.transaction()`, `core.sales.*`, `core.holdings.HOLDING_COLUMNS`, `core.constants.DEFAULT_RISK_FREE_RATE_PCT`.
- Produces: `POST /api/sales`, `GET /api/sales`, `DELETE /api/sales?id=`.

- [ ] **Step 1: Create `app/routers/sales.py`**

```python
"""
Sales ledger — record what was sold and report realized gains.

POST   /api/sales          — record a full or partial sell
GET    /api/sales          — every sale plus the realized-gains summary
DELETE /api/sales?id=xxx   — undo a sale, restoring the shares

Deliberately separate from /api/portfolio_analysis, which is the heaviest
endpoint in the app and flirts with the 30 s Vercel timeout. Realized gains
need NO price fetch, so the Winnings card paints even on a run where the
analysis times out.

Every route is scoped by the caller's user_id from the JWT.
"""

import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import CurrentUser, get_current_user
from app.core.constants import DEFAULT_RISK_FREE_RATE_PCT
from app.core.db import get_db
from app.core.holdings import HOLDING_COLUMNS, row_to_holding
from app.core.sales import (
    SaleValidationError,
    compute_sale_metrics,
    summarize_sales,
    validate_sale,
)

router = APIRouter()

_SALE_COLUMNS = (
    "id, holding_id, symbol, name, sector, quantity, buy_price, buy_date, "
    "sell_price, sell_date, notes, created_at"
)


def _row_to_sale(row) -> dict:
    return {
        "id": row[0], "holding_id": row[1], "symbol": row[2], "name": row[3],
        "sector": row[4], "quantity": row[5], "buy_price": row[6],
        "buy_date": row[7], "sell_price": row[8], "sell_date": row[9],
        "notes": row[10], "created_at": row[11],
    }


def _risk_free_rate_pct(db) -> float:
    try:
        row = db.execute(
            "SELECT value FROM settings WHERE key = 'risk_free_rate'"
        ).fetchone()
        return float(row[0]) if row else DEFAULT_RISK_FREE_RATE_PCT
    except Exception:
        return DEFAULT_RISK_FREE_RATE_PCT


@router.post("/api/sales", status_code=201)
def record_sale(body: dict, user: CurrentUser = Depends(get_current_user)):
    try:
        holding_id = body.get("holding_id")
        if not holding_id:
            raise HTTPException(status_code=400, detail="Missing required field: holding_id")

        db = get_db()

        # Deliberately NOT fetch_open_holdings: this is a by-id lookup and it
        # must see the row regardless of remaining quantity so the error
        # message can say "you hold 0 shares" rather than "not found".
        row = db.execute(
            f"SELECT {HOLDING_COLUMNS} FROM portfolio WHERE id = %s AND user_id = %s",
            (holding_id, user.id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Holding not found: {holding_id}")
        holding = row_to_holding(row)

        clean = validate_sale(
            holding=holding,
            quantity=body.get("quantity"),
            sell_price=body.get("sell_price"),
            sell_date=body.get("sell_date"),
            today=date.today(),
        )

        now = datetime.utcnow().isoformat() + "Z"
        sale_id = str(uuid.uuid4())

        with db.transaction() as tx:
            # `quantity >= %s` in the WHERE clause makes the decrement itself
            # the over-sell guard, so two rapid submits cannot both succeed.
            updated = tx.execute(
                "UPDATE portfolio SET quantity = quantity - %s, updated_at = %s "
                "WHERE id = %s AND user_id = %s AND quantity >= %s "
                "RETURNING quantity",
                (clean["quantity"], now, holding_id, user.id, clean["quantity"]),
            ).fetchone()
            if updated is None:
                raise SaleValidationError(
                    f"You hold {holding['quantity']} shares — "
                    f"you cannot sell {clean['quantity']}."
                )

            tx.execute(
                f"INSERT INTO portfolio_sales ({_SALE_COLUMNS}, user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    sale_id, holding_id, holding["symbol"], holding["name"],
                    holding.get("sector") or "", clean["quantity"],
                    float(holding["buy_price"]), holding["buy_date"],
                    clean["sell_price"], clean["sell_date"],
                    body.get("notes", ""), now, user.id,
                ),
            )
            remaining = updated[0]

        sale = {
            "id": sale_id, "holding_id": holding_id, "symbol": holding["symbol"],
            "name": holding["name"], "sector": holding.get("sector") or "",
            "quantity": clean["quantity"], "buy_price": float(holding["buy_price"]),
            "buy_date": holding["buy_date"], "sell_price": clean["sell_price"],
            "sell_date": clean["sell_date"], "notes": body.get("notes", ""),
            "created_at": now,
        }
        return {
            "sale": compute_sale_metrics(sale, _risk_free_rate_pct(db)),
            "holding": {**holding, "quantity": remaining, "updated_at": now},
        }

    except SaleValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sales")
def get_sales(user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        rows = db.execute(
            f"SELECT {_SALE_COLUMNS} FROM portfolio_sales "
            "WHERE user_id = %s ORDER BY sell_date DESC, created_at DESC",
            (user.id,),
        ).fetchall()

        rfr = _risk_free_rate_pct(db)
        priced = [compute_sale_metrics(_row_to_sale(r), rfr) for r in rows]

        currency_row = db.execute(
            "SELECT value FROM settings WHERE key = 'currency'"
        ).fetchone()

        return {
            "sales": priced,
            "summary": summarize_sales(priced),
            "currency": currency_row[0] if currency_row else "EGP",
            "risk_free_rate_pct": rfr,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/sales")
def delete_sale(id: str = Query(...), user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        with db.transaction() as tx:
            row = tx.execute(
                "SELECT holding_id, quantity FROM portfolio_sales "
                "WHERE id = %s AND user_id = %s",
                (id, user.id),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Sale not found: {id}")
            holding_id, quantity = row[0], int(row[1])

            tx.execute(
                "DELETE FROM portfolio_sales WHERE id = %s AND user_id = %s",
                (id, user.id),
            )
            # If the user hard-deleted the holding, there is nothing to restore
            # the shares to — the sale still goes, and restored stays None.
            restored = tx.execute(
                "UPDATE portfolio SET quantity = quantity + %s, updated_at = %s "
                "WHERE id = %s AND user_id = %s RETURNING quantity",
                (quantity, datetime.utcnow().isoformat() + "Z", holding_id, user.id),
            ).fetchone()

        return {
            "deleted": id,
            "holding_id": holding_id,
            "restored_quantity": restored[0] if restored else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 2: Register the router**

In `app/main.py`, add `sales,` to the `from app.routers import (...)` block (after `portfolio_analysis,`) and to the `for router_module in (...)` tuple in the same position.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all pass — nothing here changes existing behaviour.

- [ ] **Step 4: Manual end-to-end verification**

With the API running and a valid bearer token in `$TOKEN`, against a holding of 100 shares:

```bash
curl -s -X POST localhost:8000/api/sales -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"holding_id":"<id>","quantity":40,"sell_price":60,"sell_date":"2026-09-01"}'
```

Expected: 201; `holding.quantity` is 60; `sale.realized_pnl` matches `(60 - buy_price) * 40`.

Then check that over-selling is refused and that `GET /api/sales` totals are right:

```bash
curl -s -X POST localhost:8000/api/sales -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"holding_id":"<id>","quantity":999,"sell_price":60}'
curl -s localhost:8000/api/sales -H "Authorization: Bearer $TOKEN"
```

Expected: 400 naming 60 remaining shares; then the sale listed with a correct summary.

Finally confirm undo restores the position exactly:

```bash
curl -s -X DELETE "localhost:8000/api/sales?id=<sale_id>" -H "Authorization: Bearer $TOKEN"
curl -s localhost:8000/api/portfolio -H "Authorization: Bearer $TOKEN"
```

Expected: `restored_quantity` 100, and the holding is back to 100 shares.

- [ ] **Step 5: Commit**

```bash
git add app/routers/sales.py app/main.py
git commit -m "feat(api): sales ledger endpoints"
```

---

### Task 7: Frontend types and API wrappers

**Files:**
- Modify: `egx-api-fe/src/app/lib/types.ts` (append after the Portfolio section, ~line 510)
- Modify: `egx-api-fe/src/app/lib/api.ts` (append after `fetchPortfolioAnalysis`, ~line 147)

**Interfaces:**
- Consumes: the `/api/sales` response shape from Task 6.
- Produces: `Sale`, `SymbolRealized`, `SalesSummary`, `SalesResponse`; `fetchSales()`, `recordSale(body)`, `deleteSale(id)`.

- [ ] **Step 1: Add the types**

Append to `src/app/lib/types.ts`:

```typescript
// ============================================================
// Sales / Realized gains
// ============================================================

export interface Sale {
  id: string;
  holding_id: string;
  symbol: string;
  name: string;
  sector: string;
  quantity: number;
  buy_price: number;
  buy_date: string;
  sell_price: number;
  sell_date: string;
  notes: string;
  created_at: string;
  cost: number;
  proceeds: number;
  realized_pnl: number;
  /** Null when the buy price was 0 — the EGP figure is still exact. */
  realized_pnl_pct: number | null;
  days_held: number;
  /** Null under 30 days held: annualizing a quick flip is nonsense. */
  annualized_return_pct: number | null;
  /** Null whenever annualized_return_pct is null. */
  beat_t_bill: boolean | null;
}

export interface SymbolRealized {
  symbol: string;
  name: string;
  sector: string;
  sales_count: number;
  quantity: number;
  cost: number;
  proceeds: number;
  realized_pnl: number;
  realized_pnl_pct: number | null;
}

export interface SalesSummary {
  total_realized_pnl: number;
  /** Cost-weighted, never a mean of percentages. */
  total_realized_pnl_pct: number | null;
  total_proceeds: number;
  total_cost: number;
  win_count: number;
  loss_count: number;
  /** Of the trades long enough to annualize, how many beat the T-bill. */
  beat_t_bill_count: number;
  annualizable_count: number;
  best_trade: Sale | null;
  worst_trade: Sale | null;
  by_symbol: SymbolRealized[];
}

export interface SalesResponse {
  sales: Sale[];
  summary: SalesSummary;
  currency: string;
  risk_free_rate_pct: number;
}
```

- [ ] **Step 2: Add the API wrappers**

In `src/app/lib/api.ts`, add `Sale`, `SalesResponse`, `PortfolioHolding` to the type imports if not present, then append:

```typescript
// ---- Sales / Realized gains ----

export async function fetchSales(): Promise<SalesResponse> {
  return fetchJSON<SalesResponse>(`${BASE}/sales`);
}

export async function recordSale(body: {
  holding_id: string;
  quantity: number;
  sell_price: number;
  sell_date: string;
  notes?: string;
}): Promise<{ sale: Sale; holding: PortfolioHolding }> {
  return fetchJSON<{ sale: Sale; holding: PortfolioHolding }>(`${BASE}/sales`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function deleteSale(
  id: string
): Promise<{ deleted: string; holding_id: string; restored_quantity: number | null }> {
  return fetchJSON(`${BASE}/sales?id=${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}
```

- [ ] **Step 3: Typecheck**

Run from `egx-api-fe`: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/app/lib/types.ts src/app/lib/api.ts
git commit -m "feat(fe): sales types and API wrappers"
```

---

### Task 8: `SellHoldingForm` and the Sell entry points

**Files:**
- Create: `egx-api-fe/src/app/components/SellHoldingForm.tsx`
- Modify: `egx-api-fe/src/app/components/HoldingsTable.tsx:92-105,280-295,449-463`

**Interfaces:**
- Consumes: `HoldingAnalysis` (for the row), `PortfolioHolding`.
- Produces: `<SellHoldingForm holding onSubmit onCancel />` where `onSubmit(data: {quantity: number; sell_price: number; sell_date: string; notes: string}) => Promise<void> | void`; `HoldingsTable` gains `onSell: (id: string) => void`.

- [ ] **Step 1: Create the form**

```tsx
"use client";

import { useState } from "react";

interface SellHoldingFormProps {
  holding: {
    symbol: string;
    name: string;
    quantity: number;
    buy_price: number;
  };
  onSubmit: (data: {
    quantity: number;
    sell_price: number;
    sell_date: string;
    notes: string;
  }) => Promise<void> | void;
  onCancel?: () => void;
}

export default function SellHoldingForm({
  holding,
  onSubmit,
  onCancel,
}: SellHoldingFormProps) {
  // Pre-filled to the whole position: selling out entirely is the common case.
  const [quantity, setQuantity] = useState(holding.quantity.toString());
  const [sellPrice, setSellPrice] = useState("");
  const [sellDate, setSellDate] = useState(new Date().toISOString().slice(0, 10));
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const qty = parseInt(quantity);
  const price = parseFloat(sellPrice);
  const validQty = Number.isFinite(qty) && qty > 0 && qty <= holding.quantity;
  const validPrice = Number.isFinite(price) && price > 0;

  // Live preview so the number is confirmed before saving, not after.
  const pnl = validQty && validPrice ? (price - holding.buy_price) * qty : null;
  const pnlPct =
    validPrice && holding.buy_price > 0
      ? (price / holding.buy_price - 1) * 100
      : null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validQty || !validPrice || submitting) return;
    setSubmitting(true);
    try {
      await onSubmit({
        quantity: qty,
        sell_price: price,
        sell_date: sellDate,
        notes,
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-xl border border-white/5 bg-white/[0.02] p-6"
    >
      <h3 className="mb-1 text-sm font-medium text-white/70">
        Sell <span className="font-mono text-white">{holding.symbol}</span>
      </h3>
      <p className="mb-4 text-xs text-white/40">
        Bought at {holding.buy_price.toFixed(2)} EGP · {holding.quantity} shares held
      </p>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs text-white/40">
            Quantity to sell *
          </label>
          <input
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            min={1}
            max={holding.quantity}
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-[16px] text-white outline-none focus:border-accent/50 md:text-sm"
            required
          />
          <p className="mt-1 text-[10px] text-white/30">
            of {holding.quantity} shares
          </p>
          {quantity !== "" && !validQty && (
            <p className="mt-1 text-[10px] text-loss/70">
              Enter between 1 and {holding.quantity} shares
            </p>
          )}
        </div>

        <div>
          <label className="mb-1 block text-xs text-white/40">
            Sell Price (EGP) *
          </label>
          <input
            type="number"
            value={sellPrice}
            onChange={(e) => setSellPrice(e.target.value)}
            min={0.01}
            step={0.01}
            placeholder="95.00"
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-[16px] text-white placeholder-white/30 outline-none focus:border-accent/50 md:text-sm"
            required
          />
        </div>

        <div>
          <label className="mb-1 block text-xs text-white/40">Sell Date</label>
          <input
            type="date"
            value={sellDate}
            max={new Date().toISOString().slice(0, 10)}
            onChange={(e) => setSellDate(e.target.value)}
            className="w-full min-w-0 appearance-none rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-left text-[16px] text-white outline-none focus:border-accent/50 md:text-sm"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs text-white/40">
            Notes (optional)
          </label>
          <input
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Why did you sell?"
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-[16px] text-white placeholder-white/30 outline-none focus:border-accent/50 md:text-sm"
          />
        </div>
      </div>

      {pnl !== null && (
        <div className="mt-4 rounded-lg border border-white/5 bg-white/[0.03] p-3">
          <p className="text-xs text-white/40">This sale realizes</p>
          <p
            className={`font-mono text-lg font-semibold ${
              pnl >= 0 ? "text-gain" : "text-loss"
            }`}
          >
            {pnl >= 0 ? "+" : ""}
            {pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })} EGP
            {pnlPct !== null && (
              <span className="ml-2 text-sm">
                ({pnlPct >= 0 ? "+" : ""}
                {pnlPct.toFixed(2)}%)
              </span>
            )}
          </p>
        </div>
      )}

      <div className="mt-4 flex gap-3">
        <button
          type="submit"
          disabled={!validQty || !validPrice || submitting}
          className="flex min-h-[44px] items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-charcoal-dark transition-opacity hover:opacity-90 disabled:opacity-30"
        >
          {submitting ? "Recording…" : "Record Sale"}
        </button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="min-h-[44px] rounded-lg border border-white/10 px-4 py-2 text-sm text-white/50 hover:text-white"
          >
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
```

- [ ] **Step 2: Add the `onSell` prop to `HoldingsTable`**

In the `HoldingsTableProps` interface (line 92-96) add:

```typescript
  onSell: (id: string) => void;
```

Add `onSell,` to the destructured params in the component signature (line 98-102).

- [ ] **Step 3: Add the mobile entry point**

In the expanded mobile card's action row (line 280-295), insert a Sell button between Edit and Delete:

```tsx
                    <button
                      onClick={() => h.id && onSell(h.id)}
                      className="min-h-[44px] flex-1 rounded-lg border border-gain/20 py-2 text-sm font-medium text-gain"
                    >
                      Sell
                    </button>
```

- [ ] **Step 4: Add the desktop entry point**

In the desktop actions cell (line 449-463), insert between the Edit and Del buttons:

```tsx
                          <button
                            onClick={() => h.id && onSell(h.id)}
                            className="text-xs text-gain/70 hover:text-gain"
                          >
                            Sell
                          </button>
```

- [ ] **Step 5: Typecheck**

Run: `npx tsc --noEmit`
Expected: one error — `portfolio/page.tsx` does not pass the required `onSell` prop. That is expected and Task 11 fixes it. Do not add a placeholder handler to silence it.

- [ ] **Step 6: Commit**

```bash
git add src/app/components/SellHoldingForm.tsx src/app/components/HoldingsTable.tsx
git commit -m "feat(fe): sell form and holdings-table entry points"
```

---

### Task 9: `RealizedGainsCard`

**Files:**
- Create: `egx-api-fe/src/app/components/RealizedGainsCard.tsx`

**Interfaces:**
- Consumes: `SalesSummary`, `LearnTooltip`.
- Produces: `<RealizedGainsCard summary riskFreeRatePct />`.

- [ ] **Step 1: Create the component**

```tsx
"use client";

import LearnTooltip from "./LearnTooltip";
import type { SalesSummary } from "../lib/types";

interface RealizedGainsCardProps {
  summary: SalesSummary;
  riskFreeRatePct: number;
}

function egp(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toLocaleString(undefined, {
    maximumFractionDigits: 0,
  })}`;
}

export default function RealizedGainsCard({
  summary,
  riskFreeRatePct,
}: RealizedGainsCardProps) {
  if (!summary.by_symbol.length) {
    return (
      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-8 text-center">
        <p className="text-sm text-white/40">No sales recorded yet</p>
        <p className="mt-1 text-xs text-white/30">
          Sell a holding and your realized winnings will be tracked here.
        </p>
      </div>
    );
  }

  const positive = summary.total_realized_pnl >= 0;

  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4 md:p-6">
      <h2 className="mb-4 text-sm font-medium text-white/70">
        <LearnTooltip
          term="Realized Winnings"
          explanation="Profit you have actually banked by selling, in EGP. Unlike the P&L on your open holdings, this cannot go back down — the trade is closed. It counts capital gains only; dividends are not included."
        >
          Realized Winnings
        </LearnTooltip>
      </h2>

      {/* Headline */}
      <div className="mb-5">
        <p
          className={`font-mono text-3xl font-bold md:text-4xl ${
            positive ? "text-gain" : "text-loss"
          }`}
        >
          {egp(summary.total_realized_pnl)} EGP
        </p>
        {summary.total_realized_pnl_pct !== null && (
          <p className="mt-1 text-sm text-white/40">
            {summary.total_realized_pnl_pct >= 0 ? "+" : ""}
            {summary.total_realized_pnl_pct.toFixed(2)}% on{" "}
            {summary.total_cost.toLocaleString(undefined, {
              maximumFractionDigits: 0,
            })}{" "}
            EGP invested
          </p>
        )}
      </div>

      {/* Support row */}
      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="rounded-lg bg-white/[0.03] p-3">
          <p className="text-[10px] uppercase tracking-wide text-white/30">Record</p>
          <p className="mt-1 font-mono text-sm text-white">
            <span className="text-gain">{summary.win_count}W</span>
            {" / "}
            <span className="text-loss">{summary.loss_count}L</span>
          </p>
        </div>
        <div className="rounded-lg bg-white/[0.03] p-3">
          <p className="text-[10px] uppercase tracking-wide text-white/30">Proceeds</p>
          <p className="mt-1 font-mono text-sm text-white">
            {summary.total_proceeds.toLocaleString(undefined, {
              maximumFractionDigits: 0,
            })}
          </p>
        </div>
        {summary.best_trade && (
          <div className="rounded-lg bg-white/[0.03] p-3">
            <p className="text-[10px] uppercase tracking-wide text-white/30">Best</p>
            <p className="mt-1 font-mono text-sm text-gain">
              {summary.best_trade.symbol} {egp(summary.best_trade.realized_pnl)}
            </p>
          </div>
        )}
        {summary.worst_trade &&
          summary.worst_trade.id !== summary.best_trade?.id && (
            <div className="rounded-lg bg-white/[0.03] p-3">
              <p className="text-[10px] uppercase tracking-wide text-white/30">Worst</p>
              <p className="mt-1 font-mono text-sm text-loss">
                {summary.worst_trade.symbol} {egp(summary.worst_trade.realized_pnl)}
              </p>
            </div>
          )}
      </div>

      {/* T-bill context. A fact, not an aggregate — annualized returns over
          trades of different lengths cannot honestly be averaged. */}
      {summary.annualizable_count > 0 && (
        <p className="mb-5 text-xs text-white/40">
          <LearnTooltip
            term="Versus risk-free cash"
            explanation={`Egypt's T-bill rate is about ${riskFreeRatePct.toFixed(
              0
            )}% — the highest risk-free rate of any major market. A gain that took years to earn can still be worth less than leaving the money in T-bills over the same period. Trades held under 30 days are excluded, because annualizing a few days of return produces nonsense.`}
          >
            {summary.beat_t_bill_count} of {summary.annualizable_count}
          </LearnTooltip>{" "}
          closed trades beat the {riskFreeRatePct.toFixed(0)}% T-bill over the
          period you held them.
        </p>
      )}

      {/* Per-stock breakdown */}
      <div>
        <p className="mb-2 text-xs font-medium text-white/50">By stock</p>
        <div className="space-y-2">
          {summary.by_symbol.map((s) => (
            <div
              key={s.symbol}
              className="flex items-center justify-between gap-3 rounded-lg bg-white/[0.03] px-3 py-2.5"
            >
              <div className="min-w-0">
                <p className="font-mono text-sm font-medium text-white">
                  {s.symbol}
                </p>
                <p className="truncate text-[10px] text-white/30">
                  {s.quantity} shares · {s.sales_count}{" "}
                  {s.sales_count === 1 ? "sale" : "sales"}
                </p>
              </div>
              <div className="shrink-0 text-right">
                <p
                  className={`font-mono text-sm font-semibold ${
                    s.realized_pnl >= 0 ? "text-gain" : "text-loss"
                  }`}
                >
                  {egp(s.realized_pnl)}
                </p>
                {s.realized_pnl_pct !== null && (
                  <p className="text-[10px] text-white/30">
                    {s.realized_pnl_pct >= 0 ? "+" : ""}
                    {s.realized_pnl_pct.toFixed(1)}%
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Check the LearnTooltip interface matches**

Open `src/app/components/LearnTooltip.tsx` and confirm the prop names are `term` and `explanation` and that it renders `children`. If they differ, adjust the two call sites above to match — do not change `LearnTooltip`.

- [ ] **Step 3: Typecheck**

Run: `npx tsc --noEmit`
Expected: only the known `onSell` error from Task 8.

- [ ] **Step 4: Commit**

```bash
git add src/app/components/RealizedGainsCard.tsx
git commit -m "feat(fe): realized gains card"
```

---

### Task 10: `ClosedPositionsTable`

**Files:**
- Create: `egx-api-fe/src/app/components/ClosedPositionsTable.tsx`

**Interfaces:**
- Consumes: `Sale`.
- Produces: `<ClosedPositionsTable sales riskFreeRatePct onDelete />` where `onDelete(id: string) => void`.

- [ ] **Step 1: Create the component**

```tsx
"use client";

import { useState } from "react";
import type { Sale } from "../lib/types";

interface ClosedPositionsTableProps {
  sales: Sale[];
  riskFreeRatePct: number;
  onDelete: (id: string) => void;
}

function pnlClass(value: number): string {
  return value >= 0 ? "text-gain" : "text-loss";
}

function signed(value: number, digits = 0): string {
  return `${value >= 0 ? "+" : ""}${value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
  })}`;
}

/** "Held 412 days · +3.9%/yr vs 25% T-bill", or null when too short to annualize. */
function tBillLine(sale: Sale, riskFreeRatePct: number): string | null {
  if (sale.annualized_return_pct === null) return null;
  const verdict = sale.beat_t_bill ? "beat" : "lost to";
  return `${signed(sale.annualized_return_pct, 1)}%/yr — ${verdict} the ${riskFreeRatePct.toFixed(
    0
  )}% T-bill`;
}

export default function ClosedPositionsTable({
  sales,
  riskFreeRatePct,
  onDelete,
}: ClosedPositionsTableProps) {
  const [confirmDelete, setConfirmDelete] = useState<{
    id: string;
    symbol: string;
  } | null>(null);

  if (!sales.length) return null;

  return (
    <>
      <details className="rounded-xl border border-white/5 bg-white/[0.02]">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-white/70 md:px-6">
          Closed Positions
          <span className="ml-2 text-xs text-white/30">({sales.length})</span>
        </summary>

        {/* Mobile cards */}
        <div className="space-y-3 px-4 pb-4 md:hidden">
          {sales.map((s) => {
            const line = tBillLine(s, riskFreeRatePct);
            return (
              <div key={s.id} className="rounded-lg bg-white/[0.03] p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="font-mono text-sm font-medium text-white">
                      {s.symbol}
                    </p>
                    <p className="mt-0.5 text-[10px] text-white/30">
                      {s.quantity} shares · {s.buy_price.toFixed(2)} →{" "}
                      {s.sell_price.toFixed(2)} EGP
                    </p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className={`font-mono text-sm font-semibold ${pnlClass(s.realized_pnl)}`}>
                      {signed(s.realized_pnl)}
                    </p>
                    {s.realized_pnl_pct !== null && (
                      <p className="text-[10px] text-white/30">
                        {signed(s.realized_pnl_pct, 1)}%
                      </p>
                    )}
                  </div>
                </div>
                <p className="mt-2 text-[10px] text-white/30">
                  Sold {s.sell_date} · held {s.days_held} days
                </p>
                {line && <p className="mt-0.5 text-[10px] text-white/40">{line}</p>}
                <button
                  onClick={() => setConfirmDelete({ id: s.id, symbol: s.symbol })}
                  className="mt-2 min-h-[44px] w-full rounded-lg border border-white/10 text-xs text-white/40"
                >
                  Undo this sale
                </button>
              </div>
            );
          })}
        </div>

        {/* Desktop table */}
        <div className="hidden overflow-x-auto px-6 pb-4 md:block">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5 text-left text-xs text-white/40">
                <th className="py-2 pr-4 font-medium">Symbol</th>
                <th className="py-2 pr-4 font-medium">Qty</th>
                <th className="py-2 pr-4 font-medium">Buy → Sell</th>
                <th className="py-2 pr-4 font-medium">Sold</th>
                <th className="py-2 pr-4 font-medium">Held</th>
                <th className="py-2 pr-4 font-medium">Realized</th>
                <th className="py-2 pr-4 font-medium">vs T-bill</th>
                <th className="py-2 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {sales.map((s) => {
                const line = tBillLine(s, riskFreeRatePct);
                return (
                  <tr key={s.id} className="border-b border-white/5">
                    <td className="py-3 pr-4 font-mono text-white">{s.symbol}</td>
                    <td className="py-3 pr-4 font-mono text-white/60">{s.quantity}</td>
                    <td className="py-3 pr-4 font-mono text-white/60">
                      {s.buy_price.toFixed(2)} → {s.sell_price.toFixed(2)}
                    </td>
                    <td className="py-3 pr-4 text-white/40">{s.sell_date}</td>
                    <td className="py-3 pr-4 text-white/40">{s.days_held}d</td>
                    <td className={`py-3 pr-4 font-mono font-semibold ${pnlClass(s.realized_pnl)}`}>
                      {signed(s.realized_pnl)}
                      {s.realized_pnl_pct !== null && (
                        <span className="ml-1 text-xs font-normal text-white/30">
                          {signed(s.realized_pnl_pct, 1)}%
                        </span>
                      )}
                    </td>
                    <td className="py-3 pr-4 text-xs text-white/40">{line ?? "—"}</td>
                    <td className="py-3 text-right">
                      <button
                        onClick={() => setConfirmDelete({ id: s.id, symbol: s.symbol })}
                        className="text-xs text-white/30 hover:text-white/60"
                      >
                        Undo
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>

      {confirmDelete && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center px-4">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setConfirmDelete(null)}
          />
          <div className="relative w-full max-w-sm rounded-2xl bg-charcoal p-6">
            <p className="text-base font-semibold text-white">Undo this sale?</p>
            <p className="mt-2 text-sm text-white/50">
              The{" "}
              <span className="font-mono text-white/80">{confirmDelete.symbol}</span>{" "}
              sale will be removed from your winnings and the shares returned to
              your portfolio. If you deleted the holding itself, the shares
              cannot be restored.
            </p>
            <div className="mt-6 flex gap-3">
              <button
                onClick={() => setConfirmDelete(null)}
                className="min-h-[44px] flex-1 rounded-xl border border-white/10 text-sm text-white/60 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onDelete(confirmDelete.id);
                  setConfirmDelete(null);
                }}
                className="min-h-[44px] flex-1 rounded-xl bg-accent text-sm font-semibold text-charcoal-dark"
              >
                Undo
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: only the known `onSell` error from Task 8.

- [ ] **Step 3: Commit**

```bash
git add src/app/components/ClosedPositionsTable.tsx
git commit -m "feat(fe): closed positions table"
```

---

### Task 11: Wire the portfolio page

**Files:**
- Modify: `egx-api-fe/src/app/portfolio/page.tsx`

**Interfaces:**
- Consumes: everything from Tasks 7-10.
- Produces: the finished feature.

- [ ] **Step 1: Add imports and state**

Add to the imports:

```tsx
import SellHoldingForm from "../components/SellHoldingForm";
import RealizedGainsCard from "../components/RealizedGainsCard";
import ClosedPositionsTable from "../components/ClosedPositionsTable";
import { fetchSales, recordSale, deleteSale } from "../lib/api";
import type { SalesResponse } from "../lib/types";
```

Add beside the existing state declarations:

```tsx
  const [sales, setSales] = useState<SalesResponse | null>(null);
  const [sellingId, setSellingId] = useState<string | null>(null);
```

- [ ] **Step 2: Load sales independently of the analysis**

Add after `loadPortfolio`:

```tsx
  // Sales are loaded on their own: they need no price fetch, so the Winnings
  // card paints immediately even when portfolio analysis is slow or fails.
  const loadSales = useCallback(async () => {
    try {
      setSales(await fetchSales());
    } catch {
      // A sales failure must not take down the portfolio page.
    }
  }, []);

  useEffect(() => {
    loadSales();
  }, [loadSales]);
```

- [ ] **Step 3: Add the sell handlers**

```tsx
  const handleSell = (id: string) => {
    setSellingId(id);
  };

  const handleSellSubmit = async (data: {
    quantity: number;
    sell_price: number;
    sell_date: string;
    notes: string;
  }) => {
    if (!sellingId) return;
    try {
      await recordSale({ holding_id: sellingId, ...data });
      setSellingId(null);
      await Promise.all([refreshAfterMutation(), loadSales()]);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleDeleteSale = async (id: string) => {
    try {
      await deleteSale(id);
      await Promise.all([refreshAfterMutation(), loadSales()]);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const sellingHolding = sellingId
    ? portfolio?.portfolio.find((h) => h.id === sellingId)
    : null;
```

- [ ] **Step 4: Extend the body-scroll lock to the sell modal**

Change the lock effect's guard (line 42) and dependency (line 55) so it also fires for the sell modal:

```tsx
    if (!showForm && !sellingId) return;
```

```tsx
  }, [showForm, sellingId]);
```

- [ ] **Step 5: Render the sell modal**

Insert after the existing add/edit form block:

```tsx
        {sellingId && sellingHolding && (
          <>
            <div className="fixed inset-0 z-[60] flex flex-col bg-charcoal-dark md:hidden">
              <div
                className="flex shrink-0 items-center justify-between border-b border-white/10 px-4 pb-3"
                style={{ paddingTop: "calc(env(safe-area-inset-top) + 12px)" }}
              >
                <h2 className="text-lg font-bold text-white">Record Sale</h2>
                <button
                  onClick={() => setSellingId(null)}
                  className="min-h-[44px] min-w-[44px] text-sm text-white/50"
                >
                  Cancel
                </button>
              </div>
              <div
                className="flex-1 overflow-y-auto p-4"
                style={{ WebkitOverflowScrolling: "touch" }}
              >
                <SellHoldingForm
                  holding={sellingHolding}
                  onSubmit={handleSellSubmit}
                  onCancel={() => setSellingId(null)}
                />
              </div>
            </div>

            <div className="mb-6 hidden md:block">
              <SellHoldingForm
                holding={sellingHolding}
                onSubmit={handleSellSubmit}
                onCancel={() => setSellingId(null)}
              />
            </div>
          </>
        )}
```

- [ ] **Step 6: Fix the empty state so winnings survive selling out**

Today the page has a three-way ternary: loading → "No holdings yet" → the full
page. A user who sells every position would hit the middle branch and lose
sight of their winnings entirely. Split that middle branch in two.

Find the existing condition at line 310:

```tsx
        ) : !portfolio?.portfolio.length && !showForm ? (
```

Change **only that line** to also require that there are no sales, so the
original first-run empty state now appears solely when the user has never held
or sold anything:

```tsx
        ) : !portfolio?.portfolio.length && !showForm && !sales?.sales.length ? (
```

Then, immediately after that branch's closing `</div>` and before the existing
`) : (`, insert a second branch for "sold out but has history":

```tsx
        ) : !portfolio?.portfolio.length && !showForm ? (
          <div className="space-y-6">
            {sales && (
              <RealizedGainsCard
                summary={sales.summary}
                riskFreeRatePct={sales.risk_free_rate_pct}
              />
            )}
            {sales && (
              <ClosedPositionsTable
                sales={sales.sales}
                riskFreeRatePct={sales.risk_free_rate_pct}
                onDelete={handleDeleteSale}
              />
            )}
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-8 text-center">
              <p className="text-sm text-white/40">You have no open positions.</p>
              <button
                onClick={() => setShowForm(true)}
                className="mt-3 rounded-lg bg-accent px-6 py-2 text-sm font-medium text-charcoal-dark"
              >
                Add a Stock
              </button>
            </div>
          </div>
```

The result is a four-way ternary: loading → never-traded empty state →
sold-out-with-history → the full page.

- [ ] **Step 7: Render the cards in the main branch**

Inside the final `<div className="space-y-6">`, pass `onSell` to `HoldingsTable`:

```tsx
              <HoldingsTable
                holdings={analysis.holdings}
                onEdit={handleEdit}
                onDelete={handleDelete}
                onSell={handleSell}
              />
```

And add the two new sections immediately after the `PortfolioSummary` block:

```tsx
            {sales && sales.sales.length > 0 && (
              <RealizedGainsCard
                summary={sales.summary}
                riskFreeRatePct={sales.risk_free_rate_pct}
              />
            )}
            {sales && sales.sales.length > 0 && (
              <ClosedPositionsTable
                sales={sales.sales}
                riskFreeRatePct={sales.risk_free_rate_pct}
                onDelete={handleDeleteSale}
              />
            )}
```

- [ ] **Step 8: Typecheck and build**

Run: `npx tsc --noEmit` — expected: clean, including the previously-known `onSell` error.
Run: `npm run build` — expected: succeeds.

- [ ] **Step 9: Manual verification on a mobile viewport**

With the backend running, in the browser at 375px wide:

1. Expand a holding → tap **Sell** → the modal fills the screen and quantity is pre-filled to the full position.
2. Enter a price → the live P&L preview updates before you submit.
3. Sell **part** of the position → the holding remains with the reduced quantity, and the Winnings card and Closed Positions both appear.
4. Sell the **rest** → the holding disappears from the table, and the Winnings card still shows both sales.
5. **Undo** the second sale from Closed Positions → the holding reappears with the correct quantity.
6. Try to sell more shares than held → the submit button stays disabled.

- [ ] **Step 10: Commit**

```bash
git add src/app/portfolio/page.tsx
git commit -m "feat(fe): wire sell tracking into the portfolio page"
```

---

### Task 12: Learn page concept and CLAUDE.md

**Files:**
- Modify: `egx-api-fe/src/app/learn/page.tsx`
- Modify: `egx-api-be/CLAUDE.md` and `egx-api-fe/CLAUDE.md` (identical edits)

- [ ] **Step 1: Add the Learn concept**

In the Risk Management section of `learn/page.tsx`, following the existing `Concept` pattern (see the `cash_underperformer` concept at line 416 for the shape):

```tsx
          <Concept
            id="realized_gains"
            title="Realized vs Unrealized — Banking a Win"
            definition="An unrealized gain is profit on paper: your stock is up but you still own it, so the number moves every day and can vanish. A realized gain is profit you have banked by selling. It cannot go back down."
            whyItMatters="Beginners often judge themselves on paper gains, which flatter in a rising market and punish in a falling one. Your realized record is the honest scoreboard: it is what actually happened. But size alone is misleading — a 10% gain earned in a month and a 10% gain earned over three years are completely different results."
            howToUse="Check the Winnings card after each sale. Look at the annualized return next to each closed trade, not just the EGP figure. With T-bills near 25%, a small gain held for years actually lost to risk-free cash — that is a lesson about position sizing and patience, not a reason to trade more often. Trades held under 30 days show no annualized figure, because annualizing a few days of return produces meaningless numbers."
          />
```

- [ ] **Step 2: Update both CLAUDE.md copies**

Make these edits identically in `egx-api-be/CLAUDE.md` and `egx-api-fe/CLAUDE.md`:

1. **Directory layout** — add under `routers/`: `sales.py  # GET/POST/DELETE /api/sales (realized gains)`. Add under `core/`: `holdings.py  # The one spelling of the open-holdings query`, `returns.py  # Position-level return maths, shared open/closed`, `sales.py  # Realized-gains maths (pure)`.

2. **API Endpoints** — add a section after `GET /api/portfolio`:

```markdown
### GET /api/sales, POST, DELETE

Realized gains ledger. `POST` records a full or partial sell: it inserts a
`portfolio_sales` row snapshotting the cost basis and decrements
`portfolio.quantity`, **both inside one `db.transaction()`** — `db.commit()` is
a no-op and each `execute()` takes its own autocommit connection, so without
the transaction a failure between the two statements would invent or lose
shares. The `quantity >= %s` guard lives in the UPDATE's WHERE clause, so two
rapid submits cannot both succeed.

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
```

3. **Database schema** — add:

```sql
portfolio_sales (id, user_id, holding_id, symbol, name, sector, quantity,
                 buy_price, buy_date, sell_price, sell_date, notes, created_at)
                 -- Cost basis is SNAPSHOTTED, not joined: a sale is a
                 -- historical fact and must not change when the holding it
                 -- came from is later edited or deleted.
                 -- portfolio.quantity = 0 means fully sold; the row is kept
                 -- as the undo anchor and filtered out of every read by
                 -- core/holdings.fetch_open_holdings.
```

4. **Portfolio page section** — add: "**Sell** action per holding (full or partial) → `RealizedGainsCard` + collapsed `ClosedPositionsTable`."

5. **Frontend components** — add `SellHoldingForm`, `RealizedGainsCard`, `ClosedPositionsTable` to the Portfolio views list, and add `onSell` to the `HoldingsTable` description.

6. **Correct the stale `cash_available` claims — the setting does not exist in the code.** Remove "Cash balance management via `updateCash()` → `PUT /api/settings`" from the Portfolio page section; remove `cash_available=50000` from the `GET/PUT /api/settings` description; remove `cash_available` from the `settings` schema comment.

7. **Correct the "No auth" line** under *Things to Know*: `portfolio.py` and every other router scope their queries by `user.id` from a JWT Bearer token, and `main.py` seeds users via `seed_users_from_env`.

- [ ] **Step 3: Verify the docs match reality**

Run from the repo root:

```bash
diff egx-api-be/CLAUDE.md egx-api-fe/CLAUDE.md
```

Expected: no output — the copies are identical.

- [ ] **Step 4: Build and run the full suite one last time**

Run from `egx-api-be`: `python -m pytest tests/ -v` — expected: all pass.
Run from `egx-api-fe`: `npm run build` — expected: succeeds.

- [ ] **Step 5: Commit**

```bash
# in egx-api-fe
git add src/app/learn/page.tsx CLAUDE.md
git commit -m "docs: realized gains concept and CLAUDE.md update"

# in egx-api-be
git add CLAUDE.md
git commit -m "docs: sales ledger and corrections to stale claims"
```
