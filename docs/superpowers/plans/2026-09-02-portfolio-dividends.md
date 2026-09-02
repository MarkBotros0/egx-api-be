# Portfolio Dividends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user record cash distributions ("profit share") received on EGX holdings, and fold that money into the Winnings card, the per-stock breakdown and each open holding.

**Architecture:** A new symbol-anchored `portfolio_dividends` table with its own pure-maths core module and thin write routes. The existing `GET /api/sales` is extended to serve the combined realized ledger, so the combined headline is computed in tested Python rather than in TSX. `summarize_sales` is replaced by `summarize_realized`, which owns both halves — there is never a second summariser.

**Tech Stack:** Python 3 / FastAPI / psycopg3 / Neon Postgres · Next.js 14 App Router / React 18 / TypeScript / Tailwind

**Spec:** `egx-api-be/docs/superpowers/specs/2026-09-02-portfolio-dividends-design.md`

**Repos:** backend `D:\Projects\egx-api\egx-api-be`, frontend `D:\Projects\egx-api\egx-api-fe`. **Separate git repositories**, both on branch `feat/dividends`. Commit in the repo you edited.

## Global Constraints

- **Test command is `./.venv/Scripts/python.exe -m pytest`** from `egx-api-be`. A bare `python` on this machine has no pytest. Frontend checks: `npx tsc --noEmit` and `npm run build` from `egx-api-fe`.
- **psycopg3 placeholders are `%s`**, never `?`.
- **A literal `%` in SQL MUST be written `%%`.** `tests/test_users_and_roles.py::test_no_sql_has_an_unescaped_percent` walks the AST of every `execute()` call and fails on a bare `%`.
- **Every route is scoped by `user.id`** from the JWT (`CurrentUser = Depends(get_current_user)`).
- **`amount` is the total EGP actually received, already net of withholding tax.** Never presented as gross. The app computes no tax.
- **Framing rule, binding on every user-facing string in this feature:** with T-bills near 25%, no EGX dividend yield is competitive as income. A dividend is presented as **evidence the company generates real cash**, never as income, and is never compared to the T-bill rate.
- **Mobile-first:** `md:` (768px) breakpoint · `min-h-[44px]` touch targets · `space-y-3 md:hidden` + `hidden md:block` for card/table switching · full-screen modal on mobile, inline on desktop · `text-[16px] md:text-sm` on every input (prevents iOS zoom).
- **Colours:** Tailwind `gain` (#00ff88), `loss` (#ff3355), `accent` (#4488ff), `charcoal` (#12121a), `charcoal-dark` (#0a0a0f). All money EGP.
- **Never modify `portfolio_sales` semantics.** A dividend reduces no position and closes no cost basis.
- **CLAUDE.md is mirrored** in both repos and the two copies must stay byte-identical.

---

## File Structure

**Backend (`egx-api-be`)**

| File | Responsibility |
|---|---|
| `app/core/dividends.py` | **new** — pure validation/enrichment/rollup + the one spelling of the dividend queries |
| `app/core/sales.py` | `summarize_sales` removed; the rest unchanged |
| `app/core/db.py` | `portfolio_dividends` DDL + index in `init_db` |
| `app/routers/dividends.py` | **new** — `POST` / `DELETE /api/dividends` |
| `app/routers/sales.py` | `GET` extended: returns `dividends`, summary from `summarize_realized` |
| `app/routers/portfolio_analysis.py` | `dividends_collected` + `dividends_symbol_shared` per holding |
| `app/routers/users.py` | `DELETE` also clears `portfolio_dividends` |
| `app/main.py` | register the dividends router |
| `tests/test_dividends.py` | **new** — all pure-function tests + grep guards |
| `tests/test_sell_tracking.py` | `summarize_sales` → `summarize_realized(…, [])`, assertions byte-identical |

**Frontend (`egx-api-fe`)**

| File | Responsibility |
|---|---|
| `src/app/lib/types.ts` | `Dividend`; fields added to `SalesSummary`, `SymbolRealized`, `SalesResponse`, `HoldingAnalysis` |
| `src/app/lib/api.ts` | `recordDividend`, `deleteDividend` |
| `src/app/components/AddDividendForm.tsx` | **new** |
| `src/app/components/DividendsTable.tsx` | **new** |
| `src/app/components/RealizedGainsCard.tsx` | combined headline + split line + per-symbol dividends |
| `src/app/components/HoldingsTable.tsx` | dividend pill + expanded-detail line + `onAddDividend` |
| `src/app/portfolio/page.tsx` | wiring, modal, widened render gates |
| `src/app/learn/page.tsx` | `dividends` Concept anchor |

---

## Task 1: Dividend validation and enrichment

**Files:**
- Create: `egx-api-be/app/core/dividends.py`
- Test: `egx-api-be/tests/test_dividends.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DividendValidationError`, `validate_dividend(*, symbol, amount, pay_date, shares, today) -> dict`, `enrich_dividend(row: dict) -> dict`, `is_duplicate(existing: list, candidate: dict) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `egx-api-be/tests/test_dividends.py`:

```python
"""
Dividend ledger — validation, enrichment and the realized-winnings rollup.

Pure-function tests only. tests/ has no Postgres fixture, which is exactly why
core/dividends.py keeps its maths independent of its queries.
"""

import ast
from datetime import date
from pathlib import Path

import pytest

from app.core.dividends import (
    DividendValidationError,
    enrich_dividend,
    is_duplicate,
    validate_dividend,
)

TODAY = date(2026, 9, 2)


def _valid(**overrides):
    base = {
        "symbol": "COMI",
        "amount": 1200.0,
        "pay_date": "2026-08-15",
        "shares": 500,
        "today": TODAY,
    }
    base.update(overrides)
    return base


# --- validation ---------------------------------------------------------

def test_a_valid_dividend_is_normalized():
    out = validate_dividend(**_valid())
    assert out == {
        "symbol": "COMI",
        "amount": 1200.0,
        "pay_date": "2026-08-15",
        "shares": 500,
    }


def test_symbol_is_upper_cased():
    assert validate_dividend(**_valid(symbol="comi"))["symbol"] == "COMI"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_missing_symbol_is_rejected(blank):
    with pytest.raises(DividendValidationError, match="Pick a stock"):
        validate_dividend(**_valid(symbol=blank))


@pytest.mark.parametrize("bad", [0, -1, -0.01])
def test_a_non_positive_amount_is_rejected(bad):
    with pytest.raises(DividendValidationError, match="greater than 0"):
        validate_dividend(**_valid(amount=bad))


@pytest.mark.parametrize("bad", ["abc", None, ""])
def test_a_non_numeric_amount_is_rejected(bad):
    with pytest.raises(DividendValidationError, match="must be a number"):
        validate_dividend(**_valid(amount=bad))


def test_pay_date_defaults_to_today_when_blank():
    assert validate_dividend(**_valid(pay_date=""))["pay_date"] == "2026-09-02"


def test_an_unparseable_pay_date_is_rejected():
    with pytest.raises(DividendValidationError, match="must be a date"):
        validate_dividend(**_valid(pay_date="last tuesday"))


def test_a_future_pay_date_is_rejected():
    with pytest.raises(DividendValidationError, match="cannot be in the future"):
        validate_dividend(**_valid(pay_date="2026-09-03"))


def test_todays_pay_date_is_accepted():
    assert validate_dividend(**_valid(pay_date="2026-09-02"))["pay_date"] == "2026-09-02"


# A dividend is symbol-anchored, so there is no single holding whose buy date
# could bound it — and the user may record one against a position already sold.
def test_a_very_old_pay_date_is_accepted():
    assert validate_dividend(**_valid(pay_date="2019-04-01"))["pay_date"] == "2019-04-01"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_shares_is_optional_and_stored_as_none(blank):
    assert validate_dividend(**_valid(shares=blank))["shares"] is None


@pytest.mark.parametrize("bad", [0, -5, 1.5, "many"])
def test_a_bad_share_count_is_rejected(bad):
    with pytest.raises(DividendValidationError, match="whole number"):
        validate_dividend(**_valid(shares=bad))


def test_a_numeric_string_amount_is_accepted():
    assert validate_dividend(**_valid(amount="1200"))["amount"] == 1200.0


# --- enrichment ---------------------------------------------------------

def test_amount_per_share_is_computed_when_shares_are_known():
    out = enrich_dividend({"amount": 1200.0, "shares": 500})
    assert out["amount_per_share"] == 2.4


@pytest.mark.parametrize("shares", [None, 0])
def test_amount_per_share_is_null_without_a_share_count(shares):
    assert enrich_dividend({"amount": 1200.0, "shares": shares})["amount_per_share"] is None


def test_enrichment_preserves_every_original_field():
    row = {"id": "d1", "symbol": "COMI", "amount": 1200.0, "shares": 500, "notes": "Q2"}
    out = enrich_dividend(row)
    for key, value in row.items():
        assert out[key] == value


# --- duplicate guard ----------------------------------------------------
# The primary surface is a phone. A double-tapped submit is the likeliest way
# this ledger goes silently wrong, and unlike a duplicate sale it corrupts no
# share count, so it leaves no other trace.

EXISTING = [
    {"symbol": "COMI", "pay_date": "2026-08-15", "amount": 1200.0},
    {"symbol": "HRHO", "pay_date": "2026-07-01", "amount": 300.0},
]


def test_an_exact_repeat_is_a_duplicate():
    assert is_duplicate(EXISTING, {"symbol": "COMI", "pay_date": "2026-08-15", "amount": 1200.0})


@pytest.mark.parametrize("candidate", [
    {"symbol": "COMI", "pay_date": "2026-08-15", "amount": 1200.5},
    {"symbol": "COMI", "pay_date": "2026-08-16", "amount": 1200.0},
    {"symbol": "SWDY", "pay_date": "2026-08-15", "amount": 1200.0},
])
def test_a_differing_field_is_not_a_duplicate(candidate):
    assert not is_duplicate(EXISTING, candidate)


def test_nothing_is_a_duplicate_of_an_empty_ledger():
    assert not is_duplicate([], {"symbol": "COMI", "pay_date": "2026-08-15", "amount": 1200.0})
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest tests/test_dividends.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'app.core.dividends'`.

- [ ] **Step 3: Write the implementation**

Create `egx-api-be/app/core/dividends.py`:

```python
"""
Dividend ledger — the money a company paid you for holding it.

A dividend is NOT a sale. It reduces no position, closes no cost basis, and has
no undo semantics, which is why it lives in its own table rather than in
portfolio_sales.

This module holds the pure maths AND the one spelling of the dividend queries,
the same shape as core/holdings.py. The queries take `db` as a parameter, so
they stay fakeable and the maths stays testable against a tests/ directory with
no Postgres fixture. Three routers read dividends; one spelling here is what
stops three spellings growing out there.

Framing rule (from CLAUDE.md, binding on every string this feature renders):
with T-bills near 25%, no EGX dividend yield is competitive as income. A
dividend is evidence the company generates real cash — never income, and never
compared to the T-bill.
"""

from datetime import date, datetime
from typing import Optional


class DividendValidationError(ValueError):
    """A dividend that is not internally consistent. Maps to HTTP 400."""


def _parse_date(value: str) -> Optional[date]:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def validate_dividend(*, symbol, amount, pay_date, shares, today: date) -> dict:
    """
    Check a dividend request. Returns normalized
    {symbol, amount, pay_date, shares}.

    Raises DividendValidationError with a message written for the user, not the
    log. Keyword-only so a new field can never shift an existing argument.
    """
    if symbol is None or not str(symbol).strip():
        raise DividendValidationError("Pick a stock.")
    symbol = str(symbol).strip().upper()

    if isinstance(amount, bool) or amount is None or amount == "":
        raise DividendValidationError("Amount must be a number.")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise DividendValidationError("Amount must be a number.")
    if amount != amount or amount in (float("inf"), float("-inf")):
        raise DividendValidationError("Amount must be a number.")
    if amount <= 0:
        raise DividendValidationError("Amount must be greater than 0.")

    if pay_date is None or not str(pay_date).strip():
        parsed = today
    else:
        parsed = _parse_date(pay_date)
        if parsed is None:
            raise DividendValidationError("Pay date must be a date like 2026-09-01.")
    if parsed > today:
        raise DividendValidationError("Pay date cannot be in the future.")

    # No buy-date lower bound: a dividend is anchored to the SYMBOL, so there is
    # no single holding whose purchase could bound it, and it is legitimate to
    # record one against a position already sold.

    if shares is None or (isinstance(shares, str) and not shares.strip()):
        shares = None
    else:
        if isinstance(shares, bool):
            raise DividendValidationError("Shares must be a whole number of shares.")
        try:
            as_float = float(shares)
        except (TypeError, ValueError):
            raise DividendValidationError("Shares must be a whole number of shares.")
        if as_float != int(as_float) or int(as_float) <= 0:
            raise DividendValidationError("Shares must be a whole number of shares.")
        shares = int(as_float)

    return {
        "symbol": symbol,
        "amount": amount,
        "pay_date": parsed.isoformat(),
        "shares": shares,
    }


def enrich_dividend(row: dict) -> dict:
    """
    Add amount_per_share to one dividend.

    Computed server-side so the card and any other consumer cannot disagree.
    None when the share count is unknown or zero — `amount` is authoritative and
    is never derived from it.
    """
    shares = row.get("shares")
    amount = float(row.get("amount") or 0)
    per_share = round(amount / shares, 4) if shares else None
    return {**row, "amount_per_share": per_share}


def is_duplicate(existing: list, candidate: dict) -> bool:
    """
    True when this exact symbol + pay_date + amount is already on record.

    The primary surface is a phone, where a double-tapped submit is the most
    likely way this ledger goes wrong. A duplicate SALE at least leaves a wrong
    share count; a duplicate dividend leaves no trace at all, so it is caught
    here instead.
    """
    for row in existing:
        if (
            str(row.get("symbol") or "").upper() == str(candidate.get("symbol") or "").upper()
            and str(row.get("pay_date") or "")[:10] == str(candidate.get("pay_date") or "")[:10]
            and abs(float(row.get("amount") or 0) - float(candidate.get("amount") or 0)) < 1e-9
        ):
            return True
    return False
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest tests/test_dividends.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/egx-api/egx-api-be && git add app/core/dividends.py tests/test_dividends.py && git commit -m "feat: dividend validation, enrichment and duplicate guard"
```

---

## Task 2: `summarize_realized` replaces `summarize_sales`

**Files:**
- Modify: `egx-api-be/app/core/dividends.py` (append `summarize_realized`)
- Modify: `egx-api-be/app/core/sales.py` (delete `summarize_sales`)
- Modify: `egx-api-be/app/routers/sales.py` (import swap only — the GET body changes in Task 5)
- Modify: `egx-api-be/tests/test_sell_tracking.py` (rename call sites)
- Test: `egx-api-be/tests/test_dividends.py`

**Interfaces:**
- Consumes: `sales.compute_sale_metrics` output shape (`cost`, `proceeds`, `realized_pnl`, `realized_pnl_pct`, `beat_t_bill`, `symbol`, `name`, `sector`, `quantity`, `id`).
- Produces: `summarize_realized(priced_sales: list, dividends: list) -> dict`.

**Why a replacement and not an addition:** two functions producing overlapping Winnings figures is the divergence class documented in *One Score Per Stock* — the dashboard once showed 66 "Buy" where the detail page showed 45 "Hold" for exactly this reason. One function owns both halves.

- [ ] **Step 1: Write the failing tests**

Append to `egx-api-be/tests/test_dividends.py`:

```python
from app.core.dividends import summarize_realized

SALES = [
    # +2,000 on 10,000 cost
    {"id": "s1", "symbol": "COMI", "name": "CIB", "sector": "Banks", "quantity": 100,
     "cost": 10000.0, "proceeds": 12000.0, "realized_pnl": 2000.0,
     "realized_pnl_pct": 20.0, "beat_t_bill": True},
    # -500 on 5,000 cost
    {"id": "s2", "symbol": "SWDY", "name": "Elsewedy", "sector": "Industrial", "quantity": 50,
     "cost": 5000.0, "proceeds": 4500.0, "realized_pnl": -500.0,
     "realized_pnl_pct": -10.0, "beat_t_bill": False},
]

DIVIDENDS = [
    {"id": "d1", "symbol": "COMI", "name": "CIB", "sector": "Banks", "amount": 1200.0,
     "pay_date": "2026-08-15"},
    # HRHO was never sold — it must still appear in the breakdown.
    {"id": "d2", "symbol": "HRHO", "name": "EFG", "sector": "Financials", "amount": 300.0,
     "pay_date": "2026-07-01"},
]


def test_an_empty_ledger_is_zeroed_not_null():
    s = summarize_realized([], [])
    assert s["total_realized_pnl"] == 0.0
    assert s["total_dividends"] == 0.0
    assert s["total_winnings"] == 0.0
    assert s["dividend_count"] == 0
    assert s["by_symbol"] == []
    assert s["best_trade"] is None


def test_total_winnings_is_gains_plus_dividends():
    s = summarize_realized(SALES, DIVIDENDS)
    assert s["total_realized_pnl"] == 1500.0
    assert s["total_dividends"] == 1500.0
    assert s["total_winnings"] == 3000.0
    assert s["dividend_count"] == 2


# Dividends have no matching cost in this ledger — the shares producing them may
# still be held — so adding them to a numerator whose denominator is CLOSED-trade
# cost would make the percentage describe nothing.
def test_the_headline_percentage_ignores_dividends():
    without = summarize_realized(SALES, [])
    with_divs = summarize_realized(SALES, DIVIDENDS)
    assert without["total_realized_pnl_pct"] == with_divs["total_realized_pnl_pct"]
    assert with_divs["total_realized_pnl_pct"] == 10.0  # 1500 / 15000


# A dividend maps onto no single trade, so folding it into a per-trade verdict
# would make the line unverifiable.
def test_the_t_bill_counts_ignore_dividends():
    without = summarize_realized(SALES, [])
    with_divs = summarize_realized(SALES, DIVIDENDS)
    assert without["beat_t_bill_count"] == with_divs["beat_t_bill_count"] == 1
    assert without["annualizable_count"] == with_divs["annualizable_count"] == 2


def test_best_and_worst_stay_sales_only():
    s = summarize_realized(SALES, DIVIDENDS)
    assert s["best_trade"]["id"] == "s1"
    assert s["worst_trade"]["id"] == "s2"


def test_by_symbol_includes_a_symbol_that_was_never_sold():
    s = summarize_realized(SALES, DIVIDENDS)
    hrho = next(r for r in s["by_symbol"] if r["symbol"] == "HRHO")
    assert hrho["sales_count"] == 0
    assert hrho["cost"] == 0
    assert hrho["realized_pnl"] == 0
    assert hrho["realized_pnl_pct"] is None
    assert hrho["dividends"] == 300.0
    assert hrho["total_winnings"] == 300.0


def test_by_symbol_merges_dividends_into_a_sold_symbol():
    s = summarize_realized(SALES, DIVIDENDS)
    comi = next(r for r in s["by_symbol"] if r["symbol"] == "COMI")
    assert comi["realized_pnl"] == 2000.0
    assert comi["dividends"] == 1200.0
    assert comi["total_winnings"] == 3200.0


def test_by_symbol_is_sorted_by_total_winnings_descending():
    s = summarize_realized(SALES, DIVIDENDS)
    order = [r["symbol"] for r in s["by_symbol"]]
    assert order == ["COMI", "HRHO", "SWDY"]  # 3200, 300, -500


def test_a_dividend_only_ledger_still_produces_a_breakdown():
    s = summarize_realized([], DIVIDENDS)
    assert s["total_winnings"] == 1500.0
    assert len(s["by_symbol"]) == 2
    assert s["total_realized_pnl_pct"] is None


def test_summarize_sales_is_gone_so_there_is_only_one_summariser():
    import app.core.sales as sales_module
    assert not hasattr(sales_module, "summarize_sales"), (
        "Two summarisers producing overlapping Winnings figures is the "
        "divergence class documented in One Score Per Stock."
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest tests/test_dividends.py -q
```

Expected: `ImportError: cannot import name 'summarize_realized'`.

- [ ] **Step 3: Write the implementation**

Append to `egx-api-be/app/core/dividends.py`:

```python
def summarize_realized(priced_sales: list, dividends: list) -> dict:
    """
    Roll closed trades AND dividends up into the Winnings card's numbers.

    Takes sales already through sales.compute_sale_metrics, which is where the
    per-trade T-bill comparison happened, so this needs no rate of its own.

    Three figures deliberately stay capital-gains-only:

      total_realized_pnl_pct  — cost-weighted over CLOSED trades. Dividends have
                                no matching cost here (the shares may still be
                                held), so adding them would make the percentage
                                describe nothing.
      beat_t_bill_count /     — facts about individual closed trades. A dividend
      annualizable_count        maps onto no single trade.
      best_trade / worst_trade — labelled "trade" and returning a whole Sale the
                                card reads fields off.

    by_symbol is a UNION: a stock you still hold and collect on has no sale, and
    would otherwise be missing from the breakdown entirely.
    """
    total_dividends = sum(float(d.get("amount") or 0) for d in dividends)

    total_cost = sum(s["cost"] for s in priced_sales)
    total_proceeds = sum(s["proceeds"] for s in priced_sales)
    total_pnl = total_proceeds - total_cost

    annualizable = [s for s in priced_sales if s["beat_t_bill"] is not None]

    by_symbol: dict = {}

    def _bucket(symbol, name, sector):
        return by_symbol.setdefault(
            symbol,
            {
                "symbol": symbol, "name": name or symbol, "sector": sector or "",
                "sales_count": 0, "quantity": 0, "cost": 0.0, "proceeds": 0.0,
                "dividends": 0.0,
            },
        )

    for s in priced_sales:
        agg = _bucket(s["symbol"], s.get("name"), s.get("sector"))
        agg["sales_count"] += 1
        agg["quantity"] += int(s["quantity"])
        agg["cost"] += s["cost"]
        agg["proceeds"] += s["proceeds"]

    for d in dividends:
        agg = _bucket(d["symbol"], d.get("name"), d.get("sector"))
        agg["dividends"] += float(d.get("amount") or 0)

    rollup = []
    for agg in by_symbol.values():
        pnl = agg["proceeds"] - agg["cost"]
        rollup.append({
            **agg,
            "cost": round(agg["cost"], 2),
            "proceeds": round(agg["proceeds"], 2),
            "dividends": round(agg["dividends"], 2),
            "realized_pnl": round(pnl, 2),
            "realized_pnl_pct": round(pnl / agg["cost"] * 100, 2) if agg["cost"] > 0 else None,
            "total_winnings": round(pnl + agg["dividends"], 2),
        })
    rollup.sort(key=lambda r: r["total_winnings"], reverse=True)

    return {
        "total_realized_pnl": round(total_pnl, 2),
        "total_realized_pnl_pct": round(total_pnl / total_cost * 100, 2) if total_cost > 0 else None,
        "total_proceeds": round(total_proceeds, 2),
        "total_cost": round(total_cost, 2),
        "total_dividends": round(total_dividends, 2),
        "dividend_count": len(dividends),
        "total_winnings": round(total_pnl + total_dividends, 2),
        "win_count": sum(1 for s in priced_sales if s["realized_pnl"] > 0),
        "loss_count": sum(1 for s in priced_sales if s["realized_pnl"] < 0),
        "beat_t_bill_count": sum(1 for s in annualizable if s["beat_t_bill"]),
        "annualizable_count": len(annualizable),
        "best_trade": max(priced_sales, key=lambda s: s["realized_pnl"]) if priced_sales else None,
        "worst_trade": min(priced_sales, key=lambda s: s["realized_pnl"]) if priced_sales else None,
        "by_symbol": rollup,
    }
```

- [ ] **Step 4: Delete `summarize_sales` from `app/core/sales.py`**

Remove the entire `def summarize_sales(priced_sales: list) -> dict:` function and its docstring. Leave `SaleValidationError`, `_parse_date`, `validate_sale` and `compute_sale_metrics` untouched.

- [ ] **Step 5: Update the import in `app/routers/sales.py`**

Change:

```python
from app.core.sales import (
    SaleValidationError,
    compute_sale_metrics,
    summarize_sales,
    validate_sale,
)
```

to:

```python
from app.core.dividends import summarize_realized
from app.core.sales import (
    SaleValidationError,
    compute_sale_metrics,
    validate_sale,
)
```

and in `get_sales`, change `"summary": summarize_sales(priced),` to
`"summary": summarize_realized(priced, []),`. **The `[]` is temporary** — Task 5
replaces it with the real dividend list. This step only keeps the suite green.

- [ ] **Step 6: Rename the call sites in `tests/test_sell_tracking.py`**

In the import block, replace `summarize_sales,` with nothing and add
`from app.core.dividends import summarize_realized` beside the existing imports.

Then replace every `summarize_sales(priced)` with `summarize_realized(priced, [])`
and `summarize_sales([])` with `summarize_realized([], [])`.

**Every assertion in those tests stays byte-identical.** They are the regression
gate proving the capital-gains numbers did not move. If one needs changing, stop
and report it — that is a real behaviour change, not a rename.

- [ ] **Step 7: Run the whole suite**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest -q
```

Expected: all pass, including every pre-existing test.

- [ ] **Step 8: Commit**

```bash
cd /d/Projects/egx-api/egx-api-be && git add -A && git commit -m "feat: summarize_realized owns gains and dividends as one summary"
```

---

## Task 3: Table and the one spelling of the dividend queries

**Files:**
- Modify: `egx-api-be/app/core/db.py` (in `init_db`, after the `portfolio_sales` block ending at the `idx_portfolio_sales_user` index)
- Modify: `egx-api-be/app/core/dividends.py` (append the query section)

**Interfaces:**
- Consumes: `enrich_dividend` from Task 1.
- Produces: `DIVIDEND_COLUMNS`, `row_to_dividend(row) -> dict`, `fetch_dividends(db, user_id) -> list[dict]`, `fetch_dividend_totals(db, user_id) -> dict`.

- [ ] **Step 1: Add the DDL to `init_db`**

In `egx-api-be/app/core/db.py`, immediately after the
`CREATE INDEX IF NOT EXISTS idx_portfolio_sales_user` statement, insert:

```python
    # Dividends — cash the company paid you for holding it.
    #
    # Symbol-anchored, deliberately with NO holding_id. A sale carries one
    # because DELETE /api/sales must restore shares to a specific position; a
    # dividend restores nothing, so the column would buy no behaviour — and
    # would cost correctness, since deleting a holding would then orphan or
    # destroy the record of money genuinely received.
    #
    # `amount` is the total EGP that ACTUALLY LANDED, already net of Egypt's
    # 5-10% dividend withholding tax. The app computes no tax and must never
    # present this as a gross figure. `shares` is optional and exists only so
    # the UI can show an approximate per-share number; amount is never derived
    # from it.
    db.execute("""
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
        )
    """)
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_portfolio_dividends_user "
        "ON portfolio_dividends(user_id, symbol)"
    )
```

- [ ] **Step 2: Append the query section to `app/core/dividends.py`**

```python
# --- the one spelling of the dividend queries ---------------------------
#
# These live beside the maths, the same shape as core/holdings.py, because
# THREE routers read dividends — dividends.py, sales.py and
# portfolio_analysis.py. One spelling here is what stops three growing there.

DIVIDEND_COLUMNS = (
    "id, symbol, name, sector, amount, pay_date, shares, notes, created_at"
)


def row_to_dividend(row) -> dict:
    """Map a DIVIDEND_COLUMNS row tuple to the dict the API returns."""
    return {
        "id": row[0],
        "symbol": row[1],
        "name": row[2],
        "sector": row[3],
        "amount": row[4],
        "pay_date": row[5],
        "shares": row[6],
        "notes": row[7],
        "created_at": row[8],
    }


def fetch_dividends(db, user_id: str) -> list:
    """This user's dividends, enriched, newest pay_date first."""
    if not user_id:
        return []
    rows = db.execute(
        f"SELECT {DIVIDEND_COLUMNS} FROM portfolio_dividends "
        "WHERE user_id = %s ORDER BY pay_date DESC, created_at DESC",
        (user_id,),
    ).fetchall()
    return [enrich_dividend(row_to_dividend(r)) for r in rows]


def fetch_dividend_totals(db, user_id: str) -> dict:
    """
    {symbol: total EGP collected}. Empty dict when there is no user.

    Used by portfolio_analysis to put a figure against each open holding. One
    indexed aggregate with no price fetch, so it costs nothing against the 30 s
    Vercel budget.
    """
    if not user_id:
        return {}
    rows = db.execute(
        "SELECT symbol, SUM(amount) FROM portfolio_dividends "
        "WHERE user_id = %s GROUP BY symbol",
        (user_id,),
    ).fetchall()
    return {r[0]: round(float(r[1] or 0), 2) for r in rows}
```

- [ ] **Step 3: Run the suite to confirm nothing regressed**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest -q
```

Expected: all pass. In particular `test_no_sql_has_an_unescaped_percent` must
still pass — none of the new SQL contains a literal `%`.

- [ ] **Step 4: Commit**

```bash
cd /d/Projects/egx-api/egx-api-be && git add -A && git commit -m "feat: portfolio_dividends table and its single query spelling"
```

---

## Task 4: `POST` / `DELETE /api/dividends`

**Files:**
- Create: `egx-api-be/app/routers/dividends.py`
- Modify: `egx-api-be/app/main.py`

**Interfaces:**
- Consumes: `validate_dividend`, `enrich_dividend`, `is_duplicate`, `fetch_dividends`, `DividendValidationError` from `app.core.dividends`.
- Produces: `router` (registered in `main.py`).

- [ ] **Step 1: Write the router**

Create `egx-api-be/app/routers/dividends.py`:

```python
"""
Dividends — record cash a company paid you for holding it.

POST   /api/dividends          — record one
DELETE /api/dividends?id=xxx   — remove one

Reads are NOT here: dividends come back on GET /api/sales alongside closed
trades, so the Winnings card gets one fetch and the combined headline is
computed in tested Python rather than in the browser.

Unlike a sale, recording a dividend is a SINGLE statement — it decrements no
share count — so there is nothing for db.transaction() to keep atomic. Deleting
one restores nothing for the same reason.

Every route is scoped by the caller's user_id from the JWT.
"""

import uuid
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import CurrentUser, get_current_user
from app.core.db import get_db
from app.core.dividends import (
    DividendValidationError,
    enrich_dividend,
    fetch_dividends,
    is_duplicate,
    validate_dividend,
)

router = APIRouter()


@router.post("/api/dividends", status_code=201)
def record_dividend(body: dict, user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()

        clean = validate_dividend(
            symbol=body.get("symbol"),
            amount=body.get("amount"),
            pay_date=body.get("pay_date"),
            shares=body.get("shares"),
            today=date.today(),
        )

        # A double-tapped submit on a phone is the likeliest way this ledger
        # goes silently wrong: a duplicate dividend corrupts no share count, so
        # nothing else would ever reveal it.
        if is_duplicate(fetch_dividends(db, user.id), clean):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"You already recorded a dividend of {clean['amount']:.2f} EGP "
                    f"for {clean['symbol']} on {clean['pay_date']}."
                ),
            )

        now = datetime.utcnow().isoformat() + "Z"
        dividend_id = str(uuid.uuid4())
        name = body.get("name") or clean["symbol"]
        sector = body.get("sector") or ""
        notes = body.get("notes", "")

        db.execute(
            "INSERT INTO portfolio_dividends "
            "(id, user_id, symbol, name, sector, amount, pay_date, shares, notes, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                dividend_id, user.id, clean["symbol"], name, sector,
                clean["amount"], clean["pay_date"], clean["shares"], notes, now,
            ),
        )

        return {
            "dividend": enrich_dividend({
                "id": dividend_id,
                "symbol": clean["symbol"],
                "name": name,
                "sector": sector,
                "amount": clean["amount"],
                "pay_date": clean["pay_date"],
                "shares": clean["shares"],
                "notes": notes,
                "created_at": now,
            })
        }

    except DividendValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/dividends")
def delete_dividend(id: str = Query(...), user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        # RETURNING makes the delete its own existence check, so a 404 cannot
        # race a concurrent delete into a false success.
        row = db.execute(
            "DELETE FROM portfolio_dividends WHERE id = %s AND user_id = %s RETURNING id",
            (id, user.id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Dividend not found: {id}")
        return {"deleted": id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 2: Register the router in `app/main.py`**

Find the existing `sales` import and `include_router` call and add `dividends`
beside it, matching the file's established style exactly. Both the import and
the `app.include_router(dividends.router)` line are required.

- [ ] **Step 3: Verify the app imports cleanly**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -c "from app.main import app; print([r.path for r in app.routes if 'dividend' in r.path])"
```

Expected: `['/api/dividends', '/api/dividends']` (POST and DELETE).

- [ ] **Step 4: Run the suite**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/egx-api/egx-api-be && git add -A && git commit -m "feat: POST and DELETE /api/dividends"
```

---

## Task 5: `GET /api/sales` serves the combined ledger

**Files:**
- Modify: `egx-api-be/app/routers/sales.py`

**Interfaces:**
- Consumes: `fetch_dividends`, `summarize_realized`.
- Produces: `GET /api/sales` response gains `dividends: [...]`; `summary` gains `total_dividends`, `dividend_count`, `total_winnings`, and `by_symbol[].dividends` / `by_symbol[].total_winnings`.

- [ ] **Step 1: Update the module docstring**

Change the header block of `app/routers/sales.py` so the first lines read:

```python
"""
The realized ledger — what was closed, what was paid out, and what that came to.

POST   /api/sales          — record a full or partial sell
GET    /api/sales          — closed trades AND dividends, plus the combined summary
DELETE /api/sales?id=xxx   — undo a sale, restoring the shares

The GET serves BOTH ledgers on purpose. The Winnings headline is capital gains
plus dividends, and computing that sum here keeps it in tested Python — done in
the browser it would be the one number on the page with no test behind it.
Writes are split: sales here, dividends in routers/dividends.py.

Deliberately separate from /api/portfolio_analysis, which is the heaviest
endpoint in the app and flirts with the 30 s Vercel timeout. Neither ledger
needs a price fetch, so the Winnings card paints even on a run where the
analysis times out.

Every route is scoped by the caller's user_id from the JWT.
"""
```

- [ ] **Step 2: Add the import**

```python
from app.core.dividends import fetch_dividends, summarize_realized
```

replacing the `from app.core.dividends import summarize_realized` line added in
Task 2.

- [ ] **Step 3: Wire dividends into `get_sales`**

In `get_sales`, after `priced = [...]`, add:

```python
        dividends = fetch_dividends(db, user.id)
```

change the summary line to:

```python
            "summary": summarize_realized(priced, dividends),
```

and add `"dividends": dividends,` to the returned dict, directly after
`"sales": priced,`.

- [ ] **Step 4: Run the suite**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/egx-api/egx-api-be && git add -A && git commit -m "feat: GET /api/sales returns dividends and the combined summary"
```

---

## Task 6: User deletion clears the new table

**Files:**
- Modify: `egx-api-be/app/routers/users.py`
- Test: `egx-api-be/tests/test_dividends.py`

**Interfaces:** none produced.

**Why:** no table has a foreign key to `users`, so nothing cascades. Without
this, a deleted user's dividends survive as invisible orphan rows.

- [ ] **Step 1: Write the failing test**

Append to `egx-api-be/tests/test_dividends.py`:

```python
def test_deleting_a_user_also_deletes_their_dividends():
    """
    No table has an FK to users, so nothing cascades. A missed table leaves
    invisible orphan rows behind every deleted account.
    """
    source = Path(__file__).resolve().parents[1] / "app" / "routers" / "users.py"
    text = source.read_text(encoding="utf-8")
    assert "portfolio_dividends" in text, (
        "DELETE /api/users must clear portfolio_dividends inside its transaction"
    )


def test_every_user_scoped_table_is_cleared_in_one_transaction():
    """All five deletes must be inside the same db.transaction() block."""
    source = Path(__file__).resolve().parents[1] / "app" / "routers" / "users.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            body = ast.dump(node)
            if "portfolio_sales" in body:
                for table in ("portfolio_dividends", "portfolio", "watchlist",
                              "user_settings", "users"):
                    assert table in body, f"{table} is outside the transaction"
                return
    raise AssertionError("no transaction block deleting portfolio_sales was found")
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest tests/test_dividends.py -q -k user
```

Expected: both fail — `portfolio_dividends` is not in `users.py`.

- [ ] **Step 3: Add the delete**

In `app/routers/users.py`, inside the existing `with db.transaction() as tx:`
block, add the dividends delete immediately after the sales delete:

```python
            tx.execute("DELETE FROM portfolio_sales WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM portfolio_dividends WHERE user_id = %s", (user_id,))
            tx.execute("DELETE FROM portfolio WHERE user_id = %s", (user_id,))
```

Extend the comment above the block to name the new table.

- [ ] **Step 4: Run the suite**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/egx-api/egx-api-be && git add -A && git commit -m "fix: deleting a user clears their dividends too"
```

---

## Task 7: Dividends per open holding in portfolio analysis

**Files:**
- Modify: `egx-api-be/app/routers/portfolio_analysis.py`

**Interfaces:**
- Consumes: `fetch_dividend_totals` from `app.core.dividends`.
- Produces: every holding row in the response gains `dividends_collected: float` and `dividends_symbol_shared: bool`.

**Critical:** `_analyze(holdings, user_id)` also serves `POST
/api/portfolio_analysis`, where holdings come from the request body. The
dividend lookup **must key off `user_id`**, never off anything derived from
`holdings`.

- [ ] **Step 1: Add the import**

```python
from app.core.dividends import fetch_dividend_totals
```

- [ ] **Step 2: Fetch once, near the top of `_analyze`**

Immediately before the per-holding loop begins (alongside the other
initialisation such as `total_current_value = 0`), add:

```python
    # Dividends are symbol-anchored, so this is one indexed aggregate with no
    # price fetch — it costs nothing against the 30 s budget. It must key off
    # user_id and NOT off `holdings`, because POST /api/portfolio_analysis
    # takes holdings from the request body.
    dividends_by_symbol = fetch_dividend_totals(get_db(), user_id)

    # Nothing stops two portfolio rows sharing a symbol, and a dividend belongs
    # to the SYMBOL, not to one purchase lot. When that happens the UI labels
    # the figure as the symbol's total rather than the row's own — splitting it
    # by today's share count would be fiction, since the counts differed when
    # the dividend was paid.
    _symbol_counts: dict = {}
    for _h in holdings:
        _sym = (_h.get("symbol") or "").upper()
        _symbol_counts[_sym] = _symbol_counts.get(_sym, 0) + 1
```

`get_db` is already imported in this module; if it is not, add
`from app.core.db import get_db`.

- [ ] **Step 3: Add the fields to the successful holding dict**

In the `analysis = {` dict (the one beginning `"id": h.get("id"),`), add after
`"days_held": days_held,`:

```python
                "dividends_collected": dividends_by_symbol.get(symbol.upper(), 0.0),
                "dividends_symbol_shared": _symbol_counts.get(symbol.upper(), 0) > 1,
```

**`.upper()` on both lookups is not optional.** `_symbol_counts` is keyed
upper-cased, `validate_dividend` upper-cases before insert, and
`POST /api/portfolio` upper-cases too — so every key in play is upper. A lookup
that skips it works today and returns 0 the first time any path stores a
lower-case symbol, which reads as the money having vanished.

- [ ] **Step 4: Add the same fields to the error-row append**

Find every place an error row is appended (the branch that sets `"error":` on a
holding whose price feed failed) and add both fields there too, using the
holding's own symbol:

```python
                "dividends_collected": dividends_by_symbol.get(
                    (h.get("symbol") or "").upper(), 0.0
                ),
                "dividends_symbol_shared": _symbol_counts.get(
                    (h.get("symbol") or "").upper(), 0
                ) > 1,
```

A holding whose feed is down already keeps its Actions cell so it stays
sellable. A dividend figure that vanished on a feed error would read as the
money having disappeared.

- [ ] **Step 5: Run the suite**

```bash
cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/egx-api/egx-api-be && git add -A && git commit -m "feat: dividends collected per open holding"
```

---

## Task 8: Frontend types and API wrappers

**Files:**
- Modify: `egx-api-fe/src/app/lib/types.ts`
- Modify: `egx-api-fe/src/app/lib/api.ts`

**Interfaces:**
- Consumes: the backend response shapes from Tasks 4, 5 and 7.
- Produces: `Dividend`; extended `SalesSummary`, `SymbolRealized`, `SalesResponse`, `HoldingAnalysis`; `recordDividend(body)`, `deleteDividend(id)`.

- [ ] **Step 1: Add the types**

In `src/app/lib/types.ts`, in the `// Sales / Realized gains` section, add:

```ts
export interface Dividend {
  id: string;
  symbol: string;
  name: string;
  sector: string;
  /** Total EGP received, already net of withholding tax. Never a gross figure. */
  amount: number;
  pay_date: string;
  /** Optional — shares held when paid. Null when the user did not record it. */
  shares: number | null;
  notes: string;
  created_at: string;
  /** Computed server-side; null when shares is null or 0. */
  amount_per_share: number | null;
}
```

Add to `SymbolRealized`:

```ts
  dividends: number;
  total_winnings: number;
```

Add to `SalesSummary`:

```ts
  total_dividends: number;
  dividend_count: number;
  /** total_realized_pnl + total_dividends. The card's headline. */
  total_winnings: number;
```

Add to `SalesResponse`:

```ts
  dividends: Dividend[];
```

Add to **`HoldingAnalysis`** — the per-holding interface `HoldingsTable` types
its `holdings` prop with. It is `HoldingAnalysis`, **not** `StockAnalysis`;
confirm the name in `types.ts` before editing.

```ts
  /** EGP collected against this SYMBOL, 0 when none. */
  dividends_collected?: number;
  /** True when the user has more than one open holding of this symbol, in
   *  which case dividends_collected is the symbol's total, not this row's. */
  dividends_symbol_shared?: boolean;
```

Both are optional so a stale cached analysis response still type-checks.

- [ ] **Step 2: Add the API wrappers**

In `src/app/lib/api.ts`, directly after `deleteSale`, add:

```ts
// ---- Dividends ----
// Reads arrive on fetchSales(): the Winnings headline is gains + dividends and
// that sum is computed server-side, so the card needs one fetch, not two.

export async function recordDividend(body: {
  symbol: string;
  name?: string;
  sector?: string;
  amount: number;
  pay_date: string;
  shares?: number | null;
  notes?: string;
}): Promise<{ dividend: Dividend }> {
  return fetchJSON<{ dividend: Dividend }>(`${BASE}/dividends`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function deleteDividend(id: string): Promise<{ deleted: string }> {
  return fetchJSON(`${BASE}/dividends?id=${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}
```

Add `Dividend` to the existing type import block at the top of `api.ts`.

- [ ] **Step 3: Type-check**

```bash
cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
cd /d/Projects/egx-api/egx-api-fe && git add -A && git commit -m "feat: dividend types and API wrappers"
```

---

## Task 9: `RealizedGainsCard` shows total money made

**Files:**
- Modify: `egx-api-fe/src/app/components/RealizedGainsCard.tsx`

**Interfaces:**
- Consumes: `SalesSummary` with `total_winnings`, `total_dividends`, `dividend_count`, and `by_symbol[].dividends` / `.total_winnings`.
- Produces: no new props.

**Framing rule:** no string in this card may present a dividend as income or
compare it to the T-bill.

- [ ] **Step 1: Replace the tooltip text**

The current tooltip ends *"It counts capital gains only; dividends are not
included."* — that sentence is now false. Replace the whole `explanation` with:

```tsx
          explanation="Money you have actually banked: profit from selling, plus any dividends the companies paid you. Unlike the P&L on your open holdings this cannot go back down — the trades are closed and the cash is received."
```

- [ ] **Step 2: Make the headline the combined figure**

Replace the headline block:

```tsx
  const positive = summary.total_winnings >= 0;
```

and inside the `{/* Headline */}` div, change the value to
`{egp(summary.total_winnings)} EGP`.

Directly beneath the headline value, before the existing percentage line, add
the split — rendered only when there are dividends, so a user who records none
sees today's card unchanged:

```tsx
        {summary.total_dividends > 0 && (
          <p className="mt-1 font-mono text-xs text-white/50">
            {egp(summary.total_realized_pnl)} from sales
            <span className="mx-1.5 text-white/20">·</span>
            <span className="text-gain">
              {summary.total_dividends.toLocaleString(undefined, {
                maximumFractionDigits: 0,
              })}
            </span>{" "}
            in dividends
            <span className="ml-1 text-white/30">
              ({summary.dividend_count})
            </span>
          </p>
        )}
```

- [ ] **Step 3: Make the percentage line say what it describes**

The headline above it now includes more than closed trades, so the existing
percentage line must name its own scope. Change its text to:

```tsx
          <p className="mt-1 text-sm text-white/40">
            {summary.total_realized_pnl_pct >= 0 ? "+" : ""}
            {summary.total_realized_pnl_pct.toFixed(2)}% on{" "}
            {summary.total_cost.toLocaleString(undefined, {
              maximumFractionDigits: 0,
            })}{" "}
            EGP invested in closed trades
          </p>
```

- [ ] **Step 4: Show dividends in the per-stock breakdown**

In the `by_symbol.map` block, replace the right-hand figure column so the
symbol's total leads and the dividend component is named beneath it:

```tsx
              <div className="shrink-0 text-right">
                <p
                  className={`font-mono text-sm font-semibold ${
                    s.total_winnings >= 0 ? "text-gain" : "text-loss"
                  }`}
                >
                  {egp(s.total_winnings)}
                </p>
                {s.dividends > 0 && (
                  <p className="text-[10px] text-white/40">
                    incl.{" "}
                    {s.dividends.toLocaleString(undefined, {
                      maximumFractionDigits: 0,
                    })}{" "}
                    dividends
                  </p>
                )}
                {s.dividends === 0 && s.realized_pnl_pct !== null && (
                  <p className="text-[10px] text-white/30">
                    {s.realized_pnl_pct >= 0 ? "+" : ""}
                    {s.realized_pnl_pct.toFixed(1)}%
                  </p>
                )}
              </div>
```

And in the left-hand label, handle a symbol that was never sold:

```tsx
                <p className="truncate text-[10px] text-white/30">
                  {s.sales_count === 0
                    ? "Dividends only — not sold"
                    : `${s.quantity} shares · ${s.sales_count} ${
                        s.sales_count === 1 ? "sale" : "sales"
                      }`}
                </p>
```

- [ ] **Step 5: Type-check and build**

```bash
cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit && npm run build
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/egx-api/egx-api-fe && git add -A && git commit -m "feat: Winnings card headline is gains plus dividends"
```

---

## Task 10: `AddDividendForm`

**Files:**
- Create: `egx-api-fe/src/app/components/AddDividendForm.tsx`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: default export `AddDividendForm` with props
  `{ symbols, presetSymbol?, onSubmit, onCancel?, error?, onDismissError? }`,
  plus the named export `DividendSymbolOption`. `onSubmit` receives
  `{symbol, name, sector, amount, pay_date, shares, notes}`. Name, sector and
  the share count are looked up from `symbols` by the picked symbol — they are
  **not** separate props.

The error/scroll-lock/iOS-zoom decisions below are carried from
`SellHoldingForm`, where each was a real shipped defect.

- [ ] **Step 1: Write the component**

Create `egx-api-fe/src/app/components/AddDividendForm.tsx`:

```tsx
"use client";

import { useState } from "react";

export interface DividendSymbolOption {
  symbol: string;
  name: string;
  sector: string;
  /** Shares currently held, when this came from an open holding. */
  shares?: number | null;
}

interface AddDividendFormProps {
  /** Symbols the user can pick from — open holdings AND past sales, so a
   *  dividend can be logged against a position already closed. */
  symbols: DividendSymbolOption[];
  presetSymbol?: string;
  onSubmit: (data: {
    symbol: string;
    name: string;
    sector: string;
    amount: number;
    pay_date: string;
    shares: number | null;
    notes: string;
  }) => Promise<void> | void;
  onCancel?: () => void;
  /** Rejection from the API, rendered inside the form. On mobile the form
   *  fills the viewport, so a banner on the page behind it is invisible. */
  error?: string | null;
  /** Called on the first edit after a rejection so a stale message clears. */
  onDismissError?: () => void;
}

export default function AddDividendForm({
  symbols,
  presetSymbol,
  onSubmit,
  onCancel,
  error = null,
  onDismissError,
}: AddDividendFormProps) {
  const [symbol, setSymbol] = useState(presetSymbol ?? symbols[0]?.symbol ?? "");
  const [amount, setAmount] = useState("");
  const [payDate, setPayDate] = useState(new Date().toISOString().slice(0, 10));
  const [shares, setShares] = useState(() => {
    const preset = symbols.find((s) => s.symbol === presetSymbol);
    return preset?.shares != null ? String(preset.shares) : "";
  });
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Every field edit clears the last rejection: the message described the
  // values that were submitted, not the ones now on screen.
  const edited = () => {
    if (error) onDismissError?.();
  };

  const picked = symbols.find((s) => s.symbol === symbol);
  const amountNum = parseFloat(amount);
  const sharesNum = shares.trim() === "" ? null : parseInt(shares, 10);

  const validAmount = Number.isFinite(amountNum) && amountNum > 0;
  const validShares =
    sharesNum === null || (Number.isFinite(sharesNum) && sharesNum > 0);
  const canSubmit = Boolean(symbol) && validAmount && validShares && !submitting;

  const perShare =
    validAmount && sharesNum ? amountNum / sharesNum : null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      await onSubmit({
        symbol,
        name: picked?.name ?? symbol,
        sector: picked?.sector ?? "",
        amount: amountNum,
        pay_date: payDate,
        shares: sharesNum,
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
      <h3 className="mb-1 text-sm font-medium text-white/70">Record a dividend</h3>
      <p className="mb-4 text-xs text-white/40">
        Cash the company paid you for holding it. Enter what actually landed in
        your account — already after the dividend tax.
      </p>

      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <label className="mb-1 block text-xs text-white/40">Stock *</label>
          <select
            value={symbol}
            onChange={(e) => {
              setSymbol(e.target.value);
              const next = symbols.find((s) => s.symbol === e.target.value);
              setShares(next?.shares != null ? String(next.shares) : "");
              edited();
            }}
            className="min-h-[44px] w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-[16px] text-white outline-none focus:border-accent/50 md:text-sm"
            required
          >
            {!symbols.length && <option value="">No stocks yet</option>}
            {symbols.map((s) => (
              <option key={s.symbol} value={s.symbol} className="bg-charcoal">
                {s.symbol}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs text-white/40">
            Amount received (EGP) *
          </label>
          <input
            type="number"
            value={amount}
            onChange={(e) => {
              setAmount(e.target.value);
              edited();
            }}
            min={0.01}
            step={0.01}
            placeholder="1200.00"
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-[16px] text-white placeholder-white/30 outline-none focus:border-accent/50 md:text-sm"
            required
          />
          <p className="mt-1 text-[10px] text-white/30">
            The total that reached your account
          </p>
        </div>

        <div>
          <label className="mb-1 block text-xs text-white/40">Pay Date</label>
          {/* No `min`: a dividend is anchored to the stock, not to one
              purchase, so no buy date bounds it. `max` is today. */}
          <input
            type="date"
            value={payDate}
            max={new Date().toISOString().slice(0, 10)}
            onChange={(e) => {
              setPayDate(e.target.value);
              edited();
            }}
            className="w-full min-w-0 appearance-none rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-left text-[16px] text-white outline-none focus:border-accent/50 md:text-sm"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs text-white/40">
            Shares held (optional)
          </label>
          <input
            type="number"
            value={shares}
            onChange={(e) => {
              setShares(e.target.value);
              edited();
            }}
            min={1}
            placeholder="500"
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-[16px] text-white placeholder-white/30 outline-none focus:border-accent/50 md:text-sm"
          />
          <p className="mt-1 text-[10px] text-white/30">
            Only used to show a per-share figure
          </p>
        </div>

        <div className="md:col-span-2">
          <label className="mb-1 block text-xs text-white/40">
            Notes (optional)
          </label>
          <input
            type="text"
            value={notes}
            onChange={(e) => {
              setNotes(e.target.value);
              edited();
            }}
            placeholder="e.g. 2025 annual dividend"
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-[16px] text-white placeholder-white/30 outline-none focus:border-accent/50 md:text-sm"
          />
        </div>
      </div>

      {perShare !== null && (
        <div className="mt-4 rounded-lg border border-white/5 bg-white/[0.03] p-3">
          <p className="text-xs text-white/40">That works out to</p>
          <p className="font-mono text-lg font-semibold text-gain">
            {perShare.toFixed(2)} EGP per share
          </p>
        </div>
      )}

      {/* Sits directly above the buttons so the rejection is on screen at the
          same place the user just tapped — the mobile form is a full-screen
          modal and a banner on the page behind it would never be seen. */}
      {error && (
        <div
          role="alert"
          className="mt-4 rounded-lg border border-loss/30 bg-loss/10 p-3 text-xs text-loss"
        >
          {error}
        </div>
      )}

      <div className="mt-4 flex gap-3">
        <button
          type="submit"
          disabled={!canSubmit}
          className="flex min-h-[44px] items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-charcoal-dark transition-opacity hover:opacity-90 disabled:opacity-30"
        >
          {submitting ? "Recording…" : "Record Dividend"}
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

- [ ] **Step 2: Type-check**

```bash
cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
cd /d/Projects/egx-api/egx-api-fe && git add -A && git commit -m "feat: AddDividendForm"
```

---

## Task 11: `DividendsTable`

**Files:**
- Create: `egx-api-fe/src/app/components/DividendsTable.tsx`

**Interfaces:**
- Consumes: `Dividend[]` from Task 8.
- Produces: default export with props `{ dividends: Dividend[]; onDelete: (id: string) => void }`.

Follows `ClosedPositionsTable` exactly: collapsed `<details>`, mobile cards /
desktop table, delete behind a confirmation dialog.

- [ ] **Step 1: Write the component**

Create `egx-api-fe/src/app/components/DividendsTable.tsx`:

```tsx
"use client";

import { useState } from "react";
import type { Dividend } from "../lib/types";

interface DividendsTableProps {
  dividends: Dividend[];
  onDelete: (id: string) => void;
}

function egp(value: number, digits = 0): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export default function DividendsTable({
  dividends,
  onDelete,
}: DividendsTableProps) {
  const [confirmDelete, setConfirmDelete] = useState<{
    id: string;
    symbol: string;
  } | null>(null);

  if (!dividends.length) return null;

  const total = dividends.reduce((sum, d) => sum + d.amount, 0);

  return (
    <>
      <details className="rounded-xl border border-white/5 bg-white/[0.02]">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-white/70 md:px-6">
          Dividends Received
          <span className="ml-2 text-xs text-white/30">
            ({dividends.length} · {egp(total)} EGP)
          </span>
        </summary>

        {/* Mobile cards */}
        <div className="space-y-3 px-4 pb-4 md:hidden">
          {dividends.map((d) => (
            <div key={d.id} className="rounded-lg bg-white/[0.03] p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="font-mono text-sm font-medium text-white">
                    {d.symbol}
                  </p>
                  <p className="mt-0.5 text-[10px] text-white/30">
                    Paid {d.pay_date}
                    {d.amount_per_share !== null && (
                      <> · {d.amount_per_share.toFixed(2)} EGP/share</>
                    )}
                  </p>
                </div>
                <p className="shrink-0 font-mono text-sm font-semibold text-gain">
                  +{egp(d.amount)}
                </p>
              </div>
              {d.notes && (
                <p className="mt-2 truncate text-[10px] text-white/30">{d.notes}</p>
              )}
              <button
                onClick={() => setConfirmDelete({ id: d.id, symbol: d.symbol })}
                className="mt-2 min-h-[44px] w-full rounded-lg border border-white/10 text-xs text-white/40"
              >
                Remove
              </button>
            </div>
          ))}
        </div>

        {/* Desktop table */}
        <div className="hidden overflow-x-auto px-6 pb-4 md:block">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5 text-left text-xs text-white/40">
                <th className="py-2 pr-4 font-medium">Symbol</th>
                <th className="py-2 pr-4 font-medium">Paid</th>
                <th className="py-2 pr-4 font-medium">Amount</th>
                <th className="py-2 pr-4 font-medium">Per Share</th>
                <th className="py-2 pr-4 font-medium">Notes</th>
                <th className="py-2 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {dividends.map((d) => (
                <tr key={d.id} className="border-b border-white/5">
                  <td className="py-3 pr-4 font-mono text-white">{d.symbol}</td>
                  <td className="py-3 pr-4 text-white/40">{d.pay_date}</td>
                  <td className="py-3 pr-4 font-mono font-semibold text-gain">
                    +{egp(d.amount)}
                  </td>
                  <td className="py-3 pr-4 font-mono text-white/60">
                    {d.amount_per_share !== null
                      ? d.amount_per_share.toFixed(2)
                      : "—"}
                  </td>
                  <td className="py-3 pr-4 text-xs text-white/40">
                    {d.notes || "—"}
                  </td>
                  <td className="py-3 text-right">
                    <button
                      onClick={() =>
                        setConfirmDelete({ id: d.id, symbol: d.symbol })
                      }
                      className="text-xs text-white/30 hover:text-white/60"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
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
            <p className="text-base font-semibold text-white">
              Remove this dividend?
            </p>
            <p className="mt-2 text-sm text-white/50">
              The{" "}
              <span className="font-mono text-white/80">
                {confirmDelete.symbol}
              </span>{" "}
              dividend will be removed from your winnings. Your holdings are not
              affected.
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
                Remove
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
```

- [ ] **Step 2: Type-check**

```bash
cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
cd /d/Projects/egx-api/egx-api-fe && git add -A && git commit -m "feat: DividendsTable"
```

---

## Task 12: Dividend pill and action in `HoldingsTable`

**Files:**
- Modify: `egx-api-fe/src/app/components/HoldingsTable.tsx`

**Interfaces:**
- Consumes: `dividends_collected`, `dividends_symbol_shared` from Task 8.
- Produces: new required prop `onAddDividend: (holding: HoldingAnalysis) => void`.
  The element type is `HoldingAnalysis` — that is what `HoldingsTableProps.holdings`
  is already typed with in this file.

**Do NOT add a table column.** The desktop table coordinates `colSpan={12}` on
the expanded row against `colSpan={10}` on the error row; a thirteenth column
means editing both in lockstep, and a colSpan mismatch in this exact file is a
recorded shipped bug. A pill carries the same information at no structural risk.

- [ ] **Step 1: Add the prop**

In `interface HoldingsTableProps`, after `onSell`, add:

```tsx
  onAddDividend: (holding: HoldingAnalysis) => void;
```

and destructure `onAddDividend` in the component signature.

- [ ] **Step 2: Add the shared pill helper**

Above the component, add:

```tsx
/** Dividends are anchored to the SYMBOL, not to one purchase lot. When the user
 *  holds the same symbol in more than one row, this figure is that symbol's
 *  total — so it says so, rather than pretending to be this row's own. */
function DividendPill({ holding }: { holding: HoldingAnalysis }) {
  const amount = holding.dividends_collected ?? 0;
  if (amount <= 0) return null;
  return (
    <span className="ml-2 inline-flex items-center rounded-full bg-gain/10 px-2 py-0.5 text-[10px] font-medium text-gain">
      +{amount.toLocaleString(undefined, { maximumFractionDigits: 0 })} div
      {holding.dividends_symbol_shared ? " (all lots)" : ""}
    </span>
  );
}
```

- [ ] **Step 3: Render the pill beside the symbol, both breakpoints**

In the mobile card header, immediately after
`<span className="font-mono text-sm font-medium text-white">{h.symbol}</span>`,
add `<DividendPill holding={h} />`.

In the desktop Symbol cell (the `<td>` rendering `{h.symbol}` in the non-error
row), add `<DividendPill holding={h} />` directly after the symbol text.

- [ ] **Step 4: Add the labelled line to the expanded detail**

In both the mobile expanded block and the desktop expanded row (`colSpan={12}`),
add a stat entry matching the surrounding pattern, rendered only when there are
dividends:

```tsx
{(h.dividends_collected ?? 0) > 0 && (
  <div>
    <p className="text-[10px] uppercase tracking-wide text-white/30">
      Dividends {h.dividends_symbol_shared ? `(all ${h.symbol} lots)` : "collected"}
    </p>
    <p className="mt-0.5 font-mono text-sm text-gain">
      +{(h.dividends_collected ?? 0).toLocaleString(undefined, {
        maximumFractionDigits: 0,
      })}{" "}
      EGP
    </p>
  </div>
)}
```

- [ ] **Step 5: Add the "Dividend" action beside "Sell"**

In both the mobile action row and the desktop Actions cell, add a button
alongside the existing Sell one, styled to match its siblings:

```tsx
<button
  onClick={() => onAddDividend(h)}
  className="min-h-[44px] text-xs text-white/40 hover:text-gain"
>
  Dividend
</button>
```

Add it to the **error-row** action cells too. A holding whose price feed is down
is already sellable; there is no reason a dividend on it cannot be recorded.

- [ ] **Step 6: Type-check and build**

```bash
cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit && npm run build
```

Expected: `tsc` reports `portfolio/page.tsx` is missing the new required
`onAddDividend` prop. That is expected and is fixed in Task 13. Confirm there is
**no other** error, then proceed.

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/egx-api/egx-api-fe && git add -A && git commit -m "feat: dividend pill and action on holdings"
```

---

## Task 13: Wire the portfolio page

**Files:**
- Modify: `egx-api-fe/src/app/portfolio/page.tsx`

**Interfaces:**
- Consumes: `AddDividendForm`, `DividendsTable`, `recordDividend`, `deleteDividend`, and the widened `SalesResponse`.
- Produces: nothing.

- [ ] **Step 1: Add imports and state**

Import `AddDividendForm` and `DividendsTable` beside the existing component
imports, and `recordDividend`, `deleteDividend` beside `recordSale`.

Add state next to the sell state:

```tsx
  const [dividendFor, setDividendFor] = useState<{
    symbol: string;
    name: string;
    sector: string;
    shares: number | null;
  } | null>(null);
  const [dividendError, setDividendError] = useState<string | null>(null);
```

- [ ] **Step 2: Extend the scroll lock**

The lock and the modal's render condition must be the same expression — a lock
firing on a value the modal does not render on freezes the page with nothing on
screen. Change:

```tsx
    if (!showForm && !sellModalOpen) return;
```

to:

```tsx
    if (!showForm && !sellModalOpen && !dividendFor) return;
```

and add `dividendFor` to that effect's dependency array.

- [ ] **Step 3: Add the handlers**

```tsx
  const handleAddDividend = (holding: {
    symbol: string;
    name?: string;
    sector?: string;
    quantity?: number;
  }) => {
    // Reopening starts clean — a rejection from a previous attempt describes
    // values that are no longer on screen.
    setDividendError(null);
    setDividendFor({
      symbol: holding.symbol,
      name: holding.name ?? holding.symbol,
      sector: holding.sector ?? "",
      shares: holding.quantity ?? null,
    });
  };

  const closeDividendForm = () => {
    setDividendError(null);
    setDividendFor(null);
  };

  const handleDividendSubmit = async (data: {
    symbol: string;
    name: string;
    sector: string;
    amount: number;
    pay_date: string;
    shares: number | null;
    notes: string;
  }) => {
    try {
      setDividendError(null);
      await recordDividend(data);
      setDividendFor(null);
      // A dividend changes no share count, so the heavy analysis only needs
      // re-running for the per-holding figure — which comes from it.
      await Promise.all([refreshAfterMutation(), loadSales()]);
    } catch (e: any) {
      // Into the form, not the page banner: on mobile the form covers the whole
      // viewport, so a banner behind it is invisible and the rejection reads as
      // the button doing nothing. This is also where the 409 duplicate message
      // surfaces.
      setDividendError(e.message);
    }
  };

  const handleDeleteDividend = async (id: string) => {
    try {
      await deleteDividend(id);
      await Promise.all([refreshAfterMutation(), loadSales()]);
    } catch (e: any) {
      setError(e.message);
    }
  };
```

- [ ] **Step 4: Build the symbol list**

Above the `return`, add:

```tsx
  // Open holdings AND symbols with a trading history, so a dividend can be
  // recorded against a position that has already been sold.
  const dividendSymbols = useMemo(() => {
    const map = new Map<
      string,
      { symbol: string; name: string; sector: string; shares: number | null }
    >();
    for (const s of sales?.sales ?? []) {
      map.set(s.symbol, {
        symbol: s.symbol, name: s.name, sector: s.sector, shares: null,
      });
    }
    // Open holdings win — they carry a live share count.
    for (const h of portfolio?.portfolio ?? []) {
      map.set(h.symbol, {
        symbol: h.symbol,
        name: h.name ?? h.symbol,
        sector: h.sector ?? "",
        shares: h.quantity ?? null,
      });
    }
    return Array.from(map.values()).sort((a, b) =>
      a.symbol.localeCompare(b.symbol)
    );
  }, [sales, portfolio]);
```

Add `useMemo` to the `react` import if it is not already there.

- [ ] **Step 5: Render the modal**

Directly after the existing sell modal block (`{sellingHolding && ( … )}`), add
the same structure for dividends — full-screen on mobile, centred card on
desktop, matching the sell modal's markup and z-index exactly:

```tsx
        {dividendFor && (
          <>
            {/* Mobile: full-screen */}
            <div className="fixed inset-0 z-[60] flex flex-col bg-charcoal-dark md:hidden">
              <div className="flex items-center justify-between border-b border-white/5 px-4 py-3">
                <h2 className="text-sm font-medium text-white">Record Dividend</h2>
                <button
                  onClick={closeDividendForm}
                  className="min-h-[44px] px-2 text-sm text-white/50"
                >
                  Close
                </button>
              </div>
              <div
                className="flex-1 overflow-y-auto p-4"
                style={{ WebkitOverflowScrolling: "touch" }}
              >
                <AddDividendForm
                  symbols={dividendSymbols}
                  presetSymbol={dividendFor.symbol}
                  onSubmit={handleDividendSubmit}
                  onCancel={closeDividendForm}
                  error={dividendError}
                  onDismissError={() => setDividendError(null)}
                />
              </div>
            </div>
            {/* Desktop: inline card */}
            <div className="mb-6 hidden md:block">
              <AddDividendForm
                symbols={dividendSymbols}
                presetSymbol={dividendFor.symbol}
                onSubmit={handleDividendSubmit}
                onCancel={closeDividendForm}
                error={dividendError}
                onDismissError={() => setDividendError(null)}
              />
            </div>
          </>
        )}
```

- [ ] **Step 6: Widen the three render gates**

Without this a user whose only record is a dividend sees nothing.

The never-traded empty state — change:

```tsx
          !sales?.sales.length ? (
```

to:

```tsx
          !sales?.sales.length &&
          !sales?.dividends.length ? (
```

The full-page Winnings card and closed-positions gates — change both
`{sales && sales.sales.length > 0 && (` conditions to:

```tsx
            {sales && (sales.sales.length > 0 || sales.dividends.length > 0) && (
              <RealizedGainsCard … />
            )}
            {sales && sales.sales.length > 0 && (
              <ClosedPositionsTable … />
            )}
            {sales && (
              <DividendsTable
                dividends={sales.dividends}
                onDelete={handleDeleteDividend}
              />
            )}
```

`ClosedPositionsTable` keeps its sales-only gate — it renders nothing for an
empty list anyway, and so does `DividendsTable`.

In the **sold-out branch** (`/* Sold out, but there is a trading history … */`),
add `DividendsTable` after `ClosedPositionsTable`, with the same
`onDelete={handleDeleteDividend}`.

- [ ] **Step 7: Pass the new prop to `HoldingsTable`**

```tsx
              <HoldingsTable
                holdings={analysis.holdings}
                onEdit={handleEdit}
                onDelete={handleDelete}
                onSell={handleSell}
                onAddDividend={handleAddDividend}
              />
```

- [ ] **Step 8: Type-check and build**

```bash
cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit && npm run build
```

Expected: both clean, including the `onAddDividend` error from Task 12.

- [ ] **Step 9: Commit**

```bash
cd /d/Projects/egx-api/egx-api-fe && git add -A && git commit -m "feat: wire dividends into the portfolio page"
```

---

## Task 14: Learn page concept and CLAUDE.md

**Files:**
- Modify: `egx-api-fe/src/app/learn/page.tsx`
- Modify: `egx-api-be/CLAUDE.md`
- Modify: `egx-api-fe/CLAUDE.md`

**Interfaces:** none.

- [ ] **Step 1: Add the Learn concept**

In `src/app/learn/page.tsx`, immediately after the `Concept` carrying
`id="realized_gains"`, add:

```tsx
          <Concept
            id="dividends"
            title="Dividends (Profit Share)"
            definition="A dividend is cash a company pays you out of its profits, just for holding the shares — no selling involved. On the EGX it is announced per share, and what reaches your account is already after the 5-10% dividend tax."
            whyItMatters="With Egyptian T-bills near 25%, no EGX dividend yield competes as income — even a strong 8% loses to simply leaving the money in T-bills. What a dividend IS good evidence of is that the company generates real cash rather than accounting profit. Judge it that way, not as an income stream."
            howToUse="Record what actually landed in your account, not the announced gross. The app adds it to your realized winnings and shows it against the stock that paid it, so a bank holding that is flat on price but pays steadily does not read as dead money. A dividend worth more than about 15% of the share price is usually a special payout or a collapsed price — not income quality."
          />
```

`Concept` takes exactly `{ id?, title, definition, whyItMatters, howToUse }` —
confirm against the component definition at the top of the file.

**Do not** write any string comparing a collected dividend favourably to the
T-bill rate. That framing rule is in CLAUDE.md and binds this whole feature.

- [ ] **Step 2: Update CLAUDE.md in the backend repo**

Make these edits to `egx-api-be/CLAUDE.md`:

1. **Directory Layout** — add `dividends.py` under `core/` ("Dividend ledger
   maths + the one spelling of its queries") and under `routers/`
   (`POST/DELETE /api/dividends`).
2. **API Endpoints** — a new section after the `/api/sales` one:

```markdown
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

**Accepted display consequence:** nothing stops two `portfolio` rows sharing a
symbol. When that happens the per-holding figure is that SYMBOL's total, labelled
"(all lots)". Splitting it by today's share count would be fiction — the counts
differed when the dividend was paid. No aggregate is affected: every total sums
the ledger directly and never reaches it through holdings.
```

3. **`GET /api/sales`** — note that it now returns `dividends` and that
   `summary` comes from `summarize_realized`, and that
   `total_realized_pnl_pct`, `beat_t_bill_count` / `annualizable_count` and
   `best_trade` / `worst_trade` all stay **capital-gains-only**, with the reason.
4. **Database Schema** — add, after the `portfolio_sales` block:

```sql
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
```
5. **`/api/users` DELETE** — the list of cleared tables becomes five, with
   `portfolio_dividends` named.
6. **Frontend Components** — add `AddDividendForm`, `DividendsTable`; note
   `RealizedGainsCard`'s headline is now gains + dividends and that
   `HoldingsTable` shows a dividend pill rather than a column, with the colSpan
   reason.
7. **Portfolio page section** — mention the Dividend action per holding.

- [ ] **Step 3: Mirror to the frontend repo**

```bash
cp /d/Projects/egx-api/egx-api-be/CLAUDE.md /d/Projects/egx-api/egx-api-fe/CLAUDE.md
cp /d/Projects/egx-api/egx-api-be/CLAUDE.md /d/Projects/egx-api/CLAUDE.md
diff /d/Projects/egx-api/egx-api-be/CLAUDE.md /d/Projects/egx-api/egx-api-fe/CLAUDE.md && echo "IDENTICAL"
```

Expected: `IDENTICAL`.

- [ ] **Step 4: Type-check and build**

```bash
cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit && npm run build
```

Expected: both clean.

- [ ] **Step 5: Commit both repos**

```bash
cd /d/Projects/egx-api/egx-api-be && git add -A && git commit -m "docs: dividends in CLAUDE.md"
cd /d/Projects/egx-api/egx-api-fe && git add -A && git commit -m "docs: dividends Learn concept and CLAUDE.md"
```

---

## Final Verification

- [ ] Backend suite green: `cd /d/Projects/egx-api/egx-api-be && ./.venv/Scripts/python.exe -m pytest -q`
- [ ] Frontend clean: `cd /d/Projects/egx-api/egx-api-fe && npx tsc --noEmit && npm run build`
- [ ] `git status` clean in both repos

## Manual Verification (cannot be automated — no DB or browser in the suite)

1. `portfolio_dividends` lands on a cold start with 10 columns and its index.
2. Record a dividend; `GET /api/sales` returns it and `total_winnings` equals
   gains plus dividends.
3. Submit the identical dividend again → **409**, and **no second row written**.
4. Record a dividend on a symbol with no sales → appears in `by_symbol` with
   `sales_count: 0`.
5. Delete a holding that has dividends → dividends survive, Winnings unchanged.
6. Delete a user → zero rows remain in `portfolio_dividends` for them.
7. At 375px: modal opens and scroll-locks; an invalid amount errors **inside**
   the modal; the 409 message is readable; `DividendsTable` expands; the pill
   shows on a holding; a duplicated symbol shows "(all lots)".
