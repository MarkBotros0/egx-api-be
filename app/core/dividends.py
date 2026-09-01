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
