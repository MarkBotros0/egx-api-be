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
