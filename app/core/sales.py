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

from app.core.returns import (annualized_cash_rate_pct, annualized_return,
                               days_between)


class SaleValidationError(ValueError):
    """A sell request that is not internally consistent. Maps to HTTP 400."""


def _parse_date(value: str) -> Optional[date]:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _coerce_quantity(quantity) -> int:
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        try:
            if float(quantity) != int(float(quantity)):
                raise ValueError
            quantity = int(float(quantity))
        except (TypeError, ValueError):
            raise SaleValidationError("Quantity must be a whole number of shares.")
    if quantity <= 0:
        raise SaleValidationError("Quantity must be at least 1 share.")
    return quantity


def _coerce_price(sell_price) -> float:
    try:
        sell_price = float(sell_price)
    except (TypeError, ValueError):
        raise SaleValidationError("Sell price must be a number.")
    if sell_price <= 0:
        raise SaleValidationError("Sell price must be greater than 0.")
    return sell_price


def _coerce_sell_date(sell_date, today: date) -> date:
    if sell_date in (None, ""):
        parsed = today
    else:
        parsed = _parse_date(sell_date)
        if parsed is None:
            raise SaleValidationError("Sell date must be a date like 2026-09-01.")
    if parsed > today:
        raise SaleValidationError("Sell date cannot be in the future.")
    return parsed


def sort_lots(lots: list[dict]) -> list[dict]:
    """
    Oldest purchase first.

    All three keys are in the sort so the order is TOTAL: two lots bought the
    same day would otherwise be split in whatever order the database happened
    to return, and the same sell request could allocate differently on a retry.
    """
    return sorted(
        lots,
        key=lambda lot: (
            str(lot.get("buy_date") or ""),
            str(lot.get("created_at") or ""),
            str(lot.get("id") or ""),
        ),
    )


def plan_sale_allocation(lots: list[dict], quantity: int) -> list[dict]:
    """
    Split one sell across the open lots of a symbol, oldest first (FIFO).

    Buying the same symbol twice leaves two `portfolio` rows, but the user owns
    one position and sells a share count out of it. This turns that count into
    per-lot parts, each carrying THAT lot's cost basis and buy date — the
    caller writes one `portfolio_sales` row per part.

    One blended row was the alternative and it is wrong: `compute_sale_metrics`
    annualizes each trade over its own holding window and grades it against the
    policy rate that prevailed across it, so a blended basis would invent a
    purchase that never happened. Per-lot rows also keep DELETE /api/sales
    restoring shares to the lot they came from.

    Returns a copy of each lot with `quantity` set to the part being sold and
    `lot_quantity` to what that lot held before it.
    """
    quantity = int(quantity)
    open_lots = [lot for lot in lots if int(lot.get("quantity") or 0) > 0]
    held = sum(int(lot["quantity"]) for lot in open_lots)
    if quantity > held:
        raise SaleValidationError(
            f"You hold {held} shares — you cannot sell {quantity}."
        )

    remaining = quantity
    plan: list[dict] = []
    for lot in sort_lots(open_lots):
        if remaining <= 0:
            break
        available = int(lot["quantity"])
        take = min(available, remaining)
        plan.append({**lot, "quantity": take, "lot_quantity": available})
        remaining -= take
    return plan


def validate_position_sale(*, lots: list[dict], quantity, sell_price, sell_date,
                           today: date) -> dict:
    """
    Check a sell request against the whole POSITION — every open lot of one
    symbol — and return the FIFO plan alongside the normalized fields.

    Returns {quantity, sell_price, sell_date, allocation}. Raises
    SaleValidationError with a message written for the user, not the log.
    """
    open_lots = [lot for lot in lots if int(lot.get("quantity") or 0) > 0]
    held = sum(int(lot["quantity"]) for lot in open_lots)

    quantity = _coerce_quantity(quantity)
    if quantity > held:
        raise SaleValidationError(
            f"You hold {held} shares — you cannot sell {quantity}."
        )
    sell_price = _coerce_price(sell_price)
    parsed = _coerce_sell_date(sell_date, today)

    allocation = plan_sale_allocation(open_lots, quantity)

    # Only the lots actually being sold constrain the date. Selling 150 of a
    # 200+100 position touches the January lot alone, so a February sell date
    # is legitimate even though a March lot exists.
    consumed_buys = [d for d in (_parse_date(p.get("buy_date") or "") for p in allocation)
                     if d is not None]
    if consumed_buys:
        newest = max(consumed_buys)
        if parsed < newest:
            where = ("the buy date" if len(allocation) == 1
                     else "the buy date of the newest lot you are selling")
            raise SaleValidationError(
                f"Sell date cannot be before {where} ({newest.isoformat()})."
            )

    return {
        "quantity": quantity,
        "sell_price": sell_price,
        "sell_date": parsed.isoformat(),
        "allocation": allocation,
    }


def compute_sale_metrics(sale: dict, risk_free_rate_pct: float,
                         rate_steps=None) -> dict:
    """
    Add realized P&L, holding period and the T-bill verdict to one sale.

    `rate_steps` is the dated policy-rate history — [(date, annual_pct), ...]
    from `macro_series.get_risk_free_steps`. When supplied, the trade is graded
    against the rate that actually prevailed across ITS holding window rather
    than against today's scalar, which over-credited every trade closed while
    the rate was higher than it is now (27.25% in March 2024 against 19% today).

    It stays OPTIONAL, and `risk_free_rate_pct` remains the fallback, so a
    caller with no history — or a window the history does not reach — degrades
    to the previous behaviour instead of failing. The hurdle actually applied is
    returned as `t_bill_hurdle_pct` either way, because a card cannot explain a
    verdict it cannot see.
    """
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

    hurdle = (annualized_cash_rate_pct(rate_steps, sale["buy_date"], sale["sell_date"])
              if rate_steps else None)
    if hurdle is None:
        hurdle = float(risk_free_rate_pct)

    return {
        **sale,
        "cost": round(cost, 2),
        "proceeds": round(proceeds, 2),
        "realized_pnl": round(realized_pnl, 2),
        "realized_pnl_pct": round(realized_pnl_pct, 2) if realized_pnl_pct is not None else None,
        "days_held": days_held,
        "annualized_return_pct": round(ann, 1) if ann is not None else None,
        "t_bill_hurdle_pct": round(hurdle, 2),
        "beat_t_bill": (ann > hurdle) if ann is not None else None,
    }
