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
