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
