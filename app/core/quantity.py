"""
Share quantities — whole shares on the EGX, fractional shares in the US.

A market fact (`MarketConfig.share_decimals`): EG 0 — the EGX has no
fractional shares; US 6 — US brokers hold fractional positions, quoted to six
decimal places. The columns are NUMERIC(24, 6) so the database stores exactly
what the user entered.

Arithmetic is DECIMAL, never float: a position of 0.1 + 0.2 shares sold in
full must leave exactly 0, not 5.6e-17 shares that keep a "sold" lot open.

On the wire a whole quantity is an int (EG responses are byte-identical to
before) and a fractional one a float with at most six decimals.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Union

Number = Union[int, float]


def to_decimal(value) -> Decimal:
    """Exact Decimal of a stored or parsed quantity (int, float, str, Decimal)."""
    if isinstance(value, Decimal):
        return value
    if value is None:
        return Decimal(0)
    # str(), never Decimal(float): Decimal(0.1) is 0.1000000000000000055…
    return Decimal(str(value))


def out(value) -> Number:
    """Wire form: int when whole, else float (≤ 6 decimals survive a float)."""
    d = to_decimal(value)
    if d == d.to_integral_value():
        return int(d)
    return float(d.normalize())


def fmt(value) -> str:
    """For messages: '300', '1.5', '0.123456' — no trailing zeros, no exponent."""
    d = to_decimal(value)
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def parse_shares(value, share_decimals: int, *, what: str = "Quantity") -> Decimal:
    """
    A user-entered share count, validated against the market's rule.
    Raises ValueError with a message written for the user.
    """
    bad = (f"{what} must be a whole number of shares." if share_decimals == 0
           else f"{what} must be a number of shares.")
    if isinstance(value, bool) or value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(bad)
    try:
        d = to_decimal(value.strip() if isinstance(value, str) else value)
    except (InvalidOperation, ValueError):
        raise ValueError(bad)
    if not d.is_finite():
        raise ValueError(bad)
    if share_decimals == 0:
        if d != d.to_integral_value():
            raise ValueError(f"{what} must be a whole number of shares.")
        if d < 1:
            raise ValueError(f"{what} must be at least 1 share.")
        return Decimal(int(d))
    if d <= 0:
        raise ValueError(f"{what} must be greater than 0.")
    if -d.normalize().as_tuple().exponent > share_decimals:
        raise ValueError(f"{what} can have at most {share_decimals} decimal places.")
    return d
