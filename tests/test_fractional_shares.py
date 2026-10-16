"""
Fractional shares: whole on the EGX, up to six decimals in the US.

A market fact (MarketConfig.share_decimals). Decimal arithmetic so a position
sold in fractional parts closes at exactly zero, and whole quantities stay
ints on the wire so EG responses are unchanged. No network, no Postgres.
"""
from __future__ import annotations

import os
from datetime import date
from decimal import Decimal

os.environ.setdefault("AUTH_SECRET", "test-secret-for-the-gate")
os.environ.setdefault("DATABASE_URL", "postgresql://unused/unused")

import pytest  # noqa: E402

from app.core.markets import EG, US  # noqa: E402
from app.core.quantity import fmt, out, parse_shares  # noqa: E402
from app.core.sales import (  # noqa: E402
    SaleValidationError, compute_sale_metrics, group_sale_orders,
    plan_sale_allocation, validate_position_sale,
)

TODAY = date(2026, 10, 16)


def test_share_rules_are_market_facts():
    assert EG.share_decimals == 0 and US.share_decimals == 6


@pytest.mark.parametrize("value", ["3", 3, 3.0, "3.000"])
def test_eg_accepts_whole_shares_and_returns_ints(value):
    assert out(parse_shares(value, EG.share_decimals)) == 3
    assert isinstance(out(parse_shares(value, EG.share_decimals)), int)


@pytest.mark.parametrize("value, msg", [
    (1.5, "whole number"), ("0.5", "whole number"), (0, "at least 1"),
    (None, "whole number"), (True, "whole number"), ("abc", "whole number"),
])
def test_eg_refuses_fractions_with_its_old_wording(value, msg):
    with pytest.raises(ValueError, match=msg):
        parse_shares(value, EG.share_decimals)


@pytest.mark.parametrize("value, expected", [
    ("0.5", 0.5), (0.25, 0.25), ("1.123456", 1.123456), (10, 10), ("2.50", 2.5),
])
def test_us_accepts_fractions_to_six_places(value, expected):
    assert out(parse_shares(value, US.share_decimals)) == expected


@pytest.mark.parametrize("value", ["1.1234567", 0, -1, "nan", "inf"])
def test_us_refuses_bad_quantities(value):
    with pytest.raises(ValueError):
        parse_shares(value, US.share_decimals)


def test_wire_and_message_forms():
    assert out(Decimal("300.000000")) == 300 and isinstance(out(Decimal("300.000000")), int)
    assert out(Decimal("0.500000")) == 0.5
    assert fmt(Decimal("0.100000")) == "0.1" and fmt(300) == "300"


def _lot(id_, qty, buy_date="2026-01-05"):
    return {"id": id_, "symbol": "AAPL", "quantity": qty, "buy_price": 100.0,
            "buy_date": buy_date, "created_at": buy_date}


def test_fractional_lots_sell_out_to_exactly_zero():
    """0.1 + 0.2 as floats is 0.30000000000000004; as shares it must be 0.3."""
    lots = [_lot("a", Decimal("0.1"), "2026-01-01"), _lot("b", Decimal("0.2"), "2026-02-01")]
    plan = plan_sale_allocation(lots, Decimal("0.3"))
    assert [(p["id"], p["quantity"]) for p in plan] == [("a", 0.1), ("b", 0.2)]
    assert sum(Decimal(str(p["quantity"])) for p in plan) == Decimal("0.3")


def test_us_sale_validates_against_fractional_holdings():
    lots = [_lot("a", Decimal("1.5"))]
    clean = validate_position_sale(lots=lots, quantity="0.75", sell_price=120,
                                   sell_date="2026-10-01", today=TODAY,
                                   share_decimals=US.share_decimals)
    assert clean["quantity"] == 0.75
    with pytest.raises(SaleValidationError, match="You hold 1.5 shares — you cannot sell 2"):
        validate_position_sale(lots=lots, quantity=2, sell_price=120,
                               sell_date="2026-10-01", today=TODAY,
                               share_decimals=US.share_decimals)


def test_eg_sale_still_refuses_a_fraction():
    with pytest.raises(SaleValidationError, match="whole number of shares"):
        validate_position_sale(lots=[_lot("a", 10)], quantity=1.5, sell_price=10,
                               sell_date="2026-10-01", today=TODAY)


def test_fractional_sale_metrics_and_orders():
    sale = {"symbol": "AAPL", "name": "Apple", "sector": "Tech", "notes": "", "quantity": 0.5, "buy_price": 200.0, "sell_price": 210.0,
            "buy_date": "2026-01-02", "sell_date": "2026-10-01"}
    m = compute_sale_metrics(sale, 4.2)
    assert m["cost"] == 100.0 and m["proceeds"] == 105.0 and m["realized_pnl"] == 5.0
    orders = group_sale_orders([{**m, "id": "s1", "sale_group_id": "g", "created_at": "t", "lots_count": 1},
                                {**m, "id": "s2", "sale_group_id": "g", "created_at": "t", "quantity": 0.25,
                                 "cost": 50.0, "proceeds": 52.5}])
    assert orders[0]["quantity"] == 0.75


def test_dividend_shares_follow_the_market():
    from app.core.dividends import DividendValidationError, validate_dividend
    us = validate_dividend(symbol="AAPL", amount=1.3, pay_date="2026-08-14", shares="5.25",
                           today=TODAY, share_decimals=US.share_decimals)
    assert us["shares"] == 5.25
    with pytest.raises(DividendValidationError, match="whole number"):
        validate_dividend(symbol="COMI", amount=10, pay_date="2026-08-14", shares="5.25",
                          today=TODAY)


def test_add_holding_validates_per_market(monkeypatch):
    from fastapi import HTTPException
    from alert_fakes import FakeDB
    import app.routers.portfolio as portfolio_mod
    from app.core.auth import CurrentUser

    db = FakeDB()
    monkeypatch.setattr(portfolio_mod, "get_db", lambda: db)
    monkeypatch.setattr(portfolio_mod.universe, "resolve_exchange", lambda *a: "NASDAQ")
    monkeypatch.setenv("ENABLED_MARKETS", "EG,US")
    me = CurrentUser(id="u1", email="u@example.com", name="U", role="user")

    got = portfolio_mod.add_holding({"symbol": "aapl", "buy_price": 200, "quantity": "0.25"},
                                    market=US, user=me)
    assert got["quantity"] == 0.25
    [(_, params)] = db.statements("INSERT INTO portfolio")
    assert params[7] == Decimal("0.25")

    with pytest.raises(HTTPException) as e:
        portfolio_mod.add_holding({"symbol": "comi", "buy_price": 10, "quantity": 1.5},
                                  market=EG, user=me)
    assert e.value.status_code == 400 and "whole number" in e.value.detail

    got = portfolio_mod.add_holding({"symbol": "comi", "buy_price": 10, "quantity": 3},
                                    market=EG, user=me)
    assert got["quantity"] == 3 and isinstance(got["quantity"], int)


def test_share_columns_are_exact_numeric():
    import inspect
    from app.core import db as db_mod
    src = inspect.getsource(db_mod.init_db)
    assert "quantity INTEGER" not in src and "shares     INTEGER" not in src
    assert db_mod.SHARE_COLUMNS == (("portfolio", "quantity"), ("portfolio_sales", "quantity"),
                                    ("portfolio_dividends", "shares"))


def test_markets_route_sends_the_input_rules():
    from datetime import date
    from app.core.markets import public_market
    assert public_market(EG, date(2026, 10, 16))["share_decimals"] == 0
    assert public_market(EG, date(2026, 10, 16))["price_decimals"] == 3
    us = public_market(US, date(2026, 10, 16))
    assert (us["share_decimals"], us["price_decimals"]) == (6, 4)
