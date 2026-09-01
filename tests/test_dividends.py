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
