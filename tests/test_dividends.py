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


# --- summarize_realized --------------------------------------------------

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
