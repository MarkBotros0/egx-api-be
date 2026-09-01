"""
Tests for sell tracking and realized gains.

Run from egx-api-be:  python -m pytest tests/test_sell_tracking.py -v
"""

from contextlib import contextmanager
from datetime import date

import pytest

from app.core.db import _DB


# ---------------------------------------------------------------------------
# Fakes — the DB wrapper is tested without a real Postgres
# ---------------------------------------------------------------------------

class _FakeCursor:
    description = None

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, log):
        self.log = log
        self.committed = False
        self.rolled_back = False

    def execute(self, sql, params=()):
        self.log.append(sql)
        return _FakeCursor()

    @contextmanager
    def transaction(self):
        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        self.committed = True


class _FakePool:
    def __init__(self):
        self.conns = []
        self.log = []

    @contextmanager
    def connection(self):
        conn = _FakeConn(self.log)
        self.conns.append(conn)
        yield conn


def test_transaction_runs_every_statement_on_one_connection():
    # execute() takes a FRESH pooled connection per call, so a sale insert and
    # its quantity decrement issued through it could half-land.
    pool = _FakePool()
    db = _DB(pool)

    with db.transaction() as tx:
        tx.execute("UPDATE portfolio SET quantity = 1")
        tx.execute("INSERT INTO portfolio_sales VALUES (1)")

    assert len(pool.conns) == 1, "both statements must share one connection"
    assert pool.conns[0].committed
    assert len(pool.log) == 2


def test_transaction_rolls_back_on_error():
    pool = _FakePool()
    db = _DB(pool)

    with pytest.raises(RuntimeError):
        with db.transaction() as tx:
            tx.execute("UPDATE portfolio SET quantity = 1")
            raise RuntimeError("over-sold")

    assert pool.conns[0].rolled_back
    assert not pool.conns[0].committed


from app.core.returns import (
    MIN_DAYS_FOR_ANNUALIZATION,
    annualized_return,
    days_between,
)


def test_annualization_is_suppressed_below_thirty_days():
    # A +5% week annualizes to five figures. The UI must show nothing.
    assert annualized_return(5.0, 29) is None
    assert annualized_return(5.0, MIN_DAYS_FOR_ANNUALIZATION) is not None


def test_annualized_return_over_one_year_is_the_plain_return():
    assert annualized_return(10.0, 365) == pytest.approx(10.0, abs=0.01)


def test_two_year_gain_annualizes_below_the_t_bill():
    # The whole point of the T-bill line: +8% held two years LOST to cash.
    ann = annualized_return(8.0, 730)
    assert ann == pytest.approx(3.92, abs=0.05)
    assert ann < 25.0


def test_total_loss_floors_at_minus_one_hundred():
    assert annualized_return(-100.0, 365) == -100.0
    assert annualized_return(-150.0, 365) == -100.0


def test_days_between_counts_calendar_days():
    assert days_between("2026-01-01", date(2026, 1, 31)) == 30
    assert days_between("2026-01-01T00:00:00Z", date(2026, 1, 31)) == 30


def test_days_between_returns_zero_on_unparseable_input():
    assert days_between("not-a-date", date(2026, 1, 31)) == 0


from pathlib import Path

_ROUTERS = Path(__file__).resolve().parents[1] / "app" / "routers"


def test_open_holdings_filter_is_spelled_once():
    """
    portfolio.py and portfolio_analysis.py each issue their own SELECT against
    the portfolio table — portfolio_analysis does NOT call /api/portfolio. If
    either hand-writes the `quantity > 0` filter, the two can drift and a
    fully-sold holding reappears in the risk metrics as a phantom position.
    """
    for name in ("portfolio.py", "portfolio_analysis.py"):
        src = (_ROUTERS / name).read_text(encoding="utf-8")
        assert "quantity > 0" not in src, (
            f"{name} hand-writes the open-holdings filter; "
            "call core.holdings.fetch_open_holdings instead"
        )
        assert "fetch_open_holdings" in src, (
            f"{name} must read holdings through core.holdings.fetch_open_holdings"
        )


from app.core.sales import (
    SaleValidationError,
    compute_sale_metrics,
    summarize_sales,
    validate_sale,
)

TODAY = date(2026, 9, 1)


def _holding(quantity=100, buy_price=50.0, buy_date="2026-01-01"):
    return {
        "id": "h1", "symbol": "COMI", "name": "Commercial International Bank",
        "sector": "Banks", "quantity": quantity, "buy_price": buy_price,
        "buy_date": buy_date,
    }


def _sale(quantity=100, buy_price=50.0, sell_price=60.0,
          buy_date="2026-01-01", sell_date="2026-09-01", symbol="COMI"):
    return {
        "id": "s1", "holding_id": "h1", "symbol": symbol,
        "name": "Commercial International Bank", "sector": "Banks",
        "quantity": quantity, "buy_price": buy_price, "buy_date": buy_date,
        "sell_price": sell_price, "sell_date": sell_date, "notes": "",
        "created_at": "2026-09-01T00:00:00Z",
    }


# ---- validation ----

def test_partial_sell_is_accepted():
    out = validate_sale(holding=_holding(quantity=100), quantity=40,
                        sell_price=60.0, sell_date="2026-09-01", today=TODAY)
    assert out == {"quantity": 40, "sell_price": 60.0, "sell_date": "2026-09-01"}


def test_selling_the_whole_position_is_accepted():
    out = validate_sale(holding=_holding(quantity=100), quantity=100,
                        sell_price=60.0, sell_date="2026-09-01", today=TODAY)
    assert out["quantity"] == 100


def test_over_selling_is_rejected_and_names_the_remaining_quantity():
    with pytest.raises(SaleValidationError) as exc:
        validate_sale(holding=_holding(quantity=100), quantity=101,
                      sell_price=60.0, sell_date="2026-09-01", today=TODAY)
    assert "100" in str(exc.value)


@pytest.mark.parametrize("bad_quantity", [0, -5, "abc", None, 1.5])
def test_non_positive_or_non_integer_quantity_is_rejected(bad_quantity):
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=bad_quantity,
                      sell_price=60.0, sell_date="2026-09-01", today=TODAY)


@pytest.mark.parametrize("bad_price", [0, -1.0, "abc", None])
def test_non_positive_sell_price_is_rejected(bad_price):
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=10,
                      sell_price=bad_price, sell_date="2026-09-01", today=TODAY)


def test_sell_date_before_buy_date_is_rejected():
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(buy_date="2026-06-01"), quantity=10,
                      sell_price=60.0, sell_date="2026-05-31", today=TODAY)


def test_future_sell_date_is_rejected():
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=10, sell_price=60.0,
                      sell_date="2026-09-02", today=TODAY)


def test_sell_date_defaults_to_today():
    out = validate_sale(holding=_holding(), quantity=10, sell_price=60.0,
                        sell_date=None, today=TODAY)
    assert out["sell_date"] == "2026-09-01"


def test_unparseable_sell_date_is_rejected():
    with pytest.raises(SaleValidationError):
        validate_sale(holding=_holding(), quantity=10, sell_price=60.0,
                      sell_date="tomorrow", today=TODAY)


# ---- per-sale metrics ----

def test_realized_pnl_and_pct():
    m = compute_sale_metrics(_sale(quantity=100, buy_price=50.0, sell_price=60.0),
                             risk_free_rate_pct=25.0)
    assert m["cost"] == 5000.0
    assert m["proceeds"] == 6000.0
    assert m["realized_pnl"] == 1000.0
    assert m["realized_pnl_pct"] == pytest.approx(20.0)


def test_a_loss_is_negative_not_absolute():
    m = compute_sale_metrics(_sale(buy_price=60.0, sell_price=50.0),
                             risk_free_rate_pct=25.0)
    assert m["realized_pnl"] == -1000.0
    assert m["realized_pnl_pct"] == pytest.approx(-16.67, abs=0.01)


def test_a_two_year_eight_percent_win_did_not_beat_the_t_bill():
    # The reason the T-bill line exists at all.
    m = compute_sale_metrics(
        _sale(buy_price=100.0, sell_price=108.0,
              buy_date="2024-09-01", sell_date="2026-09-01"),
        risk_free_rate_pct=25.0,
    )
    assert m["realized_pnl"] > 0
    assert m["beat_t_bill"] is False


def test_a_quick_flip_reports_no_annualized_figure():
    m = compute_sale_metrics(
        _sale(buy_date="2026-08-20", sell_date="2026-09-01"),
        risk_free_rate_pct=25.0,
    )
    assert m["days_held"] == 12
    assert m["annualized_return_pct"] is None
    assert m["beat_t_bill"] is None


def test_zero_buy_price_reports_null_pct_but_exact_egp():
    m = compute_sale_metrics(_sale(buy_price=0.0, sell_price=60.0),
                             risk_free_rate_pct=25.0)
    assert m["realized_pnl_pct"] is None
    assert m["realized_pnl"] == 6000.0


def test_metrics_preserve_the_original_sale_fields():
    m = compute_sale_metrics(_sale(), risk_free_rate_pct=25.0)
    assert m["id"] == "s1" and m["symbol"] == "COMI"


# ---- summary ----

def test_empty_summary_is_zeroed_not_null():
    s = summarize_sales([])
    assert s["total_realized_pnl"] == 0
    assert s["total_realized_pnl_pct"] is None
    assert s["win_count"] == 0 and s["loss_count"] == 0
    assert s["by_symbol"] == []
    assert s["best_trade"] is None and s["worst_trade"] is None


def test_total_pct_is_cost_weighted_not_a_mean_of_percentages():
    # 10000 cost -> +1000 (+10%), 1000 cost -> +500 (+50%).
    # Mean of percentages says +30%. Cost-weighted truth is +13.64%.
    priced = [
        compute_sale_metrics(_sale(quantity=100, buy_price=100.0, sell_price=110.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=100.0, sell_price=150.0,
                                   symbol="SWDY"), 25.0),
    ]
    s = summarize_sales(priced)
    assert s["total_realized_pnl"] == 1500.0
    assert s["total_realized_pnl_pct"] == pytest.approx(13.64, abs=0.01)


def test_wins_and_losses_are_counted_and_extremes_identified():
    priced = [
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=60.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=40.0,
                                   symbol="SWDY"), 25.0),
    ]
    s = summarize_sales(priced)
    assert s["win_count"] == 1 and s["loss_count"] == 1
    assert s["best_trade"]["symbol"] == "COMI"
    assert s["worst_trade"]["symbol"] == "SWDY"


def test_by_symbol_aggregates_multiple_sales_and_sorts_by_pnl():
    priced = [
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=60.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=70.0), 25.0),
        compute_sale_metrics(_sale(quantity=10, buy_price=50.0, sell_price=55.0,
                                   symbol="SWDY"), 25.0),
    ]
    s = summarize_sales(priced)
    assert [b["symbol"] for b in s["by_symbol"]] == ["COMI", "SWDY"]
    comi = s["by_symbol"][0]
    assert comi["sales_count"] == 2 and comi["quantity"] == 20
    assert comi["realized_pnl"] == 300.0
    assert comi["realized_pnl_pct"] == pytest.approx(30.0)


def test_t_bill_counts_ignore_trades_too_short_to_annualize():
    priced = [
        # Two years, +8% — annualizable, loses to the T-bill.
        compute_sale_metrics(_sale(buy_price=100.0, sell_price=108.0,
                                   buy_date="2024-09-01", sell_date="2026-09-01"), 25.0),
        # One year, +40% — annualizable, beats it.
        compute_sale_metrics(_sale(buy_price=100.0, sell_price=140.0,
                                   buy_date="2025-09-01", sell_date="2026-09-01"), 25.0),
        # Twelve days — not annualizable, must not count either way.
        compute_sale_metrics(_sale(buy_date="2026-08-20", sell_date="2026-09-01"), 25.0),
    ]
    s = summarize_sales(priced)
    assert s["annualizable_count"] == 2
    assert s["beat_t_bill_count"] == 1
