"""
Grading a closed trade against the rate that actually prevailed while it ran.

WHY THIS EXISTS
---------------
`compute_sale_metrics` took ONE scalar risk-free rate and applied it to every
trade in the ledger. The CBE policy rate has ranged from 8.25% to 27.25% over
the period this app covers, so one scalar is wrong for every era but the
current one.

It errs in BOTH directions, which is the part that makes it more than cosmetic.
Measured against the real EGINTR series: a +25% gain held through 2024 faced a
true hurdle of 26.07% and LOST to cash, where the flat 19% called it a win; a
+18% gain over 2019 faced 14.69% and BEAT cash, where the flat 19% called it a
loss. `beat_t_bill_count` on the Winnings card was era-dependent noise, not a
consistent bias that could be mentally corrected for.

`macro_series` now holds EGINTR back to 2001-05, so the honest hurdle is
computable: compound the policy rate day by day across the holding window, then
annualize it so it is comparable to the trade's own annualized return.

These are pure-function tests — tests/ has no Postgres fixture by design.
"""

from __future__ import annotations

import pytest

from app.core.returns import annualized_cash_rate_pct
from app.core.sales import compute_sale_metrics


# ---------------------------------------------------------------------------
# annualized_cash_rate_pct — the cash benchmark itself
# ---------------------------------------------------------------------------

def test_a_flat_rate_returns_itself():
    """
    The anchor property. If the rate never moved, the equivalent annual rate
    over any window IS that rate — no compounding artifact, no drift. Every
    other case is a deviation from this one.
    """
    steps = [("2020-01-01", 19.0)]
    got = annualized_cash_rate_pct(steps, "2024-01-01", "2025-01-01")
    assert got == pytest.approx(19.0, abs=1e-9)


def test_a_flat_rate_returns_itself_over_a_short_window_too():
    """Annualizing must not depend on the window being a whole year."""
    steps = [("2020-01-01", 19.0)]
    got = annualized_cash_rate_pct(steps, "2024-01-01", "2024-03-01")
    assert got == pytest.approx(19.0, abs=1e-9)


def test_a_rate_cut_midway_lands_between_the_two_rates():
    steps = [("2020-01-01", 27.25), ("2024-07-01", 19.0)]
    got = annualized_cash_rate_pct(steps, "2024-01-01", "2025-01-01")
    assert 19.0 < got < 27.25


def test_the_longer_held_rate_dominates():
    """
    Time-weighted, not a simple average of the levels present. A window that
    spent eleven of twelve months at 27.25% must sit near 27.25%.
    """
    steps = [("2020-01-01", 27.25), ("2024-12-01", 19.0)]
    got = annualized_cash_rate_pct(steps, "2024-01-01", "2025-01-01")
    assert got > 26.0


def test_the_earliest_known_rate_is_carried_backwards():
    """
    A trade that starts before our history does still needs a hurdle, and the
    nearest historical rate is a far better estimate than today's. Falling back
    to the current scalar is exactly the bug being fixed, so the left edge must
    not do it.
    """
    steps = [("2024-01-01", 27.25)]
    got = annualized_cash_rate_pct(steps, "2023-01-01", "2023-06-01")
    assert got == pytest.approx(27.25, abs=1e-9)


def test_no_rate_history_yields_no_hurdle():
    """So the caller can fall back deliberately rather than inventing a rate."""
    assert annualized_cash_rate_pct([], "2024-01-01", "2025-01-01") is None


def test_a_zero_length_window_yields_no_hurdle():
    steps = [("2020-01-01", 19.0)]
    assert annualized_cash_rate_pct(steps, "2024-01-01", "2024-01-01") is None


def test_an_inverted_window_yields_no_hurdle():
    steps = [("2020-01-01", 19.0)]
    assert annualized_cash_rate_pct(steps, "2025-01-01", "2024-01-01") is None


def test_unsorted_steps_are_handled():
    """The caller should not have to guarantee ordering to get a right answer."""
    jumbled = [("2024-07-01", 19.0), ("2020-01-01", 27.25)]
    ordered = [("2020-01-01", 27.25), ("2024-07-01", 19.0)]
    assert (annualized_cash_rate_pct(jumbled, "2024-01-01", "2025-01-01")
            == annualized_cash_rate_pct(ordered, "2024-01-01", "2025-01-01"))


# ---------------------------------------------------------------------------
# compute_sale_metrics — the verdict that reaches the Winnings card
# ---------------------------------------------------------------------------

def _sale(**over):
    base = {
        "symbol": "COMI", "quantity": 100,
        "buy_price": 100.0, "sell_price": 125.0,
        "buy_date": "2024-01-01", "sell_date": "2025-01-01",
    }
    base.update(over)
    return base


def test_a_trade_is_graded_against_the_rate_that_prevailed_not_todays():
    """
    THE REGRESSION. +25% over a year beats today's 19% but loses to the 27.25%
    that was actually on offer while the trade ran. Grading it against today's
    rate calls it a win that it was not.
    """
    steps = [("2020-01-01", 27.25)]

    lenient = compute_sale_metrics(_sale(), 19.0)
    honest = compute_sale_metrics(_sale(), 19.0, rate_steps=steps)

    assert lenient["beat_t_bill"] is True
    assert honest["beat_t_bill"] is False


def test_the_dated_hurdle_can_also_ACQUIT_a_trade_the_flat_rate_failed():
    """
    The mirror of the test above, and the reason this is a correctness fix
    rather than a tightening. Before 2022 the CBE sat WELL BELOW 19%, so the
    flat rate failed trades that genuinely beat the cash on offer at the time.
    A fix that only ever raised the hurdle would be a different bug.
    """
    steps = [("2018-01-01", 14.0)]
    sale = _sale(buy_date="2019-01-01", sell_date="2020-01-01", sell_price=118.0)

    assert compute_sale_metrics(sale, 19.0)["beat_t_bill"] is False
    assert compute_sale_metrics(sale, 19.0, rate_steps=steps)["beat_t_bill"] is True


def test_the_hurdle_actually_used_is_reported():
    """
    The card cannot explain a verdict it cannot see. Without this the user is
    told they lost to cash but not to WHICH cash rate.
    """
    steps = [("2020-01-01", 27.25)]
    out = compute_sale_metrics(_sale(), 19.0, rate_steps=steps)
    assert out["t_bill_hurdle_pct"] == pytest.approx(27.25, abs=0.05)


def test_without_history_the_scalar_still_applies():
    """Backward compatible: every existing caller keeps its behaviour."""
    out = compute_sale_metrics(_sale(), 19.0)
    assert out["beat_t_bill"] is True
    assert out["t_bill_hurdle_pct"] == pytest.approx(19.0, abs=1e-9)


def test_empty_history_falls_back_to_the_scalar_rather_than_failing():
    out = compute_sale_metrics(_sale(), 19.0, rate_steps=[])
    assert out["beat_t_bill"] is True
    assert out["t_bill_hurdle_pct"] == pytest.approx(19.0, abs=1e-9)


def test_a_trade_too_short_to_annualize_still_has_no_verdict():
    """
    A dated hurdle must not resurrect a comparison the holding period cannot
    support. Under MIN_DAYS_FOR_ANNUALIZATION there is no annualized return to
    compare, so there is no verdict either.
    """
    short = _sale(buy_date="2024-01-01", sell_date="2024-01-10")
    out = compute_sale_metrics(short, 19.0, rate_steps=[("2020-01-01", 27.25)])
    assert out["annualized_return_pct"] is None
    assert out["beat_t_bill"] is None


# ---------------------------------------------------------------------------
# get_risk_free_steps — reading the history out of macro_series
# ---------------------------------------------------------------------------

class _StubDB:
    def __init__(self, rows):
        self.rows = rows
        self.sql = None
        self.params = None

    def execute(self, sql, params=()):
        self.sql, self.params = sql, params
        return self

    def fetchall(self):
        return self.rows


def test_the_policy_rate_history_comes_back_as_dated_steps():
    from app.core.macro_series import get_risk_free_steps

    db = _StubDB([("2024-01-01", 27.25), ("2026-02-01", 19.0)])
    assert get_risk_free_steps(db) == [("2024-01-01", 27.25), ("2026-02-01", 19.0)]


def test_the_steps_come_from_the_same_series_the_macro_card_shows():
    """
    Otherwise the rate on screen and the rate trades are graded against could
    drift apart silently — two spellings of one fact.
    """
    from app.core.macro_series import RISK_FREE_SERIES, get_risk_free_steps

    db = _StubDB([])
    get_risk_free_steps(db)
    assert RISK_FREE_SERIES in db.params


def test_no_history_yields_no_steps_rather_than_raising():
    """The Winnings card must still paint when macro_series is empty."""
    from app.core.macro_series import get_risk_free_steps

    assert get_risk_free_steps(_StubDB([])) == []


def test_a_broken_rate_history_does_not_take_down_the_ledger():
    """
    Realized gains need no price fetch and must paint even when other things
    fail — that is why the sales router is separate from portfolio_analysis.
    A macro read that raises must not undo that.
    """
    from app.core.macro_series import get_risk_free_steps

    class Exploding:
        def execute(self, *a, **k):
            raise RuntimeError("connection lost")

    assert get_risk_free_steps(Exploding()) == []


# ---------------------------------------------------------------------------
# The wiring. Without this the whole feature silently never reaches production.
# ---------------------------------------------------------------------------

def test_the_sales_router_actually_passes_the_dated_history():
    """
    A pure function nobody calls with the new argument is a no-op. This project
    has shipped that failure before — the `%` in get_weights_from_db meant saved
    weights were never once read back, and nothing noticed for months.

    Walks the AST of routers/sales.py and fails if any compute_sale_metrics call
    omits rate_steps.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "routers" / "sales.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
           == "compute_sale_metrics"
    ]
    assert calls, "compute_sale_metrics is no longer called from the sales router"

    missing = [c.lineno for c in calls
               if not any(kw.arg == "rate_steps" for kw in c.keywords)
               and len(c.args) < 3]
    assert not missing, (
        f"compute_sale_metrics called without rate_steps at line(s) {missing} — "
        "those trades are graded against today's rate instead of the rate that "
        "prevailed while they ran"
    )


def test_the_rate_history_is_read_once_per_request_not_once_per_sale():
    """
    A ledger of 50 closed trades must not make 50 macro queries. The steps are
    the same for every trade, so they are fetched once and passed down.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "routers" / "sales.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            continue
        inner = [n for n in ast.walk(node)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", None))
                    == "get_risk_free_steps"]
        assert not inner, (
            f"get_risk_free_steps is called inside a comprehension at line "
            f"{node.lineno} — that is one DB round trip per closed trade"
        )


# ---------------------------------------------------------------------------
# The backtest side. scripts/backtest.py ran the rate FLAT and withheld
# Risk-Adjusted's verdict because of it; dated rates are what lift that.
# ---------------------------------------------------------------------------

def test_risk_adjusted_stays_confounded_when_there_is_no_rate_history():
    """
    THE GUARD. The verdict may only be lifted by the data that justifies
    lifting it. A backtest run on a machine with no macro history must keep
    withholding, not quietly start reporting a number it cannot support.
    """
    from scripts.backtest import confounded_categories

    assert confounded_categories([]) == {"risk_adjusted"}
    assert confounded_categories(None) == {"risk_adjusted"}


def test_risk_adjusted_is_released_once_dated_rates_exist():
    from scripts.backtest import confounded_categories

    assert confounded_categories([("2020-01-01", 19.0)]) == set()


def test_the_backtest_grades_each_date_against_its_own_trailing_year():
    """
    score_risk_adjusted compares a TRAILING ONE-YEAR stock return against cash,
    so the hurdle must be the cash return over that same trailing year — not a
    point rate, and not today's.
    """
    from datetime import date

    from scripts.backtest import rate_for_date

    steps = [("2015-01-01", 12.0), ("2023-01-01", 27.25)]
    assert rate_for_date(steps, date(2020, 6, 1), fallback=19.0) == pytest.approx(12.0, abs=1e-9)
    assert rate_for_date(steps, date(2024, 6, 1), fallback=19.0) == pytest.approx(27.25, abs=1e-9)


def test_the_backtest_falls_back_to_the_flat_rate_without_history():
    from datetime import date

    from scripts.backtest import rate_for_date

    assert rate_for_date([], date(2024, 6, 1), fallback=19.0) == 19.0


def test_the_backtest_never_uses_a_rate_published_after_the_scoring_date():
    """
    Unlike the sales ledger — which scores an outcome that already happened —
    the backtest simulates a DECISION at a past date, so a rate set after that
    date is look-ahead. A 2024 hike must not touch a 2020 scoring date.
    """
    from datetime import date

    from scripts.backtest import rate_for_date

    without_future = rate_for_date([("2015-01-01", 12.0)], date(2020, 6, 1), fallback=19.0)
    with_future = rate_for_date(
        [("2015-01-01", 12.0), ("2023-01-01", 27.25)], date(2020, 6, 1), fallback=19.0
    )
    assert without_future == with_future


def test_a_pandas_timestamp_is_accepted_as_a_window_bound():
    """
    The backtest walks a pandas DatetimeIndex, so every scoring date arrives as
    a Timestamp while the steps parse to datetime.date. Comparing the two raises
    TypeError in pandas, which inside score_symbol would be swallowed by its
    bare  — every symbol would silently score zero rows and
    the run would report an empty panel with no error.
    """
    import pandas as pd

    steps = [("2015-01-01", 12.0)]
    got = annualized_cash_rate_pct(steps, pd.Timestamp("2023-06-01"),
                                   pd.Timestamp("2024-06-01"))
    assert got == pytest.approx(12.0, abs=1e-9)


def test_rate_for_date_accepts_the_timestamp_the_backtest_actually_passes():
    import pandas as pd

    from scripts.backtest import rate_for_date

    got = rate_for_date([("2015-01-01", 12.0)], pd.Timestamp("2020-06-01"),
                        fallback=19.0)
    assert got == pytest.approx(12.0, abs=1e-9)


# ---------------------------------------------------------------------------
# A panel must record HOW it was scored. analyze_backtest reads this to decide
# whether Risk-Adjusted's row is evidence or a caveat, and guessing from the
# local machine would let an old flat-rate panel be read as a dated one.
# ---------------------------------------------------------------------------

def test_a_panel_with_no_stamp_is_treated_as_flat_rate():
    """
    Conservative by default. Every panel built before this existed is a
    flat-rate panel, and reading one as dated would present a confounded number
    as evidence — the exact failure the withholding exists to prevent.
    """
    from scripts.backtest import read_run_meta

    meta = read_run_meta("/no/such/panel.pkl")
    assert meta["dated_risk_free"] is False
    assert meta["confounded"] == ["risk_adjusted"]


def test_a_dated_run_stamps_the_panel_and_clears_the_caveat(tmp_path):
    from scripts.backtest import read_run_meta, write_run_meta

    panel = str(tmp_path / "panel.pkl")
    write_run_meta(panel, [("2001-05-01", 11.0), ("2026-08-01", 19.0)])

    meta = read_run_meta(panel)
    assert meta["dated_risk_free"] is True
    assert meta["rate_steps"] == 2
    assert meta["confounded"] == []


def test_a_flat_run_stamps_the_panel_and_keeps_the_caveat(tmp_path):
    from scripts.backtest import read_run_meta, write_run_meta

    panel = str(tmp_path / "panel.pkl")
    write_run_meta(panel, [])

    meta = read_run_meta(panel)
    assert meta["dated_risk_free"] is False
    assert meta["confounded"] == ["risk_adjusted"]
    assert meta["flat_rate_pct"] is not None


def test_the_analyzer_reads_the_stamp_rather_than_a_module_constant():
    """
    CONFOUNDED_CATEGORIES was a module constant, so the caveat described the
    machine running the analysis instead of the run that produced the panel.
    Analysing a flat-rate panel on a machine that has rate history would have
    silently dropped the caveat.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "analyze_backtest.py"
    text = src.read_text(encoding="utf-8")
    assert "CONFOUNDED_CATEGORIES" not in text, (
        "the analyzer still imports a module-level constant; it must read the "
        "panel's own stamp via read_run_meta"
    )
    assert "read_run_meta" in text
    ast.parse(text)
