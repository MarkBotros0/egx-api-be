"""
The 20-year annual fundamentals archive, and the look-ahead guard on it.

The guard is the point of this module. Everything else here is plumbing; the
`first_usable_date` filter is what stands between a fundamental backtest and a
result that can be made to say anything. scripts/backtest.py currently refuses
to score fundamentals at all for exactly this reason, and it will only be safe
to lift that refusal while these tests pass.

Pure-function tests plus a stub db, matching the convention in tests/ — there is
no Postgres fixture, which is why the maths and the queries are separable.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.core.fundamentals_annual import (
    MIN_EXPECTED_SYMBOLS,
    MIN_FISCAL_YEAR,
    REPORTING_LAG_DAYS,
    WRITE_BATCH,
    fetch_annual_rows,
    first_usable_date,
    get_annual_asof,
    refresh_annual_fundamentals,
)


def _row(symbol, years, eps=None, **arrays):
    """One scanner row: `d` positionally aligned to TV_HISTORY_COLUMNS."""
    n = len(years)
    return {"d": [
        symbol,
        years,
        eps if eps is not None else [1.0] * n,
        arrays.get("dps", [0.5] * n),
        arrays.get("net_income", [100.0] * n),
        arrays.get("revenue", [500.0] * n),
        arrays.get("total_assets", [900.0] * n),
        arrays.get("gross_profit", [None] * n),
        arrays.get("total_debt", [50.0] * n),
    ]}


# ---------------------------------------------------------------------------
# The look-ahead guard
# ---------------------------------------------------------------------------

def test_a_fiscal_year_is_not_knowable_on_its_last_day():
    """
    The single most important property here. FY2024's figures were not
    available on 2024-12-31; treating them as if they were lets any factor
    appear to work.
    """
    usable = first_usable_date(2024)
    assert usable > "2024-12-31"
    expected = (date(2024, 12, 31) + timedelta(days=REPORTING_LAG_DAYS)).isoformat()
    assert usable == expected


def test_the_reporting_lag_is_at_least_the_regulatory_deadline():
    """EGX requires annual financials within 90 days; the lag must not be tighter."""
    assert REPORTING_LAG_DAYS >= 90


def test_get_annual_asof_refuses_rows_that_were_not_yet_published():
    """
    The read path is the enforcement point, because a backtest that queries the
    table directly would silently bypass every guard in this module.
    """
    stored = [
        # (symbol, fiscal_year, eps, dps, ni, rev, assets, gp, debt, usable)
        ("COMI", 2024, 14.7, 2.3, 1.0, 2.0, 3.0, None, 4.0, "2025-04-30"),
        ("COMI", 2023, 7.7, 0.5, 1.0, 2.0, 3.0, None, 4.0, "2024-04-29"),
    ]

    class _DB:
        def __init__(self, as_of):
            self.as_of = as_of

        def execute(self, sql, params):
            assert "first_usable_date <= %s" in sql, (
                "the as-of filter was removed from the query — this is the "
                "look-ahead guard"
            )
            cutoff = params[0]
            rows = [r for r in stored if r[9] <= cutoff]
            # DISTINCT ON (symbol) ... ORDER BY fiscal_year DESC
            best = {}
            for r in sorted(rows, key=lambda x: -x[1]):
                best.setdefault(r[0], r)
            self._rows = list(best.values())
            return self

        def fetchall(self):
            return self._rows

    # Mid-2024: FY2023 is knowable, FY2024 is not.
    got = get_annual_asof(_DB("2024-07-01"), "2024-07-01")
    assert got["COMI"]["fiscal_year"] == 2023

    # Mid-2025: FY2024 has been published.
    got = get_annual_asof(_DB("2025-07-01"), "2025-07-01")
    assert got["COMI"]["fiscal_year"] == 2024


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_pre_2012_years_are_refused_at_ingest():
    """
    The rows exist in the feed but the data does not: usable diluted EPS by
    fiscal year falls from 225 (2018) and 160 (2012) to 63 (2011), 23 (2007)
    and 15 (2006). Including them would weight a cross-sectional test toward
    whichever handful of large caps happened to report that far back.
    """
    rows = fetch_annual_rows([_row("COMI", [2013, 2012, 2011, 2006])])
    years = {r["fiscal_year"] for r in rows}
    assert years == {2013, 2012}
    assert min(years) >= MIN_FISCAL_YEAR


def test_a_year_with_no_figures_at_all_is_dropped():
    """
    An all-null record would make a coverage count look better than it is —
    the same failure `refresh_pe_data` guards against with its skip-empty rule.
    """
    rows = fetch_annual_rows([_row(
        "AAAA", [2020, 2019],
        eps=[None, 2.0], dps=[None, 1.0], net_income=[None, 1.0],
        revenue=[None, 1.0], total_assets=[None, 1.0],
        gross_profit=[None, None], total_debt=[None, 1.0],
    )])
    assert {r["fiscal_year"] for r in rows} == {2019}


def test_arrays_stay_aligned_to_their_fiscal_years():
    """
    Every array is indexed by POSITION against fiscal_period_fy_h. An off-by-one
    here would silently attach 2024's earnings to 2023 and be invisible in any
    aggregate.
    """
    rows = fetch_annual_rows([_row("BBBB", [2024, 2023, 2022],
                                   eps=[3.0, 2.0, 1.0])])
    by_year = {r["fiscal_year"]: r["eps_diluted"] for r in rows}
    assert by_year == {2024: 3.0, 2023: 2.0, 2022: 1.0}


def test_a_short_array_does_not_shift_later_years():
    """A ragged array must yield None, never the next year's value."""
    rows = fetch_annual_rows([_row("CCCC", [2024, 2023], eps=[5.0])])
    by_year = {r["fiscal_year"]: r["eps_diluted"] for r in rows}
    assert by_year[2024] == 5.0
    assert by_year[2023] is None


def test_a_truncated_response_is_refused_without_writing():
    """
    Mirrors refresh_pe_data: a partial write that updates a third of the
    universe is worse than no write, because nothing distinguishes the two
    afterwards.
    """
    class _DB:
        def __init__(self):
            self.writes = 0

        def execute(self, *_a, **_k):
            self.writes += 1
            return self

    db = _DB()
    thin = fetch_annual_rows([_row(f"S{i:03d}", [2024, 2023]) for i in range(10)])
    result = refresh_annual_fundamentals(db, thin)
    assert result["success"] is False
    assert db.writes == 0, "a truncated response was partially written"
    assert str(MIN_EXPECTED_SYMBOLS) in result["error"]


class _CountingDB:
    """Records every statement so a test can count ROUND TRIPS, not just rows."""

    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None, *_a, **_k):
        self.statements.append((sql, params))
        return self

    @property
    def writes(self):
        return len(self.statements)


def test_a_full_response_writes_every_record():
    db = _CountingDB()
    rows = fetch_annual_rows(
        [_row(f"S{i:03d}", [2024, 2023]) for i in range(MIN_EXPECTED_SYMBOLS + 5)]
    )
    result = refresh_annual_fundamentals(db, rows)
    assert result["success"] is True
    assert result["written"] == len(rows)


def test_the_archive_is_written_in_BATCHES_not_one_row_at_a_time():
    """
    The live archive is ~2,555 records and this runs inside a request with a
    30-second ceiling. One statement per row is the shape that cost the FX
    backfill two of its three attempts — it either crawls or the pooler cuts it
    off partway. Reverting to a per-row loop must fail here, not in production
    at 4am.
    """
    db = _CountingDB()
    rows = fetch_annual_rows(
        [_row(f"S{i:03d}", [2024, 2023]) for i in range(MIN_EXPECTED_SYMBOLS + 5)]
    )
    result = refresh_annual_fundamentals(db, rows)

    assert result["written"] == len(rows), "every record must still be written"
    expected_batches = (len(rows) + WRITE_BATCH - 1) // WRITE_BATCH
    assert db.writes == expected_batches, (
        f"{len(rows)} rows took {db.writes} round trips; batched at "
        f"{WRITE_BATCH} it should take {expected_batches}"
    )
    assert db.writes < len(rows) / 10, "this is not meaningfully batched"


def test_a_repeated_symbol_year_cannot_abort_its_batch():
    """
    Postgres refuses an ON CONFLICT DO UPDATE that touches the same row twice in
    one statement, so a feed repeating a (symbol, fiscal_year) would take the
    whole batch down with it — a failure per-row upserts never had to care
    about. Same lesson as macro_series.upsert_many.
    """
    db = _CountingDB()
    rows = fetch_annual_rows(
        [_row(f"S{i:03d}", [2024, 2023]) for i in range(MIN_EXPECTED_SYMBOLS + 5)]
    )
    duplicated = rows + rows[:5]          # five records arrive twice
    result = refresh_annual_fundamentals(db, duplicated)

    assert result["success"] is True
    assert result["written"] == len(rows), "duplicates must collapse, not double"

    keys = []
    for sql, params in db.statements:
        # 11 columns per record, positionally: symbol, fiscal_year, ...
        for i in range(0, len(params), 11):
            keys.append((params[i], params[i + 1]))
    assert len(keys) == len(set(keys)), "a (symbol, fiscal_year) repeated in one batch"


@pytest.mark.parametrize("year", [2012, 2018, 2024])
def test_usable_dates_are_ordered_with_their_years(year):
    assert first_usable_date(year) < first_usable_date(year + 1)
