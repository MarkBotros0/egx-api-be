"""
Dated Egyptian macro series, so a historical date can be graded against the
numbers that were actually knowable then.

WHY THIS EXISTS
---------------
`settings.risk_free_rate` is one scalar. It is the Sharpe hurdle, the Sortino
hurdle, the whole input to `score_risk_adjusted`, and the bar realized trades
are graded against — and until 2026-09-02 it had been 25% for years while the
CBE cut to 19%. A single number cannot be right for both 2019 and today, so
`scripts/backtest.py` runs the rate flat and explicitly withholds the
Risk-Adjusted verdict because of it.

The same gap makes the app's return figures misleading in a second, larger way:
EGX30 went 8.25x in EGP over twenty years and 0.94x in USD. Nothing on screen
says so.

Both need history, and both are reachable through infrastructure already here.

THE RELEASE LAG IS THE POINT, NOT THE VALUE
-------------------------------------------
A macro bar's timestamp is its REFERENCE PERIOD, not its publication date.
August's inflation is not knowable in August. Every row therefore carries
`released_at`, and `get_macro_at` filters on it — the same discipline
`fundamentals_annual.first_usable_date` enforces, and for the same reason: using
a figure before it existed lets any macro-conditioned result be manufactured.

An FX rate is different and is deliberately lagged zero: a price is knowable the
day it prints.

WHERE THE WORK HAPPENS
----------------------
Nightly, ONE keyless POST to the ECONOMICS scanner returns every series' current
value — cheap enough to ride along on the existing cron slot. History is a
different matter: pulling 300 monthly bars per series through the vendored
client takes seconds each and would blow the 30-second serverless budget, so
backfill is an OFFLINE script (`scripts/backfill_macro.py`) run once.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.core.tradingview import DEFAULT_TIMEOUT_SECONDS, HEADERS

ECONOMICS_SCAN_URL = "https://scanner.tradingview.com/economics2/scan"

# code -> (label, TradingView exchange, release lag in days, unit)
#
# The lag is how long after the reference period the figure becomes public.
# Sources: CPI and inflation land around the 10th of the following month,
# reserves around the 7th, money supply roughly two months back. They are
# deliberately generous — being late costs a backtest almost nothing, being
# early is look-ahead that invalidates it.
SERIES = {
    "EGINTR": ("CBE policy rate", "ECONOMICS", 0, "percent"),
    "EGIRYY": ("Inflation rate YoY", "ECONOMICS", 10, "percent"),
    "EGCPI": ("Consumer price index", "ECONOMICS", 10, "index"),
    "EGFER": ("FX reserves", "ECONOMICS", 7, "usd"),
    "EGM2": ("Money supply M2", "ECONOMICS", 60, "egp"),
    "EGUR": ("Unemployment rate", "ECONOMICS", 45, "percent"),
    # A price, not a release: knowable the day it prints, hence lag 0. Same
    # symbol the macro card already uses, so the two cannot disagree.
    "USDEGP": ("USD/EGP", "FX_IDC", 0, "rate"),
}

# The policy rate is a stand-in for a risk-free rate. There is no free
# machine-readable Egyptian T-bill auction series — cbe.org.eg rejects
# automated requests — so this is the honest ceiling, and the gap must be
# stated wherever a Sharpe ratio is explained rather than papered over.
RISK_FREE_SERIES = "EGINTR"
FX_SERIES = "USDEGP"


def released_at(period: str, lag_days: int) -> str:
    """When a figure for `period` (ISO date) actually became public."""
    return (date.fromisoformat(period[:10])
            + timedelta(days=lag_days)).isoformat()


def fetch_current() -> list:
    """
    One keyless POST for every ECONOMICS series' latest value.

    Returns [{series_code, value}]. Raises on HTTP/JSON error — the caller owns
    its failure policy, matching `tradingview.scan`.
    """
    codes = [c for c, (_, ex, _, _) in SERIES.items() if ex == "ECONOMICS"]
    body = json.dumps({
        "columns": ["name", "close"],
        "filter": [{"left": "name", "operation": "in_range", "right": codes}],
        "range": [0, len(codes) + 5],
    }).encode("utf-8")
    req = urllib.request.Request(ECONOMICS_SCAN_URL, data=body, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read())

    out = []
    for row in (payload.get("data") or []):
        d = row.get("d") or []
        if len(d) < 2 or d[0] is None or d[1] is None:
            continue
        try:
            out.append({"series_code": str(d[0]).upper(), "value": float(d[1])})
        except (TypeError, ValueError):
            continue
    return out


def upsert(db, series_code: str, period: str, value: float,
           source: str = "tradingview") -> None:
    """One dated observation. Idempotent on (series_code, observed_period)."""
    meta = SERIES.get(series_code)
    lag = meta[2] if meta else 0
    db.execute(
        """
        INSERT INTO macro_series
            (series_code, observed_period, value, released_at, source, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (series_code, observed_period) DO UPDATE SET
            value       = EXCLUDED.value,
            released_at = EXCLUDED.released_at,
            source      = EXCLUDED.source,
            updated_at  = EXCLUDED.updated_at
        """,
        (series_code, period[:10], value, released_at(period, lag), source,
         datetime.now(timezone.utc).isoformat()),
    )


def upsert_many(db, series_code: str, rows, source: str = "tradingview") -> int:
    """
    Many dated observations for ONE series, in a single round trip.

    Same semantics as `upsert` — idempotent on (series_code, observed_period),
    `released_at` derived from the series' publication lag — expressed as one
    multi-row INSERT instead of one statement per row.

    THIS IS WHY THE FX BACKFILL FINISHES. USD/EGP is 5,000 daily bars, and at
    one statement per row over a pooled Neon connection the run either crawled
    for many minutes or was cut off partway: the first attempt exited having
    written two thirds of the series, the second lost 3,689 rows when the pooler
    closed the connection mid-transaction. Batched, the same 5,000 rows land in
    a handful of round trips, which is short enough that neither failure mode
    has room to happen.

    Callers still batch (see `scripts/backfill_macro.WRITE_BATCH`) so that one
    bad batch cannot cost the whole series. Returns the number of rows sent.
    """
    # Deduplicate WITHIN the batch, keeping the last value for a period.
    # Postgres refuses "ON CONFLICT DO UPDATE" that would touch the same row
    # twice in one statement, so a feed that repeats a date would abort the
    # whole batch rather than one row. Per-row upserts never had to care.
    deduped: dict = {}
    for period, value in rows:
        deduped[str(period)[:10]] = value
    rows = list(deduped.items())
    if not rows:
        return 0
    meta = SERIES.get(series_code)
    lag = meta[2] if meta else 0
    now = datetime.now(timezone.utc).isoformat()

    params: list = []
    for period, value in rows:
        period = str(period)[:10]
        params.extend([series_code, period, float(value),
                       released_at(period, lag), source, now])
    values_sql = ", ".join(["(%s, %s, %s, %s, %s, %s)"] * len(rows))
    db.execute(
        f"""
        INSERT INTO macro_series
            (series_code, observed_period, value, released_at, source, updated_at)
        VALUES {values_sql}
        ON CONFLICT (series_code, observed_period) DO UPDATE SET
            value       = EXCLUDED.value,
            released_at = EXCLUDED.released_at,
            source      = EXCLUDED.source,
            updated_at  = EXCLUDED.updated_at
        """,
        tuple(params),
    )
    return len(rows)


def refresh_current(db) -> dict:
    """
    Nightly: append today's reading for each ECONOMICS series.

    The reference period for a "current" reading is taken as today. That is
    imprecise for a monthly statistic — August's inflation is stamped with the
    date we observed it, not with August — but it is honest in the only way that
    matters here: the row cannot be read before the date we learned it, so no
    look-ahead is introduced. The offline backfill writes true reference periods.
    """
    try:
        rows = fetch_current()
    except Exception as e:
        return {"success": False, "written": 0,
                "error": f"{type(e).__name__}: {e}"}
    if not rows:
        return {"success": False, "written": 0, "error": "no series returned"}

    today = date.today().isoformat()
    written = 0
    for r in rows:
        try:
            upsert(db, r["series_code"], today, r["value"], source="scanner_current")
            written += 1
        except Exception:
            continue
    return {"success": True, "written": written,
            "series": sorted({r["series_code"] for r in rows})}


def get_macro_at(db, series_code: str, as_of: str) -> Optional[float]:
    """
    The latest value of `series_code` that had been PUBLISHED by `as_of`.

    Filters on `released_at`, never on `observed_period`. That distinction is
    the whole point of this module: a bar's timestamp is its reference period,
    and using August's inflation in August is look-ahead.
    """
    try:
        row = db.execute(
            "SELECT value FROM macro_series "
            "WHERE series_code = %s AND released_at <= %s "
            "ORDER BY observed_period DESC LIMIT 1",
            (series_code.upper(), as_of[:10]),
        ).fetchone()
    except Exception:
        return None
    return float(row[0]) if row and row[0] is not None else None


def get_risk_free_at(db, as_of: str, default: Optional[float] = None) -> Optional[float]:
    """
    The policy rate in force on `as_of`, or `default` when history is missing.

    This is what lets scripts/backtest.py stop running the rate flat and finally
    issue the Risk-Adjusted verdict it currently withholds.
    """
    value = get_macro_at(db, RISK_FREE_SERIES, as_of)
    return value if value is not None else default


def get_fx_at(db, as_of: str) -> Optional[float]:
    """USD/EGP on `as_of`. A price, so there is no publication lag to respect."""
    return get_macro_at(db, FX_SERIES, as_of)
