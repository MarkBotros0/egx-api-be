"""
Scheduled maintenance jobs, driven by an EXTERNAL scheduler (cron-job.org).

WHY THESE ARE CHUNKED
---------------------
An external scheduler removes the limit on how OFTEN a job can run. It does not
remove the limit on how LONG one request may take: this backend runs on
Vercel's Python runtime with a 30-second ceiling, and scoring or measuring the
whole EGX universe was clocked at over 400 seconds cold, because every symbol
pulls 400 bars through a client that retries hard on socket timeouts.

So each call here processes a SLICE and returns a cursor. Point the scheduler
at the endpoint every few minutes and the universe completes in under an hour,
with every individual request finishing well inside the timeout. A stalled or
failed call costs one chunk rather than the whole run, which is strictly better
than a single all-or-nothing nightly job.

STATE LIVES IN POSTGRES, NEVER IN THE PROCESS
---------------------------------------------
Serverless containers do not persist between invocations, so the cursor is
passed by the caller and the results are upserted per symbol. There is
deliberately no "finalize" step: cross-sectional percentiles are computed at
READ time by `risk_grade.grade_universe`, which means a partly-refreshed table
is still coherent instead of being half-ranked against yesterday.

NEVER WIPE ON FAILURE
---------------------
Matching `refresh_pe_data`: a partial refresh that updates 50 symbols and
silently leaves 150 stale is worse than everything being stale, because nothing
on screen distinguishes the two. Rows are only ever upserted, never cleared,
and each carries its own `measured_at` so the read path can report the real
freshness of its thinnest corner.

AUTH
----
This route is in `PUBLIC_ENDPOINTS` because an external scheduler carries no
user token, and it is guarded by the `CRON_SECRET` env var exactly as
`/api/pe/refresh` is guarded by `PE_REFRESH_SECRET`. If the env var is unset
the guard is skipped, which is what makes local development workable — set it
in production.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from app.core.cache import get as cache_get, make_key, set as cache_set
from app.core.constants import INTERNAL_BARS_MIN
from app.core.db import get_db
from app.core.risk_grade import is_tradeable, measure

router = APIRouter()

# Symbols per call. Was 20, which timed out in production: the vendored client
# wraps every fetch in @retry(Exception, tries=20, delay=0.5) around a call made
# with timeout=-1, so ONE symbol the feed has no data for burns ~10 seconds
# before giving up. A chunk with four dead names exceeded the budget on its own.
DEFAULT_CHUNK = 4
MAX_CHUNK = 60

# Hard wall-clock budget for the whole call, well inside Vercel's 30s and inside
# an external scheduler's own timeout. When it expires the handler STOPS and
# returns what it has.
#
# Stopping early is safe here precisely because selection is stalest-first:
# whatever did not get measured is still the stalest thing in the table, so the
# next call picks it up with no cursor to maintain. This is the property that
# design was chosen for, and this is the case that needed it.
DEADLINE_SECONDS = 15.0


def _require_secret(supplied: Optional[str]) -> None:
    expected = os.environ.get("CRON_SECRET")
    if expected and supplied != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


# At or above this many consecutive refusals a symbol is demoted: still retried,
# but only once everything the feed actually serves is fresh. Low enough that a
# genuinely dead name stops eating the budget within one pass, high enough to
# ride out a transient outage.
FAILURE_DEMOTION_THRESHOLD = 3


def select_stalest(universe: list, measured: dict, limit: int,
                   failures: Optional[dict] = None) -> list:
    """
    The `limit` symbols whose measurement is oldest, never-measured first.

    This replaced an auto-advancing cursor, and the reason is worth keeping:
    a cursor has to advance past a chunk that FAILED, or one permanently broken
    symbol pins it and starves the rest of the universe. Advancing means the
    failed chunk is skipped until the next full pass — on a daily job, a whole
    day.

    Choosing the stalest instead is self-healing by construction. A symbol that
    fails stays stale, so the very next call picks it up again, and no state has
    to be stored or kept in sync. It also makes the SCHEDULE forgiving: any
    number of calls per day converges, and firing more often than strictly
    necessary just keeps everything fresher rather than wasting a pass.

    `measured` is {symbol: measured_at ISO string}. A symbol absent from it has
    never been measured and sorts first — the empty string precedes any
    timestamp. Ties break on symbol so the order is deterministic.

    `failures` demotes symbols the feed keeps refusing. Measured: 84 of the 166
    symbols in the ticker file have NEVER returned data from tvDatafeed, and the
    failures are PERSISTENT rather than transient — the identical set fails on
    every pass. Each refusal costs about six seconds, so without demotion half
    the budget goes to symbols that will never work and the healthy half never
    finishes.

    They are demoted, NOT excluded: a symbol that starts working resets to zero
    on its first success, and the ordering still reaches them once everything
    else is fresh.
    """
    failures = failures or {}

    def rank(sym):
        demoted = 1 if failures.get(sym, 0) >= FAILURE_DEMOTION_THRESHOLD else 0
        return (demoted, measured.get(sym) or "", sym)

    return sorted(universe, key=rank)[:limit]


def _snapshot_state(db) -> tuple:
    """({symbol: measured_at}, {symbol: consecutive_failures}) in one query."""
    rows = db.execute(
        "SELECT symbol, measured_at, COALESCE(consecutive_failures, 0) "
        "FROM risk_snapshot"
    ).fetchall()
    return ({r[0]: r[1] for r in rows}, {r[0]: r[2] for r in rows})


def is_isin(symbol: str) -> bool:
    """
    Bonds and structured instruments, whose "symbol" is really an ISIN
    (EGS30AJ1C016, EGS48271C018-EGP). They are not common stocks and would
    pollute a cross-sectional ranking.

    The test is LENGTH, not the "EGS" prefix. An Egyptian ISIN is 12 characters
    (EG + 10), sometimes with a -EGP suffix, while **EGSA is a real
    four-character EGX ticker** that a prefix rule silently discards.
    """
    s = symbol.upper()
    return len(s) >= 12 and s.startswith("EG")


def plan_chunk(universe: list, cursor: int, limit: int) -> dict:
    """
    Which symbols this call handles, and where the caller goes next.

    Pure, so the cursor arithmetic the scheduler depends on is testable without
    a database or a network — tests/ has no Postgres fixture by design.

    `cursor` wraps to 0 once the universe is exhausted, so a scheduler that
    simply replays the returned cursor loops forever without special-casing the
    end. A cursor past the end yields an empty slice rather than an error: the
    universe shrinks when a stock delists, and a scheduler holding a stale
    cursor should quietly restart, not start alerting.
    """
    total = len(universe)
    start = cursor if 0 <= cursor < total else 0
    chunk = universe[start:start + limit]
    next_cursor = start + len(chunk)
    remaining = max(0, total - next_cursor)
    return {
        "symbols": chunk,
        "cursor": next_cursor if remaining else 0,
        "remaining": remaining,
        "universe": total,
    }


def _universe() -> list:
    """
    Every EGX symbol worth measuring, from the static ticker file.

    Reads `data/egx_tickers.json` through index_membership rather than through
    `tickers._load_tickers()`, which can fire a 10-second TradingView POST on a
    cold container — the same trap documented for the scoring hot path.
    """
    from app.core.index_membership import _load

    return [s for s in sorted(_load().keys()) if not is_isin(s)]


def _benchmark_close():
    """EGX30 closes for beta, through the shared cache key every path uses."""
    try:
        from app.vendor.egxpy import get_OHLCV_data
        ck = make_key("egx30", "EGX", "Daily", INTERNAL_BARS_MIN)
        df = cache_get(ck)
        if df is None:
            raw = get_OHLCV_data("EGX30", "EGX", "Daily", INTERNAL_BARS_MIN)
            if raw is not None and not raw.empty:
                raw.columns = [c.lower() for c in raw.columns]
                cache_set(ck, raw)
                df = raw
        if df is not None and "close" in df.columns:
            return df["close"]
    except Exception:
        pass
    return None


@router.post("/api/cron/risk_snapshot")
def risk_snapshot(
    cursor: Optional[int] = Query(default=None, ge=0),
    limit: int = Query(DEFAULT_CHUNK, ge=1, le=MAX_CHUNK),
    x_cron_secret: Optional[str] = Header(default=None),
):
    """
    Measure one slice of the universe and upsert it.

    **Omit `cursor` and the endpoint picks the stalest symbols itself.** That is
    the intended production mode: an external scheduler that cannot read a
    response body — as cron-job.org cannot — simply fires one fixed URL on an
    interval, and the universe converges. Nothing has to be stored, nothing can
    desynchronise, and a chunk that fails is retried on the very next call
    because it stays stale.

    Passing `cursor` explicitly walks the universe in fixed slices instead,
    which is useful for a controlled manual sweep. It stores no state either.
    """
    _require_secret(x_cron_secret)

    from app.vendor.egxpy import get_OHLCV_data

    universe = _universe()
    if not universe:
        raise HTTPException(status_code=500, detail="No universe available")

    db = get_db()
    measured_at_map, failure_map = _snapshot_state(db)
    if cursor is None:
        slice_ = select_stalest(universe, measured_at_map, limit, failure_map)
        plan = {"cursor": None, "remaining": None, "universe": len(universe)}
    else:
        plan = plan_chunk(universe, cursor, limit)
        slice_ = plan["symbols"]

    if not slice_:
        return {"processed": 0, "written": 0, "failed": [],
                "cursor": 0, "remaining": 0, "universe": len(universe),
                "note": "nothing to measure"}

    benchmark = _benchmark_close()
    now = datetime.now(timezone.utc).isoformat()

    def _upsert(symbol: str, stats: Optional[dict], tradeable: bool,
                ok: bool) -> None:
        """
        Record the attempt, ALWAYS — including when there was nothing to
        measure.

        This is load-bearing under stalest-first selection. A symbol with no row
        is maximally stale, so if a failure wrote nothing it would be re-picked
        on every single call. The EGX has ~34 effectively dead names and plenty
        of short-history ones; left unwritten they would fill a 20-slot batch
        forever and the rest of the universe would never refresh again. Verified
        in simulation before this guard existed.

        A null sigma with tradeable=False is also the honest record: we looked,
        and there is nothing usable here. `grade_universe` already ignores null
        sigmas and `/api/risk` ranks only tradeable rows, so such a symbol
        appears with a raw measurement and no band rather than a fabricated one.
        """
        db.execute(
            """
            INSERT INTO risk_snapshot
                (symbol, measured_at, sigma_63_ann_pct, sigma_ewma_ann_pct,
                 beta, turnover_egp, traded_share, last_price, tradeable,
                 above_sma200, rsi_14, consecutive_failures)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                measured_at        = EXCLUDED.measured_at,
                sigma_63_ann_pct   = EXCLUDED.sigma_63_ann_pct,
                sigma_ewma_ann_pct = EXCLUDED.sigma_ewma_ann_pct,
                beta               = EXCLUDED.beta,
                turnover_egp       = EXCLUDED.turnover_egp,
                traded_share       = EXCLUDED.traded_share,
                last_price         = EXCLUDED.last_price,
                tradeable          = EXCLUDED.tradeable,
                above_sma200       = EXCLUDED.above_sma200,
                rsi_14             = EXCLUDED.rsi_14,
                -- Reset on success, accumulate on refusal. A symbol that
                -- starts working again un-demotes itself on the first good
                -- fetch, which is why this is a counter and not a blocklist.
                consecutive_failures = CASE
                    WHEN %s THEN 0
                    ELSE COALESCE(risk_snapshot.consecutive_failures, 0) + 1
                END
            """,
            (symbol, now,
             (stats or {}).get("sigma_63_ann_pct"),
             (stats or {}).get("sigma_ewma_ann_pct"),
             (stats or {}).get("beta"),
             (stats or {}).get("turnover_egp"),
             (stats or {}).get("traded_share"),
             (stats or {}).get("last_price"),
             tradeable,
             (stats or {}).get("above_sma200"),
             (stats or {}).get("rsi_14"),
             0 if ok else 1,
             ok),
        )

    started = time.monotonic()
    written, skipped, failed, timed_out = 0, 0, [], False
    for symbol in slice_:
        if time.monotonic() - started > DEADLINE_SECONDS:
            # Out of budget. Everything unprocessed stays stale and is picked
            # up first next call, so nothing is lost by stopping here.
            timed_out = True
            break
        try:
            df = get_OHLCV_data(symbol, "EGX", "Daily", INTERNAL_BARS_MIN)
        except Exception:
            df = None

        usable = df is not None and not df.empty
        if usable:
            df.columns = [str(c).lower() for c in df.columns]
            usable = "close" in df.columns and "volume" in df.columns

        if not usable:
            failed.append(symbol)
            _upsert(symbol, None, False, ok=False)
            continue

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        stats = measure(close, volume, benchmark_close=benchmark)
        if stats is None:
            skipped += 1
            _upsert(symbol, None, False, ok=False)
            continue

        _upsert(symbol, stats, is_tradeable(close, volume), ok=True)
        written += 1

    # How much of the universe is still carrying a measurement from before this
    # run started. This is the number to watch on the scheduler: it should fall
    # to 0 within a pass and stay there. If it never reaches 0, some symbols are
    # failing every time and `failed` names them.
    after, after_failures = _snapshot_state(db)   # one query, not one per symbol
    stale_remaining = sum(1 for s in universe if (after.get(s) or "") < now)
    # Symbols the feed keeps refusing. Expected to be large on this market:
    # roughly half the ticker file has never resolved. It is reported so a
    # persistently growing number is visible rather than mysterious.
    demoted = sum(1 for v in after_failures.values()
                  if v >= FAILURE_DEMOTION_THRESHOLD)

    elapsed = round(time.monotonic() - started, 1)
    return {
        "requested": len(slice_),
        "processed": written + skipped + len(failed),
        "written": written,
        "elapsed_seconds": elapsed,
        # Watch this: persistently true means the feed is slow enough that the
        # chunk should shrink, or the schedule should fire more often.
        "deadline_hit": timed_out,
        "skipped_insufficient_history": skipped,
        # Named, not just counted: a symbol that fails every night is a feed
        # problem, and a bare count hides which one.
        "failed": failed,
        "mode": "stalest-first" if cursor is None else "explicit-cursor",
        "cursor": plan["cursor"],
        "stale_remaining": stale_remaining,
        "demoted_symbols": demoted,
        "universe": plan["universe"],
        "measured_at": now,
    }
