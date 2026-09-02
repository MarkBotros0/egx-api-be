"""
Walk-forward backtest of the composite score.

WHAT THIS ANSWERS
-----------------
Does a higher composite score actually precede better forward returns on the
EGX? Nothing in this app has ever established that. Every correctness fix so far
proved the score is internally consistent and means what it says — a property
entirely independent of whether it predicts anything.

METHOD
------
Fetch a long daily history per symbol once, cached to disk. Walk forward monthly.
At each date score every eligible symbol using ONLY bars up to that date, then
measure forward returns at 21 / 63 / 126 trading days.

The primary metric is the INFORMATION COEFFICIENT: at each date, the Spearman
rank correlation between score and forward return ACROSS stocks; then the mean
and t-statistic of those per-date ICs. Each date contributes one observation, so
overlapping forward windows cannot inflate significance the way they would if
42,000 symbol-dates were treated as independent samples.

IC also happens to be the metric most robust to this dataset's three biggest
distortions — survivorship, EGP devaluations, and market-wide moves — because
all three shift the whole cross-section at once and therefore largely cancel in
a rank correlation, while they dominate absolute bucket returns.

WHAT IT DELIBERATELY DOES NOT TEST
----------------------------------
Fundamentals. `pe_data` holds a single snapshot of today, and
`fundamentals_history` only began collecting on 2026-08-25. Passing today's P/E
into a 2024 score would be look-ahead bias severe enough to manufacture any
result. Quality is therefore scored on its technical inputs only.

Risk-Adjusted USED to be withheld too, because the policy rate ran flat at one
current value across twenty years in which it ranged 8.25%-27.25%. `macro_series`
now supplies dated rates, so each scoring date is graded against the cash return
over its own trailing year and the verdict stands — but only when that history is
present. With no database the run falls back to flat and re-withholds; see
`confounded_categories`.

Run:  python -m scripts.backtest            (from egx-api-be)
      python -m scripts.backtest --quick    (small universe, for a smoke test)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.composite import (  # noqa: E402
    CATEGORY_ORDER,
    DEFAULT_WEIGHTS,
    classify_signal,
    compute_composite,
)
from app.core.constants import (  # noqa: E402
    DEFAULT_RISK_FREE_RATE_PCT,
    MACRO_TREND_DOWN_PCT,
    MACRO_TREND_UP_PCT,
)
from app.core.extras_builder import build_composite_extras  # noqa: E402
from app.core.returns import annualized_cash_rate_pct  # noqa: E402
from app.core.index_membership import get_index_membership  # noqa: E402
from app.core.indicators import compute_all  # noqa: E402

BENCHMARK = "EGX30"
BARS_TO_FETCH = 5000          # ~20 years; the practical server-side ceiling
MIN_HISTORY_BARS = 250        # a year of prior data before a stock is scorable
REBALANCE_EVERY = 21          # trading days between scoring dates (~monthly)
HORIZONS = (21, 63, 126)      # forward-return windows, in trading days

# The FALLBACK rate, used only when no dated history is available.
#
# This used to be the only rate: Egypt's policy rate ran from roughly 8% (2020)
# to 27.25% (2024) while the app stored one current value, so rather than
# hardcode half-remembered rates into a validation exercise the whole run went
# flat and WITHHELD Risk-Adjusted's verdict.
#
# `macro_series` now holds EGINTR monthly back to 2001-05, so each scoring date
# can be graded against the cash return over ITS OWN trailing year — which is
# what `score_risk_adjusted` compares against. The withholding is lifted only
# when that history is actually present; see `confounded_categories`.
RISK_FREE_RATE = float(DEFAULT_RISK_FREE_RATE_PCT)


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def confounded_categories(rate_steps) -> set:
    """
    Which category verdicts must NOT be read as evidence.

    The verdict may only be lifted by the data that justifies lifting it. A run
    on a machine with no macro history keeps withholding rather than quietly
    reporting a number it cannot support — the failure mode this whole script
    exists to avoid.
    """
    return set() if rate_steps else {"risk_adjusted"}


def run_meta_path(panel_path: str) -> str:
    return f"{panel_path}.meta.json"


def write_run_meta(panel_path: str, rate_steps) -> dict:
    """
    Stamp a panel with HOW it was scored, beside the panel itself.

    `analyze_backtest` needs to know whether Risk-Adjusted's row is evidence or
    a caveat, and that is a property of the RUN, not of the machine reading it.
    Deriving it locally would let an old flat-rate panel be analysed on a
    machine that happens to have rate history and silently lose the caveat.
    """
    meta = {
        "dated_risk_free": bool(rate_steps),
        "rate_steps": len(rate_steps or []),
        "flat_rate_pct": None if rate_steps else RISK_FREE_RATE,
        "confounded": sorted(confounded_categories(rate_steps)),
    }
    with open(run_meta_path(panel_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def read_run_meta(panel_path: str) -> dict:
    """
    How a panel was scored. CONSERVATIVE when unstamped.

    Every panel built before this existed is a flat-rate panel, so an absent
    stamp must read as flat — presenting a confounded number as evidence is the
    precise failure the withholding exists to prevent.
    """
    try:
        with open(run_meta_path(panel_path), encoding="utf-8") as f:
            meta = json.load(f)
        meta.setdefault("confounded", sorted(confounded_categories(
            [1] if meta.get("dated_risk_free") else [])))
        return meta
    except Exception:
        return {
            "dated_risk_free": False,
            "rate_steps": 0,
            "flat_rate_pct": RISK_FREE_RATE,
            "confounded": sorted(confounded_categories(None)),
        }


def rate_for_date(rate_steps, as_of, fallback: float = RISK_FREE_RATE) -> float:
    """
    The cash return over the trailing year ending `as_of`, as an annual rate.

    A TRAILING YEAR, not a point rate, because `score_risk_adjusted` compares a
    trailing one-year stock return against cash and the two must span the same
    window.

    NO LOOK-AHEAD BY CONSTRUCTION. Unlike the sales ledger — which scores an
    outcome that already happened — this simulates a DECISION at a past date, so
    a rate set afterwards must not reach it. Only steps inside [as_of - 1y,
    as_of] contribute, so a 2024 hike cannot touch a 2020 scoring date. (EGINTR
    carries a zero publication lag, so a rate in force is a rate that was known.)
    """
    if not rate_steps:
        return float(fallback)
    start = as_of - timedelta(days=365)
    got = annualized_cash_rate_pct(rate_steps, start, as_of)
    return float(got) if got is not None else float(fallback)


def load_rate_steps() -> list:
    """
    Dated policy rates from `macro_series`, cached to disk after the first read.

    Returns [] when the database is unreachable, which keeps this script
    runnable offline — and keeps Risk-Adjusted withheld, which is the correct
    behaviour when the data behind the verdict is missing.
    """
    path = os.path.join(CACHE_DIR, "policy_rate.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        from app.core.db import get_db
        from app.core.macro_series import get_risk_free_steps
        steps = get_risk_free_steps(get_db())
    except Exception:
        return []
    if steps:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(steps, f)
    return steps

# The vendored client wraps every call in @retry(Exception, tries=20, delay=0.5,
# backoff=0) — a flat half-second retry twenty times, which hammers the endpoint
# rather than backing off. Throttle ourselves and treat "no data" as permanent.
FETCH_PAUSE_SECONDS = 0.4

# HARD ceiling per symbol, enforced by killing a child process.
#
# This is not belt-and-braces, it is required: the vendored client calls
# `get_hist(..., timeout=-1)`, i.e. NO timeout, so an unresponsive websocket
# blocks forever — and the 20-try retry decorator wrapped around it can re-enter
# that hang twenty times. Observed directly: a full-universe run wedged on one
# symbol and made zero progress for minutes while looping on "timed out,
# retrying in 0.5 seconds". A thread cannot be killed in Python, so the fetch
# runs in a child process that can be terminated.
FETCH_TIMEOUT_SECONDS = 45


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol.upper()}.pkl")


class CacheUnreadableError(RuntimeError):
    """
    A cache entry exists but could not be unpickled.

    Deliberately NOT the same outcome as a cached None, which means the feed
    genuinely had no data for that symbol. `load_history` used to collapse the
    two into a bare `except Exception: return None`, and on 2026-08-26 that bit:
    the pickles had been written by the project venv (pandas 3.0.2) and a run
    under the system Python (pandas 2.3.3) raised

        NotImplementedError: (<StringDtype(storage='python', na_value=nan)>, ...)

    on EVERY entry. Swallowed, that read as "the entire EGX has no history".
    The run happened to die on the benchmark and print "FATAL: no benchmark
    history"; had the benchmark been readable it would have built a panel
    missing most of the universe with nothing on screen saying so. A backtest
    that quietly runs on 10% of its data is worse than one that crashes.
    """


# The cache is version-coupled by construction — pickled DataFrames do not move
# between pandas versions — so record what wrote it.
#
# Limitation worth knowing: this is one stamp for a whole directory, written by
# whoever first fetched into it. Fetch new symbols under a different pandas and
# the directory becomes genuinely mixed while the stamp still names the original
# writer. The stamp is therefore advisory; `_read_cache` raising is the
# guarantee.

def _stamp_path() -> str:
    return os.path.join(CACHE_DIR, "VERSION")


def _running_versions() -> dict:
    return {"pandas": pd.__version__, "python": sys.version.split()[0]}


def write_cache_stamp() -> None:
    """
    Record the interpreter writing cache entries, if not already recorded.

    Called only where entries are actually written. Stamping on a plain read
    would claim this interpreter wrote entries it did not.
    """
    path = _stamp_path()
    if os.path.exists(path):
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_running_versions(), f)
    except OSError:
        pass          # a diagnostic, never a reason to fail a run


def read_cache_stamp() -> dict:
    """The recorded versions, or {} if unstamped or unparseable."""
    try:
        with open(_stamp_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Swallowing is right HERE and nowhere else in this section: the stamp
        # is a hint about data, not the data. Absent means unknown, and
        # `cache_stamp_warning` treats unknown as silent.
        return {}


def cache_stamp_warning() -> str | None:
    """
    One line naming an interpreter mismatch before a long run starts, or None.

    An unstamped cache is unknown, not mismatched — warning on it would train
    the reader to ignore the warning.
    """
    stamp = read_cache_stamp()
    if not stamp:
        return None
    now = _running_versions()
    diff = [k for k in ("pandas", "python") if stamp.get(k) and stamp[k] != now[k]]
    if not diff:
        return None
    moved = ", ".join(f"{k} {stamp[k]} -> {now[k]}" for k in diff)
    return (
        f"WARNING: {CACHE_DIR} was written by a different interpreter ({moved}).\n"
        f"         Pickled frames are not portable across pandas versions. If "
        f"reads fail, re-run\n"
        f"         with the venv that wrote the cache, or delete the cache and "
        f"re-fetch."
    )


def _unreadable_message(path: str, exc: BaseException) -> str:
    """
    Say what is actually known, and only offer the version theory when it is
    live. Under the SAME pandas that wrote the cache, an unreadable entry is a
    truncated or half-written file — sending the reader off to compare
    interpreters that already agree buries the one-file remedy.
    """
    stamp = read_cache_stamp()
    head = f"{path} exists but could not be unpickled: {type(exc).__name__}: {exc}"

    if stamp.get("pandas") == pd.__version__:
        return (
            f"{head}\n"
            f"  The cache was written by this same pandas ({pd.__version__}), so "
            f"this is a corrupt\n"
            f"  or half-written entry rather than a version mismatch. Delete "
            f"{path}\n"
            f"  and re-fetch that symbol."
        )

    wrote = (f"was written by pandas {stamp['pandas']}" if stamp.get("pandas")
             else "predates the VERSION stamp, so its writer is unknown")
    return (
        f"{head}\n"
        f"  The cache {wrote}; this process is running pandas {pd.__version__}.\n"
        f"  Pickled DataFrames are not portable across pandas versions. Re-run "
        f"with the venv\n"
        f"  that wrote the cache (./.venv/Scripts/python.exe -m scripts.backtest), "
        f"or delete\n"
        f"  {CACHE_DIR} and re-fetch."
    )


def _read_cache(path: str):
    """
    Unpickle one cache entry.

    Returns the cached object, which is legitimately None for a symbol the feed
    had no data for. A missing file is None too — a stalled fetch writes
    nothing, and absence is not corruption. Anything else raises: an entry we
    cannot read is a fact about this run, and must never be reported as a fact
    about the market.

    The except is broad on purpose. The failure that motivated this was
    NotImplementedError raised by pandas' own unpickling, not
    pickle.UnpicklingError, so narrowing to pickle's own errors would re-swallow
    exactly the incident.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise CacheUnreadableError(_unreadable_message(path, exc)) from exc


def cache_failure_report(failures: dict, total: int) -> str | None:
    """
    The end-of-run tally of unreadable entries, or None if every symbol loaded.

    Per-symbol errors scroll past during a run that takes half an hour; the
    block at the bottom is what actually gets read. It has to state the count
    against the universe size, or a panel built from a fraction of the universe
    reads as a complete one.
    """
    if not failures:
        return None
    shown = sorted(failures)[:10]
    more = len(failures) - len(shown)
    return (
        "\n" + "!" * 74 + "\n"
        f"PARTIAL PANEL — {len(failures)} of {total} symbols had an UNREADABLE "
        f"cache entry and\ncontributed no rows. They are missing from every "
        f"number above; do not read\nthis run as covering the universe.\n"
        f"  affected: {', '.join(shown)}"
        f"{f' (+{more} more)' if more else ''}\n\n"
        f"  {next(iter(failures.values()))}\n"
        + "!" * 74
    )


def _fetch_worker(symbols: list) -> None:
    """
    Child-process body: fetch a BATCH, writing each symbol as it completes.

    Batched because Windows spawns a fresh interpreter per process, and
    re-importing pandas/numpy/app.core cost more than the fetch itself —
    measured at ~37 s per symbol one-at-a-time versus ~6 s for the fetch alone.
    Writing incrementally is what makes a mid-batch kill safe: whatever landed
    stays cached and only the remainder is retried.
    """
    from app.vendor.egxpy import get_OHLCV_data

    for symbol in symbols:
        path = _cache_path(symbol)
        if os.path.exists(path):
            continue
        try:
            df = get_OHLCV_data(symbol, "EGX", "Daily", BARS_TO_FETCH)
        except Exception:
            df = None

        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = None

        with open(path, "wb") as f:
            pickle.dump(df, f)
        time.sleep(FETCH_PAUSE_SECONDS)


def fetch_batch(symbols: list) -> int:
    """
    Fetch a batch in one child process, killing it if it STOPS MAKING PROGRESS.

    A fixed per-batch budget would either cut short a healthy batch or waste
    minutes on a wedged one. Watching the cache directory instead means the
    child is killed only when it has genuinely stalled — which is the failure
    mode the vendored client produces, since it passes `timeout=-1` (no
    timeout) and retries a hung socket up to twenty times.

    Returns how many of the batch are now cached. Uncached symbols are simply
    retried by the next pass, so this is safe to call repeatedly.
    """
    import multiprocessing as mp

    todo = [s for s in symbols if not os.path.exists(_cache_path(s))]
    if not todo:
        return len(symbols)

    write_cache_stamp()
    proc = mp.Process(target=_fetch_worker, args=(todo,), daemon=True)
    proc.start()

    done = last_change = 0
    stall_started = time.time()
    while proc.is_alive():
        proc.join(2)
        done = sum(1 for s in todo if os.path.exists(_cache_path(s)))
        if done != last_change:
            last_change, stall_started = done, time.time()
        elif time.time() - stall_started > FETCH_TIMEOUT_SECONDS:
            proc.terminate()
            proc.join(5)
            break

    return sum(1 for s in symbols if os.path.exists(_cache_path(s)))


def load_history(symbol: str, refetch: bool = False):
    """
    Daily OHLCV for one symbol, from the disk cache.

    None means no data — either the feed had none for this symbol, or the fetch
    stalled and wrote nothing. An entry that exists but cannot be read raises
    CacheUnreadableError instead; see that class for what conflating the two
    cost. Callers that treat None as "skip this symbol" must NOT catch it.

    Read-only by design: workers must never fetch. main() warms the cache
    serially first, because parallel cold fetches would multiply the vendored
    client's retry storm across processes.
    """
    path = _cache_path(symbol)
    if not refetch and os.path.exists(path):
        return _read_cache(path)

    fetch_batch([symbol])
    return _read_cache(path)


def universe() -> list:
    """Symbols to test: the static ticker file plus whatever the feed lists."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    syms = set()
    try:
        with open(os.path.join(here, "data", "egx_tickers.json"), encoding="utf-8") as f:
            syms |= {t["symbol"].upper() for t in json.load(f)}
    except Exception:
        pass
    try:
        from app.core.tradingview import scan
        syms |= {(r.get("d") or [""])[0].strip().upper()
                 for r in scan(["name"]) if (r.get("d") or [""])[0]}
    except Exception:
        pass
    return sorted(s for s in syms if s and s != BENCHMARK)


# ---------------------------------------------------------------------------
# Historical macro regime
# ---------------------------------------------------------------------------

def macro_at(bench_close: pd.Series) -> dict | None:
    """
    Rebuild the EGX30 regime as `fetch_macro` would have seen it.

    fetch_macro itself reads the DB, the network AND the clock, and writing to
    macro_data would pollute the live cache — so it is reproduced here from the
    benchmark series instead. The production quirks are reproduced exactly, or
    this would be testing something the app does not do:
      - iloc[-min(20, n)] is 19 bar-INTERVALS back, not 20
      - `if (monthly_change and ...)` treats an exact 0.0 as falsy -> sideways
    Only "bearish" modulates anything; bullish and sideways are exact no-ops.
    """
    n = len(bench_close)
    if n < 5:
        return None
    current = float(bench_close.iloc[-1])
    month_ago = float(bench_close.iloc[-min(20, n)])
    change = ((current - month_ago) / month_ago * 100) if month_ago else None
    trend = ("bullish" if (change and change > MACRO_TREND_UP_PCT)
             else "bearish" if (change and change < MACRO_TREND_DOWN_PCT)
             else "sideways")
    return {"egx30": {"trend": trend}}


# ---------------------------------------------------------------------------
# Scoring one symbol across all dates
# ---------------------------------------------------------------------------

def rebalance_calendar(bench_close: pd.Series) -> list:
    """
    The COMMON scoring dates, taken from the benchmark's trading calendar.

    Every symbol must be scored on the same dates or the primary metric breaks:
    the Information Coefficient is a rank correlation computed ACROSS stocks on
    a given date. Scoring each symbol at offsets from its own first bar (the
    obvious implementation) produced 4,721 distinct dates for 221 symbols —
    a median of 6 stocks per date, far too thin a cross-section to correlate.
    """
    idx = bench_close.index
    usable = len(idx) - max(HORIZONS)
    return [idx[i] for i in range(MIN_HISTORY_BARS, usable, REBALANCE_EVERY)]


def score_symbol(symbol: str, bench_close: pd.Series, dates: list,
                 rate_steps: list | None = None) -> list:
    """
    Every (date, score, category scores, forward returns) row for one symbol,
    evaluated on the shared rebalance calendar.

    Runs in a worker process; returns plain data only.
    """
    df = load_history(symbol)
    if df is None or len(df) < MIN_HISTORY_BARS + max(HORIZONS):
        return []

    tier = get_index_membership(symbol)
    close = df["close"]
    out = []

    for as_of in dates:
        # Bars STRICTLY up to the rebalance date. searchsorted keeps this exact
        # even when the symbol halted on a day the index traded.
        i = int(df.index.searchsorted(as_of, side="right"))
        if i < MIN_HISTORY_BARS or i - 1 + max(HORIZONS) >= len(df):
            continue
        sl = df.iloc[:i]

        # Align the benchmark BY DATE. Slicing by position silently misaligns
        # whenever a stock halts on a day the index trades.
        bench = bench_close[bench_close.index <= as_of]
        if len(bench) < 60:
            continue

        try:
            indicators = compute_all(sl)
            built = build_composite_extras(
                sl, indicators,
                interval="Daily",
                egx30_close=bench,
                include_multi_timeframe=True,
                # The rate that actually prevailed over THIS date's
                # trailing year, not one flat number for twenty years.
                risk_free_rate_pct=rate_for_date(rate_steps, as_of),
                # Fundamentals are a today-only snapshot — passing them here
                # would be look-ahead. This is what fundamentals_history will
                # eventually make possible.
                pe_ratio=None, dividend_yield=None, loss_making=None,
                index_membership=tier,
            )
            result = compute_composite(
                indicators,
                extras=built["extras"],
                weights=DEFAULT_WEIGHTS,          # pinned, not read from the DB
                macro=macro_at(bench),
            )
        except Exception:
            continue

        row = {
            "symbol": symbol,
            "date": as_of,
            "score": result["score"],
            "signal": result["signal"],
            # As of the scoring date, for the liquid-only cut. A result that
            # only holds among names trading a few thousand shares a day is not
            # a result a retail investor could have acted on.
            "avg_volume": float(sl["volume"].tail(20).mean()),
        }
        for name in CATEGORY_ORDER:
            row[f"cat_{name}"] = result["categories"][name]["score"]

        p0 = float(close.iloc[i - 1])
        if p0 <= 0:
            continue
        for h in HORIZONS:
            pf = float(close.iloc[i - 1 + h])
            row[f"fwd_{h}"] = (pf / p0 - 1.0) * 100.0
        out.append(row)

    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def information_coefficient(panel: pd.DataFrame, score_col: str, horizon: int):
    """
    Mean per-date Spearman rank correlation between a score and forward return.

    Cross-sectional by construction: each date yields ONE observation, so
    overlapping forward windows cannot inflate the t-statistic.
    """
    fwd = f"fwd_{horizon}"
    per_date = []
    for _, grp in panel.groupby("date"):
        g = grp[[score_col, fwd]].dropna()
        # A rank correlation needs spread; a date where every stock scored the
        # same is not evidence either way.
        if len(g) < 10 or g[score_col].nunique() < 3:
            continue
        # Spearman computed as Pearson on ranks rather than via
        # pandas' method="spearman", which imports scipy. This keeps the
        # backtest runnable on the production venv without adding a dependency
        # that Vercel would then install for no reason. `.rank()` uses average
        # ranks for ties, which is the standard definition.
        ic = g[score_col].rank().corr(g[fwd].rank())
        if ic == ic:
            per_date.append(ic)

    if len(per_date) < 5:
        return {"ic": None, "n_dates": len(per_date), "t_stat": None, "ci95": None}

    arr = np.array(per_date, dtype=float)
    mean, sd, n = arr.mean(), arr.std(ddof=1), len(arr)
    se = sd / np.sqrt(n)
    return {
        "ic": round(float(mean), 4),
        "n_dates": n,
        "t_stat": round(float(mean / se), 2) if se > 0 else None,
        "ci95": (round(float(mean - 1.96 * se), 4), round(float(mean + 1.96 * se), 4)),
        "hit_rate": round(float((arr > 0).mean()), 3),
    }


def bucket_returns(panel: pd.DataFrame, horizon: int, by: str = "signal"):
    """Mean forward return per signal band or score decile."""
    fwd = f"fwd_{horizon}"
    if by == "signal":
        key = panel["signal"]
        order = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
    else:
        key = pd.qcut(panel["score"], 10, labels=False, duplicates="drop")
        order = sorted(k for k in key.dropna().unique())

    rows = []
    for k in order:
        sel = panel[key == k][fwd].dropna()
        if len(sel) == 0:
            continue
        rows.append({
            "bucket": k if by != "signal" else str(k),
            "n": len(sel),
            "mean_fwd_pct": round(float(sel.mean()), 2),
            "median_fwd_pct": round(float(sel.median()), 2),
            "win_rate": round(float((sel > 0).mean()), 3),
        })
    return rows


def sanity_checks(panel: pd.DataFrame) -> dict:
    """
    Prove the harness can detect "no signal" before any positive result is
    believed. A shuffled score must produce IC ~ 0; if it does not, the
    measured IC is an artefact of the plumbing rather than of the score.
    """
    rng = np.random.default_rng(12345)
    shuffled = panel.copy()
    shuffled["score"] = rng.permutation(shuffled["score"].values)
    placebo = information_coefficient(shuffled, "score", HORIZONS[0])
    return {
        "placebo_ic": placebo["ic"],
        "placebo_t": placebo["t_stat"],
        "passes": placebo["ic"] is not None and abs(placebo["ic"]) < 0.03,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small universe smoke test")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "panel.pkl"))
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    stamp_warning = cache_stamp_warning()
    if stamp_warning:
        print(stamp_warning)

    # Dated policy rates decide whether Risk-Adjusted's verdict can be read at
    # all, so this is resolved BEFORE any scoring and stated in the output. The
    # previous CONFOUNDED_CATEGORIES was a module constant nothing ever read —
    # the withholding existed only in a docstring.
    rate_steps = load_rate_steps()
    confounded = confounded_categories(rate_steps)
    if rate_steps:
        print(f"policy rate: {len(rate_steps)} dated steps, "
              f"{rate_steps[0][0]} .. {rate_steps[-1][0]} — each scoring date is "
              f"graded against its own trailing year")
    else:
        print(f"policy rate: NO dated history — running flat at "
              f"{RISK_FREE_RATE}%; WITHHOLDING {sorted(confounded)}")

    syms = universe()
    if args.quick:
        syms = syms[:25]
    print(f"universe: {len(syms)} symbols")

    print(f"fetching benchmark {BENCHMARK} ...")
    bench_df = load_history(BENCHMARK)
    if bench_df is None:
        print("FATAL: no benchmark data — relative strength and beta need it")
        return 1
    bench_close = bench_df["close"]
    print(f"  {len(bench_close)} bars, {bench_close.index[0].date()} .. {bench_close.index[-1].date()}")

    # Warm the disk cache serially: parallel cold fetches would multiply the
    # vendored client's flat-retry storm across processes.
    print("fetching histories (cached after first run) ...")
    t0 = time.time()
    missing = [s for s in syms if not os.path.exists(_cache_path(s))]
    BATCH = 20
    for i in range(0, len(missing), BATCH):
        chunk = missing[i:i + BATCH]
        fetch_batch(chunk)
        n = i + len(chunk)
        rate = (time.time() - t0) / max(n, 1)
        print(f"  {n}/{len(missing)}  ({time.time()-t0:.0f}s, {rate:.1f}s/sym, "
              f"~{rate*(len(missing)-n)/60:.0f}min left)", flush=True)

    still_missing = [s for s in syms if not os.path.exists(_cache_path(s))]
    print(f"  fetch done in {time.time()-t0:.0f}s; "
          f"{len(still_missing)} stalled and were not cached (re-run to retry)")

    dates = rebalance_calendar(bench_close)
    print(f"scoring across {args.workers} workers "
          f"on {len(dates)} shared rebalance dates "
          f"({dates[0].date()} .. {dates[-1].date()}) ...")
    t0 = time.time()
    rows = []
    # Tracked separately from ordinary per-symbol errors: an unreadable cache
    # means we never saw that symbol's data at all, which changes what the panel
    # is rather than what one symbol scored.
    unreadable: dict = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(score_symbol, s, bench_close, dates, rate_steps): s
                for s in syms}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rows.extend(fut.result())
            except CacheUnreadableError as e:
                unreadable[futs[fut]] = str(e)
                print(f"  {futs[fut]}: CACHE UNREADABLE\n{e}")
            except Exception as e:
                print(f"  {futs[fut]}: {type(e).__name__} {e}")
            if n % 50 == 0:
                print(f"  {n}/{len(syms)} symbols  ({time.time()-t0:.0f}s)")

    report = cache_failure_report(unreadable, len(syms))

    if not rows:
        print("FATAL: no scored rows produced")
        if report:
            print(report)
        return 1

    panel = pd.DataFrame(rows)
    panel.to_pickle(args.out)
    print(f"\npanel: {len(panel):,} symbol-dates, "
          f"{panel['symbol'].nunique()} symbols, {panel['date'].nunique()} dates")
    print(f"span: {panel['date'].min().date()} .. {panel['date'].max().date()}")
    write_run_meta(args.out, rate_steps)
    print(f"saved -> {args.out}")
    print(f"stamped -> {run_meta_path(args.out)}")

    if report:
        # Last thing printed, and a non-zero exit, so an incomplete panel is not
        # picked up by analyze_backtest as if it covered the universe.
        print(report)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
