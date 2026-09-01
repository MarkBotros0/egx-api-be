"""
GET /api/analysis — Fetch OHLCV data and compute all technical indicators for a stock.

Also supports ?mode=batch&symbols=A,B,C for lightweight composite-only batch scoring.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import CurrentUser, get_optional_user
from app.core.cache import get, set, make_key
from app.core.constants import (
    BATCH_DEADLINE_SECONDS,
    BATCH_MAX_SYMBOLS,
    BATCH_WORKERS,
    DEFAULT_RISK_FREE_RATE_PCT,
    DIVERGENCE_LOOKBACK_FULL,
    INTERNAL_BARS_MIN,
    USER_BARS_MAX,
    USER_BARS_MIN,
)
from app.core.db import get_db
from app.core.indicators import (
    compute_all, support_resistance, fibonacci_levels, ma_crossovers,
    compute_beta, daily_returns, sma, cumulative_returns,
)
from app.core.composite import (
    compute_composite, get_weights_from_db, weights_hash, DEFAULT_WEIGHTS,
)
from app.core.extras_builder import build_composite_extras
from app.core.index_membership import get_index_membership
from app.core.levels import compute_key_levels, compute_entry_exit
from app.core.macro_fetch import fetch_macro
from app.core.pe_fetch import get_pe_for_symbol
from app.core.forecast import expected_move, monte_carlo_forecast


def _last_non_null(seq):
    """Return the final non-None, non-NaN element of a list, or None."""
    if not seq:
        return None
    for v in reversed(seq):
        if v is None:
            continue
        # NaN check without numpy dependency
        if isinstance(v, float) and v != v:
            continue
        return float(v)
    return None


def _read_risk_free_rate(db) -> float:
    """T-bill rate as a percent, from settings. Falls back to the default."""
    if db is None:
        return float(DEFAULT_RISK_FREE_RATE_PCT)
    try:
        row = db.execute("SELECT value FROM settings WHERE key = 'risk_free_rate'").fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return float(DEFAULT_RISK_FREE_RATE_PCT)


def _get_egx30_close(exchange: str, interval: str, bars: int):
    """
    Benchmark closes for beta / relative strength, TTL-cached.

    The cache key is shared with the portfolio endpoint so every page scores
    against the identical benchmark series.
    """
    try:
        from app.vendor.egxpy import get_OHLCV_data
        ck = make_key("egx30", exchange, interval, bars)
        egx30_df = get(ck)
        if egx30_df is None:
            raw = get_OHLCV_data("EGX30", "EGX", interval, bars)
            if raw is not None and not raw.empty:
                raw.columns = [c.lower() for c in raw.columns]
                set(ck, raw)
                egx30_df = raw
        if egx30_df is not None and "close" in egx30_df.columns:
            return egx30_df["close"]
    except Exception:
        pass
    return None


router = APIRouter()


def _compute_batch_one(symbol: str, interval: str, weights: dict,
                       macro: Optional[dict] = None,
                       egx30_close=None,
                       risk_free_rate_pct: Optional[float] = None,
                       pe_ratio: Optional[float] = None,
                       dividend_yield: Optional[float] = None,
                       loss_making: Optional[bool] = None,
                       index_membership: Optional[str] = None) -> tuple:
    """
    Score one symbol for the dashboard grid.

    Uses the SAME window, lookback and extras builder as the stock detail
    page, so a card's score and signal equal what the user sees after
    tapping it. Anything cheaper here reintroduces the card-vs-detail
    divergence documented in core/extras_builder.py.
    """
    try:
        from app.vendor.egxpy import get_OHLCV_data

        df = get_OHLCV_data(symbol, "EGX", interval, INTERNAL_BARS_MIN)
        if df is None or df.empty:
            return symbol, {"error": "no data"}

        df.columns = [c.lower() for c in df.columns]
        close = df["close"]
        current_price = float(close.iloc[-1])

        indicators = compute_all(df)

        built = build_composite_extras(
            df, indicators,
            interval=interval,
            egx30_close=egx30_close,
            include_multi_timeframe=True,
            risk_free_rate_pct=risk_free_rate_pct,
            pe_ratio=pe_ratio,
            dividend_yield=dividend_yield,
            loss_making=loss_making,
            index_membership=index_membership,
            divergence_lookback=DIVERGENCE_LOOKBACK_FULL,
        )

        comp = compute_composite(
            indicators,
            extras=built["extras"],
            weights=weights,
            macro=macro,
        )

        prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        sparkline = [float(x) for x in close.iloc[-30:].tolist()]

        return symbol, {
            "score": comp["score"],
            "signal": comp["signal"],
            "price": current_price,
            "change": change,
            "change_pct": change_pct,
            "sparkline": sparkline,
        }

    except Exception as e:
        return symbol, {"error": str(e)}


def scoring_cache_context(user_id: str = None):
    """
    The three non-price inputs that make a composite score what it is:
    weights, macro regime, risk-free rate. Returns (weights, macro, tags).

    Extracted so `_handle_batch` (which WRITES per-symbol score cache entries)
    and `read_cached_scores` (which READS them) derive the key from one place.
    Building the key twice is how a reader silently gets zero hits forever and
    a feature quietly reports "no data" while the data is right there.

    `user_id` selects whose weights apply. It does NOT need to enter the cache
    key: `composite_cache_key` already folds in `weights_hash(weights)`, so two
    users with different sliders land on different keys automatically, and two
    users with the same sliders correctly SHARE one entry. Adding user_id would
    fragment the cache for no gain.
    """
    try:
        db = get_db()
        weights = get_weights_from_db(db, user_id)
    except Exception:
        db = None
        weights = dict(DEFAULT_WEIGHTS)

    macro = None
    if db is not None:
        try:
            macro = fetch_macro(db)
        except Exception:
            macro = None

    macro_tag = str(((macro or {}).get("egx30") or {}).get("trend") or "n/a")
    risk_free_rate_pct = _read_risk_free_rate(db)
    tags = (weights_hash(weights), macro_tag, f"rfr{risk_free_rate_pct:g}")
    return db, weights, macro, risk_free_rate_pct, tags


def composite_cache_key(symbol: str, interval: str, tags) -> str:
    """The one place a per-symbol composite cache key is spelled."""
    w_hash, macro_tag, rfr_tag = tags
    return make_key("composite", symbol.upper(), interval, w_hash, macro_tag, rfr_tag)


def read_cached_scores(symbols: list, interval: str = "Daily") -> dict:
    """
    Composite scores for `symbols` that are ALREADY cached. Never fetches.

    The market-condition reading needs a market-wide average, and scoring 79
    symbols cannot complete inside a serverless request — measured at over
    400 s against a cold cache, because each symbol pulls 400 bars through a
    client that retries hard on socket timeouts. So the reading consumes what
    the dashboard has already scored and reports its coverage honestly.

    Pinned to the ANONYMOUS weights context (user_id=None). The regime bands
    were calibrated by scripts/backtest.py at default weights; averaging scores
    computed under one user's custom sliders would invalidate that calibration.
    Anonymous dashboard traffic is what warms these entries, so the reader
    keeps hitting them.
    """
    _, _, _, _, tags = scoring_cache_context(None)
    out = {}
    for sym in symbols:
        cached = get(composite_cache_key(sym, interval, tags))
        if cached is not None and "error" not in cached and cached.get("score") is not None:
            out[sym.upper()] = cached
    return out


def _handle_batch(symbols_str: str, interval: str, user_id: str = None):
    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="Missing required parameter: symbols")
    if len(symbols) > BATCH_MAX_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Maximum {BATCH_MAX_SYMBOLS} symbols per batch")

    symbols = list(dict.fromkeys(symbols))

    # Weights, macro regime and risk-free rate are the non-price scoring
    # inputs, so all three belong in the cache key: without them a bullish→
    # bearish flip, or a settings change, serves stale scores for the whole
    # TTL. Derived through the shared helper so `read_cached_scores` cannot
    # spell the same key differently and silently find nothing.
    db, weights, macro, risk_free_rate_pct, tags = scoring_cache_context(user_id)

    # EGX30 fetched ONCE for the whole batch (not per symbol) and under the
    # same cache key /api/analysis uses, so relative strength is measured
    # against an identical benchmark series on both pages.
    egx30_close = _get_egx30_close("EGX", interval, INTERNAL_BARS_MIN)

    scores = {}
    errors = []
    todo = []

    for sym in symbols:
        ck = composite_cache_key(sym, interval, tags)
        cached = get(ck)
        if cached is not None:
            if "error" in cached:
                errors.append({"symbol": sym, "error": cached["error"]})
            else:
                scores[sym] = cached
        else:
            todo.append(sym)

    if todo:
        pool = ThreadPoolExecutor(max_workers=BATCH_WORKERS)
        try:
            def _cache_on_done(sym: str):
                ck = composite_cache_key(sym, interval, tags)
                def _cb(f):
                    try:
                        _s, r = f.result()
                        if "error" not in r:
                            set(ck, r)
                    except Exception:
                        pass
                return _cb

            futures: dict = {}
            for s in todo:
                # Fundamentals are a single indexed row per symbol — cheap
                # enough to read per symbol, and score_quality's bands need
                # them to match what the detail page scores.
                pe_ratio = dividend_yield = loss_making = None
                if db is not None:
                    try:
                        pe_row = get_pe_for_symbol(db, s)
                        if pe_row:
                            pe_ratio = pe_row.get("pe_ratio")
                            dividend_yield = pe_row.get("dividend_yield")
                            loss_making = pe_row.get("loss_making")
                    except Exception:
                        pe_ratio = dividend_yield = loss_making = None
                f = pool.submit(_compute_batch_one, s, interval, weights, macro,
                                egx30_close, risk_free_rate_pct, pe_ratio,
                                dividend_yield, loss_making,
                                get_index_membership(s))
                # Stragglers that finish AFTER we've returned still self-cache
                # via this callback — a frontend retry a few seconds later hits
                # a warm cache and fills in the '--' cards.
                f.add_done_callback(_cache_on_done(s))
                futures[f] = s

            deadline = time.monotonic() + BATCH_DEADLINE_SECONDS
            for fut, sym in futures.items():
                remaining = deadline - time.monotonic()
                try:
                    if remaining <= 0:
                        raise FuturesTimeoutError()
                    _sym, result = fut.result(timeout=remaining)
                except FuturesTimeoutError:
                    result = {"error": "upstream timeout"}
                except Exception as e:
                    result = {"error": str(e)}
                if "error" in result:
                    errors.append({"symbol": sym, "error": result["error"]})
                else:
                    scores[sym] = result
        finally:
            # Don't block on stuck threads — Vercel recycles the container.
            pool.shutdown(wait=False)

    return {"scores": scores, "errors": errors}


@router.get("/api/analysis")
def get_analysis(
    symbol: Optional[str] = Query(None),
    exchange: str = Query("EGX"),
    interval: str = Query("Daily"),
    bars: int = Query(200),
    mode: Optional[str] = Query(None),
    symbols: Optional[str] = Query(None),
    # Optional, not required: the dashboard is a public page (middleware.ts
    # guards only /portfolio), so demanding a token here would break anonymous
    # browsing. Anonymous callers score under the global/default weights.
    user: Optional[CurrentUser] = Depends(get_optional_user),
):
    try:
        user_id = user.id if user else None

        # Batch mode
        if mode == "batch":
            return _handle_batch(symbols or "", interval.capitalize(), user_id)

        if not symbol:
            raise HTTPException(status_code=400, detail="Missing required parameter: symbol")

        symbol = symbol.upper()
        exchange = exchange.upper()
        interval = interval.capitalize()
        bars = min(max(bars, USER_BARS_MIN), USER_BARS_MAX)

        db = get_db()
        weights = get_weights_from_db(db, user_id)
        w_hash = weights_hash(weights)

        # Macro regime and the T-bill rate are both SCORING INPUTS (macro
        # modulates the final score; Risk-Adjusted compares against the rate),
        # so both must be in the cache key. Without them, an EGX30 regime flip
        # updated the dashboard immediately while this endpoint served a
        # pre-flip score for the rest of the TTL — the same stock reading
        # "Hold" on one page and "Buy" on the other in the same minute.
        # P/E is deliberately NOT keyed: it changes once nightly, so the TTL
        # already bounds staleness, and keying it would force a DB read before
        # every cache hit.
        try:
            macro = fetch_macro(db)
        except Exception:
            macro = None
        macro_tag = str(((macro or {}).get("egx30") or {}).get("trend") or "n/a")
        risk_free_rate_pct = _read_risk_free_rate(db)
        rfr_tag = f"rfr{risk_free_rate_pct:g}"

        cache_key = make_key("analysis", symbol, exchange, interval, bars,
                             w_hash, macro_tag, rfr_tag)
        cached = get(cache_key)
        if cached:
            return cached

        from app.vendor.egxpy import get_OHLCV_data
        import pandas as pd

        internal_bars = max(bars, INTERNAL_BARS_MIN)
        df = get_OHLCV_data(symbol, exchange, interval, internal_bars)

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        df.columns = [c.lower() for c in df.columns]

        indicators_full = compute_all(df)

        df_trimmed = df.iloc[-bars:]
        actual_bars = len(df_trimmed)

        indicators = {}
        for key, vals in indicators_full.items():
            indicators[key] = vals[-actual_bars:]

        # Cumulative return is a running product from the FIRST bar of the
        # series, so slicing its tail leaves it anchored to a bar the user
        # can't see (a 60-bar chart showing -8% reported "+143%"). Recompute
        # it over the visible window instead.
        try:
            indicators["cumulative_returns"] = cumulative_returns(df_trimmed["close"]).tolist()
        except Exception:
            pass

        dates = [str(idx)[:10] for idx in df_trimmed.index]
        ohlcv = {
            "dates": dates,
            "open": df_trimmed["open"].tolist(),
            "high": df_trimmed["high"].tolist(),
            "low": df_trimmed["low"].tolist(),
            "close": df_trimmed["close"].tolist(),
            "volume": [int(v) for v in df_trimmed["volume"].tolist()],
        }

        close = df_trimmed["close"]

        # Benchmark series, shared cache key with the batch + portfolio paths.
        egx30_close = _get_egx30_close(exchange, interval, internal_bars)

        # Fundamentals from the nightly feed; None when no stored row.
        pe_info = None
        try:
            pe_info = get_pe_for_symbol(db, symbol)
        except Exception:
            pe_info = None

        # Same builder the dashboard cards and the portfolio page use, so all
        # three produce the same score for the same symbol. Built before `stats`
        # because the 52-week range is one of its by-products.
        built = build_composite_extras(
            df, indicators_full,
            interval=interval,
            egx30_close=egx30_close,
            include_multi_timeframe=True,
            risk_free_rate_pct=risk_free_rate_pct,
            pe_ratio=pe_info.get("pe_ratio") if pe_info else None,
            dividend_yield=pe_info.get("dividend_yield") if pe_info else None,
            loss_making=pe_info.get("loss_making") if pe_info else None,
            index_membership=get_index_membership(symbol),
            divergence_lookback=DIVERGENCE_LOOKBACK_FULL,
        )
        divergences = built["divergences"]
        volume_price = built["volume_price"]
        multi_timeframe = built["multi_timeframe"]
        bb_squeeze = built["bb_squeeze"]

        # 52-week range comes from the builder, which reads the FULL internal
        # frame and the true intraday high/low columns. Computing it here off
        # the trimmed frame made "52W High" a function of the chart's bar-count
        # selector, and reading closes missed intraday extremes. It lives in the
        # builder so score_quality's drawdown and this number are the same fact.
        stats = {
            "current_price": float(close.iloc[-1]),
            "previous_close": float(close.iloc[-2]) if len(close) > 1 else None,
            "change": float(close.iloc[-1] - close.iloc[-2]) if len(close) > 1 else 0,
            "change_pct": float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0,
            "high_52w": built["high_52w"],
            "low_52w": built["low_52w"],
            "avg_volume": int(df_trimmed["volume"].tail(20).mean()),
        }

        sr = support_resistance(df["high"], df["low"], df["close"])
        fib = fibonacci_levels(df["high"], df["low"])

        sma_50_full = sma(df["close"], 50)
        sma_200_full = sma(df["close"], 200)
        all_dates = [str(idx)[:10] for idx in df.index]
        crossovers = ma_crossovers(sma_50_full, sma_200_full, all_dates)

        close_full = df["close"]

        beta = None
        try:
            if egx30_close is not None:
                beta = compute_beta(daily_returns(close_full), daily_returns(egx30_close))
                if beta is not None:
                    beta = round(beta, 2)
        except Exception:
            beta = None

        # Descriptive forecasts — NOT predictions. Expected-move is a 1-sigma
        # historical band (~68% of days fall inside). Monte Carlo projects 60
        # days of plausible price paths from historical mu/sigma. Both are
        # statistical ranges, not directional signals.
        forecast = None
        try:
            returns_full = daily_returns(close_full)
            forecast = {
                "expected_move": expected_move(returns_full),
                "monte_carlo": monte_carlo_forecast(
                    returns_full, float(close_full.iloc[-1])
                ),
            }
        except Exception:
            forecast = None

        composite = compute_composite(
            indicators_full,
            extras=built["extras"],
            weights=weights,
            macro=macro,
        )

        # Key levels + entry/exit zones — consumed by the KeyLevelsCard /
        # EntryExitCard on the stock detail page. Purely a presentation layer
        # over support_resistance + RSI/Stochastic/ATR; doesn't alter the score.
        rsi_latest = _last_non_null(indicators_full.get("rsi"))
        stoch_k_latest = _last_non_null(indicators_full.get("stochastic_k"))
        atr_latest = _last_non_null(indicators_full.get("atr"))
        key_levels = compute_key_levels(
            float(close_full.iloc[-1]), sr, high=df["high"], low=df["low"]
        )
        entry_exit = compute_entry_exit(
            float(close_full.iloc[-1]), sr,
            rsi_latest=rsi_latest,
            stoch_k_latest=stoch_k_latest,
            atr_latest=atr_latest,
        )

        result = {
            "symbol": symbol,
            "interval": interval,
            "bars": actual_bars,
            "ohlcv": ohlcv,
            "indicators": indicators,
            "stats": stats,
            "beta": beta,
            "support_resistance": sr,
            "fibonacci": fib,
            "crossovers": crossovers,
            "composite_score": composite,
            "divergences": divergences,
            "volume_price": volume_price,
            "multi_timeframe": multi_timeframe,
            "bb_squeeze": bb_squeeze,
            "key_levels": key_levels,
            "entry_exit": entry_exit,
            "pe": pe_info,
            "forecast": forecast,
        }

        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing analysis: {str(e)}")
