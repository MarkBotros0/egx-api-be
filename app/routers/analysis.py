"""
GET /api/analysis — Fetch OHLCV data and compute all technical indicators for a stock.

Also supports ?mode=batch&symbols=A,B,C for lightweight composite-only batch scoring.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.cache import get, set, make_key
from app.core.db import get_db
from app.core.indicators import (
    compute_all, support_resistance, fibonacci_levels, ma_crossovers,
    compute_beta, daily_returns, sma, rsi, macd,
    detect_divergences, volume_price_confirmation, multi_timeframe_alignment,
)
from app.core.composite import (
    compute_composite, get_weights_from_db, weights_hash, DEFAULT_WEIGHTS,
)

router = APIRouter()

_BATCH_MAX_SYMBOLS = 24
_BATCH_BARS = 220
_BATCH_WORKERS = 6
# One slow upstream fetch must not hang the whole chunk — return partial
# results once this budget is spent. Keep comfortably under Vercel's 30s.
_BATCH_DEADLINE_S = 20.0


def _compute_batch_one(symbol: str, interval: str, weights: dict) -> tuple:
    try:
        from egxpy.download import get_OHLCV_data

        df = get_OHLCV_data(symbol, "EGX", interval, _BATCH_BARS)
        if df is None or df.empty:
            return symbol, {"error": "no data"}

        df.columns = [c.lower() for c in df.columns]
        close = df["close"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])

        indicators = compute_all(df)

        try:
            sma_50_series = sma(close, 50)
            sma_200_series = sma(close, 200)
            dates_list = [str(idx)[:10] for idx in df.index]
            crossovers = ma_crossovers(sma_50_series, sma_200_series, dates_list)
        except Exception:
            crossovers = {"current_signal": None, "days_since_cross": None}

        try:
            rsi_series = rsi(close)
            macd_line_series, _, _ = macd(close)
            divergences = {
                "rsi": detect_divergences(close, rsi_series, lookback=30),
                "macd": detect_divergences(close, macd_line_series, lookback=30),
            }
        except Exception:
            divergences = {"rsi": {}, "macd": {}}

        try:
            volume_price = volume_price_confirmation(close, volume)
        except Exception:
            volume_price = None

        bb_squeeze = False
        try:
            bb_u = indicators.get("bollinger_upper") or []
            bb_l = indicators.get("bollinger_lower") or []
            bb_m = indicators.get("bollinger_middle") or []
            if bb_u and bb_l and bb_m and len(bb_u) >= 130:
                widths = [
                    (u - l) / m if m else None
                    for u, l, m in zip(bb_u, bb_l, bb_m)
                ]
                widths_valid = [w for w in widths[-130:] if w is not None and w == w]
                if widths_valid:
                    current_w = widths_valid[-1]
                    avg_w = sum(widths_valid) / len(widths_valid)
                    bb_squeeze = current_w < avg_w * 0.7
        except Exception:
            bb_squeeze = False

        obv_rising = None
        price_rising_20d = None
        obv_full = indicators.get("obv") or []
        if len(obv_full) >= 21 and obv_full[-1] is not None and obv_full[-21] is not None:
            obv_rising = obv_full[-1] > obv_full[-21]
        if len(close) >= 21:
            price_rising_20d = float(close.iloc[-1]) > float(close.iloc[-21])

        comp = compute_composite(
            indicators,
            extras={
                "current_price": current_price,
                "divergences": divergences,
                "volume_price": volume_price,
                "bb_squeeze": bb_squeeze,
                "obv_rising": obv_rising,
                "price_rising_20d": price_rising_20d,
                "golden_cross_active": crossovers.get("current_signal") == "golden_cross"
                                        and (crossovers.get("days_since_cross") or 99) < 10,
            },
            weights=weights,
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


def _handle_batch(symbols_str: str, interval: str):
    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="Missing required parameter: symbols")
    if len(symbols) > _BATCH_MAX_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Maximum {_BATCH_MAX_SYMBOLS} symbols per batch")

    symbols = list(dict.fromkeys(symbols))

    try:
        weights = get_weights_from_db(get_db())
    except Exception:
        weights = dict(DEFAULT_WEIGHTS)
    w_hash = weights_hash(weights)

    scores = {}
    errors = []
    todo = []

    for sym in symbols:
        ck = make_key("composite", sym, interval, w_hash)
        cached = get(ck)
        if cached is not None:
            if "error" in cached:
                errors.append({"symbol": sym, "error": cached["error"]})
            else:
                scores[sym] = cached
        else:
            todo.append(sym)

    if todo:
        pool = ThreadPoolExecutor(max_workers=_BATCH_WORKERS)
        try:
            def _cache_on_done(sym: str):
                ck = make_key("composite", sym, interval, w_hash)
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
                f = pool.submit(_compute_batch_one, s, interval, weights)
                # Stragglers that finish AFTER we've returned still self-cache
                # via this callback — a frontend retry a few seconds later hits
                # a warm cache and fills in the '--' cards.
                f.add_done_callback(_cache_on_done(s))
                futures[f] = s

            deadline = time.monotonic() + _BATCH_DEADLINE_S
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
):
    try:
        # Batch mode
        if mode == "batch":
            return _handle_batch(symbols or "", interval.capitalize())

        if not symbol:
            raise HTTPException(status_code=400, detail="Missing required parameter: symbol")

        symbol = symbol.upper()
        exchange = exchange.upper()
        interval = interval.capitalize()
        bars = min(max(bars, 30), 5000)

        db = get_db()
        weights = get_weights_from_db(db)
        w_hash = weights_hash(weights)

        cache_key = make_key("analysis", symbol, exchange, interval, bars, w_hash)
        cached = get(cache_key)
        if cached:
            return cached

        from egxpy.download import get_OHLCV_data
        import pandas as pd

        internal_bars = max(bars, 400)
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
        stats = {
            "current_price": float(close.iloc[-1]),
            "previous_close": float(close.iloc[-2]) if len(close) > 1 else None,
            "change": float(close.iloc[-1] - close.iloc[-2]) if len(close) > 1 else 0,
            "change_pct": float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0,
            "high_52w": float(close.tail(min(252, len(close))).max()),
            "low_52w": float(close.tail(min(252, len(close))).min()),
            "avg_volume": int(df_trimmed["volume"].tail(20).mean()),
        }

        sr = support_resistance(df["high"], df["low"], df["close"])
        fib = fibonacci_levels(df["high"], df["low"])

        sma_50_full = sma(df["close"], 50)
        sma_200_full = sma(df["close"], 200)
        all_dates = [str(idx)[:10] for idx in df.index]
        crossovers = ma_crossovers(sma_50_full, sma_200_full, all_dates)

        beta = None
        try:
            egx30_cache_key = make_key("egx30", exchange, interval, internal_bars)
            egx30_df = get(egx30_cache_key)
            if egx30_df is None:
                egx30_raw = get_OHLCV_data("EGX30", "EGX", interval, internal_bars)
                if egx30_raw is not None and not egx30_raw.empty:
                    egx30_raw.columns = [c.lower() for c in egx30_raw.columns]
                    set(egx30_cache_key, egx30_raw)
                    egx30_df = egx30_raw
            if egx30_df is not None:
                stock_rets = daily_returns(df["close"])
                market_rets = daily_returns(egx30_df["close"])
                beta = compute_beta(stock_rets, market_rets)
                if beta is not None:
                    beta = round(beta, 2)
        except Exception:
            beta = None

        close_full = df["close"]
        rsi_series = rsi(close_full)
        macd_line_series, _, _ = macd(close_full)
        divergences = {
            "rsi": detect_divergences(close_full, rsi_series, lookback=60),
            "macd": detect_divergences(close_full, macd_line_series, lookback=60),
        }

        volume_price = volume_price_confirmation(close_full, df["volume"])

        multi_timeframe = None
        if interval == "Daily":
            try:
                weekly_cache_key = make_key("weekly", symbol, exchange, 100)
                weekly_df = get(weekly_cache_key)
                if weekly_df is None:
                    weekly_raw = get_OHLCV_data(symbol, exchange, "Weekly", 100)
                    if weekly_raw is not None and not weekly_raw.empty:
                        weekly_raw.columns = [c.lower() for c in weekly_raw.columns]
                        set(weekly_cache_key, weekly_raw)
                        weekly_df = weekly_raw
                if weekly_df is not None:
                    multi_timeframe = multi_timeframe_alignment(close_full, weekly_df["close"])
            except Exception:
                multi_timeframe = None

        bb_squeeze = False
        try:
            bb_upper_full = indicators_full.get("bollinger_upper") or []
            bb_lower_full = indicators_full.get("bollinger_lower") or []
            bb_middle_full = indicators_full.get("bollinger_middle") or []
            if (bb_upper_full and bb_lower_full and bb_middle_full and len(bb_upper_full) >= 130):
                widths = [
                    (u - l) / m if m else None
                    for u, l, m in zip(bb_upper_full, bb_lower_full, bb_middle_full)
                ]
                widths_valid = [w for w in widths[-130:] if w is not None and w == w]
                if widths_valid:
                    current_w = widths_valid[-1]
                    avg_w = sum(widths_valid) / len(widths_valid)
                    bb_squeeze = current_w < avg_w * 0.7
        except Exception:
            bb_squeeze = False

        obv_full = indicators_full.get("obv") or []
        obv_rising = None
        price_rising_20d = None
        if len(obv_full) >= 21 and obv_full[-1] is not None and obv_full[-21] is not None:
            obv_rising = obv_full[-1] > obv_full[-21]
        if len(close_full) >= 21:
            price_rising_20d = float(close_full.iloc[-1]) > float(close_full.iloc[-21])

        composite = compute_composite(
            indicators_full,
            extras={
                "current_price": float(close_full.iloc[-1]),
                "divergences": divergences,
                "volume_price": volume_price,
                "bb_squeeze": bb_squeeze,
                "obv_rising": obv_rising,
                "price_rising_20d": price_rising_20d,
                "golden_cross_active": crossovers.get("current_signal") == "golden_cross"
                                       and (crossovers.get("days_since_cross") or 99) < 10,
            },
            weights=weights,
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
        }

        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing analysis: {str(e)}")
