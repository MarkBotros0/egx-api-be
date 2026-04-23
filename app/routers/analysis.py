"""
GET /api/analysis — Fetch OHLCV data and compute all technical indicators for a stock.

Also supports ?mode=batch&symbols=A,B,C for lightweight composite-only batch scoring.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.cache import get, set, make_key
from app.core.constants import (
    BATCH_BARS,
    BATCH_DEADLINE_SECONDS,
    BATCH_MAX_SYMBOLS,
    BATCH_WORKERS,
    BB_SQUEEZE_LOOKBACK_BARS,
    BB_SQUEEZE_RATIO,
    DIVERGENCE_LOOKBACK_BATCH,
    DIVERGENCE_LOOKBACK_FULL,
    INTERNAL_BARS_MIN,
    TRADING_DAYS_PER_YEAR,
    USER_BARS_MAX,
    USER_BARS_MIN,
)
from app.core.db import get_db
from app.core.indicators import (
    compute_all, support_resistance, fibonacci_levels, ma_crossovers,
    compute_beta, daily_returns, sma, rsi, macd,
    detect_divergences, volume_price_confirmation, multi_timeframe_alignment,
    relative_strength, annualized_return, atr,
)
from app.core.composite import (
    compute_composite, get_weights_from_db, weights_hash, DEFAULT_WEIGHTS,
)
from app.core.levels import compute_key_levels, compute_entry_exit
from app.core.macro_fetch import fetch_macro
from app.core.pe_fetch import get_pe_for_symbol


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

router = APIRouter()


def _compute_batch_one(symbol: str, interval: str, weights: dict,
                       macro: Optional[dict] = None) -> tuple:
    try:
        from egxpy.download import get_OHLCV_data

        df = get_OHLCV_data(symbol, "EGX", interval, BATCH_BARS)
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
                "rsi": detect_divergences(close, rsi_series, lookback=DIVERGENCE_LOOKBACK_BATCH),
                "macd": detect_divergences(close, macd_line_series, lookback=DIVERGENCE_LOOKBACK_BATCH),
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
            if bb_u and bb_l and bb_m and len(bb_u) >= BB_SQUEEZE_LOOKBACK_BARS:
                widths = [
                    (u - l) / m if m else None
                    for u, l, m in zip(bb_u, bb_l, bb_m)
                ]
                widths_valid = [w for w in widths[-BB_SQUEEZE_LOOKBACK_BARS:] if w is not None and w == w]
                if widths_valid:
                    current_w = widths_valid[-1]
                    avg_w = sum(widths_valid) / len(widths_valid)
                    bb_squeeze = current_w < avg_w * BB_SQUEEZE_RATIO
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
                # Batch keeps it lightweight — full 8-category breakdown lives
                # on the stock detail page; only pass history so the risk-adjusted
                # scorer's min-history gate triggers correctly.
                "history_days": len(close),
            },
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


def _handle_batch(symbols_str: str, interval: str):
    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="Missing required parameter: symbols")
    if len(symbols) > BATCH_MAX_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Maximum {BATCH_MAX_SYMBOLS} symbols per batch")

    symbols = list(dict.fromkeys(symbols))

    try:
        db = get_db()
        weights = get_weights_from_db(db)
    except Exception:
        db = None
        weights = dict(DEFAULT_WEIGHTS)
    w_hash = weights_hash(weights)

    # Macro is shared across all batch symbols and TTL-cached per-hour
    macro = None
    if db is not None:
        try:
            macro = fetch_macro(db)
        except Exception:
            macro = None
    macro_trend = ((macro or {}).get("egx30") or {}).get("trend") or "n/a"
    # Include macro regime in the cache key so scores invalidate when the
    # modulation changes (bullish → bearish flip).
    macro_tag = str(macro_trend)

    scores = {}
    errors = []
    todo = []

    for sym in symbols:
        ck = make_key("composite", sym, interval, w_hash, macro_tag)
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
                ck = make_key("composite", sym, interval, w_hash, macro_tag)
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
                f = pool.submit(_compute_batch_one, s, interval, weights, macro)
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
        bars = min(max(bars, USER_BARS_MIN), USER_BARS_MAX)

        db = get_db()
        weights = get_weights_from_db(db)
        w_hash = weights_hash(weights)

        cache_key = make_key("analysis", symbol, exchange, interval, bars, w_hash)
        cached = get(cache_key)
        if cached:
            return cached

        from egxpy.download import get_OHLCV_data
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
            "high_52w": float(close.tail(min(TRADING_DAYS_PER_YEAR, len(close))).max()),
            "low_52w": float(close.tail(min(TRADING_DAYS_PER_YEAR, len(close))).min()),
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
            "rsi": detect_divergences(close_full, rsi_series, lookback=DIVERGENCE_LOOKBACK_FULL),
            "macd": detect_divergences(close_full, macd_line_series, lookback=DIVERGENCE_LOOKBACK_FULL),
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
            if (bb_upper_full and bb_lower_full and bb_middle_full and len(bb_upper_full) >= BB_SQUEEZE_LOOKBACK_BARS):
                widths = [
                    (u - l) / m if m else None
                    for u, l, m in zip(bb_upper_full, bb_lower_full, bb_middle_full)
                ]
                widths_valid = [w for w in widths[-BB_SQUEEZE_LOOKBACK_BARS:] if w is not None and w == w]
                if widths_valid:
                    current_w = widths_valid[-1]
                    avg_w = sum(widths_valid) / len(widths_valid)
                    bb_squeeze = current_w < avg_w * BB_SQUEEZE_RATIO
        except Exception:
            bb_squeeze = False

        obv_full = indicators_full.get("obv") or []
        obv_rising = None
        price_rising_20d = None
        if len(obv_full) >= 21 and obv_full[-1] is not None and obv_full[-21] is not None:
            obv_rising = obv_full[-1] > obv_full[-21]
        if len(close_full) >= 21:
            price_rising_20d = float(close_full.iloc[-1]) > float(close_full.iloc[-21])

        # --- New extras for the 8-category composite ---

        # Trend consistency: fraction of last 20 bars with close above SMA20
        trend_consistency = None
        try:
            sma20 = sma(close_full, 20)
            last20 = close_full.iloc[-20:]
            last20_sma = sma20.iloc[-20:]
            paired = [(c, s) for c, s in zip(last20, last20_sma) if s == s]
            if paired:
                trend_consistency = sum(1 for c, s in paired if c > s) / len(paired)
        except Exception:
            trend_consistency = None

        # Current drawdown: price vs last-252-day peak (fraction, e.g. -0.15)
        current_drawdown_pct = None
        try:
            window = close_full.tail(min(TRADING_DAYS_PER_YEAR, len(close_full)))
            peak = float(window.max())
            cur = float(close_full.iloc[-1])
            if peak > 0:
                current_drawdown_pct = (cur - peak) / peak
        except Exception:
            current_drawdown_pct = None

        # Annualized return + annualized volatility
        ann_return_pct = annualized_return(close_full, lookback=TRADING_DAYS_PER_YEAR)
        volatility_annualized_pct = None
        try:
            daily_vol = daily_returns(close_full).std()
            if daily_vol == daily_vol:  # not NaN
                volatility_annualized_pct = float(daily_vol) * (TRADING_DAYS_PER_YEAR ** 0.5) * 100.0
        except Exception:
            volatility_annualized_pct = None

        # ATR as % of price
        atr_pct_of_price = None
        try:
            atr_series = atr(df["high"], df["low"], df["close"])
            last_atr = float(atr_series.dropna().iloc[-1])
            cur = float(close_full.iloc[-1])
            if cur > 0:
                atr_pct_of_price = last_atr / cur * 100.0
        except Exception:
            atr_pct_of_price = None

        # Relative strength vs EGX30 (reusing egx30_df fetched above for beta)
        rs = None
        try:
            if egx30_df is not None and "close" in egx30_df.columns:
                rs = relative_strength(close_full, egx30_df["close"], lookback=30)
        except Exception:
            rs = None

        # Risk-free rate (T-bill) — read from settings
        risk_free_rate_pct = 25.0
        try:
            row = db.execute("SELECT value FROM settings WHERE key = 'risk_free_rate'").fetchone()
            if row and row[0] is not None:
                risk_free_rate_pct = float(row[0])
        except Exception:
            pass

        # Macro context (cheap — TTL-cached)
        try:
            macro = fetch_macro(db)
        except Exception:
            macro = None

        # P/E from the nightly EGX scrape; None when no stored row.
        pe_info = None
        try:
            pe_info = get_pe_for_symbol(db, symbol)
        except Exception:
            pe_info = None

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
                # New 8-category inputs
                "multi_timeframe": multi_timeframe,
                "trend_consistency": trend_consistency,
                "current_drawdown_pct": current_drawdown_pct,
                "annualized_return_pct": ann_return_pct,
                "volatility_annualized_pct": volatility_annualized_pct,
                "atr_pct_of_price": atr_pct_of_price,
                "history_days": len(close_full),
                "risk_free_rate_pct": risk_free_rate_pct,
                "relative_strength": rs,
                "pe_ratio": pe_info.get("pe_ratio") if pe_info else None,
            },
            weights=weights,
            macro=macro,
        )

        # Key levels + entry/exit zones — consumed by the KeyLevelsCard /
        # EntryExitCard on the stock detail page. Purely a presentation layer
        # over support_resistance + RSI/Stochastic/ATR; doesn't alter the score.
        rsi_latest = _last_non_null(indicators_full.get("rsi"))
        stoch_k_latest = _last_non_null(indicators_full.get("stochastic_k"))
        atr_latest = _last_non_null(indicators_full.get("atr"))
        key_levels = compute_key_levels(float(close_full.iloc[-1]), sr)
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
        }

        set(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing analysis: {str(e)}")
