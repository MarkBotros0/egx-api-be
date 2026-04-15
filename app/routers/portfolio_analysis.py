"""
/api/portfolio_analysis — Analyze a portfolio of stock holdings.

GET  — Read holdings from Turso and analyze them
POST — Accept holdings in request body (body: {portfolio: [...], cash_available: float})
"""

from datetime import date, datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException

from app.core.db import get_db
from app.core.macro_fetch import fetch_macro
from app.core.composite import compute_composite, get_weights_from_db, DEFAULT_WEIGHTS
from app.core.constants import (
    BIG_LOSS_PCT,
    CONCENTRATION_CRITICAL_PCT,
    CONCENTRATION_WARNING_PCT,
    CORRELATION_HIGH_THRESHOLD,
    CORRELATION_NEGATIVE_THRESHOLD,
    CURRENT_DRAWDOWN_WARNING_PCT,
    DEFAULT_RISK_FREE_RATE_PCT,
    DIVERGENCE_LOOKBACK_PORTFOLIO,
    MAX_DRAWDOWN_WARNING_PCT,
    MONTE_CARLO_FORECAST_DAYS,
    MONTE_CARLO_SIMULATIONS,
    PORTFOLIO_FETCH_BARS_MAX,
    PORTFOLIO_FETCH_BARS_MIN,
    PROFIT_TARGET_PCT,
    SCORE_BUY_MAX,
    SCORE_STRONG_SELL_MAX,
    SECTOR_ALERT_PCT,
    STOCK_ALERT_PCT,
    TRADING_DAYS_PER_YEAR,
    VAR_PERCENTILE,
)
from app.core.indicators import (
    rsi as calc_rsi, sma as calc_sma, volatility as calc_volatility,
    daily_returns as calc_daily_returns, compute_beta, obv as calc_obv,
    stochastic as calc_stochastic, atr as calc_atr,
    support_resistance, fibonacci_levels, ma_crossovers,
    macd as calc_macd, bollinger_bands as calc_bollinger,
    adx as calc_adx, mfi as calc_mfi,
    detect_divergences, volume_price_confirmation,
    relative_strength as calc_relative_strength,
    annualized_return as calc_annualized_return,
    liquidity_score as calc_liquidity,
)

router = APIRouter()


def _days_between(date_str: str, today: date) -> int:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (today - d).days
    except Exception:
        return 0


def _annualized_return(total_return_pct: float, days_held: int) -> float:
    if days_held <= 0:
        return 0.0
    return ((1 + total_return_pct / 100) ** (365 / days_held) - 1) * 100


def _analyze(holdings, cash):
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings provided")

    from egxpy.download import get_OHLCV_data
    from app.core.cache import get as cache_get, set as cache_set, make_key
    import pandas as pd
    import numpy as np

    today = date.today()
    stock_analyses = []
    signals = []
    total_invested = 0
    total_current_value = 0
    sector_values = {}
    stock_values = {}
    all_returns = {}

    egx30_returns = None
    egx30_close = None
    egx30_df = None
    try:
        egx30_cache_key = make_key("egx30", "EGX", "Daily", 300)
        egx30_df = cache_get(egx30_cache_key)
        if egx30_df is None:
            egx30_raw = get_OHLCV_data("EGX30", "EGX", "Daily", 300)
            if egx30_raw is not None and not egx30_raw.empty:
                egx30_raw.columns = [c.lower() for c in egx30_raw.columns]
                cache_set(egx30_cache_key, egx30_raw)
                egx30_df = egx30_raw
        if egx30_df is not None:
            egx30_returns = calc_daily_returns(egx30_df["close"])
            egx30_close = egx30_df["close"]
    except Exception:
        pass

    try:
        db = get_db()
        rfr_row = db.execute("SELECT value FROM settings WHERE key = 'risk_free_rate'").fetchone()
        risk_free_annual = float(rfr_row[0]) / 100 if rfr_row else DEFAULT_RISK_FREE_RATE_PCT / 100
    except Exception:
        risk_free_annual = DEFAULT_RISK_FREE_RATE_PCT / 100
    risk_free_rate_pct = risk_free_annual * 100

    try:
        weights = get_weights_from_db(get_db())
    except Exception:
        weights = dict(DEFAULT_WEIGHTS)

    # Macro fetched once up-front — feeds both per-holding composite modulation
    # and the portfolio-level macro_egx30 signal below.
    macro_data = None
    try:
        macro_data = fetch_macro(get_db())
    except Exception:
        macro_data = None

    composite_scores_collected = []

    for h in holdings:
        symbol = h.get("symbol", "").upper()
        buy_price = float(h.get("buy_price", 0))
        quantity = int(h.get("quantity", 0))
        buy_date = h.get("buy_date", "")
        target_price = h.get("target_price")
        stop_loss = h.get("stop_loss")
        sector = h.get("sector", "Unknown")

        invested = buy_price * quantity
        total_invested += invested

        try:
            days_held = _days_between(buy_date, today)
            fetch_bars = min(max(PORTFOLIO_FETCH_BARS_MIN, days_held), PORTFOLIO_FETCH_BARS_MAX)
            df = get_OHLCV_data(symbol, "EGX", "Daily", fetch_bars)
            if df is None or df.empty:
                stock_analyses.append({"symbol": symbol, "error": "Could not fetch market data"})
                continue

            df.columns = [c.lower() for c in df.columns]
            close = df["close"]
            current_price = float(close.iloc[-1])

            current_value = current_price * quantity
            total_current_value += current_value
            pnl = (current_price - buy_price) * quantity
            pnl_pct = (current_price / buy_price - 1) * 100 if buy_price > 0 else 0
            ann_return = _annualized_return(pnl_pct, days_held)

            rsi_series = calc_rsi(close, 14)
            current_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None

            sma_50 = calc_sma(close, 50)
            current_sma_50 = float(sma_50.iloc[-1]) if not np.isnan(sma_50.iloc[-1]) else None
            above_sma = current_price > current_sma_50 if current_sma_50 else None

            sma_200 = calc_sma(close, 200)
            current_sma_200 = float(sma_200.iloc[-1]) if len(sma_200.dropna()) > 0 else None

            vol_series = calc_volatility(close, 20)
            current_vol = float(vol_series.iloc[-1]) if not np.isnan(vol_series.iloc[-1]) else None

            stock_rets = calc_daily_returns(close)
            all_returns[symbol] = stock_rets

            beta = None
            if egx30_returns is not None:
                beta = compute_beta(stock_rets, egx30_returns)
                if beta is not None:
                    beta = round(beta, 2)

            atr_series = calc_atr(df["high"], df["low"], close)
            current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else None
            atr_pct = round(current_atr / current_price * 100, 1) if current_atr and current_price > 0 else None

            obv_series = calc_obv(close, df["volume"])
            obv_rising = float(obv_series.iloc[-1]) > float(obv_series.iloc[-min(20, len(obv_series))]) if len(obv_series) >= 5 else None
            price_rising = current_price > float(close.iloc[-min(20, len(close))]) if len(close) >= 5 else None
            if obv_rising is not None and price_rising is not None:
                if price_rising and obv_rising:
                    obv_trend = "confirming"
                elif price_rising and not obv_rising:
                    obv_trend = "diverging_bearish"
                elif not price_rising and obv_rising:
                    obv_trend = "diverging_bullish"
                else:
                    obv_trend = "confirming_bearish"
            else:
                obv_trend = None

            stoch_k, stoch_d = calc_stochastic(df["high"], df["low"], close)
            current_stoch_k = float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else None
            current_stoch_d = float(stoch_d.iloc[-1]) if not np.isnan(stoch_d.iloc[-1]) else None

            sr = support_resistance(df["high"], df["low"], close)
            fib = fibonacci_levels(df["high"], df["low"])
            dates_list = [str(idx)[:10] for idx in df.index]
            crossovers = ma_crossovers(sma_50, sma_200, dates_list)
            trend = "bullish" if crossovers["current_signal"] == "golden_cross" else \
                    "bearish" if crossovers["current_signal"] == "death_cross" else None

            vol_20_avg = float(df["volume"].tail(20).mean())
            vol_5_avg = float(df["volume"].tail(5).mean())
            volume_trend = "increasing" if vol_5_avg > vol_20_avg * 1.1 else (
                "decreasing" if vol_5_avg < vol_20_avg * 0.9 else "stable"
            )

            try:
                adx_series, plus_di_series, minus_di_series = calc_adx(df["high"], df["low"], close)
                current_adx = float(adx_series.iloc[-1]) if len(adx_series) and not np.isnan(adx_series.iloc[-1]) else None
                current_plus_di = float(plus_di_series.iloc[-1]) if len(plus_di_series) and not np.isnan(plus_di_series.iloc[-1]) else None
                current_minus_di = float(minus_di_series.iloc[-1]) if len(minus_di_series) and not np.isnan(minus_di_series.iloc[-1]) else None
            except Exception:
                adx_series = plus_di_series = minus_di_series = None
                current_adx = current_plus_di = current_minus_di = None

            try:
                mfi_series = calc_mfi(df["high"], df["low"], close, df["volume"])
                current_mfi = float(mfi_series.iloc[-1]) if len(mfi_series) and not np.isnan(mfi_series.iloc[-1]) else None
            except Exception:
                mfi_series = None
                current_mfi = None

            try:
                macd_line_series, _macd_signal_series, macd_hist_series = calc_macd(close)
            except Exception:
                macd_line_series = _macd_signal_series = macd_hist_series = None

            try:
                bb_upper_series, bb_middle_series, bb_lower_series = calc_bollinger(close)
            except Exception:
                bb_upper_series = bb_middle_series = bb_lower_series = None

            divergences_h = {
                "rsi": detect_divergences(close, rsi_series, lookback=DIVERGENCE_LOOKBACK_PORTFOLIO) if rsi_series is not None else {},
                "macd": detect_divergences(close, macd_line_series, lookback=DIVERGENCE_LOOKBACK_PORTFOLIO) if macd_line_series is not None else {},
            }

            volume_price_h = volume_price_confirmation(close, df["volume"])

            def _tolist(s):
                try:
                    return s.tolist() if s is not None else []
                except Exception:
                    return []

            holding_indicators = {
                "sma_20": _tolist(calc_sma(close, 20)),
                "sma_50": _tolist(sma_50),
                "sma_200": _tolist(sma_200),
                "rsi": _tolist(rsi_series),
                "macd_histogram": _tolist(macd_hist_series),
                "stochastic_k": _tolist(stoch_k),
                "stochastic_d": _tolist(stoch_d),
                "adx": _tolist(adx_series),
                "plus_di": _tolist(plus_di_series),
                "minus_di": _tolist(minus_di_series),
                "mfi": _tolist(mfi_series),
                "bollinger_upper": _tolist(bb_upper_series),
                "bollinger_middle": _tolist(bb_middle_series),
                "bollinger_lower": _tolist(bb_lower_series),
                "obv": _tolist(obv_series),
            }

            # New per-holding extras for the 8-category composite
            try:
                last20 = close.iloc[-20:]
                sma20_h = calc_sma(close, 20)
                last20_sma = sma20_h.iloc[-20:]
                paired = [(c, s) for c, s in zip(last20, last20_sma) if s == s]
                trend_consistency_h = (sum(1 for c, s in paired if c > s) / len(paired)) if paired else None
            except Exception:
                trend_consistency_h = None

            try:
                window = close.tail(min(TRADING_DAYS_PER_YEAR, len(close)))
                peak = float(window.max())
                current_drawdown_h = (current_price - peak) / peak if peak > 0 else None
            except Exception:
                current_drawdown_h = None

            ann_return_pct_h = calc_annualized_return(close, lookback=TRADING_DAYS_PER_YEAR)
            try:
                daily_vol = stock_rets.std()
                volatility_ann_pct_h = float(daily_vol) * (TRADING_DAYS_PER_YEAR ** 0.5) * 100.0 if daily_vol == daily_vol else None
            except Exception:
                volatility_ann_pct_h = None

            rs_h = None
            try:
                if egx30_close is not None:
                    rs_h = calc_relative_strength(close, egx30_close, lookback=30)
            except Exception:
                rs_h = None

            liquidity_h = None
            try:
                liquidity_h = calc_liquidity(df["volume"], index_membership=None, lookback=20)
            except Exception:
                liquidity_h = None

            try:
                composite_h = compute_composite(
                    holding_indicators,
                    extras={
                        "current_price": current_price,
                        "divergences": divergences_h,
                        "volume_price": volume_price_h,
                        "bb_squeeze": False,
                        "obv_rising": obv_rising,
                        "price_rising_20d": price_rising,
                        "golden_cross_active": crossovers.get("current_signal") == "golden_cross"
                                               and (crossovers.get("days_since_cross") or 99) < 10,
                        # multi_timeframe is omitted — weekly fetch per-holding is too costly for portfolio timeout.
                        "multi_timeframe": None,
                        "trend_consistency": trend_consistency_h,
                        "current_drawdown_pct": current_drawdown_h,
                        "annualized_return_pct": ann_return_pct_h,
                        "volatility_annualized_pct": volatility_ann_pct_h,
                        "atr_pct_of_price": atr_pct,
                        "history_days": len(close),
                        "risk_free_rate_pct": risk_free_rate_pct,
                        "relative_strength": rs_h,
                    },
                    weights=weights,
                    macro=macro_data,
                )
            except Exception:
                composite_h = None

            dist_to_target = None
            dist_to_stop = None
            if target_price:
                target_price = float(target_price)
                dist_to_target = (target_price / current_price - 1) * 100
            if stop_loss:
                stop_loss = float(stop_loss)
                dist_to_stop = (stop_loss / current_price - 1) * 100

            sector_values[sector] = sector_values.get(sector, 0) + current_value
            stock_values[symbol] = stock_values.get(symbol, 0) + current_value

            analysis = {
                "id": h.get("id"),
                "symbol": symbol,
                "name": h.get("name", symbol),
                "sector": sector,
                "quantity": quantity,
                "buy_price": buy_price,
                "buy_date": buy_date,
                "current_price": current_price,
                "current_value": round(current_value, 2),
                "invested": round(invested, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
                "annualized_return": round(ann_return, 2),
                "rsi": round(current_rsi, 1) if current_rsi else None,
                "sma_50": round(current_sma_50, 2) if current_sma_50 else None,
                "above_sma": above_sma,
                "volatility": round(current_vol, 4) if current_vol else None,
                "volume_trend": volume_trend,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "dist_to_target": round(dist_to_target, 2) if dist_to_target is not None else None,
                "dist_to_stop": round(dist_to_stop, 2) if dist_to_stop is not None else None,
                "beta": beta,
                "atr": round(current_atr, 2) if current_atr else None,
                "atr_pct": atr_pct,
                "obv_trend": obv_trend,
                "stochastic_k": round(current_stoch_k, 1) if current_stoch_k is not None else None,
                "stochastic_d": round(current_stoch_d, 1) if current_stoch_d is not None else None,
                "supports": sr["supports"][:3],
                "resistances": sr["resistances"][:3],
                "fibonacci": fib,
                "trend": trend,
                "golden_cross_active": crossovers["current_signal"] == "golden_cross",
                "sma_200": round(current_sma_200, 2) if current_sma_200 else None,
                "adx": round(current_adx, 1) if current_adx is not None else None,
                "plus_di": round(current_plus_di, 1) if current_plus_di is not None else None,
                "minus_di": round(current_minus_di, 1) if current_minus_di is not None else None,
                "mfi": round(current_mfi, 1) if current_mfi is not None else None,
                "divergences": divergences_h,
                "volume_price": volume_price_h,
                "composite_score": composite_h["score"] if composite_h else None,
                "composite_signal": composite_h["signal"] if composite_h else None,
                "composite_breakdown": composite_h["categories"] if composite_h else None,
            }
            stock_analyses.append(analysis)

            if composite_h and composite_h.get("score") is not None:
                composite_scores_collected.append(composite_h["score"])

            # --- Signals ---
            if composite_h:
                c_score = composite_h["score"]
                c_signal = composite_h["signal"]
                if c_score >= SCORE_BUY_MAX:
                    signals.append({"type": "strong_buy_composite", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} composite score is {c_score:.0f} — Strong Buy across multiple indicators.",
                        "explanation": "The composite score blends trend, momentum, volume, volatility, and divergence into one number. A score ≥80 means most categories are aligned bullishly — a high-conviction setup.",
                        "learn_concept": "composite_score"})
                elif c_score <= SCORE_STRONG_SELL_MAX:
                    signals.append({"type": "strong_sell_composite", "severity": "action_required", "symbol": symbol,
                        "message": f"{symbol} composite score is {c_score:.0f} — Strong Sell. Most indicators are bearish.",
                        "explanation": "When the composite score falls below 20, trend, momentum, volume, volatility, and divergence are all flashing bearish.",
                        "learn_concept": "composite_score"})
                elif c_signal == "Buy":
                    signals.append({"type": "buy_composite", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} composite score is {c_score:.0f} — Buy signal from combined indicators.",
                        "explanation": "Multiple indicators agree this stock is in a favorable state.",
                        "learn_concept": "composite_score"})
                elif c_signal == "Sell":
                    signals.append({"type": "sell_composite", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} composite score is {c_score:.0f} — Sell signal from combined indicators.",
                        "explanation": "Multiple indicators agree this stock is weakening.",
                        "learn_concept": "composite_score"})

            for ind_name, div in (("RSI", divergences_h["rsi"]), ("MACD", divergences_h["macd"])):
                if not div:
                    continue
                if div.get("bullish"):
                    signals.append({"type": "divergence_bullish", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol}: bullish {ind_name} divergence — price lower low but {ind_name} higher low.",
                        "explanation": "Bullish divergence means sellers are losing strength even though price dropped.",
                        "learn_concept": "divergence"})
                elif div.get("bearish"):
                    signals.append({"type": "divergence_bearish", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol}: bearish {ind_name} divergence — price higher high but {ind_name} lower high.",
                        "explanation": "Bearish divergence means buyers are losing momentum despite the price rising.",
                        "learn_concept": "divergence"})

            if dist_to_stop is not None and dist_to_stop > -10:
                prio = "action_required" if dist_to_stop > -5 else "warning"
                signals.append({"type": "stop_loss", "severity": prio, "symbol": symbol,
                    "message": f"{symbol} is {abs(dist_to_stop):.1f}% away from your stop-loss at {stop_loss:.2f} EGP.",
                    "explanation": "A stop-loss is a pre-set price where you sell to limit losses.",
                    "learn_concept": "stop_loss"})

            if crossovers["current_signal"] == "death_cross" and crossovers["days_since_cross"] is not None and crossovers["days_since_cross"] <= 5:
                signals.append({"type": "death_cross", "severity": "action_required", "symbol": symbol,
                    "message": f"DEATH CROSS on {symbol}. The 50-day average crossed below the 200-day average.",
                    "explanation": "A Death Cross is a widely-watched bearish signal.",
                    "learn_concept": "golden_death_cross"})

            if crossovers["current_signal"] == "golden_cross" and crossovers["days_since_cross"] is not None and crossovers["days_since_cross"] <= 5:
                signals.append({"type": "golden_cross", "severity": "opportunity", "symbol": symbol,
                    "message": f"GOLDEN CROSS on {symbol}! The 50-day average just crossed above the 200-day average.",
                    "explanation": "A Golden Cross often precedes sustained uptrends.",
                    "learn_concept": "golden_death_cross"})

            if obv_trend == "diverging_bearish":
                signals.append({"type": "obv_divergence", "severity": "warning", "symbol": symbol,
                    "message": f"{symbol}: Price is rising BUT volume is declining.",
                    "explanation": "When price goes up but OBV goes down, the rally may not be sustainable.",
                    "learn_concept": "obv"})
            elif obv_trend == "diverging_bullish":
                signals.append({"type": "obv_accumulation", "severity": "opportunity", "symbol": symbol,
                    "message": f"{symbol}: Price is dropping but smart money may be accumulating (OBV rising).",
                    "explanation": "When volume flows in despite falling prices, it can indicate institutional accumulation.",
                    "learn_concept": "obv"})

            if current_stoch_k is not None and current_stoch_d is not None:
                prev_k = float(stoch_k.iloc[-2]) if len(stoch_k) > 1 and not np.isnan(stoch_k.iloc[-2]) else None
                prev_d = float(stoch_d.iloc[-2]) if len(stoch_d) > 1 and not np.isnan(stoch_d.iloc[-2]) else None
                if prev_k is not None and prev_d is not None:
                    if current_stoch_k < 20 and prev_k <= prev_d and current_stoch_k > current_stoch_d:
                        signals.append({"type": "stochastic_oversold", "severity": "opportunity", "symbol": symbol,
                            "message": f"{symbol} Stochastic shows oversold conditions with a bullish crossover.",
                            "explanation": "The Stochastic is below 20 (oversold) and %K crossed above %D — bullish reversal signal.",
                            "learn_concept": "stochastic"})

            if sr["supports"]:
                nearest_support = sr["supports"][0]
                dist_to_support = (current_price - nearest_support["price"]) / current_price * 100
                if 0 < dist_to_support < 3:
                    signals.append({"type": "near_support", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} is near support at {nearest_support['price']:.2f} EGP (tested {nearest_support['strength']} times).",
                        "explanation": "Support levels are prices where the stock has historically bounced.",
                        "learn_concept": "support_resistance"})
                elif dist_to_support < 0:
                    next_support = sr["supports"][1]["price"] if len(sr["supports"]) > 1 else None
                    msg = f"{symbol} broke below support at {nearest_support['price']:.2f} EGP."
                    if next_support:
                        msg += f" Next support at {next_support:.2f} EGP."
                    signals.append({"type": "support_broken", "severity": "action_required", "symbol": symbol,
                        "message": msg,
                        "explanation": "When a stock breaks below a support level, it often continues falling.",
                        "learn_concept": "support_resistance"})

            if sr["resistances"]:
                nearest_resistance = sr["resistances"][0]
                dist_to_resistance = (nearest_resistance["price"] - current_price) / current_price * 100
                if 0 < dist_to_resistance < 3:
                    signals.append({"type": "near_resistance", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} approaching resistance at {nearest_resistance['price']:.2f} EGP.",
                        "explanation": "Resistance levels are prices where the stock has historically been rejected.",
                        "learn_concept": "support_resistance"})

            if beta is not None:
                if beta > 1.3:
                    signals.append({"type": "high_beta", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} is highly volatile (beta {beta:.2f}).",
                        "explanation": "High-beta stocks amplify market moves.",
                        "learn_concept": "beta"})
                elif beta < 0.8:
                    signals.append({"type": "low_beta", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} is defensive (beta {beta:.2f}).",
                        "explanation": "Low-beta stocks are less volatile.",
                        "learn_concept": "beta"})

            if current_atr and current_price > 0:
                signals.append({"type": "atr_stop", "severity": "info", "symbol": symbol,
                    "message": f"{symbol} ATR is {current_atr:.2f} EGP ({atr_pct}% of price). Suggested stop-loss: {current_atr*1.5:.2f}-{current_atr*2:.2f} EGP below entry.",
                    "explanation": "ATR measures typical daily price movement. Set stops 1.5-2x ATR below entry.",
                    "learn_concept": "atr"})

            if dist_to_target is not None and 0 < dist_to_target < 10:
                signals.append({"type": "target_reached", "severity": "opportunity", "symbol": symbol,
                    "message": f"{symbol} is {dist_to_target:.1f}% away from your target price of {target_price:.2f} EGP.",
                    "explanation": "Consider taking partial profits or setting a trailing stop.",
                    "learn_concept": "stop_loss"})

            if dist_to_target is not None and dist_to_target <= 0:
                signals.append({"type": "target_hit", "severity": "opportunity", "symbol": symbol,
                    "message": f"{symbol} has reached your target price! Current: {current_price:.2f} EGP, Target: {target_price:.2f} EGP.",
                    "explanation": "Your stock hit the target you set. Review: set a new target or take profits.",
                    "learn_concept": "stop_loss"})

            if current_rsi is not None:
                if current_rsi > 70:
                    signals.append({"type": "rsi_overbought", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} RSI is at {current_rsi:.0f} (overbought >70).",
                        "explanation": "RSI above 70 means the stock has been rising fast — may be due for a pullback.",
                        "learn_concept": "rsi"})
                elif current_rsi < 30:
                    signals.append({"type": "rsi_oversold", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} RSI is at {current_rsi:.0f} (oversold <30). Could be a buying opportunity.",
                        "explanation": "RSI below 30 means the stock has been falling fast — it might bounce back.",
                        "learn_concept": "rsi"})

            if above_sma is not None and not above_sma:
                signals.append({"type": "below_sma", "severity": "info", "symbol": symbol,
                    "message": f"{symbol} is trading below its 50-day SMA ({current_sma_50:.2f} EGP).",
                    "explanation": "Trading below the 50-day SMA suggests the stock's momentum has weakened.",
                    "learn_concept": "sma"})

            if pnl_pct < BIG_LOSS_PCT:
                signals.append({"type": "big_loss", "severity": "warning", "symbol": symbol,
                    "message": f"Your position in {symbol} has lost {abs(pnl_pct):.1f}%. Review if your original thesis still holds.",
                    "explanation": f"A {abs(BIG_LOSS_PCT)}%+ loss is significant. Ask yourself: has the reason you bought changed?",
                    "learn_concept": "stop_loss"})

            if pnl_pct > PROFIT_TARGET_PCT:
                signals.append({"type": "profit_taking", "severity": "info", "symbol": symbol,
                    "message": f"{symbol} has gained {pnl_pct:.1f}%. Consider taking partial profits.",
                    "explanation": "Taking partial profits lets you secure gains while keeping upside exposure.",
                    "learn_concept": "stop_loss"})

            # --- New signals from the 8-category engine ---

            # Cash underperformer: annualized return < risk-free AND held >90 days
            if days_held > 90 and ann_return < risk_free_rate_pct:
                signals.append({"type": "cash_underperformer", "severity": "warning", "symbol": symbol,
                    "message": f"{symbol} has returned {ann_return:.0f}% annualized — less than the {risk_free_rate_pct:.0f}% T-bill rate.",
                    "explanation": "Holding this stock is earning you less than risk-free cash. Either your thesis needs to play out soon, or capital is better placed in T-bills.",
                    "learn_concept": "risk_adjusted_return"})

            # Relative strength — leader or laggard vs EGX30 over 30 days
            if rs_h is not None and rs_h.get("alpha_pct") is not None:
                alpha = rs_h["alpha_pct"]
                if rs_h.get("leader"):
                    signals.append({"type": "relative_strength_leader", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} is outperforming EGX30 by {alpha:+.1f}% over 30 days — a market leader.",
                        "explanation": "Stocks that lead the index tend to keep leading in the short-term. Institutional money is favouring this name.",
                        "learn_concept": "relative_strength"})
                elif rs_h.get("laggard"):
                    signals.append({"type": "relative_strength_laggard", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} is lagging EGX30 by {abs(alpha):.1f}% over 30 days.",
                        "explanation": "Persistent laggards drag down a portfolio. Consider switching to a leader unless your thesis is long-term and patient.",
                        "learn_concept": "relative_strength"})

            # MFI extremes
            if current_mfi is not None:
                if current_mfi < 20:
                    signals.append({"type": "mfi_extreme", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} MFI is {current_mfi:.0f} — money has fled (possible bounce).",
                        "explanation": "MFI is RSI weighted by volume. Below 20 means selling exhaustion; historically a reversal zone (not guaranteed).",
                        "learn_concept": "mfi"})
                elif current_mfi > 80:
                    signals.append({"type": "mfi_extreme", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} MFI is {current_mfi:.0f} — heavy buying may be exhausted.",
                        "explanation": "MFI above 80 often marks short-term tops as the volume-backed rally runs out of buyers.",
                        "learn_concept": "mfi"})

            # ADX strong-trend info (direction from DI±)
            if current_adx is not None and current_adx > 30 and current_plus_di is not None and current_minus_di is not None:
                direction = "up" if current_plus_di > current_minus_di else "down"
                signals.append({"type": "adx_strong_trend", "severity": "info", "symbol": symbol,
                    "message": f"{symbol} is in a strong {direction}trend (ADX {current_adx:.0f}).",
                    "explanation": "ADX above 30 means the current trend is strong and reliable — trend-following signals carry more weight right now.",
                    "learn_concept": "adx"})

            # Liquidity warning — thin volume
            if liquidity_h and liquidity_h.get("thin"):
                signals.append({"type": "low_liquidity_warning", "severity": "warning", "symbol": symbol,
                    "message": f"{symbol} trades on thin volume (avg {liquidity_h['avg_volume']:,} shares/day).",
                    "explanation": "Thin liquidity means wider bid/ask spreads and difficulty exiting the position quickly. A beginner should keep position sizes small here.",
                    "learn_concept": "liquidity"})

        except Exception as e:
            stock_analyses.append({"symbol": symbol, "error": f"Analysis failed: {str(e)}"})

    # Portfolio-level metrics
    total_portfolio_value = total_current_value + cash
    total_pnl = total_current_value - total_invested
    total_pnl_pct = (total_current_value / total_invested - 1) * 100 if total_invested > 0 else 0

    sector_allocation = {}
    for sec, val in sector_values.items():
        sector_allocation[sec] = round(val / total_portfolio_value * 100, 1) if total_portfolio_value > 0 else 0

    stock_concentration = {}
    for sym, val in stock_values.items():
        stock_concentration[sym] = round(val / total_portfolio_value * 100, 1) if total_portfolio_value > 0 else 0

    div_score = 100
    for sym, pct in stock_concentration.items():
        if pct > CONCENTRATION_WARNING_PCT:
            div_score -= (pct - CONCENTRATION_WARNING_PCT) * 2
    for sec, pct in sector_allocation.items():
        if pct > CONCENTRATION_CRITICAL_PCT:
            div_score -= (pct - CONCENTRATION_CRITICAL_PCT) * 1.5
    div_score = max(0, min(100, div_score))

    for sec, pct in sector_allocation.items():
        if pct > SECTOR_ALERT_PCT:
            signals.append({"type": "sector_concentration", "severity": "warning", "symbol": None,
                "message": f"{pct:.0f}% of your portfolio is in {sec}. Consider diversifying.",
                "explanation": "Sector concentration risk means if something bad happens to one industry, a large chunk of your portfolio suffers.",
                "learn_concept": "correlation"})

    for sym, pct in stock_concentration.items():
        if pct > STOCK_ALERT_PCT:
            signals.append({"type": "stock_concentration", "severity": "warning", "symbol": sym,
                "message": f"{sym} makes up {pct:.0f}% of your portfolio.",
                "explanation": "Having more than 30-35% in a single stock is risky.",
                "learn_concept": "correlation"})

    total_weight = sum(stock_values.values())
    weighted_rsi = 0
    rsi_count = 0
    for sa in stock_analyses:
        if "rsi" in sa and sa.get("rsi") is not None and sa["symbol"] in stock_values:
            w = stock_values[sa["symbol"]] / total_weight if total_weight > 0 else 0
            weighted_rsi += sa["rsi"] * w
            rsi_count += 1

    import numpy as np

    sharpe_ratio = sortino_ratio = max_drawdown_info = None
    var_95_pct = var_95_egp = cvar_95_pct = cvar_95_egp = None
    correlation_data = monte_carlo_data = avg_correlation = None

    valid_returns = {sym: rets.dropna() for sym, rets in all_returns.items() if len(rets.dropna()) >= 20}

    if valid_returns and total_current_value > 0:
        returns_df = pd.DataFrame(valid_returns).dropna()

        if len(returns_df) >= 20:
            port_weights = {sym: stock_values.get(sym, 0) / total_current_value for sym in returns_df.columns}
            portfolio_daily = sum(returns_df[sym] * port_weights.get(sym, 0) for sym in returns_df.columns)

            rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
            excess = portfolio_daily - rf_daily

            if excess.std() > 0:
                sharpe_ratio = round(float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 2)

            downside = excess[excess < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino_ratio = round(float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 2)

            cumulative = (1 + portfolio_daily).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_dd = float(drawdown.min())
            max_dd_idx = drawdown.idxmin()
            peak_idx = cumulative[:max_dd_idx].idxmax() if max_dd_idx is not None else None
            current_dd = float(drawdown.iloc[-1])
            max_drawdown_info = {
                "value": round(max_dd, 4),
                "peak_date": str(peak_idx)[:10] if peak_idx is not None else None,
                "trough_date": str(max_dd_idx)[:10] if max_dd_idx is not None else None,
                "current_drawdown": round(current_dd, 4),
            }

            var_pct = float(np.percentile(portfolio_daily.values, VAR_PERCENTILE))
            var_95_pct = round(var_pct, 4)
            var_95_egp = round(total_portfolio_value * abs(var_pct), 0)
            tail = portfolio_daily[portfolio_daily <= var_pct]
            if len(tail) > 0:
                cvar_pct_val = float(tail.mean())
                cvar_95_pct = round(cvar_pct_val, 4)
                cvar_95_egp = round(total_portfolio_value * abs(cvar_pct_val), 0)

            if len(returns_df.columns) >= 2:
                corr = returns_df.corr()
                correlation_data = {
                    "symbols": list(corr.columns),
                    "matrix": [[round(float(corr.iloc[i, j]), 2) for j in range(len(corr.columns))] for i in range(len(corr.columns))],
                }
                corr_values = []
                symbols_list = list(corr.columns)
                for i in range(len(symbols_list)):
                    for j in range(i + 1, len(symbols_list)):
                        c = float(corr.iloc[i, j])
                        corr_values.append(c)
                        if c > CORRELATION_HIGH_THRESHOLD:
                            signals.append({"type": "high_correlation", "severity": "warning", "symbol": None,
                                "message": f"{symbols_list[i]} and {symbols_list[j]} are highly correlated ({c:.2f}).",
                                "explanation": "High correlation means these stocks rise and fall together.",
                                "learn_concept": "correlation"})
                        elif c < CORRELATION_NEGATIVE_THRESHOLD:
                            signals.append({"type": "negative_correlation", "severity": "info", "symbol": None,
                                "message": f"{symbols_list[i]} and {symbols_list[j]} have negative correlation ({c:.2f}). Great for diversification.",
                                "explanation": "Negatively correlated stocks tend to move in opposite directions.",
                                "learn_concept": "correlation"})
                avg_correlation = round(float(np.mean(corr_values)), 2) if corr_values else None

            mu = float(portfolio_daily.mean())
            sigma = float(portfolio_daily.std())
            if sigma > 0:
                n_sims, n_days = MONTE_CARLO_SIMULATIONS, MONTE_CARLO_FORECAST_DAYS
                sims = np.random.normal(mu, sigma, (n_sims, n_days))
                paths = np.cumprod(1 + sims, axis=1)
                p5  = np.percentile(paths, 5,  axis=0)
                p25 = np.percentile(paths, 25, axis=0)
                p50 = np.percentile(paths, 50, axis=0)
                p75 = np.percentile(paths, 75, axis=0)
                p95 = np.percentile(paths, 95, axis=0)
                final_values = paths[:, -1]
                prob_loss = float(np.mean(final_values < 1.0))
                monte_carlo_data = {
                    "days": n_days,
                    "initial_value": round(total_portfolio_value, 0),
                    "probability_of_loss": round(prob_loss, 2),
                    "worst_case_5pct":   round(float(np.percentile(final_values, 5)),  4),
                    "pessimistic_25pct": round(float(np.percentile(final_values, 25)), 4),
                    "median":            round(float(np.percentile(final_values, 50)), 4),
                    "optimistic_75pct":  round(float(np.percentile(final_values, 75)), 4),
                    "best_case_95pct":   round(float(np.percentile(final_values, 95)), 4),
                    "percentiles": {
                        "p5":  [round(float(v), 4) for v in p5],
                        "p25": [round(float(v), 4) for v in p25],
                        "p50": [round(float(v), 4) for v in p50],
                        "p75": [round(float(v), 4) for v in p75],
                        "p95": [round(float(v), 4) for v in p95],
                    },
                }

            if sharpe_ratio is not None and sharpe_ratio < 0:
                signals.append({"type": "negative_sharpe", "severity": "action_required", "symbol": None,
                    "message": f"Your portfolio's Sharpe ratio is {sharpe_ratio:.2f}. You're earning LESS than the risk-free rate (~{risk_free_annual*100:.0f}%).",
                    "explanation": f"With Egypt's T-bill rate at ~{risk_free_annual*100:.0f}%, you could earn guaranteed returns with zero risk.",
                    "learn_concept": "sharpe_ratio"})

            if max_drawdown_info and max_drawdown_info["value"] < -MAX_DRAWDOWN_WARNING_PCT:
                signals.append({"type": "severe_drawdown", "severity": "action_required", "symbol": None,
                    "message": f"Your portfolio's max drawdown has been {max_drawdown_info['value']*100:.1f}%.",
                    "explanation": f"A drawdown over {int(MAX_DRAWDOWN_WARNING_PCT*100)}% means your portfolio lost more than that share of its value at some point.",
                    "learn_concept": "max_drawdown"})

            if max_drawdown_info and max_drawdown_info["current_drawdown"] < -CURRENT_DRAWDOWN_WARNING_PCT:
                signals.append({"type": "current_drawdown", "severity": "warning", "symbol": None,
                    "message": f"Your portfolio is currently in a {abs(max_drawdown_info['current_drawdown'])*100:.1f}% drawdown from its peak.",
                    "explanation": "Your portfolio value is below its recent high.",
                    "learn_concept": "max_drawdown"})

    # macro_data was fetched at the top of _analyze (see earlier block)
    if macro_data:
        egx30 = macro_data.get("egx30", {})
        if egx30.get("value"):
            trend_word = egx30.get("trend", "sideways")
            monthly = egx30.get("monthly_change_pct")
            msg = f"EGX30 is at {egx30['value']:,.0f}"
            if monthly is not None:
                msg += f" ({monthly:+.1f}% this month)"
            msg += f". The overall market is {trend_word}."
            signals.append({"type": "macro_egx30", "severity": "info", "symbol": None,
                "message": msg,
                "explanation": "The EGX30 index reflects the overall market direction.",
                "learn_concept": "egx30_benchmark"})

    severity_order = {"action_required": 0, "warning": 1, "opportunity": 2, "info": 3}
    signals.sort(key=lambda s: severity_order.get(s["severity"], 4))

    return {
        "holdings": stock_analyses,
        "portfolio_metrics": {
            "total_value": round(total_portfolio_value, 2),
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current_value, 2),
            "cash_available": round(cash, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "sector_allocation": sector_allocation,
            "stock_concentration": stock_concentration,
            "diversification_score": round(div_score, 0),
            "weighted_rsi": round(weighted_rsi, 1) if rsi_count > 0 else None,
            "num_holdings": len(holdings),
            "avg_composite_score": (
                round(sum(composite_scores_collected) / len(composite_scores_collected), 1)
                if composite_scores_collected else None
            ),
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown_info,
            "var_95_pct": var_95_pct,
            "var_95_egp": var_95_egp,
            "cvar_95_pct": cvar_95_pct,
            "cvar_95_egp": cvar_95_egp,
            "avg_correlation": avg_correlation,
        },
        "correlation_matrix": correlation_data,
        "monte_carlo": monte_carlo_data,
        "macro": macro_data,
        "signals": signals,
        "disclaimer": "This is educational analysis for learning purposes only, not financial advice.",
    }


@router.get("/api/portfolio_analysis")
def get_portfolio_analysis():
    try:
        db = get_db()
        rows = db.execute(
            "SELECT id, symbol, name, buy_price, buy_date, quantity, notes, sector, "
            "target_price, stop_loss, created_at, updated_at FROM portfolio"
        ).fetchall()
        holdings = [
            {
                "id": r[0], "symbol": r[1], "name": r[2], "buy_price": r[3],
                "buy_date": r[4], "quantity": r[5], "notes": r[6], "sector": r[7],
                "target_price": r[8], "stop_loss": r[9],
            }
            for r in rows
        ]
        return _analyze(holdings, 0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/api/portfolio_analysis")
def post_portfolio_analysis(body: dict):
    try:
        holdings = body.get("portfolio", [])
        return _analyze(holdings, 0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
