"""
/api/portfolio_analysis — Analyze a portfolio of stock holdings.

GET  — Read the current user's holdings from Postgres and analyze them
POST — Accept holdings in request body (body: {portfolio: [...]})
"""

import zlib
from datetime import date
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import CurrentUser, get_current_user
from app.core.db import get_db
from app.core.dividends import fetch_dividend_totals
from app.core.holdings import fetch_open_holdings
from app.core.macro_fetch import fetch_macro
from app.core.composite import compute_composite, get_weights_from_db, DEFAULT_WEIGHTS
from app.core.levels import compute_key_levels, compute_entry_exit
from app.core.extras_builder import build_composite_extras
from app.core.index_membership import get_index_membership
from app.core.pe_fetch import get_pe_for_symbol
from app.core.returns import (  # noqa: F401  (MIN_DAYS_FOR_ANNUALIZATION re-exported)
    MIN_DAYS_FOR_ANNUALIZATION,
    annualized_return as _annualized_return,
    days_between as _days_between,
)
from app.core.constants import (
    BIG_LOSS_PCT,
    CORRELATION_HIGH_THRESHOLD,
    CORRELATION_NEGATIVE_THRESHOLD,
    CURRENT_DRAWDOWN_WARNING_PCT,
    DEFAULT_RISK_FREE_RATE_PCT,
    DIVERGENCE_LOOKBACK_FULL,
    INTERNAL_BARS_MIN,
    MAX_DRAWDOWN_WARNING_PCT,
    MONTE_CARLO_FORECAST_DAYS,
    MONTE_CARLO_SIMULATIONS,
    PROFIT_TARGET_PCT,
    SCORE_BUY_MAX,
    SCORE_STRONG_SELL_MAX,
    SECTOR_ALERT_PCT,
    STOCK_ALERT_PCT,
    STOP_LOSS_ATR_MULTIPLIER,
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
)

router = APIRouter()


def _analyze(holdings, user_id: str = None):
    if not holdings:
        raise HTTPException(status_code=400, detail="No holdings provided")

    from app.vendor.egxpy import get_OHLCV_data
    from app.core.cache import get as cache_get, set as cache_set, make_key
    import pandas as pd
    import numpy as np

    today = date.today()
    stock_analyses = []
    signals = []
    total_invested = 0
    total_current_value = 0

    # Dividends are symbol-anchored, so this is one indexed aggregate with no
    # price fetch — it costs nothing against the 30 s budget. It must key off
    # user_id and NOT off `holdings`, because POST /api/portfolio_analysis
    # takes holdings from the request body.
    dividends_by_symbol = fetch_dividend_totals(get_db(), user_id)

    # Nothing stops two portfolio rows sharing a symbol, and a dividend belongs
    # to the SYMBOL, not to one purchase lot. When that happens the UI labels
    # the figure as the symbol's total rather than the row's own — splitting it
    # by today's share count would be fiction, since the counts differed when
    # the dividend was paid.
    _symbol_counts: dict = {}
    for _h in holdings:
        _sym = (_h.get("symbol") or "").upper()
        _symbol_counts[_sym] = _symbol_counts.get(_sym, 0) + 1

    sector_values = {}
    stock_values = {}
    all_returns = {}
    # Holdings we could not price. Their cost basis must NOT land in
    # total_invested — counting cost with no matching value fabricates a loss.
    excluded_holdings = []

    egx30_returns = None
    egx30_close = None
    egx30_df = None
    try:
        # Same window and cache key as /api/analysis so beta and relative
        # strength are computed against an identical benchmark series on
        # every page. A shorter window here gave the same stock a different
        # beta on the portfolio page than on its detail page.
        egx30_cache_key = make_key("egx30", "EGX", "Daily", INTERNAL_BARS_MIN)
        egx30_df = cache_get(egx30_cache_key)
        if egx30_df is None:
            egx30_raw = get_OHLCV_data("EGX30", "EGX", "Daily", INTERNAL_BARS_MIN)
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
        weights = get_weights_from_db(get_db(), user_id)
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
        # An empty-string sector renders as "45% of your portfolio is in ."
        sector = (h.get("sector") or "Unknown").strip() or "Unknown"

        invested = buy_price * quantity
        counted_in_totals = False

        try:
            days_held = _days_between(buy_date, today)
            # Fixed window (not a function of days_held): a shorter fetch left
            # SMA200 with a single valid point, so golden/death crosses could
            # never be detected for a recently-bought holding, and made beta
            # depend on how long the user had owned the stock.
            df = get_OHLCV_data(symbol, "EGX", "Daily", INTERNAL_BARS_MIN)
            if df is None or df.empty:
                # `id` travels with the error row on purpose: the spec promises
                # a sale can be recorded even when the price feed is down, and
                # the Sell button on the error row needs the holding id.
                stock_analyses.append({
                    "id": h.get("id"),
                    "symbol": symbol,
                    "error": "Could not fetch market data",
                    "dividends_collected": dividends_by_symbol.get(
                        (h.get("symbol") or "").upper(), 0.0
                    ),
                    "dividends_symbol_shared": _symbol_counts.get(
                        (h.get("symbol") or "").upper(), 0
                    ) > 1,
                })
                excluded_holdings.append({
                    "symbol": symbol,
                    "invested": round(invested, 2),
                    "error": "Could not fetch market data",
                })
                continue

            df.columns = [c.lower() for c in df.columns]
            close = df["close"]
            current_price = float(close.iloc[-1])

            current_value = current_price * quantity
            total_current_value += current_value
            # Cost basis is only counted once we have a matching market value.
            total_invested += invested
            counted_in_totals = True
            pnl = (current_price - buy_price) * quantity
            pnl_pct = (current_price / buy_price - 1) * 100 if buy_price > 0 else 0
            ann_return = _annualized_return(pnl_pct, days_held)

            rsi_series = calc_rsi(close, 14)
            current_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None

            sma_50 = calc_sma(close, 50)
            current_sma_50 = float(sma_50.iloc[-1]) if not np.isnan(sma_50.iloc[-1]) else None
            above_sma = current_price > current_sma_50 if current_sma_50 is not None else None

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
            atr_pct = round(current_atr / current_price * 100, 1) if current_atr is not None and current_price > 0 else None

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
            key_levels_h = compute_key_levels(current_price, sr, high=df["high"], low=df["low"])
            entry_exit_h = compute_entry_exit(
                current_price, sr,
                rsi_latest=current_rsi,
                stoch_k_latest=current_stoch_k,
                atr_latest=current_atr,
            )
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

            # Divergences and volume/price confirmation are produced by
            # build_composite_extras below (they are scoring inputs), so they
            # are not computed here — doing both ran the expensive divergence
            # scan twice per holding.
            divergences_h = {"rsi": {}, "macd": {}}
            volume_price_h = None

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

            liquidity_h = None

            pe_info_h = None
            try:
                pe_info_h = get_pe_for_symbol(get_db(), symbol)
            except Exception:
                pe_info_h = None

            # Same builder as /api/analysis (both full and batch paths), so a
            # holding's score here equals its score on the dashboard card and
            # on its own detail page. multi_timeframe is now included: it is
            # derived by resampling the daily frame, so it costs no extra
            # fetch and no longer has to be skipped for the timeout budget.
            try:
                built_h = build_composite_extras(
                    df, holding_indicators,
                    # Portfolio holdings are always analysed on daily bars.
                    interval="Daily",
                    egx30_close=egx30_close,
                    include_multi_timeframe=True,
                    risk_free_rate_pct=risk_free_rate_pct,
                    pe_ratio=pe_info_h.get("pe_ratio") if pe_info_h else None,
                    dividend_yield=pe_info_h.get("dividend_yield") if pe_info_h else None,
                    loss_making=pe_info_h.get("loss_making") if pe_info_h else None,
                    index_membership=get_index_membership(symbol),
                    divergence_lookback=DIVERGENCE_LOOKBACK_FULL,
                )
                divergences_h = built_h["divergences"]
                volume_price_h = built_h["volume_price"]
                rs_h = built_h["extras"].get("relative_strength")
                # From the builder, so the thin-volume warning and the liquidity
                # penalty inside score_volume can never disagree.
                liquidity_h = built_h["liquidity"]
                composite_h = compute_composite(
                    holding_indicators,
                    extras=built_h["extras"],
                    weights=weights,
                    macro=macro_data,
                )
            except Exception:
                composite_h = None
                rs_h = None

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
                "dividends_collected": dividends_by_symbol.get(symbol.upper(), 0.0),
                "dividends_symbol_shared": _symbol_counts.get(symbol.upper(), 0) > 1,
                # None until MIN_DAYS_FOR_ANNUALIZATION — see _annualized_return.
                "annualized_return": round(ann_return, 2) if ann_return is not None else None,
                # `is not None` throughout: a genuinely flat/suspended stock has
                # volatility 0.0 and ATR 0.0, which is real data, not missing data.
                "rsi": round(current_rsi, 1) if current_rsi is not None else None,
                "sma_50": round(current_sma_50, 2) if current_sma_50 is not None else None,
                "above_sma": above_sma,
                "volatility": round(current_vol, 4) if current_vol is not None else None,
                "volume_trend": volume_trend,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "dist_to_target": round(dist_to_target, 2) if dist_to_target is not None else None,
                "dist_to_stop": round(dist_to_stop, 2) if dist_to_stop is not None else None,
                "beta": beta,
                "atr": round(current_atr, 2) if current_atr is not None else None,
                "atr_pct": atr_pct,
                "obv_trend": obv_trend,
                "stochastic_k": round(current_stoch_k, 1) if current_stoch_k is not None else None,
                "stochastic_d": round(current_stoch_d, 1) if current_stoch_d is not None else None,
                "supports": sr["supports"][:3],
                "resistances": sr["resistances"][:3],
                "fibonacci": fib,
                "trend": trend,
                "golden_cross_active": crossovers["current_signal"] == "golden_cross",
                "sma_200": round(current_sma_200, 2) if current_sma_200 is not None else None,
                "adx": round(current_adx, 1) if current_adx is not None else None,
                "plus_di": round(current_plus_di, 1) if current_plus_di is not None else None,
                "minus_di": round(current_minus_di, 1) if current_minus_di is not None else None,
                "mfi": round(current_mfi, 1) if current_mfi is not None else None,
                "divergences": divergences_h,
                "volume_price": volume_price_h,
                "composite_score": composite_h["score"] if composite_h else None,
                "composite_signal": composite_h["signal"] if composite_h else None,
                "composite_breakdown": composite_h["categories"] if composite_h else None,
                "key_levels": key_levels_h,
                "entry_exit": entry_exit_h,
                "pe": pe_info_h,
                "liquidity": liquidity_h,
            }
            stock_analyses.append(analysis)

            if composite_h and composite_h.get("score") is not None:
                # (score, value) so the portfolio average is value-weighted and
                # duplicate lots of one symbol don't count twice.
                composite_scores_collected.append((composite_h["score"], current_value))

            # --- Signals ---
            if composite_h:
                c_score = composite_h["score"]
                c_signal = composite_h["signal"]
                # Band edges must match composite.classify_signal exactly
                # (lower bound inclusive), or a score of exactly 20 gets an
                # "action required: Strong Sell" alert while its own badge
                # in the same payload reads "Sell".
                # These describe CONDITION, and their severity is `info`, not
                # opportunity/action_required. The backtest found the score
                # cannot rank stocks, so an "act now" framing on a condition
                # reading was a promise the evidence does not support — and on
                # the weak side it pointed the wrong way, since low-scored
                # stocks historically bounced.
                if c_score >= SCORE_BUY_MAX:
                    signals.append({"type": "very_strong_composite", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} scores {c_score:.0f} — nearly every category is aligned positively.",
                        "explanation": "The composite blends 8 categories into one number describing the stock's present technical condition. A score ≥80 means most of them agree. It is a description, not a forecast: testing over 2007-2026 found no evidence that high scores precede better returns than low ones.",
                        "learn_concept": "composite_score"})
                elif c_score < SCORE_STRONG_SELL_MAX:
                    signals.append({"type": "very_weak_composite", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} scores {c_score:.0f} — nearly every category is negative.",
                        "explanation": "Below 20, almost every category is reading poorly. Note this is NOT an exit signal: over 2007-2026 the lowest-scoring stocks bounced about as often as the highest-scoring ones, so treat it as a description of current weakness rather than a reason to sell.",
                        "learn_concept": "composite_score"})
                elif c_signal == "Strong":
                    signals.append({"type": "strong_composite", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} scores {c_score:.0f} — most indicators are reading positively.",
                        "explanation": "Multiple categories currently describe this stock favourably. That is a statement about its condition today, not about where it goes next.",
                        "learn_concept": "composite_score"})
                elif c_signal == "Weak":
                    signals.append({"type": "weak_composite", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} scores {c_score:.0f} — most indicators are reading poorly.",
                        "explanation": "Multiple categories currently describe this stock unfavourably. Check the breakdown for which ones, rather than acting on the number itself.",
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

            # dist_to_stop = (stop / price - 1) * 100, so it turns POSITIVE
            # once price falls below the stop. Without a separate branch an
            # already-breached stop reported as "X% away from your stop" —
            # indistinguishable from a stop still X% below the price.
            if dist_to_stop is not None and dist_to_stop >= 0:
                signals.append({"type": "stop_breached", "severity": "action_required", "symbol": symbol,
                    "message": f"{symbol} is trading {dist_to_stop:.1f}% BELOW your stop-loss at {stop_loss:.2f} EGP — your exit rule has triggered.",
                    "explanation": "You set this stop-loss to cap your loss on this position. Price has passed it, so the rule you wrote before buying says to sell. Acting on your own pre-set rule is what stops a small loss becoming a large one.",
                    "learn_concept": "stop_loss"})
            elif dist_to_stop is not None and dist_to_stop > -10:
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

            # Levels come from key_levels_h (proximity-ordered), NOT from
            # sr["supports"][0] — that array is sorted by STRENGTH, so index 0
            # is the most-tested level, which is often nowhere near the price
            # and can even sit above it. Using it emitted "broke below support"
            # alerts citing a level the KeyLevels card showed as still intact.
            ns_sig = key_levels_h.get("nearest_support")
            if ns_sig:
                # distance_pct is signed: negative = support is below price.
                below_by = -ns_sig["distance_pct"]
                if 0 < below_by < 3:
                    signals.append({"type": "near_support", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} is near support at {ns_sig['price']:.2f} EGP (tested {ns_sig['strength']} times).",
                        "explanation": "Support levels are prices where the stock has historically bounced.",
                        "learn_concept": "support_resistance"})
                elif ns_sig["distance_pct"] > 0:
                    # Support sits ABOVE current price = price broke through it.
                    lower = [s for s in sr["supports"]
                             if s.get("price") is not None and s["price"] < current_price]
                    next_support = max(lower, key=lambda s: s["price"])["price"] if lower else None
                    msg = f"{symbol} broke below support at {ns_sig['price']:.2f} EGP."
                    if next_support is not None:
                        msg += f" Next support at {next_support:.2f} EGP."
                    signals.append({"type": "support_broken", "severity": "action_required", "symbol": symbol,
                        "message": msg,
                        "explanation": "When a stock breaks below a support level, it often continues falling.",
                        "learn_concept": "support_resistance"})

            # Nothing overhead at all: the stock is at/near new highs and has
            # no prior level to push back against it. The card used to bury
            # this by naming a long-cleared level as "nearest resistance".
            if key_levels_h.get("clear_air_above"):
                signals.append({"type": "clear_air_above", "severity": "opportunity", "symbol": symbol,
                    "message": f"{symbol} has no resistance overhead — it is trading above every level in its recent history.",
                    "explanation": "Nothing in the past year of trading sits above this price to act as a ceiling, so there is no obvious level where sellers have previously stepped in. That removes a headwind, but it also means there is no chart-based profit target above — set your target from your own plan, and trail your stop up rather than waiting for a level that isn't there.",
                    "learn_concept": "support_resistance"})

            nr_sig = key_levels_h.get("nearest_resistance")
            if nr_sig:
                # distance_pct is signed: positive = resistance is above price.
                above_by = nr_sig["distance_pct"]
                if 0 < above_by < 3:
                    signals.append({"type": "near_resistance", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} approaching resistance at {nr_sig['price']:.2f} EGP.",
                        "explanation": "Resistance levels are prices where the stock has historically been rejected.",
                        "learn_concept": "support_resistance"})
                elif above_by < 0:
                    # Price cleared the level — the bullish mirror of a support
                    # break, which previously had no branch at all and so was
                    # silently never reported.
                    higher = [r for r in sr["resistances"]
                              if r.get("price") is not None and r["price"] > current_price]
                    next_resistance = min(higher, key=lambda r: r["price"])["price"] if higher else None
                    msg = f"{symbol} broke above resistance at {nr_sig['price']:.2f} EGP."
                    if next_resistance is not None:
                        msg += f" Next resistance at {next_resistance:.2f} EGP."
                    signals.append({"type": "resistance_broken", "severity": "opportunity", "symbol": symbol,
                        "message": msg,
                        "explanation": "Breaking above a level that previously rejected the stock often starts a new leg up. Old resistance frequently becomes new support — consider trailing your stop-loss up to just below it.",
                        "learn_concept": "support_resistance"})

            # Entry/exit zones add momentum confirmation on top of raw level
            # proximity — "price near support AND RSI not overbought" is a
            # stronger buy-zone cue than "price near support" alone.
            ez = entry_exit_h.get("entry_zone") if entry_exit_h else None
            if ez and ez.get("active"):
                pr = ez.get("price_range") or {}
                sl = ez.get("suggested_stop_loss")
                conf = ez.get("confidence") or "low"
                # High-confidence buy-zones are the rare confluence cue we
                # want to surface aggressively; low-confidence zones are
                # hints, not calls to action.
                #
                # This line used to read `sev = "opportunity"` unconditionally,
                # which contradicted the comment directly above it. A `low`
                # zone is the LEFTOVER bucket — nothing disqualified it and
                # nothing confirmed it, typically an untested support with
                # unremarkable RSI — and it was being rendered at the loudest
                # non-alert tier the panel has. The exit side two branches down
                # had always graded correctly; only entry didn't.
                sev = "opportunity" if conf in ("high", "medium") else "info"
                prefix = {"high": "HIGH-CONFIDENCE ", "medium": "", "low": "Possible "}[conf]
                msg = (
                    f"{prefix}entry zone on {symbol}: buy band "
                    f"{pr.get('low', 0):.2f}–{pr.get('high', 0):.2f} EGP"
                )
                if sl is not None:
                    msg += f", suggested stop-loss {sl:.2f}"
                msg += f". Confidence: {conf}."
                signals.append({"type": "entry_zone_active", "severity": sev, "symbol": symbol,
                    "message": msg,
                    "explanation": "Entry zones combine a tested support level with non-overbought momentum — a beginner-friendly way to time a buy. Always size your position and set a stop-loss before entering.",
                    "learn_concept": "entry_exit_zones"})

            xz = entry_exit_h.get("exit_zone") if entry_exit_h else None
            if xz and xz.get("active"):
                pr = xz.get("price_range") or {}
                conf = xz.get("confidence") or "low"
                sev = "warning" if conf in ("high", "medium") else "info"
                prefix = {"high": "STRONG ", "medium": "", "low": "Possible "}[conf]
                msg = (
                    f"{prefix}exit zone on {symbol}: trim band "
                    f"{pr.get('low', 0):.2f}–{pr.get('high', 0):.2f} EGP. Confidence: {conf}."
                )
                signals.append({"type": "exit_zone_active", "severity": sev, "symbol": symbol,
                    "message": msg,
                    "explanation": "Exit zones combine resistance with overbought momentum — a cue to trim, take partial profits, or tighten your stop-loss. Not always a full sell, especially in strong uptrends that break through resistance.",
                    "learn_concept": "entry_exit_zones"})

            if beta is not None:
                # Magnitude drives volatility, not the raw value: a beta of
                # -2.10 is twice as volatile as the index, not "defensive".
                if abs(beta) > 1.3:
                    signals.append({"type": "high_beta", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} is highly volatile (beta {beta:.2f}).",
                        "explanation": "High-beta stocks amplify market moves.",
                        "learn_concept": "beta"})
                elif beta < 0:
                    signals.append({"type": "inverse_beta", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} tends to move AGAINST the market (beta {beta:.2f}).",
                        "explanation": "A negative beta means the stock has historically risen when EGX30 fell, and vice versa. Small negative betas are often just noise, but a consistent one makes the stock a useful diversifier.",
                        "learn_concept": "beta"})
                elif beta < 0.8:
                    signals.append({"type": "low_beta", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} is defensive (beta {beta:.2f}).",
                        "explanation": "Low-beta stocks are less volatile.",
                        "learn_concept": "beta"})

            if current_atr is not None and current_atr > 0 and current_price > 0:
                # One stop-loss convention app-wide: 1.5x ATR below the nearest
                # support (see constants.STOP_LOSS_ATR_MULTIPLIER). Quote an
                # actual price — the old message printed ATR *distances*
                # formatted like prices ("stop-loss: 1.80-2.40 EGP").
                if ns_sig:
                    suggested_stop = ns_sig["price"] - STOP_LOSS_ATR_MULTIPLIER * current_atr
                    anchor = f"({STOP_LOSS_ATR_MULTIPLIER:g}x ATR below support at {ns_sig['price']:.2f})"
                else:
                    suggested_stop = current_price - STOP_LOSS_ATR_MULTIPLIER * current_atr
                    anchor = f"({STOP_LOSS_ATR_MULTIPLIER:g}x ATR below the current price — no clear support level found)"
                signals.append({"type": "atr_stop", "severity": "info", "symbol": symbol,
                    "message": f"{symbol} ATR is {current_atr:.2f} EGP ({atr_pct}% of price). Suggested stop-loss: {suggested_stop:.2f} EGP {anchor}.",
                    "explanation": f"ATR measures typical daily price movement. Placing your stop {STOP_LOSS_ATR_MULTIPLIER:g}x ATR below the nearest support keeps normal daily noise from stopping you out while still capping the loss.",
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

            # Cash underperformer: annualized return < risk-free AND held >90 days.
            # Says "your position in X" because ann_return annualizes the
            # user's own buy price, not the stock's market return — the
            # Risk-Adjusted category uses the latter and the two can differ.
            if days_held > 90 and ann_return is not None and ann_return < risk_free_rate_pct:
                signals.append({"type": "cash_underperformer", "severity": "warning", "symbol": symbol,
                    "message": f"Your position in {symbol} has returned {ann_return:.0f}% annualized since you bought it — less than the {risk_free_rate_pct:.0f}% T-bill rate.",
                    "explanation": "Holding this stock has earned you less than risk-free cash would have. Either your thesis needs to play out soon, or capital is better placed in T-bills. (This measures your purchase, not the stock's own 12-month performance — the Risk-Adjusted score covers that.)",
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

            # P/E signals — only fired when the EGX P/E scrape has a row for the symbol.
            # Loss-making comes from diluted EPS: the fundamentals feed reports
            # a NULL P/E for loss-makers, never a negative one, so the old
            # `pe < 0` branch could never fire.
            if pe_info_h and pe_info_h.get("loss_making"):
                signals.append({"type": "pe_loss_making", "severity": "warning", "symbol": symbol,
                    "message": f"{symbol} is loss-making over the last twelve months.",
                    "explanation": "The company isn't profitable right now, so earnings-based valuation doesn't apply — lean on trend, relative strength, and macro instead.",
                    "learn_concept": "pe_ratio"})

            # Bands match score_quality's, which are centred on the EGX median
            # P/E of ~12 rather than a developed-market idea of cheap.
            if pe_info_h and pe_info_h.get("pe_ratio") is not None:
                pe = float(pe_info_h["pe_ratio"])
                if pe < 3:
                    signals.append({"type": "pe_implausibly_low", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} P/E is {pe:.1f} — implausibly low, not a bargain.",
                        "explanation": "A P/E under 3 on the EGX almost always means the earnings were one-off, or the share price has already collapsed. Check the last results before treating this as cheap.",
                        "learn_concept": "pe_ratio"})
                elif pe < 8:
                    signals.append({"type": "pe_undervalued", "severity": "opportunity", "symbol": symbol,
                        "message": f"{symbol} P/E is {pe:.1f} — cheap versus the EGX median of ~12.",
                        "explanation": "A low P/E can indicate value, but it can also mean the market expects earnings to fall. Always combine with trend and relative strength before acting.",
                        "learn_concept": "pe_ratio"})
                elif pe >= 25:
                    signals.append({"type": "pe_overvalued", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} P/E is {pe:.1f} — expensive versus the EGX median of ~12.",
                        "explanation": "A high P/E means the market is paying a lot for each EGP of current earnings. Only justified by strong, confirmed growth expectations.",
                        "learn_concept": "pe_ratio"})

            # Dividend yield. Framed as evidence of cash generation, never as
            # income — no EGX yield competes with a ~25% T-bill.
            if pe_info_h and pe_info_h.get("dividend_yield") is not None:
                dy = float(pe_info_h["dividend_yield"])
                if dy >= 15:
                    signals.append({"type": "dividend_yield_extreme", "severity": "warning", "symbol": symbol,
                        "message": f"{symbol} yields {dy:.1f}% — check before counting on it.",
                        "explanation": "A yield this high is usually a one-off special dividend, or it only looks high because the share price collapsed. Confirm the payout recurs before treating it as a reason to hold.",
                        "learn_concept": "dividend_yield"})
                elif dy >= 4:
                    signals.append({"type": "dividend_yield_solid", "severity": "info", "symbol": symbol,
                        "message": f"{symbol} yields {dy:.1f}% — an above-median EGX payer.",
                        "explanation": "Read this as evidence the company generates real cash and returns it, not as income: even this yield loses to the ~25% T-bill. It is a quality marker, not a reason to buy on its own.",
                        "learn_concept": "dividend_yield"})

        except Exception as e:
            # See above — the id keeps the Sell action reachable on error rows.
            stock_analyses.append({
                "id": h.get("id"),
                "symbol": symbol,
                "error": f"Analysis failed: {str(e)}",
                "dividends_collected": dividends_by_symbol.get(
                    (h.get("symbol") or "").upper(), 0.0
                ),
                "dividends_symbol_shared": _symbol_counts.get(
                    (h.get("symbol") or "").upper(), 0
                ) > 1,
            })
            if not counted_in_totals:
                excluded_holdings.append({
                    "symbol": symbol,
                    "invested": round(invested, 2),
                    "error": f"Analysis failed: {str(e)}",
                })

    # Portfolio-level metrics
    total_portfolio_value = total_current_value
    total_pnl = total_current_value - total_invested
    total_pnl_pct = (total_current_value / total_invested - 1) * 100 if total_invested > 0 else 0

    sector_allocation = {}
    for sec, val in sector_values.items():
        sector_allocation[sec] = round(val / total_portfolio_value * 100, 1) if total_portfolio_value > 0 else 0

    stock_concentration = {}
    for sym, val in stock_values.items():
        stock_concentration[sym] = round(val / total_portfolio_value * 100, 1) if total_portfolio_value > 0 else 0

    # Penalties start at the SAME thresholds that raise the warning signals
    # below. When they differed, a portfolio could score "Diversification
    # 100/100" while an alert right beside it said "45% in Banking".
    div_score = 100
    for sym, pct in stock_concentration.items():
        if pct > STOCK_ALERT_PCT:
            div_score -= (pct - STOCK_ALERT_PCT) * 2
    for sec, pct in sector_allocation.items():
        if pct > SECTOR_ALERT_PCT:
            div_score -= (pct - SECTOR_ALERT_PCT) * 1.5
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
                "explanation": f"Having more than {STOCK_ALERT_PCT}% in a single stock is risky — one company's bad news would move your whole portfolio.",
                "learn_concept": "correlation"})

    # Weight each ROW by its own value and renormalize over the rows actually
    # included. Using stock_values[symbol] double-counted a symbol held in two
    # lots, and skipping a None-RSI holding without renormalizing dragged the
    # portfolio RSI toward 0 (reading as "oversold" on missing data).
    weighted_rsi = None
    rsi_count = 0
    _rsi_num = 0.0
    _rsi_weight = 0.0
    for sa in stock_analyses:
        if sa.get("rsi") is not None and sa.get("current_value") is not None:
            _rsi_num += sa["rsi"] * sa["current_value"]
            _rsi_weight += sa["current_value"]
            rsi_count += 1
    if rsi_count > 0 and _rsi_weight > 0:
        weighted_rsi = _rsi_num / _rsi_weight

    import numpy as np

    sharpe_ratio = sortino_ratio = max_drawdown_info = None
    var_95_pct = var_95_egp = cvar_95_pct = cvar_95_egp = None
    correlation_data = monte_carlo_data = avg_correlation = None

    valid_returns = {sym: rets.dropna() for sym, rets in all_returns.items() if len(rets.dropna()) >= 20}

    if valid_returns and total_current_value > 0:
        returns_df = pd.DataFrame(valid_returns).dropna()

        if len(returns_df) >= 20:
            # Renormalize over the holdings actually present in returns_df.
            # Dividing by total_current_value left the excluded holdings'
            # weight unallocated, so portfolio_daily came out scaled by k<1
            # while rf_daily was not — pushing Sharpe more negative the more
            # holdings were dropped, and understating VaR and drawdown.
            included_value = sum(stock_values.get(sym, 0) for sym in returns_df.columns)
            if included_value > 0:
                port_weights = {sym: stock_values.get(sym, 0) / included_value for sym in returns_df.columns}
            else:
                port_weights = {sym: 1.0 / len(returns_df.columns) for sym in returns_df.columns}
            portfolio_daily = sum(returns_df[sym] * port_weights.get(sym, 0) for sym in returns_df.columns)

            rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS_PER_YEAR) - 1
            excess = portfolio_daily - rf_daily

            if excess.std() > 0:
                sharpe_ratio = round(float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 2)

            # True downside deviation: RMS of the negative part over ALL
            # observations. Taking the std of just the negative subset
            # measures spread within the losses (and divides by their count),
            # which systematically inflates Sortino.
            downside_dev = float(np.sqrt(np.mean(np.minimum(excess.values, 0.0) ** 2)))
            if downside_dev > 0:
                sortino_ratio = round(float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR)), 2)

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
                # Seed from the portfolio composition + today's date. Without a
                # seed (and with no response cache on this endpoint) "Chance of
                # Loss" and the whole cone changed on every page refresh with
                # identical inputs. Same holdings on the same day -> same cone;
                # it moves when the portfolio or the market data moves.
                _seed_src = "|".join(sorted(f"{s}:{v:.2f}" for s, v in stock_values.items()))
                _seed = (zlib.crc32(_seed_src.encode()) ^ zlib.crc32(today.isoformat().encode())) & 0xFFFFFFFF
                rng = np.random.default_rng(_seed)
                # Log-return sampling with geometric drift (mu - sigma^2/2).
                # Feeding an arithmetic mean into a multiplicative cumprod
                # biases the median path upward and understates loss odds.
                log_mu = mu - 0.5 * sigma ** 2
                sims = rng.normal(log_mu, sigma, (n_sims, n_days))
                paths = np.exp(np.cumsum(sims, axis=1))
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

            # Phrased as a simulation, not as history: portfolio_daily applies
            # TODAY's weights across the whole analysis window, including dates
            # before the user owned any of it.
            if max_drawdown_info and max_drawdown_info["value"] < -MAX_DRAWDOWN_WARNING_PCT:
                signals.append({"type": "severe_drawdown", "severity": "action_required", "symbol": None,
                    "message": f"This mix of holdings would have fallen {abs(max_drawdown_info['value'])*100:.1f}% at its worst over the analysis window.",
                    "explanation": f"A drawdown over {int(MAX_DRAWDOWN_WARNING_PCT*100)}% means this combination of stocks, at your current weights, lost more than that share of its value at some point in the past — including dates before you owned them. It measures how bumpy the mix is, not what your account actually did.",
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

    # Tell the user when totals don't cover everything they own, rather than
    # letting an unpriced holding quietly distort the headline numbers.
    if excluded_holdings:
        _syms = ", ".join(e["symbol"] for e in excluded_holdings)
        _excluded_invested = sum(e["invested"] for e in excluded_holdings)
        signals.append({"type": "holdings_excluded", "severity": "warning", "symbol": None,
            "message": f"{len(excluded_holdings)} holding(s) could not be priced ({_syms}) and are left out of every total below.",
            "explanation": f"Market data was unavailable for these, so {_excluded_invested:,.0f} EGP of cost basis is excluded from your P&L, allocation and risk metrics. The figures shown describe the rest of your portfolio. This is usually temporary — try refreshing later.",
            "learn_concept": None})

    severity_order = {"action_required": 0, "warning": 1, "opportunity": 2, "info": 3}
    signals.sort(key=lambda s: severity_order.get(s["severity"], 4))

    return {
        "holdings": stock_analyses,
        "excluded_holdings": excluded_holdings,
        "portfolio_metrics": {
            "total_value": round(total_portfolio_value, 2),
            "total_invested": round(total_invested, 2),
            "excluded_invested": round(sum(e["invested"] for e in excluded_holdings), 2) if excluded_holdings else 0,
            "excluded_count": len(excluded_holdings),
            "total_current_value": round(total_current_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "sector_allocation": sector_allocation,
            "stock_concentration": stock_concentration,
            "diversification_score": round(div_score, 0),
            "weighted_rsi": round(weighted_rsi, 1) if weighted_rsi is not None else None,
            "num_holdings": len(holdings),
            # Value-weighted so a large position counts more than a token one,
            # and a symbol split across two lots isn't counted twice.
            "avg_composite_score": (
                round(sum(s * v for s, v in composite_scores_collected)
                      / sum(v for _, v in composite_scores_collected), 1)
                if composite_scores_collected and sum(v for _, v in composite_scores_collected) > 0
                else None
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
def get_portfolio_analysis(user: CurrentUser = Depends(get_current_user)):
    try:
        db = get_db()
        holdings = fetch_open_holdings(db, user.id)
        return _analyze(holdings, user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/api/portfolio_analysis")
def post_portfolio_analysis(
    body: dict,
    user: CurrentUser = Depends(get_current_user),
):
    try:
        holdings = body.get("portfolio", [])
        return _analyze(holdings, user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
