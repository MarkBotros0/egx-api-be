"""
Technical Analysis Indicators — computed from scratch using pandas/numpy.

Each function takes a DataFrame with OHLCV columns and returns the computed
indicator as a pandas Series. All formulas are annotated with explanations
so you can learn what each indicator measures and why traders use it.

No external TA libraries — everything is built from first principles.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(close: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (SMA)

    What it is:  The average closing price over the last `period` days.
    Why it matters:  Smooths out noise to reveal the underlying trend.
                     When the price is above its SMA the trend is bullish;
                     below it, bearish.
    Common periods: 20 (short-term), 50 (medium), 200 (long-term).

    Formula: SMA = sum(close[i-period+1 : i+1]) / period
    """
    return close.rolling(window=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA)

    What it is:  A weighted moving average that gives MORE weight to recent
                 prices, so it reacts faster than the SMA.
    Why it matters:  Catches trend changes earlier than SMA. Used in MACD
                     calculation and for short-term trading signals.

    Formula: EMA_today = close_today * k + EMA_yesterday * (1 - k)
             where k = 2 / (period + 1)
    """
    return close.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# RSI — Relative Strength Index
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)

    What it is:  A momentum oscillator (0-100) measuring the speed and
                 magnitude of recent price changes.
    Why it matters:
      - RSI > 70 → the stock may be OVERBOUGHT (price rose too fast,
        could pull back)
      - RSI < 30 → the stock may be OVERSOLD (price dropped too much,
        could bounce)
    How traders use it:  Look for RSI divergences (price makes new high
                         but RSI doesn't) as early reversal signals.

    Formula:
      1. delta = close.diff()
      2. gain = max(delta, 0);  loss = max(-delta, 0)
      3. avg_gain = EMA of gains over `period`
      4. avg_loss = EMA of losses over `period`
      5. RS = avg_gain / avg_loss
      6. RSI = 100 - (100 / (1 + RS))
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# MACD — Moving Average Convergence Divergence
# ---------------------------------------------------------------------------

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9):
    """
    MACD (Moving Average Convergence Divergence)

    What it is:  Shows the relationship between two EMAs. When the fast EMA
                 crosses above the slow EMA the trend is turning bullish.
    Components:
      - MACD Line = EMA(fast) - EMA(slow)
      - Signal Line = EMA of the MACD line (smoothed version)
      - Histogram = MACD Line - Signal Line (shows momentum strength)
    Why it matters:
      - MACD crosses above Signal → bullish signal (consider buying)
      - MACD crosses below Signal → bearish signal (consider selling)
      - Histogram growing → momentum is strengthening
      - Histogram shrinking → momentum is fading

    Returns: (macd_line, signal_line, histogram) as pd.Series tuple
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands

    What it is:  A volatility envelope around a moving average.
      - Middle Band = SMA(period)
      - Upper Band  = Middle + num_std × standard_deviation
      - Lower Band  = Middle - num_std × standard_deviation
    Why it matters:
      - When price touches the upper band → possibly overbought
      - When price touches the lower band → possibly oversold
      - Bands squeeze together → low volatility, big move coming
      - Bands expand → high volatility period
    How traders use it:  Look for "Bollinger Squeeze" (bands narrow) as a
                         signal that a breakout is imminent.

    Returns: (upper, middle, lower) as pd.Series tuple
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return upper, middle, lower


# ---------------------------------------------------------------------------
# Returns & Volatility
# ---------------------------------------------------------------------------

def daily_returns(close: pd.Series) -> pd.Series:
    """
    Daily Returns (percentage change)

    What it is:  How much the price changed from yesterday to today, as a %.
    Formula:  return = (price_today - price_yesterday) / price_yesterday
    Why it matters:  The building block for all risk/return analysis.
    """
    return close.pct_change()


def volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Historical Volatility (rolling standard deviation of returns)

    What it is:  Measures how much the price fluctuates. Higher volatility
                 = more risk AND more opportunity.
    Formula:  std_dev of daily returns over a rolling window
    Why it matters:
      - High volatility → bigger price swings, riskier
      - Low volatility → calmer, more predictable
    Traders use it to size positions — invest less in volatile stocks.
    """
    returns = close.pct_change()
    return returns.rolling(window=period).std()


def cumulative_returns(close: pd.Series) -> pd.Series:
    """
    Cumulative Returns

    What it is:  Total return since the first data point, expressed as a
                 fraction (0.15 = 15% gain, -0.10 = 10% loss).
    Formula:  (1 + daily_return).cumprod() - 1
    Why it matters:  Shows the big picture — how much you would have gained
                     or lost if you held the stock from the start.
    """
    returns = close.pct_change()
    return (1 + returns).cumprod() - 1


# ---------------------------------------------------------------------------
# ATR — Average True Range
# ---------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR)

    What it is:  Measures the average daily price range, accounting for gaps.
    Why it matters:  Essential for setting stop-losses. ATR tells you how much
                     a stock typically moves in a day, so you can set stops
                     that won't get triggered by normal noise.
    Formula:
      TR = max(high - low, |high - prev_close|, |low - prev_close|)
      ATR = rolling mean of TR over `period` days
    """
    prev_close = close.shift(1)
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - prev_close).abs(),
        "lc": (low - prev_close).abs(),
    }).max(axis=1)
    return tr.rolling(window=period).mean()


# ---------------------------------------------------------------------------
# OBV — On-Balance Volume
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV)

    What it is:  A cumulative volume indicator. Volume is added on up days
                 and subtracted on down days.
    Why it matters:  OBV confirms price trends. If price is rising but OBV
                     is falling, the rally may be losing steam (divergence).
                     If OBV rises while price is flat, smart money may be
                     accumulating — watch for a breakout.
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    """
    Stochastic Oscillator (%K and %D)

    What it is:  Compares the closing price to the price range over a period.
                 Oscillates between 0 and 100.
    Why it matters:
      - %K > 80 = overbought (may drop)
      - %K < 20 = oversold (may bounce)
      - When %K crosses above %D from below 20 → bullish signal
      - When %K crosses below %D from above 80 → bearish signal

    Formula:
      %K = (close - lowest_low) / (highest_high - lowest_low) * 100
      %D = SMA(%K, d_period)

    Returns: (k_line, d_line) as pd.Series tuple
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = highest_high - lowest_low
    k_line = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    d_line = k_line.rolling(window=d_period).mean()
    return k_line, d_line


# ---------------------------------------------------------------------------
# ADX — Average Directional Index (Trend Strength)
# ---------------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    ADX (Average Directional Index) — measures how STRONG a trend is (0-100).

    What it is:  ADX does NOT tell you direction, only trend strength.
                 The direction comes from +DI and -DI:
                   +DI > -DI  → uptrend
                   -DI > +DI  → downtrend
    Scale:
      ADX < 20  → no trend (choppy/sideways) → trend signals are unreliable
      ADX 20-40 → developing trend → moderately reliable
      ADX > 40  → strong trend → trust trend-following signals
      ADX > 60  → extremely strong trend (rare)

    Formula (Wilder's method):
      +DM = max(high - prev_high, 0) if that exceeds the downward move, else 0
      -DM = max(prev_low - low, 0) if that exceeds the upward move, else 0
      TR  = max(high-low, |high-prev_close|, |low-prev_close|)
      Wilder-smooth each with alpha = 1/period
      +DI = 100 * smoothed(+DM) / smoothed(TR)
      -DI = 100 * smoothed(-DM) / smoothed(TR)
      DX  = 100 * |+DI - -DI| / (+DI + -DI)
      ADX = Wilder-smoothed DX

    Returns: (adx_series, plus_di, minus_di) as pd.Series tuple.
    """
    up_move = high.diff()
    down_move = -low.diff()

    # +DM: only when up_move > down_move AND up_move > 0
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    # -DM: only when down_move > up_move AND down_move > 0
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # True Range (same formula as ATR above)
    prev_close = close.shift(1)
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - prev_close).abs(),
        "lc": (low - prev_close).abs(),
    }).max(axis=1)

    # Wilder's smoothing is equivalent to EMA with alpha = 1/period
    alpha = 1.0 / period
    atr_wilder = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_smooth / atr_wilder.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_smooth / atr_wilder.replace(0, np.nan))

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_series = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx_series, plus_di, minus_di


# ---------------------------------------------------------------------------
# MFI — Money Flow Index (Volume-Weighted RSI)
# ---------------------------------------------------------------------------

def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI) — RSI weighted by volume.

    What it is:  Like RSI but incorporates volume. Tells you whether money is
                 flowing INTO or OUT OF the stock, not just whether price is
                 moving.
    Scale (0-100):
      MFI > 80 → overbought (heavy buying may be exhausted)
      MFI < 20 → oversold (heavy selling may be exhausted)

    Why it's better than RSI alone:
      - RSI says "price went up"; MFI says "price went up AND lots of money
        pushed it". A rise on low MFI is weak (few buyers); a rise on high MFI
        is strong (heavy buying, likely sustained).

    Formula:
      typical_price = (high + low + close) / 3
      raw_money_flow = typical_price * volume
      positive_flow = raw_money_flow on days where TP rose
      negative_flow = raw_money_flow on days where TP fell
      money_flow_ratio = sum(positive, period) / sum(negative, period)
      MFI = 100 - 100 / (1 + money_flow_ratio)
    """
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_diff < 0, 0.0)

    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    # Avoid division by zero: if negative_sum is 0, ratio is effectively infinite → MFI=100
    ratio = positive_sum / negative_sum.replace(0, np.nan)
    mfi_series = 100.0 - (100.0 / (1.0 + ratio))
    # Where negative_sum == 0 but positive_sum > 0, MFI should be 100
    mfi_series = mfi_series.where(~((negative_sum == 0) & (positive_sum > 0)), 100.0)
    return mfi_series


# ---------------------------------------------------------------------------
# Divergence Detection (price vs indicator)
# ---------------------------------------------------------------------------

def detect_divergences(close: pd.Series, indicator: pd.Series,
                       lookback: int = 60, swing_window: int = 5) -> dict:
    """
    Detect bullish / bearish divergences between price and an indicator
    (RSI or MACD line) over the most recent `lookback` bars.

    Regular Bullish divergence: price makes a LOWER low, but indicator makes
        a HIGHER low → selling pressure weakening → potential reversal UP.
    Regular Bearish divergence: price makes a HIGHER high, but indicator makes
        a LOWER high → buying pressure weakening → potential reversal DOWN.
    Hidden Bullish: price HIGHER low + indicator LOWER low → trend continuation up.
    Hidden Bearish: price LOWER high + indicator HIGHER high → trend continuation down.

    A swing high/low is a bar that is the max/min within ±swing_window bars.
    We find the two most recent swing highs and the two most recent swing lows
    in the lookback window, then compare.

    Returns:
        {
          "bullish": bool,
          "bearish": bool,
          "hidden_bullish": bool,
          "hidden_bearish": bool,
          "detail": str | None
        }
    """
    empty = {"bullish": False, "bearish": False,
             "hidden_bullish": False, "hidden_bearish": False, "detail": None}

    if close is None or indicator is None:
        return empty

    c = close.iloc[-lookback:].reset_index(drop=True)
    ind = indicator.iloc[-lookback:].reset_index(drop=True)

    if len(c) < swing_window * 2 + 3 or len(ind) != len(c):
        return empty

    highs = []  # list of indices where a swing high occurred
    lows = []

    # Bar i is a swing high if it's the max in [i-w, i+w] (excluding the edges
    # of the window to avoid false signals from incomplete patterns).
    for i in range(swing_window, len(c) - swing_window):
        window_high = c.iloc[i - swing_window: i + swing_window + 1].max()
        window_low = c.iloc[i - swing_window: i + swing_window + 1].min()
        if c.iloc[i] == window_high:
            highs.append(i)
        if c.iloc[i] == window_low:
            lows.append(i)

    result = dict(empty)

    # Compare two most recent swing highs for bearish / hidden bearish divergence
    if len(highs) >= 2:
        i2, i1 = highs[-1], highs[-2]  # i1 = older, i2 = newer
        price_now, price_prev = float(c.iloc[i2]), float(c.iloc[i1])
        ind_now, ind_prev = float(ind.iloc[i2]), float(ind.iloc[i1])
        if not (np.isnan(ind_now) or np.isnan(ind_prev)):
            if price_now > price_prev and ind_now < ind_prev:
                result["bearish"] = True
                result["detail"] = (
                    f"Price higher high ({price_now:.2f} > {price_prev:.2f}) "
                    f"but indicator lower high ({ind_now:.2f} < {ind_prev:.2f})"
                )
            elif price_now < price_prev and ind_now > ind_prev:
                result["hidden_bearish"] = True

    # Compare two most recent swing lows for bullish / hidden bullish divergence
    if len(lows) >= 2:
        i2, i1 = lows[-1], lows[-2]
        price_now, price_prev = float(c.iloc[i2]), float(c.iloc[i1])
        ind_now, ind_prev = float(ind.iloc[i2]), float(ind.iloc[i1])
        if not (np.isnan(ind_now) or np.isnan(ind_prev)):
            if price_now < price_prev and ind_now > ind_prev:
                result["bullish"] = True
                if result["detail"] is None:
                    result["detail"] = (
                        f"Price lower low ({price_now:.2f} < {price_prev:.2f}) "
                        f"but indicator higher low ({ind_now:.2f} > {ind_prev:.2f})"
                    )
            elif price_now > price_prev and ind_now < ind_prev:
                result["hidden_bullish"] = True

    return result


# ---------------------------------------------------------------------------
# Volume-Price Confirmation
# ---------------------------------------------------------------------------

def volume_price_confirmation(close: pd.Series, volume: pd.Series,
                              lookback: int = 20) -> dict:
    """
    Classify the most recent bar by whether volume CONFIRMS the price move.

    Big price move + big volume  → "confirmed"      (real, likely to hold)
    Big price move + low volume  → "unconfirmed"    (suspicious, may not hold)
    Flat price + heavy volume    → "accumulation"   (someone building a position)
    Flat price + low volume      → "quiet"
    Otherwise                     → "normal"

    Returns {
        "classification": str,   ("confirmed_up"|"confirmed_down"|"unconfirmed_up"|
                                   "unconfirmed_down"|"accumulation"|"quiet"|"normal")
        "price_change_pct": float,   (% change on the last bar)
        "volume_ratio": float,       (latest volume / avg volume over `lookback` bars)
    }
    """
    empty = {"classification": "normal", "price_change_pct": 0.0, "volume_ratio": 0.0}
    if close is None or volume is None or len(close) < lookback + 1:
        return empty

    price_change = close.pct_change().iloc[-1]
    avg_volume = volume.rolling(window=lookback).mean().iloc[-1]
    latest_volume = volume.iloc[-1]

    if avg_volume is None or np.isnan(avg_volume) or avg_volume == 0:
        return empty

    volume_ratio = float(latest_volume) / float(avg_volume)
    change_pct = float(price_change) if not np.isnan(price_change) else 0.0

    classification = "normal"
    abs_change = abs(change_pct)

    if abs_change > 0.02:  # >2% move
        if volume_ratio > 1.5:
            classification = "confirmed_up" if change_pct > 0 else "confirmed_down"
        else:
            classification = "unconfirmed_up" if change_pct > 0 else "unconfirmed_down"
    elif abs_change < 0.005:  # <0.5% move (essentially flat)
        if volume_ratio > 2.0:
            classification = "accumulation"
        elif volume_ratio < 0.5:
            classification = "quiet"

    return {
        "classification": classification,
        "price_change_pct": round(change_pct * 100, 2),
        "volume_ratio": round(volume_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Multi-Timeframe Alignment
# ---------------------------------------------------------------------------

def multi_timeframe_alignment(daily_close: pd.Series,
                              weekly_close: pd.Series) -> dict:
    """
    Compare trend direction on the daily and weekly timeframes. When they
    agree, signals are significantly more reliable.

    Trend direction is measured via the slope sign of SMA50 over the last
    5 bars (daily) or 3 bars (weekly).

    Returns {
        "daily_trend": "up" | "down" | "sideways",
        "weekly_trend": "up" | "down" | "sideways",
        "aligned": bool,                 (both up or both down)
        "alignment_score": int,          (100 aligned, 50 one sideways, 0 conflicting)
    }
    """
    def _trend(close_series: pd.Series, sma_period: int, slope_window: int) -> str:
        if close_series is None or len(close_series) < sma_period + slope_window:
            return "sideways"
        sma_series = close_series.rolling(window=sma_period).mean()
        recent = sma_series.iloc[-slope_window:].dropna()
        if len(recent) < 2:
            return "sideways"
        delta = float(recent.iloc[-1] - recent.iloc[0])
        base = float(recent.iloc[0])
        if base == 0:
            return "sideways"
        pct = delta / base
        if pct > 0.005:
            return "up"
        if pct < -0.005:
            return "down"
        return "sideways"

    daily_trend = _trend(daily_close, 50, 5)
    weekly_trend = _trend(weekly_close, 10, 3)  # SMA10 on weekly ≈ SMA50 on daily

    aligned = daily_trend == weekly_trend and daily_trend != "sideways"

    if aligned:
        alignment_score = 100
    elif daily_trend == "sideways" or weekly_trend == "sideways":
        alignment_score = 50
    else:
        alignment_score = 0  # conflicting (one up, one down)

    return {
        "daily_trend": daily_trend,
        "weekly_trend": weekly_trend,
        "aligned": aligned,
        "alignment_score": alignment_score,
    }


# ---------------------------------------------------------------------------
# Support & Resistance Levels
# ---------------------------------------------------------------------------

def support_resistance(high: pd.Series, low: pd.Series, close: pd.Series,
                       window: int = 20):
    """
    Auto-detect support and resistance levels.

    Finds local minima (support) and maxima (resistance) using a rolling
    window, then clusters nearby levels (within 2%) and ranks by frequency.

    Returns: {"supports": [{"price": float, "strength": int}, ...],
              "resistances": [{"price": float, "strength": int}, ...]}
    """
    supports = []
    resistances = []
    low_vals = low.values
    high_vals = high.values

    for i in range(window, len(close) - window):
        if low_vals[i] == np.min(low_vals[i - window:i + window + 1]):
            supports.append(float(low_vals[i]))
        if high_vals[i] == np.max(high_vals[i - window:i + window + 1]):
            resistances.append(float(high_vals[i]))

    return {
        "supports": _cluster_levels(supports),
        "resistances": _cluster_levels(resistances),
    }


def _cluster_levels(levels, threshold=0.02):
    """Group price levels within threshold% of each other."""
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for level in levels[1:]:
        if (level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    result = [{"price": round(float(np.mean(c)), 2), "strength": len(c)} for c in clusters]
    result.sort(key=lambda x: x["strength"], reverse=True)
    return result[:5]


# ---------------------------------------------------------------------------
# Fibonacci Retracement Levels
# ---------------------------------------------------------------------------

def fibonacci_levels(high: pd.Series, low: pd.Series, lookback: int = 60):
    """
    Fibonacci Retracement Levels

    What it is:  Key price levels derived from the Fibonacci sequence (23.6%,
                 38.2%, 50%, 61.8%, 78.6%) between a recent high and low.
    Why it matters:  These levels often act as support/resistance. The 61.8%
                     retracement is considered the strongest level.

    Returns: {"high": float, "low": float, "levels": {"23.6%": float, ...}}
    """
    recent_high = float(high.iloc[-lookback:].max()) if len(high) >= lookback else float(high.max())
    recent_low = float(low.iloc[-lookback:].min()) if len(low) >= lookback else float(low.min())
    diff = recent_high - recent_low

    return {
        "high": round(recent_high, 2),
        "low": round(recent_low, 2),
        "levels": {
            "0%": round(recent_high, 2),
            "23.6%": round(recent_high - 0.236 * diff, 2),
            "38.2%": round(recent_high - 0.382 * diff, 2),
            "50%": round(recent_high - 0.5 * diff, 2),
            "61.8%": round(recent_high - 0.618 * diff, 2),
            "78.6%": round(recent_high - 0.786 * diff, 2),
            "100%": round(recent_low, 2),
        },
    }


# ---------------------------------------------------------------------------
# Moving Average Crossovers (Golden Cross / Death Cross)
# ---------------------------------------------------------------------------

def ma_crossovers(sma_50_series: pd.Series, sma_200_series: pd.Series, dates: list):
    """
    Detect Golden Cross and Death Cross events.

    Golden Cross: 50-day SMA crosses ABOVE 200-day SMA → bullish
    Death Cross:  50-day SMA crosses BELOW 200-day SMA → bearish

    Returns: {
        "golden_cross": date_str | None,  (most recent golden cross)
        "death_cross": date_str | None,   (most recent death cross)
        "current_signal": "golden_cross" | "death_cross" | None,
        "days_since_cross": int | None,
    }
    """
    s50 = sma_50_series.values
    s200 = sma_200_series.values

    last_golden = None
    last_death = None

    for i in range(1, len(s50)):
        if np.isnan(s50[i]) or np.isnan(s200[i]) or np.isnan(s50[i - 1]) or np.isnan(s200[i - 1]):
            continue
        if s50[i] > s200[i] and s50[i - 1] <= s200[i - 1]:
            last_golden = i
        elif s50[i] < s200[i] and s50[i - 1] >= s200[i - 1]:
            last_death = i

    result = {
        "golden_cross": None,
        "death_cross": None,
        "current_signal": None,
        "days_since_cross": None,
    }

    if last_golden is not None and last_golden < len(dates):
        result["golden_cross"] = dates[last_golden]
    if last_death is not None and last_death < len(dates):
        result["death_cross"] = dates[last_death]

    # Determine current signal (whichever cross happened most recently)
    if last_golden is not None or last_death is not None:
        if last_golden is not None and (last_death is None or last_golden > last_death):
            result["current_signal"] = "golden_cross"
            result["days_since_cross"] = len(dates) - 1 - last_golden
        elif last_death is not None:
            result["current_signal"] = "death_cross"
            result["days_since_cross"] = len(dates) - 1 - last_death

    return result


# ---------------------------------------------------------------------------
# Beta vs Benchmark
# ---------------------------------------------------------------------------

def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Beta — measures a stock's volatility relative to the market.

    Beta > 1: stock is MORE volatile than the market
    Beta = 1: stock moves with the market
    Beta < 1: stock is LESS volatile (defensive)

    Formula: Beta = Cov(stock, market) / Var(market)
    """
    aligned = pd.DataFrame({"stock": stock_returns, "market": market_returns}).dropna()
    if len(aligned) < 20:
        return None
    cov_matrix = np.cov(aligned["stock"], aligned["market"])
    market_var = np.var(aligned["market"])
    if market_var == 0:
        return None
    return float(cov_matrix[0][1] / market_var)


# ---------------------------------------------------------------------------
# Helper: compute all indicators at once
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> dict:
    """
    Compute all technical indicators for a DataFrame with OHLCV columns.

    Returns a dict of indicator name → list of values (aligned by index).
    NaN values appear where the indicator needs more history to compute
    (e.g., first 19 values of a 20-period SMA will be NaN).
    """
    close = df["close"]

    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    ema_12 = ema(close, 12)
    ema_26 = ema(close, 26)
    rsi_14 = rsi(close, 14)
    macd_line, signal_line, macd_hist = macd(close)
    bb_upper, bb_middle, bb_lower = bollinger_bands(close)
    returns = daily_returns(close)
    vol = volatility(close)
    cum_returns = cumulative_returns(close)

    # New indicators
    sma_200 = sma(close, 200)
    atr_14 = atr(df["high"], df["low"], close)
    obv_series = obv(close, df["volume"])
    stoch_k, stoch_d = stochastic(df["high"], df["low"], close)
    adx_series, plus_di, minus_di = adx(df["high"], df["low"], close)
    mfi_series = mfi(df["high"], df["low"], close, df["volume"])

    return {
        "sma_20": sma_20.tolist(),
        "sma_50": sma_50.tolist(),
        "sma_200": sma_200.tolist(),
        "ema_12": ema_12.tolist(),
        "ema_26": ema_26.tolist(),
        "rsi": rsi_14.tolist(),
        "macd_line": macd_line.tolist(),
        "macd_signal": signal_line.tolist(),
        "macd_histogram": macd_hist.tolist(),
        "bollinger_upper": bb_upper.tolist(),
        "bollinger_middle": bb_middle.tolist(),
        "bollinger_lower": bb_lower.tolist(),
        "daily_returns": returns.tolist(),
        "volatility": vol.tolist(),
        "cumulative_returns": cum_returns.tolist(),
        "atr": atr_14.tolist(),
        "obv": obv_series.tolist(),
        "stochastic_k": stoch_k.tolist(),
        "stochastic_d": stoch_d.tolist(),
        "adx": adx_series.tolist(),
        "plus_di": plus_di.tolist(),
        "minus_di": minus_di.tolist(),
        "mfi": mfi_series.tolist(),
    }
