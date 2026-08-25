"""
Single source of truth for the `extras` dict fed to `compute_composite`.

WHY THIS MODULE EXISTS
----------------------
The composite score is computed in three places — the stock detail page
(`/api/analysis`), the dashboard cards (`/api/analysis?mode=batch`) and the
portfolio page (`/api/portfolio_analysis`). Each used to assemble its own
`extras` dict by hand, and they disagreed: the batch path omitted the inputs
that `score_quality`, `score_risk_adjusted` and `score_relative_strength`
need, so those three scorers returned None and `compute_composite`
renormalized over the remaining 5 of 8 categories.

The categories that vanished were exactly the punitive ones — laggards vs
EGX30, sub-T-bill returns, drawdown quality — so the dashboard read a stock
more generously than its own detail page. Measured on identical data: 66
"Buy" on the card, 45 "Hold" on the detail page. A user tapped a Buy and
landed on a Hold.

Every caller now builds extras HERE. A category may still be dropped when
its input genuinely isn't computable (a stock with 60 bars of history has no
1-year return), but it is dropped for the same reason on every page.

The builder is pure: no DB, no network. Callers pass in what they fetched.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from app.core.constants import (
    BB_SQUEEZE_LOOKBACK_BARS,
    BB_SQUEEZE_RATIO,
    DIVERGENCE_LOOKBACK_FULL,
    TRADING_DAYS_PER_YEAR,
)
from app.core.indicators import (
    annualized_return,
    daily_returns,
    detect_divergences,
    liquidity_score,
    ma_crossovers,
    macd,
    multi_timeframe_alignment,
    relative_strength,
    rsi as calc_rsi,
    sma,
    volume_price_confirmation,
)

# Weekly resampling anchor. EGX trades Sun-Thu, so Thursday is the week's
# closing session.
WEEKLY_RESAMPLE_RULE = "W-THU"

# Bars needed on each side of a 20-bar comparison window.
_TREND_WINDOW = 20

# Averaging window for the liquidity check, in bars.
_LIQUIDITY_LOOKBACK = 20

# How many bars make a year, per interval. EVERY annualization, 1-year window
# and history gate must scale by this — otherwise weekly bars get treated as
# trading days and the numbers are wrong by multiples, not rounding.
BARS_PER_YEAR = {
    "Daily": TRADING_DAYS_PER_YEAR,   # 252
    "Weekly": 52,
    "Monthly": 12,
}

# The next timeframe UP, for multi-timeframe alignment. Comparing an interval
# against itself is meaningless, so Monthly (no higher frame in this app) is
# excluded rather than compared to itself.
_HIGHER_TIMEFRAME_RULE = {
    "Daily": WEEKLY_RESAMPLE_RULE,
    "Weekly": "ME",   # month end
}

# Minimum history before annualizing is meaningful, in BARS, per interval.
# ~6 months: 120 trading days, 26 weeks, 6 months.
MIN_BARS_FOR_ANNUALIZATION = {"Daily": 120, "Weekly": 26, "Monthly": 6}


def _safe_last(series) -> Optional[float]:
    """Final non-NaN float of a Series, or None."""
    try:
        s = series.dropna()
        if len(s) == 0:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def _compute_bb_squeeze(indicators: dict) -> bool:
    """Is the current Bollinger width unusually narrow vs its recent average?"""
    try:
        bb_u = indicators.get("bollinger_upper") or []
        bb_l = indicators.get("bollinger_lower") or []
        bb_m = indicators.get("bollinger_middle") or []
        if not (bb_u and bb_l and bb_m) or len(bb_u) < BB_SQUEEZE_LOOKBACK_BARS:
            return False
        widths = [(u - l) / m if m else None for u, l, m in zip(bb_u, bb_l, bb_m)]
        valid = [w for w in widths[-BB_SQUEEZE_LOOKBACK_BARS:] if w is not None and w == w]
        if not valid:
            return False
        return valid[-1] < (sum(valid) / len(valid)) * BB_SQUEEZE_RATIO
    except Exception:
        return False


def _higher_timeframe(close: pd.Series, base_interval: str) -> Optional[pd.Series]:
    """
    Resample `close` to the next timeframe UP from `base_interval`.

    Resampling rather than fetching a separate series is deliberate: it costs
    no extra network call (so the batch path can afford it), and it guarantees
    both views are the same underlying data. A separately-fetched weekly
    series can disagree with the daily one at the provider, which would
    reintroduce card-vs-detail score divergence.

    Returns None when there is no higher frame to compare against (Monthly).
    Resampling a weekly series to weekly is a no-op, so without this gate the
    alignment check would compare a timeframe with itself.
    """
    rule = _HIGHER_TIMEFRAME_RULE.get(base_interval)
    if rule is None:
        return None
    try:
        s = close.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        higher = s.resample(rule).last().dropna()
        return higher if len(higher) >= 10 else None
    except Exception:
        return None


def build_composite_extras(
    df: pd.DataFrame,
    indicators: dict,
    *,
    interval: str = "Daily",
    egx30_close: Optional[pd.Series] = None,
    weekly_close: Optional[pd.Series] = None,
    include_multi_timeframe: bool = True,
    risk_free_rate_pct: Optional[float] = None,
    pe_ratio: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    loss_making: Optional[bool] = None,
    index_membership: Optional[str] = None,
    divergence_lookback: int = DIVERGENCE_LOOKBACK_FULL,
) -> dict:
    """
    Assemble every input `compute_composite` consumes, plus the by-products
    endpoints serve in their responses (so nothing is computed twice).

    Arguments:
      df:                  Full internal OHLCV history (lowercase columns).
                           Pass the UNTRIMMED frame — indicators need the
                           long window even when the response shows fewer bars.
      indicators:          Output of `compute_all(df)` over that same frame.
      interval:            "Daily" | "Weekly" | "Monthly" — the interval `df`
                           is sampled at. REQUIRED for correctness on the
                           non-daily views: annualization, the 1-year drawdown
                           window, the minimum-history gate and the relative
                           strength window all scale by bars-per-year. Passing
                           the default while handing in weekly bars makes the
                           app treat 252 weeks as one year.
      egx30_close:         Benchmark closes for relative strength. None ->
                           the relative_strength category is skipped.
      weekly_close:        Pre-computed weekly closes. None -> resampled from
                           `df` (preferred; see _weekly_from_daily).
      include_multi_timeframe:
                           Escape hatch for timeout pressure. Leaving this
                           True is strongly preferred — turning it off makes
                           `score_quality` weaker on that path only, which is
                           the class of divergence this module exists to
                           prevent. Only flip it if the portfolio endpoint
                           starts brushing the 30 s Vercel ceiling.
      risk_free_rate_pct:  T-bill rate as a PERCENT (e.g. 25.0).
      pe_ratio:            Trailing P/E, or None when unknown.
      dividend_yield:      PERCENT (e.g. 3.12). 0.0 means "pays nothing" —
                           real data, distinct from None ("unknown").
      loss_making:         From diluted EPS < 0, or None when unknown.
      index_membership:    "EGX30"/"EGX70"/"EGX100"/"NILEX" or None. Selects
                           which volume floors the liquidity check uses; None
                           falls back to EGX100 inside liquidity_score.
      divergence_lookback: Bars scanned for divergences.

    Returns:
      {
        "extras": {...},            # -> compute_composite(extras=...)
        "divergences": {...},       # by-products, for the response body
        "volume_price": {...} | None,
        "bb_squeeze": bool,
        "crossovers": {...},
        "multi_timeframe": {...} | None,
        "liquidity": {...} | None,
        "high_52w": float | None,
        "low_52w": float | None,
      }
    """
    close = df["close"]
    bars_per_year = BARS_PER_YEAR.get(interval, TRADING_DAYS_PER_YEAR)

    # --- Crossovers (also feeds golden_cross_active) ---
    try:
        sma_50 = sma(close, 50)
        sma_200 = sma(close, 200)
        dates_list = [str(idx)[:10] for idx in df.index]
        crossovers = ma_crossovers(sma_50, sma_200, dates_list)
    except Exception:
        crossovers = {"current_signal": None, "days_since_cross": None}

    # --- Divergences ---
    try:
        rsi_series = calc_rsi(close)
        macd_line, _, _ = macd(close)
        divergences = {
            "rsi": detect_divergences(close, rsi_series, lookback=divergence_lookback),
            "macd": detect_divergences(close, macd_line, lookback=divergence_lookback),
        }
    except Exception:
        divergences = {"rsi": {}, "macd": {}}

    # --- Volume/price confirmation ---
    try:
        volume_price = volume_price_confirmation(close, df["volume"])
    except Exception:
        volume_price = None

    bb_squeeze = _compute_bb_squeeze(indicators)

    # --- OBV / price direction over the last 20 bars ---
    # Both use the SAME [-21] index so "is OBV rising" and "is price rising"
    # describe the same window on every code path.
    obv_rising = None
    price_rising_20d = None
    obv_full = indicators.get("obv") or []
    if len(obv_full) >= _TREND_WINDOW + 1 and obv_full[-1] is not None and obv_full[-(_TREND_WINDOW + 1)] is not None:
        obv_rising = obv_full[-1] > obv_full[-(_TREND_WINDOW + 1)]
    if len(close) >= _TREND_WINDOW + 1:
        price_rising_20d = float(close.iloc[-1]) > float(close.iloc[-(_TREND_WINDOW + 1)])

    # --- Multi-timeframe alignment (vs the next frame UP) ---
    multi_timeframe = None
    if include_multi_timeframe:
        try:
            wk = weekly_close if weekly_close is not None else _higher_timeframe(close, interval)
            if wk is not None:
                multi_timeframe = multi_timeframe_alignment(close, wk)
        except Exception:
            multi_timeframe = None

    # --- Trend consistency: share of last 20 bars closing above SMA20 ---
    # Bars sitting exactly ON the average are excluded rather than counted as
    # "below". A suspended stock whose price is frozen has close == SMA20 on
    # every bar, which scored 0.0 — read by score_quality as "steady
    # downtrend" and penalised, when in truth the price simply never moved.
    # With no bar off the average, this stays None and the band is skipped.
    trend_consistency = None
    try:
        sma20 = sma(close, 20)
        moved = [
            (c, s)
            for c, s in zip(close.iloc[-_TREND_WINDOW:], sma20.iloc[-_TREND_WINDOW:])
            if s == s and c != s
        ]
        if moved:
            trend_consistency = sum(1 for c, s in moved if c > s) / len(moved)
    except Exception:
        trend_consistency = None

    # --- 52-week extremes, from the FULL frame and true intraday high/low ---
    # `bars_per_year` bars, not 252 — on weekly data a fixed 252 made this a
    # ~5-year window labelled as a 1-year one.
    #
    # The peak comes from `high`, not `close`. Reading closes made score_quality
    # say "near recent peak" off one number while StatsPanel rendered "52W High"
    # off another, on the same screen. `analysis.py` used to compute these
    # separately with its own copy of the bars-per-year table; it now reads them
    # from here so there is one definition.
    yearly_window = min(bars_per_year, len(df))
    high_52w = low_52w = None
    try:
        high_52w = float(df["high"].tail(yearly_window).max())
        low_52w = float(df["low"].tail(yearly_window).min())
    except Exception:
        high_52w = low_52w = None

    # --- Current drawdown vs the 52-week high (FRACTION, e.g. -0.15) ---
    current_drawdown_pct = None
    try:
        # Fall back to closes when there is no high column (synthetic frames).
        peak = high_52w
        if peak is None:
            peak = float(close.tail(yearly_window).max())
        if peak > 0:
            current_drawdown_pct = (float(close.iloc[-1]) - peak) / peak
    except Exception:
        current_drawdown_pct = None

    # --- Liquidity, index-aware, normalized to shares per DAY ---
    # liquidity_score's floors are daily share counts. Handing it weekly bars
    # would make one stock look ~5x more liquid on the Weekly view than on the
    # Daily view of itself — the same interval-calibration class of bug as the
    # annualization ones above.
    liquidity = None
    try:
        bars_per_day = TRADING_DAYS_PER_YEAR / bars_per_year
        liquidity = liquidity_score(
            df["volume"] / bars_per_day,
            index_membership=index_membership,
            lookback=_LIQUIDITY_LOOKBACK,
        )
    except Exception:
        liquidity = None

    # --- Annualized return + annualized volatility ---
    # Both scale by bars_per_year. With the daily constants hardcoded, a stock
    # that doubled over five years of weekly bars reported "+94% annualized"
    # (true: ~15%), and its volatility was inflated by sqrt(252/52) = 2.2x.
    # Those two feed Risk-Adjusted, which compares against the 25% T-bill.
    ann_return_pct = annualized_return(
        close, lookback=bars_per_year, periods_per_year=bars_per_year
    )
    volatility_annualized_pct = None
    try:
        per_bar_vol = daily_returns(close).std()
        if per_bar_vol == per_bar_vol:  # not NaN
            volatility_annualized_pct = float(per_bar_vol) * (bars_per_year ** 0.5) * 100.0
    except Exception:
        volatility_annualized_pct = None

    # --- ATR as % of price (reuses the already-computed ATR series) ---
    atr_pct_of_price = None
    try:
        atr_vals = indicators.get("atr") or []
        last_atr = None
        for v in reversed(atr_vals):
            if v is not None and v == v:
                last_atr = float(v)
                break
        cur = float(close.iloc[-1])
        if last_atr is not None and cur > 0:
            atr_pct_of_price = last_atr / cur * 100.0
    except Exception:
        atr_pct_of_price = None

    # --- Relative strength vs the benchmark ---
    # ~6 weeks of market time on every interval (30 daily bars, 6 weekly,
    # 2 monthly). A fixed 30 would compare 30 weeks of stock action against
    # the "30-day" label the Learn page and signals use.
    rs_lookback = max(2, round(bars_per_year * 30 / TRADING_DAYS_PER_YEAR))
    rs = None
    try:
        if egx30_close is not None and len(egx30_close) > 0:
            rs = relative_strength(close, egx30_close, lookback=rs_lookback)
    except Exception:
        rs = None

    extras = {
        "current_price": float(close.iloc[-1]),
        "divergences": divergences,
        "volume_price": volume_price,
        "bb_squeeze": bb_squeeze,
        "obv_rising": obv_rising,
        "price_rising_20d": price_rising_20d,
        "golden_cross_active": (
            crossovers.get("current_signal") == "golden_cross"
            and (crossovers.get("days_since_cross") or 99) < 10
        ),
        "multi_timeframe": multi_timeframe,
        "trend_consistency": trend_consistency,
        "current_drawdown_pct": current_drawdown_pct,
        "annualized_return_pct": ann_return_pct,
        "volatility_annualized_pct": volatility_annualized_pct,
        "atr_pct_of_price": atr_pct_of_price,
        # Expressed in TRADING-DAY equivalents so score_risk_adjusted's
        # minimum-history gate means the same span of real time on every
        # interval. Raw bar counts let 120 weekly bars (2.3 years) and 120
        # daily bars (6 months) both pass the same threshold.
        "history_days": int(len(close) * TRADING_DAYS_PER_YEAR / bars_per_year),
        "risk_free_rate_pct": risk_free_rate_pct,
        "relative_strength": rs,
        "pe_ratio": pe_ratio,
        "dividend_yield": dividend_yield,
        "loss_making": loss_making,
        "liquidity": liquidity,
    }

    return {
        "extras": extras,
        "divergences": divergences,
        "volume_price": volume_price,
        "bb_squeeze": bb_squeeze,
        "crossovers": crossovers,
        "multi_timeframe": multi_timeframe,
        "liquidity": liquidity,
        "high_52w": high_52w,
        "low_52w": low_52w,
    }
