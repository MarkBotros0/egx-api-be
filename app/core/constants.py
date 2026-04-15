"""
Centralized tuning knobs and thresholds for the EGX backend.

Constants here are values that (a) are referenced from multiple files,
(b) have differed across files in the past, or (c) are likely to be
tuned. Values that are conventional indicator parameters (RSI=14 etc.)
remain as defaults in `indicators.py` — they're standard, self-documenting
at call sites, and centralizing them adds boilerplate without clarity.
Per-algorithm thresholds with self-documenting locals (RSI overbought=70,
ADX strong-trend=25, etc.) also stay inline.
"""

# === Cache TTLs (seconds) ===

DEFAULT_CACHE_TTL_SECONDS = 900            # In-memory analysis cache (15 min)
TICKERS_CACHE_TTL_SECONDS = 12 * 3600      # Ticker list rarely changes (12 h)
MACRO_CACHE_TTL_SECONDS = 3600             # Macro data refresh cadence (1 h)


# === Vercel timeout budget ===

# Vercel Python serverless functions have a 30 s hard limit. The batch
# composite endpoint stops waiting for stragglers after this many seconds
# and returns partial results, leaving plenty of margin for response
# serialization and cold-start jitter.
BATCH_DEADLINE_SECONDS = 20.0

# ThreadPool worker count for the batched composite endpoint.
BATCH_WORKERS = 6


# === Bar fetch limits ===

BATCH_BARS = 220                           # Bars fetched per symbol in batched composite analysis
INTERNAL_BARS_MIN = 400                    # Min bars fetched internally so SMA200 is valid even when caller asks for fewer
USER_BARS_MIN = 30                         # Lower bound on user-requested `bars` query param
USER_BARS_MAX = 5000                       # Upper bound on user-requested `bars` query param
BATCH_MAX_SYMBOLS = 24                     # Max symbols accepted per batched composite request
HISTORICAL_MAX_SYMBOLS = 20                # Max symbols accepted per /historical request
COMPARE_MIN_SYMBOLS = 2
COMPARE_MAX_SYMBOLS = 10
PORTFOLIO_FETCH_BARS_MIN = 200             # Floor on bars per holding (so SMA50/RSI are stable)
PORTFOLIO_FETCH_BARS_MAX = 500             # Ceiling on bars per holding (caps Vercel-timeout exposure)


# === Trading calendar ===

TRADING_DAYS_PER_YEAR = 252                # Used for annualizing Sharpe/Sortino and for the 52-week window


# === Divergence detection windows ===

# Full /api/analysis can afford the longer window; batched + per-holding
# portfolio paths cap at 30 to fit the per-symbol time budget. Keep both
# values explicit so the difference is auditable.
DIVERGENCE_LOOKBACK_FULL = 60
DIVERGENCE_LOOKBACK_BATCH = 30
DIVERGENCE_LOOKBACK_PORTFOLIO = 30


# === Composite score signal cutoffs ===

# Used by both the scoring engine and portfolio signal generator.
# Keep in sync with SCORE_*_MAX in egx-api-fe/src/app/lib/constants.ts.
SCORE_STRONG_SELL_MAX = 20
SCORE_SELL_MAX = 40
SCORE_HOLD_MAX = 60
SCORE_BUY_MAX = 80
# (>80 is Strong Buy)


# === Bollinger Band squeeze detection ===

BB_SQUEEZE_LOOKBACK_BARS = 130             # Comparison window for "is BB width unusually narrow?"
BB_SQUEEZE_RATIO = 0.7                     # Width below 70% of recent average = squeeze


# === Default settings (DB seeds + fallbacks) ===

DEFAULT_CASH_AVAILABLE_EGP = 50_000        # Initial portfolio cash row in settings
DEFAULT_RISK_FREE_RATE_PCT = 25            # Egypt T-bill rate ~25% — high by global standards


# === Monte Carlo / risk metrics ===

MONTE_CARLO_SIMULATIONS = 1000             # Path count; vectorized via numpy
MONTE_CARLO_FORECAST_DAYS = 60             # Horizon in trading days (~3 months)
VAR_PERCENTILE = 5                         # 5th-percentile cutoff for daily returns (Value-at-Risk 95%)
MAX_DRAWDOWN_WARNING_PCT = 0.20            # Drawdown beyond -20% triggers an action_required signal
CURRENT_DRAWDOWN_WARNING_PCT = 0.05        # Currently in a drawdown of ≥5% from peak triggers a warning


# === Portfolio risk thresholds ===

CONCENTRATION_WARNING_PCT = 30             # Single position > 30% of portfolio = penalty starts in diversification score
CONCENTRATION_CRITICAL_PCT = 50            # Sector > 50% of portfolio = larger diversification-score penalty
SECTOR_ALERT_PCT = 40                      # Sector > 40% triggers a `sector_concentration` warning signal
STOCK_ALERT_PCT = 35                       # Single stock > 35% triggers a `stock_concentration` warning signal
CORRELATION_HIGH_THRESHOLD = 0.7           # Pairwise corr > 0.7 = "high" (warning signal)
CORRELATION_NEGATIVE_THRESHOLD = -0.3      # Pairwise corr < -0.3 = "good for diversification" (info signal)
PROFIT_TARGET_PCT = 20                     # Unrealized gain > 20% triggers profit-taking reminder
BIG_LOSS_PCT = -15                         # Unrealized loss worse than -15% triggers big_loss warning


# === Macro direction thresholds ===

# Monthly EGX30 % change buckets used to label the index as up / down / stable.
MACRO_TREND_UP_PCT = 2
MACRO_TREND_DOWN_PCT = -2
# USD/EGP daily change above this magnitude is labeled up/down rather than stable.
USDEGP_DIRECTION_THRESHOLD_PCT = 0.1


# === Compare-page defaults ===

COMPARE_DEFAULT_LOOKBACK_DAYS = 180        # Default window when caller omits start/end
HISTORICAL_DEFAULT_LOOKBACK_DAYS = 365     # Default window for /historical when caller omits start/end
