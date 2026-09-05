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

# TTL for a REMEMBERED FAILURE. Short, because a refusal is far more perishable
# than a close: the feed may recover within the quarter-hour a good result is
# held for. But not zero, because learning that the feed has nothing for a
# symbol costs ~6 seconds inside the vendored client's @retry(tries=20), and
# re-paying that on every dashboard load — which is what caching successes only
# meant — was a large part of why the grid was slow.
ERROR_CACHE_TTL_SECONDS = 120
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

# At or above this many consecutive refusals, the feed is treated as having no
# data for a symbol at all. Two readers depend on the SAME number and it lives
# here so they cannot drift: `routers/cron.py` demotes such a symbol behind
# everything healthy in its refresh queue, and `core/card_snapshot.py` marks it
# "no price feed" on the dashboard. If those disagreed, a symbol could be
# demoted out of the refresh queue while the grid still presented it as a card
# that ought to be loading — which is precisely the indefinite "--" this whole
# surface was rebuilt to remove. Low enough that a genuinely dead name stops
# eating the budget within one pass, high enough to ride out a transient outage.
FAILURE_DEMOTION_THRESHOLD = 3


# === Bar fetch limits ===

# INTERNAL_BARS_MIN is the ONE fetch window for every path that scores a
# stock — detail page, dashboard batch, and per-holding portfolio analysis.
# They must match: the composite score depends on the window (volatility,
# support/resistance, SMA200, beta), so different windows meant the same
# stock scored differently depending on which page you opened.
INTERNAL_BARS_MIN = 400                    # Min bars fetched internally so SMA200 is valid even when caller asks for fewer
USER_BARS_MIN = 30                         # Lower bound on user-requested `bars` query param
USER_BARS_MAX = 5000                       # Upper bound on user-requested `bars` query param
BATCH_MAX_SYMBOLS = 24                     # Max symbols accepted per batched composite request
HISTORICAL_MAX_SYMBOLS = 20                # Max symbols accepted per /historical request
COMPARE_MIN_SYMBOLS = 2
COMPARE_MAX_SYMBOLS = 10


# === Trading calendar ===

TRADING_DAYS_PER_YEAR = 252                # Used for annualizing Sharpe/Sortino and for the 52-week window


# === Divergence detection windows ===

# ONE lookback for every path. The batch and portfolio paths used to scan a
# shorter window, which meant a divergence visible on the detail page was
# invisible on the card for the same stock — and the divergence category
# (8% of the score) differed between them.
DIVERGENCE_LOOKBACK_FULL = 60


# === Composite score signal cutoffs ===

# Used by both the scoring engine and portfolio signal generator.
# Keep in sync with SCORE_*_MAX in egx-api-fe/src/app/lib/constants.ts.
SCORE_STRONG_SELL_MAX = 20
SCORE_SELL_MAX = 40
SCORE_HOLD_MAX = 60
SCORE_BUY_MAX = 80
# (>80 is Strong Buy)


# === Stop-loss convention ===

# THE house convention, used everywhere a stop-loss is suggested:
# stop = nearest_support - STOP_LOSS_ATR_MULTIPLIER x ATR.
# Anchoring to support (not to entry price) keeps the number objective and
# computable before the user has bought anything. Referenced by
# core/levels.py, core/entry_price.py and routers/portfolio_analysis.py —
# do not re-derive a different multiplier at a call site.
STOP_LOSS_ATR_MULTIPLIER = 1.5
STOP_LOSS_FALLBACK_PCT = 0.02              # Used when ATR is unavailable: 2% below support


# === Bollinger Band squeeze detection ===

BB_SQUEEZE_LOOKBACK_BARS = 130             # Comparison window for "is BB width unusually narrow?"
BB_SQUEEZE_RATIO = 0.7                     # Width below 70% of recent average = squeeze


# === Default settings (DB seeds + fallbacks) ===

# The CBE overnight deposit rate, which is the app's stand-in for a risk-free
# rate. Still very high by global standards, so Sharpe ratios here look poor
# next to developed markets — say so whenever one is explained.
#
# THIS IS A MARKET FACT WITH A DATE ON IT, NOT A TUNING KNOB. It was 25 from the
# day the app shipped until 2026-09-02, by which point the CBE had cut 825bp
# since April 2025 (to 20.00% in Dec-2025 and 19.00% on 12 Feb 2026, held at the
# 20 Aug 2026 meeting). Six hundred basis points of staleness is not cosmetic:
# this one number is the Sharpe hurdle, the Sortino hurdle, the whole input to
# score_risk_adjusted (13% of the composite), and the bar realized trades are
# graded against via beat_t_bill_count. Too high, and the app understates every
# Sharpe ratio and fails trades that genuinely beat cash.
#
# Caveat to state on screen wherever it matters: this is the POLICY rate, not a
# 91-day T-bill auction yield. There is no free machine-readable Egyptian T-bill
# series — cbe.org.eg rejects automated requests — so the bill rate is
# approximated by the policy rate rather than printed from an auction.
#
# Phase 3 replaces this constant with a dated series in `macro_series`, fed from
# ECONOMICS:EGINTR (keyless, and reachable through the already-vendored client),
# so historical dates get the rate that was actually in force instead of today's.
DEFAULT_RISK_FREE_RATE_PCT = 19            # CBE overnight deposit, as of 2026-08-20

# The value that shipped before the correction above. `init_db` upgrades rows
# still holding it, and ONLY rows still holding it, so an admin who deliberately
# set a rate keeps theirs. Delete this once no deployment can still be on 25.
STALE_RISK_FREE_RATE_PCT = 25

# Percentage points a year that ignoring dividends costs a return figure.
#
# MEASURED, not assumed: eight liquid EGX names priced from Yahoo's split- and
# dividend-event history, comparing price CAGR against total-return CAGR over
# their full listings. Median 3.70 pp/yr; COMI 3.55, SWDY 3.85, ETEL 6.44,
# ABUK 10.22, TMGH 0.88.
#
# It matters because `score_risk_adjusted` compares a PRICE return against the
# policy rate, which is a TOTAL return — cash pays you its yield, and the
# stock's dividends were being discarded before the comparison. The bias runs
# one way: every stock looks worse against cash than it was.
#
# It is DISCLOSED rather than added back, deliberately. `score_quality` already
# rewards dividend yield as evidence a company generates real cash; adding the
# same yield into the return comparison would pay a stock twice for one fact and
# leave a number that means neither thing — the failure the liquidity band was
# explicitly designed to avoid. core/corporate_actions.py builds a proper
# total-return series for offline validation; correcting the live score needs a
# per-symbol series, not a current-yield shortcut.
DIVIDEND_DRAG_PP_PER_YEAR = 3.70


# === Monte Carlo / risk metrics ===

MONTE_CARLO_SIMULATIONS = 1000             # Path count; vectorized via numpy
MONTE_CARLO_FORECAST_DAYS = 60             # Horizon in trading days (~3 months)
VAR_PERCENTILE = 5                         # 5th-percentile cutoff for daily returns (Value-at-Risk 95%)
MAX_DRAWDOWN_WARNING_PCT = 0.20            # Drawdown beyond -20% triggers an action_required signal
CURRENT_DRAWDOWN_WARNING_PCT = 0.05        # Currently in a drawdown of ≥5% from peak triggers a warning


# === Portfolio risk thresholds ===

# These two thresholds drive BOTH the warning signals and the diversification
# score's penalties. They were separate values once, so a portfolio could show
# "Diversification 100/100" beside a "45% in Banking — consider diversifying"
# alert. One threshold per concept keeps the score and the alerts consistent.
SECTOR_ALERT_PCT = 40                      # Sector > 40%: `sector_concentration` warning + diversification-score penalty
STOCK_ALERT_PCT = 35                       # Single stock > 35%: `stock_concentration` warning + diversification-score penalty
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


# === News feed ===

# A story older than this is not news. Chosen from measurement, not taste:
# across a 24-symbol sample the newest story was 0 days old for ETEL and
# 275 days old for ACGC. A 7-day window empties the feed for most holdings;
# 90 days lets that 275-day-old item render as news.
NEWS_RECENCY_DAYS = 30

# Ceiling on symbols fetched per request. Holdings win the budget, then
# watchlist, then EGX30. Measured: 24 symbols fan out in 1.30s at 8 workers.
NEWS_MAX_SYMBOLS = 40
NEWS_FETCH_WORKERS = 8

# Wall clock for the whole fan-out, mirroring BATCH_DEADLINE_SECONDS. The
# source is fast, but an unbounded fan-out is how the dashboard broke. This
# doubles as the tripwire in the spec: routinely tripping it means the
# on-demand design no longer holds and the pe_data snapshot pattern applies.
NEWS_DEADLINE_SECONDS = 8.0
NEWS_REQUEST_TIMEOUT_SECONDS = 6.0

# Per symbol, after the recency filter. COMI alone returns 98 stories and the
# raw payload across 22 symbols measured 419 KB; this is what keeps the
# response small enough for a phone on a mobile connection.
NEWS_MAX_ITEMS_PER_SYMBOL = 10


# === Dividend history / calendar ===

# Per-symbol dividend history comes from Yahoo on demand (core/dividend_history).
# Dividends move a few times a year at most, so the cached answer can sit far
# longer than a price — no reason to re-fetch a stable list every 15 minutes.
DIVIDEND_HISTORY_TTL_SECONDS = 6 * 3600     # 6h
DIVIDEND_HISTORY_TIMEOUT_SECONDS = 6.0      # one Yahoo GET; generous for a cold serverless container
