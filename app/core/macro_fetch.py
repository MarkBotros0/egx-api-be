"""
Macro data helper — fetches and caches Egyptian macro indicators.

Data sources:
  - EGX30: via egxpy (reliable, same data source as the rest of the app)
  - USD/EGP: via egxpy or public source (best-effort)
  - CBE interest rate: stored in settings table (manually updateable)

Cached rows live in the Postgres `macro_data` table with a TTL checked on read.

All scraping is wrapped in try/except — returns partial/null data, never crashes.
"""

import math
from datetime import datetime, timedelta
from typing import Optional

from app.core.markets import EG
from app.core.constants import (
    DEFAULT_RISK_FREE_RATE_PCT,
    MACRO_CACHE_TTL_SECONDS,
    MACRO_TREND_DOWN_PCT,
    MACRO_TREND_UP_PCT,
    USDEGP_DIRECTION_THRESHOLD_PCT,
)


def fetch_macro(db):
    """
    Fetch macro data, using the Postgres cache if fresh enough.
    Returns a dict with interest_rate, usd_egp, and egx30 data,
    or None if nothing is available.
    """
    now = datetime.utcnow()
    cache_cutoff = (now - timedelta(seconds=MACRO_CACHE_TTL_SECONDS)).isoformat()

    # Check cache freshness
    try:
        rows = db.execute(
            "SELECT key, value, previous_value, change_pct, updated_at FROM macro_data"
        ).fetchall()
        cached = {r[0]: {"value": r[1], "previous_value": r[2], "change_pct": r[3], "updated_at": r[4]} for r in rows}

        # If we have cached data fresher than the macro TTL, return it
        if cached and all(
            r.get("updated_at", "") > cache_cutoff
            for r in cached.values()
            if r.get("updated_at")
        ):
            return _format_cached(cached, db)
    except Exception:
        cached = {}

    # Fetch fresh data
    result = {}

    # EGX30 — most reliable, uses same data source
    result["egx30"] = _fetch_egx30()

    # USD/EGP — best-effort
    result["usd_egp"] = _fetch_usdegp()

    # CBE interest rate — from settings table
    result["interest_rate"] = _fetch_interest_rate(db)

    # Store in cache
    _store_macro(db, result)

    return result


def read_cached_macro(db):
    """Latest cached macro regardless of age; never touches the network.

    For callers on a tight time budget (the alert hook): a stale regime is
    better than a fetch that can outlive the request. None when there is
    nothing cached or the read fails.
    """
    try:
        rows = db.execute(
            "SELECT key, value, previous_value, change_pct, updated_at FROM macro_data"
        ).fetchall()
        cached = {r[0]: {"value": r[1], "previous_value": r[2], "change_pct": r[3], "updated_at": r[4]} for r in rows}
        if not cached:
            return None
        return _format_cached(cached, db)
    except Exception:
        return None


def _index_reading(close) -> dict:
    """
    Level, daily change and the 20-session trend of an index close series.
    The EGX30 card's maths (moved verbatim from _fetch_egx30), shared with the
    display-only reading another market's macro card shows.
    """
    current = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else None
    change_pct = ((current - prev) / prev * 100) if prev else None

    # Monthly trend (compare to 20 trading days ago)
    month_ago = float(close.iloc[-min(20, len(close))]) if len(close) >= 5 else None
    monthly_change = ((current - month_ago) / month_ago * 100) if month_ago else None

    direction = None
    if monthly_change is not None:
        if monthly_change > MACRO_TREND_UP_PCT:
            direction = "up"
        elif monthly_change < MACRO_TREND_DOWN_PCT:
            direction = "down"
        else:
            direction = "stable"

    return {
        "value": round(current, 2),
        "change_pct": round(change_pct, 2) if change_pct else None,
        "monthly_change_pct": round(monthly_change, 2) if monthly_change else None,
        "direction": direction,
        "trend": "bullish" if (monthly_change and monthly_change > MACRO_TREND_UP_PCT) else
                 "bearish" if (monthly_change and monthly_change < MACRO_TREND_DOWN_PCT) else "sideways",
    }


def _fetch_egx30():
    """Fetch EGX30 index data via egxpy."""
    try:
        from app.vendor.egxpy import get_OHLCV_data
        df = get_OHLCV_data(EG.benchmark_symbol, EG.benchmark_exchange, "Daily", 30)
        if df is None or df.empty:
            return {"value": None, "change_pct": None, "direction": None}
        df.columns = [c.lower() for c in df.columns]
        return _index_reading(df["close"])
    except Exception:
        return {"value": None, "change_pct": None, "direction": None}


def _fetch_usdegp():
    """Fetch USD/EGP exchange rate — best-effort via egxpy."""
    try:
        from app.vendor.egxpy import get_OHLCV_data
        df = get_OHLCV_data("USDEGP", "FX_IDC", "Daily", 10)
        if df is None or df.empty:
            return {"value": None, "change_pct": None, "direction": None}
        df.columns = [c.lower() for c in df.columns]
        current = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else None
        change_pct = ((current - prev) / prev * 100) if prev else None
        direction = "up" if (change_pct and change_pct > USDEGP_DIRECTION_THRESHOLD_PCT) else \
                    "down" if (change_pct and change_pct < -USDEGP_DIRECTION_THRESHOLD_PCT) else "stable"
        return {
            "value": round(current, 2),
            "change_pct": round(change_pct, 2) if change_pct else None,
            "direction": direction,
        }
    except Exception:
        return {"value": None, "change_pct": None, "direction": None}


def fetch_macro_for(db, market):
    """
    The macro regime that modulates `market`'s scores, or None.

    Only EG has one today: `fetch_macro` is the EGX30 trend, USD/EGP and the
    CBE rate, and modulating a US score by the EGX30 regime would be wrong.
    Other markets get None, which skips modulation exactly as a macro outage
    does. US stays None on purpose: its calibration backtest ran without
    modulation. Its macro CARD comes from `fetch_macro_display`, which never
    reaches scoring.
    """
    if market.code != EG.code:
        return None
    return fetch_macro(db)


def fetch_macro_display(db, market) -> Optional[dict]:
    """
    A DISPLAY-ONLY macro reading for a market whose scores take no macro
    modulation (US): its benchmark's level and 20-session trend, and its
    risk-free rate. None for EG, whose card is `fetch_macro`.

    It never feeds scoring. The US calibration backtest ran WITHOUT macro
    modulation, so modulating US scores would detach them from their
    calibration — `fetch_macro_for` stays None for US, and this function is
    deliberately separate from it. Nothing is written to `macro_data` (that
    table is the EG regime the alert cron reads); the reading is held in the
    in-memory cache for MACRO_CACHE_TTL_SECONDS. The benchmark comes through
    `get_benchmark_close` at the scoring window, so it usually hits the cache
    the analysis paths already warmed. Never raises.
    """
    if market.code == EG.code:
        return None
    from app.core import cache
    from app.core.benchmark import get_benchmark_close
    from app.core.constants import INTERNAL_BARS_MIN

    key = cache.make_key(market.code, "macro_display")
    hit = cache.get(key)
    if hit is not None:
        return hit
    empty = {"value": None, "change_pct": None, "direction": None}
    try:
        close = get_benchmark_close(market, "Daily", INTERNAL_BARS_MIN)
        index = _index_reading(close) if close is not None and len(close) else dict(empty)
    except Exception:
        index = dict(empty)
    # The up/down/stable and bullish/bearish labels come from ±2% buckets
    # chosen for the EGX30 (MACRO_TREND_*_PCT). Not measured for this market,
    # so they are not sent: the card shows the measured change, no verdict.
    index["direction"] = None
    index.pop("trend", None)
    try:
        rate = read_risk_free_rate(db, market)
    except Exception:
        rate = market.risk_free_fallback_pct
    out = {
        "market": market.code,
        "currency": market.currency,
        "egx30": None,
        "usd_egp": None,
        "benchmark": {**index, "label": market.benchmark_label},
        # No direction: nothing computed one for this rate.
        "interest_rate": {"value": round(float(rate), 2) if rate is not None else None,
                          "direction": None,
                          "label": market.risk_free_label},
        # Says so on the wire: this reading does not move any score.
        "scoring": False,
    }
    if index.get("value") is not None:
        cache.set(key, out, ttl=MACRO_CACHE_TTL_SECONDS)
    return out


def read_risk_free_rate(db, market=EG) -> float:
    """
    `market`'s risk-free rate as a percent, from `settings`. Falls back to the
    market's default.

    Non-EG markets read `settings.risk_free_rate:<CODE>` (written for US by
    `refresh_us_risk_free`), else `market.risk_free_fallback_pct`. They never
    read the EG key: Egypt's policy rate is no hurdle for a US stock.

    For EG: the CBE policy rate, from `settings`. Falls back to the default.

    The ONE spelling of this read. It is a scoring input — `score_risk_adjusted`
    compares every stock's annualized return against it, and it is folded into
    the composite cache key as `rfr_tag` — so two readers disagreeing about it
    would mean two different scores for one stock. It had grown three separate
    copies (here, in `analysis.py`, and about to gain a third in the snapshot
    cron); they agreed, but nothing made them.

    Note this is the POLICY rate, not a 91-day T-bill auction yield: cbe.org.eg
    rejects automated requests and no free machine-readable bill series exists,
    so the bill rate is approximated. Say so wherever it is explained.
    """
    if market.code != EG.code:
        fallback = float(market.risk_free_fallback_pct or DEFAULT_RISK_FREE_RATE_PCT)
        if db is None:
            return fallback
        try:
            # The key is spelled as a literal, like the writer's, from the
            # registry's own code — never from request input.
            row = db.execute(
                f"SELECT value FROM settings WHERE key = 'risk_free_rate:{market.code}'"
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
        except Exception:
            pass
        return fallback

    if db is None:
        return float(DEFAULT_RISK_FREE_RATE_PCT)
    try:
        row = db.execute(
            "SELECT value FROM settings WHERE key = 'risk_free_rate'"
        ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return float(DEFAULT_RISK_FREE_RATE_PCT)


# The US risk-free series: the 3-month T-bill yield, in percent.
US_RISK_FREE_SYMBOL = "US03MY"
US_RISK_FREE_EXCHANGE = "TVC"


def refresh_us_risk_free(db) -> Optional[float]:
    """
    Store the last daily close of TVC:US03MY as `settings.risk_free_rate:US`.

    The same few daily bars are also upserted into `macro_series` (US03MY),
    so the dated history US sales are graded against keeps growing after the
    one-off backfill. That write is best-effort; the scalar is what matters.

    Returns the value, or None when the feed has nothing or anything fails —
    never raises, so the universe cron it rides on cannot go down with it.
    A failure leaves the previously stored value in place.
    """
    try:
        from app.vendor.egxpy import get_OHLCV_data
        df = get_OHLCV_data(US_RISK_FREE_SYMBOL, US_RISK_FREE_EXCHANGE, "Daily", 5)
        if df is None or df.empty:
            return None
        df.columns = [str(c).lower() for c in df.columns]
        value = float(df["close"].iloc[-1])
        if not math.isfinite(value):
            return None
        db.execute(
            "INSERT INTO settings (key, value) VALUES ('risk_free_rate:US', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (str(value),),
        )
        try:
            from app.core.macro_series import US_RISK_FREE_SERIES, upsert_many
            upsert_many(db, US_RISK_FREE_SERIES,
                        [(p.date().isoformat(), float(v))
                         for p, v in df["close"].items() if math.isfinite(float(v))])
        except Exception as e:
            print(f"[risk_free] US03MY history append failed: {type(e).__name__}")
        return value
    except Exception:
        return None


def _fetch_interest_rate(db):
    """Read CBE interest rate from settings table."""
    return {"value": read_risk_free_rate(db), "direction": "stable"}


def _store_macro(db, data):
    """Store macro data in the Postgres cache."""
    try:
        now = datetime.utcnow().isoformat()
        for key, info in data.items():
            val = info.get("value")
            change = info.get("change_pct")
            # For EGX30 store monthly_change_pct in previous_value so it survives caching
            prev_val = info.get("monthly_change_pct") if key == "egx30" else info.get("previous_value")
            if val is not None:
                db.execute(
                    "INSERT INTO macro_data (key, value, previous_value, change_pct, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET "
                    "value = EXCLUDED.value, "
                    "previous_value = EXCLUDED.previous_value, "
                    "change_pct = EXCLUDED.change_pct, "
                    "updated_at = EXCLUDED.updated_at",
                    (key, val, prev_val, change, now)
                )
        db.commit()
    except Exception:
        pass


def _format_cached(cached, db):
    """Format cached macro_data rows into the response structure."""
    result = {}

    if "egx30" in cached:
        c = cached["egx30"]
        monthly = c.get("previous_value")  # stored in previous_value during _store_macro
        if monthly is not None:
            monthly = float(monthly)
        direction = ("up" if (monthly and monthly > MACRO_TREND_UP_PCT) else
                     "down" if (monthly and monthly < MACRO_TREND_DOWN_PCT) else "stable")
        result["egx30"] = {
            "value": c["value"],
            "change_pct": c["change_pct"],
            "monthly_change_pct": monthly,
            "direction": direction,
            "trend": ("bullish" if (monthly and monthly > MACRO_TREND_UP_PCT) else
                      "bearish" if (monthly and monthly < MACRO_TREND_DOWN_PCT) else "sideways"),
        }
    else:
        result["egx30"] = {"value": None, "change_pct": None, "direction": None}

    if "usd_egp" in cached:
        c = cached["usd_egp"]
        result["usd_egp"] = {
            "value": c["value"],
            "change_pct": c["change_pct"],
            "direction": "up" if (c["change_pct"] and c["change_pct"] > USDEGP_DIRECTION_THRESHOLD_PCT) else
                         "down" if (c["change_pct"] and c["change_pct"] < -USDEGP_DIRECTION_THRESHOLD_PCT) else "stable",
        }
    else:
        result["usd_egp"] = {"value": None, "change_pct": None, "direction": None}

    result["interest_rate"] = _fetch_interest_rate(db)

    return result
