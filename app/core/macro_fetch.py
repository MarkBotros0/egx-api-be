"""
Macro data helper — fetches and caches Egyptian macro indicators.

Data sources:
  - EGX30: via egxpy (reliable, same data source as the rest of the app)
  - USD/EGP: via egxpy or public source (best-effort)
  - CBE interest rate: stored in settings table (manually updateable)

All scraping is wrapped in try/except — returns partial/null data, never crashes.
"""

from datetime import datetime, timedelta

from app.core.constants import (
    DEFAULT_RISK_FREE_RATE_PCT,
    MACRO_CACHE_TTL_SECONDS,
    MACRO_TREND_DOWN_PCT,
    MACRO_TREND_UP_PCT,
    USDEGP_DIRECTION_THRESHOLD_PCT,
)


def fetch_macro(db):
    """
    Fetch macro data, using Turso cache if fresh enough.
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


def _fetch_egx30():
    """Fetch EGX30 index data via egxpy."""
    try:
        from egxpy.download import get_OHLCV_data
        df = get_OHLCV_data("EGX30", "EGX", "Daily", 30)
        if df is None or df.empty:
            return {"value": None, "change_pct": None, "direction": None}
        df.columns = [c.lower() for c in df.columns]
        current = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else None
        change_pct = ((current - prev) / prev * 100) if prev else None

        # Monthly trend (compare to 20 trading days ago)
        month_ago = float(df["close"].iloc[-min(20, len(df))]) if len(df) >= 5 else None
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
    except Exception:
        return {"value": None, "change_pct": None, "direction": None}


def _fetch_usdegp():
    """Fetch USD/EGP exchange rate — best-effort via egxpy."""
    try:
        from egxpy.download import get_OHLCV_data
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


def _fetch_interest_rate(db):
    """Read CBE interest rate from settings table."""
    try:
        row = db.execute("SELECT value FROM settings WHERE key = 'risk_free_rate'").fetchone()
        rate = float(row[0]) if row else float(DEFAULT_RISK_FREE_RATE_PCT)
        return {
            "value": rate,
            "direction": "stable",
        }
    except Exception:
        return {"value": float(DEFAULT_RISK_FREE_RATE_PCT), "direction": "stable"}


def _store_macro(db, data):
    """Store macro data in Turso cache."""
    try:
        now = datetime.utcnow().isoformat()
        for key, info in data.items():
            val = info.get("value")
            change = info.get("change_pct")
            # For EGX30 store monthly_change_pct in previous_value so it survives caching
            prev_val = info.get("monthly_change_pct") if key == "egx30" else info.get("previous_value")
            if val is not None:
                db.execute(
                    "INSERT OR REPLACE INTO macro_data (key, value, previous_value, change_pct, updated_at) VALUES (?, ?, ?, ?, ?)",
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
