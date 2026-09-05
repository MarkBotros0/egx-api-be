"""
Per-stock dividend history — dated, multi-year — from Yahoo's chart API.

WHY YAHOO, AND WHY IT IS THE ONLY OPTION
----------------------------------------
The EGX publishes no machine-readable dividend calendar (its site is behind an
F5 bot challenge — see news_fetch). The TradingView scanner pe_fetch already
calls carries only the ONE most-recent coupon per stock. `fundamentals_annual`
has annual DPS *amounts* but no ex-dates. So the one keyless source of DATED,
multi-year dividend history is Yahoo:

    https://query1.finance.yahoo.com/v8/finance/chart/<SYM>.CA?events=div

Verified live: COMI.CA 15 dividends back to 2010, SWDY.CA 16, EFID.CA 12, and
the amounts match EGX's own filings (COMI 6.00 on ~7 Apr 2026). One cheap HTTP
GET per symbol, so this rides an on-demand, self-fetching card — no table.

THE HONESTY LINE: Yahoo gives the EX-DATE and AMOUNT of PAST coupons. There is
no reliable forward date anywhere, so anything about the NEXT payment is an
ESTIMATE from history (`summarize_cadence`), never a promise.

`parse_dividends` and `summarize_cadence` are pure — the whole surface tests
without a network call. `fetch_dividends` is the thin GET around them.
"""

from __future__ import annotations

import calendar
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from app.core.constants import DIVIDEND_HISTORY_TIMEOUT_SECONDS

# events=div gives dividends; a wide window so we get the whole history. The
# .CA suffix is Yahoo's code for the Egyptian Exchange.
_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    "?interval=1d&period1=1104537600&period2={p2}&events=div"
)

# A browser-ish UA — Yahoo 429s an unadorned urllib agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def yahoo_symbol(symbol: str) -> str:
    """EGX ticker -> Yahoo code, e.g. 'COMI' -> 'COMI.CA'."""
    return f"{(symbol or '').strip().upper()}.CA"


def parse_dividends(payload) -> list[dict]:
    """
    Yahoo chart JSON -> `[{ex_date: ISO, amount: float, year: int}]`, newest
    first. Empty (not an error) when the stock has never paid — a real fact.

    A row with no usable amount or timestamp is dropped rather than rendered as
    a broken half-row.
    """
    try:
        result = (payload or {}).get("chart", {}).get("result") or []
        events = (result[0] or {}).get("events", {}) if result else {}
        raw = (events or {}).get("dividends", {}) or {}
    except (AttributeError, IndexError, TypeError):
        return []

    out = []
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        amount = entry.get("amount")
        ts = entry.get("date")
        if amount is None or ts is None:
            continue
        try:
            amount = float(amount)
            when = datetime.fromtimestamp(int(ts), timezone.utc)
        except (TypeError, ValueError):
            continue
        out.append({
            "ex_date": when.strftime("%Y-%m-%d"),
            "amount": amount,
            "year": when.year,
        })

    out.sort(key=lambda d: d["ex_date"], reverse=True)
    return out


def summarize_cadence(dividends: list[dict]) -> dict:
    """
    The pattern behind the payments, for the card's estimate line. Everything
    here is descriptive of the PAST — `typical_month` is the mode of past
    ex-date months, not a forecast.

    `payments_per_year` is the modal count of coupons in a year, so a
    semiannual payer reads 2. Null-shaped (not a crash) for an empty history.
    """
    if not dividends:
        return {
            "last_ex_date": None,
            "typical_month": None,
            "typical_month_name": None,
            "payments_per_year": None,
            "count": 0,
        }

    months = [int(d["ex_date"][5:7]) for d in dividends]
    typical_month = Counter(months).most_common(1)[0][0]

    per_year = Counter(d["year"] for d in dividends)
    payments_per_year = Counter(per_year.values()).most_common(1)[0][0]

    return {
        "last_ex_date": dividends[0]["ex_date"],  # newest first
        "typical_month": typical_month,
        "typical_month_name": calendar.month_abbr[typical_month],
        "payments_per_year": payments_per_year,
        "count": len(dividends),
    }


def fetch_dividends(
    symbol: str, timeout: float = DIVIDEND_HISTORY_TIMEOUT_SECONDS
) -> list[dict]:
    """One Yahoo GET for one symbol's dividend history. Raises on HTTP/JSON
    error — the caller owns the never-500 policy."""
    p2 = int(datetime.now(timezone.utc).timestamp()) + 86_400
    url = _URL.format(sym=yahoo_symbol(symbol), p2=p2)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return parse_dividends(payload)


# ---------------------------------------------------------------------------
# Persisted table (dividend_events) — the store the refresh appends to
# ---------------------------------------------------------------------------

def upsert_dividends(db, symbol: str, dividends: list[dict], source: str) -> int:
    """
    Append coupons to `dividend_events`, one row per (symbol, ex_date).
    Idempotent by PRIMARY KEY — a re-seen coupon is a no-op — so the nightly
    refresh can hand its latest coupon straight in and only NEW dates land.
    Returns how many rows were actually inserted.
    """
    if not dividends:
        return 0
    sym = (symbol or "").strip().upper()
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for d in dividends:
        ex_date = d.get("ex_date")
        if not ex_date:
            continue
        cur = db.execute(
            "INSERT INTO dividend_events (symbol, ex_date, amount, source, created_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (symbol, ex_date) DO NOTHING",
            (sym, ex_date, d.get("amount"), source, now),
        )
        # psycopg exposes rowcount; a DO NOTHING that hit the conflict is 0.
        inserted += getattr(cur, "rowcount", 0) or 0
    return inserted


def read_dividends(db, symbol: str) -> list[dict]:
    """Every stored coupon for one symbol, newest ex-date first."""
    rows = db.execute(
        "SELECT ex_date, amount FROM dividend_events WHERE symbol = %s "
        "ORDER BY ex_date DESC",
        ((symbol or "").strip().upper(),),
    ).fetchall()
    return [
        {"ex_date": r[0], "amount": r[1], "year": int(r[0][:4])}
        for r in rows if r[0]
    ]


def read_calendar(db) -> list[dict]:
    """
    Latest coupon per symbol from the table, newest ex-date first — the
    /dividend_calendar read model. One row per symbol via DISTINCT ON.
    """
    rows = db.execute(
        "SELECT DISTINCT ON (symbol) symbol, ex_date, amount "
        "FROM dividend_events ORDER BY symbol, ex_date DESC"
    ).fetchall()
    out = [{"symbol": r[0], "ex_date": r[1], "amount": r[2]} for r in rows if r[1]]
    out.sort(key=lambda d: d["ex_date"], reverse=True)
    return out
