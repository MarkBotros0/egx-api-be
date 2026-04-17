# P/E Ratio — Deferred Implementation Plan

Status: **blocked** — waiting for a saved copy of https://www.egx.com.eg/en/MarketPECompanies.aspx
(the site was unreachable at plan time, so the HTML parser cannot be written safely).

Once you have that fixture, this doc is the implementation checklist.

---

## Why this is deferred

The scraper must match real EGX page markup. Writing a parser against a guess
is a silent-failure risk: 80% of rows can fail to match and the feature still
"looks working" on the dashboard. We will not do that.

**Unblock step:** once `egx.com.eg` is reachable, open
`https://www.egx.com.eg/en/MarketPECompanies.aspx` in a browser, right-click
→ `Save as...` → select "Webpage, HTML only", and save it to:

```
egx-api-be/tests/fixtures/market_pe_companies.html
```

Commit that file. The parser will be written against it.

---

## Scope summary

A nightly scrape of the EGX P/E page → a Postgres table → served via
`GET /api/pe` → shown on stock detail, holdings rows, and portfolio advice.
If a nightly fetch fails, the **last successfully-fetched values stay in the
DB and continue to serve** — never wiped.

---

## 1. Database schema

Add to `egx-api-be/app/core/db.py::init_db`:

```sql
CREATE TABLE IF NOT EXISTS pe_data (
    symbol TEXT PRIMARY KEY,
    pe_ratio DOUBLE PRECISION,
    eps DOUBLE PRECISION,
    close_price DOUBLE PRECISION,
    data_date TEXT,              -- the "as of" date EGX prints on the page
    updated_at TEXT NOT NULL     -- when WE last successfully saved this row
);
CREATE INDEX IF NOT EXISTS idx_pe_data_updated ON pe_data(updated_at);
```

And a tracking row in `settings`:

```sql
-- seeded with empty value; refresh job writes ISO timestamp on each success
INSERT INTO settings (key, value) VALUES ('pe_last_successful_fetch', '')
ON CONFLICT (key) DO NOTHING;
INSERT INTO settings (key, value) VALUES ('pe_last_attempt_status', '')
ON CONFLICT (key) DO NOTHING;
```

The `pe_last_successful_fetch` ISO string is the freshness signal the UI
reads. `pe_last_attempt_status` records `ok` / `error: <message>` from the
last refresh run, regardless of outcome, so the UI can show "stale since X,
last attempt failed with Y".

**Do NOT wipe `pe_data` on a failed refresh.** Upsert each row as it comes in;
rows that don't appear in the latest fetch keep their old values until the
next successful appearance. That preserves last-known-good coverage when EGX
temporarily drops a ticker or the scrape fails.

---

## 2. Scraper module

New file: `egx-api-be/app/core/pe_fetch.py`

```python
"""
EGX P/E scraper.

Fetches https://www.egx.com.eg/en/MarketPECompanies.aspx, parses the
listed-companies P/E table, and upserts rows into pe_data.

Defensive: on fetch/parse failure, logs to settings.pe_last_attempt_status
and returns without modifying the existing pe_data rows. The read path
(get_pe_for_symbol below) always returns whatever was last stored.
"""

import re
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup  # requires: pip install beautifulsoup4 lxml

PE_URL = "https://www.egx.com.eg/en/MarketPECompanies.aspx"
HTTP_TIMEOUT_SECONDS = 20


def fetch_pe_from_egx() -> dict[str, dict]:
    """
    Returns {symbol: {"pe_ratio": float|None, "eps": float|None,
                       "close_price": float|None, "data_date": str|None}}.

    Raises on HTTP failure or unparseable HTML — caller handles fallback.
    """
    # IMPORTANT: implement AGAINST the saved HTML fixture first, then against
    # the live URL. See egx-api-be/tests/fixtures/market_pe_companies.html.
    resp = httpx.get(
        PE_URL,
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": "Mozilla/5.0 (compatible; EGX-Analytics/1.0)"},
    )
    resp.raise_for_status()
    return parse_pe_html(resp.text)


def parse_pe_html(html: str) -> dict[str, dict]:
    """
    Parse the saved-page HTML. WRITE THIS FUNCTION AGAINST THE FIXTURE.

    Expected columns (verify from fixture, adjust if wrong):
      - Symbol / Reuters Code
      - Company Name
      - Close Price
      - EPS
      - P/E Ratio
    """
    soup = BeautifulSoup(html, "lxml")
    # TODO: locate the correct <table>. The page likely has multiple tables
    # (header, navigation, the main P/E table). Prefer selecting by id or by
    # column-header text ("P/E").
    raise NotImplementedError(
        "Parser not yet written — see egx-api-be/docs/pe_ratio_plan.md"
    )


def refresh_pe_data(db) -> dict:
    """
    Called by the cron handler / manual refresh route.
    Never wipes existing rows on failure.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        rows = fetch_pe_from_egx()
    except Exception as e:
        _write_setting(db, "pe_last_attempt_status", f"error: {type(e).__name__}: {e}")
        return {"success": False, "count": 0, "error": str(e)}

    count = 0
    for symbol, info in rows.items():
        db.execute(
            """
            INSERT INTO pe_data (symbol, pe_ratio, eps, close_price, data_date, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                pe_ratio    = EXCLUDED.pe_ratio,
                eps         = EXCLUDED.eps,
                close_price = EXCLUDED.close_price,
                data_date   = EXCLUDED.data_date,
                updated_at  = EXCLUDED.updated_at
            """,
            (symbol, info.get("pe_ratio"), info.get("eps"),
             info.get("close_price"), info.get("data_date"), now),
        )
        count += 1

    _write_setting(db, "pe_last_successful_fetch", now)
    _write_setting(db, "pe_last_attempt_status", "ok")
    return {"success": True, "count": count}


def get_pe_for_symbol(db, symbol: str) -> dict | None:
    """
    Read-side helper. Returns the last stored P/E row or None.
    Never calls the network — the refresh job owns that.
    """
    row = db.execute(
        "SELECT pe_ratio, eps, close_price, data_date, updated_at "
        "FROM pe_data WHERE symbol = %s",
        (symbol,),
    ).fetchone()
    if not row:
        return None
    return {
        "pe_ratio": row[0],
        "eps": row[1],
        "close_price": row[2],
        "data_date": row[3],
        "fetched_at": row[4],
    }


def _write_setting(db, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO settings (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )
```

### Symbol matching — planning note

EGX's P/E page almost certainly uses Reuters-style codes (e.g. `COMI.CA`),
while `egxpy` uses bare symbols (e.g. `COMI`). Once the fixture is available:

1. Extract a sample of codes from the scraped table and compare to
   `get_EGXdata()` output from egxpy.
2. If they differ, add a normalization step (strip `.CA` suffix / map prefixes)
   inside `parse_pe_html` so `pe_data.symbol` matches the rest of the app.
3. Log any symbols in the scrape that **don't** correspond to an egxpy ticker
   — that is your coverage blind spot.

---

## 3. Router

New file: `egx-api-be/app/routers/pe.py`

```python
from fastapi import APIRouter, HTTPException, Header
import os

from app.core.db import get_db
from app.core.pe_fetch import refresh_pe_data, get_pe_for_symbol

router = APIRouter()


@router.get("/api/pe")
def get_pe(symbol: str | None = None):
    db = get_db()
    if symbol:
        data = get_pe_for_symbol(db, symbol.upper())
        if not data:
            raise HTTPException(status_code=404, detail=f"No P/E data for {symbol}")
        return {"symbol": symbol.upper(), **data}
    # No symbol: return all rows + freshness metadata
    rows = db.execute(
        "SELECT symbol, pe_ratio, eps, close_price, data_date, updated_at FROM pe_data"
    ).fetchall()
    last_row = db.execute(
        "SELECT value FROM settings WHERE key = 'pe_last_successful_fetch'"
    ).fetchone()
    status_row = db.execute(
        "SELECT value FROM settings WHERE key = 'pe_last_attempt_status'"
    ).fetchone()
    return {
        "data": [
            {"symbol": r[0], "pe_ratio": r[1], "eps": r[2],
             "close_price": r[3], "data_date": r[4], "fetched_at": r[5]}
            for r in rows
        ],
        "last_successful_fetch": last_row[0] if last_row else None,
        "last_attempt_status": status_row[0] if status_row else None,
    }


@router.post("/api/pe/refresh")
def trigger_refresh(x_refresh_secret: str | None = Header(default=None)):
    """
    Manual + cron-triggered refresh.
    Protected by PE_REFRESH_SECRET env var; required in production.
    """
    expected = os.environ.get("PE_REFRESH_SECRET")
    if expected and x_refresh_secret != expected:
        raise HTTPException(status_code=403, detail="Forbidden")
    db = get_db()
    return refresh_pe_data(db)
```

Register in `app/main.py`:

```python
from app.routers import pe as pe_router
app.include_router(pe_router.router)
```

---

## 4. Daily refresh

Add `vercel.json` in the backend deployment directory:

```json
{
  "crons": [
    {
      "path": "/api/pe/refresh",
      "schedule": "0 4 * * *"
    }
  ]
}
```

Runs 04:00 UTC daily (≈06:00 Cairo, before EGX opens at 10:00). Vercel Cron
on the Hobby plan runs once per day, which fits our once-daily freshness need.

Set `PE_REFRESH_SECRET` as a Vercel env var and forward it via header if you
use a non-Vercel external scheduler.

---

## 5. Composite score integration

P/E goes into the `quality` category (no new 9th weight slider needed —
`compute_composite` already renormalizes when a scorer's input is missing).

In `app/core/composite.py::score_quality`, accept a new `pe_ratio` extra:

```python
# P/E sub-component: reward 0 < pe < 20 (fair/cheap), neutral 20-30,
# penalize >30 or negative (loss-making). When pe is None the band is skipped
# and the score is built from the existing multi_timeframe/consistency/drawdown
# components alone.
```

Thresholds (Egypt context: inflation-linked interest rate ~25%, so P/E
expectations are tighter than developed markets):
- `pe < 0` → score 15, reason "Company is loss-making"
- `0 <= pe < 10` → score 85, reason "Very cheap on earnings (P/E < 10)"
- `10 <= pe < 20` → score 70, reason "Reasonably valued (P/E 10–20)"
- `20 <= pe < 30` → score 50, reason "Fully valued (P/E 20–30)"
- `pe >= 30` → score 25, reason "Expensive on earnings (P/E > 30)"

Combine with existing quality inputs as an equal-weight average.

Pass `pe_ratio` into `extras` in `analysis.py` and `portfolio_analysis.py`:

```python
pe_info = get_pe_for_symbol(db, symbol)
extras["pe_ratio"] = pe_info["pe_ratio"] if pe_info else None
```

---

## 6. Signals (portfolio_analysis.py)

```python
if pe_info and pe_info.get("pe_ratio") is not None:
    pe = pe_info["pe_ratio"]
    if pe < 0:
        signals.append({
            "type": "pe_loss_making", "severity": "warning", "symbol": symbol,
            "message": f"{symbol} is loss-making (negative P/E).",
            "explanation": "A negative P/E means the company isn't profitable. "
                           "Earnings-based valuation doesn't apply — rely on "
                           "trend and relative strength instead.",
            "learn_concept": "pe_ratio",
        })
    elif pe < 10:
        signals.append({
            "type": "pe_undervalued", "severity": "opportunity", "symbol": symbol,
            "message": f"{symbol} P/E is {pe:.1f} — potentially undervalued.",
            "explanation": "A low P/E can indicate value, but also that the "
                           "market expects earnings to fall. Combine with "
                           "trend and relative strength before acting.",
            "learn_concept": "pe_ratio",
        })
    elif pe > 30:
        signals.append({
            "type": "pe_overvalued", "severity": "warning", "symbol": symbol,
            "message": f"{symbol} P/E is {pe:.1f} — expensive vs earnings.",
            "explanation": "A high P/E means the market is paying a lot for "
                           "each EGP of earnings. Only justified by strong "
                           "growth expectations.",
            "learn_concept": "pe_ratio",
        })
```

---

## 7. Frontend

Types (`egx-api-fe/src/app/lib/types.ts`):

```typescript
export interface PEData {
  pe_ratio: number | null;
  eps: number | null;
  close_price: number | null;
  data_date: string | null;
  fetched_at: string;
}

// Add to AnalysisResponse:
pe?: PEData | null;

// Add to HoldingAnalysis:
pe?: PEData | null;
```

Backend wires `pe` field into both response shapes alongside the other
extras in `analysis.py` / `portfolio_analysis.py`.

UI surfaces:
1. **Stock detail** — new row inside `StatsPanel`:
   `P/E Ratio: 12.3  (EPS 4.52 EGP · as of 2026-04-16)` with a `LearnTooltip`.
2. **HoldingsTable** — add `P/E` column (desktop) and a field in the mobile
   card. Color-coded: green <15, amber 15–30, red >30 or negative.
3. **Dashboard `StockCard`** — optional: a tiny `P/E 12.3` chip under the
   sparkline. Cheap (one DB read).
4. **Freshness banner** — if `pe_last_successful_fetch` is >48h old, show
   a small warning strip on the stock detail + portfolio pages: "P/E data
   last updated X days ago — EGX feed may be temporarily unavailable."

---

## 8. Learn page

Add a new Concept card in `egx-api-fe/src/app/learn/page.tsx`, inside
the **Market Basics** or new **Fundamentals** section:

```tsx
<Concept
  id="pe_ratio"
  title="P/E Ratio (Price-to-Earnings)"
  definition="How many EGP investors are paying for every 1 EGP of annual profit."
  whyItMatters="..."
  howToUse="..."
/>
```

Keep the tone beginner-friendly and mention the Egypt context:
> In markets with T-bill rates around 25%, a P/E above ~20 needs strong
> growth expectations to be worth it versus just holding cash.

---

## 9. CLAUDE.md

Add:
- `pe_data` table to the schema block.
- `pe_fetch.py` to the core module list.
- `/api/pe` and `/api/pe/refresh` under API Endpoints.
- `pe_ratio` to the list of Learn anchors and the composite `quality`
  category input list.

---

## Acceptance checklist

- [ ] `tests/fixtures/market_pe_companies.html` committed
- [ ] `parse_pe_html` works against the fixture; round-trips ≥ 200 rows
- [ ] Symbol match rate vs egxpy verified (log of unmatched rows)
- [ ] `pe_data` + `settings` rows migrate cleanly on a fresh DB
- [ ] `/api/pe/refresh` with correct secret returns `{success: true, count: N}`
- [ ] `/api/pe/refresh` on upstream failure preserves prior `pe_data` rows
- [ ] Stock detail, holdings table, and Learn page all render P/E
- [ ] Vercel cron fires once and the settings row updates
- [ ] Staleness banner appears after simulated 48h gap
