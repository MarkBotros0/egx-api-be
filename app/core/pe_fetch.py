"""
EGX P/E scraper.

The source page (https://www.egx.com.eg/en/MarketPECompanies.aspx) is an
ASP.NET WebForms GridView with extremely stable span IDs. The exposed columns
are only:

    Company Name  |  P/E Ratio  |  DY(%)

No symbol / Reuters code / EPS / close price / data date is present on the
page — so the scraper has to match full English company names to the
egxpy-compatible ticker universe that powers the rest of the app.

Matching precedence (first hit wins):
    1. explicit override in data/egx_pe_name_overrides.json
    2. exact normalized-name match against the universe
    3. unique first-two-tokens prefix match
    4. token-jaccard ≥ 0.6 best match

Rows where the page prints P/E = "0" are stored as NULL (EGX uses 0 as a
"no earnings data / loss-making" sentinel — storing 0.0 would trick
score_quality into reading the stock as "very cheap", which is the exact
silent-failure mode we want to avoid).

On fetch or parse failure, the existing pe_data rows are left untouched;
the read path always returns last-known-good.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from typing import Optional


PE_URL = "https://www.egx.com.eg/en/MarketPECompanies.aspx"
HTTP_TIMEOUT_SECONDS = 20

_OVERRIDES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "egx_pe_name_overrides.json"
)

# Common corporate suffix/filler words to strip during normalization. Order matters
# for multi-word forms like "s.a.e." — we pre-collapse dots below before matching.
_STOP_WORDS = {
    "sae", "saea", "company", "companies", "co", "corporation", "corp",
    "group", "holding", "holdings", "for", "and", "the", "inc", "ltd",
    "limited", "formerly", "new",
}

# Pattern matches the ASP.NET GridView span IDs:
#   ctl00_C_I_GridView1_ctl02_lblCompanyName
#   ctl00_C_I_GridView1_ctl02_lblPE
#   ctl00_C_I_GridView1_ctl02_lblYEILD
_ROW_PATTERN = re.compile(
    r'<span id="ctl00_C_I_GridView1_ctl(\d+)_lbl(CompanyName|PE|YEILD)">([^<]*)</span>'
)


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """
    Aggressively normalize a company name for matching.

    - Lowercase
    - Drop parenthetical aliases ("(formerly ...)", "(Kabo)")
    - Replace punctuation with spaces
    - Drop common corporate filler words
    - Collapse whitespace
    """
    s = (name or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[.,&\-/–—]", " ", s)
    tokens = [t for t in s.split() if t and t not in _STOP_WORDS]
    return " ".join(tokens)


def _first_tokens(name: str, n: int = 2) -> str:
    tokens = _normalize_name(name).split()
    return " ".join(tokens[:n]) if tokens else ""


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _load_overrides() -> dict:
    """Return {EXACT_COMPANY_NAME_AS_ON_PAGE: SYMBOL}. Empty dict if file missing."""
    try:
        with open(_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v.upper() for k, v in data.items() if isinstance(v, str)}
    except Exception:
        return {}


def _load_universe() -> dict:
    """
    Build {symbol: name} from the ticker universe used by the rest of the app
    (static JSON + TradingView live fetch + discovered_tickers).

    Falls back to static JSON only if `tickers` router is unavailable.
    """
    try:
        from app.routers.tickers import _load_tickers
        return {t["symbol"].upper(): t["name"] for t in _load_tickers()}
    except Exception:
        try:
            path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "egx_tickers.json"
            )
            with open(path, "r", encoding="utf-8") as f:
                return {t["symbol"].upper(): t["name"] for t in json.load(f)}
        except Exception:
            return {}


def match_symbol(company_name: str, universe: dict, overrides: dict,
                 jaccard_floor: float = 0.6) -> Optional[str]:
    """
    Resolve an EGX-page company name to a ticker symbol. Returns None if
    no confident match can be made.
    """
    if not company_name:
        return None

    # 1. Explicit override
    sym = overrides.get(company_name) or overrides.get(company_name.strip())
    if sym:
        return sym.upper()

    nk = _normalize_name(company_name)
    if not nk:
        return None

    # Index the universe by normalized name and by first-two-tokens
    norm_to_syms: dict = {}
    prefix_to_syms: dict = {}
    for s, name in universe.items():
        k = _normalize_name(name)
        norm_to_syms.setdefault(k, []).append(s)
        pk = _first_tokens(name, 2)
        prefix_to_syms.setdefault(pk, []).append(s)

    # 2. Exact normalized match, only if unambiguous
    if nk in norm_to_syms and len(norm_to_syms[nk]) == 1:
        return norm_to_syms[nk][0]

    # 3. Unique first-two-tokens prefix
    pk = _first_tokens(company_name, 2)
    if pk and pk in prefix_to_syms and len(prefix_to_syms[pk]) == 1:
        return prefix_to_syms[pk][0]

    # 4. Token-jaccard best-match
    best_score = 0.0
    best_sym = None
    for cand_norm, syms in norm_to_syms.items():
        if len(syms) != 1:
            continue
        j = _token_jaccard(nk, cand_norm)
        if j > best_score:
            best_score = j
            best_sym = syms[0]
    if best_score >= jaccard_floor:
        return best_sym
    return None


# ---------------------------------------------------------------------------
# HTML fetch + parse
# ---------------------------------------------------------------------------

def fetch_pe_html() -> str:
    """Fetch the live EGX P/E page. Raises on HTTP error."""
    req = urllib.request.Request(
        PE_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def parse_pe_html(html: str) -> list:
    """
    Parse the P/E GridView into a list of row dicts:
        [{"company_name": str, "pe_ratio": float|None, "dividend_yield": float|None}]

    P/E == 0 on the page is returned as None (see module docstring). Same for DY%
    when it appears as "0".
    """
    rows: dict = {}
    for ctl_idx, field, raw_value in _ROW_PATTERN.findall(html):
        # ASP.NET sometimes HTML-encodes ampersands in names — decode the
        # most common entities without pulling in an HTML parser dependency.
        value = (
            raw_value.replace("&amp;", "&")
            .replace("&nbsp;", " ")
            .replace("&#39;", "'")
            .strip()
        )
        row = rows.setdefault(ctl_idx, {})
        if field == "CompanyName":
            row["company_name"] = value
        elif field == "PE":
            row["pe_ratio"] = _parse_float_or_none(value)
        elif field == "YEILD":
            row["dividend_yield"] = _parse_float_or_none(value)

    out = []
    for row in rows.values():
        if not row.get("company_name"):
            continue
        out.append({
            "company_name": row["company_name"],
            "pe_ratio": row.get("pe_ratio"),
            "dividend_yield": row.get("dividend_yield"),
        })
    return out


def _parse_float_or_none(value: str) -> Optional[float]:
    """Parse a P/E / DY cell. '0', '', '-' and non-numeric → None."""
    if value is None:
        return None
    v = value.strip()
    if not v or v in {"-", "--", "N/A", "n/a"}:
        return None
    try:
        f = float(v.replace(",", ""))
    except ValueError:
        return None
    if f == 0.0:
        return None
    return f


# ---------------------------------------------------------------------------
# DB writes / reads
# ---------------------------------------------------------------------------

def refresh_pe_data(db, html: Optional[str] = None) -> dict:
    """
    Fetch the EGX P/E page and upsert matched rows into pe_data.

    Passing `html` (e.g. a test fixture) skips the HTTP call — used by tests
    and local dev to avoid hitting egx.com.eg.

    Never wipes existing rows. On fetch/parse failure, existing rows remain
    and `pe_last_attempt_status` is updated to the error.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        if html is None:
            html = fetch_pe_html()
        parsed = parse_pe_html(html)
    except Exception as e:
        _write_setting(db, "pe_last_attempt_status", f"error: {type(e).__name__}: {e}")
        return {"success": False, "count": 0, "error": str(e)}

    if not parsed:
        _write_setting(db, "pe_last_attempt_status", "error: no rows parsed")
        return {"success": False, "count": 0, "error": "no rows parsed"}

    universe = _load_universe()
    overrides = _load_overrides()

    matched = 0
    unmatched_names = []
    for row in parsed:
        cname = row["company_name"]
        symbol = match_symbol(cname, universe, overrides)
        if symbol is None:
            unmatched_names.append(cname)
            continue
        db.execute(
            """
            INSERT INTO pe_data (symbol, company_name, pe_ratio, dividend_yield, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                company_name   = EXCLUDED.company_name,
                pe_ratio       = EXCLUDED.pe_ratio,
                dividend_yield = EXCLUDED.dividend_yield,
                updated_at     = EXCLUDED.updated_at
            """,
            (symbol, cname, row.get("pe_ratio"), row.get("dividend_yield"), now),
        )
        matched += 1

    _write_setting(db, "pe_last_successful_fetch", now)
    _write_setting(db, "pe_last_attempt_status", "ok")
    _write_setting(db, "pe_unmatched_names", "\n".join(unmatched_names))

    return {
        "success": True,
        "count": matched,
        "total_rows": len(parsed),
        "unmatched": len(unmatched_names),
    }


def get_pe_for_symbol(db, symbol: str) -> Optional[dict]:
    """Read-side helper. Returns the last stored P/E row or None."""
    row = db.execute(
        "SELECT company_name, pe_ratio, dividend_yield, updated_at "
        "FROM pe_data WHERE symbol = %s",
        (symbol.upper(),),
    ).fetchone()
    if not row:
        return None
    return {
        "company_name": row[0],
        "pe_ratio": row[1],
        "dividend_yield": row[2],
        "fetched_at": row[3],
    }


def _write_setting(db, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO settings (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )
