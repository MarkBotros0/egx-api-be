"""
Per-market facts the OFFLINE scripts need: where the cache lives, which series
the backtest benchmarks against, which index members the regime averages.

The live app's facts live in app/core/markets.py. The one deliberate
difference is the US backtest benchmark: the live app uses SP:SPX, but
tvDatafeed serves SPX only from 2016-07 (2,587 bars), while AMEX:SPY tracks the
S&P 500 back to 2006-11 (5,000 bars). The backtest calendar, beta, relative
strength and the regime's forward returns therefore use SPY, and the bundle
says so.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
EG_CACHE_DIR = os.path.join(SCRIPTS_DIR, ".cache")
UNIVERSE_FILE = "universe.json"


@dataclass(frozen=True)
class CacheSpec:
    code: str
    cache_dir: str
    benchmark_symbol: str
    benchmark_exchange: str
    risk_free_series: str
    regime_indexes: tuple
    regime_universe_label: str
    frame_exclude: tuple = ()


def spec_for(code: str, cache_dir: str | None = None) -> CacheSpec:
    c = (code or "EG").strip().upper()
    if c == "EG":
        return CacheSpec("EG", cache_dir or EG_CACHE_DIR, "EGX30", "EGX", "EGINTR",
                         ("EGX30", "EGX70"), "EGX30 and EGX70")
    if c == "US":
        return CacheSpec("US", cache_dir or os.path.join(EG_CACHE_DIR, "US"), "SPY", "AMEX",
                         "US03MY", ("SPX",), "S&P 500", frame_exclude=("SPY",))
    raise ValueError(f"no backtest cache spec for market {code!r}")


def read_universe_rows(spec: CacheSpec) -> list:
    """[{symbol, exchange, indices}] — EG from the ticker file, else the cached universe.json."""
    if spec.code == "EG":
        with open(os.path.join(REPO_ROOT, "data", "egx_tickers.json"), encoding="utf-8") as f:
            return [{"symbol": t["symbol"].upper(), "exchange": "EGX",
                     "indices": [t["index"].upper()] if t.get("index") else []}
                    for t in json.load(f)]
    with open(os.path.join(spec.cache_dir, UNIVERSE_FILE), encoding="utf-8") as f:
        return json.load(f)


def write_universe_rows(spec: CacheSpec, rows: list) -> None:
    os.makedirs(spec.cache_dir, exist_ok=True)
    with open(os.path.join(spec.cache_dir, UNIVERSE_FILE), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)


def regime_members(spec: CacheSpec) -> set:
    """The symbols /api/market_regime averages over for this market."""
    want = set(spec.regime_indexes)
    return {r["symbol"].upper() for r in read_universe_rows(spec)
            if want & set(r.get("indices") or [])}
