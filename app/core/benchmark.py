"""
The market's benchmark index closes (EGX30 for EG, S&P 500 for US).

Beta, relative strength and the macro regime all compare a stock against this
series, so it is fetched through ONE function under ONE cache key per market —
three hand-rolled copies (analysis, cron, portfolio_analysis) used to exist and
could drift. See CLAUDE.md "One Score Per Stock", rule 4.
"""
from __future__ import annotations

from app.core import cache
from app.core.markets import MarketConfig


def _get_ohlcv(symbol, exchange, interval, bars):
    from app.vendor.egxpy import get_OHLCV_data
    return get_OHLCV_data(symbol, exchange, interval, bars)


def cache_clear_for_tests() -> None:
    for k in [k for k in cache._store if ":benchmark:" in k]:
        del cache._store[k]


def get_benchmark_close(market: MarketConfig, interval: str, bars: int):
    ck = cache.make_key(market.code, "benchmark", interval, bars)
    df = cache.get(ck)
    if df is None:
        try:
            raw = _get_ohlcv(market.benchmark_symbol, market.benchmark_exchange, interval, bars)
        except Exception:
            return None
        if raw is None or raw.empty:
            return None
        raw.columns = [c.lower() for c in raw.columns]
        cache.set(ck, raw)
        df = raw
    return df["close"] if "close" in df.columns else None
