import json

import pandas as pd

from app.core import benchmark, tradingview
from app.core.markets import EG, US


def test_scan_url_per_market():
    assert tradingview.scan_url(EG) == "https://scanner.tradingview.com/egypt/scan"
    assert tradingview.scan_url(US) == "https://scanner.tradingview.com/america/scan"


def test_scan_filters_on_market_exchanges(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"data": []}'

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"] = req.full_url, json.loads(req.data)
        return Resp()

    monkeypatch.setattr(tradingview.urllib.request, "urlopen", fake_urlopen)
    tradingview.scan(["close"], market=US)
    assert seen["url"].endswith("/america/scan")
    assert {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]} in seen["body"]["filter"]


def test_scan_default_market_is_byte_identical_to_today(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"data": []}'

    monkeypatch.setattr(tradingview.urllib.request, "urlopen",
                        lambda req, timeout: (seen.setdefault("body", json.loads(req.data)), Resp())[1])
    tradingview.scan(["close"])
    assert {"left": "exchange", "operation": "equal", "right": "EGX"} in seen["body"]["filter"]


def test_benchmark_uses_market_symbol_and_caches(monkeypatch):
    calls = []
    df = pd.DataFrame({"Close": [1.0, 2.0]})

    def fake_ohlcv(symbol, exchange, interval, bars):
        calls.append((symbol, exchange, interval, bars))
        return df.copy()

    monkeypatch.setattr(benchmark, "_get_ohlcv", fake_ohlcv)
    benchmark.cache_clear_for_tests()
    s1 = benchmark.get_benchmark_close(US, "Daily", 400)
    s2 = benchmark.get_benchmark_close(US, "Daily", 400)
    assert list(s1) == [1.0, 2.0] and list(s2) == [1.0, 2.0]
    assert calls == [("SPX", "SP", "Daily", 400)]


def test_benchmark_returns_none_on_feed_failure(monkeypatch):
    def boom(*a):
        raise RuntimeError("feed down")
    monkeypatch.setattr(benchmark, "_get_ohlcv", boom)
    benchmark.cache_clear_for_tests()
    assert benchmark.get_benchmark_close(EG, "Daily", 400) is None
