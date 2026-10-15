"""The shared scanner helper: default body unchanged, plus symbol sets, tickers, filters, sort."""
import json

from app.core import tradingview
from app.core.markets import EG, US


class _Resp:
    def __init__(self, payload): self._b = json.dumps(payload).encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _capture(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return _Resp({"data": [{"s": "NYSE:KO", "d": ["KO"]}]})

    monkeypatch.setattr(tradingview.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_scan_default_body_unchanged(monkeypatch):
    seen = _capture(monkeypatch)
    tradingview.scan(["name"], market=EG)
    assert seen["body"]["filter"] == [
        {"left": "exchange", "operation": "equal", "right": "EGX"},
        {"left": "type", "operation": "equal", "right": "stock"},
    ]
    assert "symbols" not in seen["body"]


def test_scan_symbolset(monkeypatch):
    seen = _capture(monkeypatch)
    tradingview.scan(["name"], market=US, symbolset="SYML:SP;SPX", limit=600)
    assert seen["url"].endswith("/america/scan")
    assert seen["body"]["symbols"] == {"symbolset": ["SYML:SP;SPX"]}
    assert seen["body"]["range"] == [0, 600]
    assert "filter" not in seen["body"]


def test_scan_tickers_and_extra_filter_and_sort(monkeypatch):
    seen = _capture(monkeypatch)
    tradingview.scan(["name"], market=US, tickers=["NYSE:KO"])
    assert seen["body"]["symbols"] == {"tickers": ["NYSE:KO"]}
    seen = _capture(monkeypatch)
    f = {"left": "description", "operation": "match", "right": "coca"}
    tradingview.scan(["name"], market=US, extra_filter=[f], sort_by="market_cap_basic", limit=20)
    assert seen["body"]["filter"][-1] == f
    assert seen["body"]["filter"][1] == {"left": "type", "operation": "in_range", "right": ["stock", "fund"]}
    assert seen["body"]["sort"] == {"sortBy": "market_cap_basic", "sortOrder": "desc"}
