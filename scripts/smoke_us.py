"""Manual live check of the US data path. Run: .venv/bin/python scripts/smoke_us.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.markets import US
from app.core import universe as U
from app.core.tradingview import scan

def main():
    counts = {}
    for idx in US.indices:
        counts[idx.id] = len(scan(["name", "description", "sector", "type"], market=US, symbolset=idx.symbolset, limit=1000))
    print("index sizes", counts)
    assert counts["SPX"] >= 495 and counts["NDX"] >= 95 and counts["DJI"] == 30
    for sym, ex in [("AAPL", "NASDAQ"), ("BRK.B", "NYSE"), ("VOO", "AMEX")]:
        got = U.lookup_listing(US, sym)
        print(sym, got)
        assert got and got["exchange"] == ex
    assert U.lookup_listing(US, "ZZZZQX") is None
    print("search coca:", [r["symbol"] for r in U.search_symbols(US, "coca")][:5])
    from app.vendor.egxpy import get_OHLCV_data
    for sym, ex in [("BRK.B", "NYSE"), ("SPX", "SP"), ("US03MY", "TVC")]:
        df = get_OHLCV_data(sym, ex, "Daily", 5)
        print(sym, ex, None if df is None else float(df["close"].iloc[-1]))
        assert df is not None and not df.empty
    print("OK")

if __name__ == "__main__":
    main()
