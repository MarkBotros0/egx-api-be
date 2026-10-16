"""scripts/backtest.py per market: cache dir, universe, exchanges, benchmark, macro."""
import os
import pickle

import pandas as pd
import pytest

from calibration_fixtures import make_frame, write_frames, write_json


@pytest.fixture
def bt(monkeypatch):
    import scripts.backtest as bt
    yield bt
    bt.configure("EG")
    monkeypatch.delenv("BACKTEST_MARKET", raising=False)
    monkeypatch.delenv("BACKTEST_CACHE_DIR", raising=False)


def test_specs_per_market():
    from scripts.market_cache import EG_CACHE_DIR, spec_for
    eg, us = spec_for("EG"), spec_for("us")
    assert eg.cache_dir == EG_CACHE_DIR and eg.cache_dir.endswith(os.path.join("scripts", ".cache"))
    assert (eg.benchmark_symbol, eg.benchmark_exchange, eg.risk_free_series) == ("EGX30", "EGX", "EGINTR")
    assert us.cache_dir == os.path.join(EG_CACHE_DIR, "US")
    assert (us.benchmark_symbol, us.benchmark_exchange, us.risk_free_series) == ("SPY", "AMEX", "US03MY")
    assert us.regime_indexes == ("SPX",) and us.frame_exclude == ("SPY",)
    with pytest.raises(ValueError):
        spec_for("XX")


def test_index_member_tier():
    from app.core.index_membership import index_member_tier
    from app.core.markets import US
    assert index_member_tier(US) == "US large-cap"


def test_configure_switches_globals_and_back(bt, tmp_path):
    spec = bt.configure("US", str(tmp_path))
    assert (bt.CACHE_DIR, bt.BENCHMARK) == (str(tmp_path), "SPY")
    assert os.environ["BACKTEST_MARKET"] == "US"
    assert spec.code == "US"
    bt.configure("EG")
    assert bt.BENCHMARK == "EGX30" and bt.CACHE_DIR.endswith(".cache")


def test_fetch_worker_asks_each_symbols_own_exchange(bt, tmp_path, monkeypatch):
    bt.configure("US", str(tmp_path))
    seen = []

    def fake(symbol, exchange, interval, bars):
        seen.append((symbol, exchange, interval, bars))
        return make_frame(30, 1)

    monkeypatch.setattr("app.vendor.egxpy.get_OHLCV_data", fake)
    bt._fetch_worker([("KO", "NYSE"), ("AAPL", "NASDAQ"), ("SPY", "AMEX")])
    assert seen == [("KO", "NYSE", "Daily", 5000), ("AAPL", "NASDAQ", "Daily", 5000),
                    ("SPY", "AMEX", "Daily", 5000)]
    assert os.path.exists(tmp_path / "KO.pkl")
    assert bt.exchange_for("SPY", {}) == "AMEX"
    assert bt.exchange_for("KO", {"KO": "NYSE"}) == "NYSE"
    assert bt.exchange_for("ZZZ", {}) is None


def test_us_universe_from_file_keeps_index_members_and_is_cached(bt, tmp_path):
    spec = bt.configure("US", str(tmp_path))
    uf = tmp_path / "in.json"
    write_json(uf, [
        {"symbol": "ko", "exchange": "NYSE", "indices": ["SPX", "DJI"]},
        {"symbol": "AAPL", "exchange": "NASDAQ", "indices": ["SPX", "NDX", "DJI"]},
        {"symbol": "XYZQ", "exchange": "NYSE", "indices": []},
    ])
    rows = bt.universe_rows_for(spec, str(uf))
    assert [r["symbol"] for r in rows] == ["AAPL", "KO"]
    from scripts.market_cache import read_universe_rows, regime_members
    assert read_universe_rows(spec) == rows
    assert regime_members(spec) == {"AAPL", "KO"}


def test_us_backtest_applies_no_macro_modulation(bt):
    bench = make_frame(300, 3)["close"]
    bt.configure("US")
    assert bt.market_macro(bench) is None
    bt.configure("EG")
    assert set(bt.market_macro(bench)) == {"egx30"}


def test_quick_us_run_on_a_fixture_cache_writes_a_panel(bt, tmp_path):
    cache = tmp_path / "US"
    frames = {s: make_frame(700, i) for i, s in enumerate(["AAA", "BBB", "CCC"])}
    frames["SPY"] = make_frame(700, 99, price=400.0)
    write_frames(str(cache), frames)
    with open(cache / "policy_rate.pkl", "wb") as f:
        pickle.dump([], f)                       # no DB: flat rate, withheld
    uf = tmp_path / "u.json"
    write_json(uf, [{"symbol": s, "exchange": "NYSE", "indices": ["SPX"]}
                    for s in ("AAA", "BBB", "CCC")])
    rc = bt.main(["--market", "US", "--cache-dir", str(cache), "--universe-file", str(uf),
                  "--workers", "1", "--quick"])
    assert rc == 0
    panel = pd.read_pickle(cache / "panel.pkl")
    assert set(panel["symbol"]) == {"AAA", "BBB", "CCC"}
    assert panel["date"].nunique() >= 10
    assert (cache / "panel.pkl.meta.json").exists()


def test_fetch_only_stops_before_scoring(bt, tmp_path):
    cache = tmp_path / "US"
    write_frames(str(cache), {"SPY": make_frame(700, 9), "AAA": make_frame(700, 1)})
    with open(cache / "policy_rate.pkl", "wb") as f:
        pickle.dump([], f)
    uf = tmp_path / "u.json"
    write_json(uf, [{"symbol": "AAA", "exchange": "NYSE", "indices": ["SPX"]}])
    assert bt.main(["--market", "US", "--cache-dir", str(cache), "--universe-file", str(uf),
                    "--fetch-only"]) == 0
    assert not (cache / "panel.pkl").exists()


def test_a_us_run_without_rate_history_falls_back_to_the_us_rate(bt):
    import pandas as pd
    bt.configure("US")
    assert bt.flat_rate() == 4.0
    assert bt.rate_for_date([], pd.Timestamp("2020-01-01")) == 4.0
    bt.configure("EG")
    assert bt.flat_rate() == bt.RISK_FREE_RATE
