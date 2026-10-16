"""Synthetic OHLCV frames and cache directories for the offline scripts' tests."""
import json
import os
import pickle

import numpy as np
import pandas as pd


def make_frame(n: int, seed: int, start: str = "2020-01-01", price: float = 50.0,
               volume: float = 2_000_000.0, vol_spread: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = rng.normal(0.0003, 0.012 + 0.003 * (seed % 5), n)
    close = price * np.exp(np.cumsum(rets))
    vols = volume * np.exp(rng.normal(0.0, vol_spread, n))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": vols}, index=idx)


def write_frames(cache_dir: str, frames: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for name, df in frames.items():
        with open(os.path.join(cache_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(df, f)


def write_json(path: str, value) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f)
