"""Regression: backtest timestamps must align with the env window start."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\zhisa\src")
sys.path.insert(0, r"D:\zhisa")

from zhisa.backtest.engine import run_backtest
from zhisa.env.trading_env import EnvConfig


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.002))
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": rng.uniform(1, 2, n)}, index=idx)


def _random_policy(_obs):
    return int(sys.maxsize) % 9


def test_timestamps_start_at_window_not_zero():
    df = _df()
    cfg = EnvConfig(window=128, image_size=64)
    res = run_backtest(df, _random_policy, cfg=cfg)
    ts = pd.DatetimeIndex(res.timestamps)
    assert ts[0] == df.index[128], f"first step maps to bar {ts[0]} != bar 128"
    # last timestamp never exceeds the frame
    assert ts[-1] <= df.index[-1]
    # length alignment: equity incl. initial + steps => timestamps cover
    # the same seq as equity (bounds-guarded)
    assert len(ts) <= len(res.equity)


def test_timestamps_align_with_bars_in_window():
    df = _df()
    res = run_backtest(df, _random_policy, cfg=EnvConfig(window=32, image_size=64))
    ts = pd.DatetimeIndex(res.timestamps)
    assert ts[0] == df.index[32]
    # an interior point: k-th step ↔ bar 32+k
    k = min(50, len(ts) - 1)
    assert ts[k] == df.index[32 + k]