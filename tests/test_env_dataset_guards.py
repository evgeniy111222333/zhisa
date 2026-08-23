"""Regressions: step-after-termination guard + dataset lookback configurable."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, r"D:\zhisa\src")
sys.path.insert(0, r"D:\zhisa")

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.features.normalization import NormalizationSpec


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.002))
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": rng.uniform(1, 2, n)}, index=idx)


def test_step_after_termination_does_not_crash():
    # With an OPEN position at the terminal bar, the original env crashed on
    # iloc[len(df)] inside mark-to-market/liquidation; both boundary and
    # post-terminal calls must be safe now.
    for action in (1,):  # holds a position
        env = TradingEnv(_df(n=240), cfg=EnvConfig(window=16, image_size=32))
        env.reset()
        done = False
        while not done:
            obs, _, done, _, _ = env.step(1)
        assert done
        # boundary call must complete without raising (clamped terminal price)
        obs1, r1, d1, t1, info1 = env.step(1)
        assert (d1 or t1) and np.isfinite(r1)
        # OOB call -> safe no-op, not IndexError
        obs2, r2, d2, t2, info2 = env.step(1)
        assert d2 is True and info2.get("error") == "episode_already_terminated"
        assert r2 == 0.0


def test_lookback_wired_to_normalization_spec():
    spec = SampleSpec(chart_window=32, feature_window=32, image_size=32,
                      horizons=(4, 16, 64))
    df = _df(n=400)
    t = 300
    for lb in (128, 384):
        ds = MarketDataset(df, spec=spec, cache_charts=False, compute_targets=False,
                           normalization=NormalizationSpec(mode="robust_z", lookback=lb))
        # history slice for window ending at t+1: rows [max(0,t-lb) .. t]
        hist = ds._history_window(t, t + 1)
        expect = min(lb, t) + 1
        assert hist.shape[0] == expect, f"lookback={lb}: got {hist.shape[0]} != {expect}"


def test_default_lookback_still_256():
    spec = SampleSpec(chart_window=32, feature_window=32, image_size=32,
                      horizons=(4, 16, 64))
    ds = MarketDataset(_df(n=400), spec=spec, cache_charts=False, compute_targets=False)
    assert ds.normalization.lookback == 256
    t = 300
    assert ds._history_window(t, t + 1).shape[0] == 257  # rows 44..300
    t2 = 200
    assert ds._history_window(t2, t2 + 1).shape[0] == t2 + 1  # clamped at 0