"""Performance benchmarks for the ZHISA pipeline.

Run with:  pytest benchmarks/ --benchmark-only
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.rendering.chart_renderer import render_chart
from zhisa.features.ohlcv import compute_ohlcv_features
from zhisa.features.time import compute_time_features
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.models.policy import build_default_policy


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    return generate_market(MarketConfig(n_bars=10_000, seed=0))


def test_bench_synthetic_generation(benchmark):
    cfg = MarketConfig(n_bars=5_000, seed=0)
    benchmark(generate_market, cfg)


def test_bench_feature_engineering(benchmark, market):
    benchmark(compute_ohlcv_features, market)


def test_bench_time_features(benchmark, market):
    benchmark(compute_time_features, market)


def test_bench_chart_rendering(benchmark, market):
    window = market.iloc[:128]
    benchmark(render_chart, window, 64)


def test_bench_env_step(benchmark, market):
    env = TradingEnv(market, cfg=EnvConfig(window=16, image_size=16))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    actions = [int(rng.integers(0, 9)) for _ in range(200)]

    def run():
        for a in actions:
            env.step(a)

    benchmark(run)


def test_bench_model_forward(benchmark):
    model = build_default_policy(
        in_numeric_features=20, in_context_features=10,
        window=16, image_size=16, n_actions=9,
    )
    model.eval()
    chart = torch.rand(8, 3, 16, 16)
    num = torch.rand(8, 16, 20)
    ctx = torch.rand(8, 10)

    with torch.no_grad():
        benchmark(model, chart=chart, numeric=num, context=ctx)
