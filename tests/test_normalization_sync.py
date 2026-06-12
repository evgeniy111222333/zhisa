import numpy as np
import pandas as pd
import pytest
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.env.trading_env import EnvConfig, TradingEnv

def test_normalization_sync_dataset_vs_env():
    # 1) Generate synthetic market data
    df = generate_market(MarketConfig(n_bars=200, seed=42))

    # 2) Create MarketDataset (t=0 matches first env observation)
    spec = SampleSpec(chart_window=32, feature_window=32)
    ds = MarketDataset(df, spec=spec)

    # 3) Create TradingEnv with the same window
    env_cfg = EnvConfig(window=32)
    env = TradingEnv(df, cfg=env_cfg)
    env_obs, _ = env.reset(seed=42)

    # 4) Verify step-by-step equivalence for the first 20 windows
    for t in range(20):
        # Dataset sample at step t
        ds_sample = ds[t]
        ds_numeric = ds_sample["numeric"].numpy()

        # Environment numeric features at current step
        env_numeric = env_obs["numeric"]

        # Check shapes are identical
        assert ds_numeric.shape == env_numeric.shape, f"Shape mismatch at step {t}!"

        # Check values are identical within float32 precision
        np.testing.assert_allclose(
            ds_numeric,
            env_numeric,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Discrepancy in normalized features at step {t}!"
        )

        # Step environment to advance to t + 1
        env_obs, reward, term, trunc, info = env.step(0)
        if term or trunc:
            break

    print("Verification successful: Normalization is perfectly synchronized between MarketDataset and TradingEnv!")
