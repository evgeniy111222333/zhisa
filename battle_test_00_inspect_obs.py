"""Get actual obs shapes from env so we can build a real test."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"D:\zhisa\src")
from zhisa.env.trading_env import TradingEnv, EnvConfig

df = pd.read_parquet(r"D:\zhisa\data\synth\synth.parquet")
env = TradingEnv(df, cfg=EnvConfig(episode_length=20))
obs, _ = env.reset(seed=0)
for k, v in obs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")
