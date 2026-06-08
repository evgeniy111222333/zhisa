"""Minimal backtest debug - check if env.step hangs."""
import sys
import numpy as np
import pandas as pd
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.env.actions import DiscreteAction

df = pd.read_parquet('D:/zhisa/data/synth/synth.parquet')
print(f"DF: {df.shape}", flush=True)

cfg = EnvConfig(episode_length=50)
env = TradingEnv(df, cfg=cfg)
obs, _ = env.reset(seed=0)
print(f"After reset: t={env._t}, equity={env._equity:.6f}", flush=True)

np.random.seed(42)
for i in range(80):
    a = int(np.random.randint(0, 9))
    obs, r, term, trunc, info = env.step(a)
    if i < 5 or i % 10 == 0 or term or trunc:
        print(f"  step {i}: action={a} pos={info['position']:.4f} eq={info['equity']:.6f} r={r:.6f} term={term} trunc={trunc} reason={info.get('exit_reason')}", flush=True)
    if term or trunc:
        print(f"  Episode ended at step {i}", flush=True)
        break
print("DONE", flush=True)
