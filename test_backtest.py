"""Backtest smoke test: random policy, then S2 checkpoint policy."""
import sys
import numpy as np
import pandas as pd

from zhisa.backtest.engine import run_backtest, random_policy, buy_and_hold_benchmark
from zhisa.backtest.metrics import compute_metrics
from zhisa.env.trading_env import EnvConfig

df = pd.read_parquet('D:/zhisa/data/synth/synth.parquet')
print(f"DF: {df.shape}, range: {df.index[0]} -> {df.index[-1]}", flush=True)

# Random policy, capped at 200 steps (env is slow due to chart render ~350ms/step)
res = run_backtest(df, random_policy(seed=0), cfg=EnvConfig(episode_length=200))
m = res.metrics
print(f"\n=== Random policy ({len(res.equity)} steps) ===", flush=True)
print(f"  total_return         : {m.total_return*100:.4f} %", flush=True)
print(f"  annualised_vol       : {m.annualised_vol*100:.4f} %", flush=True)
print(f"  sharpe               : {m.sharpe:.4f}", flush=True)
print(f"  sortino              : {m.sortino:.4f}", flush=True)
print(f"  max_drawdown         : {m.max_drawdown*100:.4f} %", flush=True)
print(f"  n_trades             : {m.n_trades}", flush=True)
print(f"  profit_factor        : {m.profit_factor:.4f}", flush=True)
print(f"  equity range         : [{res.equity.min():.6f}, {res.equity.max():.6f}]", flush=True)
print(f"  any NaN/Inf?         : {not np.all(np.isfinite(res.equity))}", flush=True)

# Buy & hold benchmark
bnh = buy_and_hold_benchmark(df)
m_bnh = compute_metrics(bnh)
print(f"\n=== Buy & hold ===", flush=True)
print(f"  total_return         : {m_bnh.total_return*100:.4f} %", flush=True)
print(f"  annualised_vol       : {m_bnh.annualised_vol*100:.4f} %", flush=True)
print(f"  max_drawdown         : {m_bnh.max_drawdown*100:.4f} %", flush=True)

