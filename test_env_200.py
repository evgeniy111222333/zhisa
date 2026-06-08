"""200-step random rollout to confirm env is stable after fix."""
import pandas as pd
import numpy as np
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.env.actions import DiscreteAction

df = pd.read_parquet('D:/zhisa/data/synth/synth.parquet')
cfg = EnvConfig(episode_length=200)
env = TradingEnv(df, cfg=cfg)
obs, _ = env.reset(seed=0)

np.random.seed(42)
bad_steps = []
max_eq = 0.0
min_eq = 1e18
rewards = []
n_steps = 0
for i in range(200):
    a = int(np.random.randint(0, 9))
    obs, r, term, trunc, info = env.step(a)
    n_steps = i + 1
    if not np.isfinite(info['equity']) or not np.isfinite(r):
        bad_steps.append((i, a, DiscreteAction(a).name, info['equity'], r))
    max_eq = max(max_eq, info['equity'])
    min_eq = min(min_eq, info['equity'])
    rewards.append(r)
    if term or trunc:
        reason = info.get('exit_reason') or 'unknown'
        print(f'Step {i}: ended reason={reason}')
        break

print(f'After {n_steps} steps:')
print(f'  Max equity: {max_eq:.6f}')
print(f'  Min equity: {min_eq:.6f}')
print(f'  Final equity: {info["equity"]:.6f}')
print(f'  Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]')
print(f'  Mean reward: {np.mean(rewards):.4f}')
print(f'  Bad (non-finite) steps: {len(bad_steps)}')
if bad_steps:
    print(f'  First bad: {bad_steps[0]}')
