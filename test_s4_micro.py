"""Tiny PPO smoke - just 1 episode, 10 steps, 1 iter, no epochs."""
import time
import numpy as np
import pandas as pd
import torch
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig
from zhisa.training.s4_rl import PPOConfig, PPOTrainer
from zhisa.training.optim import OptimConfig
from zhisa.models.policy import build_default_policy
from zhisa.data.dataset import MarketDataset, SampleSpec

print("Loading...", flush=True)
df = pd.read_parquet('D:/zhisa/data/synth/synth.parquet')
print(f"DF: {df.shape}", flush=True)

spec = SampleSpec(chart_window=32, image_size=32)
probe_df = generate_market(MarketConfig(n_bars=300, seed=0))
probe_ds = MarketDataset(probe_df, spec=spec)
n_feat = probe_ds._features.shape[1]
n_ctx = probe_ds._time_features.shape[1]

model = build_default_policy(
    in_numeric_features=n_feat, in_context_features=n_ctx,
    window=spec.chart_window, image_size=spec.image_size,
    n_actions=9, n_regime_classes=spec.n_regime_states,
)

env_cfg = EnvConfig(episode_length=20, max_leverage=3.0)
ppo_cfg = PPOConfig(
    n_episodes=1, max_steps_per_episode=10,
    n_epochs=1, minibatch_size=4,
    device="cpu", optim=OptimConfig(lr=3e-4, weight_decay=0.01),
    env_cfg=env_cfg, seed=0, log_every=1,
)

trainer = PPOTrainer(model, ppo_cfg)
print("Trainer built, starting fit...", flush=True)

# Warmup: a few env steps to amortise first-call cost.
import time as _t
from zhisa.env.trading_env import TradingEnv
env_test = TradingEnv(df, cfg=env_cfg)
obs, _ = env_test.reset(seed=0)
for i in range(5):
    t0 = _t.perf_counter()
    obs, r, term, trunc, info = env_test.step(0)
    print(f"  warmup step {i}: {1000*(_t.perf_counter()-t0):.0f}ms", flush=True)
    if term or trunc:
        break

print("Now running PPO fit...", flush=True)
t0 = time.perf_counter()
result = trainer.fit(df)
t1 = time.perf_counter()
print(f"\nFit done in {t1-t0:.1f}s", flush=True)

for h in result["history"]:
    print(f"  it={h['iteration']:>3}  ep={h['n_episodes']:>2}  steps={h['rollout_steps']:>4}  "
          f"mean_return={h['mean_return']:>14.4f}  policy={h['policy_loss']:>8.4f}  "
          f"value={h['value_loss']:>8.4f}  entropy={h['entropy']:>8.4f}  total={h['total_loss']:>8.4f}",
          flush=True)

returns = [h["mean_return"] for h in result["history"]]
all_finite = all(np.isfinite(r) for r in returns)
print(f"\nAll returns finite? {all_finite}")
print(f"Max abs return: {max(abs(r) for r in returns):.4f}")
