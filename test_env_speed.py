"""Time each env step to find slow path."""
import time
import numpy as np
import pandas as pd
from zhisa.env.trading_env import TradingEnv, EnvConfig

df = pd.read_parquet('D:/zhisa/data/synth/synth.parquet')
cfg = EnvConfig(episode_length=20)
env = TradingEnv(df, cfg=cfg)
obs, _ = env.reset(seed=0)

np.random.seed(42)
times = []
for i in range(15):
    t0 = time.perf_counter()
    a = int(np.random.randint(0, 9))
    obs, r, term, trunc, info = env.step(a)
    times.append(time.perf_counter() - t0)
    print(f"  step {i}: {1000*(time.perf_counter()-t0):.0f}ms", flush=True)
    if term or trunc:
        break

print(f"\nCompleted {len(times)} steps in {sum(times):.2f}s")
print(f"Per-step: mean={1000*np.mean(times):.1f}ms")

