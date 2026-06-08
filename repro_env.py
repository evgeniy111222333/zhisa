"""Quick reproduction of env step issues."""
import pandas as pd
import numpy as np
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.env.actions import DiscreteAction

df = pd.read_parquet('data/synth/synth.parquet')
print(f'DF shape: {df.shape}, columns: {list(df.columns)}')
print(f'Index dtype: {df.index.dtype}, first: {df.index[0]}, last: {df.index[-1]}')
print(f'NaN in close: {df["close"].isna().sum()}')
print(f'Inf in close: {np.isinf(df["close"]).sum()}')
print(f'close min/max: {df["close"].min()}, {df["close"].max()}')

# Use default config (same as smoke test)
cfg = EnvConfig()
env = TradingEnv(df, cfg=cfg)
obs, _ = env.reset(seed=0)
print(f'\nEnvConfig: initial_equity={cfg.initial_equity}, fee_bps={cfg.fee_bps}, '
      f'slippage_bps_per_unit={cfg.slippage_bps_per_unit}, '
      f'market_depth_units={cfg.market_depth_units}, max_leverage={cfg.max_leverage}')
print(f'Obs shapes: chart={obs["chart"].shape}, numeric={obs["numeric"].shape}, context={obs["context"].shape}')

# Take 30 random actions, watch for NaN/Inf
print(f'\n{"step":>4} {"action":>6} {"name":>14} {"position":>10} {"equity":>14} {"reward":>14} {"price":>10}')
np.random.seed(0)
for i in range(30):
    a = int(np.random.randint(0, 9))
    try:
        obs, r, term, trunc, info = env.step(a)
        name = DiscreteAction(a).name
        print(f'{i:>4} {a:>6} {name:>14} {info["position"]:>10.4f} {info["equity"]:>14.6f} {r:>14.6f} {info["price"]:>10.4f}'
              f'   term={term} trunc={trunc}')
        if term or trunc:
            print(f'  Episode ended: exit_reason={info.get("exit_reason")}')
            break
        if not np.isfinite(info["equity"]) or not np.isfinite(r):
            print(f'  >>> NON-FINITE VALUE <<<')
            break
    except Exception as e:
        print(f'  >>> EXCEPTION: {e}')
        break
