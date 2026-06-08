"""Trace env internals to find the source of overflow."""
import pandas as pd
import numpy as np
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.env.actions import DiscreteAction

df = pd.read_parquet('data/synth/synth.parquet')
cfg = EnvConfig()
env = TradingEnv(df, cfg=cfg)
obs, _ = env.reset(seed=0)

# Monkey-patch _mark_to_market to log
orig_mtm = env._mark_to_market
def traced_mtm(price=None):
    p = price if price is not None else float(env.df["close"].iloc[env._t])
    ret = (p / max(env._avg_entry, 1e-12)) - 1.0
    out = orig_mtm(price)
    print(f"  MTM: price={p:.4f} avg_entry={env._avg_entry:.6e} ret={ret:.6e} "
          f"pos={env._position} lev={cfg.max_leverage} "
          f"=> equity+pos*lev*ret = {env._equity} + {env._position}*{cfg.max_leverage}*{ret:.6e} = {out:.6e}")
    return out
env._mark_to_market = traced_mtm

print(f"init_equity={cfg.initial_equity}, max_lev={cfg.max_leverage}, "
      f"risk.max_pos_per_inst={env._risk.limits.max_position_per_instrument}, "
      f"risk.max_gross={env._risk.limits.max_gross_exposure}")

np.random.seed(0)
for i in range(15):
    a = int(np.random.randint(0, 9))
    name = DiscreteAction(a).name
    print(f"\n--- step {i}: action={a} {name} ---")
    print(f"  pre: pos={env._position} avg_entry={env._avg_entry:.4f} equity={env._equity:.6f} t={env._t}")
    obs, r, term, trunc, info = env.step(a)
    print(f"  post: pos={env._position} avg_entry={env._avg_entry:.4f} equity={info['equity']:.6e} reward={r:.6e} t={env._t}")
    if not np.isfinite(info['equity']):
        print("  >>> NON-FINITE EQUITY <<<")
        break
    if term or trunc:
        print(f"  Episode ended: {info.get('exit_reason')}")
        break
