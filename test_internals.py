import argparse
import torch
import numpy as np
import os
from pathlib import Path

from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.scripts._real_data import load_market_dataframe
from zhisa.models.policy import build_default_policy
from zhisa.scripts.backtest import TorchModelPolicy, _checkpoint_policy_config

def main():
    print("=== ZHiSA Internal Backtest Diagnostics ===")
    
    class DummyArgs:
        data_source = "tsdb"
        tsdb_root = "data/tsdb"
        symbol = "BTC/USDT"
        timeframe = "5m"
        latest_bars = 500
        bars = 500
        seed = 0
    
    args = DummyArgs()
    df = load_market_dataframe(args, seed=0, default_bars=500)
    print(f"[OK] Loaded DataFrame: shape={df.shape}, columns={list(df.columns)}")
    
    ckpt_path = "artifacts/s4/model_btc_rl.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _checkpoint_policy_config(ckpt)
    
    model = build_default_policy(
        in_numeric_features=int(cfg.get("in_numeric_features", 32)),
        in_context_features=int(cfg.get("in_context_features", 10)),
        window=int(cfg.get("window", 32)),
        image_size=int(cfg.get("image_size", EnvConfig.image_size)),
        n_actions=int(cfg.get("n_actions", 9)),
        n_regime_classes=int(cfg.get("n_regime_classes", 4)),
    )
    model.load_state_dict(ckpt["model"])
    policy = TorchModelPolicy(model, device="cpu")
    print(f"[OK] Policy loaded successfully. Device: {policy.device}")
    
    os.environ["ZHISA_FAST_RENDER"] = "1"
    
    env_cfg = EnvConfig(seed=0, window=int(cfg.get("window", 32)), image_size=int(cfg.get("image_size", 32)))
    env = TradingEnv(df, cfg=env_cfg)
    
    obs, info = env.reset()
    print("\n--- STEP 0 (Reset) Internals ---")
    print(f"Obs Keys: {list(obs.keys())}")
    print(f"Chart: shape={obs['chart'].shape}, dtype={obs['chart'].dtype}, min={obs['chart'].min():.3f}, max={obs['chart'].max():.3f}")
    print(f"Numeric: shape={obs['numeric'].shape}, dtype={obs['numeric'].dtype}, vals={obs['numeric'][:5].round(3)}...")
    print(f"Context: shape={obs['context'].shape}, dtype={obs['context'].dtype}, vals={obs['context'][:5].round(3)}...")
    
    logits = policy.logits(obs)
    action = policy(obs)
    print(f"\n--- POLICY FORWARD PASS Internals ---")
    print(f"Logits: {logits.numpy().round(3)}")
    print(f"Selected Action (argmax): {action}")
    
    obs, r, term, trunc, info = env.step(action)
    print(f"\n--- STEP 1 (Env Response) Internals ---")
    print(f"Reward: {r:.6f}")
    print(f"Info Equity: {info['equity']:.2f}")
    print(f"Info Position: {info['position']}")
    print(f"Info Price: {info['price']:.2f}")
    print("[OK] Internals Diagnostic Complete.")

if __name__ == '__main__':
    main()
