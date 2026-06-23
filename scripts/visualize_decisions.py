import argparse
import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from zhisa.backtest.engine import run_backtest
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.features.ohlcv import compute_ohlcv_features

def _checkpoint_policy_config(ckpt: dict) -> dict:
    for key in ("model_config", "policy_config", "config"):
        cfg = ckpt.get(key)
        if isinstance(cfg, dict) and "window" in cfg and "in_numeric_features" in cfg:
            return cfg
    return {}

def _model_policy(model, device: str = "cpu"):
    model.eval()
    model.to(device)

    def _p(obs):
        with torch.no_grad():
            chart = torch.from_numpy(obs["chart"]).unsqueeze(0).to(device)
            num = torch.from_numpy(obs["numeric"]).unsqueeze(0).to(device)
            ctx = torch.from_numpy(obs["context"]).unsqueeze(0).to(device)
            out = model(chart=chart, numeric=num, context=ctx)
            return int(out["policy_logits"].argmax(dim=-1).item())

    return _p

def main():
    parser = argparse.ArgumentParser(description="Visualize model decisions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="artifacts/eval")
    add_market_data_args(parser)
    args = parser.parse_args()

    # Load Model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = _checkpoint_policy_config(ckpt)
    model = build_default_policy(
        in_numeric_features=60,
        in_context_features=10,
        window=128,
        image_size=128,
        n_actions=9,
        n_regime_classes=4,
    )
    # Exclude reconstructor keys from policy network
    policy_state = {k: v for k, v in ckpt["model"].items() if not k.startswith("reconstructor.")}
    model.load_state_dict(policy_state, strict=False)
    policy = _model_policy(model)

    env_cfg = EnvConfig()
    env_cfg.window = 128
    env_cfg.image_size = 128

    # Load Data
    df = load_market_dataframe(args, seed=42, default_bars=args.bars)
    if len(df) == 0:
        print("Dataframe is empty!")
        return
    print(f"Loaded df shape: {df.shape}")
    print(f"Loaded df columns: {list(df.columns)}")
        
    # We explicitly compute features here to extract the funding z-score for plotting
    features = compute_ohlcv_features(df)
    zscore = features.get("ctx_funding_zscore_7d", pd.Series(0.0, index=df.index))

    df_backtest = df.copy()

    # Run Backtest
    print(f"Running backtest on {len(df_backtest)} bars...")
    result = run_backtest(df_backtest, policy, cfg=env_cfg)

    # Output paths
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "trade_log.csv"
    img_path = out_dir / "decisions_chart.png"

    # Align dataframe indices with result lengths (which are shorter due to window size warmup)
    result_len = len(result.positions)
    aligned_df_index = df.index[-result_len:]
    aligned_zscore = zscore.iloc[-result_len:]

    # Save Text Log (CSV)
    print("Saving text log...")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Price", "Action", "TargetPos", "FundingZScore", "Equity"])
        
        last_pos = 0.0
        for i in range(result_len):
            current_pos = result.positions[i]
            if current_pos != last_pos:
                writer.writerow([
                    str(aligned_df_index[i]),
                    f"{result.prices[i]:.2f}",
                    result.actions[i],
                    current_pos,
                    f"{aligned_zscore.iloc[i]:.4f}",
                    f"{result.equity[i]:.4f}"
                ])
                last_pos = current_pos

    # Plot
    print("Generating chart...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    x = np.arange(result_len)

    
    # 1. Price + Position Background
    ax1 = axes[0]
    ax1.plot(x, result.prices, color='black', linewidth=1.5, label='Price')
    ax1.set_title("Price & Position (Green=Long, Red=Short)", fontsize=14)
    ax1.set_ylabel("Price")
    
    # Fill background based on positions
    # positions array contains values like -1, 0, 1
    # We can use fill_between to color the background
    ax1.fill_between(x, ax1.get_ylim()[0], ax1.get_ylim()[1], where=(result.positions > 0.01), color='green', alpha=0.15)
    ax1.fill_between(x, ax1.get_ylim()[0], ax1.get_ylim()[1], where=(result.positions < -0.01), color='red', alpha=0.15)

    # 2. Funding Z-Score
    ax2 = axes[1]
    ax2.plot(x, aligned_zscore.values, color='purple', linewidth=1.5, label='Funding Z-Score')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axhline(3, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(-3, color='red', linestyle=':', alpha=0.5)
    ax2.set_title("Context: Funding Rate Z-Score", fontsize=12)
    ax2.set_ylabel("Z-Score")
    
    # 3. Equity
    ax3 = axes[2]
    ax3.plot(x, result.equity, color='blue', linewidth=1.5, label='Equity')
    ax3.set_title("Equity Curve", fontsize=12)
    ax3.set_ylabel("Account Value")
    
    # formatting x-ticks
    step = max(1, len(x) // 10)
    ticks = x[::step]
    labels = [aligned_df_index[i].strftime("%Y-%m-%d %H:%M") for i in ticks]
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels, rotation=45)
    
    plt.tight_layout()
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {img_path}")

if __name__ == "__main__":
    main()
