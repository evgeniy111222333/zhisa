import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zhisa.models.policy import build_default_policy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_decisions
from visualize_decisions import _checkpoint_policy_config
from zhisa.env.trading_env import TradingEnv, EnvConfig
from zhisa.scripts._real_data import load_market_dataframe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/s1/model.pt")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    args = parser.parse_args()

    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = _checkpoint_policy_config(ckpt)
    
    in_num = int(cfg.get("in_numeric_features", 32))
    in_ctx = int(cfg.get("in_context_features", 10))
    window = int(cfg.get("window", 32))
    
    model = build_default_policy(
        in_numeric_features=in_num,
        in_context_features=in_ctx,
        window=window,
        image_size=int(cfg.get("image_size", EnvConfig.image_size)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loading {args.symbol} data...")
    class DummyArgs:
        symbol = args.symbol
        bars = 500
        source = "binance_futures"
    
    df = load_market_dataframe(DummyArgs(), seed=42, default_bars=500)
    
    if in_num <= 35:
        keep_cols = [c for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume", "timestamp"]]
        df = df[keep_cols]

    env_cfg = EnvConfig()
    env_cfg.window = window
    env = TradingEnv(df, cfg=env_cfg)
    
    obs, _ = env.reset()
    
    chart = torch.from_numpy(obs["chart"]).unsqueeze(0).float()
    num = torch.from_numpy(obs["numeric"]).unsqueeze(0).float()
    
    with torch.no_grad():
        v_emb = model.vision(chart)
        n_emb, _ = model.numeric(num)
        
    print(f"Vision embedding shape: {v_emb.shape}")
    print(f"Numeric embedding shape: {n_emb.shape}")
    
    if v_emb.dim() == 2:
        v_emb = v_emb.unsqueeze(1)
        
    v_norm = torch.nn.functional.normalize(v_emb[0], p=2, dim=-1)
    n_norm = torch.nn.functional.normalize(n_emb[0], p=2, dim=-1)
    
    sim_matrix = torch.matmul(v_norm, n_norm.t()).numpy()
    
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Min sim: {sim_matrix.min():.4f}, Max sim: {sim_matrix.max():.4f}, Mean sim: {sim_matrix.mean():.4f}")
    
    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debug_vision.png"
    
    plt.figure(figsize=(12, 6))
    plt.plot(v_emb[0].numpy(), label='Vision Embedding (Image)', alpha=0.8, linewidth=2)
    plt.plot(n_emb[0].numpy(), label='Numeric Embedding (OHLCV)', alpha=0.8, linewidth=2)
    plt.title('Vision vs Numeric Embeddings (What the model "sees" internally)')
    plt.xlabel('Embedding Dimension (0 to 127)')
    plt.ylabel('Activation Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
