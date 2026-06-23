import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import build_default_policy
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["ZHISA_FAST_RENDER"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading recent 15000 bars from history...")
    df = pd.read_parquet(args.dataset).sort_index().iloc[-15000:]
    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128)
    ds = MarketDataset(df, spec=spec, compute_targets=True)
    loader = DataLoader(ds, batch_size=64, collate_fn=multimodal_collate, shuffle=False)
    
    model = build_default_policy(
        in_numeric_features=ds._features_df.shape[1],
        in_context_features=ds._time_features_df.shape[1],
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("reconstructor.")}, strict=False)
    model.eval()
    model.to(device)
    
    embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Embeddings"):
            c = batch.chart.to(device)
            n = batch.numeric.to(device)
            ctx = batch.context.to(device)
            out = model(chart=c, numeric=n, context=ctx)
            embs.append(out["embedding"].cpu().numpy())
            
    X = np.concatenate(embs)
    df_times = df.index[spec.chart_window:]
    close_prices = df["close"].values[spec.chart_window:]
    
    target_dates = [
        "2026-03-01 00:00:00",
        "2026-04-15 12:00:00",
        "2026-05-10 08:00:00"
    ]
    
    SIM_STEPS = 120 # 30 hours
    
    plt.figure(figsize=(15, 10))
    
    for i, t_date_str in enumerate(target_dates):
        target_dt = pd.to_datetime(t_date_str).tz_localize('UTC')
        diffs = abs(df_times - target_dt)
        target_idx = diffs.argmin()
        actual_target_dt = df_times[target_idx]
        
        print(f"\nProcessing Date: {actual_target_dt}")
        
        if target_idx + SIM_STEPS >= len(X):
            print("  Error: Too close to end.")
            continue
            
        current_state = X[target_idx].reshape(1, -1)
        valid_history_end = target_idx - SIM_STEPS - 1
        
        if valid_history_end < 100:
            print("  Error: Not enough history.")
            continue
            
        X_history = X[:valid_history_end]
        
        knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn.fit(X_history)
        distances, indices = knn.kneighbors(current_state)
        
        best_analog_idx = indices[0][0]
        
        # Analog Trajectory (Top 1)
        base_price_analog = close_prices[best_analog_idx]
        future_prices_analog = close_prices[best_analog_idx : best_analog_idx + SIM_STEPS + 1]
        trajectory_analog = (future_prices_analog - base_price_analog) / base_price_analog
        
        # Reality Trajectory
        base_price_real = close_prices[target_idx]
        future_prices_real = close_prices[target_idx : target_idx + SIM_STEPS + 1]
        trajectory_real = (future_prices_real - base_price_real) / base_price_real
        
        # Subplot
        plt.subplot(3, 1, i + 1)
        plt.plot(trajectory_analog * 100, color='blue', linewidth=2, linestyle='--', label='S1 Forecast (Top-1 Analog)')
        plt.plot(trajectory_real * 100, color='red', linewidth=2, label='REALITY (Actual Future)')
        plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        plt.title(f"Target Date: {actual_target_dt.strftime('%Y-%m-%d %H:%M')}")
        plt.ylabel("Return (%)")
        if i == 2:
            plt.xlabel("15-Minute Steps (120 steps = 30 hours)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "backtest_multi.png")
    plt.savefig(plot_path)
    print(f"\nSaved multi-date backtest plot to {plot_path}")

if __name__ == "__main__":
    main()
