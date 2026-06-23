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
    parser.add_argument("--target-date", type=str, default="2026-03-01 00:00:00", help="Date to start simulation")
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
    
    target_dt = pd.to_datetime(args.target_date).tz_localize('UTC')
    
    diffs = abs(df_times - target_dt)
    target_idx = diffs.argmin()
    actual_target_dt = df_times[target_idx]
    
    print(f"\nTarget Date requested: {target_dt}")
    print(f"Closest available time in dataset: {actual_target_dt}")
    
    SIM_STEPS = 120 # 30 hours
    
    if target_idx + SIM_STEPS >= len(X):
        print("Error: Target date is too close to the end of the dataset to get 30 hours of future reality.")
        return
        
    current_state = X[target_idx].reshape(1, -1)
    
    valid_history_end = target_idx - SIM_STEPS - 1
    
    if valid_history_end < 100:
        print("Error: Not enough history before target date for KNN.")
        return
        
    X_history = X[:valid_history_end]
    
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(X_history)
    distances, indices = knn.kneighbors(current_state)
    
    print("\nFound Top 10 Historical Analogs (from strict past):")
    analog_trajectories = []
    
    for rank, hist_idx in enumerate(indices[0]):
        t = df_times[hist_idx]
        d = distances[0][rank]
        
        base_price = close_prices[hist_idx]
        future_prices = close_prices[hist_idx : hist_idx + SIM_STEPS + 1]
        trajectory = (future_prices - base_price) / base_price
        analog_trajectories.append(trajectory)
        print(f"  Analog {rank+1} | Time: {t} | Dist: {d:.4f}")
        
    analog_trajectories = np.array(analog_trajectories)
    mean_trajectory = analog_trajectories.mean(axis=0)
    
    base_price_real = close_prices[target_idx]
    future_prices_real = close_prices[target_idx : target_idx + SIM_STEPS + 1]
    trajectory_real = (future_prices_real - base_price_real) / base_price_real
    
    plt.figure(figsize=(12, 7))
    for i, traj in enumerate(analog_trajectories):
        plt.plot(traj * 100, color='gray', alpha=0.2, label='Historical Analogs' if i==0 else "")
        
    plt.plot(mean_trajectory * 100, color='blue', linewidth=3, linestyle='--', label='S1 Forecast (Mean of Analogs)')
    plt.plot(trajectory_real * 100, color='red', linewidth=3, label='REALITY (Actual Future)')
    
    plt.axhline(0, color='black', linestyle='-')
    plt.title(f"S1 Forecast vs Reality (30-Hour Simulation starting {actual_target_dt.strftime('%Y-%m-%d %H:%M')})")
    plt.xlabel("15-Minute Steps (120 steps = 30 hours)")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.out_dir, "backtest_30h.png")
    plt.savefig(plot_path)
    print(f"\nSaved backtest plot to {plot_path}")

if __name__ == "__main__":
    main()
