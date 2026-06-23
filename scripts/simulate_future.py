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
    print(f"Using device: {device}")
    
    print("Loading recent 10000 bars from history for analog matching...")
    df = pd.read_parquet(args.dataset).sort_index().iloc[-10000:]
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
        for batch in tqdm(loader, desc="Extracting 10k Embeddings"):
            c = batch.chart.to(device)
            n = batch.numeric.to(device)
            ctx = batch.context.to(device)
            out = model(chart=c, numeric=n, context=ctx)
            embs.append(out["embedding"].cpu().numpy())
            
    X = np.concatenate(embs)
    
    df_times = df.index[spec.chart_window:].tolist()
    close_prices = df["close"].values[spec.chart_window:]
    
    print("\n" + "="*50)
    print("PART 1: DEEP DIVE INTO NEURON #113 (FEAR NEURON)")
    print("="*50)
    neuron_vals = X[:, 113]
    top_5_idx = np.argsort(neuron_vals)[-5:][::-1]
    
    n113_out = []
    for rank, idx in enumerate(top_5_idx):
        t = df_times[idx]
        val = neuron_vals[idx]
        future_idx = min(idx + 10, len(close_prices)-1)
        r = (close_prices[future_idx] - close_prices[idx]) / close_prices[idx]
        msg = f"Rank {rank+1} | Time: {t} | Neuron Value: {val:.3f} | Subsequent 2.5h Return: {r:.2%}"
        print(msg)
        n113_out.append(msg)
        
    with open(os.path.join(args.out_dir, "neuron113.txt"), "w") as f:
        f.write("\n".join(n113_out))
        
    print("\n" + "="*50)
    print("PART 2: 30-HOUR SIMULATION (ANALOG FORECASTING)")
    print("="*50)
    
    SIM_STEPS = 120 # 30 hours
    valid_analog_idx = len(X) - SIM_STEPS - 1
    
    X_history = X[:valid_analog_idx]
    current_state = X[-1].reshape(1, -1)
    
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(X_history)
    distances, indices = knn.kneighbors(current_state)
    
    print(f"Current State Time: {df_times[-1]}")
    print("Found Top 10 Historical Analogs:")
    
    analog_trajectories = []
    for rank, hist_idx in enumerate(indices[0]):
        t = df_times[hist_idx]
        d = distances[0][rank]
        
        base_price = close_prices[hist_idx]
        future_prices = close_prices[hist_idx : hist_idx + SIM_STEPS + 1]
        trajectory = (future_prices - base_price) / base_price
        analog_trajectories.append(trajectory)
        
        final_ret = trajectory[-1]
        print(f"  Analog {rank+1} | Time: {t} | Dist: {d:.4f} | 30h Actual Return: {final_ret:.2%}")
        
    analog_trajectories = np.array(analog_trajectories)
    mean_trajectory = analog_trajectories.mean(axis=0)
    
    plt.figure(figsize=(10, 6))
    for i, traj in enumerate(analog_trajectories):
        plt.plot(traj * 100, color='gray', alpha=0.3, label='Historical Analogs' if i==0 else "")
    plt.plot(mean_trajectory * 100, color='blue', linewidth=3, label='Mean Simulated Forecast')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("30-Hour Market Simulation (Analog Forecasting via S1 Latent Space)")
    plt.xlabel("15-Minute Steps (120 steps = 30 hours)")
    plt.ylabel("Cumulative Return Forecast (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.out_dir, "simulation_30h.png")
    plt.savefig(plot_path)
    print(f"\nSaved simulation plot to {plot_path}")
    
    final_mean_ret = mean_trajectory[-1]
    msg = f"SIMULATION RESULT: Over the next 30 hours, the model projects an average cumulative return of {final_mean_ret:.2%}."
    print(msg)
    with open(os.path.join(args.out_dir, "simulation_result.txt"), "w") as f:
        f.write(msg)

if __name__ == "__main__":
    main()
