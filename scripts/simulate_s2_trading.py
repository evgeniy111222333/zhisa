import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ.setdefault("ZHISA_FAST_RENDER", "1")

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from torch.utils.data import DataLoader

def _longest_contiguous(frame: pd.DataFrame, timeframe: str = "15min") -> pd.DataFrame:
    expected = pd.Timedelta(timeframe)
    ids = frame.index.to_series().diff().ne(expected).cumsum()
    segments = [part for _, part in frame.groupby(ids, sort=False)]
    return max(segments, key=len)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--target-date", type=str, default="2024-03-01 00:00:00")
    parser.add_argument("--fee", type=float, default=0.0005) # 0.05%
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = dict(checkpoint["model_config"])
    if isinstance(raw_config.get("vision_channels"), list):
        raw_config["vision_channels"] = tuple(raw_config["vision_channels"])
    model = PolicyNetwork(PolicyConfig(**raw_config))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    
    spec = SampleSpec(
        chart_window=int(raw_config["window"]),
        feature_window=int(raw_config["window"]),
        image_size=int(raw_config["image_size"]),
        n_regime_states=int(raw_config["n_regime_classes"]),
    )
    
    # 2. Load Data
    print(f"Loading dataset from {args.dataset} for {args.symbol}...")
    
    # Load from TSDB raw data to have full history
    df_path = os.path.join("data", "tsdb", args.symbol.replace('/', '_'), "15m", "data.parquet")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Cannot find full history at {df_path}")
    df = pd.read_parquet(df_path).sort_index()
        
    df = _longest_contiguous(df)
    
    target_dt = pd.to_datetime(args.target_date).tz_localize('UTC')
    times = df.index
    diffs = abs(times - target_dt)
    target_idx = diffs.argmin()
    actual_target_dt = times[target_idx]
    
    print(f"Requested Target: {target_dt}")
    print(f"Actual Start:     {actual_target_dt}")
    
    start_idx = target_idx - spec.chart_window
    end_idx = target_idx + args.steps + 64 + 1
    
    if start_idx < 0 or end_idx >= len(df):
        raise ValueError("Target date is too close to the edges of the dataset.")
        
    sim_df = df.iloc[start_idx:end_idx]
    ds = MarketDataset(
        sim_df,
        spec=spec,
        cache_charts=False,
        compute_targets=False,
    )
    
    loader = DataLoader(ds, batch_size=1, collate_fn=multimodal_collate, shuffle=False)
    
    print("Starting Step-by-Step Simulation...")
    
    initial_balance = 10000.0
    balance = initial_balance
    position = 0 # 0 = Cash, 1 = Long, -1 = Short
    
    history_balance = []
    history_actions = []
    history_prices = []
    history_times = []
    
    sim_prices = sim_df['close'].values[spec.chart_window - 1:]
    sim_times = sim_df.index[spec.chart_window - 1:]
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Trading")):
            if i >= args.steps:
                break
                
            c = batch.chart.to(device)
            n = batch.numeric.to(device)
            ctx = batch.context.to(device)
            
            out = model(chart=c, numeric=n, context=ctx)
            direction_logits = out["direction"][0]
            pred_class = torch.argmax(direction_logits).item()
            
            desired_position = pred_class - 1
            
            current_price = sim_prices[i]
            next_price = sim_prices[i+1]
            current_time = sim_times[i]
            
            action_taken = None
            if desired_position != position:
                fee_multiplier = abs(desired_position - position)
                fee_cost = balance * args.fee * fee_multiplier
                balance -= fee_cost
                position = desired_position
                
                if position == 1:
                    action_taken = "BUY"
                elif position == -1:
                    action_taken = "SELL"
                else:
                    action_taken = "CLOSE"
                    
            if action_taken:
                history_actions.append({"time": current_time, "action": action_taken, "price": current_price, "step": i})
            
            if position != 0:
                return_pct = (next_price - current_price) / current_price
                pnl = balance * return_pct * position
                balance += pnl
                
            history_balance.append(balance)
            history_prices.append(current_price)
            history_times.append(current_time)
            
    print(f"\nSimulation Complete!")
    print(f"Final Balance: ${balance:.2f} (Return: {((balance/initial_balance)-1)*100:.2f}%)")
    print(f"Total Trades:  {len(history_actions)}")
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    ax1.plot(history_times, history_prices, color='white', alpha=0.7, label='Price')
    
    for act in history_actions:
        t = act["time"]
        p = act["price"]
        if act["action"] == "BUY":
            ax1.scatter(t, p, color='lime', marker='^', s=150, zorder=5, label='BUY' if 'BUY' not in ax1.get_legend_handles_labels()[1] else "")
        elif act["action"] == "SELL":
            ax1.scatter(t, p, color='red', marker='v', s=150, zorder=5, label='SELL' if 'SELL' not in ax1.get_legend_handles_labels()[1] else "")
        elif act["action"] == "CLOSE":
            ax1.scatter(t, p, color='yellow', marker='x', s=100, zorder=5, label='CLOSE' if 'CLOSE' not in ax1.get_legend_handles_labels()[1] else "")
            
    ax1.set_title(f"S2 Epoch 8 Trading Simulator | {args.symbol} | Start: {actual_target_dt.strftime('%Y-%m-%d')}")
    ax1.set_ylabel("Price (USDT)")
    ax1.grid(True, alpha=0.2)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    ax2.plot(history_times, history_balance, color='cyan', linewidth=2, label='Equity ($10k Start)')
    ax2.axhline(initial_balance, color='gray', linestyle='--', alpha=0.5)
    
    ax2.fill_between(history_times, history_balance, initial_balance, where=(np.array(history_balance) >= initial_balance), color='green', alpha=0.1)
    ax2.fill_between(history_times, history_balance, initial_balance, where=(np.array(history_balance) < initial_balance), color='red', alpha=0.1)
    
    ax2.set_ylabel("Equity (USD)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "s2_trading_sim.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved visualization to {plot_path}")

if __name__ == "__main__":
    main()
