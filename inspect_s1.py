import sys
import torch
import pprint
from pathlib import Path

sys.path.insert(0, r"D:\zhisa\src")
from zhisa.scripts._real_data import load_market_dataframe
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.models.policy import build_default_policy

print("="*50)
print("S1 INFERENCE DEBUG (5 Epochs Checkpoint)")
print("="*50)

# Load data
class FakeArgs:
    data_source = "tsdb"
    tsdb_root = "data/tsdb"
    symbol = "BTC/USDT"
    timeframe = "5m"
    with_futures_context = True
    futures_context_root = "data/futures_context/binance_usdm"

args = FakeArgs()
print("1. Loading real market data (1000 bars)...")
df = load_market_dataframe(args, default_bars=1000)

spec = SampleSpec(chart_window=32, feature_window=32, image_size=32)
ds = MarketDataset(df, spec=spec)
n_feat = ds._features.shape[1]
n_ctx = ds._time_features.shape[1]

# Load S1 Checkpoint
checkpoint_path = "artifacts/s1/model_btc.pt"
print(f"2. Loading checkpoint: {checkpoint_path}")
sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Build Model
model = build_default_policy(
    in_numeric_features=n_feat,
    in_context_features=n_ctx,
    window=spec.chart_window,
    image_size=spec.image_size,
    n_actions=9,
    n_regime_classes=spec.n_regime_states,
)

# Load weights
from zhisa.training.s1_ssl import _filter_matching_state_dict
filtered = _filter_matching_state_dict(sd["model"], model)
model.load_state_dict(filtered, strict=False)
model.eval()

# Inference on first 3 samples
print("\n3. Running Inference on 3 Real Market Snapshots:")
for i in [0, 100, 200]:
    sample = ds[i]
    
    # Add batch dimension
    obs = {
        "chart": torch.tensor(sample["chart"]).unsqueeze(0),
        "numeric": torch.tensor(sample["numeric"]).unsqueeze(0),
        "context": torch.tensor(sample["context"]).unsqueeze(0),
    }
    
    with torch.no_grad():
        features = model.encode(obs["chart"], obs["numeric"], obs["context"])
        out = model(chart=obs["chart"], numeric=obs["numeric"], context=obs["context"])
        
    print(f"\n--- Snapshot ID {i} ---")
    print(f"Vision Chart: {obs['chart'].shape}")
    print(f"Numeric Stats: {obs['numeric'].shape}")
    print(f"Time Context: {obs['context'].shape}")
    
    print(f"\n=> Learned Brain Features (Vector of {features.shape[-1]} numbers):")
    # Print the first 10 numbers to see what they look like
    feats_preview = features[0, :10].numpy().round(3).tolist()
    print(f"   Preview: {feats_preview} ...")
    print(f"   Feature StdDev (Activation): {features.std().item():.4f}")
    
    # S1 didn't train logits/value, so they will be random, but we can print them.
    # The brain features ARE trained!
    
print("\nDebug complete!")
