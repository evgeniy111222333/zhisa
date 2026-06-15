import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, r"D:\zhisa\src")
from zhisa.scripts._real_data import load_market_dataframe
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.models.policy import build_default_policy

print("="*60)
print("S1 DEEP ANALYSIS: ABLATION, TEMPORAL SIMILARITY & WEIGHTS")
print("="*60)

class FakeArgs:
    data_source = "tsdb"
    tsdb_root = "data/tsdb"
    symbol = "BTC/USDT"
    timeframe = "5m"
    with_futures_context = True
    futures_context_root = "data/futures_context/binance_usdm"

args = FakeArgs()
print("1. Loading 500 contiguous market snapshots...")
df = load_market_dataframe(args, default_bars=600)
spec = SampleSpec(chart_window=32, feature_window=32, image_size=32)
ds = MarketDataset(df, spec=spec)
n_feat = ds._features.shape[1]
n_ctx = ds._time_features.shape[1]

print("2. Loading S1 Checkpoint...")
checkpoint_path = "artifacts/s1/model_btc.pt"
sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

model = build_default_policy(
    in_numeric_features=n_feat, in_context_features=n_ctx,
    window=spec.chart_window, image_size=spec.image_size,
    n_actions=9, n_regime_classes=spec.n_regime_states,
)
from zhisa.training.s1_ssl import _filter_matching_state_dict
model.load_state_dict(_filter_matching_state_dict(sd["model"], model), strict=False)
model.eval()

# Gather 500 samples
N = min(500, len(ds))
charts, numerics, contexts = [], [], []
for i in range(N):
    sample = ds[i]
    charts.append(torch.tensor(sample["chart"]))
    numerics.append(torch.tensor(sample["numeric"]))
    contexts.append(torch.tensor(sample["context"]))

charts = torch.stack(charts)
numerics = torch.stack(numerics)
contexts = torch.stack(contexts)

@torch.no_grad()
def get_emb(c, n, ctx):
    return model.encode(c, n, ctx)

print("\n3. Running Modality Ablation (Sensitivity Analysis)...")
base_emb = get_emb(charts, numerics, contexts)

# Zero out each modality to see how much the embedding shifts
emb_no_chart = get_emb(torch.zeros_like(charts), numerics, contexts)
emb_no_num = get_emb(charts, torch.zeros_like(numerics), contexts)
emb_no_ctx = get_emb(charts, numerics, torch.zeros_like(contexts))

def shift_metric(base, ablated):
    # Euclidean distance normalized
    return F.mse_loss(base, ablated).item()

shift_chart = shift_metric(base_emb, emb_no_chart)
shift_num = shift_metric(base_emb, emb_no_num)
shift_ctx = shift_metric(base_emb, emb_no_ctx)
total_shift = shift_chart + shift_num + shift_ctx

print(f"   Vision (Chart) Reliance:  {shift_chart/total_shift*100:.1f}%  (Shift: {shift_chart:.4f})")
print(f"   Numeric Reliance:         {shift_num/total_shift*100:.1f}%  (Shift: {shift_num:.4f})")
print(f"   Context Reliance:         {shift_ctx/total_shift*100:.1f}%  (Shift: {shift_ctx:.4f})")

print("\n4. Temporal Smoothness (Cosine Similarity)...")
# Compare frame T with T+k
sims_1 = F.cosine_similarity(base_emb[:-1], base_emb[1:]).mean().item()
sims_10 = F.cosine_similarity(base_emb[:-10], base_emb[10:]).mean().item()
sims_100 = F.cosine_similarity(base_emb[:-100], base_emb[100:]).mean().item()

print(f"   Sim(T, T+1)   [5 min]:   {sims_1:.4f}  (Should be high)")
print(f"   Sim(T, T+10)  [50 min]:  {sims_10:.4f}")
print(f"   Sim(T, T+100) [8.3 hrs]: {sims_100:.4f}  (Should be much lower)")

print("\n5. Weight Norms Analysis (Are sub-networks alive?)...")
def get_norm(prefix):
    norms = [p.norm().item() for n, p in model.named_parameters() if prefix in n]
    return np.mean(norms) if norms else 0.0

print(f"   Vision Encoder Mean L2:  {get_norm('vision_encoder'):.4f}")
print(f"   Numeric Encoder Mean L2: {get_norm('numeric'):.4f}")
print(f"   Context Encoder Mean L2: {get_norm('context'):.4f}")
print(f"   Fusion Layer Mean L2:    {get_norm('fusion'):.4f}")

print("\nDone.")
