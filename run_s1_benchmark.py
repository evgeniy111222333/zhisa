import os
import torch
import pandas as pd
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.models.policy import build_default_policy
import warnings
warnings.filterwarnings("ignore")

def run_benchmark():
    print("Loading data...")
    path = r"D:\zhisa\data\prepared\s1_15m_v1\symbols\BTC_USDT.parquet"
    df = pd.read_parquet(path)
    
    # Take a test slice (end of dataset)
    df = df.tail(1000)
    
    print("Initializing MarketDataset...")
    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128)
    os.environ["ZHISA_FAST_RENDER"] = "1"
    ds = MarketDataset(df, spec=spec, compute_targets=False)
    
    print("Building model...")
    n_feat = ds._features_df.shape[1]
    n_ctx = ds._time_features_df.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=n_ctx,
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
    )
    
    ckpt_path = r"d:\zhisa\artifacts\s1\model.pt"
    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    
    print("\n" + "="*70)
    print("S1 ACTIONS BENCHMARK (REAL-TIME REPLAY)")
    print("="*70)
    
    actions_map = {
        0: "STRONG SHORT (-1.0)",
        1: "SHORT (-0.75)",
        2: "LIGHT SHORT (-0.5)",
        3: "WEAK SHORT (-0.25)",
        4: "HOLD (0.0)",
        5: "WEAK LONG (+0.25)",
        6: "LIGHT LONG (+0.5)",
        7: "LONG (+0.75)",
        8: "STRONG LONG (+1.0)",
    }
    
    # Start offset
    start_idx = len(ds) - 50
    results = []
    
    for i in range(20):
        idx = start_idx + i
        batch = ds[idx]
        
        c = batch["chart"].unsqueeze(0)
        n = batch["numeric"].unsqueeze(0)
        ctx = batch["context"].unsqueeze(0)
        
        with torch.no_grad():
            out = model(chart=c, numeric=n, context=ctx)
            
        logits = out["policy_logits"][0]
        emb = out["embedding"][0]
        
        chosen_action = logits.argmax().item()
        confidence = logits.softmax(dim=0)[chosen_action].item() * 100
        
        row_idx = idx + spec.chart_window - 1
        ts = df.index[row_idx]
        price = df.iloc[row_idx]['close']
        funding = df.iloc[row_idx].get('ctx_funding_rate', 0.0)
        
        norm = emb.norm().item()
        
        print(f"Step {i+1:02d} | Time: {ts.strftime('%Y-%m-%d %H:%M')} | Price: ${price:.1f} | Fund: {funding:.5f}")
        print(f"       Embedding Power (Focus): {norm:.2f}")
        print(f"       Action Chosen: {actions_map[chosen_action]} (Confidence: {confidence:.1f}%)")
        print("-" * 70)
        
        results.append({
            "step": i+1,
            "time": ts.strftime('%Y-%m-%d %H:%M'),
            "price": price,
            "funding": funding,
            "focus": norm,
            "action": actions_map[chosen_action],
            "confidence": confidence
        })
        
    return results

if __name__ == "__main__":
    run_benchmark()
