import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings

# Suppress sklearn convergence warnings for linear probes
warnings.filterwarnings("ignore")

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import build_default_policy
from torch.utils.data import DataLoader

def run_suite():
    os.environ["ZHISA_FAST_RENDER"] = "1"
    
    print("Loading data...")
    df = pd.read_parquet(r"d:\zhisa\data\prepared\s1_15m_v1\symbols\BTC_USDT.parquet")
    # Take 2000 bars for evaluation
    df = df.sort_index().iloc[:2000] 
    
    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128)
    ds = MarketDataset(df, spec=spec, compute_targets=True)
    loader = DataLoader(ds, batch_size=128, collate_fn=multimodal_collate, shuffle=False)
    
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
    ckpt = torch.load(r"d:\zhisa\artifacts\s1\model.pt", map_location="cpu")
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    
    # Exclude reconstructor keys from policy network
    policy_state = {k: v for k, v in model_state.items() if not k.startswith("reconstructor.")}
    model.load_state_dict(policy_state, strict=False)
    model.eval()
    
    embs = []
    embs_no_ctx = []
    embs_masked_img = []
    actions = []
    
    labels_dir = []
    labels_vol = []
    labels_regime = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # 1. Normal inference
            out = model(chart=batch.chart, numeric=batch.numeric, context=batch.context)
            emb = out["embedding"]
            logits = out["policy_logits"]
            
            # 2. No context inference
            out_no_ctx = model(chart=batch.chart, numeric=batch.numeric, context=torch.zeros_like(batch.context))
            emb_no_ctx = out_no_ctx["embedding"]
            
            # 3. Masked image inference (mask out the rightmost 30% of the image to simulate missing latest price action)
            masked_chart = batch.chart.clone()
            W = masked_chart.shape[-1]
            mask_start = int(W * 0.7)
            masked_chart[:, :, :, mask_start:] = 0
            out_masked = model(chart=masked_chart, numeric=batch.numeric, context=batch.context)
            emb_masked = out_masked["embedding"]
            
            embs.append(emb.cpu().numpy())
            embs_no_ctx.append(emb_no_ctx.cpu().numpy())
            embs_masked_img.append(emb_masked.cpu().numpy())
            actions.append(logits.argmax(dim=-1).cpu().numpy())
            
            labels_dir.append(batch.label_dir.cpu().numpy())
            labels_vol.append(batch.label_vol.cpu().numpy())
            labels_regime.append(batch.label_regime.cpu().numpy())
            
    # Concatenate
    X = np.concatenate(embs)
    X_no_ctx = np.concatenate(embs_no_ctx)
    X_masked = np.concatenate(embs_masked_img)
    acts = np.concatenate(actions)
    
    y_dir = np.concatenate(labels_dir)
    y_vol = np.concatenate(labels_vol)
    y_regime = np.concatenate(labels_regime)
    
    print("\n" + "="*50)
    print("S1 REPRESENTATION LEARNING BENCHMARK SUITE")
    print("="*50)
    
    # Test 1: Predictive Return Probe
    y_dir_cls = y_dir + 1
    X_train, X_test, y_train, y_test = train_test_split(X, y_dir_cls, test_size=0.3, shuffle=False)
    clf_dir = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc_dir = accuracy_score(y_test, clf_dir.predict(X_test))
    print(f"Test 1 | Return Probe Accuracy            : {acc_dir:.1%}")
    
    # Test 2: Volatility Probe
    X_train, X_test, y_train, y_test = train_test_split(X, y_vol, test_size=0.3, shuffle=False)
    reg_vol = Ridge().fit(X_train, y_train)
    r2_vol = r2_score(y_test, reg_vol.predict(X_test))
    print(f"Test 2 | Volatility Probe R2 Score        : {r2_vol:.3f}")
    
    # Test 3: Regime Classification Probe
    X_train, X_test, y_train, y_test = train_test_split(X, y_regime, test_size=0.3, shuffle=False)
    clf_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc_reg = accuracy_score(y_test, clf_reg.predict(X_test))
    print(f"Test 3 | Regime Probe Accuracy            : {acc_reg:.1%}")
    
    # Test 4: Context Sensitivity
    l2_ctx = np.linalg.norm(X - X_no_ctx, axis=-1).mean()
    print(f"Test 4 | Context Sensitivity (L2 dist)    : {l2_ctx:.3f}")
    
    # Test 5: Masking Robustness
    l2_mask = np.linalg.norm(X - X_masked, axis=-1).mean()
    print(f"Test 5 | Masking Robustness (L2 dist)     : {l2_mask:.3f}")
    
    # Test 6: Zero-Shot Action Distribution
    unique, counts = np.unique(acts, return_counts=True)
    dist = {int(k): round(float(v)/len(acts)*100, 1) for k, v in zip(unique, counts)}
    print(f"Test 6 | Action Distribution (%)          : {dist}")
    
    # Test 7: Action Consistency
    consistent = (acts[:-1] == acts[1:]).mean()
    print(f"Test 7 | Action Consistency (t vs t+1)    : {consistent:.1%}")
    
    # Test 8: Embedding Norm Stability
    norms = np.linalg.norm(X, axis=-1)
    print(f"Test 8 | Embedding Norm Stability         : {norms.mean():.2f} (std: {norms.std():.4f})")
    
    print("="*50)

if __name__ == "__main__":
    run_suite()
