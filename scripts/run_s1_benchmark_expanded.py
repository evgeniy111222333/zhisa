import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import build_default_policy
from torch.utils.data import DataLoader

def run_suite(args):
    os.environ["ZHISA_FAST_RENDER"] = "1"
    
    print(f"Loading data from {args.dataset}...")
    df = pd.read_parquet(args.dataset)
    df = df.sort_index().iloc[-args.bars:] 
    
    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128)
    ds = MarketDataset(df, spec=spec, compute_targets=True)
    loader = DataLoader(ds, batch_size=64, collate_fn=multimodal_collate, shuffle=False)
    
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
    
    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    policy_state = {k: v for k, v in model_state.items() if not k.startswith("reconstructor.")}
    model.load_state_dict(policy_state, strict=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    embs = []
    embs_no_ctx = []
    embs_masked_img = []
    
    labels_dir = []
    labels_vol = []
    labels_regime = []
    
    returns_1step = []
    returns_5step = []
    returns_10step = []
    
    close_prices = df["close"].values
    
    print("Running inference...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            chart = batch.chart.to(device)
            numeric = batch.numeric.to(device)
            context = batch.context.to(device)
            
            out = model(chart=chart, numeric=numeric, context=context)
            emb = out["embedding"]
            
            out_no_ctx = model(chart=chart, numeric=numeric, context=torch.zeros_like(context))
            emb_no_ctx = out_no_ctx["embedding"]
            
            masked_chart = chart.clone()
            W = masked_chart.shape[-1]
            mask_start = int(W * 0.7)
            masked_chart[:, :, :, mask_start:] = 0
            out_masked = model(chart=masked_chart, numeric=numeric, context=context)
            emb_masked = out_masked["embedding"]
            
            embs.append(emb.cpu().numpy())
            embs_no_ctx.append(emb_no_ctx.cpu().numpy())
            embs_masked_img.append(emb_masked.cpu().numpy())
            
            labels_dir.append(batch.label_dir.cpu().numpy())
            labels_vol.append(batch.label_vol.cpu().numpy())
            labels_regime.append(batch.label_regime.cpu().numpy())
            
    X = np.concatenate(embs)
    X_no_ctx = np.concatenate(embs_no_ctx)
    X_masked = np.concatenate(embs_masked_img)
    
    y_dir = np.concatenate(labels_dir) + 1 
    y_vol = np.concatenate(labels_vol)
    y_regime = np.concatenate(labels_regime)
    
    start_idx = spec.chart_window
    for i in range(len(X)):
        current_idx = start_idx + i
        if current_idx + 10 < len(close_prices):
            r1 = (close_prices[current_idx+1] - close_prices[current_idx]) / close_prices[current_idx]
            r5 = (close_prices[current_idx+5] - close_prices[current_idx]) / close_prices[current_idx]
            r10 = (close_prices[current_idx+10] - close_prices[current_idx]) / close_prices[current_idx]
        else:
            r1, r5, r10 = 0.0, 0.0, 0.0
        returns_1step.append(1 if r1 > 0 else 0)
        returns_5step.append(1 if r5 > 0 else 0)
        returns_10step.append(1 if r10 > 0 else 0)
        
    y_ret1 = np.array(returns_1step)
    y_ret5 = np.array(returns_5step)
    y_ret10 = np.array(returns_10step)
    
    print("\n" + "="*60)
    print("S1 REPRESENTATION LEARNING EXPERIMENTS & BENCHMARKS")
    print(f"Dataset: {os.path.basename(args.dataset)} | Samples: {len(X)}")
    print("="*60)
    
    def run_probe(X, y, is_classification=True, name=""):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        if is_classification:
            clf_lin = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            acc_lin = accuracy_score(y_test, clf_lin.predict(X_test))
            clf_nl = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
            acc_nl = accuracy_score(y_test, clf_nl.predict(X_test))
            
            # Dummy baseline
            dummy_acc = max(np.mean(y_test == 0), np.mean(y_test == 1), np.mean(y_test == 2)) if len(np.unique(y_test)) > 1 else 1.0
            print(f"[{name}]")
            print(f"  Baseline (Majority): {dummy_acc:.1%}")
            print(f"  Linear Probe Acc:    {acc_lin:.1%}")
            print(f"  RF Probe Acc:        {acc_nl:.1%}\n")
        else:
            reg_lin = Ridge().fit(X_train, y_train)
            r2_lin = r2_score(y_test, reg_lin.predict(X_test))
            reg_nl = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
            r2_nl = r2_score(y_test, reg_nl.predict(X_test))
            print(f"[{name}]")
            print(f"  Linear Probe R2:     {r2_lin:.3f}")
            print(f"  RF Probe R2:         {r2_nl:.3f}\n")
            
    print("--- 1. MARKET PREDICTION CAPABILITY ---")
    run_probe(X, y_dir, is_classification=True, name="Short-term Direction (Label Dir)")
    run_probe(X, y_ret1, is_classification=True, name="Strict 1-Step Future Return (>0)")
    run_probe(X, y_ret5, is_classification=True, name="Strict 5-Step Future Return (>0)")
    run_probe(X, y_ret10, is_classification=True, name="Strict 10-Step Future Return (>0)")
    
    print("--- 2. RISK & REGIME UNDERSTANDING ---")
    run_probe(X, y_vol, is_classification=False, name="Future Volatility Prediction")
    run_probe(X, y_regime, is_classification=True, name="Market Regime Classification")
    
    print("--- 3. REPRESENTATION ROBUSTNESS ---")
    l2_ctx = np.linalg.norm(X - X_no_ctx, axis=-1).mean()
    print(f"[Context Sensitivity]")
    print(f"  Avg L2 Distance (no context): {l2_ctx:.3f}\n")
    
    l2_mask = np.linalg.norm(X - X_masked, axis=-1).mean()
    print(f"[Image Masking]")
    print(f"  Avg L2 Distance (30% right masked): {l2_mask:.3f}\n")
    
    print("--- 4. LATENT SPACE ANALYSIS ---")
    norms = np.linalg.norm(X, axis=-1)
    print(f"[Embedding Norms] Mean: {norms.mean():.2f} | Std: {norms.std():.4f}")
    
    cos_sims = []
    for i in range(len(X)-1):
        num = np.dot(X[i], X[i+1])
        den = np.linalg.norm(X[i]) * np.linalg.norm(X[i+1])
        if den > 0:
            cos_sims.append(num / den)
    print(f"[Temporal Continuity] Avg Cosine Sim (t vs t+1): {np.mean(cos_sims):.3f}")
    
    pca = PCA(n_components=10)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    print(f"[Information Density] Var explained by Top 10 PCA: {evr.sum():.1%}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to parquet dataset")
    parser.add_argument("--bars", type=int, default=3000, help="Number of recent bars to evaluate")
    args = parser.parse_args()
    run_suite(args)
