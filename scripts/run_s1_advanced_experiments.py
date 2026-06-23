import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore")

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import build_default_policy
from torch.utils.data import DataLoader

def extract_embeddings(df_path, model, spec, bars=2000, device="cpu"):
    df = pd.read_parquet(df_path).sort_index().iloc[-bars:]
    ds = MarketDataset(df, spec=spec, compute_targets=True)
    loader = DataLoader(ds, batch_size=64, collate_fn=multimodal_collate, shuffle=False)
    
    embs, labels_dir, labels_regime, labels_vol = [], [], [], []
    raw_context = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {os.path.basename(df_path)}"):
            c = batch.chart.to(device)
            n = batch.numeric.to(device)
            ctx = batch.context.to(device)
            out = model(chart=c, numeric=n, context=ctx)
            embs.append(out["embedding"].cpu().numpy())
            labels_dir.append(batch.label_dir.cpu().numpy())
            labels_regime.append(batch.label_regime.cpu().numpy())
            labels_vol.append(batch.label_vol.cpu().numpy())
            raw_context.append(ctx.cpu().numpy())
            
    return {
        "X": np.concatenate(embs),
        "y_dir": np.concatenate(labels_dir) + 1,
        "y_regime": np.concatenate(labels_regime),
        "y_vol": np.concatenate(labels_vol),
        "ctx": np.concatenate(raw_context),
        "loader": loader
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["ZHISA_FAST_RENDER"] = "1"
    
    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128)
    
    # Load one dummy dataset to build the model
    dummy_df = pd.read_parquet(os.path.join(args.data_dir, "BTC_USDT.parquet")).iloc[:1000]
    dummy_ds = MarketDataset(dummy_df, spec=spec, compute_targets=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = build_default_policy(
        in_numeric_features=dummy_ds._features_df.shape[1],
        in_context_features=dummy_ds._time_features_df.shape[1],
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
    
    print("\n--- EXTRACTING EMBEDDINGS ---")
    data_btc = extract_embeddings(os.path.join(args.data_dir, "BTC_USDT.parquet"), model, spec, 3000, device)
    data_eth = extract_embeddings(os.path.join(args.data_dir, "ETH_USDT.parquet"), model, spec, 1500, device)
    data_doge = extract_embeddings(os.path.join(args.data_dir, "DOGE_USDT.parquet"), model, spec, 1500, device)
    
    print("\n--- EXP 1: ZERO-SHOT TRANSFER (CROSS-ASSET) ---")
    X_train, X_test, y_train, y_test = train_test_split(data_btc["X"], data_btc["y_dir"], test_size=0.3, shuffle=False)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc_btc = accuracy_score(y_test, clf.predict(X_test))
    acc_eth = accuracy_score(data_eth["y_dir"], clf.predict(data_eth["X"]))
    acc_doge = accuracy_score(data_doge["y_dir"], clf.predict(data_doge["X"]))
    
    print(f"BTC Test Acc (Trained on BTC): {acc_btc:.1%}")
    print(f"ETH Zero-Shot Acc (Trained on BTC): {acc_eth:.1%}")
    print(f"DOGE Zero-Shot Acc (Trained on BTC): {acc_doge:.1%}")
    
    with open(os.path.join(args.out_dir, "exp1_zeroshot.txt"), "w") as f:
        f.write(f"BTC: {acc_btc:.3f}\nETH: {acc_eth:.3f}\nDOGE: {acc_doge:.3f}\n")
        
    print("\n--- EXP 2: T-SNE VISUALIZATION ---")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(data_btc["X"][-1500:]) # Last 1500 for speed
    y_reg_tsne = data_btc["y_regime"][-1500:]
    y_dir_tsne = data_btc["y_dir"][-1500:]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_reg_tsne, palette="Set1", s=30)
    plt.title("S1 Embeddings by Market Regime")
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_dir_tsne, palette="coolwarm", s=30)
    plt.title("S1 Embeddings by Future Direction")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "tsne_clusters.png"))
    print(f"Saved t-SNE plot to {os.path.join(args.out_dir, 'tsne_clusters.png')}")
    
    print("\n--- EXP 3: K-NEAREST NEIGHBORS TRADING BOT ---")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance').fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"KNN Trading Bot Accuracy on BTC: {knn_acc:.1%} (Linear was {acc_btc:.1%})")
    with open(os.path.join(args.out_dir, "exp3_knn.txt"), "w") as f:
        f.write(f"KNN: {knn_acc:.3f}\n")
    
    print("\n--- EXP 4: FEATURE PERMUTATION IMPORTANCE (CONTEXT) ---")
    # We will perturb each context feature (dim 2 of context tensor) and measure L2 distance
    batch0 = next(iter(data_btc["loader"]))
    c0 = batch0.chart.to(device)
    n0 = batch0.numeric.to(device)
    ctx0 = batch0.context.to(device)
    with torch.no_grad():
        base_emb = model(chart=c0, numeric=n0, context=ctx0)["embedding"]
        
    n_ctx_features = ctx0.shape[1]
    importances = []
    for f_idx in range(n_ctx_features):
        ctx_perturbed = ctx0.clone()
        # Shuffle the values of this specific feature across the batch
        shuffled_indices = torch.randperm(ctx_perturbed.size(0))
        ctx_perturbed[:, f_idx] = ctx_perturbed[shuffled_indices, f_idx]
        
        with torch.no_grad():
            new_emb = model(chart=c0, numeric=n0, context=ctx_perturbed)["embedding"]
            
        l2_diff = torch.norm(base_emb - new_emb, dim=-1).mean().item()
        importances.append((f_idx, l2_diff))
        
    importances.sort(key=lambda x: x[1], reverse=True)
    print("Context Feature Importance (L2 Impact):")
    for rank, (f_idx, diff) in enumerate(importances[:5]):
        print(f"Rank {rank+1}: Feature idx {f_idx} -> Diff {diff:.4f}")
    with open(os.path.join(args.out_dir, "exp4_importance.txt"), "w") as f:
        f.write(str(importances))
        
    print("\n--- EXP 5: BLACK SWAN EVENT DETECTION ---")
    # Find the top 1% most volatile steps in the test set
    vol_threshold = np.percentile(data_btc["y_vol"], 99)
    black_swan_mask = data_btc["y_vol"] > vol_threshold
    
    normal_norms = np.linalg.norm(data_btc["X"][~black_swan_mask], axis=-1)
    swan_norms = np.linalg.norm(data_btc["X"][black_swan_mask], axis=-1)
    
    print(f"Normal Market Embedding Norm: {normal_norms.mean():.2f}")
    print(f"Black Swan Embedding Norm:    {swan_norms.mean():.2f}")
    with open(os.path.join(args.out_dir, "exp5_swan.txt"), "w") as f:
        f.write(f"Normal: {normal_norms.mean():.3f}\nSwan: {swan_norms.mean():.3f}\n")
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
