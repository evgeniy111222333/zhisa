"""Probe an S1 checkpoint: verify the representation actually learned what is intended.

Runs a battery of diagnostics on a frozen S1 representation:

1. **Instrument separation** вЂ” do the 12 market embeddings form distinct clusters?
   (centroid-distance index + silhouette on a subset).
2. **Regime / label structure** вЂ” distribution of HMM regime & direction labels, and
   how well regime labels align with the embedding geometry (silhouette).
3. **Frozen linear probe (direction)** вЂ” a shallow head on *frozen* embeddings,
   evaluated out-of-sample (balanced accuracy) vs prior/majority baseline. This is
   the classic "does the SSL representation carry predictive info" test.
4. **Temporal CPC separation** вЂ” mean cosine to the *next* state vs to random
   negatives (must be clearly positive separation).
5. **Vision<->numeric alignment** вЂ” mean cosine between the vision and numeric
   encoders on the same window (SSL alignment objective, evaluated).

Output is a JSON report. The script is CPU/GPU agnostic (--device auto=cuda).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.data.preparation import load_prepared_symbol
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.rendering.spec import RenderSpec
from zhisa.rendering.chart_renderer import render_chart


def _build_policy_from_checkpoint(path: str, device) -> PolicyNetwork:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(sd["model_config"])
    if isinstance(cfg.get("vision_channels"), list):
        cfg["vision_channels"] = tuple(cfg["vision_channels"])
    if isinstance(cfg.get("market_horizons"), list):
        cfg["market_horizons"] = tuple(int(x) for x in cfg["market_horizons"])
    allowed = {f.name for f in PolicyConfig.__dataclass_fields__.values()}
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    model = PolicyNetwork(PolicyConfig(**cfg)).to(device)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()
    return model


def _center(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def _silhouette(Z, labels, sample=12000, seed=0):
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n = len(Z)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    return float(silhouette_score(Z[idx], labels[idx], sample_size=min(sample, 8000), random_state=seed))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--symbols", default=None, help="comma list; default = all in manifest")
    ap.add_argument("--per-symbol", type=int, default=6000, help="max samples per symbol")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args(argv)

    import zhisa.rendering.chart_renderer as cr

    root = Path(args.prepared_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    symbols = args.symbols.split(",") if args.symbols else manifest["symbols"]
    symbols = [s.replace("/", "_") for s in symbols]
    instr_id = {s: i for i, s in enumerate(symbols)}

    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128, horizons=(4, 16, 64))
    device = torch.device(args.device)
    model = _build_policy_from_checkpoint(args.checkpoint, device)

    Zs, instr_labels, regime_labels, dir_labels, ts = [], [], [], [], []
    dose_t, dose_neg, align_list = [], [], []

    for symbol in symbols:
        df = pd.read_parquet(root / "symbols" / f"{symbol}.parquet")
        df = df.sort_index()
        ds = MarketDataset(
            df, spec=spec,
            cache_charts=False, chart_cache_size=-1, compute_targets=True,
            instrument_id=instr_id[symbol],
        )
        n = min(len(ds), args.per_symbol)
        rng = np.random.default_rng(hash(symbol) % 2**32)
        idx = np.sort(rng.choice(len(ds), size=n, replace=False))

        for bs in range(0, n, 64):
            inds = idx[bs:bs + 64]
            batch = multimodal_collate([ds[int(i)] for i in inds])
            with torch.no_grad():
                z = model.encode(
                    batch.chart.to(device), batch.numeric.to(device),
                    batch.context.to(device), instrument_id=batch.instrument_id.to(device),
                )
                v = model.vision(batch.chart.to(device))
                ncl, _ = model.numeric(batch.numeric.to(device))
            z = z.cpu().numpy()
            Zs.append(z)
            instr_labels.append(np.full(len(inds), instr_id[symbol], dtype=int))
            regime_labels.append(batch.label_regime.numpy())
            dir_labels.append(batch.label_dir.numpy())
            ts.append(batch.meta)
            # temporal CPC: next state by +horizon within dataset
            fut = [ds[int(i) + 4] if int(i) + 4 < len(ds) else ds[int(i)] for i in inds]
            fb = multimodal_collate(fut)
            with torch.no_grad():
                zf = model.encode(
                    fb.chart.to(device), fb.numeric.to(device),
                    fb.context.to(device), instrument_id=fb.instrument_id.to(device),
                ).cpu().numpy()
            dose_t.append(np.sum(_center(z) * _center(zf), axis=1))
            rnd = np.sort(np.random.default_rng(0).choice(len(Zs[0]) if False else len(idx), size=len(inds), replace=True))
            dose_neg.append(np.sum(_center(z) * _center(z[np.roll(np.arange(len(z)), 7)]), axis=1))
            # vision<->numeric alignment
            align_list.append(
                torch.nn.functional.cosine_similarity(v, ncl, dim=1).cpu().numpy()
            )

    Z = np.concatenate(Zs)
    instr = np.concatenate(instr_labels)
    regime = np.concatenate(regime_labels)
    direc = np.concatenate(dir_labels)

    report = {
        "checkpoint": args.checkpoint,
        "n_samples": int(len(Z)),
        "n_symbols": len(symbols),
        "finite": bool(np.isfinite(Z).all()),
        "embedding_norm_mean": float(np.linalg.norm(Z, axis=1).mean()),
        "embedding_std": float(Z.std()),
    }

    # 1) instrument separation
    instrument_silh = _silhouette(Z, instr, sample=20000)
    centroids = np.stack([
        Z[instr == i].mean(0) for i in range(len(symbols))
    ])
    c = _center(centroids)
    cents = np.sum(c * c, axis=1)
    cdist = (c @ c.T)
    off = (cdist.sum() - np.trace(cdist)) / max(len(symbols) * (len(symbols) - 1), 1)
    within = [np.mean(_center(Z[instr == i]) @ _center(Z[instr == i]).T - np.eye((instr == i).sum())) for i in range(len(symbols))]
    report["instrument_separation"] = {
        "silhouette": instrument_silh,
        "centroid_cos_offdiag_mean": float(off),
        "within_similarity_mean": float(np.mean(within)),
    }

    # 2) regime / direction label balance + regime silhouette
    reg_counts = np.bincount(regime, minlength=int(regime.max() + 1))
    dir_counts = np.bincount(direc + 1, minlength=3)
    report["label_balance"] = {
        "regime_counts": reg_counts.tolist(),
        "direction_counts": np.array([dir_counts[0], dir_counts[1], dir_counts[2]]).tolist(),
    }
    try:
        report["regime_silhouette"] = _silhouette(Z, regime, sample=20000)
    except Exception as e:
        report["regime_silhouette"] = f"err:{e}"

    # 3) frozen linear probe on direction
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = Z
    y = direc.astype(int)
    rng = np.random.default_rng(1)
    m = len(y)
    test_mask = rng.random(m) < 0.4
    tr, te = ~test_mask, test_mask
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te]))
    bal, prior_bal = {}, None
    tpred = pred
    for cls in np.unique(y):
        mte = y[te] == cls
        acc = np.mean(tpred[mte] == cls) if mte.sum() else 0.0
        bal[int(cls)] = acc
    # prior/majority baseline: always predict most common class
    maj = int(np.bincount(y[tr] + 1).argmax() - 1)
    base = float(np.mean(y[te] == maj))
    report["direction_probe"] = {
        "balanced": bal,
        "majority_baseline": float(base),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
    }

    # 4) temporal CPC separation
    tp = np.concatenate(dose_t)
    tn = np.concatenate(dose_neg)
    report["temporal_cpc"] = {
        "mean_cos_positive": float(tp.mean()),
        "mean_cos_negative": float(tn.mean()),
        "separation": float((tp - tn).mean()),
    }

    # 5) vision<->numeric alignment
    al0 = np.concatenate(align_list)
    report["alignment"] = {"mean_cos_vision_numeric": float(al0.mean())}

    out = args.report_out or "s1_probe_report.json"
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())