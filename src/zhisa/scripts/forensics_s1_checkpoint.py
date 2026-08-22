"""Deep forensics of an S1 checkpoint: internals + practical behavioural tests.

This is NOT another loss-curve report. It digs into the trained model:

- metadata/compare of ``best`` vs ``last`` checkpoints (epochs, val totals,
  weight distance);
- weight internals (near-zero fraction, norms per module, EMAв†”student
  agreement, instrument-embedding diversity);
- behaviour on REAL data (BTC/TRX 1h):
    * temporal autocorrelation of embeddings (smoothness over lags),
    * visionв†”numeric alignment,
    * cross-instrument separation,
    * CPC forward-prediction quality (cosine + top-1 among in-batch negatives),
    * masked reconstruction error vs a naive mean-baseline,
    * perturbation sensitivity (numeric-scale Lipschitz estimate);
- the same behavioural battery on a RANDOM-initialised twin в†’ "learned vs not";
- kNN chart retrieval rendered to PNGs so the result can be eyeballed.

Usage::

    python -m zhisa.scripts.forensics_s1_checkpoint \
        --best /data/out/phase1_heavy_best.pt --last /data/out/phase1_heavy_last.pt \
        --prepared-root /data/datasets/s1_1h_12m_v2 --symbols BTC_USDT,TRX_USDT \
        --out /data/out/forensics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.models.policy import PolicyConfig, PolicyNetwork
from zhisa.rendering.chart_renderer import render_chart_visualization
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer


def _policy_from_config(model_config: dict) -> PolicyNetwork:
    cfg = dict(model_config)
    if isinstance(cfg.get("vision_channels"), list):
        cfg["vision_channels"] = tuple(int(x) for x in cfg["vision_channels"])
    if isinstance(cfg.get("market_horizons"), list):
        cfg["market_horizons"] = tuple(int(x) for x in cfg["market_horizons"])
    allowed = {f.name for f in PolicyConfig.__dataclass_fields__.values()}
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    return PolicyNetwork(PolicyConfig(**cfg))


def _weight_stats(model) -> dict:
    tot_near, total = 0, 0
    per = {}
    for name, p in model.named_parameters():
        n = p.numel()
        if n == 0:
            continue
        nnz = (p.detach().abs() > 1e-6).sum().item()
        per[name] = {
            "std": float(p.detach().std().item()),
            "near_zero_frac": round(1.0 - nnz / n, 4),
        }
        total += n
        tot_near += (n - nnz)
    return {"total_params": total, "near_zero_total_frac": round(tot_near / total, 4), "per_layer": per}


def _cos_matrix(emb: torch.Tensor) -> np.ndarray:
    e = torch.nn.functional.normalize(emb, dim=-1)
    return (e @ e.T).detach().cpu().numpy()


def _embeddings(model, ds, device, n=1500, bs=64, seed=7):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    Z, T = [], []
    for s in range(0, len(idx), bs):
        b = multimodal_collate([ds[int(i)] for i in idx[s:s + bs]])
        with torch.no_grad():
            z = model.encode(
                b.chart.to(device), b.numeric.to(device),
                b.context.to(device), instrument_id=b.instrument_id.to(device),
            )
        Z.append(z.detach().cpu().numpy())
        T.append(b.meta)
    return np.concatenate(Z), idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--symbols", default="BTC_USDT,TRX_USDT")
    ap.add_argument("--out", default="/data/out/forensics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples", type=int, default=1500)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    best_payload = torch.load(args.best, map_location="cpu", weights_only=False)
    last_payload = torch.load(args.last, map_location="cpu", weights_only=False)
    report = {"best": args.best, "last": args.last}

    def meta_summary(payload) -> dict:
        ts = payload.get("trainer_state", {}) or {}
        hist = ts.get("history", []) or []
        vals = [float(h.get("val", {}).get("total", float("nan"))) for h in hist]
        return {
            "completed_epochs": ts.get("completed_epochs"),
            "step": ts.get("step"),
            "val_totals_per_epoch": [round(v, 4) if v == v else None for v in vals],
            "best_val_total": ts.get("best_val_total"),
            "warm_start_from": (payload.get("checkpoint_meta") or {}).get("warm_start_from"),
        }

    report["best_meta"] = meta_summary(best_payload)
    report["last_meta"] = meta_summary(last_payload)
    mc = best_payload.get("model_config") or {}
    report["model_config_short"] = {
        k: mc.get(k) for k in ("embed_dim", "vision_mode", "numeroc_causal", "numeric_layers",
                                "fusion_layers", "memory_layers", "n_instruments", "encoder_ff_mult")
    }

    model_best = _policy_from_config(mc)
    model_last = _policy_from_config(mc)
    # weight distance best vs last
    db = [(n, p) for n, p in model_best.named_parameters()]
    dl = dict(model_last.named_parameters())
    d2 = sum(((p.detach() - dl[n].detach()).float().pow(2).sum().item())
             for n, p in db if n in dl)
    report["best_last_weight_l2"] = round(float(np.sqrt(d2)), 3)

    report["weight_stats_best"] = _weight_stats(model_best)

    # SSLPretrainer for heads/teacher/reconstructor + EMA agreement
    ssl_cfg_src = best_payload.get("ssl_config") or {}
    tr = SSLPretrainer(
        model_best,
        SSLConfig(device="cpu", batch_size=8,
                  projection_dim=int(ssl_cfg_src.get("projection_dim", 128)),
                  hidden_dim=int(ssl_cfg_src.get("hidden_dim", 256)),
                  use_ema_teacher=True, use_masked_modeling=True,
                  use_temporal_contrast=True, use_cross_modal=True),
    )
    tr.load(args.best)
    # EMA<->student agreement computed in CPU space (teacher stays on cpu)
    if tr.teacher is not None:
        pairs = [(tp, sp) for tp, sp in zip(tr.teacher.teacher.parameters(),
                                            tr.model.parameters())
                 if tp.numel() == sp.numel()]
        ema_cos = sum(
            torch.nn.functional.cosine_similarity(
                tp.detach().flatten(), sp.detach().flatten(), dim=0
            ).item() for tp, sp in pairs
        ) / max(len(pairs), 1)
        report["ema_student_cos_mean"] = round(float(ema_cos), 4)
    model_best = tr.model.to(device)
    model_best.eval()
    for _m in (tr.proj_temporal, tr.temporal_predictor, tr.proj_vision, tr.proj_numeric,
               tr.reconstructor, tr.target_proj_temporal):
        _m.to(device)

    # instrument embedding diversity
    emb = tr.model.context.instrument_emb.weight.detach()
    C = _cos_matrix(emb)
    off = C[~np.eye(len(C), dtype=bool)]
    report["instrument_emb"] = {"max_offdiag_cos": round(float(off.max()), 4),
                                "mean_offdiag_cos": round(float(off.mean()), 4)}

    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128, horizons=(4, 16, 64))
    symbols = args.symbols.split(",")
    instr_id = {s: i for i, s in enumerate(symbols)}

    def battery(model, tag: str, ds_list, extra=None):
        Zs, instr, ts_all = [], [], []
        for sym in symbols:
            ds = ds_list[sym]
            Z, idx = _embeddings(model, ds, device, n=args.samples)
            Zs.append(Z)
            instr.append(np.full(len(Z), instr_id[sym]))
        Z = np.concatenate(Zs)
        rep = {
            "finite": bool(np.isfinite(Z).all()),
            "norm_mean": round(float(np.linalg.norm(Z, axis=1).mean()), 4),
        }
        # temporal autocorrelation per symbol (fixed start window)
        arcs = {}
        for sym in symbols:
            ds = ds_list[sym]
            kvals = np.arange(1, 13)
            cosk = []
            with torch.no_grad():
                b0 = multimodal_collate([ds[i] for i in range(0, 96, 2)])
                z0 = model.encode(b0.chart.to(device), b0.numeric.to(device),
                                  b0.context.to(device), instrument_id=b0.instrument_id.to(device)).detach().cpu()
            for k in kvals:
                bk = multimodal_collate([ds[min(i + k, len(ds) - 1)] for i in range(0, 96, 2)])
                with torch.no_grad():
                    zk = model.encode(bk.chart.to(device), bk.numeric.to(device),
                                      bk.context.to(device), instrument_id=bk.instrument_id.to(device)).detach().cpu()
                cosk.append(torch.nn.functional.cosine_similarity(z0, zk, dim=-1).mean().item())
            arcs[sym] = cosk
        rep["temporal_autocorr_cos"] = {s: [round(x, 4) for x in v] for s, v in arcs.items()}
        # vision<->numeric alignment
        ds = ds_list[symbols[0]]
        als = []
        b = multimodal_collate([ds[i] for i in range(0, 128, 4)])
        with torch.no_grad():
            v = model.plain_vision(b.chart.to(device))
            n, _ = model.numeric(b.numeric.to(device))
            als = torch.nn.functional.cosine_similarity(v, n, dim=-1).mean().item()
        rep["alignment_cos"] = round(float(als), 4)
        # CPC forward-prediction
        from zhisa.training.s1_ssl import TemporalPairDataset
        h = int(ssl_cfg_src.get("temporal_horizon", 4) or 4)
        pairs = TemporalPairDataset(ds, horizon=h)
        rng = np.random.default_rng(1)
        idx = rng.choice(len(pairs), size=min(256, len(pairs)), replace=False)
        cs, hits = [], 0
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                cur, fut = zip(*[pairs[int(i)] for i in idx[s:s + 64]])
                cb = multimodal_collate(cur)
                fb = multimodal_collate(fut)
                zc = model.encode(cb.chart.to(device), cb.numeric.to(device),
                                  cb.context.to(device), instrument_id=cb.instrument_id.to(device))
                zf = model.encode(fb.chart.to(device), fb.numeric.to(device),
                                  fb.context.to(device), instrument_id=fb.instrument_id.to(device))
                pt = tr.temporal_predictor(tr.proj_temporal(zc))
                tg = tr.target_proj_temporal(zf)
                sim = torch.nn.functional.cosine_similarity(
                    torch.nn.functional.normalize(pt, dim=-1).unsqueeze(0),
                    torch.nn.functional.normalize(tg, dim=-1).unsqueeze(1), dim=-1)
                cs.append(float(sim.diag().mean().item()))
                hits += int((sim.argmax(dim=1) == torch.arange(sim.size(0), device=sim.device)).sum().item())
        rep["cpc_forward"] = {"mean_cos": round(float(np.mean(cs)), 4),
                              "top1_in_batch": round(hits / len(idx), 4)}
        # masked reconstruction vs mean baseline
        from zhisa.training.s1_ssl import _MaskedReconstructor
        n_patch = model.numeric.n_patches
        recon = _MaskedReconstructor(model.numeric.cfg.d_model, model.numeric.cfg.patch_size,
                                     model.numeric.cfg.in_features).to(device)
        recon.load_state_dict(tr.reconstructor.state_dict())
        b = multimodal_collate([ds[i] for i in range(0, 256, 4)])
        x = b.numeric.to(device)
        n_patches = model.numeric.n_patches
        patch = model.numeric.cfg.patch_size
        patches = x.view(*x.shape[:1], n_patches, patch, -1).reshape(x.size(0), n_patches, -1)
        mask = torch.zeros_like(patches[..., 0])
        mask[:, ::5] = 1.0
        masked_win = (patches * mask.unsqueeze(-1)).view_as(x)
        with torch.no_grad():
            _, tok = model.numeric(masked_win)
            pred = recon(tok)
            if getattr(model.numeric.cfg, "causal", False):
                pred_p = pred[:, :n_patches]
            else:
                pred_p = pred[:, 1:]
            err_pred = (pred_p[mask.bool()] - patches[mask.bool()]).pow(2).mean().item()
            meanbl = patches - patches.mean(dim=1, keepdim=True)
            err_bl = (meanbl[mask.bool()]).pow(2).mean().item()
        rep["masked_recon"] = {"masked_mse": round(err_pred, 5),
                               "mean_baseline_mse": round(err_bl, 5),
                               "gain_vs_baseline": round(err_bl / max(err_pred, 1e-9), 3)}
        # perturbation Lipschitz (numeric scale)
        with torch.no_grad():
            b0 = multimodal_collate([ds[i] for i in range(0, 128, 8)])
            z0 = model.encode(b0.chart.to(device), b0.numeric.to(device),
                              b0.context.to(device), instrument_id=b0.instrument_id.to(device))
            z1 = model.encode(b0.chart.to(device), (b0.numeric.to(device) * 1.01),
                              b0.context.to(device), instrument_id=b0.instrument_id.to(device))
        rep["numeric_perturb_1pct"] = {
            "delta_cos": round(float(torch.nn.functional.cosine_similarity(z0, z1, dim=-1).mean().item()), 5),
            "delta_norm": round(float((z0 - z1).norm(dim=-1).mean().item()), 5),
        }
        rep["tag"] = tag
        return Z, np.concatenate(instr), rep

    ds_list = {}
    sym_frames = {}
    for sym in symbols:
        df = pd.read_parquet(Path(args.prepared_root) / "symbols" / f"{sym}.parquet").sort_index()
        sym_frames[sym] = df
        ds_list[sym] = MarketDataset(df, spec=spec, cache_charts=False, compute_targets=False,
                                     instrument_id=instr_id[sym])

    Zb, ib, rep_best = battery(model_best, "trained_best", ds_list)
    report["behaviour_trained"] = rep_best

    # random-init twin (same architecture, no checkpoint)
    rmodel = _policy_from_config(mc).to(device)
    rmodel.eval()
    Zr, ir, rep_rnd = battery(rmodel, "random_init", ds_list)
    report["behaviour_random"] = rep_rnd

    # instrument separation (trained)
    from sklearn.metrics import silhouette_score
    try:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Zb), size=min(4000, len(Zb)), replace=False)
        Zi = torch.nn.functional.normalize(torch.from_numpy(Zb[idx]), dim=-1).numpy()
        report["instrument_separation_silhouette"] = round(float(silhouette_score(Zi, ib[idx], sample_size=3000, random_state=0)), 4)
    except Exception as e:
        report["instrument_separation_silhouette"] = f"err:{e}"

    # kNN chart retrieval (render the visuals; best-effort)
    try:
        btc_df = sym_frames[symbols[0]]
        ds = ds_list[symbols[0]]
        nn_idx = np.random.default_rng(3).choice(min(len(ds), 3000), size=3, replace=False)
        Zq, idxq = _embeddings(model_best, ds, device, n=3000, seed=2)
        sim = _cos_matrix(torch.from_numpy(Zq))  # (3000,3000)
        order = np.argsort(-sim, axis=1)
        from PIL import Image
        created = []
        for qi, q in enumerate(nn_idx):
            nbrs = order[q][:6]
            fig_paths = []
            for ni, nbr in enumerate(nbrs):
                t0 = int(idxq[nbr])
                win = btc_df.iloc[max(0, t0) : max(0, t0) + 128][["open", "high", "low", "close", "volume"]]
                img = render_chart_visualization(win, size=128)
                arr = (np.asarray(img).clip(0.0, 1.0) * 255).astype(np.uint8)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = arr[:, :, ::-1] if arr.shape[-1] == 3 else arr
                    p = out / f"knn_q{qi}_n{ni}.png"
                    Image.fromarray(arr).save(p)
                    fig_paths.append(str(p))
                else:
                    fig_paths.append(f"skip_bad_shape:{arr.shape}")
            created.append({"query_index": int(q), "neighbors": fig_paths})
        report["knn_thumbnails"] = created
    except Exception as e:  # thumbnails are best-effort visual aid
        report["knn_thumbnails"] = f"err:{e}"

    (out / "forensics_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())