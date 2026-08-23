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


def _policy_matching_checkpoint(model_config: dict, state: dict) -> tuple[PolicyNetwork, dict]:
    """Build a :class:`PolicyNetwork` whose parameter set matches ``state`` exactly.

    Checkpoints produced before the residual-memory feature (``memory_scale`` /
    ``memory_in_norm``) have a state dict with FEWER parameters than a default
    build. Build with defaults (as before) and detect the surplus; when present,
    rebuild with ``memory_residual / memory_input_norm = False`` so weight stats,
    gradient probes and embeddings are computed on the EXACT trained
    architecture instead of a model with extra fresh modules.
    """
    model = _policy_from_config(model_config)
    mk = set(model.state_dict().keys())
    sk = set(state.keys())
    if sk <= mk and len(mk) > len(sk):
        cand = dict(model_config)
        cand["memory_residual"] = False
        cand["memory_input_norm"] = False
        rebuilt = _policy_from_config(cand)
        if set(rebuilt.state_dict().keys()) <= sk:
            return rebuilt, cand
    return model, dict(model_config)


def _load_into(model: PolicyNetwork, state: dict) -> None:
    """Load checkpoint weights filtered to shape-matched keys (strict=False)."""
    sd = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in sd and v.shape == sd[k].shape}
    model.load_state_dict(filtered, strict=False)


def _summary_end(cfg) -> bool:
    """True when the numeric encoder puts its summary/CLS token LAST."""
    return (
        getattr(cfg, "summary_position", None) == "end"
        or getattr(cfg, "causal", False)
    )


def _idxs(ds, n: int, step: int):
    """Robust index range that never runs past a short dataset."""
    return range(0, min(n, len(ds)), step)


def _weight_stats(model) -> dict:
    tot_near, total = 0, 0
    n_nan, n_inf = 0, 0
    per = {}
    for name, p in model.named_parameters():
        n = p.numel()
        if n == 0:
            continue
        x = p.detach().float()
        n_nan += int(x.isnan().sum().item())
        n_inf += int(x.isinf().sum().item())
        nnz = (x.abs() > 1e-6).sum().item()
        # std under the unbiased estimator (ddof=1) is NaN for size-1 tensors;
        # report 0.0 so the report contains no bogus NaN entries.
        std = float(x.std().item()) if n > 1 else 0.0
        per[name] = {
            "std": round(float(std), 6),
            "abs_mean": round(float(x.abs().mean().item()), 6),
            "near_zero_frac": round(1.0 - nnz / n, 4),
        }
        total += n
        tot_near += (n - nnz)
    return {
        "total_params": total,
        "near_zero_total_frac": round(tot_near / total, 4),
        "nan_params": n_nan,
        "inf_params": n_inf,
        "per_layer": per,
    }


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


def _rank_metrics(Z: np.ndarray) -> tuple[float, int]:
    """top-10 SVD variance share + effective dims (>1% variance)."""
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    if Zc.shape[0] < 3:
        return 1.0, int(min(Zc.shape[1], max(Zc.shape[0], 1)))
    sv = np.linalg.svd(Zc, compute_uv=False)
    vs = (sv ** 2) / max(float((sv ** 2).sum()), 1e-12)
    return round(float(vs[:10].sum()), 4), int((vs > 0.01).sum())


def _gradient_balance(tr: SSLPretrainer, ds, device, batches: int = 2, n_cpc: int = 64) -> dict:
    """Gradient-l2 probe per module (vision/numeric/fusion/... ) over the SSL loss.

    Reproduces the deep-diag v2 headline: vision was starved vs numeric
    (~23x smaller gradient norm), which in turn hid most representation work
    in the projection heads.
    """
    from zhisa.training.s1_ssl import TemporalPairDataset
    pairs = TemporalPairDataset(ds, horizon=int(tr.cfg.temporal_horizon or 4))
    modules = {
        "vision": tr.model.vision,
        "numeric": tr.model.numeric,
        "context": tr.model.context,
        "fusion": tr.model.fusion,
        "memory": tr.model.memory,
        "heads": tr.model.heads,
        "proj_temporal": tr.proj_temporal,
        "temporal_predictor": tr.temporal_predictor,
        "proj_vision": tr.proj_vision,
        "proj_numeric": tr.proj_numeric,
        "reconstructor": tr.reconstructor,
    }
    acc = {k: [] for k in modules}
    tr.model.eval()
    rng = np.random.default_rng(0)
    done = 0
    for _ in range(max(1, int(batches)) * 2):
        if len(pairs) == 0:
            break
        idx = rng.choice(len(pairs), size=min(n_cpc, len(pairs)), replace=False)
        try:
            cur, fut = zip(*[pairs[int(i)] for i in idx])
        except Exception:
            break
        cb = multimodal_collate(cur)
        fb = multimodal_collate(fut)
        batch = {
            "chart": cb.chart, "numeric": cb.numeric, "context": cb.context,
            "instrument_id": getattr(cb, "instrument_id", None),
        }
        batch["future_chart"] = fb.chart
        batch["future_numeric"] = fb.numeric
        batch["future_context"] = fb.context
        losses = tr._loss(batch)
        if not torch.isfinite(losses["total"]):
            continue
        tr.opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        per = {}
        ok = True
        for k, m in modules.items():
            g2 = 0.0
            for p in m.parameters():
                if p.grad is not None and torch.isfinite(p.grad).all():
                    g2 += float(p.grad.detach().pow(2).sum())
            per[k] = float(np.sqrt(g2))
            if not np.isfinite(per[k]):
                ok = False
        if ok:
            for k in modules:
                acc[k].append(per[k])
            done += 1
        if done >= int(batches):
            break
    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    tot = sum(avg.values()) or 1e-12
    return {
        "grad_l2_per_module": {k: round(v, 4) for k, v in avg.items()},
        "grad_share_per_module": {k: round(v / tot, 4) for k, v in avg.items()},
        "vision_over_numeric": round(avg["vision"] / max(avg["numeric"], 1e-12), 4),
        "grad_total_l2": round(float(np.sqrt(sum(v * v for v in avg.values()))), 4),
        "n_batches": done,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--symbols", default="BTC_USDT,TRX_USDT")
    ap.add_argument("--out", default="/data/out/forensics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--chart-window", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--num-cpc", type=int, default=256)
    ap.add_argument("--grad-batches", type=int, default=2)
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
        k: mc.get(k) for k in ("embed_dim", "vision_mode", "numeric_causal", "n_regime_classes",
                                "vision_reader", "token_fusion", "freq_branch", "two_tokens_per_bar",
                                "numeric_layers", "fusion_layers", "memory_layers", "use_memory",
                                "n_instruments", "encoder_ff_mult")
    }

    # Build a model whose parameter set matches the checkpoint EXACTLY (the
    # checkpoint may predate memory_residual/memory_input_norm).
    _match, eff_cfg = _policy_matching_checkpoint(mc, best_payload.get("model") or {})
    del _match
    report["model_config_effective"] = {
        "memory_residual": bool(eff_cfg.get("memory_residual", True)),
        "memory_input_norm": bool(eff_cfg.get("memory_input_norm", True)),
    }
    meta = best_payload.get("checkpoint_meta") or {}
    ds_meta = meta.get("dataset") or {}
    report["provenance"] = {
        "stage": meta.get("stage"),
        "dataset_root": ds_meta.get("root"),
        "dataset_timeframe": ds_meta.get("timeframe"),
        "dataset_manifest_checksum": ds_meta.get("manifest_checksum"),
        "render": (meta.get("render") or {}),
        "trading_policy_ready": meta.get("trading_policy_ready"),
    }

    # FIX: previously the L2 was computed between two freshly-random models
    # (checkpoint weights were never loaded), producing a meaningless value.
    model_best = _policy_from_config(eff_cfg)
    model_last = _policy_from_config(eff_cfg)
    _load_into(model_best, best_payload.get("model") or {})
    _load_into(model_last, last_payload.get("model") or {})
    db = [(n, p) for n, p in model_best.named_parameters()]
    dl = dict(model_last.named_parameters())
    d2 = 0.0
    for n, p in db:
        if n in dl and p.numel() == dl[n].numel():
            diff = (p.detach().float() - dl[n].detach().float())
            if torch.isfinite(diff).all():
                d2 += float(diff.pow(2).sum())
    l2 = float(np.sqrt(np.nan_to_num(d2, nan=float("inf"), posinf=float("inf"))))
    report["best_last_weight_l2"] = round(l2, 3) if l2 != float("inf") else "nan_or_inf"

    report["weight_stats_best"] = _weight_stats(model_best)

    # SSLPretrainer for heads/teacher/reconstructor + EMA agreement
    ssl_cfg_src = best_payload.get("ssl_config") or {}
    tr = SSLPretrainer(
        model_best,
        SSLConfig(device=str(device), batch_size=8,
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

    spec = SampleSpec(chart_window=int(args.chart_window), feature_window=int(args.chart_window),
                       image_size=int(args.image_size), horizons=(4, 16, 64))
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
            r0 = list(_idxs(ds, 96, 2))
            if not r0:
                arcs[sym] = []
                continue
            with torch.no_grad():
                b0 = multimodal_collate([ds[i] for i in r0])
                z0 = model.encode(b0.chart.to(device), b0.numeric.to(device),
                                  b0.context.to(device), instrument_id=b0.instrument_id.to(device)).detach().cpu()
            for k in kvals:
                rk = [min(i + k, len(ds) - 1) for i in r0]
                bk = multimodal_collate([ds[i] for i in rk])
                with torch.no_grad():
                    zk = model.encode(bk.chart.to(device), bk.numeric.to(device),
                                      bk.context.to(device), instrument_id=bk.instrument_id.to(device)).detach().cpu()
                cosk.append(torch.nn.functional.cosine_similarity(z0, zk, dim=-1).mean().item())
            arcs[sym] = cosk
        rep["temporal_autocorr_cos"] = {s: [round(x, 4) for x in v] for s, v in arcs.items()}
        # vision<->numeric alignment (trunk-level cos), per symbol
        als_per = {}
        for sym in symbols:
            dsc = ds_list[sym]
            b = multimodal_collate([dsc[i] for i in _idxs(dsc, 128, 4)])
            with torch.no_grad():
                v = model.plain_vision(b.chart.to(device))
                n, _ = model.numeric(b.numeric.to(device))
                als_per[sym] = round(float(torch.nn.functional.cosine_similarity(
                    v, n, dim=-1).mean().item()), 4)
        rep["alignment_cos_per_symbol"] = als_per
        rep["alignment_cos"] = als_per.get(symbols[0], max(als_per.values(), default=0.0))
        # CPC forward-prediction
        from zhisa.training.s1_ssl import TemporalPairDataset
        ds = ds_list[symbols[0]]
        h = int(ssl_cfg_src.get("temporal_horizon", 4) or 4)
        pairs = TemporalPairDataset(ds, horizon=h)
        rng = np.random.default_rng(1)
        idx = rng.choice(len(pairs), size=min(int(args.num_cpc), len(pairs)), replace=False)
        cs_pos, cs_neg, hits = [], [], 0
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
                eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                cs_pos.append(float(sim[eye].mean().item()))
                cs_neg.append(float(sim[~eye].mean().item()))
                hits += int((sim.argmax(dim=1) == torch.arange(sim.size(0), device=sim.device)).sum().item())
        rep["cpc_forward"] = {
            "mean_cos": round(float(np.mean(cs_pos)), 4),
            "margin": round(float(np.mean(cs_pos) - np.mean(cs_neg)), 4),
            "top1_in_batch": round(hits / max(len(idx), 1), 4),
        }
        # masked reconstruction vs mean baseline
        from zhisa.training.s1_ssl import _MaskedReconstructor
        recon = _MaskedReconstructor(model.numeric.cfg.d_model, model.numeric.cfg.patch_size,
                                     model.numeric.cfg.in_features).to(device)
        recon.load_state_dict(tr.reconstructor.state_dict())
        mr_idxs = list(_idxs(ds, 256, 4))
        n_patches = model.numeric.n_patches
        patch = model.numeric.cfg.patch_size
        if mr_idxs:
            b = multimodal_collate([ds[i] for i in mr_idxs])
            x = b.numeric.to(device)
            patches = x.view(*x.shape[:1], n_patches, patch, -1).reshape(x.size(0), n_patches, -1)
            mask = torch.zeros_like(patches[..., 0])
            mask[:, ::5] = 1.0
            masked_win = (patches * mask.unsqueeze(-1)).view_as(x)
            summary_end = _summary_end(model.numeric.cfg)
            with torch.no_grad():
                _, tok_m = model.numeric(masked_win)
                pred_m = recon(tok_m)
                pred_p = pred_m[:, :n_patches] if summary_end else pred_m[:, 1:]
                err_pred = (pred_p[mask.bool()] - patches[mask.bool()]).pow(2).mean().item()
                # recon quality on a FULLY visible window (no masking) isolates
                # the encoder's pure representational power from the masking
                # difficulty (v2 causal-mask regression diagnostic).
                _, tok_v = model.numeric(x)
                pred_v = recon(tok_v)
                pred_a = pred_v[:, :n_patches] if summary_end else pred_v[:, 1:]
                err_all = (pred_a - patches).pow(2).mean().item()
                meanbl = patches - patches.mean(dim=1, keepdim=True)
                err_bl = (meanbl[mask.bool()]).pow(2).mean().item()
            rep["masked_recon"] = {
                "masked_mse": round(err_pred, 5),
                "visible_mse": round(err_all, 5),
                "visible_over_masked": round(err_all / max(err_pred, 1e-9), 4),
                "mean_baseline_mse": round(err_bl, 5),
                "gain_vs_baseline": round(err_bl / max(err_pred, 1e-9), 3),
            }
        # perturbation robustness (numeric-scale Lipschitz estimate)
        # NOTE: name semantics — cos is the POST-perturbation similarity
        # (1.0 = direction unchanged). angle_deg and per-channel additive
        # noise are the interpretable forms.
        with torch.no_grad():
            b0 = multimodal_collate([ds[i] for i in range(0, 128, 8)])
            z0 = model.encode(b0.chart.to(device), b0.numeric.to(device),
                              b0.context.to(device), instrument_id=b0.instrument_id.to(device))
            z1 = model.encode(b0.chart.to(device), (b0.numeric.to(device) * 1.01),
                              b0.context.to(device), instrument_id=b0.instrument_id.to(device))
            col_std = b0.numeric.std(dim=(0, 1), unbiased=True).to(device)
            z2 = model.encode(b0.chart.to(device),
                              (b0.numeric.to(device) + torch.randn_like(b0.numeric.to(device)) * (0.01 * col_std)),
                              b0.context.to(device), instrument_id=b0.instrument_id.to(device))
        cos01 = torch.nn.functional.cosine_similarity(z0, z1, dim=-1).clamp(-1.0, 1.0)
        cos02 = torch.nn.functional.cosine_similarity(z0, z2, dim=-1).clamp(-1.0, 1.0)
        rep["numeric_perturb_1pct"] = {
            "post_perturb_cos_scale": round(float(cos01.mean().item()), 5),
            "post_perturb_cos_additive": round(float(cos02.mean().item()), 5),
            "angle_deg_scale": round(float(torch.acos(cos01).mean().item() * 180.0 / np.pi), 4),
            "angle_deg_additive": round(float(torch.acos(cos02).mean().item() * 180.0 / np.pi), 4),
            "delta_norm": round(float((z0 - z1).norm(dim=-1).mean().item()), 5),
        }
        rep["tag"] = tag
        # ---- v2 internals: determinism, aug invariance, chart noise, rank ----
        try:
            with torch.no_grad():
                b0 = multimodal_collate([ds[i] for i in _idxs(ds, 96, 4)])
                chart = b0.chart.to(device)
                num = b0.numeric.to(device)
                ctx = b0.context.to(device)
                inst = b0.instrument_id.to(device)
                z_base = model.encode(chart, num, ctx, instrument_id=inst)
                z_again = model.encode(chart, num, ctx, instrument_id=inst)
                max_delta = float((z_base - z_again).abs().max().item())
                # keyed augment invariance (same window, deterministic keys)
                try:
                    from zhisa.rendering.augmentations import KeyedAugmentor
                    aug = KeyedAugmentor(transforms=("mirror", "color_jitter", "crop", "gaussian_noise"),
                                         strength=0.05, crop_frac=0.9, noise_std=0.01)
                    charts_aug = torch.stack(
                        [aug.apply(chart[i], f"f:{i}") for i in range(chart.size(0))], 0
                    )
                    z_aug = model.encode(charts_aug, num, ctx, instrument_id=inst)
                    aug_cos = float(torch.nn.functional.cosine_similarity(
                        z_base, z_aug, dim=-1).mean().item())
                except Exception as exc:
                    aug_cos = f"err:{type(exc).__name__}"
                # chart pixel-noise perturb (1% Gaussian)
                z_noisy = model.encode(chart + torch.randn_like(chart) * 0.01,
                                       num, ctx, instrument_id=inst)
                cn = torch.nn.functional.cosine_similarity(z_base, z_noisy, dim=-1).clamp(-1.0, 1.0)
                chart_noise_angle = float(torch.acos(cn).mean().item() * 180.0 / np.pi)
        except Exception as exc:
            max_delta, aug_cos, chart_noise_angle = f"err:{type(exc).__name__}", None, None
        # rank / isotropy of the full embedding set (collapse check)
        try:
            Zc = Z - Z.mean(axis=0, keepdims=True)
            sv = np.linalg.svd(Zc, compute_uv=False)
            var_share = (sv ** 2) / max(float((sv ** 2).sum()), 1e-12)
            n_ = min(400, len(Z))
            sub = torch.nn.functional.normalize(torch.from_numpy(Z[:n_]), dim=-1)
            sim = (sub @ sub.t()).numpy()
            off = sim[~np.eye(n_, dtype=bool)]
            pair_cos = float(off.mean())
            rank_ratio = float(var_share[:10].sum())
            dead_dim = float((np.std(Z, axis=0) < 1e-6).mean())
        except Exception as exc:
            pair_cos, rank_ratio, dead_dim = None, f"err:{type(exc).__name__}", None
        rep["internals"] = {
            "eval_determinism_max_delta": max_delta,
            "augment_invariance_cos": aug_cos,
            "chart_noise_angle_deg": round(chart_noise_angle, 4) if isinstance(chart_noise_angle, float) else chart_noise_angle,
            "embedding_pairwise_cos_mean": round(pair_cos, 5) if isinstance(pair_cos, float) else pair_cos,
            "embedding_top10_var_share": rank_ratio,
            "embedding_dead_dim_frac": round(dead_dim, 5) if isinstance(dead_dim, float) else dead_dim,
        }
        # ---- "vision alive" gate: how much rank the CHART stream alone spans ----
        # A dead vision encoder outputs an almost-constant embedding regardless of
        # the chart (measured below: chart_only top10-SVD ~0.9995 while numeric-only
        # keeps ~10 effective dims). This is THE structural driver of low embedding
        # variation + numeric dominance, and is not evident from SSL loss metrics.
        if tag != "random_init":
            try:
                dsv = ds_list[symbols[0]]
                stride = max(1, int(spec.chart_window))
                ridx = np.arange(0, min(len(dsv) - 1, stride * 128), stride)
                Zf: list[np.ndarray] = []
                Zn: list[np.ndarray] = []
                Zc: list[np.ndarray] = []
                with torch.no_grad():
                    for s in ridx:
                        b = multimodal_collate([dsv[int(s)]])
                        c = b.chart.to(device)
                        n = b.numeric.to(device)
                        t = b.context.to(device)
                        i = b.instrument_id.to(device)
                        Zf.append(model.encode(c, n, t, instrument_id=i).cpu().numpy()[0])
                        Zc.append(model.encode(c * 0.0, n, t, instrument_id=i).cpu().numpy()[0])
                        Zn.append(model.encode(c, n * 0.0, t, instrument_id=i).cpu().numpy()[0])
                if len(Zf) >= 8:
                    f10, fed = _rank_metrics(np.stack(Zf))
                    c10, ced = _rank_metrics(np.stack(Zc))
                    n10, ned = _rank_metrics(np.stack(Zn))
                    rep["vision_alive"] = {
                        "full_top10_svd": f10, "full_eff_dim": fed,
                        "chart_only_top10_svd": c10, "chart_only_eff_dim": ced,
                        "numeric_only_top10_svd": n10, "numeric_only_eff_dim": ned,
                    }
            except Exception as exc:
                rep["vision_alive"] = f"err:{type(exc).__name__}:{exc}"
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

    # gradient-balance probe (vision/numeric starve diagnostic) on trained model
    try:
        report["gradient_balance_trained"] = _gradient_balance(
            tr, ds_list[symbols[0]], device, batches=args.grad_batches,
        )
    except Exception as exc:
        report["gradient_balance_trained"] = f"err:{type(exc).__name__}:{exc}"

    # random-init twin (same architecture as the checkpoint, no weights)
    rmodel = _policy_from_config(eff_cfg).to(device)
    rmodel.eval()
    Zr, ir, rep_rnd = battery(rmodel, "random_init", ds_list)
    report["behaviour_random"] = rep_rnd

    # instrument separation (trained + random)
    from sklearn.metrics import silhouette_score
    try:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Zb), size=min(4000, len(Zb)), replace=False)
        Zi = torch.nn.functional.normalize(torch.from_numpy(Zb[idx]), dim=-1).numpy()
        report["instrument_separation_silhouette"] = round(float(silhouette_score(Zi, ib[idx], sample_size=3000, random_state=0)), 4)
    except Exception as e:
        report["instrument_separation_silhouette"] = f"err:{e}"
    try:
        rng = np.random.default_rng(1)
        idxr = rng.choice(len(Zr), size=min(4000, len(Zr)), replace=False)
        Zir = torch.nn.functional.normalize(torch.from_numpy(Zr[idxr]), dim=-1).numpy()
        report["instrument_separation_silhouette_random"] = round(
            float(silhouette_score(Zir, ir[idxr], sample_size=3000, random_state=0)), 4)
    except Exception as e:
        report["instrument_separation_silhouette_random"] = f"err:{e}"

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