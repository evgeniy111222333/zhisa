"""Targeted probe: numeric-input perturbation sensitivity on REAL checkpoints.

The forensics metric `numeric_perturb_1pct.delta_cos` stores the mean cosine
similarity *after* a x1.01 multiplicative perturbation (1.0 = direction
unchanged). This probe deconvolves that metric and measures, on real data:

  A. multiplicative scale x1.01  (replicates forensics; cos / angle / L2)
  B. additive per-channel noise  eps ~ N(0, (0.01 * col_std)^2)
  C. additive token-wise noise   (independent per position, same scale)
  D. worst-channel ranking       (which numeric column moves the embedding most)
  E. jitter-averaging mitigation (mean of K noisy encodes vs clean)
  F. random-init twin control    (same arch, untrained)

Usage::

    python -m zhisa.scripts.probe_numeric_sensitivity \\
        --best /data/out/phase1_heavy_best.pt \\
        --warm /data/out/phaseA_v2_warm.pt  # optional \\
        --prepared-root /data/datasets/s1_1h_12m_v2 \\
        --symbols BTC_USDT,TRX_USDT --out /data/out/numeric_probe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.scripts.forensics_s1_checkpoint import _policy_from_config


def _load_policy(ckpt: str, device: torch.device):
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = payload.get("model_config") or {}
    model = _policy_from_config(mc)
    sd = payload.get("model", payload.get("state_dict"))
    if "model" not in payload and sd is None:
        sd = {k: v for k, v in payload.items() if hasattr(torch.Tensor, k) or isinstance(v, dict) and "weight" in str(v)}
    if sd is None:
        raise RuntimeError(f"cannot find state dict in {ckpt}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, mc


def _batch(ds, idx, device):
    b = multimodal_collate([ds[int(i)] for i in idx])
    return (
        b.chart.to(device),
        b.numeric.to(device),
        b.context.to(device),
        b.instrument_id.to(device),
    )


def _encode(model, chart, num, ctx, inst):
    with torch.no_grad():
        return model.encode(chart, num, ctx, instrument_id=inst)


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    cos = F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
    return float(torch.acos(cos).mean().item() * 180.0 / np.pi)


def _stats(z0, z1) -> dict:
    cos = float(F.cosine_similarity(z0, z1, dim=-1).mean().item())
    ang = _angle_deg(z0, z1)
    l2 = float((z0 - z1).norm(dim=-1).mean().item())
    rel = float(((z0 - z1).norm(dim=-1) / (z0.norm(dim=-1) + 1e-9)).mean().item())
    return {"cos": round(cos, 5), "angle_deg": round(ang, 3), "l2": round(l2, 5), "rel_l2": round(rel, 5)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best", required=True)
    ap.add_argument("--warm", default=None)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--symbols", default="BTC_USDT,TRX_USDT")
    ap.add_argument("--out", default="/data/out/numeric_probe")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128, horizons=(4, 16, 64))
    syms = args.symbols.split(",")
    ds_list, stds = {}, {}
    frames = {}
    for sym in syms:
        df = pd.read_parquet(Path(args.prepared_root) / "symbols" / f"{sym}.parquet").sort_index()
        frames[sym] = df
        ds_list[sym] = MarketDataset(df, spec=spec, cache_charts=False, compute_targets=False, instrument_id=syms.index(sym))

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(ds_list[syms[0]]), size=min(args.samples, len(ds_list[syms[0]])), replace=False))

    report: dict = {}
    ckpts = {"best": args.best}
    if args.warm:
        ckpts["warm_v2"] = args.warm

    for tag, ckpt in ckpts.items():
        model, mc = _load_policy(ckpt, device)
        m_row: dict = {"model_config": {k: mc.get(k) for k in ("embed_dim", "vision_mode", "numeric_layers", "n_instruments")}}
        # random-init twin (same arch)
        twin = _policy_from_config(dict(mc))
        twin.to(device).eval()

        all_cos_clean, all_ang_clean = [], []
        all_cos_rand, all_ang_rand = [], []
        channels: dict[str, list] = {}
        n_ch = None
        for sym in syms:
            ds = ds_list[sym]
            chart, num, ctx, inst = _batch(ds, idx, device)
            n_ch = num.shape[-1]
            z0 = _encode(model, chart, num, ctx, inst)
            z0r = _encode(twin, chart, num, ctx, inst)

            # A. multiplicative x1.01 (forensics replication)
            z1 = _encode(model, chart, num * 1.01, ctx, inst)
            sA = {"mult_1pct": _stats(z0, z1)}
            z1r = _encode(twin, chart, num * 1.01, ctx, inst)
            sA["mult_1pct_random_twin"] = _stats(z0r, z1r)

            # per-channel column std (on this batch, normalized domain)
            col_std = num.std(dim=(0, 1), unbiased=True)
            noise_scale = (0.01 * col_std).to(device)

            # B. additive per-channel (same noise pattern per sample-column)
            eps_b = torch.randn_like(num) * noise_scale
            zb = _encode(model, chart, num + eps_b, ctx, inst)
            sB = {"add_channel_1pct": _stats(z0, zb)}
            zbr = _encode(twin, chart, num + eps_b, ctx, inst)
            sB["add_channel_1pct_random_twin"] = _stats(z0r, zbr)

            # C. token-wise stronger (independent per position)
            zb_t = _encode(model, chart, num + torch.randn_like(num) * noise_scale, ctx, inst)
            sC = {"add_token_wise_1pct": _stats(z0, zb_t)}
            zbt_r = _encode(twin, chart, num + torch.randn_like(num) * noise_scale, ctx, inst)
            sC["add_token_wise_1pct_random_twin"] = _stats(z0r, zbt_r)

            # D. worst-channel: perturb ONE column at a time
            per_ch = []
            for c in range(n_ch):
                eps_c = torch.zeros_like(num)
                eps_c[..., c] = torch.randn(eps_c.shape[:-1], device=device) * noise_scale[c]
                zc = _encode(model, chart, num + eps_c, ctx, inst)
                per_ch.append((float(F.cosine_similarity(z0, zc, dim=-1).mean().item()), c))
            per_ch.sort()
            worst = [(int(c), round(cos, 5)) for cos, c in per_ch[:5]]
            sD = {"worst_5_channels": worst}

            # E. jitter-averaging mitigation on the additive token-wise case
            K = 8
            acc = None
            for _ in range(K):
                zk = _encode(model, chart, num + torch.randn_like(num) * noise_scale, ctx, inst)
                acc = zk if acc is None else acc + zk
            zavg = acc / K
            sE = {"jitter_avg_K8_vs_clean": _stats(z0, zavg),
                  "single_noisy_vs_clean": _stats(z0, zb_t)}

            all_cos_clean.append(sB["add_channel_1pct"]["cos"])
            all_ang_clean.append(sB["add_channel_1pct"]["angle_deg"])
            all_cos_rand.append(sB["add_channel_1pct_random_twin"]["cos"])
            all_ang_rand.append(sB["add_channel_1pct_random_twin"]["angle_deg"])
            for k, v in sA.items():
                channels.setdefault(k, []).append(v)
            for k, v in sB.items():
                channels.setdefault(k, []).append(v)
            for k, v in sC.items():
                channels.setdefault(k, []).append(v)

        m_row.update(
            {
                "channels": n_ch,
                "summary_channel_noise": {
                    "cos_mean": round(float(np.mean(all_cos_clean)), 5),
                    "cos_std": round(float(np.std(all_cos_clean)), 5),
                    "angle_deg_mean": round(float(np.mean(all_ang_clean)), 3),
                    "random_twin_cos_mean": round(float(np.mean(all_cos_rand)), 5),
                    "random_twin_angle_deg_mean": round(float(np.mean(all_ang_rand)), 3),
                },
                "mult_1pct": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in sA.items()},
                "add_channel_1pct": {k: {kk: float(vv) if isinstance(vv, list) else float(vv) for kk, vv in v.items()} for k, v in sB.items()},
                "add_token_wise_1pct": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in sC.items()},
                "worst_5_channels": [{"channel": int(c), "cos_after_1pct": round(cos, 5)} for cos, c in per_ch[:5]],
                "jitter_avg_K8": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in sE.items()},
            }
        )
        report[tag] = m_row
        print(f"[{tag}] channel-noise cos={m_row['summary_channel_noise']['cos_mean']} "
              f"angle={m_row['summary_channel_noise']['angle_deg_mean']}deg "
              f"(random twin cos={m_row['summary_channel_noise']['random_twin_cos_mean']})")

    (out / "numeric_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print("written:", out / "numeric_sensitivity_report.json")


if __name__ == "__main__":
    raise SystemExit(main())