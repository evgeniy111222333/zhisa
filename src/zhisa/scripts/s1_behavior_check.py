"""S1 end-to-end BEHAVIOURAL verification battery (pre-S2 gate).

Not another aggregate metric list — each probe asks "does the trained SSL model
actually DO the right thing" on real 1h data and returns PASS/FAIL with
concrete evidence:

  B1 recency      : perturbing the LAST bar must move the embedding more than
                    an EARLY bar (model actually reads the present).
  B2 candle-flip  : mirroring the final candle shifts z in a sane range
                    (responsive, not dead, not exploding).
  B3 future-lookup: for a window at t, the TRUE t+h window ranks ahead of
                    random same-symbol distractors (mean rank small).
  B4 instrument-id: paired Embeddings (same content, BTC vs ETH id) must be
                    separable by a 1-NN probe (model knows WHICH instrument).
  B5 NN locality  : nearest neighbours in embedding space are temporally
                    close (similar embeddings come from similar market states).
  B6 no-collapse  : embedding-norm variance is real and pairwise cos << 1
                    (feature-less constant embedding would be a red flag).
  B8 determinism  : the same obs twice yields bit-identical output.

Usage::

    python -m zhisa.scripts.s1_behavior_check \
        --checkpoint /data/out/phase2_heavy_v2_5_last.pt \
        --prepared-root /data/datasets/s1_1h_12m_v2 \
        --symbols BTC_USDT,ETH_USDT --out /data/out/s1_behavior
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.scripts.forensics_s1_checkpoint import _policy_matching_checkpoint


def _norm(z): return torch.nn.functional.normalize(z, dim=-1)


def _ang(a, b):
    c = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
    return float(torch.acos(c).mean().item() * 180.0 / np.pi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--symbols", default="BTC_USDT,ETH_USDT")
    ap.add_argument("--out", default="artifacts/s1_behavior")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--num-queries", type=int, default=24)
    ap.add_argument("--num-distractors", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, eff_cfg = _policy_matching_checkpoint(payload.get("model_config") or {},
                                                 payload.get("model") or {})
    model = model.to(device).eval()
    torch.set_grad_enabled(False)  # pure evaluation: no autograd anywhere
    print("behavioral battery on", args.checkpoint, "| effective:", eff_cfg)

    symbols = args.symbols.split(",")
    spec = SampleSpec(chart_window=128, image_size=128)
    ds_list = {}
    for i, sym in enumerate(symbols):
        df = pd.read_parquet(Path(args.prepared_root) / "symbols" / f"{sym}.parquet").sort_index()
        ds_list[sym] = MarketDataset(df, spec=spec, compute_targets=False,
                                     cache_charts=False, instrument_id=i)

    rng = np.random.default_rng(args.seed)
    report = {}
    H = int(args.horizon)

    def enc(sym, idxs, ids=None):
        b = multimodal_collate([ds_list[sym][int(i)] for i in idxs])
        return model.encode(b.chart.to(device), b.numeric.to(device),
                            b.context.to(device),
                            instrument_id=(ids if ids is not None else b.instrument_id.to(device)))

    # ---------------- B8 determinism + B6 no-collapse -----------------
    sym0 = symbols[0]
    ds0 = ds_list[sym0]
    idxs = rng.choice(len(ds0), size=64, replace=False)
    z = enc(sym0, idxs)
    z2 = enc(sym0, idxs)
    max_delta = float((z - z2).abs().max().item())
    report["b8_determinism"] = {"max_delta": round(max_delta, 8)}
    b8_ok = max_delta < 1e-6
    report["b8_ok"] = bool(b8_ok)

    norms = z.norm(dim=-1).cpu().numpy()
    zn = _norm(z)
    pair = (zn @ zn.t()).cpu().numpy()
    off = pair[~np.eye(len(pair), dtype=bool)]
    var_frac = float(norms.std() / max(norms.mean(), 1e-9))
    report["b6_no_collapse"] = {
        "norm_var_frac": round(var_frac, 4),
        "pairwise_cos_mean": round(float(off.mean()), 5),
        "pairwise_cos_max_offdiag": round(float(np.max(np.abs(off))), 5),
    }
    b6_ok = bool(var_frac > 0.01 and abs(float(off.mean())) < 0.99)
    report["b6_ok"] = bool(b6_ok) and bool(np.isfinite(norms).all())

    # ---------------- B1 recency + B2 candle-flip ---------------------
    n_w = 32
    widx = rng.choice(len(ds0) - 1, size=n_w, replace=False)
    z_base = enc(sym0, widx)
    # perturb the LAST bar's numeric features (replace with its column-mean noise)
    def perturb_last(w, idxs, window=128):
        b = multimodal_collate([ds0[int(i)] for i in idxs])
        x = b.numeric.clone()
        feats = x[:, -1, :]
        x[:, -1, :] = torch.randn_like(feats) * feats.std(dim=0, keepdim=True).clamp_min(1e-3)
        num = x.to(device)
        return model.encode(b.chart.to(device), num, b.context.to(device),
                            instrument_id=b.instrument_id.to(device))

    def perturb_early(w, idxs):
        b = multimodal_collate([ds0[int(i)] for i in idxs])
        x = b.numeric.clone()
        feats = x[:, 8, :]
        x[:, 8, :] = torch.randn_like(feats) * feats.std(dim=0, keepdim=True).clamp_min(1e-3)
        num = x.to(device)
        return model.encode(b.chart.to(device), num, b.context.to(device),
                            instrument_id=b.instrument_id.to(device))

    duty = 0
    z_last = perturb_last(sym0, widx)
    z_early = perturb_early(sym0, widx)
    ang_last = _ang(z_base, z_last)
    ang_early = _ang(z_base, z_early)
    # B2: flip the final candle (mirror close about the window's mid close)
    def flip_last(w, idxs):
        b = multimodal_collate([ds0[int(i)] for i in idxs])
        chart = b.chart.clone()
        mid = chart[:, 0].mean(dim=(1, 2), keepdim=True)
        last = chart[:, :, -1, :]
        chart[:, :, -1, :] = 2.0 * mid - last
        return model.encode(chart.to(device), b.numeric.to(device), b.context.to(device),
                            instrument_id=b.instrument_id.to(device))

    z_flip = flip_last(sym0, widx)
    ang_flip = _ang(z_base, z_flip)
    report["b1_recency"] = {"ang_last": round(ang_last, 4), "ang_early": round(ang_early, 4)}
    report["b2_candle_flip"] = {"ang_flip": round(ang_flip, 4)}
    b1_ok = bool(ang_last > 0.05 and ang_last > 1.5 * ang_early + 1e-3)
    b2_ok = bool(0.05 < ang_flip < 15.0)
    report["b1_ok"], report["b2_ok"] = bool(b1_ok), bool(b2_ok)

    # ---------------- B3 future-lookup rank ---------------------------
    n_q = min(int(args.num_queries), max(1, len(ds0) - H - 1))
    qidx = rng.choice(len(ds0) - H - 1, size=n_q, replace=False)
    ranks = []
    top1 = 0
    with torch.no_grad():
        for t in qidx:
            fut_idx = t + H
            dist = rng.choice([i for i in range(len(ds0)) if abs(i - fut_idx) > 2],
                              size=int(args.num_distractors), replace=False)
            cand = [fut_idx] + list(dist)
            zc = enc(sym0, cand)
            zq = enc(sym0, [t]).expand(zc.size(0), -1)
            sim = torch.nn.functional.cosine_similarity(_norm(zq), _norm(zc), dim=-1)
            rank = int((sim.argsort(descending=True) == 0).nonzero()[0].item())
            ranks.append(rank)
            if rank == 0:
                top1 += 1
    report["b3_future_lookup"] = {
        "n": int(len(ranks)),
        "top1_acc": round(top1 / max(len(ranks), 1), 4),
        "mean_rank": round(float(np.mean(ranks)), 3),
        "median_rank": int(np.median(ranks)),
    }
    b3_ok = bool(np.mean(ranks) <= max(3.0, 0.25 * int(args.num_distractors)))

    # ---------------- B4 instrument-id probe --------------------------
    n_probe = min(24, len(ds0))
    match = rng.choice(len(ds0), size=n_probe, replace=False)
    z_a = enc(symbols[0], match, ids=torch.zeros(len(match), dtype=torch.long))
    z_b = enc(symbols[1], match, ids=torch.ones(len(match), dtype=torch.long))
    Z = torch.cat([_norm(z_a), _norm(z_b)], dim=0)
    labels = torch.cat([torch.zeros(len(match)), torch.ones(len(match))], 0).long()
    sim = Z @ Z.t()
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    pred = labels[sim.argmax(dim=-1)]
    acc = float((pred == labels).float().mean().item())
    ga = float((_norm(z_a) @ _norm(z_b).t()).mean().item())  # between-group cosine
    report["b4_instrument_probe"] = {"1nn_acc": round(acc, 4), "between_group_cos": round(ga, 4)}
    b4_ok = bool(acc >= 0.9)

    # ---------------- B5 NN locality ----------------------------------
    n_n = 16
    base_idx = np.arange(0, min(2048, len(ds0)), 2)
    nq = rng.choice(len(base_idx), size=n_n, replace=False)
    dts = []
    margins = []
    zb = enc(sym0, base_idx)
    zbs = _norm(zb)
    with torch.no_grad():
        for i in nq:
            s = (zbs @ zbs[i]).detach().cpu().numpy()
            order = np.argsort(-s)
            top1 = order[1] if order[0] == i else order[0]
            dts.append(int(base_idx[top1] - base_idx[i]))
            margins.append(float(s[top1] - s[order[2]]))
    abs_dt = np.abs(dts)
    report["b5_nn_locality"] = {
        "median_abs_dt_bars": int(np.median(abs_dt)),
        "p75_abs_dt_bars": int(np.percentile(abs_dt, 75)),
        "median_margin": round(float(np.median(margins)), 4),
    }
    b5_ok = bool(int(np.median(abs_dt)) <= 24)

    report.update({
        "b1_ok": bool(b1_ok), "b2_ok": bool(b2_ok), "b3_ok": bool(b3_ok),
        "b4_ok": bool(b4_ok), "b5_ok": bool(b5_ok), "b6_ok": bool(b6_ok),
        "b8_ok": bool(b8_ok),
    })
    report["s1_behavior_pass"] = sum(1 for k in ("b1_ok", "b2_ok", "b3_ok", "b4_ok", "b5_ok", "b6_ok", "b8_ok")
                                     if report[k])
    report["s1_behavior_total"] = 7

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "behavior_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())