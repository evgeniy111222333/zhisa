"""Deep S1 diagnostics on a real checkpoint + real data (analysis only).

Measures, not claims:
  - per-module gradient norms and per-loss component values on real batches;
  - masked-recon error vs mask ratio (does the model learn beyond mean-fill?);
  - temporal CPC margin (pos_cos - max_neg_cos) per instrument;
  - vision<->numeric alignment split per instrument;
  - instrument-embedding pairwise cosine matrix (which pairs collide);
  - numeric channel health (std / zero / NaN fractions);
  - EMA teacher<->student agreement per encoder block;
  - bit-level OHLCV comparison of two prepared roots (v3.1 vs v2) — decides
    whether a chart-store can be reused across roots.

Usage::

    python deep_diagnose_s1.py --ckpt <s1.pt> --prepared-root <root> \
        [--other-root <root2 for bit-compare>] [--symbols BTC_USDT,TRX_USDT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from zhisa.data.cross_asset import symbol_logret
from zhisa.data.dataset import MarketDataset, SampleSpec, multimodal_collate
from zhisa.scripts.forensics_s1_checkpoint import _policy_from_config
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer, TemporalPairDataset, temporal_pair_collate


def _round(x, n=4):
    return float(round(x, n)) if np.isfinite(x) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--other-root", default=None, help="second prepared root for bit-compare")
    ap.add_argument("--symbols", default="BTC_USDT,TRX_USDT")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--charts-cache-dir", default=None,
                    help="compiled chart store dir; when given, datasets are "
                         "served from the store instead of rendering on the fly")
    ap.add_argument("--out", default=None, help="write the JSON report to this file")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    report: dict = {}
    out_json = Path(args.out) if args.out else None

    # ------------------------------------------------------------------
    # 0. Optional bit-level OHLCV comparison of two prepared roots
    # ------------------------------------------------------------------
    if args.other_root:
        a_root, b_root = Path(args.prepared_root), Path(args.other_root)
        diff_sum = {}
        for sym in args.symbols.split(","):
            da = pd.read_parquet(a_root / "symbols" / f"{sym}.parquet").sort_index()
            db = pd.read_parquet(b_root / "symbols" / f"{sym}.parquet").sort_index()
            cols = ["open", "high", "low", "close", "volume"]
            ca, cb = da[cols].to_numpy(dtype=np.float64), db[cols].to_numpy(dtype=np.float64)
            n = min(len(ca), len(cb))
            eq = np.equal(ca[:n], cb[:n])
            row_diff = (~eq.all(axis=1)).sum()
            col_diff = (~eq).sum(axis=0).tolist()
            # where does it differ? suffix vs interior
            first_bad = int(np.argmax(~eq.all(axis=1))) if row_diff else None
            last_bad = int(n - 1 - np.argmax((~eq.all(axis=1))[::-1])) if row_diff else None
            bit_same = bool(np.array_equal(ca[:n], cb[:n]))
            diff_sum[sym] = {
                "rows_a": len(da), "rows_b": len(db), "prefix_rows_equal": int(n),
                "bit_identical_prefix": bit_same,
                "row_diffs": int(row_diff), "col_diffs": col_diff,
                "first_bad_row": first_bad, "last_bad_row": last_bad,
                "suffix_only": bool(row_diff and first_bad is not None and n - 1 - last_bad == 0),
            }
        report["bit_compare"] = diff_sum
        print("bit-compare:", json.dumps(diff_sum))

    # ------------------------------------------------------------------
    # Build dataset + model
    # ------------------------------------------------------------------
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = payload.get("model_config") or {}
    model = _policy_from_config(mc)
    tr = SSLPretrainer(
        model,
        SSLConfig(device="cpu", batch_size=8,
                  projection_dim=int((payload.get("ssl_config") or {}).get("projection_dim", 128)),
                  hidden_dim=int((payload.get("ssl_config") or {}).get("hidden_dim", 256)),
                  use_ema_teacher=True, use_masked_modeling=True,
                  use_temporal_contrast=True, use_cross_modal=True),
    )
    tr.load(args.ckpt)
    tr.model.to(device).eval()

    spec = SampleSpec(chart_window=128, feature_window=128, image_size=128, horizons=(4, 16, 64))
    syms = args.symbols.split(",")
    instr_of = {s: i for i, s in enumerate(syms)}
    ds_list, frames = {}, {}
    for sym in syms:
        df = pd.read_parquet(Path(args.prepared_root) / "symbols" / f"{sym}.parquet").sort_index()
        frames[sym] = df
        if args.charts_cache_dir:
            from zhisa.data.chart_store import CompiledChartStore
            from zhisa.data.render_job import materialize_parallel
            from zhisa.rendering.spec import RenderSpec as RS
            store, _ = materialize_parallel(
                df, window=spec.chart_window, spec=RS(size=spec.image_size),
                n=len(df) - spec.chart_window - max(spec.horizons, default=0) - 1,
                out_root=args.charts_cache_dir, workers=2, engine="cpu",
            )
            ds_list[sym] = MarketDataset(df, spec=spec, cache_charts=True,
                                         chart_source=store,
                                         compute_targets=False, instrument_id=instr_of[sym])
        else:
            ds_list[sym] = MarketDataset(df, spec=spec, cache_charts=False,
                                         compute_targets=False, instrument_id=instr_of[sym])

    # ------------------------------------------------------------------
    # 1. Numeric channel health (from real feature matrix)
    # ------------------------------------------------------------------
    feats = ds_list[syms[0]]._features_df
    ch = {
        "n_channels": int(feats.shape[1]),
        "channels": {
            c: {"std": _round(float(feats[c].std())), "zero_frac": _round(float((feats[c] == 0).mean())),
                "nan_frac": _round(float(feats[c].isna().mean()))}
            for c in feats.columns
        },
    }
    zeroish = [c for c, v in ch["channels"].items() if (v["zero_frac"] or 0) > 0.5]
    ch["near_zero_channels"] = zeroish
    report["channel_health"] = ch
    print("channels:", ch["n_channels"], "zeroish:", zeroish or [])

    # ------------------------------------------------------------------
    # 2. Loss components + per-module gradient norms on real batches
    # ------------------------------------------------------------------
    rng = np.random.default_rng(11)
    comp = {"temporal": [], "alignment": [], "masked": []}
    grads = {}
    n_batches = 4
    for bi in range(n_batches):
        idx = rng.choice(len(ds_list[syms[0]]), size=16, replace=False)
        b = multimodal_collate([ds_list[syms[0]][int(i)] for i in idx])
        bd = tr._to_device(b)
        losses = tr._loss(bd)  # eval mode: train=False -> canonical loss
        for k in comp:
            comp[k].append(float(losses[k].item()))
        # gradient norms per block (student, one batch backprop)
        tr.model.train()
        losses["total"].backward(retain_graph=True)
        blocks = {"vision": tr.model.vision, "numeric": tr.model.numeric,
                  "fusion": tr.model.fusion, "context": tr.model.context,
                  "proj_temporal": tr.proj_temporal, "proj_vision": tr.proj_vision,
                  "proj_numeric": tr.proj_numeric, "reconstructor": tr.reconstructor}
        for name, mod in blocks.items():
            gn = torch.cat([p.grad.detach().flatten() for p in mod.parameters()
                            if p.grad is not None and p.requires_grad], dim=0)
            grads.setdefault(name, []).append(float(gn.norm().item()))
        tr.model.zero_grad(set_to_none=True)
        tr.model.eval()
    report["loss_components"] = {k: {"mean": _round(float(np.mean(v)), 5), "std": _round(float(np.std(v)), 5)} for k, v in comp.items()}
    report["grad_norms"] = {k: {"mean": _round(float(np.mean(v))), "std": _round(float(np.std(v)))} for k, v in grads.items()}
    print("loss comps:", report["loss_components"])
    print("grad norms:", report["grad_norms"])

    # ------------------------------------------------------------------
    # 3. Masked-recon error vs mask ratio
    # ------------------------------------------------------------------
    b = multimodal_collate([ds_list[syms[0]][int(i)] for i in rng.choice(len(ds_list[syms[0]]), size=8, replace=False)])
    x = b.numeric.to(device)
    from zhisa.training.s1_ssl import _MaskedReconstructor
    ratios, errs = [], []
    for r in (0.1, 0.3, 0.5, 0.7, 0.9):
        mask = torch.bernoulli(torch.full((x.size(0), tr.model.numeric.n_patches), 1.0 - r, device=device))
        mask[:, 0] = 1.0
        patch = tr.model.numeric.cfg.patch_size
        n_patches = tr.model.numeric.n_patches
        patches = x.view(x.size(0), n_patches, patch, -1).reshape(x.size(0), n_patches, -1)
        mx = (patches * mask.unsqueeze(-1)).view_as(x)
        with torch.no_grad():
            _, tok = tr.model.numeric(mx)
            pred = tr.reconstructor(tok)
            pred_p = pred[:, :n_patches] if getattr(tr.model.numeric.cfg, "causal", False) else pred[:, 1:]
        errs.append(float((pred_p[mask < 0.5] - patches[mask < 0.5]).pow(2).mean().item()))
        ratios.append(r)
    report["masked_err_vs_ratio"] = {"ratios": ratios, "errs": errs}
    print("masked err curve:", report["masked_err_vs_ratio"])

    # ------------------------------------------------------------------
    # 4. Temporal CPC margin + alignment per instrument (real pairs)
    # ------------------------------------------------------------------
    srcs = {s: TemporalPairDataset(ds_list[s], horizon=4) for s in syms}
    per_sym = {}
    for s, src in srcs.items():
        rng2 = np.random.default_rng(5)
        idx = np.sort(rng2.choice(len(src), size=min(80, len(src)), replace=False))
        pos_cos, margins, aligns = [], [], []
        for st in range(0, len(idx), 32):
            items = [src[int(i)] for i in idx[st:st + 32]]
            b = temporal_pair_collate(items)
            with torch.no_grad():
                zc = tr.model.encode(b["chart"].to(device), b["numeric"].to(device),
                                     b["context"].to(device), instrument_id=b["instrument_id"].to(device))
                zf = tr.teacher.teacher.encode(b["future_chart"].to(device), b["future_numeric"].to(device),
                                               b["future_context"].to(device), instrument_id=b["instrument_id"].to(device))
                pt = F.normalize(tr.temporal_predictor(tr.proj_temporal(zc)), dim=-1)
                tg = F.normalize(tr.target_proj_temporal(zf), dim=-1)
                sim = pt @ tg.t()
                d = torch.arange(sim.size(0))
                pos = sim.diag()
                others = sim.masked_scatter(torch.eye(sim.size(0), dtype=torch.bool, device=sim.device), torch.full_like(sim, -2.0))
                margin = (pos - others.max(dim=1).values)
                pos_cos.extend(pos.tolist()); margins.extend(margin.tolist())
                v = tr.model.plain_vision(b["chart"].to(device))
                n_, _ = tr.model.numeric(b["numeric"].to(device))
                aligns.append(F.cosine_similarity(tr.proj_vision(v), tr.proj_numeric(n_), dim=-1).mean().item())
        per_sym[s] = {
            "pos_cos_mean": _round(float(np.mean(pos_cos)), 4),
            "pos_cos_std": _round(float(np.std(pos_cos)), 4),
            "margin_mean": _round(float(np.mean(margins)), 4),
            "margin_median": _round(float(np.median(margins)), 4),
            "alignment": _round(float(np.mean(aligns)), 4),
        }
    report["per_instrument"] = per_sym
    print("per instrument:", json.dumps(per_sym))

    # ------------------------------------------------------------------
    # 5. Instrument embedding pairwise collisions
    # ------------------------------------------------------------------
    emb = tr.model.context.instrument_emb.weight.detach().numpy()
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    C = e @ e.T
    iu = np.triu_indices(len(C), 1)
    pairs = sorted(zip(C[iu], iu[0].tolist(), iu[1].tolist()), reverse=True)[:6]
    report["instrument_colllisions_top"] = [{"cos": _round(float(c), 4), "i": int(i), "j": int(j)} for c, i, j in pairs]
    print("top collisions:", report["instrument_colllisions_top"])

    # ------------------------------------------------------------------
    # 6. EMA teacher<->student per block
    # ------------------------------------------------------------------
    if tr.teacher is not None:
        ema_blocks = {}
        for blk in ("vision", "numeric", "fusion", "context"):
            tp = dict(tr.teacher.teacher.state_dict())
            sp = dict(tr.model.state_dict())
            v1_ = torch.cat([tp[k].flatten() for k in tp if k.startswith(blk + ".")])
            v2_ = torch.cat([sp[k].flatten() for k in sp if k.startswith(blk + ".")])
            if v1_.numel() and v1_.numel() == v2_.numel():
                ema_blocks[blk] = _round(float(F.cosine_similarity(v1_, v2_, dim=0).item()), 4)
        report["ema_teacher_student_per_block"] = ema_blocks
        print("ema per block:", ema_blocks)

    print(json.dumps(report, indent=1, default=str))
    if out_json is not None:
        out_json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print("report written:", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())