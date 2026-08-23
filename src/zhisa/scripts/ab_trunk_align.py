"""A/B: trunk-level alignment fix vs baseline on REAL data.

Same warm start, same loader order; variants differ only in
``weight_trunk_align`` (and momentum). Measures the lever we care about:
trunk cos(v, n_cls) WITHOUT projections, plus proj-space cos, CPC margin,
and per-block gradient norms after a fixed step budget.

Usage::

    python -m zhisa.scripts.ab_trunk_align --config configs/s1_ssl_1h_12m_heavy_v2.yaml \\
        --start-ckpt /data/out/phaseA_v2_warm.pt --prepared-root /data/datasets/s1_1h_12m_v2 \\
        --charts-cache-dir /data/charts --out /data/out/ab_trunk_align --steps 80
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from zhisa.data.dataset import SampleSpec
from zhisa.data.preparation import load_prepared_split
from zhisa.rendering.spec import RenderSpec
from zhisa.scripts.train_s1 import _market_datasets_from_frame, _ssl_config_from
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer, TemporalPairDataset, temporal_pair_collate
from zhisa.utils.seeding import set_seed


def _policy_from_ckpt(ckpt: str, cfg: dict):
    from zhisa.models.policy import PolicyConfig, PolicyNetwork
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = dict(payload.get("model_config") or {})
    allowed = {f.name for f in PolicyConfig.__dataclass_fields__.values()}
    mc = {k: v for k, v in mc.items() if k in allowed}
    if isinstance(mc.get("vision_channels"), list):
        mc["vision_channels"] = tuple(mc["vision_channels"])
    if isinstance(mc.get("market_horizons"), list):
        mc["market_horizons"] = tuple(mc["market_horizons"])
    if str(cfg.get("model", {}).get("vision_mode", "")) == "columnformer":
        mc.setdefault("vision_mode", "columnformer")
    return PolicyNetwork(PolicyConfig(**mc))


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@torch.no_grad()
def _probe(tr: SSLPretrainer, src, device: torch.device) -> dict:
    rng = np.random.default_rng(3)
    idx = rng.choice(len(src), size=min(64, len(src)), replace=False)
    pairs = [src[int(i)] for i in idx]
    b = temporal_pair_collate(pairs)
    chart, numeric, context = b["chart"], b["numeric"], b["context"]
    inst = b["instrument_id"]
    v = tr.model.plain_vision(chart.to(device))
    n_cls, _ = tr.model.numeric(numeric.to(device))
    v_proj = tr.proj_vision(v)
    n_proj = tr.proj_numeric(n_cls)
    trunk_cos = float(F.cosine_similarity(v.reshape(v.size(0), -1), n_cls, dim=-1).mean().item())
    proj_cos = float(F.cosine_similarity(v_proj, n_proj, dim=-1).mean().item())
    zc = tr.model.encode(chart.to(device), numeric.to(device), context.to(device), instrument_id=inst.to(device))
    zf = tr.teacher.teacher.encode(b["future_chart"].to(device), b["future_numeric"].to(device),
                                   b["future_context"].to(device), instrument_id=inst.to(device))
    pt = F.normalize(tr.temporal_predictor(tr.proj_temporal(zc)), dim=-1)
    tg = F.normalize(tr.target_proj_temporal(zf), dim=-1)
    sim = pt @ tg.t()
    hits = int((sim.argmax(dim=1) == torch.arange(sim.size(0), device=device)).sum().item())
    return {"trunk_cos": trunk_cos, "proj_cos": proj_cos, "top1": hits / sim.size(0),
            "margin": float((sim.diag() - sim.masked_fill(torch.eye(sim.size(0), dtype=torch.bool, device=device), -2.0).max(dim=1).values).mean().item())}


def _grads(tr: SSLPretrainer) -> dict:
    out = {}
    for blk in ("vision", "numeric", "fusion", "context"):
        mod = getattr(tr.model, blk, None)
        if mod is None:
            continue
        g = torch.cat([p.grad.detach().flatten() for p in mod.parameters()
                       if p.grad is not None and p.requires_grad], dim=0)
        out[blk] = float(g.norm().item())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--start-ckpt", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--charts-cache-dir", default=None)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--out", default="/data/out/ab_trunk_align")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = load_config(Path(args.config))
    set_seed(int(cfg.get("seed", 0)))
    chart_window = int(cfg.get("chart_window", 128))
    image_size = int(cfg.get("image_size", 128))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window, image_size=image_size)
    from zhisa.features.normalization import NormalizationSpec

    prepared_root = Path(args.prepared_root)
    manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    timeframe = str(manifest["timeframe"])
    train_frame = load_prepared_split(prepared_root, "train")
    symbols = sorted(train_frame["symbol"].unique())
    if args.symbols:
        symbols = [s for s in args.symbols.split(",") if s in set(symbols)]
    datasets = _market_datasets_from_frame(
        train_frame[train_frame["symbol"].isin(symbols)].copy(),
        spec=spec, cache_charts=True, chart_cache_size=-1,
        timeframe=timeframe,
        charts_cache_dir=args.charts_cache_dir,
        render_spec=RenderSpec(size=spec.image_size),
        render_workers=2, render_chunk=5_000, render_engine="cpu",
        normalization=NormalizationSpec(mode="rolling_z", lookback=256),
        instruments=symbols,
    )
    del train_frame
    train_ds = torch.utils.data.ConcatDataset(datasets)
    print(f"datasets: {len(datasets)} segments, len={len(train_ds)}")

    variants = {
        "base": dict(),
        "trunk_align": dict(weight_trunk_align=0.5, trunk_align_momentum=0.99),
    }
    report = {"steps": args.steps, "device": str(device), "variants": {}}
    for name, overrides in variants.items():
        ssl = _ssl_config_from(cfg)
        for k, v in overrides.items():
            setattr(ssl, k, v)
        ssl.device = args.device
        ssl.batch_size = args.batch_size
        ssl.use_cross_modal = True
        ssl.use_temporal_contrast = True
        ssl.use_masked_modeling = False  # isolate the alignment lever
        tr = SSLPretrainer(_policy_from_ckpt(args.start_ckpt, cfg), ssl)
        tr.load(args.start_ckpt)
        tr.model.to(device).train()
        src = tr._pair_source or TemporalPairDataset(train_ds, horizon=int(ssl.temporal_horizon))
        loader = tr._loader(train_ds, shuffle=True, epoch=0)
        curve = []
        tick = time.monotonic()
        for it, b in enumerate(loader):
            tr.step(b)
            if (it + 1) % args.eval_every == 0:
                p = _probe(tr, src, device)
                curve.append({"step": int(tr._step), **p})
                print(f"[{name}] step={tr._step} trunk={p['trunk_cos']:.4f} proj={p['proj_cos']:.4f} "
                      f"top1={p['top1']:.3f} margin={p['margin']:.3f} t={time.monotonic() - tick:.0f}s")
            if tr._step >= args.steps:
                break
        # gradient snapshot after one loss pass on a fresh batch
        b = next(iter(loader))
        losses = tr._loss(b)
        tr.model.zero_grad(set_to_none=True)
        losses["total"].backward()
        grads = _grads(tr)
        tr.model.zero_grad(set_to_none=True)
        report["variants"][name] = {
            "curve": curve,
            "grads_at_end": grads,
            "losses_at_end": {k: round(float(v.item()), 5) for k, v in losses.items()},
        }
        print(f"[{name}] grads={grads}")
    (out / "ab_trunk_align_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("written:", out / "ab_trunk_align_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())