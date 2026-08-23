"""A/B: Temporal CPC v2 variants (base / bank / bank+hard) on REAL data.

Real prepared S1 data + compiled chart store; each variant starts from the
same warm checkpoint and trains with a fixed deterministic loader order.
Every `--eval-every` steps we run a fixed CPC accuracy probe (random 64
pairs, in-batch negatives only -> variant-comparable) and record step time.

Usage::

    python -m zhisa.scripts.ab_temporal_cpc --config configs/s1_ssl_1h_12m_heavy_v2.yaml \\
        --start-ckpt /data/out/phaseA_v2_warm.pt \
        --prepared-root /data/datasets/s1_1h_12m_v2 --charts-cache-dir /data/charts \\
        --out /data/out/ab_temporal_cpc --steps 200 --batch-size 32 --eval-every 25
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from zhisa.data.dataset import SampleSpec
from zhisa.data.preparation import load_prepared_split
from zhisa.rendering.spec import RenderSpec
from zhisa.scripts.train_s1 import _market_datasets_from_frame, _ssl_config_from
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer, TemporalPairDataset
from zhisa.utils.seeding import set_seed


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--start-ckpt", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--charts-cache-dir", default=None)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--max-bars-per-symbol", type=int, default=None)
    ap.add_argument("--out", default="/data/out/ab_temporal_cpc")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="Fast end-to-end pass: 15 steps x batch 8 per variant "
                         "(pipeline check only, tiny numbers).")
    ap.add_argument("--in-numeric-features", default=None,
                    help="Numeric feature count for the policy: an int, 'auto' "
                         "(derive from the dataset; numeric trunk re-inits, rest "
                         "transfers shape-matched from the checkpoint), or empty "
                         "(use the checkpoint's own config).")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.smoke:
        print("SMOKE MODE: quick end-to-end pipeline check (tiny steps/batch)")
        args.steps, args.batch_size, args.eval_every = 15, 8, 15

    cfg = load_config(Path(args.config))
    set_seed(int(cfg.get("seed", 0)))
    chart_window = int(cfg.get("chart_window", 128))
    image_size = int(cfg.get("image_size", 128))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window, image_size=image_size)
    norm_mode = str((cfg.get("normalize", {}) or {}).get("mode", "rolling_z"))
    norm_lookback = int((cfg.get("normalize", {}) or {}).get("lookback", 256))
    from zhisa.features.normalization import NormalizationSpec

    norm_spec = NormalizationSpec(mode=norm_mode, lookback=norm_lookback)

    prepared_root = Path(args.prepared_root)
    manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    timeframe = str(manifest["timeframe"])
    train_frame = load_prepared_split(prepared_root, "train")
    symbols = sorted(train_frame["symbol"].unique())
    if args.symbols:
        symbols = [s for s in args.symbols.split(",") if s in set(symbols)]
    datasets = _market_datasets_from_frame(
        train_frame[train_frame["symbol"].isin(symbols)].copy(),
        spec=spec,
        cache_charts=True,
        chart_cache_size=-1,
        max_bars_per_symbol=args.max_bars_per_symbol,
        timeframe=timeframe,
        charts_cache_dir=args.charts_cache_dir,
        render_spec=RenderSpec(size=spec.image_size),
        render_workers=2,
        render_chunk=5_000,
        render_engine="cpu",
        normalization=norm_spec,
        instruments=symbols,
    )
    del train_frame
    train_ds = torch.utils.data.ConcatDataset(datasets)
    print(f"datasets: {len(datasets)} segments x {len(symbols)} symbols, len={len(train_ds)}")

    variants = {
        "base": dict(),
        "bank": dict(temporal_bank_size=1024, temporal_bank_warmup=64),
        "hard": dict(temporal_bank_size=1024, temporal_bank_warmup=64, temporal_hard_offsets=(-1, 2)),
    }

    report = {"device": str(device), "steps": args.steps, "variants": {}, "status": "partial"}

    def _dump() -> None:
        (out / "ab_report.json").write_text(json.dumps(report, indent=2, default=str))

    import atexit
    atexit.register(_dump)
    for name, overrides in variants.items():
        ssl = _ssl_config_from(cfg)
        for k, v in overrides.items():
            setattr(ssl, k, v)
        ssl.device = args.device
        ssl.batch_size = args.batch_size
        ssl.use_masked_modeling = False
        ssl.use_cross_modal = False
        n_feat_override = None
        if args.in_numeric_features and str(args.in_numeric_features).lower() != "auto":
            n_feat_override = int(args.in_numeric_features)
        elif str(args.in_numeric_features or "").lower() == "auto":
            n_feat_override = int(datasets[0]._features_df.shape[1])
        tr = SSLPretrainer(_policy_from_ckpt(args.start_ckpt, cfg, in_numeric_features=n_feat_override), ssl)
        tr.load(args.start_ckpt)
        tr.model.to(device).train()

        loader = tr._loader(train_ds, shuffle=True, epoch=0)
        tick = time.monotonic()
        curve = []
        step_times = []
        for it, b in enumerate(loader):
            t0 = time.monotonic()
            step_out = tr.step(b)
            step_times.append(time.monotonic() - t0)
            if (it + 1) % args.eval_every == 0:
                acc, cos = _cpc_probe(tr, device)
                curve.append(
                    {"step": tr._step, "top1": round(float(acc), 4), "pos_cos": round(float(cos), 4),
                     "temporal_loss": round(float(step_out["temporal"]), 4)}
                )
                print(f"[{name}] step={tr._step} top1={curve[-1]['top1']} pos_cos={curve[-1]['pos_cos']} "
                      f"loss={curve[-1]['temporal_loss']} t={time.monotonic() - tick:.0f}s")
            if tr._step >= args.steps:
                break
        report["variants"][name] = {
            "curve": curve,
            "steps_done": int(tr._step),
            "mean_step_s": round(float(np.mean(step_times[10:])), 3),
            "bank_size": int(tr._bank.size(0)) if tr._bank is not None else 0,
        }
        print(f"[{name}] done: steps={tr._step}, mean_step={report['variants'][name]['mean_step_s']}s")

    (out / "ab_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("written:", out / "ab_report.json")
    return 0


def _policy_from_ckpt(ckpt: str, cfg: dict, in_numeric_features=None):
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
    if in_numeric_features is not None:
        mc["in_numeric_features"] = int(in_numeric_features)
    return PolicyNetwork(PolicyConfig(**mc))


@torch.no_grad()
def _cpc_probe(tr: SSLPretrainer, device: torch.device) -> tuple[float, float]:
    """Fixed-format CPC accuracy: random 64 pairs, in-batch negatives only."""
    src = tr._pair_source
    if src is None:
        return 0.0, 0.0
    rng = np.random.default_rng(3)
    idx = rng.choice(len(src), size=min(64, len(src)), replace=False)
    from zhisa.training.s1_ssl import temporal_pair_collate
    b = temporal_pair_collate([src[int(i)] for i in idx])
    chart = b["chart"].to(device)
    numeric = b["numeric"].to(device)
    context = b["context"].to(device)
    inst = b["instrument_id"].to(device)
    zc = tr.model.encode(chart, numeric, context, instrument_id=inst)
    zf = tr.teacher.teacher.encode(b["future_chart"].to(device), b["future_numeric"].to(device),
                                   b["future_context"].to(device), instrument_id=inst)
    p_t = tr.temporal_predictor(tr.proj_temporal(zc))
    tg = tr.target_proj_temporal(zf).detach()
    p_t = torch.nn.functional.normalize(p_t, dim=-1)
    tg = torch.nn.functional.normalize(tg, dim=-1)
    sim = p_t @ tg.t()
    hits = int((sim.argmax(dim=1) == torch.arange(sim.size(0), device=device)).sum().item())
    cos = float(torch.nn.functional.cosine_similarity(p_t, tg, dim=-1).mean().item())
    return hits / sim.size(0), cos


if __name__ == "__main__":
    raise SystemExit(main())