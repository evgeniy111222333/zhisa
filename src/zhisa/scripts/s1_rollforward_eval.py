"""S1 roll-forward OOS evaluation (anti-'val is an echo' gate).

The SSL validation split shares the same rendering/feature distribution as
training, so an improving val can mask memorization. This script evaluates the
checkpoint on an explicit, LATER time slice that was never touched by the
trainer — the honest overfitting/behaviour gate BEFORE graduating to S2.

    python -m zhisa.scripts.s1_rollforward_eval \
        --checkpoint /data/out/phase2_heavy_v2_5_last.pt \
        --prepared-root /data/datasets/s1_1h_12m_v2 \
        --symbols BTC_USDT,ETH_USDT \
        --start 2026-01-01 --end 2026-08-23 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.scripts.forensics_s1_checkpoint import _policy_matching_checkpoint
from zhisa.training.s1_ssl import SSLConfig, SSLPretrainer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared-root", required=True)
    ap.add_argument("--symbols", default="BTC_USDT,ETH_USDT")
    ap.add_argument("--start", required=True, help="inclusive UTC start, e.g. 2026-06-01")
    ap.add_argument("--end", required=True, help="inclusive UTC end, e.g. 2026-08-23")
    ap.add_argument("--out", default="artifacts/s1_oos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chart-window", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--val-max-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, eff_cfg = _policy_matching_checkpoint(payload.get("model_config") or {},
                                                 payload.get("model") or {})
    ssl_cfg_src = payload.get("ssl_config") or {}
    tr = SSLPretrainer(
        model,
        SSLConfig(device=str(device), batch_size=64,
                  projection_dim=int(ssl_cfg_src.get("projection_dim", 128)),
                  hidden_dim=int(ssl_cfg_src.get("hidden_dim", 256)),
                  use_ema_teacher=True, use_masked_modeling=True,
                  use_temporal_contrast=True, use_cross_modal=True,
                  weight_trunk_align=float(ssl_cfg_src.get("weight_trunk_align", 0.0)),
                  trunk_align_momentum=float(ssl_cfg_src.get("trunk_align_momentum", 0.0)),
                  instrument_contrast_w=float(ssl_cfg_src.get("instrument_contrast_w", 0.0)),
                  mask_ratio=float(ssl_cfg_src.get("mask_ratio", 0.5)),
                  temperature=float(ssl_cfg_src.get("temperature", 0.1)),
                  temporal_horizon=int(ssl_cfg_src.get("temporal_horizon", 4) or 4),
                  val_max_batches=int(args.val_max_batches)),
    )
    tr.load(args.checkpoint)
    print("loaded %s (OOS roll-forward eval)" % args.checkpoint)

    spec = SampleSpec(chart_window=int(args.chart_window), image_size=int(args.image_size))
    oos_ds = None
    rows = 0
    for i, sym in enumerate(args.symbols.split(",")):
        df = pd.read_parquet(Path(args.prepared_root) / "symbols" / f"{sym}.parquet").sort_index()
        # tz-safe windowing: align the requested UTC bounds to the frame's tz.
        lo = pd.Timestamp(args.start)
        hi = pd.Timestamp(args.end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if df.index.tz is not None:
            lo = lo.tz_localize("UTC").tz_convert(df.index.tz)
            hi = hi.tz_localize("UTC").tz_convert(df.index.tz)
        else:
            lo = lo.tz_localize(None)
            hi = hi.tz_localize(None)
        mask = (df.index >= lo) & (df.index <= hi)
        if not mask.any():
            print(f"WARNING: no bars for {sym} in [{args.start}, {args.end}]")
            continue
        slice_df = df[mask]
        ds = MarketDataset(slice_df, spec=spec, compute_targets=False,
                           cache_charts=False, instrument_id=i)
        oos_ds = ds if oos_ds is None else torch.utils.data.ConcatDataset([oos_ds, ds])
        rows += len(ds)

    if oos_ds is None or rows < 4:
        print("NO OOS data in the requested slice -> gate cannot be evaluated")
        return 2

    metrics = tr.evaluate(oos_ds)
    metrics["n_samples"] = int(rows)
    metrics["window_dates"] = {"start": args.start, "end": args.end}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oos_report.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())