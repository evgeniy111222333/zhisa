"""Warm-start a vision-v2 (ColumnFormer + token fusion) S1 checkpoint from a
vision-*1 (CNN) checkpoint: copies every compatible weight (numeric encoder,
working memory, context encoder, heads, SSL projections) and leaves only the new
vision/fusion branches randomly initialised. This lets v2 be *fine-tuned* from a
finished v1 run instead of trained from scratch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from zhisa.config import load_config
from zhisa.models.policy import PolicyConfig, build_default_policy
from zhisa.training.s1_ssl import _filter_matching_state_dict


def _build_v2_policy(config_yaml: str, in_numeric_features: int, in_context_features: int, n_instruments: int):
    cfg = load_config(config_yaml)
    m = dict(cfg.get("model", {}) or {})
    if isinstance(m.get("vision_channels"), list):
        m["vision_channels"] = tuple(int(x) for x in m["vision_channels"])
    if "n_instruments" not in m or int(m.get("n_instruments", 0)) < 1:
        m["n_instruments"] = int(n_instruments)
    window = int(cfg.get("chart_window", 128) if cfg else 128)
    image_size = int(cfg.get("image_size", 128) if cfg else 128)
    n_regime = int((cfg.get("ssl", {}) or {}).get("n_regime_states", 4) if cfg else 4)
    return build_default_policy(
        in_numeric_features=int(in_numeric_features),
        in_context_features=int(in_context_features),
        window=window, image_size=image_size,
        n_actions=9, n_regime_classes=max(int(n_regime), 1),
        **m,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="v1 (CNN) S1 checkpoint")
    ap.add_argument("--config", required=True, help="v2 YAML config (model block drives architecture)")
    ap.add_argument("--target", required=True, help="output v2 S1 checkpoint")
    ap.add_argument("--in-numeric-features", type=int, default=32)
    ap.add_argument("--in-context-features", type=int, default=10)
    args = ap.parse_args(argv)

    src = torch.load(args.source, map_location="cpu", weights_only=False)
    src_cfg = src.get("model_config") or src.get("config") or {}
    n_instruments = int(src_cfg.get("n_instruments", 12) or 12)
    policy = _build_v2_policy(
        args.config, args.in_numeric_features, args.in_context_features, n_instruments
    )
    policy.train()  # keep all new branches trainable

    filtered = _filter_matching_state_dict(src.get("model", src), policy)
    n_copied = len(filtered)
    n_total = sum(1 for _ in policy.named_parameters())
    policy.load_state_dict(filtered, strict=False)

    cfg_dict = policy.cfg.__dict__.copy()
    if "vision_channels" in cfg_dict and isinstance(cfg_dict["vision_channels"], tuple):
        cfg_dict["vision_channels"] = list(cfg_dict["vision_channels"])
    if "market_horizons" in cfg_dict and isinstance(cfg_dict["market_horizons"], tuple):
        cfg_dict["market_horizons"] = list(cfg_dict["market_horizons"])

    meta = dict(src.get("checkpoint_meta", {}) or {})
    meta.update({
        "stage": "s1_ssl",
        "warm_start_from": str(Path(args.source).resolve()),
        "vision_v2": True,
    })
    payload = {
        "model": policy.state_dict(),
        "config": cfg_dict,
        "model_config": cfg_dict,
        "ssl_config": src.get("ssl_config"),
        "checkpoint_meta": meta,
        "trainer_state": {
            "step": 0, "completed_epochs": 0, "history": [],
            "best_val_total": float("inf"),
        },
    }
    for key in ("proj_temporal", "temporal_predictor", "proj_vision", "proj_numeric",
                "reconstructor", "target_proj_temporal"):
        if key in src:
            payload[key] = src[key]
    if "teacher" in src:
        payload["teacher"] = src["teacher"]

    p = Path(args.target)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)
    print(
        f"warm-started v2 -> {p}\n"
        f"  copied {n_copied}/{n_total} weights (vision+v2-fusion are fresh)\n"
        f"  v2 params: {sum(p2.numel() for p2 in policy.parameters()):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())