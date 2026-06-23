"""Validate a serious S4 run without starting training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from zhisa.env.trading_env import TradingEnv
from zhisa.scripts._rl_training import (
    build_env_config,
    build_policy_from_checkpoint,
    load_prepared_markets,
    load_trading_checkpoint,
)
from zhisa.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight a serious S4 run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg = load_config(config_path)
    payload = load_trading_checkpoint(checkpoint_path)
    model = build_policy_from_checkpoint(payload)
    episode_steps = int(cfg.get("max_steps_per_episode", 512))
    minimum = model.cfg.window + episode_steps + 2
    train, train_meta = load_prepared_markets(
        args.prepared_root, args.train_split, minimum_bars=minimum,
    )
    val, val_meta = load_prepared_markets(
        args.prepared_root, args.val_split, minimum_bars=minimum,
    )
    if train_meta["manifest_checksum"] != val_meta["manifest_checksum"]:
        raise ValueError("train/validation manifest mismatch")
    if max(frame.index.max() for frame in train) >= min(frame.index.min() for frame in val):
        raise ValueError("train and validation time ranges overlap")

    env_cfg = build_env_config(cfg, model)
    env_cfg.random_start = True
    env_cfg.episode_length = episode_steps
    probe = TradingEnv(train[0].iloc[:max(minimum, 1024)], cfg=env_cfg)
    observed = (probe.obs_numeric_dim, probe.obs_context_dim)
    expected = (model.cfg.in_numeric_features, model.cfg.in_context_features)
    if observed != expected:
        raise ValueError(f"observation mismatch: data={observed}, model={expected}")

    meta = payload.get("checkpoint_meta") or {}
    report = {
        "status": "ready",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_stage": meta.get("stage"),
        "manifest_checksum": train_meta["manifest_checksum"],
        "timeframe": train_meta["timeframe"],
        "train_markets": len(train),
        "train_rows": train_meta["rows"],
        "validation_markets": len(val),
        "validation_rows": val_meta["rows"],
        "train_end": str(max(frame.index.max() for frame in train)),
        "validation_start": str(min(frame.index.min() for frame in val)),
        "observation_dims": observed,
        "actions": model.cfg.n_actions,
        "episode_steps": episode_steps,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
