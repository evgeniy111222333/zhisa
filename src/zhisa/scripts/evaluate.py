"""Evaluate a model: print metrics, run a backtest, dump a JSON report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from zhisa.backtest.engine import run_backtest
from zhisa.backtest.reports import print_metrics
from zhisa.data.synthetic import generate_market  # kept for test monkeypatch compatibility
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.utils.seeding import set_seed


def _checkpoint_policy_config(ckpt: dict) -> dict:
    """Return the saved PolicyConfig dict from a checkpoint, if present."""
    for key in ("model_config", "policy_config", "config"):
        cfg = ckpt.get(key)
        if isinstance(cfg, dict) and "window" in cfg and "in_numeric_features" in cfg:
            return cfg
    return {}


def _model_policy(model, device: str = "cpu"):
    model.eval()
    model.to(device)

    def _p(obs):
        with torch.no_grad():
            chart = torch.from_numpy(obs["chart"]).unsqueeze(0).to(device)
            num = torch.from_numpy(obs["numeric"]).unsqueeze(0).to(device)
            ctx = torch.from_numpy(obs["context"]).unsqueeze(0).to(device)
            out = model(chart=chart, numeric=num, context=ctx)
            return int(out["policy_logits"].argmax(dim=-1).item())

    return _p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--bars", type=int, default=4000)
    parser.add_argument("--out", type=str, default="artifacts/eval/report.json")
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    set_seed(0)
    df = load_market_dataframe(args, seed=0, default_bars=args.bars)
    policy = None
    env_cfg = EnvConfig()
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = _checkpoint_policy_config(ckpt)
        model = build_default_policy(
            in_numeric_features=int(cfg.get("in_numeric_features", 32)),
            in_context_features=int(cfg.get("in_context_features", 10)),
            window=int(cfg.get("window", 32)),
            image_size=int(cfg.get("image_size", EnvConfig.image_size)),
        )
        model.load_state_dict(ckpt["model"])
        policy = _model_policy(model)
        env_cfg.window = int(cfg.get("window", env_cfg.window))
        env_cfg.image_size = int(cfg.get("image_size", env_cfg.image_size))
    if policy is None:
        rng = np.random.default_rng(0)
        def policy(_obs):
            return int(rng.integers(0, 9))
    result = run_backtest(df, policy, cfg=env_cfg)
    print_metrics(result.metrics, title="evaluation")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result.metrics.to_dict(), f, indent=2)
    print(f"Report saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
