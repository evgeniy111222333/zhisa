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
from zhisa.data.render_contract import assert_serving_render
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.scripts._rl_training import build_policy_from_checkpoint
from zhisa.utils.seeding import set_seed


def _model_policy(model, device: str = "cpu"):
    """Serving policy with a REAL rolling memory session (S4 parity).

    Keeps working-memory state across steps; the episode starts cold (zeros)
    exactly like an S4 rollout episode. History continuity is the whole point
    — the legacy closure re-passed zeros on every step.
    """
    model.eval()
    model.to(device)
    from zhisa.models.session import make_stateful_policy, session_step

    return make_stateful_policy(model)


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
        model = build_policy_from_checkpoint(ckpt)
        policy = _model_policy(model)
        env_cfg.window = int(model.cfg.window)
        env_cfg.image_size = int(model.cfg.image_size)
        assert_serving_render(ckpt, env_cfg.image_size)
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
