"""Backtest a trained policy (or a random baseline) on synthetic data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from zhisa.backtest.engine import BacktestResult, buy_and_hold_benchmark, run_backtest
from zhisa.backtest.metrics import compute_metrics
from zhisa.backtest.reports import print_metrics, save_report
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.utils.seeding import set_seed


def _random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)

    def _p(_obs):
        return int(rng.integers(0, 9))

    return _p


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
    parser = argparse.ArgumentParser(description="Backtest a policy.")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/backtest")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    df = generate_market(MarketConfig(n_bars=args.bars, seed=args.seed))

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {}) or {}
        model = build_default_policy(
            in_numeric_features=int(cfg.get("in_numeric_features", 32)),
            in_context_features=int(cfg.get("in_context_features", 10)),
            window=int(cfg.get("window", 32)),
            image_size=int(cfg.get("image_size", 32)),
            n_actions=int(cfg.get("n_actions", 9)),
            n_regime_classes=int(cfg.get("n_regime_classes", 4)),
        )
        model.load_state_dict(ckpt["model"])
        policy = _model_policy(model)
    else:
        policy = _random_policy(args.seed)
        print("No checkpoint provided; using random policy for smoke test.")

    result = run_backtest(df, policy, cfg=EnvConfig(seed=args.seed))
    print_metrics(result.metrics, title="policy")
    bh = buy_and_hold_benchmark(df)
    bh_metrics = compute_metrics(bh)
    print_metrics(bh_metrics, title="buy&hold")
    if args.out:
        save_report(result, args.out, name="policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
