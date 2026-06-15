"""Run a no-money simulated replay on historical market data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from zhisa.backtest.engine import BacktestResult, buy_and_hold_benchmark, run_backtest
from zhisa.backtest.metrics import compute_metrics
from zhisa.backtest.regime_ab import RegimeABConfig, run_regime_ab_backtest
from zhisa.backtest.reports import print_metrics, save_report
from zhisa.env.actions import DiscreteAction
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.scripts._real_data import add_market_data_args, frame_summary, load_market_dataframe
from zhisa.scripts.backtest import (
    TorchModelPolicy,
    _checkpoint_policy_config,
    _checkpoint_policy_metadata,
    _random_policy,
    _warn_if_checkpoint_not_trading_ready,
)
from zhisa.utils.seeding import set_seed


def _action_name(action: int) -> str:
    try:
        return DiscreteAction(int(action)).name
    except ValueError:
        return str(int(action))


def _load_policy(checkpoint: str | None, *, device: str = "cpu", seed: int = 0):
    env_cfg = EnvConfig(seed=seed)
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _warn_if_checkpoint_not_trading_ready(ckpt, checkpoint)
        cfg = _checkpoint_policy_config(ckpt)
        meta = _checkpoint_policy_metadata(ckpt)
        model = build_default_policy(
            in_numeric_features=int(cfg.get("in_numeric_features", 32)),
            in_context_features=int(cfg.get("in_context_features", 10)),
            window=int(cfg.get("window", 32)),
            image_size=int(cfg.get("image_size", env_cfg.image_size)),
            n_actions=int(cfg.get("n_actions", 9)),
            n_regime_classes=int(cfg.get("n_regime_classes", 4)),
        )
        model.load_state_dict(ckpt["model"])
        env_cfg.window = int(cfg.get("window", env_cfg.window))
        env_cfg.image_size = int(cfg.get("image_size", env_cfg.image_size))
        return TorchModelPolicy(model, device=device), env_cfg, "checkpoint", meta
    print("No checkpoint provided; using random policy for no-money smoke replay.")
    return _random_policy(seed), env_cfg, "random", {}


def _save_decision_log(result: BacktestResult, out_dir: Path, name: str) -> str:
    rows: list[dict] = []
    timestamps = result.timestamps
    actions = result.actions[1 : len(result.info) + 1]
    for i, info in enumerate(result.info):
        row = {
            "step": i,
            "timestamp": str(timestamps[i + 1]) if timestamps is not None and i + 1 < len(timestamps) else "",
            "action": int(actions[i]) if i < len(actions) else -1,
            "action_name": _action_name(int(actions[i])) if i < len(actions) else "UNKNOWN",
        }
        for key, value in info.items():
            if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                row[key] = value
        rows.append(row)
    path = out_dir / f"{name}_decisions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _apply_env_overrides(env_cfg: EnvConfig, args: argparse.Namespace) -> EnvConfig:
    env_cfg.seed = args.seed
    env_cfg.window = int(args.window or env_cfg.window)
    env_cfg.image_size = int(args.image_size or env_cfg.image_size)
    env_cfg.episode_length = int(args.episode_length or env_cfg.episode_length)
    env_cfg.fee_bps = float(args.fee_bps)
    env_cfg.slippage_bps_per_unit = float(args.slippage_bps_per_unit)
    env_cfg.max_leverage = float(args.max_leverage)
    env_cfg.kill_on_drawdown = bool(args.kill_on_drawdown)
    return env_cfg


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a no-money paper replay on historical data.")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/paper_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=0)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps-per-unit", type=float, default=1.5)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--kill-on-drawdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regime-ab", action="store_true")
    parser.add_argument("--benchmark-symbol", type=str, default="")
    add_market_data_args(parser, default_source="tsdb")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_market_dataframe(args, seed=args.seed, default_bars=args.bars)
    policy, env_cfg, policy_source, checkpoint_meta = _load_policy(
        args.checkpoint, device=args.device, seed=args.seed
    )
    env_cfg = _apply_env_overrides(env_cfg, args)

    safety = {
        "mode": "simulated_replay_no_orders",
        "real_orders_enabled": False,
        "exchange_order_api_used": False,
        "api_keys_required": False,
    }

    if args.regime_ab:
        ab = run_regime_ab_backtest(
            df,
            policy,
            env_cfg=env_cfg,
            cfg=RegimeABConfig(symbol=args.symbol, benchmark_symbol=args.benchmark_symbol),
            seed=args.seed,
        )
        print_metrics(ab.baseline.result.metrics, title=ab.baseline.name)
        print_metrics(ab.gated.result.metrics, title=ab.gated.name)
        save_report(ab.baseline.result, out_dir, name=ab.baseline.name)
        save_report(ab.gated.result, out_dir, name=ab.gated.name)
        baseline_log = _save_decision_log(ab.baseline.result, out_dir, ab.baseline.name)
        gated_log = _save_decision_log(ab.gated.result, out_dir, ab.gated.name)
        summary = {
            "safety": safety,
            "policy_source": policy_source,
            "checkpoint_meta": checkpoint_meta,
            "data": frame_summary(df),
            "env": {
                "window": env_cfg.window,
                "image_size": env_cfg.image_size,
                "fee_bps": env_cfg.fee_bps,
                "slippage_bps_per_unit": env_cfg.slippage_bps_per_unit,
                "max_leverage": env_cfg.max_leverage,
            },
            "comparison": ab.comparison,
            "decision_logs": {
                ab.baseline.name: baseline_log,
                ab.gated.name: gated_log,
            },
        }
    else:
        result = run_backtest(df, policy, cfg=env_cfg, seed=args.seed)
        print_metrics(result.metrics, title="paper_policy")
        save_report(result, out_dir, name="paper_policy")
        decisions = _save_decision_log(result, out_dir, "paper_policy")
        bh = buy_and_hold_benchmark(df)
        bh_metrics = compute_metrics(bh)
        print_metrics(bh_metrics, title="buy&hold")
        summary = {
            "safety": safety,
            "policy_source": policy_source,
            "checkpoint_meta": checkpoint_meta,
            "data": frame_summary(df),
            "env": {
                "window": env_cfg.window,
                "image_size": env_cfg.image_size,
                "fee_bps": env_cfg.fee_bps,
                "slippage_bps_per_unit": env_cfg.slippage_bps_per_unit,
                "max_leverage": env_cfg.max_leverage,
            },
            "metrics": result.metrics.to_dict(),
            "buy_and_hold": bh_metrics.to_dict(),
            "decision_log": decisions,
        }

    summary_path = out_dir / "paper_run_summary.json"
    _write_summary(summary_path, summary)
    print(f"Paper replay artifacts saved to: {out_dir}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
