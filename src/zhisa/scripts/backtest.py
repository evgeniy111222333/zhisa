"""Backtest a trained policy (or a random baseline) on synthetic data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from zhisa.backtest.engine import buy_and_hold_benchmark, run_backtest
from zhisa.backtest.metrics import compute_metrics
from zhisa.backtest.reports import print_metrics, save_report
from zhisa.backtest.regime_ab import RegimeABConfig, run_regime_ab_backtest
from zhisa.data.synthetic import generate_market  # kept for test monkeypatch compatibility
from zhisa.env.trading_env import EnvConfig
from zhisa.data.render_contract import assert_serving_render
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.scripts._rl_training import build_policy_from_checkpoint
from zhisa.utils.seeding import set_seed


def _checkpoint_policy_config(ckpt: dict) -> dict:
    """Return the saved PolicyConfig dict from a checkpoint, if present."""
    for key in ("model_config", "policy_config", "config"):
        cfg = ckpt.get(key)
        if isinstance(cfg, dict) and "window" in cfg and "in_numeric_features" in cfg:
            return cfg
    return {}


def _checkpoint_policy_metadata(ckpt: dict) -> dict:
    """Return checkpoint metadata used for replay/backtest guardrails."""
    meta = ckpt.get("checkpoint_meta")
    return meta if isinstance(meta, dict) else {}


def _warn_if_checkpoint_not_trading_ready(ckpt: dict, checkpoint: str | None) -> None:
    """Emit a clear warning for checkpoints whose policy head is not trade-trained."""
    meta = _checkpoint_policy_metadata(ckpt)
    if meta.get("trading_policy_ready") is False:
        stage = meta.get("stage", "unknown")
        reason = meta.get("reason", "checkpoint metadata marks this policy as not trading-ready")
        print(
            "WARNING: checkpoint is not marked as a trading-ready policy "
            f"(stage={stage}, checkpoint={checkpoint}). {reason}"
        )


def _random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)

    def _p(_obs):
        return int(rng.integers(0, 9))

    return _p


class TorchModelPolicy:
    """Callable policy that also exposes logits for regime-aware gating."""

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        from zhisa.models.session import make_stateful_policy
        # ONE rolling-memory session across the whole backtest (S4/serve parity)
        self._call = make_stateful_policy(model)
        self._state = None

    def logits(self, obs) -> torch.Tensor:
        from zhisa.models.session import session_step
        out, self._state = session_step(self.model, obs, self._state)
        return out["policy_logits"].squeeze(0).detach().cpu()

    def __call__(self, obs) -> int:
        return int(self.logits(obs).argmax(dim=-1).item())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest a policy.")
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/backtest")
    parser.add_argument("--seed", type=int, default=0)
    add_market_data_args(parser)
    parser.add_argument(
        "--regime-ab",
        action="store_true",
        help="run baseline and regime-gated variants side by side",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    df = load_market_dataframe(args, seed=args.seed, default_bars=args.bars)

    env_cfg = EnvConfig(seed=args.seed)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        _warn_if_checkpoint_not_trading_ready(ckpt, args.checkpoint)
        model = build_policy_from_checkpoint(ckpt)
        policy = TorchModelPolicy(model)
        env_cfg.window = int(model.cfg.window)
        env_cfg.image_size = int(model.cfg.image_size)
        assert_serving_render(ckpt, env_cfg.image_size)
    else:
        policy = _random_policy(args.seed)
        print("No checkpoint provided; using random policy for smoke test.")

    if args.regime_ab:
        ab = run_regime_ab_backtest(
            df,
            policy,
            env_cfg=env_cfg,
            cfg=RegimeABConfig(),
            seed=args.seed,
        )
        print_metrics(ab.baseline.result.metrics, title=ab.baseline.name)
        print_metrics(ab.gated.result.metrics, title=ab.gated.name)
        print("== regime summary ==")
        print(json.dumps(ab.gated.regime_summary, indent=2))
        bh = buy_and_hold_benchmark(df)
        print_metrics(compute_metrics(bh), title="buy&hold")
        if args.out:
            save_report(ab.baseline.result, args.out, name=ab.baseline.name)
            save_report(ab.gated.result, args.out, name=ab.gated.name)
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "regime_ab_comparison.json").open("w", encoding="utf-8") as f:
                json.dump(ab.comparison, f, indent=2)
        return 0

    result = run_backtest(df, policy, cfg=env_cfg)
    print_metrics(result.metrics, title="policy")
    bh = buy_and_hold_benchmark(df)
    bh_metrics = compute_metrics(bh)
    print_metrics(bh_metrics, title="buy&hold")
    if args.out:
        save_report(result, args.out, name="policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
