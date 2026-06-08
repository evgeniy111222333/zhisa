"""Stage-1 portfolio reinforcement learning (PPO with per-instrument action masking).

This is a small, runnable training script that wires together:
  * :class:`MarketDataset` to build a single-instrument dataset
  * :class:`PortfolioEnv` to drive the multi-instrument environment
  * :class:`PortfolioPolicyNetwork` as the actor / critic
  * :class:`PortfolioPPOTrainer` for the PPO update with mask consistency

Usage::

    python -m zhisa.scripts.train_portfolio_rl --config configs/portfolio_rl.yaml

Optional CLI flags override the YAML config; see ``--help`` for the full list.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.portfolio_env import PortfolioConfig, PortfolioEnv
from zhisa.env.trading_env import EnvConfig
from zhisa.models.portfolio_policy import (
    PortfolioPolicyConfig,
    PortfolioPolicyNetwork,
)
from zhisa.training.optim import OptimConfig
from zhisa.training.portfolio_ppo import PortfolioPPOConfig, PortfolioPPOTrainer
from zhisa.utils.logging import get_logger
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    """Resolve a sensible default device from env (GPU when available)."""
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"




_LOG = get_logger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Stage-1 portfolio PPO (multi-instrument, mask-aware).",
    )
    p.add_argument("--config", type=str, default="configs/portfolio_rl.yaml",
                   help="YAML config file.")
    p.add_argument("--bars", type=int, default=None,
                   help="Number of synthetic bars to use per instrument.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-instruments", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--minibatch", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--fusion-hidden", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--episode-length", type=int, default=None)
    p.add_argument("--gross-cap", type=float, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--history", type=str, default=None)
    return p


def _resolve_overrides(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI overrides on top of the YAML config."""
    out = dict(cfg)
    for k, v in vars(args).items():
        if v is None or k in {"config"}:
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _to_dataframe(bars: int, seed: int) -> pd.DataFrame:
    return generate_market(MarketConfig(n_bars=int(bars), seed=int(seed)))


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    cfg = _resolve_overrides(args, cfg)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    n_instruments = int(cfg.get("n_instruments", 2))
    if n_instruments < 2:
        raise ValueError("n_instruments must be >= 2 for the portfolio RL stage.")

    env_cfg_dict = cfg.get("env_cfg", {}) or {}
    env_cfg = EnvConfig(
        window=int(env_cfg_dict.get("window", 16)),
        image_size=int(env_cfg_dict.get("image_size", 32)),
        episode_length=int(env_cfg_dict.get("episode_length", 30)),
    )
    pcfg_dict = cfg.get("portfolio_cfg", {}) or {}
    pcfg = PortfolioConfig(
        n_instruments=n_instruments,
        instrument_names=pcfg_dict.get(
            "instrument_names", [f"i{i}" for i in range(n_instruments)]
        ),
        env_cfg=env_cfg,
        gross_leverage_cap=float(pcfg_dict.get("gross_leverage_cap", 1.0)),
    )

    # Probe feature dims.
    probe_df = _to_dataframe(int(cfg.get("bars", 300)), seed=seed)
    probe_env = PortfolioEnv(
        {name: probe_df for name in pcfg.instrument_names}, cfg=pcfg,
    )
    obs, _ = probe_env.reset()
    inst0 = obs["instruments"][0]
    in_numeric = int(inst0["numeric"].shape[-1])
    in_context = int(inst0["context"].shape[-1])
    portfolio_dim = int(obs["portfolio"].shape[0])

    model_dict = cfg.get("model", {}) or {}
    pp_cfg = PortfolioPolicyConfig(
        n_instruments=n_instruments,
        in_numeric_features=in_numeric,
        in_context_features=in_context,
        window=int(model_dict.get("window", env_cfg.window)),
        image_size=int(model_dict.get("image_size", env_cfg.image_size)),
        embed_dim=int(model_dict.get("embed_dim", 32)),
        fusion_hidden=int(model_dict.get("fusion_hidden", 32)),
        portfolio_dim=portfolio_dim,
    )
    model = PortfolioPolicyNetwork(pp_cfg)
    _LOG.info("model_params=%d", sum(p.numel() for p in model.parameters()))

    ppo_dict = cfg.get("ppo", {}) or {}
    lr = float(ppo_dict.get("learning_rate", 3e-4))
    ppo_cfg = PortfolioPPOConfig(
        n_instruments=n_instruments,
        n_iterations=int(ppo_dict.get("iterations", 2)),
        n_episodes=int(ppo_dict.get("episodes", 2)),
        max_steps_per_episode=int(ppo_dict.get("max_steps", env_cfg.episode_length)),
        n_epochs=int(ppo_dict.get("epochs", 1)),
        minibatch_size=int(ppo_dict.get("minibatch_size", 16)),
        gamma=float(ppo_dict.get("gamma", 0.99)),
        gae_lambda=float(ppo_dict.get("gae_lambda", 0.95)),
        clip_ratio=float(ppo_dict.get("clip_ratio", 0.2)),
        value_coef=float(ppo_dict.get("value_loss_coef", 0.5)),
        entropy_coef=float(ppo_dict.get("entropy_coef", 0.01)),
        grad_clip=float(ppo_dict.get("max_grad_norm", 0.5)),
        log_every=int(ppo_dict.get("log_every", 1)),
        device=str(ppo_dict.get("device", _default_device())),
        seed=seed,
        portfolio_dim=portfolio_dim,
        checkpoint=str(cfg.get("checkpoint", "artifacts/portfolio_rl.pt")),
    )
    ppo_cfg.optim = OptimConfig(lr=lr)
    ppo_cfg.env_cfg = env_cfg

    trainer = PortfolioPPOTrainer(model, ppo_cfg)
    data = {name: _to_dataframe(int(cfg.get("bars", 300)), seed=seed + i)
            for i, name in enumerate(pcfg.instrument_names)}

    _LOG.info("start_fit n_instruments=%d bars=%d iters=%d episodes=%d",
              n_instruments, int(cfg.get("bars", 300)),
              ppo_cfg.n_iterations, ppo_cfg.n_episodes)
    out = trainer.fit(data, env_cfg=pcfg)
    history = out.get("history", [])

    history_path = cfg.get("history", "artifacts/portfolio_rl_history.json")
    if history_path:
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=float)

    last = history[-1] if history else {}
    _LOG.info("done mean_return=%.6f mean_gross=%.3f n_updates=%d",
              float(last.get("mean_return", float("nan"))),
              float(last.get("mean_gross_leverage", float("nan"))),
              int(last.get("n_updates", 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
