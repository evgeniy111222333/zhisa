"""S4-CVaR: Train a CVaR-Constrained PPO agent on synthetic data.

Usage::

    python -m zhisa.scripts.train_s4_cvar --config configs/s4_cvar_ppo.yaml
    python -m zhisa.scripts.train_s4_cvar --config configs/s4_cvar_ppo.yaml \\
        --bars 8000 --iterations 5

The script trains a :class:`CVaRPPOTrainer` — PPO with a
Lagrangian multiplier enforcing ``-CVaR_alpha <= threshold`` on
per-episode returns. The dual multiplier is updated by ascent
after each rollout; the resulting penalty is added to the PPO
loss to shape the policy towards the constraint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zhisa.config import load_config
from zhisa.data.dataset import SampleSpec
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.training.cvar_ppo import CVaRPPOConfig, CVaRPPOTrainer
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    """Resolve a sensible default device from env (GPU when available)."""
    import os
    import torch
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"




def _build_env_cfg(cfg) -> EnvConfig:
    overrides = (cfg.get("env_cfg", {}) if cfg else {}) or {}
    base = EnvConfig()
    valid = {f for f in base.__dataclass_fields__}
    kwargs = {k: v for k, v in overrides.items() if k in valid}
    return EnvConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S4-CVaR PPO.")
    parser.add_argument("--config", type=str, default="configs/s4_cvar_ppo.yaml")
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--cvar-threshold", type=float, default=None)
    parser.add_argument("--cvar-lambda-lr", type=float, default=None)
    parser.add_argument("--cvar-warmup-iters", type=int, default=None,
                        help="Override cvar warmup iterations (default 5 for s4_cvar_v2)")
    parser.add_argument("--minibatch-size", type=int, default=None,
                        help="Override PPO mini-batch size (default 256 for s4_cvar_v2)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient override")
    parser.add_argument("--checkpoint", type=str, default="artifacts/s4_cvar/model.pt")
    parser.add_argument("--load", type=str, default=None, help="Path to existing model weights to continue training")
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)
    device = args.device or (str(cfg.get("device", _default_device())) if cfg else _default_device())

    n_bars = int(args.bars or (cfg.get("bars", 4000) if cfg else 4000))
    df = load_market_dataframe(args, seed=seed, default_bars=n_bars)

    chart_window = int(cfg.get("chart_window", 16)) if cfg else 16
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    spec = SampleSpec(
        chart_window=chart_window, feature_window=chart_window,
        image_size=image_size,
        n_regime_states=int(cfg.get("n_regime_states", 4)) if cfg else 4,
    )

    env_cfg = _build_env_cfg(cfg)
    env_cfg.window = chart_window
    env_cfg.image_size = image_size
    # Pre-render every rolling-window chart once at env construction.
    # This trades ~1-3 seconds of init time for completely removing
    # matplotlib from the per-step hot path — for a 200-iter CVaR-PPO
    # fit with 2000 step-rollouts this is typically the single
    # biggest win after the on-device history fix in Tier 1.
    # Override in YAML/config to disable.
    if not getattr(env_cfg, "prefetch_charts", False):
        env_cfg.prefetch_charts = True

    probe_len = min(len(df), max(chart_window + 80, 128))
    probe_env = TradingEnv(df.iloc[:probe_len], cfg=env_cfg)
    n_feat = probe_env.obs_numeric_dim
    n_ctx = probe_env.obs_context_dim
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=spec.n_regime_states,
    )
    if args.load:
        ckpt = torch.load(args.load, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded existing weights from {args.load}")

    # Speed-oriented defaults for the s4_cvar_v2 line:
    # * minibatch_size=256 keeps the GPU/CPU busier per PPO step (the
    #   2000-step rollouts we run split into ~8 minibatches/epoch
    #   instead of ~32, with no change to the algorithm).
    # * cvar_warmup_iters=5 lets the policy explore for the first five
    #   iterations before the dual multiplier can grow — this is
    #   mathematically equivalent to running with warmup=0 and an
    #   initial lambda=0, but saves the dual-update branch in the
    #   inner loop and (more importantly) gives the policy a head start
    #   so the first few iterations don't waste effort chasing a
    #   noisy CVaR signal.
    trainer_cfg = CVaRPPOConfig(
        n_iterations=args.iterations or (int(cfg.get("n_iterations", 5)) if cfg else 5),
        n_episodes=args.episodes or (int(cfg.get("n_episodes", 4)) if cfg else 4),
        max_steps_per_episode=args.max_steps or (int(cfg.get("max_steps_per_episode", 200)) if cfg else 200),
        n_epochs=int(cfg.get("n_epochs", 4)) if cfg else 4,
        minibatch_size=args.minibatch_size or (int(cfg.get("minibatch_size", 256)) if cfg else 256),
        clip_ratio=float(cfg.get("clip_ratio", 0.2)) if cfg else 0.2,
        value_coef=float(cfg.get("value_coef", 0.5)) if cfg else 0.5,
        entropy_coef=args.ent_coef or (float(cfg.get("entropy_coef", 0.01)) if cfg else 0.01),
        gamma=float(cfg.get("gamma", 0.99)) if cfg else 0.99,
        gae_lambda=float(cfg.get("gae_lambda", 0.95)) if cfg else 0.95,
        grad_clip=float(cfg.get("grad_clip", 1.0)) if cfg else 1.0,
        target_kl=float(cfg.get("target_kl", 0.05)) if cfg else 0.05,
        cvar_alpha=args.cvar_alpha or (float(cfg.get("cvar_alpha", 0.1)) if cfg else 0.1),
        cvar_threshold=args.cvar_threshold or (float(cfg.get("cvar_threshold", 0.1)) if cfg else 0.1),
        cvar_lambda_init=float(cfg.get("cvar_lambda_init", 0.0)) if cfg else 0.0,
        cvar_lambda_lr=args.cvar_lambda_lr or (float(cfg.get("cvar_lambda_lr", 0.05)) if cfg else 0.05),
        cvar_lambda_max=float(cfg.get("cvar_lambda_max", 100.0)) if cfg else 100.0,
        cvar_warmup_iters=args.cvar_warmup_iters or (int(cfg.get("cvar_warmup_iters", 5)) if cfg else 5),
        env_cfg=env_cfg,
        device=device,
        seed=seed,
        checkpoint=args.checkpoint,
        log_every=int(cfg.get("log_every", 1)) if cfg else 1,
    )

    trainer = CVaRPPOTrainer(model, trainer_cfg)
    result = trainer.fit(df)
    history = trainer.cvar_history
    final_lambda = history[-1]["lambda_cvar"] if history else 0.0
    final_cvar = history[-1]["cvar"] if history else 0.0
    print(
        f"S4-CVaR training complete. iterations={len(history)} "
        f"final_lambda={final_lambda:.4f} final_cvar={final_cvar:.4f}"
    )
    print(f"checkpoint saved to: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
