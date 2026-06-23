"""Train S4 CVaR-constrained PPO on real prepared or ad-hoc market data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zhisa.config import load_config
from zhisa.env.trading_env import TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.scripts._rl_training import (
    build_env_config,
    build_policy_from_checkpoint,
    load_prepared_markets,
    load_trading_checkpoint,
)
from zhisa.training.cvar_ppo import CVaRPPOConfig, CVaRPPOTrainer
from zhisa.training.optim import OptimConfig
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    import os

    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _value(cli_value, cfg, key: str, default):
    return cli_value if cli_value is not None else (cfg.get(key, default) if cfg else default)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S4 CVaR PPO.")
    parser.add_argument("--config", default="configs/s4_cvar_ppo.yaml")
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--max-bars-per-symbol", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None, help="S2b/S3 trading-policy checkpoint")
    parser.add_argument("--resume-from", default=None, help="Resume an S4-CVaR trainer checkpoint")
    parser.add_argument("--load", default=None, help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--checkpoint", default="artifacts/s4_cvar/model.pt")
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--cvar-threshold", type=float, default=None)
    parser.add_argument("--cvar-lambda-lr", type=float, default=None)
    parser.add_argument("--cvar-warmup-iters", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--fast-render", action="store_true")
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0) if cfg else 0)
    set_seed(seed)
    if args.fast_render:
        import os
        os.environ["ZHISA_FAST_RENDER"] = "1"

    init_path = args.init_checkpoint or args.load
    init_payload = load_trading_checkpoint(init_path) if init_path else None
    if init_payload:
        model = build_policy_from_checkpoint(init_payload)
    else:
        model = build_default_policy(
            in_numeric_features=32,
            in_context_features=10,
            window=int(cfg.get("chart_window", 32) if cfg else 32),
            image_size=int(cfg.get("image_size", 32) if cfg else 32),
            n_actions=9,
            n_regime_classes=int(cfg.get("n_regime_states", 4) if cfg else 4),
        )

    env_cfg = build_env_config(cfg, model)
    env_cfg.random_start = bool(args.prepared_root or env_cfg.random_start)
    episode_steps = int(_value(args.max_steps, cfg, "max_steps_per_episode", 200))
    env_cfg.episode_length = episode_steps

    data_meta = None
    val_markets = None
    if args.prepared_root:
        minimum = model.cfg.window + episode_steps + 2
        train_markets, data_meta = load_prepared_markets(
            args.prepared_root,
            args.train_split,
            minimum_bars=minimum,
            max_bars_per_symbol=args.max_bars_per_symbol,
        )
        if not args.no_validation:
            val_markets, val_meta = load_prepared_markets(
                args.prepared_root,
                args.val_split,
                minimum_bars=minimum,
                max_bars_per_symbol=args.max_bars_per_symbol,
            )
            if val_meta["manifest_checksum"] != data_meta["manifest_checksum"]:
                raise ValueError("train/validation manifest mismatch")
    else:
        n_bars = int(_value(args.bars, cfg, "bars", 4000))
        train_markets = [load_market_dataframe(args, seed=seed, default_bars=n_bars)]

    if init_payload is None:
        dimension_probe = TradingEnv(
            train_markets[0].iloc[: max(model.cfg.window + 80, 256)], cfg=env_cfg,
        )
        model = build_default_policy(
            in_numeric_features=dimension_probe.obs_numeric_dim,
            in_context_features=dimension_probe.obs_context_dim,
            window=model.cfg.window,
            image_size=model.cfg.image_size,
            n_actions=9,
            n_regime_classes=model.cfg.n_regime_classes,
        )
        env_cfg = build_env_config(cfg, model)
        env_cfg.random_start = bool(args.prepared_root or env_cfg.random_start)
        env_cfg.episode_length = episode_steps

    # Probe the exact environment/model input contract before allocating PPO.
    probe = TradingEnv(train_markets[0].iloc[: max(model.cfg.window + 80, 256)], cfg=env_cfg)
    if (probe.obs_numeric_dim, probe.obs_context_dim) != (
        model.cfg.in_numeric_features, model.cfg.in_context_features,
    ):
        raise ValueError("checkpoint observation dimensions do not match TradingEnv")

    optim_raw = (cfg.get("optim", {}) if cfg else {}) or {}
    checkpoint = Path(args.checkpoint)
    best_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_best{checkpoint.suffix}")
    trainer_cfg = CVaRPPOConfig(
        n_iterations=int(_value(args.iterations, cfg, "n_iterations", 5)),
        n_episodes=int(_value(args.episodes, cfg, "n_episodes", 4)),
        max_steps_per_episode=episode_steps,
        n_epochs=int(cfg.get("n_epochs", 4) if cfg else 4),
        minibatch_size=int(_value(args.minibatch_size, cfg, "minibatch_size", 256)),
        clip_ratio=float(cfg.get("clip_ratio", 0.2) if cfg else 0.2),
        value_coef=float(cfg.get("value_coef", 0.5) if cfg else 0.5),
        value_loss_scale=float(cfg.get("value_loss_scale", 1.0) if cfg else 1.0),
        entropy_coef=float(_value(args.ent_coef, cfg, "entropy_coef", 0.01)),
        gamma=float(cfg.get("gamma", 0.99) if cfg else 0.99),
        gae_lambda=float(cfg.get("gae_lambda", 0.95) if cfg else 0.95),
        grad_clip=float(cfg.get("grad_clip", 1.0) if cfg else 1.0),
        target_kl=float(cfg.get("target_kl", 0.05) if cfg else 0.05),
        cvar_alpha=float(_value(args.cvar_alpha, cfg, "cvar_alpha", 0.1)),
        cvar_threshold=float(_value(args.cvar_threshold, cfg, "cvar_threshold", 0.1)),
        cvar_lambda_init=float(cfg.get("cvar_lambda_init", 0.0) if cfg else 0.0),
        cvar_lambda_lr=float(_value(args.cvar_lambda_lr, cfg, "cvar_lambda_lr", 0.05)),
        cvar_lambda_max=float(cfg.get("cvar_lambda_max", 100.0) if cfg else 100.0),
        cvar_warmup_iters=int(_value(args.cvar_warmup_iters, cfg, "cvar_warmup_iters", 5)),
        env_cfg=env_cfg,
        device=str(args.device or (cfg.get("device", _default_device()) if cfg else _default_device())),
        seed=seed,
        checkpoint=str(checkpoint),
        best_checkpoint=str(best_checkpoint) if val_markets else None,
        checkpoint_every_iterations=int(cfg.get("checkpoint_every_iterations", 0) if cfg else 0),
        source_checkpoint=str(Path(init_path).resolve()) if init_path else None,
        dataset_root=data_meta["root"] if data_meta else None,
        dataset_manifest_checksum=data_meta["manifest_checksum"] if data_meta else None,
        eval_every_iterations=int(cfg.get("eval_every_iterations", 0) if cfg else 0),
        eval_episodes=int(cfg.get("eval_episodes", 12) if cfg else 12),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 0) if cfg else 0),
        early_stopping_min_delta=float(cfg.get("early_stopping_min_delta", 0.0) if cfg else 0.0),
        log_every=int(cfg.get("log_every", 1) if cfg else 1),
        optim=OptimConfig(
            lr=float(optim_raw.get("lr", 3e-5)),
            weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
            scheduler="none",
        ),
    )
    trainer = CVaRPPOTrainer(model, trainer_cfg)
    if args.resume_from:
        print(f"Resumed S4-CVaR: {trainer.load(args.resume_from)}")
    result = trainer.fit(train_markets, val_df=val_markets)
    last = result["history"][-1] if result["history"] else {}
    print(
        f"S4-CVaR training complete. iterations={len(result['history'])} "
        f"equity_return={last.get('mean_equity_return', 0.0):.6f} "
        f"final_cvar={last.get('cvar', 0.0):.6f} final_lambda={trainer.lambda_cvar:.4f}"
    )
    print(f"checkpoint saved to: {checkpoint}")
    if val_markets:
        print(f"best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
