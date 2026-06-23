"""Train a vanilla S4 PPO control run on prepared or ad-hoc market data."""
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
from zhisa.training.optim import OptimConfig
from zhisa.training.s4_rl import PPOConfig, PPOTrainer
from zhisa.utils.seeding import set_seed


def _default_device() -> str:
    import os
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def _value(cli_value, cfg, key: str, default):
    return cli_value if cli_value is not None else (cfg.get(key, default) if cfg else default)


def _build_env_cfg(cfg):
    """Backward-compatible helper with strict EnvConfig validation."""
    model = build_default_policy(window=32, image_size=32)
    return build_env_config(cfg, model)


def _build_ppo_cfg(cfg, args, env_cfg):
    """Small config helper retained for programmatic callers/tests."""
    optim_raw = (cfg.get("optim", {}) if cfg else {}) or {}
    return PPOConfig(
        n_iterations=int(cfg.get("n_iterations", 100) if cfg else 100),
        n_episodes=int(_value(getattr(args, "n_episodes", None), cfg, "n_episodes", 4)),
        max_steps_per_episode=int(_value(getattr(args, "max_steps", None), cfg, "max_steps_per_episode", 200)),
        n_epochs=int(cfg.get("n_epochs", 4) if cfg else 4),
        minibatch_size=int(cfg.get("minibatch_size", 32) if cfg else 32),
        clip_ratio=float(cfg.get("clip_ratio", 0.2) if cfg else 0.2),
        value_coef=float(cfg.get("value_coef", 0.5) if cfg else 0.5),
        value_loss_scale=float(cfg.get("value_loss_scale", 1.0) if cfg else 1.0),
        entropy_coef=float(cfg.get("entropy_coef", 0.01) if cfg else 0.01),
        gamma=float(cfg.get("gamma", 0.99) if cfg else 0.99),
        gae_lambda=float(cfg.get("gae_lambda", 0.95) if cfg else 0.95),
        grad_clip=float(cfg.get("grad_clip", 1.0) if cfg else 1.0),
        target_kl=float(cfg.get("target_kl", 0.05) if cfg else 0.05),
        device=str(cfg.get("device", _default_device()) if cfg else _default_device()),
        optim=OptimConfig(
            lr=float(optim_raw.get("lr", 3e-4)),
            weight_decay=float(optim_raw.get("weight_decay", 1e-2)),
            scheduler="none",
        ),
        env_cfg=env_cfg,
        seed=int(cfg.get("seed", 0) if cfg else 0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train vanilla S4 PPO baseline.")
    parser.add_argument("--config", default="configs/s4_rl.yaml")
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--max-bars-per-symbol", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--s2-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--checkpoint", default="artifacts/s4/policy.pt")
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--device", default=None)
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

    init_path = args.init_checkpoint or args.s2_checkpoint
    payload = load_trading_checkpoint(init_path) if init_path else None
    model = build_policy_from_checkpoint(payload) if payload else build_default_policy(
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
            args.prepared_root, args.train_split, minimum_bars=minimum,
            max_bars_per_symbol=args.max_bars_per_symbol,
        )
        if not args.no_validation:
            val_markets, _ = load_prepared_markets(
                args.prepared_root, args.val_split, minimum_bars=minimum,
                max_bars_per_symbol=args.max_bars_per_symbol,
            )
    else:
        train_markets = [load_market_dataframe(
            args, seed=seed, default_bars=int(_value(args.bars, cfg, "n_bars", 1500)),
        )]

    if payload is None:
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

    probe = TradingEnv(train_markets[0].iloc[: max(model.cfg.window + 80, 256)], cfg=env_cfg)
    if (probe.obs_numeric_dim, probe.obs_context_dim) != (
        model.cfg.in_numeric_features, model.cfg.in_context_features,
    ):
        raise ValueError("checkpoint observation dimensions do not match TradingEnv")

    optim_raw = (cfg.get("optim", {}) if cfg else {}) or {}
    checkpoint = Path(args.checkpoint)
    best_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_best{checkpoint.suffix}")
    trainer_cfg = PPOConfig(
        n_iterations=int(_value(args.n_iterations, cfg, "n_iterations", 20)),
        n_episodes=int(_value(args.n_episodes, cfg, "n_episodes", 8)),
        max_steps_per_episode=episode_steps,
        n_epochs=int(cfg.get("n_epochs", 4) if cfg else 4),
        minibatch_size=int(cfg.get("minibatch_size", 256) if cfg else 256),
        clip_ratio=float(cfg.get("clip_ratio", 0.2) if cfg else 0.2),
        value_coef=float(cfg.get("value_coef", 0.5) if cfg else 0.5),
        value_loss_scale=float(cfg.get("value_loss_scale", 1.0) if cfg else 1.0),
        entropy_coef=float(cfg.get("entropy_coef", 0.01) if cfg else 0.01),
        gamma=float(cfg.get("gamma", 0.99) if cfg else 0.99),
        gae_lambda=float(cfg.get("gae_lambda", 0.95) if cfg else 0.95),
        grad_clip=float(cfg.get("grad_clip", 1.0) if cfg else 1.0),
        target_kl=float(cfg.get("target_kl", 0.05) if cfg else 0.05),
        device=str(args.device or (cfg.get("device", _default_device()) if cfg else _default_device())),
        optim=OptimConfig(
            lr=float(optim_raw.get("lr", 3e-5)),
            weight_decay=float(optim_raw.get("weight_decay", 1e-4)),
            scheduler="none",
        ),
        env_cfg=env_cfg,
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
    )
    trainer = PPOTrainer(model, trainer_cfg)
    if args.resume_from:
        print(f"Resumed S4 PPO: {trainer.load(args.resume_from)}")
    result = trainer.fit(train_markets, val_df=val_markets)
    last = result["history"][-1] if result["history"] else {}
    print(
        f"S4 PPO training complete: iterations={len(result['history'])} "
        f"equity_return={last.get('mean_equity_return', 0.0):.6f}"
    )
    print(f"checkpoint saved to: {checkpoint}")
    if val_markets:
        print(f"best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
