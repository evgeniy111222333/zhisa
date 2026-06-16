"""Train a policy through the S4 PPO loop on a synthetic market.

By default this runs ``n_episodes`` short trading episodes against a
fresh OHLCV stream generated on every iteration by
:func:`generate_market`. The model's feature dimensionality is
auto-probed from a :class:`MarketDataset` and passed to
:func:`build_default_policy`.

Usage::

    python -m zhisa.scripts.train_s4 --config configs/s4_rl.yaml
    python -m zhisa.scripts.train_s4 --config configs/s4_rl.yaml \\
        --n-episodes 10 --max-steps 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zhisa.config import load_config
from zhisa.data.dataset import SampleSpec
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.training.optim import OptimConfig
from zhisa.training.s4_rl import PPOConfig, PPOTrainer
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
    """Return an :class:`EnvConfig` with YAML overrides applied on top."""
    overrides = (cfg.get("env_cfg", {}) if cfg else {}) or {}
    base = EnvConfig()
    valid = {f for f in base.__dataclass_fields__}
    kwargs = {k: v for k, v in overrides.items() if k in valid}
    return EnvConfig(**kwargs)


def _build_ppo_cfg(cfg, args, env_cfg: EnvConfig) -> PPOConfig:
    """Resolve all CLI/config knobs into a :class:`PPOConfig`."""
    def opt(key, default):
        return cfg.get(key, default) if cfg else default

    optim_overrides = opt("optim", {}) or {}
    return PPOConfig(
        n_iterations=int(args.n_iterations if args.n_iterations is not None
                         else opt("n_iterations", 100)),
        n_episodes=int(args.n_episodes if args.n_episodes is not None
                       else opt("n_episodes", 10)),
        max_steps_per_episode=int(args.max_steps if args.max_steps is not None
                                  else opt("max_steps_per_episode", 200)),
        n_epochs=int(opt("n_epochs", 4)),
        minibatch_size=int(opt("minibatch_size", 32)),
        clip_ratio=float(opt("clip_ratio", 0.2)),
        value_coef=float(opt("value_coef", 0.5)),
        entropy_coef=float(opt("entropy_coef", 0.01)),
        gamma=float(opt("gamma", 0.99)),
        gae_lambda=float(opt("gae_lambda", 0.95)),
        grad_clip=float(opt("grad_clip", 1.0)),
        target_kl=float(opt("target_kl", 0.05)),
        device=str(opt("device", _default_device())),
        optim=OptimConfig(
            lr=float(optim_overrides.get("lr", 3e-4)),
            weight_decay=float(optim_overrides.get("weight_decay", 1e-2)),
            warmup_steps=int(optim_overrides.get("warmup_steps", 0)),
        ),
        env_cfg=env_cfg,
        seed=int(opt("seed", 0)),
        log_every=int(opt("log_every", 1)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S4 PPO policy.")
    parser.add_argument("--config", type=str, default="configs/s4_rl.yaml")
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="artifacts/s4/policy.pt")
    parser.add_argument("--s2-checkpoint", type=str, default=None, help="Path to S2 checkpoint to load base weights from.")
    parser.add_argument("--fast-render", action="store_true", help="Use numpy renderer without matplotlib")
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)

    if args.fast_render:
        import os
        os.environ["ZHISA_FAST_RENDER"] = "1"

    chart_window = int(cfg.get("chart_window", 32)) if cfg else 32
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    n_bars = int(args.bars or (cfg.get("n_bars", 1500) if cfg else 1500))
    df = load_market_dataframe(args, seed=seed, default_bars=n_bars)

    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size)

    env_cfg = _build_env_cfg(cfg)
    env_cfg.window = chart_window
    env_cfg.image_size = image_size

    # Probe feature dim on a small slice of the same env data contract.
    probe_len = min(len(df), max(chart_window + 80, 128))
    probe_env = TradingEnv(df.iloc[:probe_len], cfg=env_cfg)
    n_feat = probe_env.obs_numeric_dim
    n_ctx = probe_env.obs_context_dim

    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=9, n_regime_classes=spec.n_regime_states,
    )

    if args.s2_checkpoint:
        print(f"Loading S2 checkpoint from {args.s2_checkpoint}...")
        import torch
        sd = torch.load(args.s2_checkpoint, map_location="cpu", weights_only=False)
        from zhisa.training.s1_ssl import _filter_matching_state_dict
        filtered = _filter_matching_state_dict(sd["model"], model)
        model.load_state_dict(filtered, strict=False)
        print("S2 weights loaded successfully.")

    ppo_cfg = _build_ppo_cfg(cfg, args, env_cfg)

    import time
    start_time = time.time()
    
    trainer = PPOTrainer(model, ppo_cfg)
    result = trainer.fit(df)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.checkpoint)

    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"S4 PPO training complete in {int(hours)}h {int(minutes)}m {int(seconds)}s.")
    history = result["history"]
    if history:
        last = history[-1]
        ret = last.get("episode_return", last.get("mean_return", 0.0))
        steps = last.get("steps", last.get("total_steps", last.get("rollout_steps", 0)))
        print(f"final episode: return={ret:.4f} steps={steps}")
    print(f"checkpoint saved to: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
