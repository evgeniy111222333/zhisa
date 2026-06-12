"""Train an S2b imitation-learning policy (BC or DAgger) on a synthetic market.

Usage::

    python -m zhisa.scripts.train_s2b --config configs/s2b_imitation.yaml
    python -m zhisa.scripts.train_s2b --config configs/s2b_imitation.yaml --trainer dagger
    python -m zhisa.scripts.train_s2b --config configs/s2b_imitation.yaml \\
        --bars 8000 --epochs 3

The script generates a fresh synthetic market, builds the policy with
probed feature dimensionality, and runs either behavioural cloning or
DAgger (Dataset Aggregation) under the chosen rule-based expert. The
resulting checkpoint is compatible with the S4 PPO trainer's
``load_state_dict``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.expert import SUPPORTED_EXPERTS, build_expert
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.env.trading_env import EnvConfig
from zhisa.models.policy import build_default_policy
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s2b_imitation import (
    BCConfig,
    BehavioralCloningTrainer,
    DAggerConfig,
    DAggerTrainer,
)
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
    """Apply YAML overrides on top of a default :class:`EnvConfig`."""
    overrides = (cfg.get("env_cfg", {}) if cfg else {}) or {}
    base = EnvConfig()
    valid = {f for f in base.__dataclass_fields__}
    kwargs = {k: v for k, v in overrides.items() if k in valid}
    return EnvConfig(**kwargs)


def _build_optim(cfg) -> OptimConfig:
    optim_overrides = (cfg.get("optim", {}) or {}) if cfg else {}
    return OptimConfig(
        lr=float(optim_overrides.get("lr", 3e-4)),
        weight_decay=float(optim_overrides.get("weight_decay", 1e-2)),
        scheduler=str(optim_overrides.get("scheduler", "cosine")),
        warmup_steps=int(optim_overrides.get("warmup_steps", 0)),
    )


def _build_loss_weights(cfg) -> LossWeights:
    w = (cfg.get("loss_weights", {}) or {}) if cfg else {}
    return LossWeights(
        direction=float(w.get("direction", 1.0)),
        volatility=float(w.get("volatility", 0.5)),
        regime=float(w.get("regime", 0.3)),
        return_pred=float(w.get("return_pred", 0.5)),
        policy=float(w.get("policy", 1.0)),
        value=float(w.get("value", 0.5)),
        uncertainty=float(w.get("uncertainty", 0.05)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S2b imitation policy (BC or DAgger).")
    parser.add_argument("--config", type=str, default="configs/s2b_imitation.yaml")
    parser.add_argument("--trainer", type=str, choices=("bc", "dagger"), default=None,
                        help="Override the trainer kind from the config.")
    parser.add_argument("--expert", type=str, choices=sorted(SUPPORTED_EXPERTS), default=None,
                        help="Override the expert kind from the config.")
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="artifacts/s2b/model.pt")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None

    set_seed(int(cfg.get("seed", 0)) if cfg else 0)

    # --- Data ---
    n_bars = args.bars or int(cfg.get("bars", 4000)) if cfg else 4000
    df = generate_market(MarketConfig(n_bars=n_bars))

    chart_window = int(cfg.get("chart_window", 32)) if cfg else 32
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    spec = SampleSpec(
        chart_window=chart_window, feature_window=chart_window,
        image_size=image_size,
        n_regime_states=int(cfg.get("n_regime_states", 4)) if cfg else 4,
    )

    # --- Model ---
    probe_ds = MarketDataset(df, spec=spec)
    n_feat = probe_ds._features.shape[1]
    n_ctx = probe_ds._time_features.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=n_ctx,
        window=spec.chart_window,
        image_size=spec.image_size,
        n_actions=9,
        n_regime_classes=spec.n_regime_states,
    )

    # --- Expert ---
    expert_kind = args.expert or (cfg.get("expert", "triple_barrier") if cfg else "triple_barrier")
    expert_kwargs = dict(cfg.get("expert_kwargs", {}) or {}) if cfg else {}
    expert_kwargs.setdefault("chart_window", chart_window)
    expert = build_expert(expert_kind, **expert_kwargs)

    # --- Optim / loss ---
    optim_cfg = _build_optim(cfg)
    loss_weights = _build_loss_weights(cfg)
    device = args.device or (str(cfg.get("device", _default_device())) if cfg else _default_device())

    # --- Trainer ---
    trainer_kind = args.trainer or (cfg.get("trainer", "bc") if cfg else "bc")
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    if trainer_kind == "bc":
        epochs = args.epochs or (int(cfg.get("bc_epochs", 3)) if cfg else 3)
        bc_cfg = BCConfig(
            epochs=epochs, batch_size=int(cfg.get("batch_size", 32)) if cfg else 32,
            grad_clip=float(cfg.get("grad_clip", 1.0)) if cfg else 1.0,
            log_every=int(cfg.get("log_every", 50)) if cfg else 50,
            device=device, seed=int(cfg.get("seed", 0)) if cfg else 0,
            optim=optim_cfg, loss_weights=loss_weights,
            checkpoint=args.checkpoint,
        )
        loss = MultiTaskLoss(loss_weights)
        trainer = BehavioralCloningTrainer(model, loss, bc_cfg)
        res = trainer.fit(df, expert, spec=spec)
        history = res["history"]
        final_loss = history[-1]["loss"] if history else float("nan")
    else:
        n_rounds = args.rounds or (int(cfg.get("dagger_rounds", 3)) if cfg else 3)
        env_cfg = _build_env_cfg(cfg)
        # Sync the env's window / image size with the model / dataset.
        env_cfg.window = chart_window
        env_cfg.image_size = image_size
        dagger_cfg = DAggerConfig(
            n_rounds=n_rounds,
            epochs_per_round=int(cfg.get("epochs_per_round", 1)) if cfg else 1,
            rollout_episodes_per_round=int(cfg.get("rollout_episodes_per_round", 2)) if cfg else 2,
            max_steps_per_episode=int(cfg.get("max_steps_per_episode", 200)) if cfg else 200,
            batch_size=int(cfg.get("batch_size", 32)) if cfg else 32,
            grad_clip=float(cfg.get("grad_clip", 1.0)) if cfg else 1.0,
            log_every=int(cfg.get("log_every", 50)) if cfg else 50,
            device=device, seed=int(cfg.get("seed", 0)) if cfg else 0,
            optim=optim_cfg, loss_weights=loss_weights,
            env_cfg=env_cfg, checkpoint=args.checkpoint,
        )
        trainer = DAggerTrainer(model, expert, dagger_cfg)
        res = trainer.fit(df, spec=spec)
        history = [{"round": r.round_idx, "loss": r.bc_loss, "n_aggregated": r.n_aggregated,
                    "n_new_pairs": r.n_new_pairs, "elapsed_s": r.elapsed_s} for r in res.rounds]
        final_loss = res.final_loss

    print(f"S2b ({trainer_kind}) training complete. final_loss={final_loss:.4f}")
    print(f"checkpoint saved to: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
