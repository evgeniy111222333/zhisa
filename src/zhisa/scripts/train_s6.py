"""Train a Decision Transformer (S6) on trajectories collected from a base policy.

Usage::

    python -m zhisa.scripts.train_s6 --config configs/s6_dt.yaml
    python -m zhisa.scripts.train_s6 --config configs/s6_dt.yaml --episodes 8
    python -m zhisa.scripts.train_s6 --config configs/s6_dt.yaml \\
        --base-policy artifacts/s2b_bc.pt

The script:

1. Generates a fresh synthetic market.
2. Builds a :class:`PolicyNetwork` whose ``encode`` will be used to
   pre-compute per-state embeddings.
3. Rolls ``--base-policy`` (or a random policy) in :class:`TradingEnv`
   for ``--episodes`` episodes to collect trajectories.
4. Pre-computes state embeddings for every visited state.
5. Builds a :class:`TrajectoryWindowDataset` and trains a
   :class:`DecisionTransformer` on it.
6. Saves a checkpoint containing both the DT body and the
   pre-embedding policy state.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.data.trajectory import collect_trajectories
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.training.decision_transformer import (
    DTConfig,
    DecisionTransformer,
    DecisionTransformerTrainer,
    build_default_dt,
    embed_trajectories,
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
    overrides = (cfg.get("env_cfg", {}) if cfg else {}) or {}
    base = EnvConfig()
    valid = {f for f in base.__dataclass_fields__}
    kwargs = {k: v for k, v in overrides.items() if k in valid}
    return EnvConfig(**kwargs)


def _load_base_policy(checkpoint: Optional[str], in_numeric_features: int, in_context_features: int,
                     window: int, image_size: int, n_regime_classes: int) -> torch.nn.Module:
    model = build_default_policy(
        in_numeric_features=in_numeric_features,
        in_context_features=in_context_features,
        window=window,
        image_size=image_size,
        n_actions=9,
        n_regime_classes=n_regime_classes,
    )
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" in payload:
            try:
                model.load_state_dict(payload["model"], strict=False)
            except Exception:
                pass
    model.eval()
    return model


def _policy_fn_from_model(model: torch.nn.Module) -> callable:
    def fn(obs: dict) -> int:
        with torch.no_grad():
            chart = torch.from_numpy(np.asarray(obs["chart"], dtype=np.float32)).unsqueeze(0)
            numeric = torch.from_numpy(np.asarray(obs["numeric"], dtype=np.float32)).unsqueeze(0)
            context = torch.from_numpy(np.asarray(obs["context"], dtype=np.float32)).unsqueeze(0)
            out = model(chart, numeric, context)
            logits = out.get("policy_logits")
            if logits is None:
                logits = out.get("action_logits")
            if logits is None:
                for v in out.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 2:
                        logits = v
                        break
            if logits is None:
                raise RuntimeError("Policy model produced no usable logits")
            return int(logits.argmax(dim=-1).item())
    return fn


def _random_policy_fn(rng: np.random.Generator, n_actions: int = 9) -> callable:
    def fn(_obs: dict) -> int:
        return int(rng.integers(0, n_actions))
    return fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S6 Decision Transformer (offline RL).")
    parser.add_argument("--config", type=str, default="configs/s6_dt.yaml")
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of rollouts to collect (overrides YAML).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max steps per rollout episode (overrides YAML).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--base-policy", type=str, default=None,
                        help="Optional path to a trained policy checkpoint for collection.")
    parser.add_argument("--random-base", action="store_true",
                        help="Collect trajectories with a uniform-random policy.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/s6_dt/dt.pt")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)

    # --- Data ---
    n_bars = args.bars or (int(cfg.get("bars", 4000)) if cfg else 4000)
    df = generate_market(MarketConfig(n_bars=n_bars))
    chart_window = int(cfg.get("chart_window", 16)) if cfg else 16
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    spec = SampleSpec(
        chart_window=chart_window, feature_window=chart_window,
        image_size=image_size,
        n_regime_states=int(cfg.get("n_regime_states", 4)) if cfg else 4,
    )

    # --- Probed feature dims ---
    probe_ds = MarketDataset(df, spec=spec)
    n_feat = probe_ds._features.shape[1]
    n_ctx = probe_ds._time_features.shape[1]

    # --- Env ---
    env_cfg = _build_env_cfg(cfg)
    env_cfg.window = chart_window
    env_cfg.image_size = image_size
    env = TradingEnv(df, cfg=env_cfg)

    # --- Base policy (for trajectory collection) ---
    rng = np.random.default_rng(seed + 1)
    if args.random_base or (cfg or {}).get("random_base", False):
        policy_fn = _random_policy_fn(rng)
    else:
        base = _load_base_policy(
            args.base_policy, in_numeric_features=n_feat, in_context_features=n_ctx,
            window=spec.chart_window, image_size=spec.image_size,
            n_regime_classes=spec.n_regime_states,
        )
        policy_fn = _policy_fn_from_model(base)

    # --- Collect trajectories ---
    n_episodes = args.episodes or (int(cfg.get("episodes", 4)) if cfg else 4)
    max_steps = args.max_steps or (int(cfg.get("max_steps_per_episode", 200)) if cfg else 200)
    trajectories = collect_trajectories(env, policy_fn, n_episodes=n_episodes, max_steps=max_steps, seed=seed)
    total_steps = sum(len(t) for t in trajectories)
    if total_steps == 0:
        raise RuntimeError("No trajectory steps collected; check env and base policy.")

    # --- Pre-compute state embeddings via the probe policy ---
    embedder = _load_base_policy(
        args.base_policy, in_numeric_features=n_feat, in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_regime_classes=spec.n_regime_states,
    )
    device = args.device or (str(cfg.get("device", _default_device())) if cfg else _default_device())
    trajectories = embed_trajectories(trajectories, embedder, device=device, batch_size=64)

    # --- Build dataset ---
    from torch.utils.data import Subset
    from zhisa.data.trajectory import TrajectoryWindowDataset
    context_length = args.context_length or (int(cfg.get("context_length", 8)) if cfg else 8)
    val_frac = float(cfg.get("val_frac", 0.0)) if cfg else 0.0
    full_dataset = TrajectoryWindowDataset(
        trajectories, context_length=context_length,
        gamma=float(cfg.get("gamma", 1.0)) if cfg else 1.0,
        n_actions=int(cfg.get("n_actions", 9)) if cfg else 9,
    )
    val_dataset = None
    if val_frac > 0.0 and len(full_dataset) > 1:
        n_val = max(1, int(round(len(full_dataset) * val_frac)))
        n_val = min(n_val, len(full_dataset) - 1)
        rng2 = np.random.default_rng(seed + 2)
        perm = rng2.permutation(len(full_dataset))
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()
        dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
    else:
        dataset = full_dataset

    # --- Build DT ---
    from zhisa.models.policy import PolicyConfig
    embed_dim = int(getattr(embedder.cfg, "embed_dim", 128))
    pcfg = PolicyConfig(
        image_size=image_size, in_numeric_features=n_feat, window=chart_window,
        in_context_features=n_ctx, embed_dim=embed_dim,
        n_actions=int(cfg.get("n_actions", 9)) if cfg else 9,
        n_regime_classes=spec.n_regime_states,
    )
    dt_model, dt_cfg = build_default_dt(
        pcfg,
        DTConfig(
            state_dim=embed_dim,
            n_actions=int(cfg.get("n_actions", 9)) if cfg else 9,
            context_length=context_length,
            d_model=int(cfg.get("d_model", 64)) if cfg else 64,
            n_heads=int(cfg.get("n_heads", 4)) if cfg else 4,
            n_layers=int(cfg.get("n_layers", 2)) if cfg else 2,
            dropout=float(cfg.get("dropout", 0.1)) if cfg else 0.1,
            learning_rate=float(cfg.get("learning_rate", 1e-3)) if cfg else 1e-3,
            batch_size=int(cfg.get("batch_size", 32)) if cfg else 32,
            epochs=args.epochs or (int(cfg.get("epochs", 3)) if cfg else 3),
            rtg_loss_weight=float(cfg.get("rtg_loss_weight", 0.0)) if cfg else 0.0,
            max_rtg_clip=float(cfg.get("max_rtg_clip", 10.0)) if cfg else 10.0,
            device=device, seed=seed, verbose=True,
        ),
    )

    trainer = DecisionTransformerTrainer(dt_model, dt_cfg)
    result = trainer.fit(dataset, val_dataset=val_dataset)

    # --- Save ---
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.checkpoint, extra={
        "context_length": context_length,
        "n_trajectories": len(trajectories),
        "n_windows": len(dataset),
    })
    print(f"S6 (Decision Transformer) training complete. final_loss={result.final_loss:.4f}")
    print(f"trajectories={len(trajectories)} steps={total_steps} windows={len(dataset)}")
    print(f"checkpoint saved to: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
