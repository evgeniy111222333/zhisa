"""S7: Train a world model + Dyna-style PPO on synthetic data.

Usage::

    python -m zhisa.scripts.train_s7 --config configs/s7_world_model.yaml
    python -m zhisa.scripts.train_s7 --config configs/s7_world_model.yaml --dyna-rounds 3

The script:

1. Generates a synthetic market and embeds it via a frozen
   :class:`PolicyNetwork`.
2. Rolls a behaviour policy (a base policy if given, otherwise
   random) in :class:`TradingEnv` to collect trajectories.
3. Trains a :class:`WorldModel` on those trajectories.
4. Runs ``--dyna-rounds`` Dyna updates: each update samples
   initial states from the real pool, imagines rollouts in the
   world model, and trains a :class:`LatentActorCritic` with
   PPO on the imagined rewards.
5. Saves a checkpoint containing the WM, the actor-critic, and
   the final imagined-return metric.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from zhisa.config import load_config
from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.data.trajectory import collect_trajectories
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.latent_actor_critic import LatentActorCritic, LatentActorCriticConfig
from zhisa.models.policy import build_default_policy
from zhisa.models.world_model import WorldModel, WorldModelConfig
from zhisa.scripts._real_data import add_market_data_args, load_market_dataframe
from zhisa.training.decision_transformer import embed_trajectories
from zhisa.training.dyna_ppo import DynaPPOConfig, DynaPPOTrainer
from zhisa.training.world_model_trainer import (
    WorldModelDataset,
    WorldModelTrainer,
    WorldModelTrainerConfig,
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


def _random_policy_fn(rng: np.random.Generator, n_actions: int = 9):
    def fn(_o):
        return int(rng.integers(0, n_actions))
    return fn


def _policy_fn_from_model(model: torch.nn.Module):
    def fn(obs):
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
                raise RuntimeError("No logits found in policy output")
            return int(logits.argmax(dim=-1).item())
    return fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train S7 World Model + Dyna PPO.")
    parser.add_argument("--config", type=str, default="configs/s7_world_model.yaml")
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wm-epochs", type=int, default=None)
    parser.add_argument("--dyna-rounds", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--random-base", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="artifacts/s7_wm/wm.pt")
    parser.add_argument("--dyna-checkpoint", type=str, default="artifacts/s7_wm/dyna.pt")
    add_market_data_args(parser)
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else None
    seed = int(cfg.get("seed", 0)) if cfg else 0
    set_seed(seed)
    device = args.device or (str(cfg.get("device", _default_device())) if cfg else _default_device())

    # --- Market + dataset probe ---
    n_bars = args.bars or (int(cfg.get("bars", 4000)) if cfg else 4000)
    if str(getattr(args, "data_source", "synthetic")) == "synthetic":
        df = generate_market(MarketConfig(n_bars=n_bars))
    else:
        df = load_market_dataframe(args, seed=seed, default_bars=n_bars)
    chart_window = int(cfg.get("chart_window", 16)) if cfg else 16
    image_size = int(cfg.get("image_size", 32)) if cfg else 32
    spec = SampleSpec(
        chart_window=chart_window, feature_window=chart_window,
        image_size=image_size,
        n_regime_states=int(cfg.get("n_regime_states", 4)) if cfg else 4,
    )
    probe_ds = MarketDataset(df, spec=spec)
    n_feat = probe_ds._features.shape[1]
    n_ctx = probe_ds._time_features.shape[1]

    # --- Embedder (frozen) + behaviour policy ---
    embedder = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=spec.n_regime_states,
    )
    embed_dim = int(embedder.cfg.embed_dim)

    # --- Env + behaviour policy ---
    env_cfg = _build_env_cfg(cfg)
    env_cfg.window = chart_window
    env_cfg.image_size = image_size
    env = TradingEnv(df, cfg=env_cfg)
    rng = np.random.default_rng(seed + 1)
    if args.random_base or (cfg or {}).get("random_base", False):
        policy_fn = _random_policy_fn(rng)
    else:
        policy_fn = _policy_fn_from_model(embedder)

    # --- Collect real trajectories ---
    n_episodes = args.episodes or (int(cfg.get("episodes", 4)) if cfg else 4)
    max_steps = args.max_steps or (int(cfg.get("max_steps_per_episode", 200)) if cfg else 200)
    trajectories = collect_trajectories(env, policy_fn, n_episodes=n_episodes, max_steps=max_steps, seed=seed)
    total_steps = sum(len(t) for t in trajectories)
    if total_steps == 0:
        raise RuntimeError("No trajectory steps collected; check env and base policy.")
    trajectories = embed_trajectories(trajectories, embedder, device=device, batch_size=64)

    # --- Train World Model ---
    wm_cfg = WorldModelConfig(
        state_dim=embed_dim,
        n_actions=int(cfg.get("n_actions", 9)) if cfg else 9,
        dynamics_hidden=int(cfg.get("dynamics_hidden", 128)) if cfg else 128,
        dynamics_layers=int(cfg.get("dynamics_layers", 1)) if cfg else 1,
    )
    wm = WorldModel(wm_cfg)
    wm_trainer = WorldModelTrainer(wm, WorldModelTrainerConfig(
        learning_rate=float(cfg.get("wm_learning_rate", 1e-3)) if cfg else 1e-3,
        batch_size=int(cfg.get("wm_batch_size", 64)) if cfg else 64,
        epochs=args.wm_epochs or (int(cfg.get("wm_epochs", 3)) if cfg else 3),
        device=device, seed=seed, verbose=True,
    ))
    wm_dataset = WorldModelDataset(trajectories)
    wm_result = wm_trainer.fit(wm_dataset)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    wm_trainer.save(args.checkpoint, extra={
        "final_state_mse": wm_result.final_state_mse,
        "final_reward_mse": wm_result.final_reward_mse,
    })
    print(
        f"WorldModel trained. final_state_mse={wm_result.final_state_mse:.4f} "
        f"final_reward_mse={wm_result.final_reward_mse:.4f}"
    )

    # --- Dyna PPO in imagination ---
    n_actions = int(cfg.get("n_actions", 9)) if cfg else 9
    ac_cfg = LatentActorCriticConfig(
        state_dim=embed_dim, n_actions=n_actions,
        hidden_dim=int(cfg.get("ac_hidden", 64)) if cfg else 64,
        n_hidden_layers=int(cfg.get("ac_layers", 1)) if cfg else 1,
    )
    ac = LatentActorCritic(ac_cfg)
    horizon = args.horizon or (int(cfg.get("dyna_horizon", 16)) if cfg else 16)
    dyna_trainer = DynaPPOTrainer(wm, ac, DynaPPOConfig(
        horizon=horizon,
        n_imagined_rollouts=int(cfg.get("dyna_rollouts", 32)) if cfg else 32,
        ppo_epochs=int(cfg.get("ppo_epochs", 4)) if cfg else 4,
        ppo_minibatch_size=int(cfg.get("ppo_minibatch_size", 32)) if cfg else 32,
        learning_rate=float(cfg.get("dyna_learning_rate", 3e-4)) if cfg else 3e-4,
        gamma=float(cfg.get("dyna_gamma", 0.99)) if cfg else 0.99,
        device=device, seed=seed, verbose=True,
    ))
    # Build the initial-state pool from real states.
    pool_size = int(cfg.get("initial_pool_size", 64)) if cfg else 64
    pool_z, pool_h = _build_initial_pool(trajectories, embed_dim, wm_cfg.dynamics_hidden, wm_cfg.dynamics_layers, pool_size, seed=seed)
    # pool_h is (N, n_layers, H); trainer expects (n_layers, N, H).
    pool_h = pool_h.permute(1, 0, 2).contiguous()
    n_rounds = args.dyna_rounds or (int(cfg.get("dyna_rounds", 2)) if cfg else 2)
    last_summary: dict = {}
    sample_size = min(int(dyna_trainer.cfg.n_imagined_rollouts), int(pool_z.size(0)))
    for r in range(n_rounds):
        idx = torch.randperm(pool_z.size(0))[:sample_size]
        last_summary = dyna_trainer.update(pool_z[idx], pool_h[:, idx])
    Path(args.dyna_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    dyna_trainer.save(args.dyna_checkpoint, extra={
        "final_imagined_return": float(last_summary.get("imagined_return", 0.0)),
    })
    print(
        f"Dyna PPO complete. rounds={n_rounds} "
        f"final_imagined_return={float(last_summary.get('imagined_return', 0.0)):.4f}"
    )
    print(f"WM checkpoint: {args.checkpoint}")
    print(f"Dyna checkpoint: {args.dyna_checkpoint}")
    return 0


def _build_initial_pool(trajectories, state_dim: int, dyn_hidden: int, dyn_layers: int, pool_size: int, seed: int = 0):
    """Sample ``pool_size`` (z, h) pairs from real trajectory observations."""
    rng = np.random.default_rng(seed)
    flat: list[tuple[np.ndarray, np.ndarray]] = []
    for traj in trajectories:
        for o in traj.obs:
            emb = o.get("state_emb")
            if emb is None:
                continue
            h = np.zeros((dyn_layers, dyn_hidden), dtype=np.float32)
            flat.append((np.asarray(emb, dtype=np.float32), h))
    if not flat:
        raise RuntimeError("No real states with embeddings to seed the pool")
    idx = rng.choice(len(flat), size=min(pool_size, len(flat)), replace=len(flat) < pool_size)
    z = np.stack([flat[i][0] for i in idx], axis=0)
    h = np.stack([flat[i][1] for i in idx], axis=0)
    return torch.from_numpy(z).float(), torch.from_numpy(h).float()


if __name__ == "__main__":
    raise SystemExit(main())
