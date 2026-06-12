"""Objective functions that build + run a ZHISA trainer from sampled params.

Each objective accepts a *base* config (mostly budget knobs) plus a
:class:`SearchSpace` and an Optuna ``Trial``, and returns the
metric value to be maximised (we always maximise; a loss is
negated to be a reward).

These objective functions are deliberately lightweight: they
re-implement the small subset of each trainer's config that's
actually searched, so the HPO loop is fast and isolated from the
heavier, fully-featured CLI scripts.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optuna
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.expert import build_expert
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.data.trajectory import collect_trajectories
from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import build_default_policy
from zhisa.training.losses import LossWeights, MultiTaskLoss
from zhisa.training.optim import OptimConfig
from zhisa.training.s2b_imitation import BCConfig, BehavioralCloningTrainer
from zhisa.utils.seeding import set_seed

from zhisa.hpopt.search_space import SearchSpace


@dataclass
class ObjectiveResult:
    """The objective's return value.

    ``value`` is what Optuna sees; ``history`` is the trainer's own
    history dict (helpful for debugging or meta-learning), and
    ``params`` is the dict of values sampled this trial.
    """

    value: float
    history: list[dict[str, Any]]
    params: dict[str, Any]
    elapsed_s: float = 0.0


ObjectiveFn = Callable[[optuna.Trial, dict[str, Any]], ObjectiveResult]


def _to_float(x: Any) -> float:
    """Coerce numpy / torch scalars to Python float, with NaN->-inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(v) or math.isinf(v):
        return float("-inf")
    return v


def _make_optim(trial_params: dict[str, Any], base_cfg: dict[str, Any]) -> OptimConfig:
    """Build an :class:`OptimConfig` from sampled params + base."""
    lr = float(trial_params.get("lr", base_cfg.get("lr", 3e-4)))
    wd = float(trial_params.get("weight_decay", base_cfg.get("weight_decay", 1e-2)))
    return OptimConfig(lr=lr, weight_decay=wd)


def _make_loss_weights(base_cfg: dict[str, Any]) -> LossWeights:
    w = base_cfg.get("loss_weights", {}) or {}
    return LossWeights(
        direction=float(w.get("direction", 1.0)),
        volatility=float(w.get("volatility", 0.5)),
        regime=float(w.get("regime", 0.3)),
        return_pred=float(w.get("return_pred", 0.5)),
        policy=float(w.get("policy", 1.0)),
        value=float(w.get("value", 0.5)),
        uncertainty=float(w.get("uncertainty", 0.05)),
    )


def _probe_features(n_bars: int, chart_window: int, image_size: int, n_regime: int) -> tuple[int, int]:
    df = generate_market(MarketConfig(n_bars=max(int(n_bars), 200)))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size, n_regime_states=n_regime)
    ds = MarketDataset(df, spec=spec)
    # ``numeric`` carries OHLCV features only; cyclic time embeddings go
    # into ``context`` (the env's contract — and the policy's default).
    n_feat = ds._features.shape[1]
    n_ctx = ds._time_features.shape[1]
    return n_feat, n_ctx


def _default_device() -> str:
    import os
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def bc_objective(trial: optuna.Trial, base_cfg: dict[str, Any]) -> ObjectiveResult:
    """Objective for the BC trainer (s2b)."""
    space = base_cfg["space"]
    params = space.sample(trial)
    seed = int(base_cfg.get("seed", 0))
    set_seed(seed)

    n_bars = int(base_cfg.get("n_bars", 2000))
    chart_window = int(base_cfg.get("chart_window", 32))
    image_size = int(base_cfg.get("image_size", 32))
    n_regime = int(base_cfg.get("n_regime_states", 4))
    device = str(base_cfg.get("device", _default_device()))

    n_feat, n_ctx = _probe_features(n_bars, chart_window, image_size, n_regime)
    df = generate_market(MarketConfig(n_bars=n_bars))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size, n_regime_states=n_regime)
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=n_regime,
    )
    expert_kind = str(base_cfg.get("expert", "triple_barrier"))
    expert = build_expert(expert_kind, chart_window=chart_window)
    optim_cfg = _make_optim(params, base_cfg)
    loss_weights = _make_loss_weights(base_cfg)

    bc_cfg = BCConfig(
        epochs=int(params.get("epochs", 3)),
        batch_size=int(params.get("batch_size", 32)),
        grad_clip=float(params.get("grad_clip", 1.0)),
        log_every=int(base_cfg.get("log_every", 1000)),
        device=device, seed=seed, optim=optim_cfg, loss_weights=loss_weights,
        checkpoint=base_cfg.get("checkpoint"),
    )
    trainer = BehavioralCloningTrainer(model, MultiTaskLoss(loss_weights), bc_cfg)
    t0 = time.perf_counter()
    res = trainer.fit(df, expert, spec=spec)
    history = res["history"]
    last = history[-1]["loss"] if history else float("nan")
    value = -_to_float(last)
    return ObjectiveResult(value=value, history=history, params=params,
                           elapsed_s=time.perf_counter() - t0)


def dagger_objective(trial: optuna.Trial, base_cfg: dict[str, Any]) -> ObjectiveResult:
    """Objective for the DAgger trainer (s2b)."""
    space = base_cfg["space"]
    params = space.sample(trial)
    seed = int(base_cfg.get("seed", 0))
    set_seed(seed)

    n_bars = int(base_cfg.get("n_bars", 2000))
    chart_window = int(base_cfg.get("chart_window", 32))
    image_size = int(base_cfg.get("image_size", 32))
    n_regime = int(base_cfg.get("n_regime_states", 4))
    device = str(base_cfg.get("device", _default_device()))

    from zhisa.training.s2b_imitation import DAggerConfig, DAggerTrainer
    from zhisa.env.trading_env import TradingEnv

    n_feat, n_ctx = _probe_features(n_bars, chart_window, image_size, n_regime)
    df = generate_market(MarketConfig(n_bars=n_bars))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size, n_regime_states=n_regime)
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=n_regime,
    )
    expert_kind = str(base_cfg.get("expert", "triple_barrier"))
    expert = build_expert(expert_kind, chart_window=chart_window)
    optim_cfg = _make_optim(params, base_cfg)
    loss_weights = _make_loss_weights(base_cfg)
    env_cfg = EnvConfig(window=chart_window, image_size=image_size,
                        episode_length=int(base_cfg.get("max_steps_per_episode", 200)))

    dagger_cfg = DAggerConfig(
        n_rounds=int(params.get("n_rounds", 3)),
        epochs_per_round=1,
        rollout_episodes_per_round=int(params.get("rollout_episodes_per_round", 2)),
        max_steps_per_episode=int(base_cfg.get("max_steps_per_episode", 200)),
        batch_size=int(params.get("batch_size", 32)),
        grad_clip=float(params.get("grad_clip", 1.0)),
        log_every=int(base_cfg.get("log_every", 1000)),
        device=device, seed=seed, optim=optim_cfg, loss_weights=loss_weights,
        env_cfg=env_cfg, checkpoint=base_cfg.get("checkpoint"),
    )
    trainer = DAggerTrainer(model, expert, dagger_cfg)
    t0 = time.perf_counter()
    res = trainer.fit(df, spec=spec)
    value = -_to_float(res.final_loss)
    history = [{"round": r.round_idx, "loss": r.bc_loss,
                "n_aggregated": r.n_aggregated} for r in res.rounds]
    return ObjectiveResult(value=value, history=history, params=params,
                           elapsed_s=time.perf_counter() - t0)


def ppo_objective(trial: optuna.Trial, base_cfg: dict[str, Any]) -> ObjectiveResult:
    """Objective for the S4 PPO trainer."""
    space = base_cfg["space"]
    params = space.sample(trial)
    seed = int(base_cfg.get("seed", 0))
    set_seed(seed)

    n_bars = int(base_cfg.get("n_bars", 1500))
    chart_window = int(base_cfg.get("chart_window", 32))
    image_size = int(base_cfg.get("image_size", 32))
    n_regime = int(base_cfg.get("n_regime_states", 4))
    device = str(base_cfg.get("device", _default_device()))

    from zhisa.training.s4_rl import PPOConfig, PPOTrainer

    n_feat, n_ctx = _probe_features(n_bars, chart_window, image_size, n_regime)
    df = generate_market(MarketConfig(n_bars=n_bars, seed=seed))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size, n_regime_states=n_regime)
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=n_regime,
    )
    optim_cfg = _make_optim(params, base_cfg)
    env_cfg = EnvConfig(window=chart_window, image_size=image_size,
                        episode_length=int(base_cfg.get("max_steps_per_episode", 200)))
    ppo_cfg = PPOConfig(
        n_episodes=int(base_cfg.get("n_episodes", 2)),
        max_steps_per_episode=int(base_cfg.get("max_steps_per_episode", 200)),
        n_epochs=int(params.get("n_epochs", 4)),
        minibatch_size=int(params.get("minibatch_size", 32)),
        clip_ratio=float(params.get("clip_ratio", 0.2)),
        value_coef=float(params.get("value_coef", 0.5)),
        entropy_coef=float(params.get("entropy_coef", 0.01)),
        gamma=float(params.get("gamma", 0.99)),
        gae_lambda=float(params.get("gae_lambda", 0.95)),
        grad_clip=float(base_cfg.get("grad_clip", 1.0)),
        target_kl=float(params.get("target_kl", 0.05)),
        device=device, optim=optim_cfg, env_cfg=env_cfg, seed=seed,
        log_every=int(base_cfg.get("log_every", 1)),
    )
    trainer = PPOTrainer(model, ppo_cfg)
    t0 = time.perf_counter()
    res = trainer.fit(df)
    history = res["history"]
    last = history[-1].get("episode_return", 0.0) if history else 0.0
    value = _to_float(last)
    return ObjectiveResult(value=value, history=history, params=params,
                           elapsed_s=time.perf_counter() - t0)


def s4_cvar_objective(trial: optuna.Trial, base_cfg: dict[str, Any]) -> ObjectiveResult:
    """Objective for the CVaR-Constrained PPO trainer."""
    space = base_cfg["space"]
    params = space.sample(trial)
    seed = int(base_cfg.get("seed", 0))
    set_seed(seed)

    n_bars = int(base_cfg.get("n_bars", 1500))
    chart_window = int(base_cfg.get("chart_window", 32))
    image_size = int(base_cfg.get("image_size", 32))
    n_regime = int(base_cfg.get("n_regime_states", 4))
    device = str(base_cfg.get("device", _default_device()))

    from zhisa.training.cvar_ppo import CVaRPPOConfig, CVaRPPOTrainer

    n_feat, n_ctx = _probe_features(n_bars, chart_window, image_size, n_regime)
    df = generate_market(MarketConfig(n_bars=n_bars, seed=seed))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size, n_regime_states=n_regime)
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=n_regime,
    )
    optim_cfg = _make_optim(params, base_cfg)
    env_cfg = EnvConfig(window=chart_window, image_size=image_size,
                        episode_length=int(base_cfg.get("max_steps_per_episode", 200)))
    trainer_cfg = CVaRPPOConfig(
        n_iterations=int(base_cfg.get("n_iterations", 2)),
        n_episodes=int(base_cfg.get("n_episodes", 2)),
        max_steps_per_episode=int(base_cfg.get("max_steps_per_episode", 200)),
        n_epochs=int(params.get("n_epochs", 4)),
        minibatch_size=int(params.get("minibatch_size", 64)),
        clip_ratio=float(params.get("clip_ratio", 0.2)),
        value_coef=float(params.get("value_coef", 0.5)),
        entropy_coef=float(params.get("entropy_coef", 0.01)),
        gamma=float(params.get("gamma", 0.99)),
        gae_lambda=float(params.get("gae_lambda", 0.95)),
        grad_clip=float(base_cfg.get("grad_clip", 1.0)),
        target_kl=float(params.get("target_kl", 0.05)),
        cvar_alpha=float(params.get("cvar_alpha", 0.1)),
        cvar_threshold=float(base_cfg.get("cvar_threshold", 0.1)),
        cvar_lambda_init=float(params.get("cvar_lambda_init", 0.0)),
        cvar_lambda_lr=float(params.get("cvar_lr", 0.05)),
        cvar_lambda_max=float(params.get("cvar_max", 100.0)),
        env_cfg=env_cfg, device=device, seed=seed,
        log_every=int(base_cfg.get("log_every", 1)),
    )
    trainer = CVaRPPOTrainer(model, trainer_cfg)
    t0 = time.perf_counter()
    trainer.fit(df)
    history = trainer.cvar_history
    last_cvar = history[-1]["cvar"] if history else 0.0
    last_lambda = history[-1]["lambda_cvar"] if history else 0.0
    # We reward improvements in CVaR (higher is better — less negative)
    # while keeping the dual multiplier low (lighter penalty).
    value = _to_float(-last_cvar) - 0.01 * _to_float(last_lambda)
    return ObjectiveResult(value=value, history=history, params=params,
                           elapsed_s=time.perf_counter() - t0)


def dt_objective(trial: optuna.Trial, base_cfg: dict[str, Any]) -> ObjectiveResult:
    """Objective for the Decision Transformer (S6)."""
    space = base_cfg["space"]
    params = space.sample(trial)
    seed = int(base_cfg.get("seed", 0))
    set_seed(seed)

    n_bars = int(base_cfg.get("n_bars", 1500))
    n_episodes = int(base_cfg.get("n_episodes", 2))
    max_steps = int(base_cfg.get("max_steps_per_episode", 100))
    chart_window = int(base_cfg.get("chart_window", 16))
    image_size = int(base_cfg.get("image_size", 32))
    n_regime = int(base_cfg.get("n_regime_states", 4))
    device = str(base_cfg.get("device", _default_device()))

    from zhisa.training.decision_transformer import (
        DTConfig, DecisionTransformerTrainer, build_default_dt, embed_trajectories,
    )
    from zhisa.models.policy import PolicyConfig
    from zhisa.data.trajectory import TrajectoryWindowDataset

    n_feat, n_ctx = _probe_features(n_bars, chart_window, image_size, n_regime)
    df = generate_market(MarketConfig(n_bars=n_bars))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size, n_regime_states=n_regime)
    env_cfg = EnvConfig(window=chart_window, image_size=image_size,
                        episode_length=max_steps)
    env = TradingEnv(df, cfg=env_cfg)

    embedder = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=n_regime,
    )

    def _policy_fn(obs):
        with torch.no_grad():
            chart = torch.from_numpy(np.asarray(obs["chart"], dtype=np.float32)).unsqueeze(0)
            numeric = torch.from_numpy(np.asarray(obs["numeric"], dtype=np.float32)).unsqueeze(0)
            context = torch.from_numpy(np.asarray(obs["context"], dtype=np.float32)).unsqueeze(0)
            out = embedder(chart, numeric, context)
            logits = out.get("policy_logits")
            return int(logits.argmax(dim=-1).item())

    rng = np.random.default_rng(seed + 1)
    trajs = collect_trajectories(env, _policy_fn, n_episodes=n_episodes,
                                 max_steps=max_steps, seed=seed)
    if sum(len(t) for t in trajs) == 0:
        return ObjectiveResult(value=float("-inf"), history=[], params=params)
    trajs = embed_trajectories(trajs, embedder, device=device, batch_size=64)

    context_length = int(params.get("context_len", 16))
    dataset = TrajectoryWindowDataset(trajs, context_length=context_length,
                                      gamma=1.0, n_actions=9)

    embed_dim = int(getattr(embedder.cfg, "embed_dim", 128))
    pcfg = PolicyConfig(image_size=image_size, in_numeric_features=n_feat,
                        window=chart_window, in_context_features=n_ctx,
                        embed_dim=embed_dim, n_actions=9, n_regime_classes=n_regime)
    dt_model, dt_cfg = build_default_dt(
        pcfg, DTConfig(
            state_dim=embed_dim, n_actions=9, context_length=context_length,
            d_model=int(base_cfg.get("d_model", 64)),
            n_heads=int(base_cfg.get("n_heads", 4)),
            n_layers=int(base_cfg.get("n_layers", 2)),
            dropout=0.1,
            learning_rate=float(params.get("lr", 3e-4)),
            batch_size=int(params.get("batch_size", 16)),
            epochs=int(params.get("epochs", 2)),
            rtg_loss_weight=float(params.get("rtg_coef", 0.0)),
            max_rtg_clip=10.0, device=device, seed=seed, verbose=False,
        ),
    )
    trainer = DecisionTransformerTrainer(dt_model, dt_cfg)
    t0 = time.perf_counter()
    res = trainer.fit(dataset)
    value = -_to_float(res.final_loss)
    return ObjectiveResult(value=value, history=[], params=params,
                           elapsed_s=time.perf_counter() - t0)


def world_model_objective(trial: optuna.Trial, base_cfg: dict[str, Any]) -> ObjectiveResult:
    """Objective for the World Model + Dyna (S7).

    The metric is the negative of the final Dyna imagined return —
    higher imagined return => better policy.
    """
    space = base_cfg["space"]
    params = space.sample(trial)
    seed = int(base_cfg.get("seed", 0))
    set_seed(seed)

    n_bars = int(base_cfg.get("n_bars", 1500))
    n_episodes = int(base_cfg.get("n_episodes", 2))
    max_steps = int(base_cfg.get("max_steps_per_episode", 100))
    chart_window = int(base_cfg.get("chart_window", 16))
    image_size = int(base_cfg.get("image_size", 32))
    n_regime = int(base_cfg.get("n_regime_states", 4))
    device = str(base_cfg.get("device", _default_device()))

    from zhisa.models.latent_actor_critic import LatentActorCritic, LatentActorCriticConfig
    from zhisa.models.world_model import WorldModel, WorldModelConfig
    from zhisa.training.decision_transformer import embed_trajectories
    from zhisa.training.dyna_ppo import DynaPPOConfig, DynaPPOTrainer
    from zhisa.training.world_model_trainer import (
        WorldModelDataset, WorldModelTrainer, WorldModelTrainerConfig,
    )

    n_feat, n_ctx = _probe_features(n_bars, chart_window, image_size, n_regime)
    df = generate_market(MarketConfig(n_bars=n_bars))
    spec = SampleSpec(chart_window=chart_window, feature_window=chart_window,
                      image_size=image_size, n_regime_states=n_regime)
    env_cfg = EnvConfig(window=chart_window, image_size=image_size,
                        episode_length=max_steps)
    env = TradingEnv(df, cfg=env_cfg)

    embedder = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=chart_window, image_size=image_size,
        n_actions=9, n_regime_classes=n_regime,
    )
    embed_dim = int(embedder.cfg.embed_dim)

    rng = np.random.default_rng(seed + 1)
    def _policy_fn(obs):
        return int(rng.integers(0, 9))
    trajs = collect_trajectories(env, _policy_fn, n_episodes=n_episodes,
                                 max_steps=max_steps, seed=seed)
    if sum(len(t) for t in trajs) == 0:
        return ObjectiveResult(value=float("-inf"), history=[], params=params)
    trajs = embed_trajectories(trajs, embedder, device=device, batch_size=64)

    wm_cfg = WorldModelConfig(state_dim=embed_dim, n_actions=9,
                              dynamics_hidden=64, dynamics_layers=1)
    wm = WorldModel(wm_cfg)
    wm_trainer = WorldModelTrainer(wm, WorldModelTrainerConfig(
        learning_rate=float(params.get("lr", 3e-4)),
        batch_size=64, epochs=int(params.get("epochs", 2)),
        device=device, seed=seed, verbose=False,
    ))
    wm_trainer.fit(WorldModelDataset(trajs))

    ac_cfg = LatentActorCriticConfig(state_dim=embed_dim, n_actions=9,
                                     hidden_dim=64, n_hidden_layers=1)
    ac = LatentActorCritic(ac_cfg)
    horizon = int(params.get("rollout_horizon", 8))
    dyna_trainer = DynaPPOTrainer(wm, ac, DynaPPOConfig(
        horizon=horizon, n_imagined_rollouts=16, ppo_epochs=2,
        ppo_minibatch_size=16, learning_rate=float(params.get("lr", 3e-4)),
        gamma=0.99, device=device, seed=seed, verbose=False,
    ))

    pool_z, pool_h = [], []
    for traj in trajs:
        for o in traj.obs:
            if o.get("state_emb") is not None:
                pool_z.append(np.asarray(o["state_emb"], dtype=np.float32))
                pool_h.append(np.zeros((1, 64), dtype=np.float32))
    if not pool_z:
        return ObjectiveResult(value=float("-inf"), history=[], params=params)
    z = torch.from_numpy(np.stack(pool_z)).float()
    h = torch.from_numpy(np.stack(pool_h)).float().permute(1, 0, 2).contiguous()
    sample = min(int(dyna_trainer.cfg.n_imagined_rollouts), int(z.size(0)))
    t0 = time.perf_counter()
    summary = dyna_trainer.update(z[:sample], h[:, :sample])
    value = _to_float(summary.get("imagined_return", 0.0))
    return ObjectiveResult(value=value, history=[summary], params=params,
                           elapsed_s=time.perf_counter() - t0)


def portfolio_ppo_objective(trial: optuna.Trial, base_cfg: dict[str, Any]) -> ObjectiveResult:
    """Objective for the multi-instrument portfolio PPO (Stage 1)."""
    space = base_cfg["space"]
    params = space.sample(trial)
    seed = int(base_cfg.get("seed", 0))
    set_seed(seed)

    from zhisa.env.portfolio_env import PortfolioConfig, PortfolioEnv
    from zhisa.models.portfolio_policy import (
        PortfolioPolicyConfig, PortfolioPolicyNetwork,
    )
    from zhisa.training.portfolio_ppo import PortfolioPPOConfig, PortfolioPPOTrainer

    n_instruments = int(params.get("n_instruments", 3))
    if n_instruments < 2:
        n_instruments = 2
    bars = int(base_cfg.get("n_bars", 300))
    chart_window = int(base_cfg.get("chart_window", 16))
    image_size = int(base_cfg.get("image_size", 32))
    episode_length = int(base_cfg.get("max_steps_per_episode", 30))
    device = str(base_cfg.get("device", _default_device()))
    gross_cap = float(params.get("max_gross_leverage", 1.5))

    data = {f"i{i}": generate_market(MarketConfig(n_bars=bars, seed=seed + i))
            for i in range(n_instruments)}
    env_cfg = EnvConfig(window=chart_window, image_size=image_size,
                        episode_length=episode_length)
    pcfg = PortfolioConfig(
        n_instruments=n_instruments,
        instrument_names=list(data.keys()),
        env_cfg=env_cfg, gross_leverage_cap=gross_cap,
    )
    probe_env = PortfolioEnv(data, cfg=pcfg)
    obs, _ = probe_env.reset()
    inst0 = obs["instruments"][0]
    in_numeric = int(inst0["numeric"].shape[-1])
    in_context = int(inst0["context"].shape[-1])
    portfolio_dim = int(obs["portfolio"].shape[0])

    model = PortfolioPolicyNetwork(PortfolioPolicyConfig(
        n_instruments=n_instruments, in_numeric_features=in_numeric,
        in_context_features=in_context, window=chart_window,
        image_size=image_size, embed_dim=32, fusion_hidden=32,
        portfolio_dim=portfolio_dim,
    ))
    optim_cfg = _make_optim(params, base_cfg)
    ppo_cfg = PortfolioPPOConfig(
        n_instruments=n_instruments, n_iterations=int(base_cfg.get("n_iterations", 1)),
        n_episodes=int(base_cfg.get("n_episodes", 1)),
        max_steps_per_episode=episode_length, n_epochs=int(params.get("n_epochs", 2)),
        minibatch_size=int(params.get("minibatch_size", 16)),
        clip_ratio=float(params.get("clip_ratio", 0.2)),
        value_coef=float(params.get("value_coef", 0.5)),
        entropy_coef=float(params.get("entropy_coef", 0.01)),
        gamma=float(params.get("gamma", 0.99)),
        gae_lambda=float(params.get("gae_lambda", 0.95)),
        grad_clip=float(base_cfg.get("grad_clip", 0.5)),
        log_every=1, device=device, seed=seed, portfolio_dim=portfolio_dim,
    )
    ppo_cfg.optim = optim_cfg
    ppo_cfg.env_cfg = env_cfg
    trainer = PortfolioPPOTrainer(model, ppo_cfg)
    t0 = time.perf_counter()
    out = trainer.fit(data, env_cfg=pcfg)
    history = out.get("history", [])
    last = history[-1].get("mean_return", 0.0) if history else 0.0
    return ObjectiveResult(value=_to_float(last), history=history, params=params,
                           elapsed_s=time.perf_counter() - t0)
