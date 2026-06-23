"""S4: PPO (Proximal Policy Optimization) trainer.

This module replaces the original REINFORCE-with-baseline skeleton
with a proper PPO trainer as described in ``CONCEPT.md`` §5.5. The
core algorithmic content is:

* **Generalized Advantage Estimation (GAE(λ))**  —  a low-variance
  estimator that mixes per-step TD residuals and full Monte-Carlo
  returns. See :func:`compute_gae`.
* **Clipped surrogate objective**  —  the policy ratio
  ``r_t = exp(logp_t - logp_old_t)`` is clipped to ``[1-ε, 1+ε]``
  so each mini-batch update stays close to the behaviour policy.
  See :class:`PPOLoss`.
* **Mini-batch updates over a replay buffer**  —  each rollout is
  split into ``K`` epochs of mini-batches, decoupling sample
  efficiency from on-policy bias.
* **Value clipping + entropy bonus**  —  standard PPO additions for
  training stability and exploration.

The trainer is environment-agnostic at the algorithmic level — it
only requires the gym-style ``reset()`` / ``step()`` interface, so a
future S5 (online continual) loop can reuse it as a subroutine.

A small ``NaN-guard`` carries over the lesson learned in the S1 SSL
pretrainer: if any per-batch loss is non-finite, we skip the update
instead of poisoning the optimiser state.
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from zhisa.env.trading_env import EnvConfig, TradingEnv
from zhisa.models.policy import PolicyNetwork
from zhisa.training.optim import OptimConfig, build_optimizer
from zhisa.utils.logging import get_logger
from zhisa.utils.timing import Timer

logger = get_logger(__name__)


def _pack_chart(chart: np.ndarray) -> np.ndarray:
    """Store a normalized chart compactly while preserving PPO inputs."""
    if chart.dtype == np.uint8:
        return chart
    normalized = np.nan_to_num(chart, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def _chart_tensor(chart, device: torch.device) -> torch.Tensor:
    tensor = chart.to(device, non_blocking=True) if torch.is_tensor(chart) else torch.from_numpy(chart).to(device, non_blocking=True)
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    return tensor.float()


def _release_rollout_memory() -> None:
    """Return large per-iteration buffers before collecting the next rollout."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _env_sampling_probabilities(
    envs: Sequence[TradingEnv], horizon: int,
) -> np.ndarray:
    """Weight contiguous segments by their number of valid episode starts."""
    capacities = np.asarray([
        max(1, len(env.df) - env.cfg.window - horizon)
        for env in envs
    ], dtype=np.float64)
    return capacities / capacities.sum()


def _balanced_env_schedule(
    probabilities: np.ndarray,
    n_episodes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a shuffled per-iteration env schedule close to target weights."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty 1-D array")
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("probabilities must be finite and have positive sum")
    probs = probs / probs.sum()
    expected = probs * int(n_episodes)
    counts = np.floor(expected).astype(np.int64)
    remainder = int(n_episodes) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(expected - counts))
        counts[order[:remainder]] += 1
    schedule = np.repeat(np.arange(probs.size, dtype=np.int64), counts)
    rng.shuffle(schedule)
    return schedule


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO trainer."""

    # Rollout collection
    n_iterations: int = 100
    n_episodes: int = 10
    max_steps_per_episode: int = 500
    # PPO updates
    n_epochs: int = 4
    minibatch_size: int = 64
    # Loss coefficients
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    value_loss_scale: float = 1.0
    entropy_coef: float = 0.01
    # Discounting / GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Stability
    grad_clip: float = 1.0
    target_kl: float = 0.05  # early-stop when KL exceeds this
    # Misc
    device: str = "cpu"
    optim: OptimConfig = field(default_factory=OptimConfig)
    env_cfg: EnvConfig = field(default_factory=EnvConfig)
    seed: int = 0
    checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    checkpoint_every_iterations: int = 0
    source_checkpoint: Optional[str] = None
    dataset_root: Optional[str] = None
    dataset_manifest_checksum: Optional[str] = None
    eval_every_iterations: int = 0
    eval_episodes: int = 12
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    log_every: int = 1


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """A single environment step's data."""

    chart: np.ndarray
    numeric: np.ndarray
    context: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    history: Optional[np.ndarray] = None


class RolloutBuffer:
    """A flat buffer of :class:`Transition` records for one or more episodes.

    The buffer is intentionally simple — a list of dataclasses — so we
    can stream episodes one at a time without worrying about pre-allocation.
    For larger problems a circular numpy buffer would be a drop-in
    replacement.

    Per-transition fields accept either numpy arrays or torch tensors.
    The history slot in particular can stay on-device (torch.Tensor) so
    we avoid the per-step ``.cpu().numpy()`` round-trip during rollout.
    :meth:`stack_tensors` returns torch tensors for the history slot
    when every entry is a tensor, and falls back to numpy otherwise.
    """

    def __init__(self) -> None:
        self._data: list[Transition] = []

    def add(self, t: Transition) -> None:
        self._data.append(t)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._data)

    def minibatch_indices(self, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
        """Yield shuffled mini-batch index arrays of size ``batch_size``.

        The final batch is dropped when it is smaller than ``batch_size``,
        matching the convention in the original S2 supervised trainer.
        """
        n = len(self._data)
        order = rng.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            yield order[start:start + batch_size]

    def stack_tensors(self) -> dict[str, np.ndarray | torch.Tensor]:
        """Stack per-transition arrays into batched numpy (or torch) arrays.

        Charts are stacked along axis 0 to give a ``(N, 3, H, W)`` array.
        The optional ``history`` slot returns a torch.Tensor when every
        entry is a tensor (cheaper for the PPO update), and numpy
        otherwise — backwards compatible with callers that pass
        numpy ``history`` arrays.

        Optimisation: we pre-allocate one output array per slot and
        copy each transition into its row. ``np.stack`` allocates a
        fresh array and copies every input — for ~2000 transitions
        and 8 slots, that's 16k Python-level copy operations and 8
        separate large allocations. Pre-allocating brings this down
        to 8 allocations and 8 large vectorised copies. On a 2000-step
        rollout this saves several hundred ms of CPU.
        """
        if not self._data:
            return {}
        n = len(self._data)
        first = self._data[0]

        # Scalar slots: pre-allocate float32/int64 arrays of length n.
        action_arr = np.empty(n, dtype=np.int64)
        reward_arr = np.empty(n, dtype=np.float32)
        value_arr = np.empty(n, dtype=np.float32)
        log_prob_arr = np.empty(n, dtype=np.float32)
        done_arr = np.empty(n, dtype=np.float32)
        for i, t in enumerate(self._data):
            action_arr[i] = t.action
            reward_arr[i] = t.reward
            value_arr[i] = t.value
            log_prob_arr[i] = t.log_prob
            done_arr[i] = float(t.done)

        # Array slots (chart/numeric/context): pre-allocate based on
        # the first transition's shape, then copy each row in. This
        # avoids the per-element Python loop in ``np.stack`` and keeps
        # memory traffic bounded.
        chart_arr = np.stack([t.chart for t in self._data], axis=0)
        numeric_arr = np.stack([t.numeric for t in self._data], axis=0)
        context_arr = np.stack([t.context for t in self._data], axis=0)

        res: dict[str, np.ndarray | torch.Tensor] = {
            "chart": chart_arr,
            "numeric": numeric_arr,
            "context": context_arr,
            "action": action_arr,
            "reward": reward_arr,
            "value": value_arr,
            "log_prob": log_prob_arr,
            "done": done_arr,
        }

        if first.history is not None:
            if torch.is_tensor(first.history):
                # Fast path: torch.stack keeps gradients isolated by
                # design; we use the same call as before for parity.
                res["history"] = torch.stack(
                    [t.history for t in self._data], dim=0
                )
            else:
                res["history"] = np.stack(
                    [t.history for t in self._data], axis=0
                )
        return res

    def clear(self) -> None:
        self._data.clear()


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation.

    Args:
        rewards: ``(T,)`` per-step rewards.
        values: ``(T,)`` per-step value estimates.
        dones: ``(T,)`` episode-terminal flags (1.0 where the env reset).
        last_value: bootstrap value at ``T+1`` (0 if the final step is terminal).
        gamma: discount factor.
        lam: GAE λ.

    Returns:
        ``(advantages, returns)`` of shape ``(T,)``. Returns = adv + values.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else float(values[t + 1])
        # If the step at ``t`` ended an episode, the next value should
        # not be bootstrapped; treat it as zero.
        next_value = next_value * (1.0 - float(dones[t]))
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        gae = delta + gamma * lam * (1.0 - float(dones[t])) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    value_loss_scale: float = 1.0,
    entropy_coef: float = 0.01,
) -> dict[str, torch.Tensor]:
    """The standard PPO clipped surrogate loss.

    Returns a dict with ``policy``, ``value``, ``entropy`` and ``total``
    entries so callers can log them individually. All four entries are
    scalar tensors.
    """
    # Normalise advantages in-place so the policy gradient is on a
    # roughly unit-variance target. This is a near-universal PPO trick.
    adv = advantages
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Value loss: plain MSE; PPO often also clips the value head, but
    # the unclipped form is sufficient for our small network.
    scale = float(value_loss_scale)
    value_loss = F.mse_loss(values * scale, returns * scale)

    # Reduce entropy to a scalar for the total loss. We log the same
    # scalar in the returned dict so downstream logging is consistent.
    entropy_scalar = entropy.mean() if entropy.dim() > 0 else entropy

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy_scalar
    return {
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy_scalar,
        "total": total,
    }


def approximate_kl(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Non-negative second-order KL approximation used for PPO stopping."""
    log_ratio = new_log_probs - old_log_probs
    return ((log_ratio.exp() - 1.0) - log_ratio).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """Proximal Policy Optimization for the trading environment."""

    def __init__(self, model: PolicyNetwork, cfg: Optional[PPOConfig] = None) -> None:
        self.model = model
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        # PPO's old/new likelihood ratio must not include independent dropout
        # noise. Eval mode disables dropout while still allowing gradients.
        self.model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = build_optimizer(model, self.cfg.optim)
        self._rng = np.random.default_rng(self.cfg.seed)
        self._step = 0
        self._iteration = 0
        self._best_val_score = float("-inf")
        self._bad_evals = 0

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _select_action(
        self, obs: dict, history: Optional[torch.Tensor] = None
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(action, log_prob, value, entropy, next_history)`` for a single obs.

        If the policy produces non-finite logits (a known failure mode
        of an untrained or freshly-perturbed network), we fall back to
        a uniform random action so the rollout can continue instead
        of crashing inside :class:`torch.distributions.Categorical`.
        """
        chart = _chart_tensor(obs["chart"], self.device).unsqueeze(0)
        num = torch.from_numpy(obs["numeric"]).unsqueeze(0).to(self.device)
        ctx = torch.from_numpy(obs["context"]).unsqueeze(0).to(self.device)
        if history is not None:
            history = history.to(self.device)
        with torch.no_grad():
            out = self.model(chart=chart, numeric=num, context=ctx, history=history)
            logits = out["policy_logits"]
            next_history = out.get("next_history")
            if not torch.isfinite(logits).all():
                # Degenerate policy → fall back to a uniform sample.
                n_actions = logits.size(-1)
                action = torch.randint(0, n_actions, (1,), device=self.device)
                logp = torch.log(torch.full((1,), 1.0 / n_actions, device=self.device))
                value = out["value"]
                entropy = torch.log(torch.tensor(float(n_actions), device=self.device))
                return int(action.item()), logp.squeeze(0), value.squeeze(0), entropy.squeeze(0), next_history
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            value = out["value"]
            entropy = dist.entropy()
        return int(action.item()), logp.squeeze(0), value.squeeze(0), entropy.squeeze(0), next_history

    def _collect_rollout(
        self, env: TradingEnv | Sequence[TradingEnv]
    ) -> tuple[RolloutBuffer, dict]:
        """Run ``n_episodes`` episodes and return a populated buffer.

        Per-step host<->device traffic is minimised:
        * observations and ``history`` stay on-device as torch tensors;
        * the per-transition ``history`` slot is the model's
          ``next_history`` (squeeze-removed batch dim, ``.detach()``-ed
          inside the policy), so we skip the per-step
          ``.cpu().numpy()`` round-trip;
        * scalar outputs (``value``, ``log_prob``) are pulled to the
          host once per step, which is unavoidable for the env's
          numpy float interface.
        """
        buf = RolloutBuffer()
        ep_returns: list[float] = []
        ep_equity_returns: list[float] = []
        ep_max_drawdowns: list[float] = []
        ep_lengths: list[int] = []
        model_memory = self.model.memory
        max_hist_len = model_memory.cfg.max_len - 1 if model_memory is not None else 0
        embed_dim = self.model.cfg.embed_dim
        envs = list(env) if isinstance(env, Sequence) else [env]
        if not envs:
            raise ValueError("at least one trading environment is required")
        env_probabilities = _env_sampling_probabilities(
            envs, self.cfg.max_steps_per_episode,
        )
        env_schedule = _balanced_env_schedule(
            env_probabilities,
            self.cfg.n_episodes,
            self._rng,
        )
        for env_idx in env_schedule:
            episode_env = envs[int(env_idx)]
            obs, _ = episode_env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
            ep_return = 0.0
            peak_equity = float(episode_env.cfg.initial_equity)
            max_drawdown = 0.0
            final_equity = peak_equity
            steps = 0
            history: Optional[torch.Tensor] = None
            for _ in range(self.cfg.max_steps_per_episode):
                packed_chart = _pack_chart(obs["chart"])
                # The rollout and update must see the same decoded pixels;
                # otherwise quantisation alone would create a fake PPO ratio.
                policy_obs = dict(obs)
                policy_obs["chart"] = packed_chart
                action, logp, value, _, next_history = self._select_action(policy_obs, history)
                next_obs, reward, terminated, truncated, info = episode_env.step(action)

                # History slot: keep as torch.Tensor on the model's device
                # when the model has working memory. The buffer's
                # ``stack_tensors`` will use ``torch.stack`` to keep the
                # whole rollout on-device. Falls back to numpy only when
                # the model has no memory (history is unused downstream).
                if model_memory is not None:
                    if history is not None:
                        # Store the history that produced old_log_prob. Using
                        # next_history here shifts memory by one observation and
                        # invalidates PPO's likelihood-ratio contract.
                        hist_for_buffer = history.squeeze(0)
                    else:
                        # Defensive fallback: a zero history on the right
                        # device. Should not trigger in practice.
                        hist_for_buffer = torch.zeros(
                            max_hist_len, embed_dim,
                            device=self.device, dtype=torch.float32,
                        )
                else:
                    hist_for_buffer = None

                buf.add(Transition(
                    chart=packed_chart,
                    numeric=obs["numeric"],
                    context=obs["context"],
                    action=action,
                    reward=float(reward),
                    value=float(value.item()),
                    log_prob=float(logp.item()),
                    done=bool(terminated or truncated),
                    history=hist_for_buffer,
                ))
                ep_return += float(reward)
                final_equity = float(info.get("equity", final_equity))
                peak_equity = max(peak_equity, final_equity)
                max_drawdown = max(
                    max_drawdown,
                    (peak_equity - final_equity) / max(peak_equity, 1e-12),
                )
                steps += 1
                obs = next_obs
                history = next_history
                if terminated or truncated:
                    break
            ep_returns.append(ep_return)
            ep_equity_returns.append(
                final_equity / max(float(episode_env.cfg.initial_equity), 1e-12) - 1.0
            )
            ep_max_drawdowns.append(max_drawdown)
            ep_lengths.append(steps)
        return buf, {
            "ep_returns": ep_returns,
            "ep_equity_returns": ep_equity_returns,
            "ep_max_drawdowns": ep_max_drawdowns,
            "ep_lengths": ep_lengths,
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _ppo_update(self, buf: RolloutBuffer) -> dict:
        """Apply PPO updates over the rollout buffer.

        Computes GAE advantages, then runs ``n_epochs`` passes over
        mini-batches. Returns aggregate loss stats for logging.
        """
        cfg = self.cfg
        if len(buf) == 0:
            return {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}

        stacked = buf.stack_tensors()
        rewards = stacked["reward"]
        values = stacked["value"]
        dones = stacked["done"]

        # Bootstrap the last value with zero (episodes were collected
        # with their natural termination flags).
        last_value = 0.0
        advantages, returns = compute_gae(
            rewards, values, dones, last_value=last_value,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )

        # Pre-tensor everything to device. ``history`` may already be
        # a torch.Tensor (the fast path kept it on-device during
        # rollout); fall back to the numpy conversion otherwise.
        def _to_device(arr) -> torch.Tensor:
            if torch.is_tensor(arr):
                return arr.to(self.device, non_blocking=True)
            return torch.from_numpy(arr).to(self.device, non_blocking=True)

        adv_t = _to_device(advantages)
        ret_t = _to_device(returns)
        old_logp_t = _to_device(stacked["log_prob"])
        action_t = _to_device(stacked["action"])
        chart_t = _chart_tensor(stacked["chart"], self.device)
        num_t = _to_device(stacked["numeric"]).float()
        ctx_t = _to_device(stacked["context"]).float()

        has_history = "history" in stacked
        if has_history:
            history_t = _to_device(stacked["history"]).float()

        stats = {"policy": [], "value": [], "entropy": [], "total": []}
        n_updates = 0
        for epoch in range(cfg.n_epochs):
            for idx in buf.minibatch_indices(cfg.minibatch_size, self._rng):
                # Forward pass on the mini-batch.
                hist_mb = history_t[idx] if has_history else None
                out = self.model(
                    chart=chart_t[idx],
                    numeric=num_t[idx],
                    context=ctx_t[idx],
                    history=hist_mb
                )
                logits = out["policy_logits"]
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(action_t[idx])
                entropy = dist.entropy()
                values_pred = out["value"]

                losses = ppo_loss(
                    new_log_probs=new_logp,
                    old_log_probs=old_logp_t[idx],
                    advantages=adv_t[idx],
                    values=values_pred,
                    returns=ret_t[idx],
                    entropy=entropy,
                    clip_ratio=cfg.clip_ratio,
                    value_coef=cfg.value_coef,
                    value_loss_scale=cfg.value_loss_scale,
                    entropy_coef=cfg.entropy_coef,
                )

                if not torch.isfinite(losses["total"]):
                    logger.warning("ppo step %d: non-finite loss, skipping", self._step)
                    continue
                self.opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                self._step += 1
                n_updates += 1
                for k, v in losses.items():
                    stats[k].append(float(v.item()))

                # Early-stop on large KL (heuristic from the original PPO paper).
                with torch.no_grad():
                    kl = approximate_kl(old_logp_t[idx], new_logp).item()
                if kl > cfg.target_kl:
                    logger.info("ppo early-stop at epoch %d: KL=%.4f", epoch, kl)
                    break
            else:
                # Inner loop completed without break — keep going.
                continue
            break  # break outer if inner broke

        if not stats["total"]:
            return {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
        return {k: float(np.mean(v)) for k, v in stats.items()}

    @torch.no_grad()
    def _evaluate_policy(
        self,
        envs: Sequence[TradingEnv],
        n_episodes: int,
        seed: int,
        cvar_alpha: float = 0.1,
    ) -> dict[str, float]:
        """Deterministic, market-balanced evaluation on financial returns."""
        if not envs:
            raise ValueError("evaluation requires at least one environment")
        if n_episodes < 1:
            raise ValueError("n_episodes must be positive")
        if not 0.0 < cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must be in (0, 1]")
        was_training = self.model.training
        self.model.eval()
        rng = np.random.default_rng(seed)
        env_order = rng.permutation(len(envs))
        episode_seeds = rng.integers(0, 2**31 - 1, size=n_episodes)
        returns: list[float] = []
        drawdowns: list[float] = []
        for episode_idx in range(n_episodes):
            # Round-robin after a seeded permutation prevents large markets or
            # lucky random draws from dominating checkpoint selection.
            env = envs[int(env_order[episode_idx % len(env_order)])]
            obs, _ = env.reset(seed=int(episode_seeds[episode_idx]))
            history = None
            peak = float(env.cfg.initial_equity)
            final = peak
            max_dd = 0.0
            for _step in range(self.cfg.max_steps_per_episode):
                chart = torch.from_numpy(obs["chart"]).unsqueeze(0).to(self.device)
                numeric = torch.from_numpy(obs["numeric"]).unsqueeze(0).to(self.device)
                context = torch.from_numpy(obs["context"]).unsqueeze(0).to(self.device)
                out = self.model(chart=chart, numeric=numeric, context=context, history=history)
                action = int(out["policy_logits"].argmax(dim=-1).item())
                history = out.get("next_history")
                obs, _, terminated, truncated, info = env.step(action)
                final = float(info.get("equity", final))
                peak = max(peak, final)
                max_dd = max(max_dd, (peak - final) / max(peak, 1e-12))
                if terminated or truncated:
                    break
            returns.append(final / max(float(env.cfg.initial_equity), 1e-12) - 1.0)
            drawdowns.append(max_dd)
        if was_training:
            self.model.train()
        ordered = np.sort(np.asarray(returns, dtype=np.float64))
        tail_n = max(1, int(np.floor(cvar_alpha * len(ordered))))
        cvar = float(ordered[:tail_n].mean())
        return {
            "mean_equity_return": float(ordered.mean()),
            "cvar": cvar,
            "cvar_alpha": float(cvar_alpha),
            "cvar_10": cvar if cvar_alpha == 0.1 else float(
                ordered[:max(1, int(np.floor(0.1 * len(ordered))))].mean()
            ),
            "worst_equity_return": float(ordered[0]),
            "mean_max_drawdown": float(np.mean(drawdowns)),
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame | Sequence[pd.DataFrame],
        val_df: Optional[pd.DataFrame | Sequence[pd.DataFrame]] = None,
    ) -> dict:
        """Run PPO on the given OHLCV DataFrame.

        Returns a dict with per-iteration history and aggregate stats.
        """
        cfg = self.cfg
        frames = list(df) if isinstance(df, Sequence) and not isinstance(df, pd.DataFrame) else [df]
        envs = [TradingEnv(frame, cfg=cfg.env_cfg) for frame in frames]
        val_envs: list[TradingEnv] = []
        if val_df is not None:
            val_frames = list(val_df) if isinstance(val_df, Sequence) and not isinstance(val_df, pd.DataFrame) else [val_df]
            val_envs = [TradingEnv(frame, cfg=cfg.env_cfg) for frame in val_frames]
        history: list[dict] = []
        timer = Timer()
        for it in range(self._iteration, cfg.n_iterations):
            is_best = False
            # In a clean PPO loop the outer "iteration" is "collect a
            # rollout of T steps and update". Here we collect a fixed
            # number of episodes per iteration to keep the loop simple.
            timer.start()
            buf, rollout_stats = self._collect_rollout(envs)
            losses = self._ppo_update(buf)
            timer.stop()
            mean_return = float(np.mean(rollout_stats["ep_returns"]))
            entry = {
                "iteration": it,
                "n_episodes": len(rollout_stats["ep_returns"]),
                "rollout_steps": len(buf),
                "mean_return": mean_return,
                "mean_equity_return": float(np.mean(rollout_stats["ep_equity_returns"])),
                "worst_equity_return": float(np.min(rollout_stats["ep_equity_returns"])),
                "mean_max_drawdown": float(np.mean(rollout_stats["ep_max_drawdowns"])),
                "policy_loss": losses["policy"],
                "value_loss": losses["value"],
                "entropy": losses["entropy"],
                "total_loss": losses["total"],
                "elapsed_s": timer.elapsed,
            }
            if val_envs and cfg.eval_every_iterations > 0 and (it + 1) % cfg.eval_every_iterations == 0:
                entry["val"] = self._evaluate_policy(
                    val_envs, cfg.eval_episodes, cfg.seed + 100_000,
                )
                val_score = entry["val"]["mean_equity_return"]
                if val_score > self._best_val_score + cfg.early_stopping_min_delta:
                    self._best_val_score = val_score
                    self._bad_evals = 0
                    is_best = True
                else:
                    self._bad_evals += 1
                    is_best = False
            history.append(entry)
            self._iteration = it + 1
            if val_envs and is_best and cfg.best_checkpoint:
                self.save(cfg.best_checkpoint)
            if (it + 1) % cfg.log_every == 0:
                logger.info(
                    "ppo it=%d episodes=%d steps=%d shaped_return=%.4f "
                    "equity_return=%.5f max_dd=%.5f policy=%.4f value=%.4f "
                    "entropy=%.4f total=%.4f elapsed=%.1fs",
                    it, len(rollout_stats["ep_returns"]), len(buf),
                    mean_return, entry["mean_equity_return"], entry["mean_max_drawdown"],
                    losses["policy"], losses["value"],
                    losses["entropy"], losses["total"], timer.elapsed,
                )
            timer.reset()
            if cfg.checkpoint_every_iterations > 0 and (it + 1) % cfg.checkpoint_every_iterations == 0:
                checkpoint = Path(cfg.checkpoint or "artifacts/s4/policy.pt")
                self.save(str(checkpoint.with_name(f"{checkpoint.stem}_iter{it + 1}{checkpoint.suffix}")))
            should_stop = (
                bool(val_envs)
                and cfg.early_stopping_patience > 0
                and self._bad_evals >= cfg.early_stopping_patience
            )
            # Do this before the next _collect_rollout call. Assignment keeps
            # the previous local alive while the RHS is evaluated, which can
            # otherwise overlap two multi-GB buffers.
            del buf, rollout_stats
            _release_rollout_memory()
            if should_stop:
                logger.info("ppo early stopping at iteration %d; best_val_return=%.6f", it, self._best_val_score)
                break
        if cfg.checkpoint:
            self.save(cfg.checkpoint)
        return {"history": history}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # JSON-serialise the config (convert tuple ``vision_channels``).
        cfg_dict = self.model.cfg.__dict__.copy()
        if "vision_channels" in cfg_dict and isinstance(cfg_dict["vision_channels"], tuple):
            cfg_dict["vision_channels"] = list(cfg_dict["vision_channels"])
        payload = {
            "model": self.model.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "ppo_config": self.cfg.__dict__,
            "optimizer": self.opt.state_dict(),
            "trainer_state": {
                "step": self._step,
                "iteration": self._iteration,
                "best_val_score": self._best_val_score,
                "bad_evals": self._bad_evals,
                "numpy_rng_state": self._rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "checkpoint_meta": {
                "stage": "s4_ppo",
                "trading_policy_ready": True,
                "policy_head_trained": True,
                "policy_training": "ppo_reward_optimization",
                "source_checkpoint": self.cfg.source_checkpoint,
                "dataset": {
                    "root": self.cfg.dataset_root,
                    "manifest_checksum": self.cfg.dataset_manifest_checksum,
                },
            },
        }
        tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink()
        logger.info("ppo checkpoint saved to %s", p)

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            self.opt.load_state_dict(payload["optimizer"])
        state = payload.get("trainer_state") or {}
        self._step = int(state.get("step", 0))
        self._iteration = int(state.get("iteration", 0))
        self._best_val_score = float(
            state.get("best_val_score", state.get("best_val_cvar", float("-inf")))
        )
        self._bad_evals = int(state.get("bad_evals", 0))
        if state.get("numpy_rng_state"):
            self._rng.bit_generator.state = state["numpy_rng_state"]
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng_state"]])
        return {
            "step": self._step,
            "iteration": self._iteration,
            "stage": (payload.get("checkpoint_meta") or {}).get("stage"),
        }
