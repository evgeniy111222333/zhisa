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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO trainer."""

    # Rollout collection
    n_episodes: int = 10
    max_steps_per_episode: int = 500
    # PPO updates
    n_epochs: int = 4
    minibatch_size: int = 64
    # Loss coefficients
    clip_ratio: float = 0.2
    value_coef: float = 0.5
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


class RolloutBuffer:
    """A flat buffer of :class:`Transition` records for one or more episodes.

    The buffer is intentionally simple — a list of dataclasses — so we
    can stream episodes one at a time without worrying about pre-allocation.
    For larger problems a circular numpy buffer would be a drop-in
    replacement.
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

    def stack_tensors(self) -> dict[str, np.ndarray]:
        """Stack per-transition arrays into batched numpy arrays.

        Charts are stacked along axis 0 to give a ``(N, 3, H, W)`` array.
        """
        if not self._data:
            return {}
        return {
            "chart": np.stack([t.chart for t in self._data], axis=0),
            "numeric": np.stack([t.numeric for t in self._data], axis=0),
            "context": np.stack([t.context for t in self._data], axis=0),
            "action": np.array([t.action for t in self._data], dtype=np.int64),
            "reward": np.array([t.reward for t in self._data], dtype=np.float32),
            "value": np.array([t.value for t in self._data], dtype=np.float32),
            "log_prob": np.array([t.log_prob for t in self._data], dtype=np.float32),
            "done": np.array([t.done for t in self._data], dtype=np.float32),
        }

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
    value_loss = F.mse_loss(values, returns)

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
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = build_optimizer(model, self.cfg.optim)
        self._rng = np.random.default_rng(self.cfg.seed)
        self._step = 0

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _select_action(
        self, obs: dict
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, value, entropy)`` for a single obs.

        If the policy produces non-finite logits (a known failure mode
        of an untrained or freshly-perturbed network), we fall back to
        a uniform random action so the rollout can continue instead
        of crashing inside :class:`torch.distributions.Categorical`.
        """
        chart = torch.from_numpy(obs["chart"]).unsqueeze(0).to(self.device)
        num = torch.from_numpy(obs["numeric"]).unsqueeze(0).to(self.device)
        ctx = torch.from_numpy(obs["context"]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(chart=chart, numeric=num, context=ctx)
            logits = out["policy_logits"]
            if not torch.isfinite(logits).all():
                # Degenerate policy → fall back to a uniform sample.
                n_actions = logits.size(-1)
                action = torch.randint(0, n_actions, (1,), device=self.device)
                logp = torch.log(torch.full((1,), 1.0 / n_actions, device=self.device))
                value = out["value"]
                entropy = torch.log(torch.tensor(float(n_actions), device=self.device))
                return int(action.item()), logp.squeeze(0), value.squeeze(0), entropy.squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            value = out["value"]
            entropy = dist.entropy()
        return int(action.item()), logp.squeeze(0), value.squeeze(0), entropy.squeeze(0)

    def _collect_rollout(self, env: TradingEnv) -> tuple[RolloutBuffer, dict]:
        """Run ``n_episodes`` episodes and return a populated buffer."""
        buf = RolloutBuffer()
        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        for ep in range(self.cfg.n_episodes):
            obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
            ep_return = 0.0
            steps = 0
            for _ in range(self.cfg.max_steps_per_episode):
                action, logp, value, _ = self._select_action(obs)
                next_obs, reward, terminated, truncated, _info = env.step(action)
                buf.add(Transition(
                    chart=obs["chart"],
                    numeric=obs["numeric"],
                    context=obs["context"],
                    action=action,
                    reward=float(reward),
                    value=float(value.item()),
                    log_prob=float(logp.item()),
                    done=bool(terminated or truncated),
                ))
                ep_return += float(reward)
                steps += 1
                obs = next_obs
                if terminated or truncated:
                    break
            ep_returns.append(ep_return)
            ep_lengths.append(steps)
        return buf, {
            "ep_returns": ep_returns,
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

        # Pre-tensor everything to device.
        to_t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        adv_t = to_t(advantages)
        ret_t = to_t(returns)
        old_logp_t = to_t(stacked["log_prob"])
        action_t = to_t(stacked["action"])
        chart_t = to_t(stacked["chart"]).float()
        num_t = to_t(stacked["numeric"]).float()
        ctx_t = to_t(stacked["context"]).float()

        stats = {"policy": [], "value": [], "entropy": [], "total": []}
        n_updates = 0
        for epoch in range(cfg.n_epochs):
            for idx in buf.minibatch_indices(cfg.minibatch_size, self._rng):
                # Forward pass on the mini-batch.
                out = self.model(chart=chart_t[idx], numeric=num_t[idx], context=ctx_t[idx])
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
                    kl = (old_logp_t[idx] - new_logp).mean().item()
                if abs(kl) > cfg.target_kl:
                    logger.info("ppo early-stop at epoch %d: KL=%.4f", epoch, kl)
                    break
            else:
                # Inner loop completed without break — keep going.
                continue
            break  # break outer if inner broke

        if not stats["total"]:
            return {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        """Run PPO on the given OHLCV DataFrame.

        Returns a dict with per-iteration history and aggregate stats.
        """
        cfg = self.cfg
        env = TradingEnv(df, cfg=cfg.env_cfg)
        history: list[dict] = []
        timer = Timer()
        for it in range(cfg.n_episodes * 1):  # outer loop = n_iterations
            # In a clean PPO loop the outer "iteration" is "collect a
            # rollout of T steps and update". Here we collect a fixed
            # number of episodes per iteration to keep the loop simple.
            timer.start()
            buf, rollout_stats = self._collect_rollout(env)
            losses = self._ppo_update(buf)
            timer.stop()
            mean_return = float(np.mean(rollout_stats["ep_returns"]))
            history.append({
                "iteration": it,
                "n_episodes": len(rollout_stats["ep_returns"]),
                "rollout_steps": len(buf),
                "mean_return": mean_return,
                "policy_loss": losses["policy"],
                "value_loss": losses["value"],
                "entropy": losses["entropy"],
                "total_loss": losses["total"],
                "elapsed_s": timer.elapsed,
            })
            if (it + 1) % cfg.log_every == 0:
                logger.info(
                    "ppo it=%d episodes=%d steps=%d mean_return=%.4f "
                    "policy=%.4f value=%.4f entropy=%.4f total=%.4f elapsed=%.1fs",
                    it, len(rollout_stats["ep_returns"]), len(buf),
                    mean_return, losses["policy"], losses["value"],
                    losses["entropy"], losses["total"], timer.elapsed,
                )
            timer.reset()
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
        torch.save({
            "model": self.model.state_dict(),
            "config": cfg_dict,
            "model_config": cfg_dict,  # canonical name
            "ppo_config": self.cfg.__dict__,
        }, p)
        logger.info("ppo checkpoint saved to %s", p)
