"""Memory sessions: THE single contract for rolling working-memory state.

Every consumer of :class:`~zhisa.models.policy.PolicyNetwork` that wants the
working memory to see real past embeddings (S2 supervised, S4 RL, backtests,
paper/live trading) runs through this module:

    state = session_start(model)                 # zeros = cold episode start
    out, state = session_step(model, obs, state) # advances the rolling state
    warm = session_warm_up(model, obs_sequence)  # or pre-warm from history

The state is the model's own ``next_history`` buffer (detached by the model),
so there is NO backprop-through-time into previous encoder passes; only the
memory module's weights learn from the rolling stream — identical in S2, S4
and serve.

Two evaluation contracts are always available:
    * cold  — ``session_start`` (fresh episode; also the S2 non-sequential
      path and any caller that passes ``history=None``);
    * warm  — real rolling history (S4 rollout pattern).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from zhisa.models.policy import PolicyNetwork


def session_start(model: PolicyNetwork, n_sessions: int = 1) -> Optional[torch.Tensor]:
    """Begin a memory session: zeroed history (cold episode start)."""
    if model.memory is None:
        return None
    dev = next(model.memory.parameters()).device
    slots = model.memory.cfg.max_len - 1
    return torch.zeros(n_sessions, slots, model.cfg.embed_dim, device=dev,
                       dtype=torch.float32)


def session_step(
    model: PolicyNetwork,
    obs: dict,
    state: Optional[torch.Tensor],
) -> tuple[dict, Optional[torch.Tensor]]:
    """One forward with rolling memory; returns ``(out, next_state)``.

    The input batch is placed on the model's device so a live/paper session on
    CUDA never mixes CPU observations with a CUDA ``history`` buffer.
    """
    dev = next(model.parameters()).device
    batch = _to_batch(obs, device=dev)
    out = model(
        chart=batch["chart"],
        numeric=batch["numeric"],
        context=batch["context"],
        history=state,
        macro_numeric=_maybe(batch, "macro_numeric"),
        instrument_id=_maybe(batch, "instrument_id"),
    )
    return out, out.get("next_history") if model.memory is not None else None


def session_warm_up(
    model: PolicyNetwork,
    obs_sequence: list[dict],
    start: Optional[torch.Tensor] = None,
) -> tuple[list[dict], Optional[torch.Tensor]]:
    """Roll through ``obs_sequence`` and return ``(outs, final_state)``.

    Used to pre-warm live/backtest sessions from recent bars and to compute
    warm-eval metrics for S2.
    """
    state = start if start is not None else session_start(model, 1)
    outs = []
    with torch.no_grad():
        for obs in obs_sequence:
            out, state = session_step(model, obs, state)
            outs.append(out)
    return outs, state


def memory_sensitivity(
    model: PolicyNetwork,
    obs_sequence: list[dict],
) -> float:
    """1 - cos(z_cold, z_warm): how much the memory actually USES history.

    ~0.001 on untrained memory; a trained memory should push this well above
    the noise floor (target >= 0.05).
    """
    if model.memory is None:
        return 0.0
    batch = _to_batch(obs_sequence[-1], device=next(model.parameters()).device)
    with torch.no_grad():
        cold = model(
            chart=batch["chart"], numeric=batch["numeric"], context=batch["context"],
            history=None, macro_numeric=_maybe(batch, "macro_numeric"),
            instrument_id=_maybe(batch, "instrument_id"),
        )["embedding"][0]
    outs, _ = session_warm_up(model, obs_sequence)
    z_warm = outs[-1]["embedding"][0]
    cos = float(torch.nn.functional.cosine_similarity(
        cold.unsqueeze(0), z_warm.unsqueeze(0), dim=-1).item())
    return round(max(0.0, 1.0 - cos), 5)


def make_stateful_policy(
    model: PolicyNetwork,
    *,
    warm_obs: Optional[list[dict]] = None,
    seed_state: Optional[torch.Tensor] = None,
) -> Callable[[dict], int]:
    """Wrap ``PolicyNetwork.forward`` into a ``policy(obs) -> action`` callable
    that keeps rolling memory state across calls (serve/backtest contract).

    ``warm_obs``: recent bars to pre-warm the session before trading.
    """
    state = seed_state
    if warm_obs is not None and model.memory is not None:
        _, state = session_warm_up(model, list(warm_obs), start=state)

    def _policy(obs: dict) -> int:
        nonlocal state
        out, state = session_step(model, obs, state)
        return int(torch.argmax(out["policy_logits"]).item())

    return _policy


def _to_batch(obs: dict, device: Optional[torch.device] = None) -> dict:
    # Per-key expected batch ranks: chart 4D, numeric/macro 3D, context 2D.
    _rank = {"chart": 4, "numeric": 3, "context": 2, "macro_numeric": 3}

    def _t(key: str):
        v = obs.get(key)
        if v is None:
            return None
        if torch.is_tensor(v):
            t = v
            if device is not None:
                t = t.to(device)
            while t.ndim < _rank.get(key, 3):
                t = t.unsqueeze(0)
            return t
        arr = np.asarray(v)
        t = torch.as_tensor(arr, dtype=torch.float32, device=device)
        while t.ndim < _rank.get(key, 3):
            t = t.unsqueeze(0)
        return t

    out = {
        "chart": _t("chart"),
        "numeric": _t("numeric"),
        "context": _t("context"),
        "macro_numeric": _t("macro_numeric"),
    }
    _iv = obs.get("instrument_id")
    if _iv is not None:
        if torch.is_tensor(_iv):
            t = _iv.to(device) if device is not None else _iv
            t = t.unsqueeze(0) if t.ndim == 0 else t
            out["instrument_id"] = t
        else:
            out["instrument_id"] = torch.as_tensor(
                np.asarray(_iv), dtype=torch.long, device=device
            ).unsqueeze(0)
    else:
        out["instrument_id"] = None
    if out["chart"] is None or out["numeric"] is None or out["context"] is None:
        raise ValueError("obs must contain chart/numeric/context")
    return out


def _maybe(batch: dict, key: str):
    v = batch.get(key)
    return None if v is None else v