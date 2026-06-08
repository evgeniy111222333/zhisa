"""Interpretability tools for the policy network.

Three lightweight techniques that don't require extra dependencies
or model surgery:

* :func:`attention_rollout` — average the attention weights across
  the fusion Transformer's layers and heads to get a per-token
  importance map (Abnar & Zuidema 2020).
* :func:`integrated_gradients` — feature attribution via the
  integrated-gradients method (Sundararajan et al. 2017). Returns
  per-input-feature importance for the *numeric* tensor (most
  useful for debugging the encoder).
* :func:`feature_ablation` — leave-one-out feature importance:
  mask each feature column, run the model, and rank features by
  the resulting logit delta. Slow but model-agnostic.

v2 additions (multi-modal & cross-instrument):

* :func:`chart_saliency` — input-gradient saliency map of the chart
  (optionally smoothed via SmoothGrad).
* :func:`per_modality_attributions` — integrated gradients run on
  each modality (chart, numeric, context) separately and combined
  into one report.
* :func:`cross_instrument_rollout` — attention rollout on the
  cross-instrument attention block of a
  :class:`PortfolioPolicyNetwork` (Stage 2).
* :func:`action_explanation` — bundle every per-sample explanation
  (chosen action, top features, saliency summary, action
  probabilities) into a single dict ready for logging or display.
* :func:`per_dataset_summary` — aggregate action_explanations over
  a :class:`MarketDataset` and return a per-feature / per-action
  summary.

All entry points take a model + a single observation and return
either a numpy array of importances or a small dict. Observations
may be ``np.ndarray`` or ``torch.Tensor``; both are accepted and
converted internally.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from zhisa.utils.logging import get_logger

logger = get_logger(__name__)


def _to_numpy(x) -> np.ndarray:
    """Convert a torch tensor or numpy array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------


def _hook_attentions(module: nn.Module, inputs, output):
    """Forward hook that stashes attention weights on the module.

    PyTorch's MultiheadAttention only returns attention weights
    when ``need_weights=True`` is passed at call time. We monkey-patch
    the MHA's ``forward`` to set that flag (see
    :func:`attention_rollout`).
    """
    module._last_attn_weights = output[1] if isinstance(output, tuple) else None


def _patched_mha_forward(self, *args, **kwargs):
    """Wrap the original MHA forward to always request attention weights.

    The original method is stashed on ``self._orig_forward`` by
    :func:`attention_rollout`. After the rollout finishes the original
    method is restored.
    """
    kwargs["need_weights"] = True
    return self._orig_forward(*args, **kwargs)


def attention_rollout(
    model: nn.Module,
    obs: dict,
    *,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Compute the attention rollout across the fusion transformer.

    Args:
        model: a :class:`PolicyNetwork` (or any model with a
            ``.fusion`` attribute containing a ``nn.TransformerEncoder``).
        obs: an observation dict with ``chart``, ``numeric``, ``context``.
        head_fusion: ``"mean"`` or ``"max"`` over attention heads.
        discard_ratio: fraction of lowest attention values to zero out
            (helps visualise the strongest relations).
        device: torch device.

    Returns:
        A ``(T+1, T+1)`` numpy array of rolled-out attention weights
        from the CLS token to all other tokens, or ``None`` if the
        model has no transformer fusion.
    """
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        logger.warning("model has no .fusion — cannot rollout")
        return None
    # The fusion module uses ``self.encoder = nn.TransformerEncoder(...)``.
    transformer = getattr(fusion, "encoder", None)
    if transformer is None or not hasattr(transformer, "layers"):
        logger.warning("fusion has no .encoder.layers — cannot rollout")
        return None
    if len(transformer.layers) == 0:
        return None
    # Temporarily monkey-patch the MHA forward to always request
    # attention weights (PyTorch's ``nn.TransformerEncoderLayer`` calls
    # ``self.self_attn(...)`` with ``need_weights=False`` for speed).
    mha_modules: list[nn.MultiheadAttention] = []
    for layer in transformer.layers:
        mha = getattr(layer, "self_attn", None)
        if isinstance(mha, nn.MultiheadAttention):
            mha_modules.append(mha)
            if not hasattr(mha, "_orig_forward"):
                mha._orig_forward = mha.forward
            # Bind the unbound method to the instance.
            import types
            mha.forward = types.MethodType(_patched_mha_forward, mha)
    # Forward pass.
    handles = [m.register_forward_hook(_hook_attentions) for m in mha_modules]
    model.eval()
    try:
        chart = torch.from_numpy(_to_numpy(obs["chart"])).unsqueeze(0).to(device)
        numeric = torch.from_numpy(_to_numpy(obs["numeric"])).unsqueeze(0).to(device)
        context = torch.from_numpy(_to_numpy(obs["context"])).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(chart=chart, numeric=numeric, context=context)
    finally:
        for h in handles:
            h.remove()
        for m in mha_modules:
            if hasattr(m, "_orig_forward"):
                m.forward = m._orig_forward
    # Stack per-layer attentions: (L, B, H, T, T).
    layer_attns: list[torch.Tensor] = []
    for m in mha_modules:
        w = getattr(m, "_last_attn_weights", None)
        if w is None:
            continue
        # ``w`` may be (B, H, T, T) or (B, T, T) — handle both.
        if w.dim() == 4:
            layer_attns.append(w)
        elif w.dim() == 3:
            layer_attns.append(w.unsqueeze(1))
    if not layer_attns:
        return None
    stacked = torch.stack(layer_attns, dim=0)  # (L, B, H, T, T)
    if head_fusion == "mean":
        fused = stacked.mean(dim=2)  # (L, B, T, T)
    elif head_fusion == "max":
        fused = stacked.max(dim=2).values
    elif head_fusion == "min":
        fused = stacked.min(dim=2).values
    else:
        raise ValueError(f"unknown head_fusion: {head_fusion!r}")
    # Optionally drop the lowest ``discard_ratio`` attention values.
    if discard_ratio > 0.0:
        L, B, T, _ = fused.shape
        flat = fused.reshape(L, B, T * T)
        n_drop = int(T * T * discard_ratio)
        if n_drop > 0:
            threshold = flat.topk(n_drop, largest=False).values.max(dim=-1).values
            mask = (flat >= threshold.unsqueeze(-1)).float()
            fused = (flat * mask).reshape(L, B, T, T)
    # Rollout: matrix-multiply the residual (I + A) across layers.
    result = torch.eye(fused.size(-1), device=fused.device).unsqueeze(0).unsqueeze(0)
    result = result.expand(fused.size(0), fused.size(1), -1, -1).clone()
    for layer in fused.unbind(dim=0):
        I = torch.eye(layer.size(-1), device=layer.device).unsqueeze(0)
        attn_with_residual = layer + I
        # Row-normalise.
        attn_with_residual = attn_with_residual / attn_with_residual.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-9)
        result = torch.matmul(attn_with_residual, result)
    # Return the full rolled-out attention matrix for the first sample.
    return result[0, 0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Integrated gradients
# ---------------------------------------------------------------------------


def integrated_gradients(
    model: nn.Module,
    obs: dict,
    *,
    target: Optional[int] = None,
    n_steps: int = 32,
    device: str = "cpu",
) -> dict:
    """Integrated gradients for the *numeric* input.

    Args:
        model: a :class:`PolicyNetwork`.
        obs: an observation dict.
        target: which class index to attribute (default: argmax of
            ``policy_logits``).
        n_steps: number of Riemann approximation steps.
        device: torch device.

    Returns:
        A dict with:
          * ``attributions`` — same shape as ``obs["numeric"]``,
            with non-negative attributions summing (approximately)
            to ``output[target] - baseline[target]``.
          * ``total`` — scalar sum of attributions.
    """
    chart = torch.from_numpy(_to_numpy(obs["chart"])).unsqueeze(0).to(device)
    numeric = torch.from_numpy(_to_numpy(obs["numeric"])).unsqueeze(0).to(device).clone()
    context = torch.from_numpy(_to_numpy(obs["context"])).unsqueeze(0).to(device)
    baseline = torch.zeros_like(numeric)
    if target is None:
        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)
        target = int(out["policy_logits"].argmax(dim=-1).item())
    # Build a path of (n_steps+1) interpolated inputs.
    path = [baseline + (float(i) / n_steps) * (numeric - baseline)
            for i in range(n_steps + 1)]
    # Compute gradients at each step.
    grads = []
    for x in path:
        x = x.detach().requires_grad_(True)
        out = model(chart=chart, numeric=x, context=context)
        score = out["policy_logits"][0, target]
        score.backward()
        grads.append(x.grad.detach().clone())
        model.zero_grad(set_to_none=True)
    avg_grad = torch.stack(grads, dim=0).mean(dim=0)
    ig = (numeric - baseline) * avg_grad
    return {
        "attributions": ig.squeeze(0).detach().cpu().numpy(),
        "total": float(ig.sum().item()),
    }


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------


def feature_ablation(
    model: nn.Module,
    obs: dict,
    *,
    target: Optional[int] = None,
    n_features: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Leave-one-feature-out importance for the *numeric* input.

    For each column ``j`` in the numeric tensor, zero it out, run
    the model, and record ``|delta_logit_j|``. The result is an
    importance vector of length ``F``.
    """
    chart = torch.from_numpy(_to_numpy(obs["chart"])).unsqueeze(0).to(device)
    numeric = torch.from_numpy(_to_numpy(obs["numeric"])).unsqueeze(0).to(device).clone()
    context = torch.from_numpy(_to_numpy(obs["context"])).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(chart=chart, numeric=numeric, context=context)
        base_logits = out["policy_logits"][0].detach().cpu().numpy()
    if target is None:
        target = int(np.argmax(base_logits))
    F = numeric.size(-1) if n_features is None else int(n_features)
    importances = np.zeros(F, dtype=np.float32)
    for j in range(F):
        masked = numeric.clone()
        masked[0, :, j] = 0.0
        with torch.no_grad():
            out = model(chart=chart, numeric=masked, context=context)
            new_logits = out["policy_logits"][0].detach().cpu().numpy()
        importances[j] = abs(base_logits[target] - new_logits[target])
    return importances


# ---------------------------------------------------------------------------
# v2: chart saliency, per-modality attributions, cross-instrument rollout,
# action explanation, dataset-level summary.
# ---------------------------------------------------------------------------


def chart_saliency(
    model: nn.Module,
    obs: dict,
    *,
    target: Optional[int] = None,
    smoothgrad_n: int = 0,
    sigma: float = 0.1,
    device: str = "cpu",
) -> np.ndarray:
    """Input-gradient saliency map of the chart image.

    The gradient of the chosen logit with respect to the chart
    pixels is computed, optionally averaged over ``smoothgrad_n``
    noise-perturbed copies (Smilkov et al. 2017).

    Args:
        model: a :class:`PolicyNetwork`.
        obs: an observation dict.
        target: which action to attribute (default: argmax of
            ``policy_logits``).
        smoothgrad_n: number of noise samples (0 = raw saliency).
        sigma: standard deviation of the noise, as a fraction of
            the chart's per-pixel std.
        device: torch device.

    Returns:
        A numpy array of shape ``(3, H, W)`` with signed
        attributions (the magnitude is meaningful; the sign is the
        direction of influence on the chosen logit).
    """
    chart = torch.from_numpy(_to_numpy(obs["chart"])).unsqueeze(0).to(device).clone()
    numeric = torch.from_numpy(_to_numpy(obs["numeric"])).unsqueeze(0).to(device)
    context = torch.from_numpy(_to_numpy(obs["context"])).unsqueeze(0).to(device)
    if target is None:
        with torch.no_grad():
            out = model(chart=chart, numeric=numeric, context=context)
        target = int(out["policy_logits"].argmax(dim=-1).item())
    n_samples = max(int(smoothgrad_n), 0) + 1
    noise_std = float(sigma) * float(chart.std().item() + 1e-9)
    accumulator = torch.zeros_like(chart)
    for k in range(n_samples):
        c = chart
        if k > 0:
            c = c + noise_std * torch.randn_like(chart)
        c = c.detach().requires_grad_(True)
        out = model(chart=c, numeric=numeric, context=context)
        score = out["policy_logits"][0, target]
        score.backward()
        accumulator = accumulator + c.grad.detach()
        model.zero_grad(set_to_none=True)
    return (accumulator / n_samples).squeeze(0).detach().cpu().numpy()


def _integrated_gradients_for(
    model: nn.Module,
    chart: torch.Tensor,
    numeric: torch.Tensor,
    context: torch.Tensor,
    *,
    target: int,
    n_steps: int,
) -> dict[str, torch.Tensor]:
    """Run IG separately on chart, numeric, context (all three enabled).

    The numerical path is the standard integrated-gradients integral.
    The chart and context paths are computed with the same Riemann
    approximation so that the three attributions are directly
    comparable in scale.
    """
    out: dict[str, torch.Tensor] = {}
    for name, x in (("chart", chart), ("numeric", numeric), ("context", context)):
        baseline = torch.zeros_like(x)
        path = [baseline + (float(i) / n_steps) * (x - baseline)
                for i in range(n_steps + 1)]
        grads = []
        for step in path:
            step = step.detach().requires_grad_(True)
            o = model(chart=step if name == "chart" else chart,
                      numeric=step if name == "numeric" else numeric,
                      context=step if name == "context" else context)
            score = o["policy_logits"][0, target]
            score.backward()
            grads.append(step.grad.detach().clone())
            model.zero_grad(set_to_none=True)
        avg = torch.stack(grads, dim=0).mean(dim=0)
        out[name] = (x - baseline) * avg
    return out


def per_modality_attributions(
    model: nn.Module,
    obs: dict,
    *,
    target: Optional[int] = None,
    n_steps: int = 16,
    device: str = "cpu",
) -> dict:
    """Integrated gradients for each modality, plus scalar totals.

    Returns a dict with:

    * ``chart`` — (3, H, W) signed attributions
    * ``numeric`` — (T, F) signed attributions
    * ``context`` — (C,) signed attributions
    * ``totals`` — dict of scalar |attribution| sums per modality
    * ``target`` — the action index that was attributed
    * ``target_logit`` — its logit value
    """
    chart = torch.from_numpy(_to_numpy(obs["chart"])).unsqueeze(0).to(device).clone()
    numeric = torch.from_numpy(_to_numpy(obs["numeric"])).unsqueeze(0).to(device).clone()
    context = torch.from_numpy(_to_numpy(obs["context"])).unsqueeze(0).to(device).clone()
    with torch.no_grad():
        out = model(chart=chart, numeric=numeric, context=context)
    if target is None:
        target = int(out["policy_logits"].argmax(dim=-1).item())
    target_logit = float(out["policy_logits"][0, target].item())
    igs = _integrated_gradients_for(
        model, chart, numeric, context, target=target, n_steps=int(n_steps),
    )
    return {
        "chart": igs["chart"].squeeze(0).detach().cpu().numpy(),
        "numeric": igs["numeric"].squeeze(0).detach().cpu().numpy(),
        "context": igs["context"].squeeze(0).detach().cpu().numpy(),
        "totals": {
            "chart": float(igs["chart"].abs().sum().item()),
            "numeric": float(igs["numeric"].abs().sum().item()),
            "context": float(igs["context"].abs().sum().item()),
        },
        "target": int(target),
        "target_logit": target_logit,
    }


def cross_instrument_rollout(
    policy: nn.Module,
    instruments: dict,
    portfolio: torch.Tensor | np.ndarray,
    *,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Attention rollout over the cross-instrument attention block.

    Requires the policy to be a :class:`PortfolioPolicyNetwork` with
    ``cross_attn`` enabled (Stage 2). Returns the (N, N) attention
    rollout matrix where ``out[i, j]`` is the influence of
    instrument ``j`` on instrument ``i``. Returns ``None`` when the
    model has no cross-attention block.

    Args:
        policy: a :class:`PortfolioPolicyNetwork`.
        instruments: dict with keys ``chart``, ``numeric``, ``context``,
            each shaped ``(B, N, ...)`` (already batched).
        portfolio: ``(B, portfolio_dim)`` portfolio summary.
        head_fusion: ``"mean"``, ``"max"`` or ``"min"`` over heads.
        discard_ratio: fraction of lowest attention values to zero
            out (improves the visualised structure).
        device: torch device.
    """
    cross = getattr(policy, "cross_attn", None)
    if cross is None:
        logger.warning("policy has no .cross_attn — cannot rollout")
        return None
    transformer = getattr(cross, "layers", None)
    if transformer is None or not hasattr(transformer, "layers") or len(transformer.layers) == 0:
        logger.warning("cross_attn has no transformer layers")
        return None
    mha_modules: list[nn.MultiheadAttention] = []
    for layer in transformer.layers:
        mha = getattr(layer, "self_attn", None)
        if isinstance(mha, nn.MultiheadAttention):
            mha_modules.append(mha)
            if not hasattr(mha, "_orig_forward"):
                mha._orig_forward = mha.forward
            import types
            mha.forward = types.MethodType(_patched_mha_forward, mha)
    handles = [m.register_forward_hook(_hook_attentions) for m in mha_modules]
    policy.eval()
    try:
        c = torch.from_numpy(_to_numpy(instruments["chart"])).to(device)
        n = torch.from_numpy(_to_numpy(instruments["numeric"])).to(device)
        ctx = torch.from_numpy(_to_numpy(instruments["context"])).to(device)
        pf = torch.from_numpy(_to_numpy(portfolio)).to(device)
        if pf.dim() == 1:
            pf = pf.unsqueeze(0)
        with torch.no_grad():
            _ = policy(instruments={"chart": c, "numeric": n, "context": ctx},
                       portfolio=pf)
    finally:
        for h in handles:
            h.remove()
        for m in mha_modules:
            if hasattr(m, "_orig_forward"):
                m.forward = m._orig_forward
    layer_attns: list[torch.Tensor] = []
    for m in mha_modules:
        w = getattr(m, "_last_attn_weights", None)
        if w is None:
            continue
        if w.dim() == 4:
            layer_attns.append(w)
        elif w.dim() == 3:
            layer_attns.append(w.unsqueeze(1))
    if not layer_attns:
        return None
    stacked = torch.stack(layer_attns, dim=0)
    if head_fusion == "mean":
        fused = stacked.mean(dim=2)
    elif head_fusion == "max":
        fused = stacked.max(dim=2).values
    elif head_fusion == "min":
        fused = stacked.min(dim=2).values
    else:
        raise ValueError(f"unknown head_fusion: {head_fusion!r}")
    if discard_ratio > 0.0:
        L, B, T, _ = fused.shape
        flat = fused.reshape(L, B, T * T)
        n_drop = int(T * T * discard_ratio)
        if n_drop > 0:
            threshold = flat.topk(n_drop, largest=False).values.max(dim=-1).values
            mask = (flat >= threshold.unsqueeze(-1)).float()
            fused = (flat * mask).reshape(L, B, T, T)
    eye = torch.eye(fused.size(-1), device=fused.device)
    eye = eye.unsqueeze(0).unsqueeze(0).expand(fused.size(0), fused.size(1), -1, -1).clone()
    result = eye
    for layer in fused.unbind(dim=0):
        I = torch.eye(layer.size(-1), device=layer.device).unsqueeze(0)
        attn_with_residual = layer + I
        attn_with_residual = attn_with_residual / attn_with_residual.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-9)
        result = torch.matmul(attn_with_residual, result)
    return result[0, 0].detach().cpu().numpy()


def _action_name(idx: int) -> str:
    """Best-effort human label for a discrete action index.

    Falls back to a generic ``"action_<idx>"`` when the discrete
    action enum isn't importable (e.g. from a non-trading model).
    """
    try:
        from zhisa.env.trading_env import DiscreteAction
        for member in DiscreteAction:
            if int(member) == int(idx):
                return str(member.name)
    except Exception:
        pass
    return f"action_{int(idx)}"


def action_explanation(
    model: nn.Module,
    obs: dict,
    *,
    target: Optional[int] = None,
    n_steps: int = 16,
    top_k_numeric: int = 5,
    device: str = "cpu",
) -> dict:
    """Bundle every per-sample explanation into a single dict.

    Combines:

    * the chosen (or explicit) target action and its logit
    * per-modality IG attributions
    * the chart saliency map
    * the top-k numeric features by |attribution|
    * the full softmax action distribution
    """
    pm = per_modality_attributions(model, obs, target=target,
                                  n_steps=int(n_steps), device=device)
    target = int(pm["target"])
    sal = chart_saliency(model, obs, target=target, device=device)
    numeric_attr = pm["numeric"]
    flat = np.abs(numeric_attr).sum(axis=0) if numeric_attr.ndim > 1 else np.abs(numeric_attr)
    if flat.size > 0:
        order = np.argsort(-flat)[: int(top_k_numeric)]
        top = [
            {"feature_index": int(j), "importance": float(flat[j]),
             "attribution": float(numeric_attr[..., j].sum() if numeric_attr.ndim > 1
                                  else numeric_attr[j])}
            for j in order
        ]
    else:
        top = []
    with torch.no_grad():
        chart = torch.from_numpy(_to_numpy(obs["chart"])).unsqueeze(0).to(device)
        numeric = torch.from_numpy(_to_numpy(obs["numeric"])).unsqueeze(0).to(device)
        context = torch.from_numpy(_to_numpy(obs["context"])).unsqueeze(0).to(device)
        out = model(chart=chart, numeric=numeric, context=context)
        probs = torch.softmax(out["policy_logits"][0], dim=-1).cpu().numpy()
    return {
        "target": target,
        "target_name": _action_name(target),
        "target_logit": pm["target_logit"],
        "action_probabilities": probs,
        "per_modality_attributions": {
            "chart": pm["chart"], "numeric": pm["numeric"], "context": pm["context"],
        },
        "modality_totals": pm["totals"],
        "chart_saliency": sal,
        "chart_saliency_summary": {
            "abs_mean": float(np.abs(sal).mean()),
            "abs_max": float(np.abs(sal).max()),
            "positive_mass": float((sal > 0).sum()),
            "negative_mass": float((sal < 0).sum()),
        },
        "top_numeric_features": top,
    }


def per_dataset_summary(
    model: nn.Module,
    samples: Sequence[dict],
    *,
    n_samples: Optional[int] = None,
    n_steps: int = 8,
    top_k_numeric: int = 5,
    device: str = "cpu",
) -> dict:
    """Aggregate :func:`action_explanation` over many observations.

    Args:
        model: a :class:`PolicyNetwork`.
        samples: an iterable of observation dicts.
        n_samples: at most this many samples to consume
            (default: all of them).
        n_steps: IG steps (smaller for speed).
        top_k_numeric: top features per sample.
        device: torch device.

    Returns:
        A summary dict with action distribution, top features, mean
        per-modality attribution, and a list of per-sample reports.
    """
    chosen: list[dict] = []
    if n_samples is not None:
        samples = list(samples)[: int(n_samples)]
    for obs in samples:
        try:
            chosen.append(action_explanation(
                model, obs, n_steps=int(n_steps),
                top_k_numeric=int(top_k_numeric), device=device,
            ))
        except Exception as exc:
            logger.warning("failed to explain one sample: %s", exc)
            continue
    if not chosen:
        return {"n_samples": 0, "action_distribution": {},
                "top_features": [], "mean_modality_totals": {},
                "samples": []}
    action_dist: dict[str, int] = {}
    for c in chosen:
        action_dist[c["target_name"]] = action_dist.get(c["target_name"], 0) + 1
    feature_scores: dict[int, float] = {}
    for c in chosen:
        for f in c["top_numeric_features"]:
            j = int(f["feature_index"])
            feature_scores[j] = feature_scores.get(j, 0.0) + float(f["importance"])
    top_features = [
        {"feature_index": int(j), "cumulative_importance": float(s)}
        for j, s in sorted(feature_scores.items(), key=lambda kv: -kv[1])[: int(top_k_numeric)]
    ]
    mean_modality = {
        k: float(np.mean([c["modality_totals"][k] for c in chosen]))
        for k in chosen[0]["modality_totals"]
    }
    return {
        "n_samples": len(chosen),
        "action_distribution": action_dist,
        "top_features": top_features,
        "mean_modality_totals": mean_modality,
        "samples": chosen,
    }
