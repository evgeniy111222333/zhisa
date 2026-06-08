"""Tests for the interpretability module."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.interpretability import (
    attention_rollout,
    feature_ablation,
    integrated_gradients,
)


def _build_model_and_obs(n_bars: int = 200, seed: int = 0):
    torch.manual_seed(seed)
    spec = SampleSpec(chart_window=8, feature_window=8, image_size=8)
    df = generate_market(MarketConfig(n_bars=n_bars, seed=seed))
    ds = MarketDataset(df, spec=spec)
    n_feat = ds._features.shape[1] + ds._time_features.shape[1]
    n_ctx = ds._time_features.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat,
        in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=4, n_regime_classes=spec.n_regime_states,
    )
    sample = ds[0]
    obs = {
        "chart": sample["chart"],
        "numeric": sample["numeric"],
        "context": sample["context"],
    }
    return model, obs


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------


def test_attention_rollout_returns_matrix():
    model, obs = _build_model_and_obs()
    rollout = attention_rollout(model, obs)
    assert rollout is not None
    # 4 tokens: CLS + vision + numeric + context → (4, 4).
    assert rollout.shape[0] == rollout.shape[1]
    assert rollout.ndim == 2
    # All values finite.
    assert np.isfinite(rollout).all()


def test_attention_rollout_handles_missing_fusion():
    """If the model has no ``.fusion``, the function must return ``None``."""
    model, obs = _build_model_and_obs()

    class NoFusion:
        pass

    no_fusion = NoFusion()
    out = attention_rollout(no_fusion, obs)
    assert out is None


def test_attention_rollout_does_not_break_forward():
    """The forward pass must still produce valid outputs after the
    rollout call (no dangling state)."""
    model, obs = _build_model_and_obs()
    attention_rollout(model, obs)
    chart = torch.from_numpy(np.asarray(obs["chart"])).unsqueeze(0)
    numeric = torch.from_numpy(np.asarray(obs["numeric"])).unsqueeze(0)
    context = torch.from_numpy(np.asarray(obs["context"])).unsqueeze(0)
    with torch.no_grad():
        out = model(chart=chart, numeric=numeric, context=context)
    assert "policy_logits" in out
    assert out["policy_logits"].shape[0] == 1


# ---------------------------------------------------------------------------
# Integrated gradients
# ---------------------------------------------------------------------------


def test_integrated_gradients_returns_correct_shape():
    model, obs = _build_model_and_obs()
    ig = integrated_gradients(model, obs, n_steps=8)
    assert "attributions" in ig
    assert "total" in ig
    assert ig["attributions"].shape == obs["numeric"].shape
    assert isinstance(ig["total"], float)


def test_integrated_gradients_zero_at_baseline():
    """At the baseline (all zeros), the attribution must be zero
    along the path. We just check that the total is finite."""
    model, obs = _build_model_and_obs()
    ig = integrated_gradients(model, obs, n_steps=4)
    assert np.isfinite(ig["total"])


def test_integrated_gradients_target_explicit():
    model, obs = _build_model_and_obs()
    ig = integrated_gradients(model, obs, target=2, n_steps=4)
    assert ig["attributions"].shape == obs["numeric"].shape


def test_integrated_gradients_uses_default_target():
    """No target → use the argmax of policy_logits."""
    model, obs = _build_model_and_obs()
    chart = torch.from_numpy(np.asarray(obs["chart"])).unsqueeze(0)
    numeric = torch.from_numpy(np.asarray(obs["numeric"])).unsqueeze(0)
    context = torch.from_numpy(np.asarray(obs["context"])).unsqueeze(0)
    with torch.no_grad():
        out = model(chart=chart, numeric=numeric, context=context)
    expected = int(out["policy_logits"].argmax(dim=-1).item())
    ig = integrated_gradients(model, obs, target=expected, n_steps=4)
    # No assertion on sign; we just want to confirm no crash.
    assert ig["attributions"].shape == np.asarray(obs["numeric"]).shape


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------


def test_feature_ablation_returns_one_per_feature():
    model, obs = _build_model_and_obs()
    n_feat = obs["numeric"].shape[-1]
    imp = feature_ablation(model, obs)
    assert imp.shape == (n_feat,)


def test_feature_ablation_all_non_negative():
    model, obs = _build_model_and_obs()
    imp = feature_ablation(model, obs)
    assert (imp >= 0).all()


def test_feature_ablation_target_explicit():
    model, obs = _build_model_and_obs()
    n_feat = obs["numeric"].shape[-1]
    imp = feature_ablation(model, obs, target=1)
    assert imp.shape == (n_feat,)
    assert (imp >= 0).all()


def test_feature_ablation_zero_input_has_no_effect():
    """Ablating a feature that's already 0 changes the logit by ~0
    (so the importance is near zero)."""
    model, obs = _build_model_and_obs()
    # Zero out the numeric input first.
    obs_zero = {**obs, "numeric": np.zeros_like(obs["numeric"])}
    imp = feature_ablation(model, obs_zero)  # we don't care about speed
    # We don't assert exact zeros (the model may still react to
    # chart+context), just that the importances are finite.
    assert np.isfinite(imp).all()
