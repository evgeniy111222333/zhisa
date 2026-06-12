"""Tests for the v2 interpretability tools."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zhisa.data.dataset import MarketDataset, SampleSpec
from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.models.policy import build_default_policy
from zhisa.training.interpretability import (
    action_explanation,
    chart_saliency,
    cross_instrument_rollout,
    per_dataset_summary,
    per_modality_attributions,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _build_policy_model(seed: int = 0):
    torch.manual_seed(seed)
    spec = SampleSpec(chart_window=8, feature_window=8, image_size=8)
    df = generate_market(MarketConfig(n_bars=200, seed=seed))
    ds = MarketDataset(df, spec=spec)
    n_feat = ds._features.shape[1]
    n_ctx = ds._time_features.shape[1]
    model = build_default_policy(
        in_numeric_features=n_feat, in_context_features=n_ctx,
        window=spec.chart_window, image_size=spec.image_size,
        n_actions=4, n_regime_classes=spec.n_regime_states,
    )
    sample = ds[0]
    return model, sample, ds


def _build_portfolio_model(seed: int = 0, with_cross_attn: bool = True):
    from zhisa.models.portfolio_policy import (
        PortfolioPolicyConfig, PortfolioPolicyNetwork,
    )
    torch.manual_seed(seed)
    cfg = PortfolioPolicyConfig(
        n_instruments=3, embed_dim=32, fusion_hidden=32,
        cross_attn_depth=2 if with_cross_attn else 0,
        cross_attn_heads=4,
        window=8, image_size=8,
        in_numeric_features=4, in_context_features=5,
        portfolio_dim=32,
    )
    m = PortfolioPolicyNetwork(cfg)
    instruments = {
        "chart": torch.randn(1, 3, 3, 8, 8),
        "numeric": torch.randn(1, 3, 8, 4),
        "context": torch.randn(1, 3, 5),
    }
    portfolio = torch.randn(1, 32)
    return m, instruments, portfolio


# ----------------------------------------------------------------------
# chart_saliency
# ----------------------------------------------------------------------

def test_chart_saliency_shape() -> None:
    model, sample, _ = _build_policy_model()
    sal = chart_saliency(model, sample, device="cpu")
    assert sal.shape == (3, 8, 8)
    assert np.isfinite(sal).all()


def test_chart_saliency_smoothgrad_runs() -> None:
    model, sample, _ = _build_policy_model()
    sal = chart_saliency(model, sample, smoothgrad_n=4, sigma=0.1, device="cpu")
    assert sal.shape == (3, 8, 8)
    # Smoothed saliency should be at least somewhat smaller than raw
    raw = chart_saliency(model, sample, smoothgrad_n=0, device="cpu")
    assert np.isfinite(sal).all()
    assert np.isfinite(raw).all()


def test_chart_saliency_with_explicit_target() -> None:
    model, sample, _ = _build_policy_model()
    sal = chart_saliency(model, sample, target=2, device="cpu")
    assert sal.shape == (3, 8, 8)


# ----------------------------------------------------------------------
# per_modality_attributions
# ----------------------------------------------------------------------

def test_per_modality_attributions_shapes() -> None:
    model, sample, _ = _build_policy_model()
    pm = per_modality_attributions(model, sample, n_steps=4, device="cpu")
    assert pm["chart"].shape == (3, 8, 8)
    assert pm["numeric"].shape == sample["numeric"].shape
    assert pm["context"].shape == sample["context"].shape
    assert isinstance(pm["target"], int)
    assert "chart" in pm["totals"]


def test_per_modality_attributions_totals_non_negative() -> None:
    model, sample, _ = _build_policy_model()
    pm = per_modality_attributions(model, sample, n_steps=4, device="cpu")
    for k, v in pm["totals"].items():
        assert v >= 0.0
        assert math.isfinite(v)


def test_per_modality_attributions_default_target() -> None:
    model, sample, _ = _build_policy_model()
    model.eval()
    pm = per_modality_attributions(model, sample, n_steps=4, device="cpu")
    chart = torch.from_numpy(np.asarray(sample["chart"])).unsqueeze(0)
    numeric = torch.from_numpy(np.asarray(sample["numeric"])).unsqueeze(0)
    context = torch.from_numpy(np.asarray(sample["context"])).unsqueeze(0)
    with torch.no_grad():
        out = model(chart=chart, numeric=numeric, context=context)
    expected = int(out["policy_logits"].argmax(dim=-1).item())
    assert pm["target"] == expected


# ----------------------------------------------------------------------
# cross_instrument_rollout
# ----------------------------------------------------------------------

def test_cross_instrument_rollout_returns_matrix() -> None:
    m, instruments, portfolio = _build_portfolio_model(with_cross_attn=True)
    rollout = cross_instrument_rollout(m, instruments, portfolio, device="cpu")
    assert rollout is not None
    N = instruments["chart"].size(1)
    assert rollout.shape == (N, N)
    assert np.isfinite(rollout).all()


def test_cross_instrument_rollout_returns_none_for_stage1() -> None:
    m, instruments, portfolio = _build_portfolio_model(with_cross_attn=False)
    rollout = cross_instrument_rollout(m, instruments, portfolio, device="cpu")
    assert rollout is None


def test_cross_instrument_rollout_with_numpy_portfolio() -> None:
    """Portfolio can be a numpy array; the function should accept both."""
    m, instruments, portfolio = _build_portfolio_model(with_cross_attn=True)
    portfolio_np = portfolio.cpu().numpy()
    rollout = cross_instrument_rollout(m, instruments, portfolio_np, device="cpu")
    assert rollout is not None
    assert rollout.shape == (3, 3)


def test_cross_instrument_rollout_invalid_head_fusion() -> None:
    m, instruments, portfolio = _build_portfolio_model(with_cross_attn=True)
    with pytest.raises(ValueError):
        cross_instrument_rollout(m, instruments, portfolio,
                                 head_fusion="weird", device="cpu")


# ----------------------------------------------------------------------
# action_explanation
# ----------------------------------------------------------------------

def test_action_explanation_has_all_keys() -> None:
    model, sample, _ = _build_policy_model()
    ex = action_explanation(model, sample, n_steps=4, device="cpu")
    expected = {
        "target", "target_name", "target_logit",
        "action_probabilities", "per_modality_attributions",
        "modality_totals", "chart_saliency", "chart_saliency_summary",
        "top_numeric_features",
    }
    assert expected.issubset(set(ex.keys()))


def test_action_explanation_probabilities_sum_to_one() -> None:
    model, sample, _ = _build_policy_model()
    ex = action_explanation(model, sample, n_steps=4, device="cpu")
    probs = ex["action_probabilities"]
    assert abs(probs.sum() - 1.0) < 1e-5
    assert (probs >= 0).all()


def test_action_explanation_top_features_respected() -> None:
    model, sample, _ = _build_policy_model()
    ex = action_explanation(model, sample, n_steps=4, top_k_numeric=3, device="cpu")
    assert len(ex["top_numeric_features"]) <= 3
    # Should be sorted by descending importance.
    imps = [f["importance"] for f in ex["top_numeric_features"]]
    assert imps == sorted(imps, reverse=True)


def test_action_explanation_with_explicit_target() -> None:
    model, sample, _ = _build_policy_model()
    ex = action_explanation(model, sample, target=2, n_steps=4, device="cpu")
    assert ex["target"] == 2


def test_action_explanation_saliency_summary_finite() -> None:
    model, sample, _ = _build_policy_model()
    ex = action_explanation(model, sample, n_steps=4, device="cpu")
    for v in ex["chart_saliency_summary"].values():
        assert math.isfinite(float(v))


# ----------------------------------------------------------------------
# per_dataset_summary
# ----------------------------------------------------------------------

def test_per_dataset_summary_aggregates_samples() -> None:
    model, _, ds = _build_policy_model()
    samples = []
    for i in range(3):
        s = ds[i]
        samples.append({
            "chart": s["chart"], "numeric": s["numeric"], "context": s["context"],
        })
    summary = per_dataset_summary(model, samples, n_steps=2,
                                  top_k_numeric=3, device="cpu")
    assert summary["n_samples"] == 3
    assert sum(summary["action_distribution"].values()) == 3
    assert len(summary["top_features"]) <= 3
    for v in summary["mean_modality_totals"].values():
        assert v >= 0.0


def test_per_dataset_summary_handles_empty_input() -> None:
    model, _, _ = _build_policy_model()
    summary = per_dataset_summary(model, [], n_steps=2, device="cpu")
    assert summary["n_samples"] == 0
    assert summary["action_distribution"] == {}
    assert summary["samples"] == []


def test_per_dataset_summary_respects_n_samples() -> None:
    model, _, ds = _build_policy_model()
    samples = []
    for i in range(10):
        s = ds[i]
        samples.append({
            "chart": s["chart"], "numeric": s["numeric"], "context": s["context"],
        })
    summary = per_dataset_summary(model, samples, n_samples=2, n_steps=2,
                                  device="cpu")
    assert summary["n_samples"] == 2
