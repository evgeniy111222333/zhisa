"""Regression tests for multi-task loss edge cases."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from zhisa.training.losses import LossWeights, MultiTaskLoss


def _minimal_outputs(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = logits.shape[0]
    return {
        "direction": torch.zeros(batch, 3),
        "policy_logits": logits,
    }


def _minimal_targets(actions: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "label_dir": torch.zeros_like(actions),
        "action": actions,
    }


def test_policy_focal_loss_uses_weighted_mean_without_double_weighting():
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor([0, 1, 1], dtype=torch.long)
    weights = torch.tensor([0.5, 2.0], dtype=torch.float32)
    loss = MultiTaskLoss(
        LossWeights(direction=0.0, policy=1.0),
        policy_class_weights=weights,
        policy_focal_gamma=1.5,
    )

    got = loss(_minimal_outputs(logits), _minimal_targets(actions))["policy"]
    ce = F.cross_entropy(logits, actions, reduction="none")
    probs = torch.softmax(logits, dim=-1).gather(1, actions[:, None]).squeeze(1)
    focal = (1.0 - probs).pow(1.5)
    sample_weights = weights[actions]
    expected = (sample_weights * focal * ce).sum() / sample_weights.sum()

    torch.testing.assert_close(got, expected)


def test_policy_class_weight_setter_preserves_buffer_state_dict():
    loss = MultiTaskLoss(LossWeights(policy=1.0))
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    loss.set_policy_class_weights(weights)

    assert "policy_class_weights" in loss._buffers
    assert loss.policy_class_weights.dtype == torch.float32
    torch.testing.assert_close(
        loss.state_dict()["policy_class_weights"],
        weights.float(),
    )


def test_return_target_scale_amplifies_regression_loss_without_changing_optimum():
    outputs = {
        "direction": torch.zeros(2, 3),
        "return_pred": torch.tensor([0.0, 0.01], dtype=torch.float32),
    }
    targets = {
        "label_dir": torch.zeros(2, dtype=torch.long),
        "label_ret": torch.tensor([0.01, -0.01], dtype=torch.float32),
    }
    unscaled = MultiTaskLoss(
        LossWeights(direction=0.0, return_pred=1.0, value=0.0, uncertainty=0.0),
        return_target_scale=1.0,
    )
    scaled = MultiTaskLoss(
        LossWeights(direction=0.0, return_pred=1.0, value=0.0, uncertainty=0.0),
        return_target_scale=100.0,
    )

    assert scaled(outputs, targets)["return_pred"] > unscaled(outputs, targets)["return_pred"] * 100

    matched = {
        "direction": torch.zeros(2, 3),
        "return_pred": targets["label_ret"].clone(),
    }
    torch.testing.assert_close(
        scaled(matched, targets)["return_pred"],
        torch.tensor(0.0),
    )


def test_return_direction_auxiliary_penalizes_wrong_sign():
    targets = {
        "label_dir": torch.tensor([1, -1], dtype=torch.long),
        "label_ret": torch.tensor([0.01, -0.01], dtype=torch.float32),
    }
    loss = MultiTaskLoss(
        LossWeights(direction=0.0, return_pred=1.0, value=0.0, uncertainty=0.0),
        return_target_scale=100.0,
        return_direction_weight=0.5,
    )
    right = {
        "direction": torch.zeros(2, 3),
        "return_pred": torch.tensor([0.01, -0.01], dtype=torch.float32),
    }
    wrong = {
        "direction": torch.zeros(2, 3),
        "return_pred": torch.tensor([-0.01, 0.01], dtype=torch.float32),
    }

    assert loss(wrong, targets)["return_pred"] > loss(right, targets)["return_pred"]


def test_return_corr_auxiliary_penalizes_reversed_ranking():
    targets = {
        "label_ret": torch.tensor([-0.02, -0.01, 0.01, 0.03], dtype=torch.float32),
    }
    loss = MultiTaskLoss(
        LossWeights(direction=0.0, return_pred=1.0, value=0.0, uncertainty=0.0),
        return_target_scale=100.0,
        return_corr_weight=0.5,
    )
    right = {
        "direction": torch.zeros(4, 3),
        "return_pred": torch.tensor([-0.02, -0.01, 0.01, 0.03], dtype=torch.float32),
    }
    reversed_order = {
        "direction": torch.zeros(4, 3),
        "return_pred": torch.tensor([0.03, 0.01, -0.01, -0.02], dtype=torch.float32),
    }

    assert loss(reversed_order, targets)["return_pred"] > loss(right, targets)["return_pred"]
    torch.testing.assert_close(
        loss(right, targets)["return_pred"],
        torch.tensor(0.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_multi_horizon_direction_and_return_losses_are_active():
    outputs = {
        "direction": torch.zeros(2, 3),
        "direction_multi": torch.tensor(
            [
                [[4.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
                [[0.0, 4.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "return_multi": torch.tensor(
            [[-0.01, 0.02], [0.0, -0.015]],
            dtype=torch.float32,
        ),
    }
    targets = {
        "label_dir": torch.zeros(2, dtype=torch.long),
        "label_dir_multi": torch.tensor([[-1, 1], [0, -1]], dtype=torch.long),
        "label_ret_multi": torch.tensor(
            [[-0.01, 0.02], [0.0, -0.015]],
            dtype=torch.float32,
        ),
    }
    loss = MultiTaskLoss(
        LossWeights(
            direction=0.0,
            direction_multi=1.0,
            return_multi=1.0,
            return_pred=0.0,
            value=0.0,
            uncertainty=0.0,
        ),
        return_target_scale=100.0,
        return_direction_weight=0.25,
    )

    good = loss(outputs, targets)
    bad_outputs = dict(outputs)
    bad_outputs["direction_multi"] = torch.zeros_like(outputs["direction_multi"])
    bad_outputs["return_multi"] = -outputs["return_multi"]
    bad = loss(bad_outputs, targets)

    assert "direction_multi" in good
    assert "return_multi" in good
    assert good["return_multi"] < bad["return_multi"]
    assert good["total"] < bad["total"]


def test_multi_horizon_losses_respect_horizon_weights():
    outputs = {
        "direction": torch.zeros(1, 3),
        "direction_multi": torch.tensor(
            [[[4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "return_multi": torch.tensor([[0.01, 0.01, -0.10]], dtype=torch.float32),
    }
    targets = {
        "label_dir": torch.zeros(1, dtype=torch.long),
        "label_dir_multi": torch.tensor([[-1, -1, 1]], dtype=torch.long),
        "label_ret_multi": torch.tensor([[0.01, 0.01, 0.10]], dtype=torch.float32),
    }
    unweighted = MultiTaskLoss(
        LossWeights(
            direction=0.0,
            direction_multi=1.0,
            return_multi=1.0,
            return_pred=0.0,
            value=0.0,
            uncertainty=0.0,
        ),
        return_target_scale=10.0,
    )
    weighted = MultiTaskLoss(
        LossWeights(
            direction=0.0,
            direction_multi=1.0,
            return_multi=1.0,
            return_pred=0.0,
            value=0.0,
            uncertainty=0.0,
        ),
        return_target_scale=10.0,
        direction_multi_horizon_weights=torch.tensor([1.0, 1.0, 0.0]),
        return_multi_horizon_weights=torch.tensor([1.0, 1.0, 0.0]),
    )

    assert weighted(outputs, targets)["direction_multi"] < unweighted(outputs, targets)["direction_multi"]
    assert weighted(outputs, targets)["return_multi"] < unweighted(outputs, targets)["return_multi"]


def test_return_multi_corr_auxiliary_respects_horizon_weights():
    outputs = {
        "direction": torch.zeros(2, 3),
        "return_multi": torch.tensor(
            [[-0.01, 0.02, -0.10], [0.01, 0.03, -0.20]],
            dtype=torch.float32,
        ),
    }
    targets = {
        "label_dir": torch.zeros(2, dtype=torch.long),
        "label_ret_multi": torch.tensor(
            [[-0.01, 0.02, 0.10], [0.01, 0.03, 0.20]],
            dtype=torch.float32,
        ),
    }
    weighted = MultiTaskLoss(
        LossWeights(
            direction=0.0,
            return_multi=1.0,
            return_pred=0.0,
            value=0.0,
            uncertainty=0.0,
        ),
        return_target_scale=100.0,
        return_corr_weight=0.5,
        return_multi_horizon_weights=torch.tensor([1.0, 1.0, 0.0]),
    )
    unweighted = MultiTaskLoss(
        LossWeights(
            direction=0.0,
            return_multi=1.0,
            return_pred=0.0,
            value=0.0,
            uncertainty=0.0,
        ),
        return_target_scale=100.0,
        return_corr_weight=0.5,
    )

    assert weighted(outputs, targets)["return_multi"] < unweighted(outputs, targets)["return_multi"]


def test_volatility_log_auxiliary_penalizes_relative_low_vol_error():
    targets = {
        "label_vol": torch.tensor([0.10, 2.00], dtype=torch.float32),
    }
    raw = MultiTaskLoss(
        LossWeights(direction=0.0, volatility=1.0, value=0.0, uncertainty=0.0),
        volatility_log_weight=0.0,
    )
    relative = MultiTaskLoss(
        LossWeights(direction=0.0, volatility=1.0, value=0.0, uncertainty=0.0),
        volatility_log_weight=1.0,
    )
    good = {"volatility": torch.tensor([0.10, 2.00], dtype=torch.float32)}
    bad_low_vol = {"volatility": torch.tensor([0.20, 2.00], dtype=torch.float32)}

    assert relative(bad_low_vol, targets)["volatility"] > raw(bad_low_vol, targets)["volatility"]
    torch.testing.assert_close(
        relative(good, targets)["volatility"],
        torch.tensor(0.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_volatility_corr_auxiliary_penalizes_wrong_ranking():
    targets = {
        "label_vol": torch.tensor([0.2, 0.5, 1.0, 2.0], dtype=torch.float32),
    }
    loss = MultiTaskLoss(
        LossWeights(direction=0.0, volatility=1.0, value=0.0, uncertainty=0.0),
        volatility_corr_weight=0.5,
    )
    right = {"volatility": torch.tensor([0.2, 0.5, 1.0, 2.0], dtype=torch.float32)}
    reversed_order = {"volatility": torch.tensor([2.0, 1.0, 0.5, 0.2], dtype=torch.float32)}

    assert loss(reversed_order, targets)["volatility"] > loss(right, targets)["volatility"]


def test_target_scales_must_be_positive():
    try:
        MultiTaskLoss(return_target_scale=0.0)
    except ValueError as exc:
        assert "return_target_scale" in str(exc)
    else:
        raise AssertionError("expected invalid return_target_scale to fail")

    try:
        MultiTaskLoss(volatility_log_weight=-0.1)
    except ValueError as exc:
        assert "volatility_log_weight" in str(exc)
    else:
        raise AssertionError("expected invalid volatility_log_weight to fail")

    try:
        MultiTaskLoss(return_corr_weight=-0.1)
    except ValueError as exc:
        assert "return_corr_weight" in str(exc)
    else:
        raise AssertionError("expected invalid return_corr_weight to fail")

    try:
        MultiTaskLoss(volatility_corr_weight=-0.1)
    except ValueError as exc:
        assert "volatility_corr_weight" in str(exc)
    else:
        raise AssertionError("expected invalid volatility_corr_weight to fail")
