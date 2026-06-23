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
