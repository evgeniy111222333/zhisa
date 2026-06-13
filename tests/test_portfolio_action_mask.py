"""Tests for the gross-leverage action mask utility."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.env.actions import DiscreteAction
from zhisa.env.portfolio_action_mask import (
    ACTION_TO_TARGET_FRACTION,
    action_to_target_fraction,
    compute_gross_leverage_mask,
    mask_logits,
)


def test_action_to_target_fraction_table():
    assert action_to_target_fraction(int(DiscreteAction.SKIP)) == 0.0
    assert action_to_target_fraction(int(DiscreteAction.SKIP), 0.75) == 0.75
    assert action_to_target_fraction(int(DiscreteAction.SKIP), -0.5) == -0.5
    assert action_to_target_fraction(int(DiscreteAction.LONG_25)) == 0.25
    assert action_to_target_fraction(int(DiscreteAction.LONG_100)) == 1.0
    assert action_to_target_fraction(int(DiscreteAction.SHORT_50)) == -0.5
    assert action_to_target_fraction(int(DiscreteAction.CLOSE)) == 0.0


def test_partial_close_is_position_relative():
    assert action_to_target_fraction(int(DiscreteAction.PARTIAL_CLOSE), 1.0) == 0.5
    assert action_to_target_fraction(int(DiscreteAction.PARTIAL_CLOSE), -0.8) == -0.4


def test_compute_mask_zero_cap_allows_only_zero_targets():
    positions = np.array([0.0, 0.0], dtype=np.float32)
    mask = compute_gross_leverage_mask(positions, gross_cap=0.0, n_actions_per=9)
    assert mask.shape == (2, 9)
    for a in range(9):
        tgt = action_to_target_fraction(a, 0.0)
        if abs(tgt) < 1e-6:
            assert mask[0, a]
        else:
            assert not mask[0, a]


def test_compute_mask_large_cap_allows_everything():
    positions = np.array([0.0, 0.0], dtype=np.float32)
    mask = compute_gross_leverage_mask(positions, gross_cap=10.0, n_actions_per=9)
    assert mask.all()


def test_compute_mask_with_open_positions():
    positions = np.array([0.5, 0.0], dtype=np.float32)
    mask = compute_gross_leverage_mask(positions, gross_cap=1.0, n_actions_per=9)
    # Instrument 0 is at 0.5; others=0; cap=1.0 -> can go to |target|<=1.0.
    for a in range(9):
        tgt = action_to_target_fraction(a, 0.5)
        if abs(tgt) <= 1.0 + 1e-5:
            assert mask[0, a]
        else:
            assert not mask[0, a]
    # Instrument 1 is at 0; others=0.5; cap=1.0 -> can go to |target|<=0.5.
    for a in range(9):
        tgt = action_to_target_fraction(a, 0.0)
        if abs(tgt) <= 0.5 + 1e-5:
            assert mask[1, a]
        else:
            assert not mask[1, a]


def test_compute_mask_conservative_for_multi_instrument():
    """When others already at cap, the only allowed actions are those that don't breach cap.

    With positions=[0.5, 0.5] and cap=1.0, any action that would move the
    instrument to |target| > 0.5 must be rejected (would push gross past 1.0).
    """
    positions = np.array([0.5, 0.5], dtype=np.float32)
    mask = compute_gross_leverage_mask(positions, gross_cap=1.0, n_actions_per=9)
    for i in range(2):
        for a in range(9):
            tgt = action_to_target_fraction(a, positions[i])
            new_gross = abs(tgt) + 0.5
            if new_gross > 1.0 + 1e-6:
                assert not mask[i, a], f"mask[{i},{a}] should be False for tgt={tgt}"
            else:
                assert mask[i, a], f"mask[{i},{a}] should be True for tgt={tgt}"


def test_compute_mask_empty():
    mask = compute_gross_leverage_mask(np.zeros(0), gross_cap=1.0, n_actions_per=9)
    assert mask.shape == (0, 9)


def test_mask_logits_unchanged_when_all_true():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, True]])
    out = mask_logits(logits, mask)
    np.testing.assert_allclose(out.numpy(), logits.numpy(), atol=1e-5)


def test_mask_logits_zeros_invalid_actions():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    out = mask_logits(logits, mask)
    assert out[0, 1].item() <= -1e8
    np.testing.assert_allclose(out[0, [0, 2]].numpy(), logits[0, [0, 2]].numpy(), atol=1e-5)


def test_mask_logits_handles_float_mask():
    logits = torch.zeros(2, 3)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    out = mask_logits(logits, mask)
    assert out[0, 1].item() <= -1e8
    assert out[1, 0].item() <= -1e8


def test_action_to_target_fraction_table_consistency():
    """All 9 actions should be mapped in the table."""
    assert len(ACTION_TO_TARGET_FRACTION) == 9
    for a in range(9):
        _ = action_to_target_fraction(a, 0.0)
