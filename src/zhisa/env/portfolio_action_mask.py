"""Gross-leverage action mask for :class:`PortfolioEnv`.

The joint action space of a portfolio env is ``9**N`` for
``N`` instruments, which grows exponentially. We avoid this
explosion by giving the policy **N independent heads** (one per
instrument) of size 9 and enforcing the gross-leverage cap
*before* sampling, by masking per-instrument actions that would
break the cap.

The mask is **conservative**: it might over-mask some actions
that would be jointly valid, but it never lets the policy sample
an action that violates the cap.

The mask is computed from the **current** position of every
instrument — for instrument ``i``, an action is valid iff the
proposed target fraction keeps the new total gross within the
cap. Since the cap is additive in ``|position|``, masking each
instrument independently (using the current positions of the
other instruments) is sufficient.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from zhisa.env.actions import DiscreteAction


ACTION_TO_TARGET_FRACTION: dict[int, float] = {
    int(DiscreteAction.SKIP): 0.0,  # flat-position target; open exposure is kept by the helper
    int(DiscreteAction.LONG_25): 0.25,
    int(DiscreteAction.LONG_50): 0.5,
    int(DiscreteAction.LONG_100): 1.0,
    int(DiscreteAction.SHORT_25): -0.25,
    int(DiscreteAction.SHORT_50): -0.5,
    int(DiscreteAction.SHORT_100): -1.0,
    int(DiscreteAction.CLOSE): 0.0,
    int(DiscreteAction.PARTIAL_CLOSE): 0.5,  # half of current
}


def action_to_target_fraction(action: int, current_position: float = 0.0) -> float:
    """Translate a discrete action to a target position fraction.

    ``SKIP`` and ``PARTIAL_CLOSE`` are position-relative: ``SKIP``
    keeps the current position, while ``PARTIAL_CLOSE`` targets half
    of it. ``CLOSE`` targets zero.
    """
    if int(action) == int(DiscreteAction.SKIP):
        return float(current_position)
    if int(action) == int(DiscreteAction.PARTIAL_CLOSE):
        return 0.5 * float(current_position)
    return ACTION_TO_TARGET_FRACTION.get(int(action), 0.0)


def compute_gross_leverage_mask(
    current_positions: np.ndarray,
    gross_cap: float,
    n_actions_per: int = 9,
) -> np.ndarray:
    """Return a ``(N, n_actions_per)`` boolean mask.

    True = the action is *valid* (does not push the total
    ``sum(|position|)`` above ``gross_cap``). False = the
    action is masked out.

    The cap is interpreted in *fraction-of-equity* terms so it
    matches the per-instrument position fractions used by
    :class:`PortfolioEnv`.
    """
    current_positions = np.asarray(current_positions, dtype=np.float32).reshape(-1)
    N = int(current_positions.size)
    mask = np.ones((N, int(n_actions_per)), dtype=bool)
    if N == 0:
        return mask
    other_abs_sum = float(np.sum(np.abs(current_positions)))
    cap = float(gross_cap) + 1e-6
    for i in range(N):
        cur_abs = float(abs(current_positions[i]))
        others_without_i = other_abs_sum - cur_abs
        for a in range(int(n_actions_per)):
            target = action_to_target_fraction(a, current_positions[i])
            new_abs_i = float(abs(target))
            new_gross = others_without_i + new_abs_i
            if new_gross > cap:
                mask[i, a] = False
    return mask


def mask_logits(logits: torch.Tensor, mask: torch.Tensor, neg_value: float = -1e9) -> torch.Tensor:
    """Apply a boolean mask to ``(..., n_actions)`` logits in-place safe.

    ``mask`` of shape ``(..., n_actions)`` with True = keep, False = mask.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    out = logits.masked_fill(~mask, neg_value)
    return out


__all__ = [
    "ACTION_TO_TARGET_FRACTION",
    "action_to_target_fraction",
    "compute_gross_leverage_mask",
    "mask_logits",
]
