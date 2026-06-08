"""Tests for the recurrent latent dynamics module."""
from __future__ import annotations

import pytest
import torch

from zhisa.models.latent_dynamics import LatentDynamics, LatentDynamicsConfig


def test_initial_state_shape():
    cfg = LatentDynamicsConfig(state_dim=8, n_actions=5, hidden_dim=16, n_layers=2)
    ld = LatentDynamics(cfg)
    h = ld.initial_state(batch_size=4)
    assert h.shape == (2, 4, 16)


def test_forward_returns_finite_outputs():
    cfg = LatentDynamicsConfig(state_dim=8, n_actions=5, hidden_dim=16, n_layers=1)
    ld = LatentDynamics(cfg)
    z = torch.randn(3, 8)
    a = torch.tensor([0, 2, 4], dtype=torch.long)
    z_next, h_next, r, d = ld(z, a)
    assert z_next.shape == (3, 8)
    assert h_next.shape == (1, 3, 16)
    assert r.shape == (3,)
    assert d.shape == (3,)
    assert torch.isfinite(z_next).all()
    assert torch.isfinite(r).all()
    assert torch.isfinite(d).all()


def test_state_persistence_via_recurrent_state():
    """Two-step dynamics: the recurrent h carries information forward."""
    cfg = LatentDynamicsConfig(state_dim=8, n_actions=5, hidden_dim=16, n_layers=1)
    torch.manual_seed(0)
    ld = LatentDynamics(cfg)
    z = torch.zeros(1, 8)
    a = torch.zeros(1, dtype=torch.long)
    # First step from a zero hidden state.
    z1, h1, _, _ = ld(z, a, h=None)
    # Second step using the carried h.
    z2, h2, _, _ = ld(z1, a, h=h1)
    # Reset to zero hidden and recompute step 2 — should differ.
    h1_zero = ld.initial_state(1)
    z2_alt, _, _, _ = ld(z1, a, h=h1_zero)
    assert not torch.allclose(z2, z2_alt)


def test_forward_sequence_shapes():
    cfg = LatentDynamicsConfig(state_dim=8, n_actions=5, hidden_dim=16, n_layers=1)
    ld = LatentDynamics(cfg)
    z_seq = torch.randn(2, 5, 8)
    a_seq = torch.randint(0, 5, (2, 5))
    z_next_seq, h_final, r_seq, d_logit_seq = ld.forward_sequence(z_seq, a_seq)
    assert z_next_seq.shape == (2, 5, 8)
    assert h_final.shape == (1, 2, 16)
    assert r_seq.shape == (2, 5)
    assert d_logit_seq.shape == (2, 5)


def test_forward_sequence_gradients_flow():
    cfg = LatentDynamicsConfig(state_dim=8, n_actions=5, hidden_dim=16, n_layers=1)
    ld = LatentDynamics(cfg)
    z = torch.randn(2, 3, 8, requires_grad=True)
    a = torch.randint(0, 5, (2, 3))
    z_next, _, r, d = ld.forward_sequence(z, a)
    loss = z_next.pow(2).mean() + r.pow(2).mean() + d.pow(2).mean()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_config_validates():
    with pytest.raises(ValueError):
        LatentDynamicsConfig(hidden_dim=0)
    with pytest.raises(ValueError):
        LatentDynamicsConfig(n_layers=0)


def test_two_layer_gru():
    cfg = LatentDynamicsConfig(state_dim=8, n_actions=5, hidden_dim=16, n_layers=2)
    ld = LatentDynamics(cfg)
    h = ld.initial_state(2)
    assert h.shape == (2, 2, 16)
    z = torch.randn(2, 8)
    a = torch.tensor([0, 1])
    z_next, h_next, _, _ = ld(z, a, h=h)
    assert z_next.shape == (2, 8)
    assert h_next.shape == (2, 2, 16)
