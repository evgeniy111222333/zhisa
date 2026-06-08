"""Tests for the CVaR (Expected Shortfall) module."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zhisa.risk.cvar import cvar_constraint_violation, cvar_numpy, cvar_torch


def test_cvar_numpy_basic():
    r = np.array([1.0, 2.0, 3.0, 4.0, -5.0, -10.0, 0.5, 0.2, 0.1, 0.0], dtype=np.float32)
    c = cvar_numpy(r, alpha=0.2)
    sorted_r = np.sort(r)
    expected = sorted_r[:2].mean()
    assert c == pytest.approx(float(expected), abs=1e-5)


def test_cvar_numpy_alpha_one_returns_mean():
    r = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    c = cvar_numpy(r, alpha=1.0)
    assert c == pytest.approx(float(r.mean()), abs=1e-5)


def test_cvar_numpy_alpha_clamps_to_one_over_n():
    r = np.array([1.0, 2.0], dtype=np.float32)
    c = cvar_numpy(r, alpha=0.01)
    assert c == pytest.approx(float(np.min(r)), abs=1e-5)


def test_cvar_numpy_empty():
    assert cvar_numpy(np.zeros(0), alpha=0.1) == 0.0


def test_cvar_torch_matches_numpy():
    rng = np.random.default_rng(0)
    r_np = rng.standard_normal(50).astype(np.float32)
    r_t = torch.from_numpy(r_np)
    a = 0.1
    np.testing.assert_allclose(
        cvar_numpy(r_np, a), float(cvar_torch(r_t, a).item()), atol=1e-5
    )


def test_cvar_torch_gradients_flow():
    r = torch.tensor([1.0, 2.0, -3.0, -4.0, 5.0], requires_grad=True)
    c = cvar_torch(r, alpha=0.4)
    c.backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()


def test_cvar_torch_empty():
    r = torch.zeros(0, dtype=torch.float32)
    c = cvar_torch(r, alpha=0.1)
    assert c.item() == 0.0


def test_cvar_constraint_violation_zero_when_safe():
    r = np.array([0.1, 0.2, 0.05, 0.3, 0.15], dtype=np.float32)
    assert cvar_constraint_violation(r, alpha=0.2, threshold=10.0) == 0.0


def test_cvar_constraint_violation_positive_when_violated():
    r = np.array([-1.0, -2.0, 0.1, 0.2, 0.3], dtype=np.float32)
    v = cvar_constraint_violation(r, alpha=0.4, threshold=0.5)
    assert v > 0.0
    expected_cvar = cvar_numpy(r, alpha=0.4)
    assert v == pytest.approx(-expected_cvar - 0.5, abs=1e-5)


def test_cvar_constraint_violation_exact_at_boundary():
    r = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    v = cvar_constraint_violation(r, alpha=1.0, threshold=1.0)
    assert v == pytest.approx(0.0, abs=1e-5)
