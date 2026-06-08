"""Unit tests for CrossInstrumentAttention and Stage-2 portfolio policy."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from zhisa.models.cross_instrument_attention import (
    CrossInstrumentAttention,
    CrossInstrumentConfig,
)
from zhisa.models.portfolio_policy import (
    PortfolioPolicyConfig,
    PortfolioPolicyNetwork,
)


# ----------------------------------------------------------------------
# CrossInstrumentAttention
# ----------------------------------------------------------------------

def _default_cfg(**kw) -> CrossInstrumentConfig:
    base = dict(embed_dim=64, depth=2, n_heads=4, n_instruments_max=8, dropout=0.0)
    base.update(kw)
    return CrossInstrumentConfig(**base)


def test_cross_attn_config_validates() -> None:
    with pytest.raises(ValueError):
        CrossInstrumentConfig(embed_dim=0)
    with pytest.raises(ValueError):
        CrossInstrumentConfig(embed_dim=64, n_heads=3)
    with pytest.raises(ValueError):
        CrossInstrumentConfig(depth=-1)


def test_cross_attn_forward_shape() -> None:
    m = CrossInstrumentAttention(_default_cfg())
    x = torch.randn(2, 4, 64)
    y = m(x)
    assert y.shape == x.shape


def test_cross_attn_with_portfolio_bias() -> None:
    m = CrossInstrumentAttention(_default_cfg())
    m.set_portfolio_dim(32)
    x = torch.randn(2, 4, 64)
    p = torch.randn(2, 32)
    y = m(x, portfolio=p)
    assert y.shape == x.shape


def test_cross_attn_depth_zero_is_identity_plus_id() -> None:
    """With depth=0 the only learnable part is the instrument-id embedding."""
    m = CrossInstrumentAttention(_default_cfg(depth=0, use_instrument_id=True))
    x = torch.randn(1, 4, 64)
    y = m(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)


def test_cross_attn_permutation_equivariant() -> None:
    """Shuffling instruments before the block shuffles the outputs the same way.

    Note: with ``use_instrument_id=True`` the model is *not*
    permutation-equivariant (the id-embedding breaks the symmetry
    on purpose, so the model can distinguish instrument 0 from
    instrument 1). We disable the id here to isolate the
    self-attention block's intrinsic equivariance.
    """
    m = CrossInstrumentAttention(_default_cfg(use_instrument_id=False))
    m.eval()
    x = torch.randn(1, 5, 64)
    perm = [3, 1, 4, 0, 2]
    y1 = m(x)
    y2 = m(x[:, perm])
    assert torch.allclose(y1[:, perm], y2, atol=1e-5)


def test_cross_attn_per_instrument_changes_with_others() -> None:
    """Cross-coupling: gradients must propagate to *all* instruments'
    outputs, even when only one instrument's input is perturbed.

    This is a weaker but more reliable check than asking the
    *output* to change a lot: with pre-norm transformers the
    LayerNorm at the end normalises the perturbation, so we check
    that the focal instrument's *gradient* depends on a remote
    instrument's input.
    """
    torch.manual_seed(0)
    m = CrossInstrumentAttention(_default_cfg(use_instrument_id=False, depth=2))
    m.eval()
    x = torch.randn(1, 3, 64, requires_grad=True)
    y = m(x)
    # Sum of focal-instrument logits.
    loss = y[:, 0].sum()
    loss.backward()
    # The gradient on the *non-focal* instruments must be non-zero,
    # proving that information from those instruments influenced
    # the focal output.
    g_other = x.grad[0, 1:].abs().sum().item()
    assert g_other > 1e-4, f"expected cross-coupling, got grad={g_other}"


def test_cross_attn_supports_various_n() -> None:
    m = CrossInstrumentAttention(_default_cfg(n_instruments_max=16))
    for n in (1, 2, 4, 8):
        x = torch.randn(2, n, 64)
        y = m(x)
        assert y.shape == (2, n, 64)


def test_cross_attn_rejects_too_many_instruments() -> None:
    m = CrossInstrumentAttention(_default_cfg(n_instruments_max=4))
    x = torch.randn(1, 8, 64)
    with pytest.raises(ValueError):
        m(x)


def test_cross_attn_rejects_wrong_dim() -> None:
    m = CrossInstrumentAttention(_default_cfg(embed_dim=64))
    with pytest.raises(ValueError):
        m(torch.randn(2, 4, 32))  # wrong embed dim


def test_cross_attn_invalid_input_rank() -> None:
    m = CrossInstrumentAttention(_default_cfg())
    with pytest.raises(ValueError):
        m(torch.randn(4, 64))  # missing N


# ----------------------------------------------------------------------
# PortfolioPolicyNetwork with cross-attention (Stage 2)
# ----------------------------------------------------------------------

def _obs(B=2, N=3, H=24, W=24, F=32, C=10, T=16, D=32) -> dict:
    return {
        "chart": torch.randn(B, N, 3, H, W),
        "numeric": torch.randn(B, N, T, F),
        "context": torch.randn(B, N, C),
    }


def test_portfolio_policy_stage1_default_no_cross_attn() -> None:
    """Default config (cross_attn_depth=0) must not instantiate cross-attn."""
    m = PortfolioPolicyNetwork(PortfolioPolicyConfig(
        n_instruments=3, embed_dim=32, fusion_hidden=32,
    ))
    assert m.cross_attn is None
    y = m(_obs(N=3, D=32), torch.randn(2, 32))
    assert y["action_logits"].shape == (2, 3, 9)
    assert y["value"].shape == (2,)


def test_portfolio_policy_stage2_has_cross_attn() -> None:
    m = PortfolioPolicyNetwork(PortfolioPolicyConfig(
        n_instruments=3, embed_dim=32, fusion_hidden=32,
        cross_attn_depth=2, cross_attn_heads=4,
    ))
    assert m.cross_attn is not None
    y = m(_obs(N=3, D=32), torch.randn(2, 32))
    assert y["action_logits"].shape == (2, 3, 9)


def test_portfolio_policy_stage2_changes_outputs_vs_stage1() -> None:
    """With a random init, Stage 1 and Stage 2 must produce different logits."""
    torch.manual_seed(123)
    cfg1 = PortfolioPolicyConfig(n_instruments=3, embed_dim=32, fusion_hidden=32,
                                 cross_attn_depth=0)
    m1 = PortfolioPolicyNetwork(cfg1)
    m1.eval()
    cfg2 = PortfolioPolicyConfig(n_instruments=3, embed_dim=32, fusion_hidden=32,
                                 cross_attn_depth=2, cross_attn_heads=4)
    m2 = PortfolioPolicyNetwork(cfg2)
    m2.eval()
    obs = _obs(N=3, D=32)
    p = torch.randn(2, 32)
    y1 = m1(obs, p)
    y2 = m2(obs, p)
    assert not torch.allclose(y1["action_logits"], y2["action_logits"], atol=1e-4)
    assert not torch.allclose(y1["value"], y2["value"], atol=1e-4)


def test_portfolio_policy_stage2_uses_instrument_id_when_enabled() -> None:
    """Without instrument-id embedding, permutation of inputs would be a symmetry.

    We use the id to break the symmetry so the model can distinguish
    instruments by position.
    """
    cfg = PortfolioPolicyConfig(
        n_instruments=3, embed_dim=32, fusion_hidden=32,
        cross_attn_depth=2, cross_attn_heads=4,
    )
    m = PortfolioPolicyNetwork(cfg)
    assert m.cross_attn.instrument_id is not None
    assert m.cross_attn.instrument_id.num_embeddings >= 3


def test_portfolio_policy_stage2_responds_to_other_instruments() -> None:
    """Mutating an *unrelated* instrument must change the focal instrument's
    action logits. This is the whole point of cross-attention.
    """
    torch.manual_seed(7)
    cfg = PortfolioPolicyConfig(
        n_instruments=3, embed_dim=32, fusion_hidden=32,
        cross_attn_depth=2, cross_attn_heads=4,
    )
    m = PortfolioPolicyNetwork(cfg)
    m.eval()
    obs1 = _obs(B=1, N=3, D=32)
    obs2 = {k: v.clone() for k, v in obs1.items()}
    # Mutate the LAST instrument (index 2) only.
    obs2["chart"][:, 2] += 5.0
    p = torch.randn(1, 32)
    y1 = m(obs1, p)
    y2 = m(obs2, p)
    # The focal instrument (index 0) should change because the model
    # attends to instrument 2.
    assert not torch.allclose(y1["action_logits"][:, 0], y2["action_logits"][:, 0], atol=1e-5)
    assert not torch.allclose(y1["value"], y2["value"], atol=1e-5)


def test_portfolio_policy_cross_attn_validates_config() -> None:
    with pytest.raises(ValueError):
        PortfolioPolicyConfig(n_instruments=2, embed_dim=32, fusion_hidden=32,
                              cross_attn_depth=-1)
    with pytest.raises(ValueError):
        PortfolioPolicyConfig(n_instruments=2, embed_dim=32, fusion_hidden=32,
                              cross_attn_depth=1, cross_attn_heads=3)


def test_portfolio_policy_save_load_roundtrip_with_cross_attn(tmp_path) -> None:
    cfg = PortfolioPolicyConfig(
        n_instruments=3, embed_dim=32, fusion_hidden=32,
        cross_attn_depth=2, cross_attn_heads=4,
    )
    m = PortfolioPolicyNetwork(cfg)
    m.eval()
    out = tmp_path / "cross_attn_test.pt"
    m.save(str(out))
    m2 = PortfolioPolicyNetwork.load(str(out))
    m2.eval()
    assert m2.cfg.cross_attn_depth == 2
    obs = _obs(N=3, D=32)
    p = torch.randn(2, 32)
    y1 = m(obs, p)
    y2 = m2(obs, p)
    assert torch.allclose(y1["action_logits"], y2["action_logits"])
    assert torch.allclose(y1["value"], y2["value"])


def test_portfolio_policy_stage2_gradient_flows() -> None:
    """Loss gradients must reach the cross-attention parameters."""
    cfg = PortfolioPolicyConfig(
        n_instruments=3, embed_dim=32, fusion_hidden=32,
        cross_attn_depth=2, cross_attn_heads=4,
    )
    m = PortfolioPolicyNetwork(cfg)
    obs = _obs(N=3, D=32)
    p = torch.randn(2, 32)
    y = m(obs, p)
    loss = y["action_logits"].sum() + y["value"].sum()
    loss.backward()
    has_grad = any(
        (p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in m.cross_attn.parameters()
    )
    assert has_grad
