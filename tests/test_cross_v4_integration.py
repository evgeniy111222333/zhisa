"""Tests: cross-asset v4 additions (#1 resid alpha, #2 vol-index, #3 lead-lag)
and #4 additive corr-bias in cross-instrument attention."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.data.cross_asset import (
    build_market_index,
    enrich_frame,
    enrich_market_frames,
    symbol_logret,
)
from zhisa.models.cross_instrument_attention import CrossInstrumentAttention, CrossInstrumentConfig


def _frame(n=600, drift=0.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(drift + rng.standard_normal(n) * noise))
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": rng.uniform(100, 200, n)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# #1 residual alpha
# ---------------------------------------------------------------------------


def test_resid_alpha_columns_and_semantics():
    rb = np.random.default_rng(0).standard_normal(800) * 0.02
    ra = 0.8 * rb + np.random.default_rng(1).standard_normal(800) * 0.001
    idx = pd.date_range("2024-01-01", periods=800, freq="1h", tz="UTC")
    fa = pd.DataFrame({"open": 100, "high": 100, "low": 100,
                       "close": 100 * np.exp(np.cumsum(ra)), "volume": 1.0}, index=idx)
    fb = pd.DataFrame({"open": 100, "high": 100, "low": 100,
                       "close": 100 * np.exp(np.cumsum(rb)), "volume": 1.0}, index=idx)
    fb2 = fb.copy()
    fb2["volume"] = 1.0
    ref = symbol_logret(fb)
    e = enrich_frame(fa, ref, windows=(256,), with_resid_alpha=True,
                     ref_logrets_wide=pd.DataFrame({"B": ref}))
    for c in ("resid_alpha_256",):
        assert c in e.columns, c
    # resid must be small when beta=0.8 and market drives the asset:
    med = float(e["resid_alpha_256"].dropna().abs().median())
    assert med < 0.01, med
    # causal: tampering future bars does not change past resid
    past = float(e.loc[e.index[300], "resid_alpha_256"])
    tam = fa.copy()
    tam.iloc[500:, 3] *= 1.5
    e2 = enrich_frame(tam, ref, windows=(256,), with_resid_alpha=True,
                      ref_logrets_wide=pd.DataFrame({"B": ref}))
    assert float(e2.loc[e2.index[300], "resid_alpha_256"]) == pytest.approx(past, abs=1e-12)


# ---------------------------------------------------------------------------
# #2 volume-weighted index
# ---------------------------------------------------------------------------


def test_volume_weighted_index_scale_preserved():
    """Bug #1 regression: vol-intensity must keep the market signal scale
    (~mean 1.0), NOT shrink it by the window size (sum) — otherwise
    rel_logret_vw_1 collapses to ~ lr and the index is lost."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=700, freq="1h", tz="UTC")
    mk = rng.standard_normal(700) * 0.005
    frames = {}
    for s, b in (("A", 1.0), ("B", 0.0)):
        close = 100 * np.exp(np.cumsum(b * mk + rng.standard_normal(700) * 0.001))
        vol = rng.uniform(50, 200, 700)
        vol[::5] *= 8.0  # occasional volume spikes
        frames[s] = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": vol}, index=idx)
    e = enrich_market_frames(frames, with_volume_ratios=True, with_vol_index=True)
    vw = e["A"]["market_index_vw_logret"].dropna()
    ref = e["A"].index  # reference market = B's logret (driven by mk)
    scale = float(vw.abs().mean())
    # market logret std is ~0.005; a 256x shrink would give ~2e-5
    assert scale > 4e-4, f"vw index scale collapsed: {scale}"
    # weight series (intensity) mean is ~1.0, not ~1/256
    wmean = float(e["A"]["vw_weight_256"].dropna().mean())
    assert 0.5 < wmean < 2.0, f"vw weight scale wrong: {wmean}"


def test_build_market_index_two_symbol_universe_not_nan():
    """Bug #2 regression: with N=2 ("others" = 1 symbol) the index must not
    be all-NaN — required periods can never exceed the reference count."""
    a, b = _frame(seed=1), _frame(seed=2)
    idx = build_market_index({"B": symbol_logret(b)}, min_periods=2)
    assert idx.notna().sum() > 0.9 * len(idx)  # not a dead column


def test_volume_weighted_index_shifts_market():
    frames = {"A": _frame(seed=1), "B": _frame(seed=2), "C": _frame(seed=3)}
    frames["B"]["volume"] *= 20.0  # B dominates liquidity
    eq = enrich_market_frames(frames, with_volume_ratios=True, with_vol_index=True,
                              with_resid_alpha=False)
    assert "market_index_vw_logret" in eq["A"].columns
    assert "rel_logret_vw_1" in eq["A"].columns
    assert "beta_vw_256" in eq["A"].columns and "corr_vw_256" in eq["A"].columns
    # B's volume share must register as the top weight
    w = eq["A"]["vw_weight_256"].dropna().median()
    assert 0.0 < w < 1.0
    # causality: future tampering of a reference volume cannot affect past vw
    pos = eq["A"]["rel_logret_vw_1"].index[200]
    old = float(eq["A"].loc[pos, "rel_logret_vw_1"])
    tam = {k: v.copy() for k, v in frames.items()}
    tam["B"].iloc[450:, -1] *= 3.0
    e2 = enrich_market_frames(tam, with_volume_ratios=True, with_vol_index=True)
    assert float(e2["A"].loc[pos, "rel_logret_vw_1"]) == pytest.approx(old, abs=1e-12)


# ---------------------------------------------------------------------------
# #3 lead-lag (causal lags only)
# ---------------------------------------------------------------------------


def test_lead_lag_columns_and_rejects_lookahead():
    frames = {"A": _frame(seed=1), "B": _frame(seed=2)}
    e = enrich_market_frames(frames, lead_lag_lags=(0, 1, 2))
    assert "leadlag_avg_0" in e["A"].columns
    assert "leadlag_avg_1" in e["A"].columns and "leadlag_avg_2" in e["A"].columns
    # negative lag (asset leads -> future info) must be rejected
    with pytest.raises(ValueError, match="causal"):
        enrich_frame(frames["A"], symbol_logret(frames["B"]),
                     lead_lag_lags=(-1,))
    # causal (no lookahead): lag values use only past B bars
    old = float(e["A"].loc[e["A"].index[300], "leadlag_B_1"])
    tam = frames["A"].copy()
    tam.iloc[500:, 3] *= 1.5
    e2 = enrich_market_frames({"A": tam, "B": frames["B"]}, lead_lag_lags=(0, 1, 2))
    assert float(e2["A"].loc[e2["A"].index[300], "leadlag_B_1"]) == pytest.approx(old, abs=1e-12)


# ---------------------------------------------------------------------------
# #4 corr-bias in cross-instrument attention
# ---------------------------------------------------------------------------


def _attn(**kw) -> CrossInstrumentAttention:
    defaults = dict(embed_dim=32, depth=1, n_heads=4, dropout=0.0,
                    use_instrument_id=False)
    defaults.update(kw)
    return CrossInstrumentAttention(CrossInstrumentConfig(**defaults))


def test_biased_attention_applies_additive_bias():
    m = _attn(use_attention_bias=True, bias_gate=5.0)
    m.eval()
    x = torch.randn(2, 4, 32)
    bias = torch.zeros(2, 4, 4)
    bias[:, 0, 1] = 5.0  # force instrument 0 to attend instrument 1
    with torch.no_grad():
        out = m(x, bias=bias)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # without bias the module REQUIRES it (guard)
    with pytest.raises(ValueError, match="bias"):
        m(x)
    # wrong shape rejected
    with pytest.raises(ValueError):
        m(x, bias=torch.zeros(2, 3, 3))


def test_biased_vs_unbiased_outputs_differ():
    m = _attn(use_attention_bias=True, bias_gate=50.0)
    m.eval()
    x = torch.randn(3, 5, 32)
    bias = torch.zeros(3, 5, 5)
    bias[:, :, 2] = 10.0
    with torch.no_grad():
        o_zero_bias = m(x, bias=torch.zeros(3, 5, 5))
        o_strong = m(x, bias=bias)
    assert float((o_strong - o_zero_bias).abs().max().item()) > 1e-3  # bias changes output
    # zero bias within the SAME module equals gate-0 (recomputed)
    cmp = _attn(use_attention_bias=True, bias_gate=50.0)
    cmp.load_state_dict(m.state_dict())
    cmp.eval()
    with torch.no_grad():
        cmp.cfg.bias_gate = 0.0
        o_gate0 = cmp(x, bias=bias)
    assert float((o_gate0 - o_zero_bias).abs().max().item()) < 1e-6


def test_biased_gate0_bit_identical_to_unbiased_full_depth():
    """Regression: with bias enabled and gate=0, the FULL encoder stack
    (all layers + FFN) must reproduce the unbiased path EXACTLY — the
    previous implementation skipped FFN/layers beyond depth=1."""
    for depth in (1, 2, 3):
        mb = _attn(use_attention_bias=True, bias_gate=0.0, depth=depth)
        mu = _attn(use_attention_bias=False, depth=depth)
        mb.load_state_dict(mu.state_dict(), strict=False)
        mb.eval()
        mu.eval()
        x = torch.randn(2, 4, 32)
        bias = torch.randn(2, 4, 4)
        with torch.no_grad():
            ob = mb(x, bias=bias)
            ou = mu(x)
        assert torch.allclose(ob, ou, atol=1e-6), f"depth={depth} biased(gate=0) != unbiased"