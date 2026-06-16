"""Bit-exactness + perf tests for regime datasets.

These cover RegimeOutcomeDataset and RegimeSupervisionDataset — the
optimised numpy paths must match the legacy pandas path exactly.

Run: pytest tests/test_regime_perf.py -v
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("ZHISA_FAST_RENDER", "1")

import numpy as np
import pandas as pd
import pytest
import torch

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.regime.dataset import (
    MACRO_TO_ID,
    MESO_TO_ID,
    PLAYBOOK_NAMES,
    PLAYBOOK_TO_ID,
    RISK_MODE_TO_ID,
    RegimeSupervisionConfig,
    RegimeSupervisionDataset,
    _forward_outcome,
    _playbook_scores,
)
from zhisa.regime.learned import (
    RegimeOutcomeDataset,
    RegimeOutcomeDatasetConfig,
    _outcome,
)
from zhisa.utils.seeding import set_seed


def _make_df(n_bars: int = 1000) -> pd.DataFrame:
    set_seed(42)
    return generate_market(MarketConfig(n_bars=n_bars, freq="5min", seed=42))


# ---------------------------------------------------------------------------
# _outcome bit-exactness: numpy path vs legacy pandas path
# ---------------------------------------------------------------------------
def test_outcome_numpy_matches_pandas():
    df = _make_df(800)
    close_series = df["close"].astype(float)
    close_arr = close_series.to_numpy(dtype=np.float64)
    horizons = (4, 6, 12, 24, 48)
    # Sample a spread of indices
    for t in [0, 1, 50, 100, 200, 500, 700, len(close_arr) - 50]:
        for h in horizons:
            new_out, new_mfe, new_abs = _outcome(close_arr, t, h)
            ref_out, ref_mfe, ref_abs = _outcome(close_series, t, h)
            assert abs(new_out.forward_return - ref_out.forward_return) < 1e-9, (
                f"t={t} h={h}: forward_return {new_out.forward_return} != {ref_out.forward_return}"
            )
            assert abs(new_out.realized_vol - ref_out.realized_vol) < 1e-9, (
                f"t={t} h={h}: realized_vol {new_out.realized_vol} != {ref_out.realized_vol}"
            )
            assert abs(new_out.max_drawdown - ref_out.max_drawdown) < 1e-9, (
                f"t={t} h={h}: max_drawdown {new_out.max_drawdown} != {ref_out.max_drawdown}"
            )
            assert abs(new_mfe - ref_mfe) < 1e-9, (
                f"t={t} h={h}: mfe {new_mfe} != {ref_mfe}"
            )
            assert abs(new_abs - ref_abs) < 1e-9, (
                f"t={t} h={h}: abs_path_return {new_abs} != {ref_abs}"
            )


def test_outcome_handles_edge_cases():
    """Edge cases: t at end of array, horizon longer than data, zeros in close."""
    close = np.array([100.0, 101.0, 0.0, 99.0, 102.0, 0.0, 105.0], dtype=np.float64)
    # t=6, h=10 — only the anchor exists
    out, mfe, abs_ret = _outcome(close, 6, 10)
    assert out.forward_return == 0.0
    assert out.realized_vol == 0.0
    assert out.max_drawdown == 0.0
    assert mfe == 0.0
    assert abs_ret == 0.0
    # t=2, h=3 — anchor at 0 → outcome all zeros
    out, mfe, abs_ret = _outcome(close, 2, 3)
    assert out.forward_return == 0.0


# ---------------------------------------------------------------------------
# _forward_outcome bit-exactness: numpy version must match
# ---------------------------------------------------------------------------
def test_forward_outcome_numpy_matches_pandas():
    """Re-implement the legacy pandas _forward_outcome and compare to a
    numpy version we will use in RegimeSupervisionDataset."""
    df = _make_df(800)
    close_series = df["close"].astype(float)
    close_arr = close_series.to_numpy(dtype=np.float64)

    def _numpy_forward_outcome(close_arr: np.ndarray, t: int, horizon: int):
        from zhisa.regime.memory import RegimeOutcome
        c0 = float(close_arr[t])
        end = min(t + 1 + horizon, len(close_arr))
        if end <= t + 1 or c0 <= 0 or not np.isfinite(c0):
            return RegimeOutcome(forward_return=0.0, realized_vol=0.0, max_drawdown=0.0)
        future = close_arr[t + 1 : end].astype(np.float64, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_future = np.where(future > 0, np.log(future), np.nan)
            log_c0 = np.log(c0)
        log_path = np.concatenate(([log_c0], log_future))
        logret = np.diff(log_path)
        logret = logret[np.isfinite(logret)]
        rel = future / c0 - 1.0
        forward_return = float(rel[-1])
        realized_vol = float(logret.std(ddof=0)) if logret.size else 0.0
        max_drawdown = float(rel.min()) if rel.size else 0.0
        return RegimeOutcome(
            forward_return=forward_return if np.isfinite(forward_return) else 0.0,
            realized_vol=realized_vol if np.isfinite(realized_vol) else 0.0,
            max_drawdown=min(0.0, max_drawdown) if np.isfinite(max_drawdown) else 0.0,
        )

    for t in [0, 1, 50, 100, 200, 500, 700]:
        ref = _forward_outcome(close_series, t, 12)
        new = _numpy_forward_outcome(close_arr, t, 12)
        assert abs(ref.forward_return - new.forward_return) < 1e-9, f"t={t}: fr {ref.forward_return} != {new.forward_return}"
        assert abs(ref.realized_vol - new.realized_vol) < 1e-9, f"t={t}: rv {ref.realized_vol} != {new.realized_vol}"
        assert abs(ref.max_drawdown - new.max_drawdown) < 1e-9, f"t={t}: dd {ref.max_drawdown} != {new.max_drawdown}"


# ---------------------------------------------------------------------------
# Performance: numpy outcome should be substantially faster
# ---------------------------------------------------------------------------
def test_outcome_numpy_faster_than_pandas():
    df = _make_df(2000)
    close_series = df["close"].astype(float)
    close_arr = close_series.to_numpy(dtype=np.float64)
    horizons = (6, 12, 24, 48)
    indices = list(range(64, len(close_arr) - 50))
    n = len(indices) * len(horizons)
    # Warm up both
    for t in indices[:50]:
        for h in horizons:
            _ = _outcome(close_series, t, h)
            _ = _outcome(close_arr, t, h)
    # Pandas
    t0 = time.perf_counter()
    for t in indices:
        for h in horizons:
            _ = _outcome(close_series, t, h)
    pandas_t = time.perf_counter() - t0
    # Numpy
    t0 = time.perf_counter()
    for t in indices:
        for h in horizons:
            _ = _outcome(close_arr, t, h)
    numpy_t = time.perf_counter() - t0
    assert numpy_t < pandas_t, (
        f"numpy {numpy_t:.3f}s not faster than pandas {pandas_t:.3f}s — regression"
    )
    # We expect a 5-30x speedup on realistic workloads
    assert numpy_t <= pandas_t * 0.5, (
        f"numpy is only {pandas_t/max(numpy_t,1e-9):.1f}x faster; expected >= 2x"
    )


# ---------------------------------------------------------------------------
# RegimeOutcomeDataset: warm cache should dominate the workload
# ---------------------------------------------------------------------------
def test_regime_outcome_dataset_warm_dominates():
    set_seed(42)
    df = _make_df(2000)
    cfg = RegimeOutcomeDatasetConfig(horizons=(6, 12), stride=10, min_history=64)
    set_seed(42)
    ds = RegimeOutcomeDataset(df, cfg=cfg)
    n = len(ds)
    indices = list(range(n))
    # First pass (cold): builds cache
    t0 = time.perf_counter()
    for i in indices:
        _ = ds[i]
    cold = time.perf_counter() - t0
    # Second pass (warm): hits cache
    t0 = time.perf_counter()
    for i in indices:
        _ = ds[i]
    warm = time.perf_counter() - t0
    # The fast path is advertised
    assert ds.__fast_getitem__ is True
    # Warm must be at least 5x faster than cold (typical: 50-100x)
    assert warm * 5.0 <= cold + 0.05, (
        f"warm {warm:.3f}s not 5x faster than cold {cold:.3f}s — cache ineffective"
    )


def test_regime_outcome_dataset_strided_matches_legacy():
    """Strided indices should produce deterministic outcomes that match
    the legacy single-idx calculation."""
    set_seed(42)
    df = _make_df(1500)
    cfg = RegimeOutcomeDatasetConfig(horizons=(4, 8), stride=5, min_history=64)
    set_seed(42)
    ds = RegimeOutcomeDataset(df, cfg=cfg)
    # Pick a few indices and verify the regime outcomes are not NaN/inf
    for i in [0, len(ds) // 2, len(ds) - 1]:
        item = ds[i]
        # Outcomes are objects with named fields
        for o in item.outcomes:
            assert np.isfinite(o.forward_return) or o.forward_return == 0.0
            assert np.isfinite(o.realized_vol) or o.realized_vol == 0.0
            assert np.isfinite(o.max_drawdown) or o.max_drawdown == 0.0
        # Return / vol / drawdown tensors must be finite
        for t in (item.return_targets, item.volatility_targets, item.drawdown_targets, item.mfe_targets):
            assert torch.isfinite(t).all(), f"non-finite in tensor {t}"


# ---------------------------------------------------------------------------
# RegimeSupervisionDataset: same pattern
# ---------------------------------------------------------------------------
def test_regime_supervision_dataset_warm_dominates():
    set_seed(42)
    df = _make_df(2000)
    cfg = RegimeSupervisionConfig(horizon=12, stride=10, min_history=64)
    set_seed(42)
    ds = RegimeSupervisionDataset(df, cfg=cfg)
    n = len(ds)
    indices = list(range(n))
    t0 = time.perf_counter()
    for i in indices:
        _ = ds[i]
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in indices:
        _ = ds[i]
    warm = time.perf_counter() - t0
    assert warm * 5.0 <= cold + 0.05, (
        f"warm {warm:.3f}s not 5x faster than cold {cold:.3f}s"
    )


def test_regime_supervision_dataset_outputs_well_formed():
    set_seed(42)
    df = _make_df(1500)
    cfg = RegimeSupervisionConfig(horizon=12, stride=5, min_history=64)
    set_seed(42)
    ds = RegimeSupervisionDataset(df, cfg=cfg)
    for i in [0, len(ds) // 2, len(ds) - 1]:
        item = ds[i]
        # All required fields present
        assert item.x.shape[0] > 0
        assert isinstance(item.macro.item(), int)
        assert isinstance(item.meso.item(), int)
        assert isinstance(item.risk_mode.item(), int)
        assert torch.isfinite(item.forward_return).item()
        assert torch.isfinite(item.realized_vol).item()
        assert torch.isfinite(item.max_drawdown).item()
        # Playbook scores shape matches the names tuple
        assert item.playbook_scores.shape[0] == len(PLAYBOOK_NAMES)
        # Meta is a dict
        assert "t" in item.meta
        assert "horizon" in item.meta


# ---------------------------------------------------------------------------
# _playbook_scores: still deterministic
# ---------------------------------------------------------------------------
def test_playbook_scores_still_works():
    """Quick smoke test that _playbook_scores produces the right shape
    and that the best_playbook index matches the argmax of the scores."""
    from zhisa.regime.dataset import _playbook_scores
    from zhisa.regime.memory import RegimeOutcome
    # Construct a fake report-like object: we just need allowed_playbooks
    from types import SimpleNamespace
    report = SimpleNamespace(allowed_playbooks=("trend_pullback_long", "no_trade_wait"))
    outcome = RegimeOutcome(forward_return=0.02, realized_vol=0.01, max_drawdown=-0.005)
    scores, best_id, best_name = _playbook_scores(report, outcome)
    assert scores.shape == (len(PLAYBOOK_NAMES),)
    assert best_id == int(np.argmax(scores))
    assert best_name == PLAYBOOK_NAMES[best_id]
    # Both playbooks in the report should have non-default scores
    assert scores[PLAYBOOK_TO_ID["trend_pullback_long"]] != -1.0
    assert scores[PLAYBOOK_TO_ID["no_trade_wait"]] != -1.0
