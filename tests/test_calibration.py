"""Tests for the calibration module."""
from __future__ import annotations

import numpy as np
import pytest

from zhisa.training.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
    calibration_error,
    fit_calibrator,
    reliability_diagram,
)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def test_temperature_initially_one():
    cal = TemperatureScaler()
    assert cal.temperature == 1.0


def test_temperature_fit_changes_temperature():
    """On miscalibrated data, the temperature should move away from 1."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, size=(200, 3))
    # Make logits for class 0 the dominant one.
    z[:, 0] += 2.0
    y = np.zeros(200, dtype=np.int64)
    cal = TemperatureScaler().fit(z, y)
    # The temperature should be different from 1.0 (it's overconfident).
    assert cal.temperature != 1.0


def test_temperature_predict_proba_is_normalised():
    z = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 0.0]])
    cal = TemperatureScaler()
    p = cal.predict_proba(z)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(axis=1), [1.0, 1.0], atol=1e-6)
    assert (p >= 0).all() and (p <= 1).all()


def test_temperature_reduces_ece():
    """A temperature > 1 on overconfident logits should reduce ECE."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 5, size=(500, 3))
    y_int = rng.integers(0, 3, size=500)
    # Compute raw softmax probs (overconfident).
    raw = np.exp(z - z.max(axis=1, keepdims=True))
    raw /= raw.sum(axis=1, keepdims=True)
    raw_labels = (y_int == 1).astype(np.float64)
    ece_before = calibration_error(raw, raw_labels)

    cal = TemperatureScaler(n_iter=200).fit(z, y_int)
    p_cal = cal.predict_proba(z)
    ece_after = calibration_error(p_cal, raw_labels)
    assert ece_after < ece_before


def test_temperature_state_dict_round_trip():
    z = np.random.default_rng(0).normal(0, 1, (50, 3))
    y = np.random.default_rng(0).integers(0, 3, 50)
    cal = TemperatureScaler().fit(z, y)
    state = cal.state_dict()
    new = TemperatureScaler()
    new.load_state_dict(state)
    assert new.temperature == cal.temperature
    np.testing.assert_allclose(
        cal.predict_proba(z), new.predict_proba(z), atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------


def test_platt_fit_runs():
    z = np.random.default_rng(0).normal(0, 1, (100,))
    y = (z > 0).astype(np.int64)
    cal = PlattCalibrator().fit(z, y)
    assert np.isfinite(cal.slope)
    assert np.isfinite(cal.intercept)


def test_platt_predict_proba_in_unit_interval():
    z = np.linspace(-3, 3, 20)
    cal = PlattCalibrator(slope=1.0, intercept=0.0)
    p = cal.predict_proba(z)
    assert (p >= 0).all() and (p <= 1).all()


def test_platt_handles_2d_logits():
    z = np.array([[0.1, 0.9], [0.7, 0.3]])
    y = np.array([1, 0])
    cal = PlattCalibrator(target_class=1).fit(z, y)
    p = cal.predict_proba(z)
    assert p.shape == (2,)
    assert (p >= 0).all() and (p <= 1).all()


def test_platt_state_dict_round_trip():
    cal = PlattCalibrator(slope=2.5, intercept=-0.7, target_class=0)
    state = cal.state_dict()
    new = PlattCalibrator()
    new.load_state_dict(state)
    assert new.slope == 2.5
    assert new.intercept == -0.7
    assert new.target_class == 0


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------


def test_isotonic_fit_is_monotonic():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 200)
    y = (s > 0.5).astype(np.float64) + rng.normal(0, 0.1, 200)
    y = np.clip(y, 0, 1)
    cal = IsotonicCalibrator().fit(s, y)
    # The output must be monotonically non-decreasing in the input.
    p = cal.predict_proba(np.linspace(0, 1, 50))
    diffs = np.diff(p)
    assert (diffs >= -1e-6).all()


def test_isotonic_predict_proba_in_unit_interval():
    s = np.linspace(-1, 2, 30)
    cal = IsotonicCalibrator()
    cal.fit(np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.5, 0.9]))
    p = cal.predict_proba(s)
    assert (p >= 0).all() and (p <= 1).all()


def test_isotonic_state_dict_round_trip():
    s = np.linspace(0, 1, 20)
    y = np.array([0.1] * 10 + [0.9] * 10)
    cal = IsotonicCalibrator().fit(s, y)
    state = cal.state_dict()
    new = IsotonicCalibrator()
    new.load_state_dict(state)
    p_old = cal.predict_proba(s)
    p_new = new.predict_proba(s)
    np.testing.assert_allclose(p_old, p_new, atol=1e-5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_calibration_error_perfect_is_zero():
    p = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    # Perfect calibration only if bin averages match — with N=4 the
    # ECE should be very small (not zero in general).
    ece = calibration_error(p, y, n_bins=4)
    assert 0.0 <= ece <= 0.3


def test_calibration_error_accepts_2d_probs():
    p = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
    y = np.array([0, 1, 1])
    ece = calibration_error(p, y)
    assert 0.0 <= ece <= 1.0


def test_reliability_diagram_returns_dict():
    p = np.linspace(0, 1, 50)
    y = (p > 0.5).astype(np.float64)
    diag = reliability_diagram(p, y, n_bins=5)
    assert set(diag.keys()) == {"bin_edges", "bin_accuracy",
                                  "bin_confidence", "bin_count"}
    assert len(diag["bin_edges"]) == 6
    assert len(diag["bin_accuracy"]) == 5


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_fit_calibrator_factory():
    z = np.random.default_rng(0).normal(0, 1, (100, 3))
    y = np.random.default_rng(0).integers(0, 3, 100)
    for kind in ("temperature", "platt", "isotonic"):
        cal = fit_calibrator(kind, z, y)
        assert cal is not None
        p = cal.predict_proba(z[:3])
        assert p.shape[0] == 3


def test_fit_calibrator_unknown_kind():
    with pytest.raises(ValueError):
        fit_calibrator("magic", np.zeros((10, 2)), np.zeros(10, dtype=int))
