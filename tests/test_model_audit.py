from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.model_audit.catalog import build_catalog
from zhisa.model_audit.metrics import action_churn, expected_calibration_error, trade_audit
from zhisa.model_audit.perturbations import mirror_prices


def test_catalog_is_complete_and_unique():
    catalog = build_catalog()
    assert [item.id for item in catalog] == list(range(1, 61))
    assert len({item.key for item in catalog}) == 60


def test_trade_audit_tracks_sides_streak_and_exposure():
    positions = np.array([0, 1, 1, 0, -1, -1, 0], dtype=float)
    equity = np.array([1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0])
    result = trade_audit(positions, equity)
    assert result.n_trades == 2
    assert result.long_win_rate == 1.0
    assert result.short_win_rate == 0.0
    assert result.max_consecutive_losses == 1
    assert result.exposure_fraction == 4 / 7


def test_calibration_and_churn_have_known_bounds():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.1)
    assert action_churn(np.array([1, 4, 2, 0])) == pytest.approx(2 / 3)


def test_price_mirror_preserves_positive_ohlc():
    frame = pd.DataFrame({
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0], "volume": [1.0, 2.0],
    })
    mirrored = mirror_prices(frame)
    assert (mirrored[["open", "high", "low", "close"]] > 0).all().all()
    assert (mirrored["high"] >= mirrored[["open", "close"]].max(axis=1)).all()
    assert (mirrored["low"] <= mirrored[["open", "close"]].min(axis=1)).all()
