"""Tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zhisa.features.indicators import atr, bollinger, ema, rsi, sma, vwap_session
from zhisa.features.ohlcv import compute_ohlcv_features
from zhisa.features.orderbook import orderbook_features
from zhisa.features.time import compute_time_features


def test_sma_ema_basic(small_market):
    s = small_market["close"]
    sma10 = sma(s, 10)
    ema10 = ema(s, 10)
    assert sma10.notna().all()
    assert ema10.notna().all()
    assert len(sma10) == len(s)


def test_rsi_range(small_market):
    r = rsi(small_market["close"], 14)
    assert (r.dropna() >= 0).all()
    assert (r.dropna() <= 100).all()


def test_atr_positive(small_market):
    a = atr(small_market, 14)
    assert (a > 0).all()


def test_bollinger_bandwidth(small_market):
    bb = bollinger(small_market["close"], 20, n_std=2.0)
    assert (bb["bandwidth"].dropna() >= 0).all()
    valid = bb["upper"].notna() & bb["mid"].notna() & bb["lower"].notna()
    assert (bb["upper"][valid] >= bb["mid"][valid]).all()
    assert (bb["mid"][valid] >= bb["lower"][valid]).all()


def test_vwap_session(small_market):
    v = vwap_session(small_market)
    # VWAP should be a weighted average; should lie within the day's range
    grouped = v.groupby(v.index.date)
    for _, g in grouped:
        lo = small_market.loc[g.index, "low"].min()
        hi = small_market.loc[g.index, "high"].max()
        assert (g >= lo).all() and (g <= hi).all()


def test_ohlcv_features_no_nan_inf(small_market):
    feats = compute_ohlcv_features(small_market)
    assert not feats.isin([np.inf, -np.inf]).any().any()
    # Some early NaNs are expected (rolling) but should be fillable
    feats_filled = feats.fillna(0.0)
    assert np.isfinite(feats_filled.to_numpy()).all()


def test_ohlcv_feature_columns(small_market):
    feats = compute_ohlcv_features(small_market)
    # Check at least a few expected columns
    for col in ("logret_1", "rv_8", "atr_14", "rsi_14", "don_pos_20", "vwap_dist"):
        assert col in feats.columns, f"missing feature {col}"


def test_time_features_shape(small_market):
    tf = compute_time_features(small_market)
    assert tf.shape[0] == len(small_market)
    assert tf.shape[1] % 2 == 0  # cyclic pairs
    # All values must be in [-1, 1]
    assert ((tf >= -1.0) & (tf <= 1.0)).all().all()


def test_orderbook_features_basic():
    bids = np.array([100.0, 99.5, 99.0])
    asks = np.array([100.5, 101.0, 101.5])
    bs = np.array([5.0, 8.0, 4.0])
    as_ = np.array([3.0, 6.0, 7.0])
    f = orderbook_features(bids, asks, bs, as_)
    assert f["spread"] == 0.5
    assert f["imbalance"] > 0.0
    assert f["midprice"] == 100.25
    assert f["weighted_mid"] > 0
