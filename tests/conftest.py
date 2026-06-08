"""Shared pytest fixtures."""
from __future__ import annotations

import os

os.environ.setdefault("ZHISA_FAST_RENDER", "1")
os.environ.setdefault("ZHISA_TEST_DEVICE", "auto")

import pytest

import numpy as np
import pandas as pd
import torch

from zhisa.data.synthetic import MarketConfig, generate_market
from zhisa.utils.seeding import set_seed


def _resolve_test_device() -> str:
    pref = os.environ.get("ZHISA_TEST_DEVICE", "auto").lower()
    if pref in {"cpu", "cuda"}:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


TEST_DEVICE = _resolve_test_device()


@pytest.fixture(autouse=True)
def _seed_everything():
    set_seed(1234)
    yield


@pytest.fixture
def small_market() -> pd.DataFrame:
    cfg = MarketConfig(n_bars=1500, freq="5min", seed=1234)
    return generate_market(cfg)


@pytest.fixture
def tiny_market() -> pd.DataFrame:
    cfg = MarketConfig(n_bars=400, freq="5min", seed=99)
    return generate_market(cfg)


@pytest.fixture
def device() -> str:
    return TEST_DEVICE
