"""Tests for utility modules: config loader, seeding, timing."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from zhisa.config import load_config, deep_merge
from zhisa.utils.seeding import get_seed, set_seed
from zhisa.utils.timing import Timer, rate_limit


def test_deep_merge_scalar_override():
    a = {"x": 1, "y": {"p": 1, "q": 2}}
    b = {"y": {"q": 99}, "z": 3}
    out = deep_merge(a, b)
    assert out == {"x": 1, "y": {"p": 1, "q": 99}, "z": 3}


def test_deep_merge_no_mutation():
    a = {"x": {"y": 1}}
    b = {"x": {"y": 2}}
    deep_merge(a, b)
    assert a == {"x": {"y": 1}}


def test_load_config_simple(tmp_path: Path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 2


def test_load_config_override(tmp_path: Path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("a: 1\nb: {c: 2}\n", encoding="utf-8")
    cfg = load_config(cfg_file, overrides=["a=42", "b.c=100"])
    assert cfg["a"] == 42
    assert cfg["b"]["c"] == 100


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_set_seed_determinism():
    set_seed(123)
    a = np.random.rand(10)
    set_seed(123)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_get_seed_matches_set():
    set_seed(7)
    assert get_seed() == 7


def test_timer_measures():
    t = Timer()
    t.start()
    time.sleep(0.05)
    elapsed = t.stop()
    assert elapsed >= 0.04
    assert elapsed < 1.0


def test_timer_accumulates():
    t = Timer()
    with t:
        time.sleep(0.02)
    t.start()
    time.sleep(0.02)
    t.stop()
    assert t.elapsed >= 0.04


def test_rate_limit_no_op_when_zero():
    t0 = time.perf_counter()
    with rate_limit(0):
        pass
    assert time.perf_counter() - t0 < 0.1
