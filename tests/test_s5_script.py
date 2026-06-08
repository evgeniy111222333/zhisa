"""Smoke tests for the S5 online continual script and config loading."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from zhisa.config import load_config
from zhisa.scripts.train_s5 import _build_continual_cfg, _build_inner_factory


def _test_env() -> dict:
    env = dict(os.environ)
    env["ZHISA_FAST_RENDER"] = "1"
    return env


def test_s5_config_loads():
    cfg = load_config(Path("configs/s5_continual.yaml"))
    assert cfg is not None
    assert cfg["n_iterations"] == 3
    assert cfg["inner_kind"] == "s1"
    assert cfg["replay_capacity"] == 64
    assert cfg["ewc_lambda"] == 1.0
    assert cfg["ewc_lambda_on_drift"] == 10.0


def test_build_continual_cfg_defaults():
    from zhisa.utils.containers import Config

    cfg = Config({})
    args = _Args()
    c = _build_continual_cfg(cfg, args)
    assert c.n_iterations == 3
    assert c.replay_capacity == 64
    assert c.ewc_lambda == 1.0
    assert c.ewc_lambda_on_drift == 10.0
    assert c.checkpoint is None


def test_build_continual_cfg_cli_overrides():
    from zhisa.utils.containers import Config

    cfg = Config({"n_iterations": 7, "drift_threshold": 1.0})
    args = _Args(n_iterations=2, checkpoint="artifacts/x.pt")
    c = _build_continual_cfg(cfg, args)
    assert c.n_iterations == 2          # CLI wins
    assert c.drift_threshold == 1.0
    assert c.checkpoint == "artifacts/x.pt"


def test_build_inner_factory_s1():
    from zhisa.utils.containers import Config
    from zhisa.data.dataset import SampleSpec

    spec = SampleSpec(chart_window=8, feature_window=8, image_size=8)
    f = _build_inner_factory(
        Config({"inner_batch_size": 8, "inner_lr": 1e-3, "inner_epochs": 1}),
        "s1", spec=spec,
    )
    assert callable(f)


def test_build_inner_factory_s2():
    from zhisa.utils.containers import Config
    from zhisa.data.dataset import SampleSpec

    spec = SampleSpec(chart_window=8, feature_window=8, image_size=8)
    f = _build_inner_factory(
        Config({"inner_batch_size": 8, "inner_lr": 1e-3, "inner_epochs": 1}),
        "s2", spec=spec,
    )
    assert callable(f)


def test_build_inner_factory_s1_without_spec_raises():
    from zhisa.utils.containers import Config
    with pytest.raises(ValueError):
        _build_inner_factory(Config({}), "s1", spec=None)


def test_build_inner_factory_s4():
    from zhisa.utils.containers import Config

    f = _build_inner_factory(
        Config({"inner_batch_size": 8, "inner_lr": 1e-3, "inner_epochs": 1}),
        "s4",
    )
    assert callable(f)


def test_build_inner_factory_unknown_exits():
    from zhisa.utils.containers import Config

    with pytest.raises(SystemExit):
        _build_inner_factory(Config({}), "s9")


def test_s5_script_runs_smoke(tmp_path):
    """A tiny S5 run should complete and write a checkpoint."""
    tiny_cfg = tmp_path / "tiny_s5.yaml"
    tiny_cfg.write_text(
        "seed: 0\n"
        "device: cpu\n"
        "chart_window: 8\n"
        "image_size: 8\n"
        "n_iterations: 2\n"
        "bars_per_iter: 120\n"
        "replay_capacity: 8\n"
        "replay_batch_size: 2\n"
        "inner_kind: s1\n"
        "inner_epochs: 1\n"
        "inner_batch_size: 4\n"
        "inner_lr: 0.001\n"
        "drift_threshold: 100.0\n"
        "drift_warmup: 100\n"
        "log_every: 1\n"
    )
    out_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable, "-m", "zhisa.scripts.train_s5",
        "--config", str(tiny_cfg),
        "--inner", "s1",
        "--checkpoint", str(out_dir / "s5_smoke.pt"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=_test_env())
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "S5 online continual training complete" in result.stdout
    assert "drift events" in result.stdout
    assert (out_dir / "s5_smoke.pt").exists()


class _Args:
    """Stand-in for argparse.Namespace used by build helpers."""
    def __init__(self, n_iterations=None, checkpoint=None):
        self.n_iterations = n_iterations
        self.checkpoint = checkpoint
