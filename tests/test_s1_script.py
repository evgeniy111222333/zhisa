"""Smoke tests for the S1 training script and SSL config loading."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from zhisa.config import load_config
from zhisa.scripts.train_s1 import _ssl_config_from


def test_s1_config_loads():
    """The shipped S1 config must load without errors and contain the expected keys."""
    cfg = load_config(Path("configs/s1_ssl.yaml"))
    assert cfg is not None
    assert "ssl" in cfg
    s = cfg["ssl"]
    assert s["projection_dim"] == 64
    assert s["temperature"] == 0.1
    assert s["use_ema_teacher"] is True


def test_ssl_config_factory_handles_missing_block():
    """If the config has no 'ssl' block, the factory should use defaults."""
    from zhisa.utils.containers import Config

    cfg = Config({"seed": 0})
    ssl = _ssl_config_from(cfg)
    assert ssl.projection_dim == 64
    assert ssl.use_ema_teacher is True
    assert ssl.epochs == 1  # SSLConfig default


def test_ssl_config_factory_handles_none():
    """The factory must work even if called with ``None`` config."""
    ssl = _ssl_config_from(None)
    assert ssl.projection_dim == 64
    assert ssl.temperature == 0.1


def test_s1_script_runs_smoke(tmp_path):
    """A minimal S1 training run should complete end-to-end and write a checkpoint."""
    out_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable, "-m", "zhisa.scripts.train_s1",
        "--bars", "200",
        "--epochs", "1",
        "--batch-size", "64",
        "--checkpoint", str(out_dir / "s1_smoke.pt"),
    ]
    import os
    env = dict(os.environ)
    env["ZHISA_FAST_RENDER"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "S1 training complete" in result.stdout
    assert (out_dir / "s1_smoke.pt").exists()
