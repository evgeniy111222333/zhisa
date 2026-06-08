"""Smoke tests for the S2b CLI script (BC + DAgger)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from zhisa.config import load_config


def test_s2b_config_loads():
    cfg = load_config(Path("configs/s2b_imitation.yaml"))
    assert cfg is not None
    assert cfg["trainer"] in ("bc", "dagger")
    assert cfg["expert"] in ("triple_barrier", "momentum", "sma_cross")
    assert "optim" in cfg
    assert "loss_weights" in cfg
    assert "env_cfg" in cfg


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    env["ZHISA_FAST_RENDER"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "zhisa.scripts.train_s2b", *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )


def test_s2b_bc_smoke(tmp_path):
    """A minimal BC run should complete and write a checkpoint."""
    out_dir = tmp_path / "artifacts"
    cmd = [
        "--bars", "200",
        "--epochs", "1",
        "--trainer", "bc",
        "--expert", "triple_barrier",
        "--checkpoint", str(out_dir / "s2b_bc_smoke.pt"),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "S2b (bc) training complete" in result.stdout
    assert (out_dir / "s2b_bc_smoke.pt").exists()


def test_s2b_dagger_smoke(tmp_path):
    """A minimal DAgger run should complete and write a checkpoint."""
    out_dir = tmp_path / "artifacts"
    cmd = [
        "--bars", "200",
        "--rounds", "1",
        "--trainer", "dagger",
        "--expert", "momentum",
        "--checkpoint", str(out_dir / "s2b_dagger_smoke.pt"),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "S2b (dagger) training complete" in result.stdout
    assert (out_dir / "s2b_dagger_smoke.pt").exists()


def test_s2b_sma_cross_expert(tmp_path):
    """The CLI should accept the sma_cross expert."""
    out_dir = tmp_path / "artifacts"
    cmd = [
        "--bars", "200",
        "--epochs", "1",
        "--trainer", "bc",
        "--expert", "sma_cross",
        "--checkpoint", str(out_dir / "s2b_sma_smoke.pt"),
    ]
    result = _run(cmd)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert (out_dir / "s2b_sma_smoke.pt").exists()


def test_s2b_unknown_expert_fails():
    """Passing an unknown expert should fail with a clean error."""
    cmd = [
        "--bars", "100",
        "--epochs", "1",
        "--trainer", "bc",
        "--expert", "nope",
    ]
    result = _run(cmd, timeout=30)
    # argparse should reject the choice.
    assert result.returncode != 0
