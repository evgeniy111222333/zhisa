"""Smoke tests for the S7 world-model training script."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 480) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["ZHISA_FAST_RENDER"] = "1"
    return subprocess.run(
        [sys.executable, *cmd],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_s7_config_loads():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "s7_world_model.yaml").read_text())
    assert "wm_epochs" in cfg
    assert "dyna_horizon" in cfg
    assert cfg["wm_epochs"] >= 1
    assert cfg["dyna_horizon"] >= 1


def test_s7_script_random_base_smoke(tmp_path):
    wm_ckpt = tmp_path / "wm.pt"
    dyna_ckpt = tmp_path / "dyna.pt"
    proc = _run([
        "-m", "zhisa.scripts.train_s7",
        "--config", "configs/s7_world_model.yaml",
        "--bars", "300",
        "--episodes", "2",
        "--max-steps", "20",
        "--wm-epochs", "1",
        "--dyna-rounds", "1",
        "--horizon", "4",
        "--random-base",
        "--checkpoint", str(wm_ckpt),
        "--dyna-checkpoint", str(dyna_ckpt),
    ])
    assert proc.returncode == 0, proc.stderr
    assert wm_ckpt.exists() and wm_ckpt.stat().st_size > 0
    assert dyna_ckpt.exists() and dyna_ckpt.stat().st_size > 0
    assert "WorldModel trained" in proc.stdout
    assert "Dyna PPO complete" in proc.stdout


def test_s7_script_unknown_config_falls_back(tmp_path):
    wm_ckpt = tmp_path / "wm.pt"
    dyna_ckpt = tmp_path / "dyna.pt"
    proc = _run([
        "-m", "zhisa.scripts.train_s7",
        "--config", "nonexistent.yaml",
        "--bars", "200",
        "--episodes", "1",
        "--max-steps", "10",
        "--wm-epochs", "1",
        "--dyna-rounds", "1",
        "--horizon", "3",
        "--random-base",
        "--checkpoint", str(wm_ckpt),
        "--dyna-checkpoint", str(dyna_ckpt),
    ])
    assert proc.returncode == 0, proc.stderr
    assert wm_ckpt.exists()
    assert dyna_ckpt.exists()
