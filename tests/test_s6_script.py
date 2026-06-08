"""Smoke tests for the S6 Decision Transformer training script."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    import os
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


def test_s6_config_loads():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "s6_dt.yaml").read_text())
    assert "context_length" in cfg
    assert cfg["d_model"] % cfg["n_heads"] == 0
    assert int(cfg["epochs"]) >= 1
    assert int(cfg["episodes"]) >= 1


def test_s6_script_random_base_smoke(tmp_path):
    out = tmp_path / "dt.pt"
    proc = _run([
        "-m", "zhisa.scripts.train_s6",
        "--config", "configs/s6_dt.yaml",
        "--bars", "300",
        "--episodes", "2",
        "--max-steps", "20",
        "--epochs", "1",
        "--context-length", "4",
        "--random-base",
        "--checkpoint", str(out),
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert out.stat().st_size > 0
    assert "training complete" in proc.stdout


def test_s6_script_unknown_config_falls_back(tmp_path):
    """If the config path is invalid, the script should still work via CLI defaults."""
    out = tmp_path / "dt.pt"
    proc = _run([
        "-m", "zhisa.scripts.train_s6",
        "--config", "nonexistent.yaml",
        "--bars", "200",
        "--episodes", "1",
        "--max-steps", "10",
        "--epochs", "1",
        "--context-length", "3",
        "--random-base",
        "--checkpoint", str(out),
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_s6_script_with_base_policy_checkpoint(tmp_path):
    """If a (random-init) base policy is given, the script should still run."""
    import torch
    from zhisa.models.policy import build_default_policy
    pol = build_default_policy(in_numeric_features=32, in_context_features=10, window=16, image_size=32)
    pol_ckpt = tmp_path / "base.pt"
    torch.save({"model": pol.state_dict()}, pol_ckpt)
    out = tmp_path / "dt.pt"
    proc = _run([
        "-m", "zhisa.scripts.train_s6",
        "--config", "configs/s6_dt.yaml",
        "--bars", "300",
        "--episodes", "2",
        "--max-steps", "20",
        "--epochs", "1",
        "--context-length", "4",
        "--base-policy", str(pol_ckpt),
        "--checkpoint", str(out),
    ], timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
