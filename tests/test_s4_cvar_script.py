"""Smoke tests for the S4-CVaR script."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
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


def test_s4_cvar_config_loads():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "s4_cvar_ppo.yaml").read_text())
    assert 0.0 < cfg["cvar_alpha"] <= 1.0
    assert cfg["cvar_threshold"] >= 0.0
    assert cfg["cvar_lambda_max"] > 0.0
    assert cfg["cvar_lambda_lr"] > 0.0
    assert cfg["n_iterations"] >= 1


def test_s4_cvar_script_smoke(tmp_path):
    out = tmp_path / "model.pt"
    proc = _run([
        "-m", "zhisa.scripts.train_s4_cvar",
        "--config", "configs/s4_cvar_ppo.yaml",
        "--bars", "300",
        "--iterations", "2",
        "--episodes", "2",
        "--max-steps", "20",
        "--cvar-alpha", "0.3",
        "--cvar-threshold", "0.05",
        "--checkpoint", str(out),
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
    assert "S4-CVaR training complete" in proc.stdout
    assert "final_lambda=" in proc.stdout
    assert "final_cvar=" in proc.stdout


def test_s4_cvar_script_unknown_config_falls_back(tmp_path):
    out = tmp_path / "model.pt"
    proc = _run([
        "-m", "zhisa.scripts.train_s4_cvar",
        "--config", "nonexistent.yaml",
        "--bars", "200",
        "--iterations", "1",
        "--episodes", "1",
        "--max-steps", "10",
        "--checkpoint", str(out),
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
