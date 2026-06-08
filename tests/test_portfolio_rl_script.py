"""Smoke test for the Stage-1 portfolio RL training script."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["ZHISA_FAST_RENDER"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True,
                          text=True, timeout=timeout)


def test_train_portfolio_rl_smoke(tmp_path):
    ckpt = tmp_path / "p.pt"
    hist = tmp_path / "hist.json"
    cmd = [
        sys.executable, "-m", "zhisa.scripts.train_portfolio_rl",
        "--bars", "200",
        "--n-instruments", "2",
        "--iterations", "1",
        "--episodes", "1",
        "--max-steps", "5",
        "--epochs", "1",
        "--minibatch", "2",
        "--learning-rate", "0.0003",
        "--embed-dim", "16",
        "--fusion-hidden", "16",
        "--window", "16",
        "--image-size", "32",
        "--episode-length", "5",
        "--gross-cap", "0.5",
        "--checkpoint", str(ckpt),
        "--history", str(hist),
        "--seed", "0",
    ]
    proc = _run(cmd, timeout=120)
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert ckpt.exists()
    payload_path = tmp_path / "p.pt"
    assert payload_path.exists()
    import torch
    payload = torch.load(str(payload_path), weights_only=False, map_location="cpu")
    assert "model" in payload
    assert "ppo_config" in payload

    assert hist.exists()
    history = json.loads(hist.read_text("utf-8"))
    assert isinstance(history, list)
    assert len(history) >= 1
    entry = history[-1]
    assert "mean_return" in entry
    assert "mean_gross_leverage" in entry


def test_train_portfolio_rl_rejects_single_instrument(tmp_path):
    cmd = [
        sys.executable, "-m", "zhisa.scripts.train_portfolio_rl",
        "--bars", "100", "--n-instruments", "1", "--iterations", "1",
        "--episodes", "1", "--max-steps", "2",
        "--checkpoint", str(tmp_path / "p.pt"),
        "--history", str(tmp_path / "h.json"),
    ]
    proc = _run(cmd, timeout=60)
    assert proc.returncode != 0
    assert "n_instruments" in proc.stderr or "n_instruments" in proc.stdout
