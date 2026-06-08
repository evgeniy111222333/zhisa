"""Subprocess smoke tests for zhisa.scripts.explain."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(args: list[str], timeout: float = 90.0) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("ZHISA_TEST_DEVICE", "cpu")
    env.setdefault("ZHISA_FAST_RENDER", "1")
    return subprocess.run(
        [PYTHON, "-m", "zhisa.scripts.explain", *args],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout,
    )


def _validate_report(payload: dict) -> None:
    assert "n_samples" in payload
    assert "action_distribution" in payload
    assert "top_features" in payload
    assert "samples" in payload
    assert "mean_modality_totals" in payload
    for s in payload["samples"]:
        for k in ("target", "target_name", "action_probabilities",
                  "modality_totals", "chart_saliency_summary",
                  "top_numeric_features"):
            assert k in s, f"missing key {k} in sample"
        s = sum(s["action_probabilities"])
        assert abs(s - 1.0) < 1e-4


def test_explain_full_workflow(tmp_path: Path) -> None:
    """End-to-end: run the explain script, check the JSON report."""
    out = tmp_path / "report.json"
    p = _run([
        "--config", "configs/explain_default.yaml",
        "--n-samples", "2",
        "--n-steps", "2",
        "--top-k", "3",
        "--out", str(out),
    ])
    assert p.returncode == 0, f"stdout={p.stdout!r}\nstderr={p.stderr!r}"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_samples"] == 2
    _validate_report(payload)


def test_explain_with_missing_checkpoint_graceful(tmp_path: Path) -> None:
    """Missing checkpoint path is tolerated; report still written."""
    out = tmp_path / "r2.json"
    p = _run([
        "--config", "configs/explain_default.yaml",
        "--checkpoint", str(tmp_path / "does_not_exist.pt"),
        "--n-samples", "1",
        "--n-steps", "2",
        "--out", str(out),
    ])
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_samples"] == 1
    _validate_report(payload)
