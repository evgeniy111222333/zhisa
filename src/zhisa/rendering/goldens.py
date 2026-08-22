"""Golden-image registry: deterministic QA for the canonical renderer.

The ideal treats the rendered image as an audited artefact. A *golden* is a
fixed ``(fixture window, RenderSpec, renderer version)`` triple whose output
pixels + SHA-256 digest are committed to disk. Every future render is compared
against the golden; any unexpected change (accidental or otherwise) surfaces
as a test failure instead of silently shifting the model's visual input
distribution.

Registry layout (content-addressed):

    <root>/
      <fixture_name>/
        <fingerprint>/
          meta.json          # fixture name + spec hash + renderer version + shape
          image.npy          # committed (H, W, 3) float32 pixels
          image.sha256       # hex digest of the raw .npy bytes
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from zhisa.rendering.chart_renderer import CANONICAL_RENDERER_VERSION, render_fingerprint, _render_ohlcv_canonical
from zhisa.rendering.spec import RenderSpec


def golden_fixture(n_bars: int = 96, seed: int = 0) -> np.ndarray:
    """A deterministic, dependency-free OHLCV fixture for goldens.

    Deliberately *not* generated with an RNG: prices are produced by closed-form
    arithmetic, so the fixture bytes are stable across numpy/platform versions
    (this is what lets goldens be compared bit-exactly on different machines).
    """
    t = np.arange(n_bars, dtype=np.float64)
    # Three waves + integer offsets -> full determinism.
    base = 100.0 + 8.0 * np.sin(t / 11.0) + 3.0 * np.sin(t / 3.7)
    body = 0.9 + 1.2 * np.sin(t / 5.0 + 1.0)
    body = np.minimum(np.maximum(body, 0.3), 2.2)
    open_ = base
    close = open_ + body
    high = np.maximum(open_, close) + 0.4 + 0.5 * np.abs(np.sin(t / 2.0))
    low = np.minimum(open_, close) - 0.4 - 0.5 * np.abs(np.cos(t / 2.0))
    volume = 500.0 + 400.0 * np.abs(np.sin(t / 4.0))
    return np.column_stack([open_, high, low, close, volume])


def _image_sha256(image: np.ndarray) -> str:
    raw = np.ascontiguousarray(image, dtype=np.float32).tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def golden_dir(root, fixture_name: str, fingerprint: str) -> Path:
    return Path(root) / fixture_name / fingerprint


def compute_digest(ohlcv: np.ndarray, spec: RenderSpec) -> str:
    """SHA-256 of the canonical render of a window under a spec."""
    return _image_sha256(_render_ohlcv_canonical(np.asarray(ohlcv), spec))


class GoldenError(AssertionError):
    """Raised when a render no longer matches its committed golden."""


def store_golden(
    root,
    fixture_name: str,
    ohlcv: np.ndarray,
    spec: RenderSpec,
    *,
    force: bool = False,
) -> dict:
    """Render ``ohlcv`` and commit it as a golden under ``root``.

    Returns the persisted meta dict. If a golden already exists for the same
    (fixture, fingerprint) and ``force=False``, it is left untouched.
    """
    fp = render_fingerprint(spec)
    gdir = golden_dir(root, fixture_name, fp)
    gdir.mkdir(parents=True, exist_ok=True)
    meta_path = gdir / "meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    image = _render_ohlcv_canonical(np.asarray(ohlcv, dtype=np.float64), spec)
    np.save(gdir / "image.npy", image)
    digest = digest_of_saved(gdir)
    meta = {
        "fixture": fixture_name,
        "renderer_version": CANONICAL_RENDERER_VERSION,
        "spec_hash": spec.content_hash(),
        "fingerprint": fp,
        "image_sha256": digest,
        "shape": [int(s) for s in image.shape],
        "n_bars": int(len(ohlcv)),
    }
    (gdir / "image.sha256").write_text(digest, encoding="utf-8")
    (gdir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def digest_of_saved(gdir) -> str:
    img_path = Path(gdir) / "image.npy"
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    arr = np.load(img_path)
    return _image_sha256(arr)


def has_golden(root, fixture_name: str, spec: RenderSpec) -> bool:
    return golden_dir(root, fixture_name, render_fingerprint(spec)).is_dir()


def assert_matches_golden(
    root,
    fixture_name: str,
    ohlcv: np.ndarray,
    spec: RenderSpec,
    *,
    atol: float = 0.0,
) -> dict:
    """Compare a fresh render against the committed golden.

    Raises :class:`GoldenError` if pixels differ beyond ``atol`` (default:
    bit-exact) or the digest of the committed image is inconsistent. Returns
    the golden meta on success.
    """
    gdir = golden_dir(root, fixture_name, render_fingerprint(spec))
    meta_path = gdir / "meta.json"
    if not meta_path.exists():
        raise GoldenError(f"no golden committed for {fixture_name}/{render_fingerprint(spec)}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # The committed artefact must be internally consistent before we compare.
    actual_saved = digest_of_saved(gdir)
    if actual_saved != meta["image_sha256"]:
        raise GoldenError(
            f"golden {fixture_name}/{render_fingerprint(spec)} is corrupt: "
            f"stored sha256 {meta['image_sha256']} != on-disk {actual_saved}"
        )

    fresh = _render_ohlcv_canonical(np.asarray(ohlcv, dtype=np.float64), spec)
    expected = np.load(gdir / "image.npy")
    if fresh.shape != expected.shape:
        raise GoldenError(
            f"golden {fixture_name} shape mismatch: fresh {fresh.shape} != expected {expected.shape}"
        )
    diff = np.abs(fresh.astype(np.float32) - expected.astype(np.float32))
    if float(diff.max()) > atol:
        raise GoldenError(
            f"golden {fixture_name}/{render_fingerprint(spec)} mismatch: "
            f"max|Δ|={float(diff.max()):.6f} > atol={atol} "
            f"({int((diff > atol).sum())}/{diff.size} pixels differ)"
        )
    return meta