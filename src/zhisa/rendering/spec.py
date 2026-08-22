"""RenderSpec — the canonical, versioned description of how a chart is drawn.

This is the single source of truth for *every* visual decision a chart needs:
image size, anti-aliasing factor, colour palette, price/volume layout, price
mapping epsilon, overlays (SMA lines), and whether volume/overlays are drawn.

The ideal architecture treats rendering as a compiled, reproducible artefact:

- ``renderer.reconfigure`` takes an OHLCV window and a ``RenderSpec`` and
  returns a tensor **deterministically** (pure function, no RNG, no env vars).
- A ``RenderSpec`` has a stable *content hash*, so a chart's visual identity
  is captured (and audited) even when the implementation evolves.
- Training, validation and inference must all use the **same** spec hash; the
  byte-equivalence contract (see ``data.chart_store``) enforces this.

Why this file exists independently of the renderer: separation of *what* the
chart must look like (spec) from *how* it is rasterised (renderer). Changing
one should not silently change the other, and both are versioned.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Tuple

# Matches the historical colour palette so a change here is intentional.
DEFAULT_BG = (0.05, 0.06, 0.09)
DEFAULT_FG = (0.85, 0.88, 0.92)
DEFAULT_GREEN = (0.20, 0.80, 0.40)
DEFAULT_RED = (0.95, 0.35, 0.30)
DEFAULT_GREY = (0.45, 0.48, 0.55)

# (period, rgb) overlays drawn on top of the price area.
DEFAULT_OVERLAYS: Tuple[Tuple[int, Tuple[float, float, float]], ...] = (
    (10, (0.23, 0.63, 1.00)),   # SMA-10  (blue)
    (30, (1.00, 0.67, 0.20)),   # SMA-30  (orange)
)

# Bump this whenever a change would alter the meaning of the rendered image
# (new overlay, changed layout, changed mapping). It participates in the
# content hash, so bumping it deliberately invalidates cached artefacts.
RENDER_SPEC_VERSION = "1.0.0"


@dataclass(frozen=True)
class RenderSpec:
    """Deterministic description of how a chart image is produced.

    All fields are immutable; two specs with equal fields produce byte-equal
    images (given the same renderer version and same OHLCV window).
    """

    size: int = 64
    # Anti-aliasing: the underlying raster is drawn at ``size * supersample``
    # then box-downsampled. Higher = smoother edges, more compute.
    supersample: int = 4
    # Fraction of vertical space reserved for the price area (the rest is the
    # volume mini-panel at the bottom).
    price_frac: float = 0.75
    # Floor applied to the price range so flat windows do not divide by zero.
    min_range: float = 1e-9
    # Colour palette.
    background: Tuple[float, float, float] = DEFAULT_BG
    foreground: Tuple[float, float, float] = DEFAULT_FG
    green: Tuple[float, float, float] = DEFAULT_GREEN
    red: Tuple[float, float, float] = DEFAULT_RED
    grey: Tuple[float, float, float] = DEFAULT_GREY
    overlays: Tuple[Tuple[int, Tuple[float, float, float]], ...] = DEFAULT_OVERLAYS
    include_volume: bool = True
    include_overlays: bool = True
    # Reserved for deterministic keyed augmentations (see ``augmentations``);
    # does not affect the base render.
    seed: int = 0
    version: str = RENDER_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.supersample < 1:
            raise ValueError("supersample must be >= 1")
        if not (0.0 < self.price_frac <= 1.0):
            raise ValueError("price_frac must be in (0, 1]")
        for col in (self.background, self.foreground, self.green, self.red, self.grey):
            for ch in col:
                if not (0.0 <= ch <= 1.0):
                    raise ValueError(f"colour channel out of [0,1]: {ch}")

    # ------------------------------------------------------------------
    # Canonical serialization + content hash
    # ------------------------------------------------------------------

    def canonical_dict(self, *, exclude: Tuple[str, ...] = ("seed", "version")) -> dict:
        """A stable, JSON-serialisable representation for hashing.

        ``exclude`` removes fields that should not change the visual identity
        of the base render (the RNG seed only matters for augmentation, and
        ``version`` is folded in separately so it can be bumped explicitly).
        """
        d = asdict(self)
        for key in exclude:
            d.pop(key, None)
        # Overlays are a tuple of tuples; make them JSON-safe lists already
        # deduped & sorted for stability (they are uniquely keyed by period).
        ov = d.get("overlays")
        if ov is not None:
            d["overlays"] = sorted((int(p), [float(c) for c in col]) for p, col in ov)
        d["size"] = int(d["size"])
        d["supersample"] = int(d["supersample"])
        d["price_frac"] = float(d["price_frac"])
        d["min_range"] = float(d["min_range"])
        d["include_volume"] = bool(d["include_volume"])
        d["include_overlays"] = bool(d["include_overlays"])
        for k in ("background", "foreground", "green", "red", "grey"):
            d[k] = [float(x) for x in d[k]]
        return d

    def content_hash(self, *, shake: str = "") -> str:
        """Stable SHA-256 of the canonical spec (bumped with ``version``)."""
        payload = {
            "version": self.version,
            **self.canonical_dict(exclude=("seed", "version")),
        }
        if shake:
            payload["shake"] = shake
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def short_hash(self, n: int = 12) -> str:
        return self.content_hash()[:n]

    def to_meta(self) -> dict:
        """Metadata block that can be persisted next to a compiled artefact.

        Round-trips through :meth:`from_meta` losslessly (JSON-safe). The
        RNG ``seed`` is kept so an augmentation keyed by the same spec reproduces
        the same randomised behaviour on reload.
        """
        d = self.canonical_dict(exclude=())
        d["version"] = self.version
        return d

    @classmethod
    def from_meta(cls, meta: dict) -> "RenderSpec":
        known = {f.name for f in fields(cls)}
        m = dict(meta)
        for key in ("background", "foreground", "green", "red", "grey"):
            if key in m:
                m[key] = tuple(float(x) for x in m[key])
        if "overlays" in m and isinstance(m["overlays"], list):
            m["overlays"] = tuple(
                (int(p), tuple(float(c) for c in col)) for p, col in m["overlays"]
            )
        kwargs = {k: v for k, v in m.items() if k in known}
        return cls(**kwargs)


def default_render_spec(size: int | None = None, **overrides) -> RenderSpec:
    """Convenience factory for a canonical spec, optionally at a given size."""
    spec = RenderSpec()
    if size is not None:
        overrides["size"] = size
    if overrides:
        spec = RenderSpec(**{**asdict(spec), **overrides})
    return spec
