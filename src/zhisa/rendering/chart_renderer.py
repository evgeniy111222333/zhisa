"""Chart image renderer: OHLCV window -> RGB tensor.

**Canonical renderer (single source of truth).**

The ideal architecture treats charting as a compiled, deterministic artefact:

- One renderer implementation, driven by a :class:`RenderSpec` (``spec.py``).
- **No runtime environment-flag divergence**: ``render_chart`` and
  ``render_chart_array`` always produce the *same* canonical pixels, whether
  or not ``ZHISA_FAST_RENDER`` is set. The old matplotlib-vs-numpy split is
  retired for the training path; the matplotlib path remains available only
  as an explicit *visualisation* tool (``render_chart_visualization``).
- Anti-aliasing is provided by **supersampling + box downsample** so candles
  and overlay lines have smooth, deterministic edges.

Produced images are ``torch.FloatTensor`` of shape ``(3, H, W)`` in [0, 1],
bit-reproducible given the same ``(ohlcv window, RenderSpec)``.
"""
from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from zhisa.rendering.spec import (
    DEFAULT_BG,
    DEFAULT_FG,
    DEFAULT_GREEN,
    DEFAULT_GREY,
    DEFAULT_RED,
    RenderSpec,
    default_render_spec,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.patches import Rectangle  # noqa: E402
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


# Bump whenever rasterisation semantics change (geometry, AA, mapping). It
# participates in every byte-equivalence / golden contract so a deliberate
# visual change cannot be masked by a stale cache.
CANONICAL_RENDERER_VERSION = "1.0.0"


def render_fingerprint(spec: RenderSpec) -> str:
    """Content-address of a render = spec hash + renderer version.

    Two artefacts with the same fingerprint are guaranteed to be visually
    identical *by construction* (given the same input window). Used by the
    compiled chart store and the golden-image registry.
    """
    import hashlib
    raw = f"{CANONICAL_RENDERER_VERSION}:{spec.content_hash()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Canonical rasteriser (pure numpy, deterministic, anti-aliased)
# ---------------------------------------------------------------------------


def _running_mean(values: np.ndarray, period: int) -> np.ndarray:
    """Trailing mean with ``min_periods=1`` (prefix-expanding head)."""
    n = len(values)
    if n < period:
        return np.cumsum(values) / np.arange(1, n + 1)
    kernel = np.ones(period) / period
    sma = np.convolve(values, kernel, mode="valid")
    pad = np.cumsum(values[: period - 1]) / np.arange(1, period)
    return np.concatenate((pad, sma))


def _render_ohlcv_canonical(ohlcv: np.ndarray, spec: RenderSpec) -> np.ndarray:
    """Render ``(N, >=5)`` OHLCV into a ``(size, size, 3)`` float32 image.

    Pure function: deterministic for a fixed ``(ohlcv, spec)``.
    """
    values = np.asarray(ohlcv, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 5:
        raise ValueError("ohlcv must have shape (N, >=5)")
    n = len(values)
    if n < 2:
        raise ValueError("ohlcv must contain at least 2 rows to render a chart")
    size, ss = spec.size, max(int(spec.supersample), 1)
    S = size * ss
    price_h = int(S * spec.price_frac)

    rgb = np.full((S, S, 3), spec.background, dtype=np.float32)

    o_arr, h_arr, l_arr, c_arr, v_arr = values[:, :5].T
    lo = float(l_arr.min())
    hi = float(h_arr.max())
    rng = max(hi - lo, float(spec.min_range))

    xs = (np.arange(n) * (S - 1) / max(n - 1, 1)).astype(int)

    def price_y(values_: np.ndarray) -> np.ndarray:
        y = price_h - ((values_ - lo) / rng * (price_h - 2)).astype(int)
        return np.clip(y, 0, price_h - 1)

    y_o, y_c = price_y(o_arr), price_y(c_arr)
    y_h, y_l = price_y(h_arr), price_y(l_arr)
    colors = np.where(
        (c_arr >= o_arr)[:, None],
        np.asarray(spec.green, dtype=np.float32),
        np.asarray(spec.red, dtype=np.float32),
    )

    # Candles: any row inside [min(wick), max(wick)] is lit, then the body is
    # painted over with the same colour (matches the historical look).
    y_grid = np.arange(price_h)[:, None]
    wick = (y_grid >= np.minimum(y_h, y_l)) & (y_grid <= np.maximum(y_h, y_l))
    body = (y_grid >= np.minimum(y_o, y_c)) & (y_grid <= np.maximum(y_o, y_c))
    py, bars = np.nonzero(wick | body)
    rgb[py, xs[bars]] = colors[bars]

    # Volume mini-panel at the bottom.
    if spec.include_volume:
        vmax = max(float(v_arr.max()), 1e-9)
        bar_heights = (v_arr / vmax * (S - price_h - 1)).astype(int)
        volume_y = np.arange(price_h, S)[:, None]
        volume_mask = volume_y >= (S - bar_heights)[None, :]
        vy, bars = np.nonzero(volume_mask)
        # vy is relative to price_h; offset back into absolute rows so the
        # volume fills the BOTTOM panel (the historical fast renderer's
        # semantics), never the top of the price area.
        rgb[vy + price_h, xs[bars]] = colors[bars]

    # Overlay SMA lines, drawn deterministically in period order.
    if spec.include_overlays:
        for period, col in sorted(spec.overlays, key=lambda ov: int(ov[0])):
            if n < int(period):
                continue
            sma = _running_mean(c_arr, int(period))
            ys = price_y(sma)
            x_grid = np.arange(S)
            line_y = np.rint(np.interp(x_grid, xs, ys)).astype(int)
            rgb[np.clip(line_y, 0, price_h - 1), x_grid] = np.asarray(col, dtype=np.float32)

    # Anti-aliasing: box downsample from (S, S) -> (size, size).
    if ss > 1:
        aligned = rgb.reshape(size, ss, size, ss, 3)
        rgb = aligned.mean(axis=(1, 3)).astype(np.float32)

    return rgb


def render_ohlcv(ohlcv: np.ndarray, spec: Optional[RenderSpec] = None) -> torch.Tensor:
    """Render ``(N, >=5)`` OHLCV to a ``(3, size, size)`` tensor (canonical)."""
    spec = spec or RenderSpec()
    rgb = _render_ohlcv_canonical(ohlcv, spec)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()


# ---------------------------------------------------------------------------
# Public entry points (single canonical path)
# ---------------------------------------------------------------------------


def render_chart(df: pd.DataFrame, size: int = 64, spec: Optional[RenderSpec] = None) -> torch.Tensor:
    """Render a candlestick + volume chart from an OHLCV window.

    Always uses the canonical renderer regardless of ``ZHISA_FAST_RENDER``.
    Returns a ``torch.FloatTensor`` of shape ``(3, size, size)`` in [0, 1].
    """
    if df is None or len(df) == 0:
        return torch.full((3, size, size), 0.5, dtype=torch.float32)
    spec = spec or RenderSpec(size=size)
    values = df[["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64, copy=False
    )
    rgb = _render_ohlcv_canonical(values, spec)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()


def render_chart_array(ohlcv: np.ndarray, size: int = 64, spec: Optional[RenderSpec] = None) -> torch.Tensor:
    """Render a contiguous ``(N, 5)`` OHLCV array without pandas overhead."""
    spec = spec or RenderSpec(size=size)
    rgb = _render_ohlcv_canonical(ohlcv, spec)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()


def render_chart_batch(
    windows: list[pd.DataFrame],
    size: int = 64,
    spec: Optional[RenderSpec] = None,
) -> torch.Tensor:
    """Batched wrapper around :func:`render_chart`."""
    return torch.stack([render_chart(w, size=size, spec=spec) for w in windows], dim=0)


# ---------------------------------------------------------------------------
# Visualization-only matplotlib path (explicit; never used by training)
# ---------------------------------------------------------------------------


def _draw_candles(ax, df: pd.DataFrame) -> None:
    if not _HAS_MPL:
        return
    x = np.arange(len(df))
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    width = 0.6
    for i in range(len(df)):
        color = _GREEN if c[i] >= o[i] else _RED
        ax.plot([i, i], [l[i], h[i]], color=color, linewidth=0.7, solid_capstyle="butt")
        body_low = min(o[i], c[i])
        body_height = max(abs(o[i] - c[i]), 1e-9 * (h[i] - l[i] + 1e-12))
        rect = Rectangle((i - width / 2, body_low), width, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.4)
        ax.add_patch(rect)
    ax.set_xlim(-0.5, len(df) - 0.5)


def _draw_overlay(ax, df: pd.DataFrame, kind: str, period: int, color: str) -> None:
    if not _HAS_MPL:
        return
    s = df["close"].rolling(period, min_periods=1).mean()
    ax.plot(np.arange(len(df)), s.to_numpy(), color=color, linewidth=1.0, label=f"{kind}{period}")


def _draw_volume(ax, df: pd.DataFrame) -> None:
    if not _HAS_MPL:
        return
    x = np.arange(len(df))
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    vmax = v.max() if v.max() > 0 else 1.0
    for i in range(len(df)):
        color = _GREEN if c[i] >= o[i] else _RED
        ax.plot([i, i], [0, v[i] / vmax], color=color, linewidth=1.0, alpha=0.55)


def _df_to_image(df: pd.DataFrame, size: int) -> np.ndarray:
    """Matplotlib-backed raster (visualization only, not deterministic SOT)."""
    if not _HAS_MPL:
        return np.full((size, size, 3), 0.5, dtype=np.float32)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    gs = fig.add_gridspec(4, 1, hspace=0.0)
    ax_price = fig.add_subplot(gs[0:3, 0])
    ax_vol = fig.add_subplot(gs[3, 0], sharex=ax_price)
    for ax in (ax_price, ax_vol):
        ax.set_facecolor(DEFAULT_BG)
        for spine in ax.spines.values():
            spine.set_color(DEFAULT_GREY)
        ax.tick_params(colors=DEFAULT_FG, labelsize=4, length=0)
    fig.patch.set_facecolor(DEFAULT_BG)

    _draw_candles(ax_price, df)
    _draw_overlay(ax_price, df, "SMA", 10, "#3aa0ff")
    _draw_overlay(ax_price, df, "SMA", 30, "#ffaa33")
    _draw_volume(ax_vol, df)
    ax_price.set_ylim(df["low"].min(), df["high"].max())
    ax_vol.set_ylim(0, 1)
    ax_vol.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    try:
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_argb()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        rgb = arr[:, :, 1:4]  # ARGB -> RGB
    except Exception:
        bio = BytesIO()
        fig.savefig(bio, format="png", facecolor=fig.get_facecolor())
        bio.seek(0)
        from PIL import Image
        rgb = np.asarray(Image.open(bio).convert("RGB"))
    plt.close(fig)
    return rgb.astype(np.float32) / 255.0


def render_chart_visualization(df: pd.DataFrame, size: int = 64) -> np.ndarray:
    """Explicitly ask for the matplotlib visualisation raster.

    This is for humans (debugging / interpretability) only. It is **not**
    a training input source, so its non-determinism is acceptable and it is
    intentionally separate from the canonical renderer used in the pipeline.
    """
    if df is None or len(df) == 0:
        return np.full((size, size, 3), 0.5, dtype=np.float32)
    rgb = _df_to_image(df, size)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        try:
            from PIL import Image
            img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size))
            rgb = np.asarray(img).astype(np.float32) / 255.0
        except Exception:
            rgb = np.full((size, size, 3), 0.5, dtype=np.float32)
    return rgb


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated; kept until all callers migrate)
# ---------------------------------------------------------------------------

# Historical pure-numpy renderer without anti-aliasing. Kept for auditability
# and for tests that compare the legacy raster against the canonical one. The
# canonical renderer is always the production path.
def _fast_render_ohlcv(ohlcv: np.ndarray, size: int) -> np.ndarray:
    if len(ohlcv) < 2:
        return np.full((size, size, 3), 0.5, dtype=np.float32)
    spec = RenderSpec(size=size, supersample=1)
    return _render_ohlcv_canonical(ohlcv, spec)


def _fast_render(df: pd.DataFrame, size: int) -> np.ndarray:
    values = df[["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64, copy=False
    )
    return _fast_render_ohlcv(values, size)


# Deprecated matplotlib-drawing helpers, kept for external imports.
_GREEN = DEFAULT_GREEN
_RED = DEFAULT_RED
_GREY = DEFAULT_GREY


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray) -> None:
    """Bresenham-style line drawing onto an HxWx3 image (clipped)."""
    H, W = img.shape[:2]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy