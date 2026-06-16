"""Chart image renderer: OHLCV DataFrame window -> RGB tensor.

The renderer is deliberately lightweight — it draws a candlestick chart
with a small set of overlays and returns a ``torch.Tensor`` of shape
``(3, H, W)`` in [0, 1]. It is used both as a model input channel and
as a debugging / interpretability tool.
"""
from __future__ import annotations

import os
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.patches import Rectangle  # noqa: E402
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


_DEFAULT_BG = (0.05, 0.06, 0.09)
_DEFAULT_FG = (0.85, 0.88, 0.92)
_GREEN = (0.20, 0.80, 0.40)
_RED = (0.95, 0.35, 0.30)
_GREY = (0.45, 0.48, 0.55)


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
        # Wick
        ax.plot([i, i], [l[i], h[i]], color=color, linewidth=0.7, solid_capstyle="butt")
        # Body
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
    if not _HAS_MPL:
        # Last-resort: a tiny placeholder image (so the rest of the pipeline
        # can still be exercised in environments without matplotlib).
        return np.full((size, size, 3), 0.5, dtype=np.float32)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    gs = fig.add_gridspec(4, 1, hspace=0.0)
    ax_price = fig.add_subplot(gs[0:3, 0])
    ax_vol = fig.add_subplot(gs[3, 0], sharex=ax_price)
    for ax in (ax_price, ax_vol):
        ax.set_facecolor(_DEFAULT_BG)
        for spine in ax.spines.values():
            spine.set_color(_GREY)
        ax.tick_params(colors=_DEFAULT_FG, labelsize=4, length=0)
    fig.patch.set_facecolor(_DEFAULT_BG)

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
        # Fallback via savefig to a buffer (works on more backends)
        bio = BytesIO()
        fig.savefig(bio, format="png", facecolor=fig.get_facecolor())
        bio.seek(0)
        from PIL import Image
        rgb = np.asarray(Image.open(bio).convert("RGB"))
    plt.close(fig)
    return rgb.astype(np.float32) / 255.0


def render_chart(df: pd.DataFrame, size: int = 64) -> torch.Tensor:
    """Render a small candlestick + volume chart from an OHLCV window.

    Returns a ``torch.FloatTensor`` of shape ``(3, H, W)`` in [0, 1].
    Falls back to a uniform grey image if matplotlib is unavailable.

    Set the environment variable ``ZHISA_FAST_RENDER=1`` to use the
    pure-numpy renderer (much faster, used by default in tests).
    """
    if df is None or len(df) == 0:
        return torch.full((3, size, size), 0.5, dtype=torch.float32)
    use_fast = os.environ.get("ZHISA_FAST_RENDER", "0") == "1"
    if use_fast:
        rgb = _fast_render(df, size)
    else:
        rgb = _df_to_image(df, size)
    # Resize to (size, size) if needed (defensive)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        try:
            from PIL import Image
            img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size))
            rgb = np.asarray(img).astype(np.float32) / 255.0
        except Exception:
            rgb = np.full((size, size, 3), 0.5, dtype=np.float32)
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return t.float()


def _fast_render(df: pd.DataFrame, size: int) -> np.ndarray:
    """A pure-numpy candlestick renderer; an order of magnitude faster
    than matplotlib and good enough for feature extraction."""
    if len(df) < 2:
        return np.full((size, size, 3), 0.5, dtype=np.float32)
    n = len(df)
    rgb = np.full((size, size, 3), 0.05, dtype=np.float32)
    
    # OPTIMIZATION: Extract arrays ONCE to avoid 40 million Pandas .iloc calls
    o_arr = df["open"].to_numpy(dtype=np.float64)
    c_arr = df["close"].to_numpy(dtype=np.float64)
    h_arr = df["high"].to_numpy(dtype=np.float64)
    l_arr = df["low"].to_numpy(dtype=np.float64)
    v_arr = df["volume"].to_numpy(dtype=np.float64)
    
    lo = float(l_arr.min())
    hi = float(h_arr.max())
    rng = max(hi - lo, 1e-9)
    # Map bars to x columns
    xs = (np.arange(n) * (size - 1) / max(n - 1, 1)).astype(int)
    # Main price area: top 75% of rows
    price_h = int(size * 0.75)
    # Candles
    for i in range(n):
        x = xs[i]
        o, c, h, l = float(o_arr[i]), float(c_arr[i]), float(h_arr[i]), float(l_arr[i])
        y_o = price_h - int((o - lo) / rng * (price_h - 2))
        y_c = price_h - int((c - lo) / rng * (price_h - 2))
        y_h = price_h - int((h - lo) / rng * (price_h - 2))
        y_l = price_h - int((l - lo) / rng * (price_h - 2))
        color = np.array(_GREEN if c >= o else _RED, dtype=np.float32)
        # Wick
        for y in range(min(y_h, y_l), max(y_h, y_l) + 1):
            rgb[y, x] = color
        # Body
        for y in range(min(y_o, y_c), max(y_o, y_c) + 1):
            rgb[y, x] = color
    # Volume bars on bottom 25%
    vmax = max(v_arr.max(), 1e-9)
    for i in range(n):
        x = xs[i]
        v = v_arr[i] / vmax
        bar_h = int(v * (size - price_h - 1))
        o, c = float(o_arr[i]), float(c_arr[i])
        color = np.array(_GREEN if c >= o else _RED, dtype=np.float32)
        for y in range(size - bar_h, size):
            rgb[y, x] = color
    # Two moving averages (SMA10 / SMA30) as small lines
    for p, col in ((10, np.array([0.23, 0.63, 1.0], dtype=np.float32)),
                   (30, np.array([1.0, 0.67, 0.20], dtype=np.float32))):
        if n < p:
            continue
        # Pure numpy SMA to avoid Pandas DataFrame creation overhead
        kernel = np.ones(p) / p
        sma = np.convolve(c_arr, kernel, mode='valid')
        # Pad beginning to match min_periods=1 behavior
        pad_len = n - len(sma)
        if pad_len > 0:
            pad = np.cumsum(c_arr[:pad_len]) / np.arange(1, pad_len + 1)
            sma = np.concatenate((pad, sma))
            
        ys = price_h - ((sma - lo) / rng * (price_h - 2)).astype(int)
        for i in range(n - 1):
            cv = np.clip(ys[i], 0, price_h - 1)
            nv = np.clip(ys[i + 1], 0, price_h - 1)
            x1, x2 = xs[i], xs[i + 1]
            for x in range(x1, x2 + 1):
                y = int(cv + (nv - cv) * (x - x1) / max(x2 - x1, 1))
                rgb[y, x] = col
    return rgb


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


def render_chart_batch(
    windows: list[pd.DataFrame],
    size: int = 64,
) -> torch.Tensor:
    """Batched wrapper around :func:`render_chart`."""
    return torch.stack([render_chart(w, size=size) for w in windows], dim=0)
