"""Time-based feature encodings (cyclic embeddings of date/time)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_time_features(
    df: pd.DataFrame,
    *,
    add_minute: bool = True,
    add_hour: bool = True,
    add_dayofweek: bool = True,
    add_dayofmonth: bool = True,
    add_month: bool = True,
) -> pd.DataFrame:
    """Cyclic sin/cos embeddings of the datetime components of ``df.index``.

    Encodes each chosen component as a 2-D (sin, cos) vector to make the
    representation truly cyclic (no ordinal ordering between, e.g., hour
    23 and hour 0).
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")
    out = pd.DataFrame(index=idx)

    def cyclic(period: float, values: np.ndarray) -> tuple[pd.Series, pd.Series]:
        ang = 2.0 * np.pi * values / period
        return np.sin(ang), np.cos(ang)

    if add_minute:
        m = idx.minute.to_numpy() + idx.second.to_numpy() / 60.0
        s, c = cyclic(60.0, m)
        out["sin_minute"], out["cos_minute"] = s, c
    if add_hour:
        h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
        s, c = cyclic(24.0, h)
        out["sin_hour"], out["cos_hour"] = s, c
    if add_dayofweek:
        s, c = cyclic(7.0, idx.dayofweek.to_numpy() + idx.hour.to_numpy() / 24.0)
        out["sin_dow"], out["cos_dow"] = s, c
    if add_dayofmonth:
        s, c = cyclic(31.0, idx.day.to_numpy())
        out["sin_dom"], out["cos_dom"] = s, c
    if add_month:
        s, c = cyclic(12.0, idx.month.to_numpy() - 1.0)
        out["sin_month"], out["cos_month"] = s, c

    return out.astype(np.float64)
