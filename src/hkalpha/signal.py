"""Turn scored events into a daily cross-sectional score (stocks x dates).

The score on date t only uses events whose entry_date <= t, so a position formed
at the close of t is tradable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def event_matrix(events: pd.DataFrame, value_col: str, dates, stocks) -> pd.DataFrame:
    e = events.dropna(subset=["entry_date", value_col])
    e = e[e["stock_code"].isin(stocks)]
    m = e.groupby(["entry_date", "stock_code"])[value_col].sum().unstack()
    return m.reindex(index=dates, columns=stocks).fillna(0.0)


def decayed(x: pd.DataFrame, half_life: float) -> pd.DataFrame:
    d = 0.5 ** (1.0 / half_life)
    vals = x.to_numpy()
    out = np.empty_like(vals)
    acc = np.zeros(vals.shape[1])
    for t in range(len(vals)):
        acc = acc * d + vals[t]
        out[t] = acc
    out[np.abs(out) < 1e-3] = 0.0
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def flow_tilt(score: pd.DataFrame, flow_z: pd.DataFrame, k: float = 0.5) -> pd.DataFrame:
    """Scale news scores up when Southbound is accumulating, down when it's distributing."""
    return score * (1 + k * flow_z.reindex_like(score).fillna(0).clip(-2, 2)).clip(lower=0)
