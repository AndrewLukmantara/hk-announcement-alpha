"""Price panel, liquidity, and flow features. Every feature is lagged so it is known at the entry close."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def build_panel(prices: pd.DataFrame, index_df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    close = prices.pivot(index="date", columns="stock_code", values="close").sort_index()
    volume = prices.pivot(index="date", columns="stock_code", values="volume").reindex_like(close)
    ret = close / close.shift(1) - 1.0
    turnover = close * volume
    adv20 = turnover.rolling(20, min_periods=10).mean()
    idx = index_df.set_index("date")["close"].sort_index().reindex(close.index).ffill()
    mkt_ret = idx / idx.shift(1) - 1.0
    return {"close": close, "ret": ret, "turnover": turnover, "adv20": adv20, "mkt_ret": mkt_ret}


def southbound_flow_z(sb: pd.DataFrame, dates, stocks, window: int = 5, lag: int = 1) -> pd.DataFrame:
    """Cross-sectional z of the change in Southbound % of issued shares.

    CCASS holdings for day d are published the next morning, so lag >= 1.
    """
    if sb is None or sb.empty:
        return pd.DataFrame(np.nan, index=dates, columns=stocks)
    pct = (sb.pivot_table(index="date", columns="stock_code", values="sb_pct", aggfunc="last")
             .reindex(index=dates, columns=stocks).ffill(limit=3))
    return cs_zscore(pct.diff(window).shift(lag))


def short_ratio_z(ss: pd.DataFrame, turnover: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    if ss is None or ss.empty:
        return pd.DataFrame(np.nan, index=turnover.index, columns=turnover.columns)
    st = (ss.pivot_table(index="date", columns="stock_code", values="short_turnover_hkd", aggfunc="sum")
            .reindex_like(turnover))
    ratio = (st / turnover.replace(0, np.nan)).clip(0, 1)
    return cs_zscore(ratio.shift(lag))


def lookup(panel: pd.DataFrame, dates: pd.Series, codes: pd.Series) -> np.ndarray:
    """Vectorised panel.loc[date, code] for many (date, code) pairs; NaN when missing."""
    s = panel.stack()  # missing pairs come back as NaN from reindex either way
    keys = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), pd.Index(codes)])
    return s.reindex(keys).to_numpy(dtype=float)
