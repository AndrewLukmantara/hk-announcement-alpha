from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(ret: pd.DataFrame, h: int) -> pd.DataFrame:
    """Cumulative log return from t+1 to t+h, aligned at t."""
    lr = np.log1p(ret.fillna(0.0))
    return lr.rolling(h).sum().shift(-h)


def rowwise_spearman(a: pd.DataFrame, b: pd.DataFrame, min_n: int = 5) -> pd.Series:
    mask = a.notna() & b.notna()
    ra = a.where(mask).rank(axis=1)
    rb = b.where(mask).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    ic = (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return ic.where(mask.sum(axis=1) >= min_n)


def ic_series(score: pd.DataFrame, ret: pd.DataFrame, eligible: pd.DataFrame, h: int) -> pd.Series:
    s = score.where(eligible & (score != 0))
    return rowwise_spearman(s, forward_returns(ret, h))


def ic_stats(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) < 10:
        return {"ic_mean": np.nan, "ic_ir": np.nan, "ic_t": np.nan, "n_days": len(ic)}
    return {"ic_mean": ic.mean(), "ic_ir": ic.mean() / ic.std(),
            "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)), "n_days": len(ic)}


def perf_stats(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan, "hit_rate": np.nan}
    eq = (1 + r).cumprod()
    out = {
        "ann_ret": (1 + r).prod() ** (252 / len(r)) - 1,
        "ann_vol": r.std() * np.sqrt(252),
        "sharpe": r.mean() / r.std() * np.sqrt(252),
        "max_dd": (eq / eq.cummax() - 1).min(),
        "hit_rate": (r[r != 0] > 0).mean(),
    }
    if turnover is not None:
        out["avg_daily_turnover"] = turnover.mean()
    return out
