"""Daily cross-sectional backtest with HK transaction costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    stamp_duty: float = 0.001
    trading_fee: float = 0.0000565
    sfc_levy: float = 0.000027
    afrc_levy: float = 0.0000015
    commission_bps: float = 3.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 100.0

    @property
    def per_side(self) -> float:
        return (self.stamp_duty + self.trading_fee + self.sfc_levy + self.afrc_levy
                + (self.commission_bps + self.slippage_bps) / 1e4)


def build_weights(score: pd.DataFrame, eligible: pd.DataFrame, mode: str = "long_short",
                  quantile: float = 0.3, min_names: int = 3) -> pd.DataFrame:
    s = score.where(eligible & (score != 0) & score.notna())
    r = s.rank(axis=1, pct=True)
    long = (r >= 1 - quantile) & (s > 0)
    short = (r <= quantile) & (s < 0)
    nl, ns = long.sum(axis=1), short.sum(axis=1)
    long = long & (nl >= min_names).to_numpy()[:, None]
    short = short & (ns >= min_names).to_numpy()[:, None]
    nl, ns = long.sum(axis=1).replace(0, np.nan), short.sum(axis=1).replace(0, np.nan)
    if mode == "long_only":
        w = long.astype(float).div(nl, axis=0)
    else:
        w = long.astype(float).div(nl, axis=0) * 0.5 - short.astype(float).div(ns, axis=0) * 0.5
    return w.fillna(0.0)


def rebalance_every(weights: pd.DataFrame, k: int) -> pd.DataFrame:
    """Only trade every k days (hold in between). Cuts turnover, which matters with 10 bps stamp duty."""
    if k <= 1:
        return weights
    keep = np.arange(len(weights)) % k == 0
    return weights[keep].reindex(weights.index).ffill().fillna(0.0)


def run_backtest(weights: pd.DataFrame, ret: pd.DataFrame, costs: Costs) -> pd.DataFrame:
    """Weights set at close t earn the t -> t+1 return; trades at close t pay costs."""
    ret = ret.reindex_like(weights).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross = (held * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = (turnover * costs.per_side).shift(1).fillna(0.0)
    borrow = held.clip(upper=0).abs().sum(axis=1) * costs.borrow_bps_annual / 1e4 / 252
    return pd.DataFrame({
        "gross": gross, "cost": cost + borrow, "net": gross - cost - borrow,
        "turnover": turnover, "n_long": (weights > 0).sum(axis=1), "n_short": (weights < 0).sum(axis=1),
    })
