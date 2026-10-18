"""Map announcement timestamps (HKT) to the first close you could realistically trade at.

HKEX sessions: 09:30-12:00, lunch 12:00-13:00, 13:00-16:00. Issuers publish mostly
after the close, before the open, or during lunch. An announcement only becomes
tradable at the first close that is at least `lag_minutes` after publication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
LUNCH_START = pd.Timedelta(hours=12)
LUNCH_END = pd.Timedelta(hours=13)
SESSION_CLOSE = pd.Timedelta(hours=16)


def _calendar(trading_days) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_days).normalize())))


def classify_session(dt: pd.Series, trading_days) -> pd.Series:
    td = _calendar(trading_days)
    day = dt.dt.normalize()
    tod = dt - day
    labels = np.select(
        [~day.isin(td), tod < SESSION_OPEN, tod < LUNCH_START, tod < LUNCH_END, tod < SESSION_CLOSE],
        ["non_trading_day", "pre_open", "morning", "lunch", "afternoon"],
        default="post_close",
    )
    return pd.Series(labels, index=dt.index)


def assign_entry_dates(ann: pd.DataFrame, trading_days, lag_minutes: int = 30) -> pd.DataFrame:
    """Add `entry_date` (first tradable close) and `session` columns. Ignores half-days."""
    td = _calendar(trading_days)
    closes = td + SESSION_CLOSE
    cutoff = (ann["datetime"] + pd.Timedelta(minutes=lag_minutes)).values
    pos = closes.searchsorted(cutoff, side="left")
    valid = pos < len(td)
    entry = np.full(len(ann), np.datetime64("NaT"), dtype="datetime64[ns]")
    entry[valid] = td.values[pos[valid]]
    out = ann.copy()
    out["entry_date"] = entry
    out["session"] = classify_session(out["datetime"], td).values
    return out
