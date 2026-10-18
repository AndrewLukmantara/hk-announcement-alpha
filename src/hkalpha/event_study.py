"""Market-model event study and the news x flow interaction regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def abnormal_returns(ret: pd.DataFrame, mkt_ret: pd.Series, events: pd.DataFrame,
                     est_window=(-250, -30), max_h: int = 20, min_obs: int = 120) -> pd.DataFrame:
    """AR[k] for k = 0..max_h relative to each event's entry date.

    AR[0] is the reaction (prior close -> entry close): not tradable.
    AR[1..h] is the drift after the entry close: what a strategy can capture.
    Suspended days (NaN returns) count as zero abnormal return.
    """
    dates = ret.index
    R = ret.to_numpy()
    M = mkt_ret.reindex(dates).to_numpy()
    cols = {c: i for i, c in enumerate(ret.columns)}
    t0s = dates.get_indexer(pd.DatetimeIndex(events["entry_date"]))
    out, idx = [], []
    for ev_id, code, t0 in zip(events.index, events["stock_code"], t0s):
        j = cols.get(code)
        a, b = t0 + est_window[0], t0 + est_window[1]
        if j is None or t0 < 0 or a < 1 or t0 + max_h >= len(dates):
            continue
        y, x = R[a:b + 1, j], M[a:b + 1]
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < min_obs:
            continue
        beta, alpha = np.polyfit(x[ok], y[ok], 1)
        win = R[t0:t0 + max_h + 1, j] - (alpha + beta * M[t0:t0 + max_h + 1])
        out.append(np.nan_to_num(win))
        idx.append(ev_id)
    return pd.DataFrame(out, index=idx, columns=range(max_h + 1))


def car_columns(ar: pd.DataFrame, horizons) -> pd.DataFrame:
    df = pd.DataFrame({"car_0": ar[0]}, index=ar.index)
    for h in horizons:
        df[f"drift_{h}"] = ar.loc[:, 1:h].sum(axis=1)
    return df


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def summarise(df: pd.DataFrame, group: str, horizons) -> pd.DataFrame:
    cols = ["car_0"] + [f"drift_{h}" for h in horizons]
    rows = []
    for g, d in df.groupby(group, observed=True):
        r = {group: g, "n": len(d)}
        for c in cols:
            r[f"{c}_bps"] = d[c].mean() * 1e4
            r[f"{c}_t"] = _tstat(d[c])
        rows.append(r)
    return pd.DataFrame(rows).sort_values(f"drift_{horizons[1] if len(horizons) > 1 else horizons[0]}_bps")


def sentiment_bucket(s: pd.Series, thr: float = 0.2) -> pd.Series:
    return pd.cut(s, [-np.inf, -thr, thr, np.inf], labels=["negative", "neutral", "positive"])


def interaction_regression(df: pd.DataFrame, y: str, sent: str):
    """y ~ sent * flow_z + sent * short_z + log_adv, SEs clustered by entry date."""
    terms = [sent]
    for f in ["flow_z", "short_z"]:
        if f in df and df[f].notna().mean() > 0.3:
            terms += [f, f"{sent}:{f}"]
    if "log_adv" in df:
        terms.append("log_adv")
    d = df.dropna(subset=[y] + [t for t in terms if ":" not in t]).copy()
    d = d[d[sent] != 0]
    if len(d) < 50:
        return None
    d["_cluster"] = pd.factorize(d["entry_date"])[0]
    model = smf.ols(f"{y} ~ {' + '.join(terms)}", data=d)
    return model.fit(cov_type="cluster", cov_kwds={"groups": d["_cluster"]})
