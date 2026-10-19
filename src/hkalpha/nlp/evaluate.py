"""Hand-labelling workflow: export a stratified sample, label it, then benchmark scorers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def export_label_sample(ann: pd.DataFrame, path: Path, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Stratify by HKEX category so rare-but-important events are represented."""
    uniq = ann.drop_duplicates("news_id")
    strata = uniq.get("category_en", pd.Series("all", index=uniq.index)).fillna("unknown")
    # HKEX joins several headline tags with " / " (thousands of combinations); stratify on the first
    # tag only, and drop the "(Headlines Revised)"-style prefixes.
    strata = (strata.str.replace(r"^\(.*?\)\s*", "", regex=True)
                    .str.split(" / ").str[0].str.rstrip("]").str.strip())
    per = max(1, n // max(1, strata.nunique()))
    # sample row labels directly: groupby().apply() drops the grouping column in pandas >= 3
    idx = [i for _, g in uniq.groupby(strata) for i in g.sample(min(len(g), per), random_state=seed).index]
    sample = uniq.loc[idx]
    if len(sample) > n:
        sample = sample.sample(n, random_state=seed)
    if len(sample) < n:
        rest = uniq.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(len(rest), n - len(sample)), random_state=seed)])
    cols = [c for c in ["news_id", "stock_code", "datetime", "title_en", "title_zh", "category_en"] if c in sample]
    sample = sample[cols].copy()
    sample["label_sentiment"] = ""   # fill with -1, -0.5, 0, 0.5, 1
    sample["label_event_type"] = ""
    sample.to_csv(path, index=False, encoding="utf-8-sig")
    return sample


def evaluate_scorers(labels: pd.DataFrame, scores: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """labels: news_id + label_sentiment; scores: news_id + scorer columns."""
    df = labels.merge(scores.drop_duplicates("news_id"), on="news_id", how="inner")
    df["label_sentiment"] = pd.to_numeric(df["label_sentiment"], errors="coerce")
    df = df.dropna(subset=["label_sentiment"])
    rows = []
    for c in cols:
        d = df.dropna(subset=[c])
        if len(d) < 10:
            continue
        rho = spearmanr(d[c], d["label_sentiment"]).statistic
        sign_acc = (np.sign(d[c].round(1)) == np.sign(d["label_sentiment"])).mean()
        rows.append({"scorer": c, "n": len(d), "spearman": rho, "sign_accuracy": sign_acc,
                     "coverage_nonzero": (d[c].abs() > 0).mean()})
    return pd.DataFrame(rows)
