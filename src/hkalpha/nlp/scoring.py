"""Run every scorer variant and return one wide frame aligned to the announcements."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .keyword import keyword_scores


def score_all(ann: pd.DataFrame, cfg: dict, cache_path: Path, llm_offline: bool = True,
              excerpts: dict | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=ann.index)
    kw = keyword_scores(ann, "both")
    out["kw_both"] = kw["sentiment"]
    out["event_type"] = kw["event_type"]
    out["kw_en"] = keyword_scores(ann, "en")["sentiment"]
    out["kw_zh"] = keyword_scores(ann, "zh")["sentiment"]
    out["category"] = keyword_scores(ann, "both", fields="category")["sentiment"]

    if cfg["nlp"].get("use_llm"):
        from .llm import llm_scores
        for mode in cfg["nlp"].get("llm_lang_modes", ["both"]):
            s = llm_scores(ann, mode, cfg, cache_path, excerpts=excerpts, offline=llm_offline)
            out[f"llm_{mode}"] = s["sentiment"]
            out[f"llm_{mode}_surprise"] = s["surprise"]
            if mode == "both":
                out["llm_event_type"] = s["event_type"]
    return out
