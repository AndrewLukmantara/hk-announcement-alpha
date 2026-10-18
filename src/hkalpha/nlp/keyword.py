"""Keyword baseline scorer (bilingual)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lexicon import LEXICON


def score_text(text: str | None, lang: str) -> tuple[float, str, int]:
    """Return (sentiment in [-1,1], dominant event_type, n_matches)."""
    if not isinstance(text, str) or not text.strip():
        return 0.0, "none", 0
    t = text.lower() if lang == "en" else text
    total, etype, best, n = 0.0, "other", -1.0, 0
    for phrase, w, et in LEXICON[lang]:
        if phrase in t:
            total += w
            n += 1
            t = t.replace(phrase, " ")
            if abs(w) > best:
                best, etype = abs(w), et
    return float(np.clip(total, -1, 1)), etype, n


def _combine(en: tuple, zh: tuple) -> tuple[float, str, int]:
    (se, te, ne), (sz, tz, nz) = en, zh
    if ne and nz:
        return (se + sz) / 2, (te if abs(se) >= abs(sz) else tz), ne + nz
    return (se, te, ne) if ne else (sz, tz, nz)


def keyword_scores(ann: pd.DataFrame, mode: str = "both", fields: str = "title") -> pd.DataFrame:
    """mode: en | zh | both. fields: title (title text) or category (HKEX headline category only)."""
    col_en, col_zh = (("title_en", "title_zh") if fields == "title" else ("category_en", "category_zh"))
    rows = []
    for en_txt, zh_txt in zip(ann.get(col_en, pd.Series([None] * len(ann))),
                              ann.get(col_zh, pd.Series([None] * len(ann)))):
        en = score_text(en_txt, "en") if mode in ("en", "both") else (0.0, "none", 0)
        zh = score_text(zh_txt, "zh") if mode in ("zh", "both") else (0.0, "none", 0)
        rows.append(_combine(en, zh))
    out = pd.DataFrame(rows, columns=["sentiment", "event_type", "n_matches"], index=ann.index)
    out["confidence"] = np.minimum(1.0, out["n_matches"] * 0.5)
    return out
