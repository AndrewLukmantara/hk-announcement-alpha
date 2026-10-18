"""Synthetic HK market with a *known* planted effect, for verifying the pipeline offline.

Planted truth (so you can check the pipeline recovers it):
  * reaction on the entry day:  3% x true sentiment
  * post-entry drift over 5 days: 2% x sentiment x (1 + 0.6 x Southbound flow z)
  * 10% of material events get a vague title ("INSIDE INFORMATION") the keyword scorer misses;
    another 16% are informative in only one language (tests the bilingual scorer).
Results on this data say nothing about real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import cs_zscore
from ..timing import assign_entry_dates

#            event_type,           base, english title,                                              chinese title,               category,           weight
TEMPLATES = [
    ("earnings_positive",  0.8, "POSITIVE PROFIT ALERT",                                   "正面盈利預告",               "Inside Information", 6),
    ("earnings_negative", -0.8, "PROFIT WARNING",                                          "盈利警告",                   "Inside Information", 6),
    ("buyback",            0.4, "VOLUNTARY ANNOUNCEMENT - SHARE BUY-BACK",                 "自願公告 - 股份回購",         "Share Buyback",       8),
    ("dividend",           0.5, "DECLARATION OF SPECIAL DIVIDEND",                         "宣派特別股息",               "Dividend",            3),
    ("placing",           -0.6, "PLACING OF NEW SHARES UNDER GENERAL MANDATE",             "根據一般授權配售新股",         "Placing",             4),
    ("rights_issue",      -0.6, "PROPOSED RIGHTS ISSUE",                                   "建議供股",                   "Rights Issue",        2),
    ("privatisation",      0.9, "PROPOSED PRIVATISATION BY WAY OF A SCHEME OF ARRANGEMENT", "建議以協議安排方式私有化",     "Privatisation",       0.5),
    ("management_change", -0.2, "RESIGNATION OF EXECUTIVE DIRECTOR",                       "執行董事辭任",               "Change in Directors", 6),
    ("litigation",        -0.5, "LITIGATION UPDATE",                                       "訴訟最新情況",               "Inside Information",  2),
    ("contract",           0.4, "STRATEGIC COOPERATION AGREEMENT",                         "戰略合作協議",               "Inside Information",  4),
    ("routine",            0.0, "MONTHLY RETURN OF EQUITY ISSUER ON MOVEMENTS IN SECURITIES", "股份發行人的證券變動月報表", "Monthly Returns",    40),
    ("routine",            0.0, "NOTICE OF BOARD MEETING",                                 "董事會會議通告",             "Board Meeting",       20),
    ("routine",            0.0, "NEXT DAY DISCLOSURE RETURN",                              "翌日披露報表",               "Next Day Disclosure", 15),
]


def generate(cfg: dict) -> dict[str, pd.DataFrame]:
    sc = cfg.get("synthetic", {})
    rng = np.random.default_rng(sc.get("seed", 7))
    n = sc.get("n_stocks", 80)
    days = pd.bdate_range(cfg["dates"]["start"], cfg["dates"]["end"])
    days = days[rng.random(len(days)) > 0.03]  # crude public holidays
    T = len(days)
    codes = [f"{c:05d}" for c in rng.choice(np.arange(1, 9999), n, replace=False)]

    mkt = rng.normal(0.0002, 0.012, T)
    beta = rng.uniform(0.6, 1.4, n)
    ret = mkt[:, None] * beta + rng.normal(0, 1, (T, n)) * rng.uniform(0.012, 0.03, n)

    sb_pct = np.clip(rng.uniform(0, 20, n) + np.cumsum(rng.normal(0, 0.05, (T, n)), axis=0), 0, 70)
    sb_df = pd.DataFrame(sb_pct, index=days, columns=codes)
    flow_z = cs_zscore(sb_df.diff(cfg["signal"]["flow_window"]).shift(1)).fillna(0).to_numpy()

    # announcements
    w = np.array([t[5] for t in TEMPLATES], dtype=float)
    w /= w.sum()
    hits = np.argwhere(rng.random((T, n)) < 0.06)
    recs = []
    for k, (t, j) in enumerate(hits):
        tpl = TEMPLATES[rng.choice(len(TEMPLATES), p=w)]
        etype, base, en, zh, cat = tpl[:5]
        u_lang = rng.random()
        if base != 0 and u_lang < 0.10:
            en, zh = "INSIDE INFORMATION", "內幕消息"
        elif base != 0 and u_lang < 0.18:
            en = "ANNOUNCEMENT"        # only the Chinese title is informative
        elif base != 0 and u_lang < 0.26:
            zh = "公告"                 # only the English title is informative
        u = rng.random()  # publication time mix typical of HKEX
        if u < 0.5:
            off = pd.Timedelta(minutes=int(rng.integers(16 * 60 + 30, 23 * 60)))
        elif u < 0.7:
            off = pd.Timedelta(minutes=int(rng.integers(7 * 60, 9 * 60)))
        elif u < 0.9:
            off = pd.Timedelta(minutes=int(rng.integers(12 * 60, 12 * 60 + 59)))
        else:
            off = pd.Timedelta(minutes=int(rng.integers(13 * 60, 15 * 60 + 59)))
        true_s = float(np.clip(base + rng.normal(0, 0.15), -1, 1)) if base != 0 else 0.0
        recs.append({"news_id": str(100000 + k), "stock_code": codes[j], "datetime": days[t] + off,
                     "title_en": en, "title_zh": zh, "category_en": cat, "category_zh": "",
                     "stock_name_en": f"STOCK {codes[j]}", "stock_name_zh": "", "_true": true_s})
    ann = assign_entry_dates(pd.DataFrame(recs), days, cfg["market"]["entry_lag_minutes"])
    pos = pd.Index(days).get_indexer(ann["entry_date"])
    col = pd.Index(codes).get_indexer(ann["stock_code"])
    for p, j, s in zip(pos, col, ann["_true"]):
        if p < 0 or s == 0:
            continue
        ret[p, j] += 0.03 * s
        drift = 0.02 * s * (1 + 0.6 * np.clip(flow_z[p, j], -2, 2)) / 5
        ret[p + 1:p + 6, j] += drift

    close = rng.uniform(5, 200, n) * np.cumprod(1 + ret, axis=0)
    turnover = np.exp(rng.normal(17.5, 1.0, n)) * np.exp(rng.normal(0, 0.4, (T, n)))
    prices = pd.DataFrame({
        "date": np.repeat(days.values, n), "stock_code": np.tile(codes, T),
        "close": close.ravel(), "volume": (turnover / close).ravel(),
    })
    prices["open"] = prices["high"] = prices["low"] = prices["close"]
    index_df = pd.DataFrame({"date": days, "close": 20000 * np.cumprod(1 + mkt)})
    sb = sb_df.stack().rename("sb_pct").reset_index().rename(columns={"level_0": "date", "level_1": "stock_code"})
    sb.columns = ["date", "stock_code", "sb_pct"]
    ss = pd.DataFrame({"date": prices["date"], "stock_code": prices["stock_code"],
                       "short_turnover_hkd": turnover.ravel() * rng.uniform(0.03, 0.2, T * n)})
    ann = ann.drop(columns=["entry_date", "session"])
    return {"announcements": ann, "prices": prices, "index": index_df, "southbound": sb, "short_selling": ss}
