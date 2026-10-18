"""HKEX daily short-selling turnover (Main Board). The public page only shows the
latest day, so run `hkalpha fetch-shortselling` daily after 16:30 HKT (cron) to build history.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ncms/ASHTMAIN.HTM"
LINE = re.compile(r"^\s*(\d{1,5})\s+(.+?)\s+([\d,]+)\s+([\d,]+)\s*$")
DATE = re.compile(r"(\d{1,2})\s+([A-Z]{3})\s+(\d{4})")


def parse_report(text: str) -> pd.DataFrame:
    m = DATE.search(text.upper())
    day = pd.to_datetime(" ".join(m.groups()), format="%d %b %Y") if m else pd.Timestamp.today().normalize()
    rows = []
    for line in re.sub(r"<[^>]+>", "", text).splitlines():
        g = LINE.match(line)
        if g:
            rows.append({"date": day, "stock_code": g.group(1).zfill(5),
                         "short_shares": int(g.group(3).replace(",", "")),
                         "short_turnover_hkd": float(g.group(4).replace(",", ""))})
    return pd.DataFrame(rows)


def fetch_today(out_path: Path) -> pd.DataFrame:
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    new = parse_report(r.text)
    out_path = Path(out_path)
    old = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    df = pd.concat([old, new], ignore_index=True).drop_duplicates(["date", "stock_code"], keep="last")
    df.to_parquet(out_path, index=False)
    print(f"short selling: {len(new)} rows for {new['date'].iloc[0].date() if len(new) else 'n/a'}")
    return df
