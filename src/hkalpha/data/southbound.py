"""Southbound Stock Connect holdings from the HKEX CCASS 'mutual market' page.

https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk shows, for a chosen date,
every SEHK stock held by mainland investors via Stock Connect: shares held and
% of issued shares. The page is an ASP.NET form (needs __VIEWSTATE) and only keeps
about 12 months of history, so start collecting early and keep a daily cron going.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project)"}


def fetch_day(day: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    page = session.get(URL, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(page.text, "lxml")
    form = {i["name"]: i.get("value", "") for i in soup.select("input[name]")}
    form.update({"txtShareholdingDate": day.strftime("%Y/%m/%d"), "btnSearch": "Search",
                 "__EVENTTARGET": "btnSearch", "__EVENTARGUMENT": ""})
    r = session.post(URL, data=form, headers=HEADERS, timeout=60)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    t = next(t for t in tables if any("Stock Code" in str(c) for c in t.columns))
    t.columns = [str(c) for c in t.columns]
    col = lambda key: next(c for c in t.columns if key.lower() in c.lower())
    clean = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    return pd.DataFrame({
        "date": day.normalize(),
        "stock_code": t[col("Stock Code")].astype(str).str.extract(r"(\d+)")[0].str.zfill(5),
        "sb_shares": clean(t[col("Shareholding")]),
        "sb_pct": clean(t[col("%")]),
    }).dropna(subset=["stock_code"])


def fetch_range(start: str, end: str, out_path: Path, sleep: float = 2.0) -> pd.DataFrame:
    out_path = Path(out_path)
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done = set(existing["date"]) if not existing.empty else set()
    s, frames = requests.Session(), []
    for day in pd.bdate_range(start, end):
        if day in done:
            continue
        try:
            frames.append(fetch_day(day, s))
            print(f"{day.date()}: ok")
        except Exception as e:
            print(f"{day.date()}: skipped ({e})")
        time.sleep(sleep)
    df = pd.concat([existing, *frames], ignore_index=True) if frames else existing
    if not df.empty:
        df.drop_duplicates(["date", "stock_code"]).to_parquet(out_path, index=False)
    return df
