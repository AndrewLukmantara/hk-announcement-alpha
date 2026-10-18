"""HKEXnews title-search scraper (English + Traditional Chinese).

Uses the JSON endpoint behind https://www1.hkexnews.hk/search/titlesearch.xhtml.
Parameters were taken from the site's own requests; if HKEX changes them, open the
title search page, run a search with your browser's network tab open, and update
PARAMS below. Be polite: one request per second, identify yourself in the UA.
"""
from __future__ import annotations

import html
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www1.hkexnews.hk/search/titleSearchServlet.do"
HOST = "https://www1.hkexnews.hk"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research project; contact: your-email@connect.hku.hk)"}
PARAMS = dict(sortDir="0", sortByOptions="DateTime", category="0", market="SEHK", stockId="-1",
              documentType="-1", title="", searchType="0", t1code="-2", t2Gcode="-2", t2code="-2",
              rowRange="5000")
_TAG = re.compile(r"<[^>]+>")


def _clean(s) -> str:
    return html.unescape(_TAG.sub(" ", s or "")).strip() if isinstance(s, str) else ""


def query_day(day: date, lang: str, session: requests.Session) -> list[dict]:
    """lang: 'E' (English) or 'ZH' (Traditional Chinese)."""
    d = day.strftime("%Y%m%d")
    r = session.get(BASE, params={**PARAMS, "fromDate": d, "toDate": d, "lang": lang},
                    headers=HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json()
    result = payload.get("result")
    rows = json.loads(result) if isinstance(result, str) else (result or [])
    if payload.get("hasNextRow"):
        print(f"  warning {d} {lang}: more rows than rowRange; raise rowRange")
    return rows


def parse_rows(rows: list[dict], lang: str) -> pd.DataFrame:
    sfx = "en" if lang == "E" else "zh"
    out = []
    for row in rows:
        codes = [c.strip() for c in re.split(r"<br\s*/?>", row.get("STOCK_CODE", "")) if c.strip()]
        names = [_clean(n) for n in re.split(r"<br\s*/?>", row.get("STOCK_NAME", ""))]
        for i, code in enumerate(codes):  # multi-issuer announcements -> one row per stock
            out.append({
                "news_id": str(row.get("NEWS_ID")),
                "stock_code": code.zfill(5),
                f"stock_name_{sfx}": names[i] if i < len(names) else "",
                "datetime": pd.to_datetime(row.get("DATE_TIME"), format="%d/%m/%Y %H:%M"),
                f"title_{sfx}": _clean(row.get("TITLE")),
                f"category_{sfx}": _clean(row.get("LONG_TEXT")),
                f"url_{sfx}": HOST + row.get("FILE_LINK", ""),
            })
    return pd.DataFrame(out)


def merge_languages(en: pd.DataFrame, zh: pd.DataFrame) -> pd.DataFrame:
    if en.empty or zh.empty:
        return en if zh.empty else zh
    m = en.merge(zh.drop(columns="datetime"), on=["news_id", "stock_code"], how="left")
    if m["title_zh"].notna().mean() < 0.5:
        # NEWS_IDs differ across languages -> align by (stock, timestamp, order within it)
        for df in (en, zh):
            df["_k"] = df.groupby(["stock_code", "datetime"]).cumcount()
        m = en.merge(zh.drop(columns="news_id"), on=["stock_code", "datetime", "_k"], how="left").drop(columns="_k")
    return m


def fetch_range(start: str, end: str, out_path: Path, sleep: float = 1.0) -> pd.DataFrame:
    out_path = Path(out_path)
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    done = set(existing["datetime"].dt.date) if not existing.empty else set()
    s = requests.Session()
    frames, day, last = [], pd.Timestamp(start).date(), pd.Timestamp(end).date()
    while day <= last:
        if day not in done:
            try:
                en = parse_rows(query_day(day, "E", s), "E")
                time.sleep(sleep)
                zh = parse_rows(query_day(day, "ZH", s), "ZH")
                time.sleep(sleep)
                if not en.empty or not zh.empty:
                    frames.append(merge_languages(en, zh))
                print(f"{day}: {len(en)} rows")
            except Exception as e:
                print(f"{day}: failed ({e}); rerun to resume")
            if len(frames) >= 20:  # checkpoint
                existing = _save(existing, frames, out_path)
                frames = []
        day += timedelta(days=1)
    return _save(existing, frames, out_path)


def _save(existing: pd.DataFrame, frames: list, path: Path) -> pd.DataFrame:
    df = pd.concat([existing, *frames], ignore_index=True) if frames else existing
    if not df.empty:
        df = df.drop_duplicates(["news_id", "stock_code"]).sort_values("datetime").reset_index(drop=True)
        df.to_parquet(path, index=False)
    return df


def fetch_pdf_excerpt(url: str, max_chars: int = 2000) -> str:
    """First page(s) of the announcement PDF, for richer LLM context (optional)."""
    import io
    from pypdf import PdfReader
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    reader = PdfReader(io.BytesIO(r.content))
    text = ""
    for page in reader.pages[:3]:
        text += (page.extract_text() or "") + "\n"
        if len(text) >= max_chars:
            break
    return re.sub(r"\s+", " ", text)[:max_chars]
