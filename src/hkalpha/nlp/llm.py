"""LLM scorer (Anthropic Messages API) with an on-disk cache.

Look-ahead control: the company name and stock code are removed from the
prompt, and the model is told to ignore any memory of price moves. You should
still treat in-sample LLM results with suspicion if the sample predates the
model's training cutoff; the honest test is the period after it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.anthropic.com/v1/messages"
EVENT_TYPES = [
    "earnings_positive", "earnings_negative", "dividend", "buyback", "placing", "rights_issue",
    "mna", "privatisation", "management_change", "litigation", "contract", "trading_halt",
    "financing", "routine", "other",
]

SYSTEM = f"""You are a sell-side equity analyst covering companies listed on the Hong Kong Stock Exchange.
You will see one HKEX announcement: its title and headline category, in English and/or Traditional Chinese,
and sometimes an excerpt of the document. The issuer is anonymised as "the Company".
Judge ONLY from the text provided. Do not use any memory of how a specific company's share price moved.

Return ONLY a JSON object, with no prose and no code fences:
{{"event_type": one of {EVENT_TYPES},
 "sentiment": number in [-1, 1]  (expected 1-5 trading day price impact for existing shareholders; 0 if routine),
 "surprise": number in [0, 1]    (how unexpected and material),
 "confidence": number in [0, 1]}}"""


def _anonymise(text: str | None, names: list[str | None]) -> str:
    if not isinstance(text, str):
        return ""
    for n in names:
        if isinstance(n, str) and len(n.strip()) >= 2:
            text = re.sub(re.escape(n.strip()), "the Company", text, flags=re.IGNORECASE)
    return re.sub(r"\(?\b(stock code|股份代號)[:：]?\s*\d{1,5}\)?", "", text, flags=re.IGNORECASE)


def build_prompt(row: pd.Series, mode: str, excerpt: str | None = None) -> str:
    names = [row.get("stock_name_en"), row.get("stock_name_zh")]
    parts = []
    if mode in ("en", "both"):
        parts.append(f"English title: {_anonymise(row.get('title_en'), names)}")
        parts.append(f"English category: {row.get('category_en') or ''}")
    if mode in ("zh", "both"):
        parts.append(f"Chinese title: {_anonymise(row.get('title_zh'), names)}")
        parts.append(f"Chinese category: {row.get('category_zh') or ''}")
    if excerpt:
        parts.append(f"Document excerpt:\n{_anonymise(excerpt, names)[:2000]}")
    return "\n".join(parts)


def parse_response(text: str) -> dict:
    text = re.sub(r"```(json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in response: {text[:200]}")
    d = json.loads(m.group(0))
    et = d.get("event_type", "other")
    return {
        "event_type": et if et in EVENT_TYPES else "other",
        "sentiment": float(np.clip(float(d.get("sentiment", 0.0)), -1, 1)),
        "surprise": float(np.clip(float(d.get("surprise", 0.5)), 0, 1)),
        "confidence": float(np.clip(float(d.get("confidence", 0.5)), 0, 1)),
    }


def call_claude(prompt: str, model: str, api_key: str, max_retries: int = 5) -> dict:
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    body = {"model": model, "max_tokens": 200, "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(max_retries):
        r = requests.post(API_URL, headers=headers, json=body, timeout=60)
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
        return parse_response(text)
    raise RuntimeError("Anthropic API: retries exhausted")


class LLMCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]

    @staticmethod
    def key(news_id, mode: str, model: str, prompt: str) -> str:
        h = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        return f"{news_id}|{mode}|{model}|{h}"

    def put(self, key: str, value: dict) -> None:
        with self.lock:
            self.data[key] = value
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def llm_scores(ann: pd.DataFrame, mode: str, cfg: dict, cache_path: Path,
               excerpts: dict | None = None, offline: bool = False) -> pd.DataFrame:
    """Score each unique announcement once (news_id), then broadcast to every stock row.

    offline=True only reads the cache (no API calls); uncached rows come back NaN.
    """
    model = cfg["nlp"]["llm_model"]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not offline and not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY or run with offline=True")
    cache = LLMCache(cache_path)
    uniq = ann.drop_duplicates("news_id").set_index("news_id")
    jobs = {}
    for nid, row in uniq.iterrows():
        prompt = build_prompt(row, mode, (excerpts or {}).get(nid))
        jobs[nid] = (cache.key(nid, mode, model, prompt), prompt)

    todo = [(nid, k, p) for nid, (k, p) in jobs.items() if k not in cache.data]
    if todo and not offline:
        print(f"[llm:{mode}] scoring {len(todo)} announcements with {model} ({len(jobs) - len(todo)} cached)")
        with ThreadPoolExecutor(max_workers=cfg["nlp"].get("llm_max_workers", 8)) as ex:
            futs = {ex.submit(call_claude, p, model, api_key): k for _, k, p in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    cache.put(futs[fut], fut.result())
                except Exception as e:  # keep going; failures stay NaN and are retried next run
                    print(f"  failed: {e}")
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)}")

    recs = {nid: cache.data.get(k) for nid, (k, _) in jobs.items()}
    df = pd.DataFrame.from_dict({n: (v or {}) for n, v in recs.items()}, orient="index")
    for c in ["sentiment", "surprise", "confidence", "event_type"]:
        if c not in df:
            df[c] = np.nan
    out = ann[["news_id"]].join(df, on="news_id").drop(columns="news_id")
    return out.set_index(ann.index)
