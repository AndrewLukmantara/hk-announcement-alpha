"""Daily prices via yfinance (free; survivorship-biased: delisted names are missing)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_yahoo(code: str) -> str:
    return f"{int(code):04d}.HK"


def fetch_prices(codes: list[str], start: str, end: str, out_path: Path, index_ticker: str,
                 index_path: Path, batch: int = 50) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        tick = {to_yahoo(c): c for c in chunk}
        raw = yf.download(list(tick), start=start, end=end, auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
        for t, code in tick.items():
            if t not in raw.columns.get_level_values(0):
                continue
            d = raw[t].dropna(how="all").reset_index()
            d.columns = [str(c).lower() for c in d.columns]
            d["stock_code"] = code
            frames.append(d[["date", "stock_code", "open", "high", "low", "close", "volume"]])
        print(f"prices: {min(i + batch, len(codes))}/{len(codes)}")
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    prices.to_parquet(out_path, index=False)

    idx = yf.download(index_ticker, start=start, end=end, auto_adjust=True, progress=False)
    idx = idx["Close"].squeeze().rename("close").reset_index()
    idx.columns = ["date", "close"]
    idx["date"] = pd.to_datetime(idx["date"]).dt.tz_localize(None)
    idx.to_parquet(index_path, index=False)
    return prices
