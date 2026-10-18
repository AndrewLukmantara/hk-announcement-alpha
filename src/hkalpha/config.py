from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = path.parent
    return cfg


def data_paths(cfg: dict, synthetic: bool = False) -> dict[str, Path]:
    root = cfg["_root"] / cfg["project"]["data_dir"]
    base = root / ("synthetic" if synthetic else "raw")
    base.mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    return {
        "announcements": base / "announcements.parquet",
        "prices": base / "prices.parquet",
        "index": base / "index.parquet",
        "southbound": base / "southbound.parquet",
        "short_selling": base / "short_selling.parquet",
        "scores": base / "scores.parquet",
        "llm_cache": root / "cache" / "llm_scores.jsonl",
        "universe": cfg["_root"] / cfg["universe"]["tickers_file"],
    }


def reports_dir(cfg: dict, synthetic: bool = False) -> Path:
    d = cfg["_root"] / cfg["project"]["reports_dir"] / ("synthetic" if synthetic else "live")
    d.mkdir(parents=True, exist_ok=True)
    return d
