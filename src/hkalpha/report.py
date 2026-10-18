from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def plot_car_paths(ar: pd.DataFrame, buckets: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for b in ["positive", "neutral", "negative"]:
        ids = buckets.index[buckets == b].intersection(ar.index)
        if len(ids) > 10:
            ax.plot(ar.columns, ar.loc[ids].mean().cumsum() * 1e4, label=f"{b} (n={len(ids)})", lw=2)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("trading days from entry (day 0 = reaction, not tradable)")
    ax.set_ylabel("mean CAR (bps)")
    ax.set_title("Cumulative abnormal return by announcement sentiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ic_decay(decay: dict[str, pd.Series], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, s in decay.items():
        ax.plot(s.index, s.values, marker="o", label=name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("forward horizon (trading days)")
    ax.set_ylabel("mean rank IC (out-of-sample)")
    ax.set_title("IC decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equity(curves: dict[str, pd.Series], oos_start, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in curves.items():
        ax.plot((1 + r.fillna(0)).cumprod(), label=name, lw=1.5)
    ax.axvline(pd.Timestamp(oos_start), color="black", ls="--", lw=1)
    ax.text(pd.Timestamp(oos_start), ax.get_ylim()[1], " out-of-sample", va="top")
    ax.set_ylabel("growth of 1 (net of costs)")
    ax.set_title("Strategy equity curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
