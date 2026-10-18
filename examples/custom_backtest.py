"""Backtest your own signal with the library API.

Run from the repo root after `hkalpha run --synthetic` (or on live data without --synthetic):
    python examples/custom_backtest.py
"""
import pandas as pd

from hkalpha.backtest import Costs, build_weights, rebalance_every, run_backtest
from hkalpha.config import data_paths, load_config
from hkalpha.features import build_panel, southbound_flow_z
from hkalpha.metrics import ic_series, ic_stats, perf_stats
from hkalpha.pipeline import load_inputs
from hkalpha.signal import decayed, event_matrix, flow_tilt
from hkalpha.timing import assign_entry_dates

SYNTHETIC = True
cfg = load_config("config.yaml")
d = load_inputs(cfg, synthetic=SYNTHETIC)
panel = build_panel(d["prices"], d["index"])
ret, dates, stocks = panel["ret"], panel["ret"].index, panel["ret"].columns
eligible = (panel["adv20"].shift(1) >= cfg["universe"]["min_adv_hkd"]) & panel["close"].notna()

# scored events written by `hkalpha run`
scores = pd.read_parquet(data_paths(cfg, SYNTHETIC)["scores"])

# --- your idea: only trade earnings + placing events, 5-day half-life, tilted by Southbound flow
events = scores[scores["event_type"].isin(["earnings_positive", "earnings_negative", "placing"])]
signal = decayed(event_matrix(events, "kw_both", dates, stocks), half_life=5)
signal = flow_tilt(signal, southbound_flow_z(d["southbound"], dates, stocks), k=0.5)

weights = rebalance_every(build_weights(signal, eligible, mode="long_short", quantile=0.3, min_names=3), 5)
bt = run_backtest(weights, ret, Costs(**cfg["backtest"]["costs"]))

oos = dates >= pd.Timestamp(cfg["dates"]["oos_start"])
print("OOS IC (5d):", {k: round(float(v), 3) for k, v in ic_stats(ic_series(signal, ret, eligible, 5)[oos]).items()})
print("OOS net:    ", {k: round(float(v), 3) for k, v in perf_stats(bt["net"][oos], bt["turnover"][oos]).items()})
