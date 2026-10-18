"""End-to-end research run: score -> time-align -> event study -> signals -> walk-forward backtest -> report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Costs, build_weights, rebalance_every, run_backtest
from .config import data_paths, reports_dir
from .event_study import (abnormal_returns, car_columns, interaction_regression, sentiment_bucket,
                          summarise)
from .features import build_panel, lookup, short_ratio_z, southbound_flow_z
from .metrics import ic_series, ic_stats, perf_stats
from .nlp.scoring import score_all
from .report import df_to_md, plot_car_paths, plot_equity, plot_ic_decay
from .signal import decayed, event_matrix, flow_tilt
from .timing import assign_entry_dates


def _read(p):
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_inputs(cfg: dict, synthetic: bool, regenerate: bool = False) -> dict[str, pd.DataFrame]:
    P = data_paths(cfg, synthetic)
    if synthetic and (regenerate or not P["prices"].exists()):
        from .data.synthetic import generate
        for k, df in generate(cfg).items():
            df.to_parquet(P[k], index=False)
    data = {k: _read(P[k]) for k in ["announcements", "prices", "index", "southbound", "short_selling"]}
    for k in ["announcements", "prices", "index"]:
        if data[k].empty:
            raise FileNotFoundError(f"missing {P[k]}: run the fetch commands first (see README)")
    for df in data.values():
        if "stock_code" in df:
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(5)
    return data


def run(cfg: dict, synthetic: bool = False, regenerate: bool = False, llm_offline: bool = True) -> dict:
    P, out = data_paths(cfg, synthetic), reports_dir(cfg, synthetic)
    d = load_inputs(cfg, synthetic, regenerate)
    start, end, oos = (pd.Timestamp(cfg["dates"][k]) for k in ("start", "end", "oos_start"))
    horizons = cfg["event_study"]["horizons"]

    # ---- panel & features
    panel = build_panel(d["prices"], d["index"])
    ret, dates, stocks = panel["ret"], panel["ret"].index, panel["ret"].columns
    eligible = (panel["adv20"].shift(1) >= cfg["universe"]["min_adv_hkd"]) & panel["close"].notna()
    flow_z = southbound_flow_z(d["southbound"], dates, stocks, cfg["signal"]["flow_window"])
    short_z = short_ratio_z(d["short_selling"], panel["turnover"])
    log_adv = np.log(panel["adv20"].shift(1))

    # ---- events: align to tradable close, score
    ev = assign_entry_dates(d["announcements"], dates, cfg["market"]["entry_lag_minutes"])
    ev = ev[ev["stock_code"].isin(stocks) & ev["entry_date"].between(start, end)].reset_index(drop=True)
    scores = score_all(ev, cfg, P["llm_cache"], llm_offline=llm_offline)
    ev = pd.concat([ev, scores], axis=1)
    ev[["news_id", "stock_code", "entry_date"] + list(scores.columns)].to_parquet(P["scores"], index=False)
    for name, panel_df in [("flow_z", flow_z), ("short_z", short_z), ("log_adv", log_adv)]:
        ev[name] = lookup(panel_df, ev["entry_date"], ev["stock_code"])
    primary = cfg["signal"]["primary"]
    if primary not in ev:
        raise KeyError(f"signal.primary={primary} not scored; enable nlp.use_llm or pick a kw_* variant")

    # ---- event study
    es = cfg["event_study"]
    ar = abnormal_returns(ret, panel["mkt_ret"], ev, tuple(es["est_window"]), max(horizons), es["min_est_obs"])
    evs = ev.join(car_columns(ar, horizons), how="inner")
    evs["bucket"] = sentiment_bucket(evs[primary])
    by_bucket = summarise(evs, "bucket", horizons)
    by_type = summarise(evs[evs["event_type"].ne("routine")], "event_type", horizons)
    by_session = ev["session"].value_counts().rename_axis("session").reset_index(name="n_events")
    reg = interaction_regression(evs, f"drift_{horizons[1] if len(horizons) > 1 else horizons[0]}", primary)
    lang_reg = None
    if {"kw_en", "kw_zh"} <= set(evs):
        # horse race: conditional on the English score, does the Chinese score predict drift?
        sub = evs[(evs["kw_en"] != 0) | (evs["kw_zh"] != 0)].copy()
        if len(sub) > 50 and (sub["kw_en"] - sub["kw_zh"]).abs().sum() > 0:
            import statsmodels.formula.api as smf
            sub["_c"] = pd.factorize(sub["entry_date"])[0]
            lang_reg = smf.ols(f"drift_{horizons[1]} ~ kw_en + kw_zh", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["_c"]})
    plot_car_paths(ar, evs["bucket"], out / "car_paths.png")

    # ---- signals: choose half-life in-sample, evaluate out-of-sample
    hls, h_sel = cfg["signal"]["half_lives"], cfg["signal"]["selection_horizon"]
    is_mask, oos_mask = dates < oos, dates >= oos
    variants = [c for c in ["kw_both", "kw_en", "kw_zh", "category", "llm_both", "llm_en", "llm_zh"]
                if c in ev and ev[c].notna().any()]
    raw = {v: event_matrix(ev, v, dates, stocks) for v in variants}

    def best_hl(x):
        stats = {hl: ic_stats(ic_series(decayed(x, hl), ret, eligible, h_sel)[is_mask])["ic_mean"] for hl in hls}
        stats = {k: v for k, v in stats.items() if np.isfinite(v)}
        return max(stats, key=stats.get) if stats else hls[0]

    signals, chosen = {}, {}
    for v in variants:
        chosen[v] = best_hl(raw[v])
        signals[v] = decayed(raw[v], chosen[v])
    signals[f"{primary}_x_flow"] = flow_tilt(signals[primary], flow_z, cfg["signal"]["flow_tilt"])
    chosen[f"{primary}_x_flow"] = chosen[primary]
    if flow_z.notna().any().any():
        signals["flow_only"] = flow_z.fillna(0.0)
        chosen["flow_only"] = "-"

    c = cfg["backtest"]
    costs = Costs(**c["costs"])
    rows, curves, decay = [], {}, {}
    k = c.get("rebalance_every_slow", 5)
    hl_slow = min([h for h in hls if h >= k] or [max(hls)])  # decay matched to the holding period
    slow = flow_tilt(decayed(raw[primary], hl_slow), flow_z, cfg["signal"]["flow_tilt"])
    runs = [(n, s, 1) for n, s in signals.items()] + [(f"{primary}_x_flow_reb{k}", slow, k)]
    chosen[f"{primary}_x_flow_reb{k}"] = hl_slow
    for name, s, every in runs:
        w = rebalance_every(build_weights(s, eligible, c["mode"], c["quantile"], c["min_names"]), every)
        bt = run_backtest(w, ret, costs)
        ic = ic_series(s, ret, eligible, h_sel)
        r_is, r_oos = perf_stats(bt["net"][is_mask]), perf_stats(bt["net"][oos_mask], bt["turnover"][oos_mask])
        rows.append({"signal": name, "half_life": chosen[name],
                     "IS_ic": ic_stats(ic[is_mask])["ic_mean"], "OOS_ic": ic_stats(ic[oos_mask])["ic_mean"],
                     "OOS_ic_t": ic_stats(ic[oos_mask])["ic_t"], "IS_sharpe": r_is["sharpe"],
                     "OOS_sharpe_gross": perf_stats(bt["gross"][oos_mask])["sharpe"],
                     "OOS_sharpe_net": r_oos["sharpe"], "OOS_ann_ret": r_oos["ann_ret"],
                     "OOS_max_dd": r_oos["max_dd"], "OOS_turnover": r_oos.get("avg_daily_turnover", np.nan)})
        if name in (primary, f"{primary}_x_flow", f"{primary}_x_flow_reb{k}", "flow_only"):
            curves[name] = bt["net"]
        if every == 1 and name in (primary, f"{primary}_x_flow", "category"):
            decay[name] = pd.Series({h: ic_stats(ic_series(s, ret, eligible, h)[oos_mask])["ic_mean"]
                                     for h in [1, 2, 3, 5, 10, 15, 20]})
    ablation = pd.DataFrame(rows)
    plot_equity(curves, oos, out / "equity.png")
    plot_ic_decay(decay, out / "ic_decay.png")

    # ---- report
    md = [f"# HK announcement alpha: {'SYNTHETIC DATA (pipeline check only)' if synthetic else 'live data'}\n",
          f"Sample {start.date()} to {end.date()}, out-of-sample from {oos.date()}. "
          f"{len(ev):,} announcement-stock events, {ev['stock_code'].nunique()} stocks, "
          f"primary scorer `{primary}`. Costs per side: {costs.per_side * 1e4:.1f} bps "
          f"(stamp duty {costs.stamp_duty * 1e4:.0f} bps).\n",
          "## Publication timing\n", df_to_md(by_session),
          "## Event study by sentiment bucket\n",
          "car_0 is the reaction from the prior close to the entry close (not tradable); "
          "drift_h is the abnormal return over the next h days (tradable).\n",
          df_to_md(by_bucket, "{:.2f}"), "![CAR paths](car_paths.png)\n",
          "## Event study by event type (non-routine)\n", df_to_md(by_type, "{:.2f}"),
          "## Interaction regression (clustered by entry date)\n",
          "```\n" + (reg.summary().as_text() if reg is not None else "not enough data") + "\n```\n",
          "## Does the Chinese title add information beyond the English one?\n",
          "Horse race of the English-only and Chinese-only keyword scores; a significant kw_zh "
          "coefficient means the Chinese version carries information the English one lacks.\n",
          "```\n" + (lang_reg.summary().tables[1].as_text() if lang_reg is not None else
                      "no language disagreement in sample") + "\n```\n",
          "## Signal ablation (half-life chosen on in-sample IC)\n", df_to_md(ablation),
          "![Equity](equity.png)\n\n![IC decay](ic_decay.png)\n"]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    ablation.to_csv(out / "ablation.csv", index=False)
    by_type.to_csv(out / "event_study_by_type.csv", index=False)
    print(f"report written to {out / 'summary.md'}")
    return {"events": evs, "ablation": ablation, "by_bucket": by_bucket, "by_type": by_type,
            "regression": reg, "out_dir": out}
