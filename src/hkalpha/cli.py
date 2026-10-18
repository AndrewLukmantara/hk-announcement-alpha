from __future__ import annotations

import argparse

import pandas as pd

from .config import data_paths, load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="hkalpha")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("fetch-announcements"); a.add_argument("--start"); a.add_argument("--end")
    sub.add_parser("fetch-prices")
    s = sub.add_parser("fetch-southbound"); s.add_argument("--start"); s.add_argument("--end")
    sub.add_parser("fetch-shortselling")
    sc = sub.add_parser("score-llm", help="call Claude for every announcement (cached)")
    sc.add_argument("--synthetic", action="store_true")
    lb = sub.add_parser("label-sample"); lb.add_argument("--n", type=int, default=400)
    ev = sub.add_parser("eval-labels"); ev.add_argument("--labels", required=True)
    r = sub.add_parser("run"); r.add_argument("--synthetic", action="store_true")
    r.add_argument("--regenerate", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    P = data_paths(cfg, getattr(args, "synthetic", False))
    start, end = cfg["dates"]["start"], cfg["dates"]["end"]

    if args.cmd == "fetch-announcements":
        from .data.hkexnews import fetch_range
        fetch_range(args.start or start, args.end or end, P["announcements"])
    elif args.cmd == "fetch-prices":
        from .data.prices import fetch_prices
        codes = pd.read_csv(P["universe"], dtype=str)["stock_code"].str.zfill(5).tolist()
        fetch_prices(codes, start, end, P["prices"], cfg["market"]["index_ticker"], P["index"])
    elif args.cmd == "fetch-southbound":
        from .data.southbound import fetch_range
        fetch_range(args.start or start, args.end or end, P["southbound"])
    elif args.cmd == "fetch-shortselling":
        from .data.short_selling import fetch_today
        fetch_today(P["short_selling"])
    elif args.cmd == "score-llm":
        from .nlp.llm import llm_scores
        ann = pd.read_parquet(P["announcements"])
        ann = ann[ann["stock_code"].astype(str).str.zfill(5).isin(
            pd.read_csv(P["universe"], dtype=str)["stock_code"].str.zfill(5))] if not args.synthetic else ann
        for mode in cfg["nlp"]["llm_lang_modes"]:
            llm_scores(ann, mode, cfg, P["llm_cache"], offline=False)
    elif args.cmd == "label-sample":
        from .nlp.evaluate import export_label_sample
        path = P["announcements"].parent / "label_sample.csv"
        export_label_sample(pd.read_parquet(P["announcements"]), path, args.n)
        print(f"wrote {path}: fill label_sentiment / label_event_type, then run eval-labels")
    elif args.cmd == "eval-labels":
        from .nlp.evaluate import evaluate_scorers
        labels = pd.read_csv(args.labels, dtype={"news_id": str})
        scores = pd.read_parquet(P["scores"])
        cols = [c for c in scores.columns if c.startswith(("kw_", "llm_", "category")) and not c.endswith("surprise")]
        print(evaluate_scorers(labels, scores, cols).to_string(index=False))
    elif args.cmd == "run":
        from .pipeline import run
        run(cfg, synthetic=args.synthetic, regenerate=args.regenerate)


if __name__ == "__main__":
    main()
