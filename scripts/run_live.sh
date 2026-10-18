#!/usr/bin/env bash
# One command for every next step, run on your own machine.
#
#   scripts/run_live.sh smoke   # 1 month of data: checks the scrapers work (~5 min)
#   scripts/run_live.sh full    # full sample from config.yaml (announcements take a few hours)
#
# Optional: export ANTHROPIC_API_KEY=... first to add the LLM scorer.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"

pip install -q -e ".[live,dev]"
pytest -q

if [ "$MODE" = "smoke" ]; then
  START=$(date -d "45 days ago" +%F); END=$(date -d "15 days ago" +%F)
else
  START=$(python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['dates']['start'])")
  END=$(python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['dates']['end'])")
fi
SB_START=$(date -d "360 days ago" +%F); YESTERDAY=$(date -d yesterday +%F)

echo "== announcements $START -> $END";   hkalpha fetch-announcements --start "$START" --end "$END"
echo "== prices";                          hkalpha fetch-prices
echo "== southbound $SB_START -> $YESTERDAY"
if [ "$MODE" = "smoke" ]; then hkalpha fetch-southbound --start "$START" --end "$END"
else hkalpha fetch-southbound --start "$SB_START" --end "$YESTERDAY"; fi
echo "== short selling (today)";           hkalpha fetch-shortselling || echo "short-selling page not available yet today"

LLM_FLAG=""
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  echo "== LLM scoring";                   hkalpha score-llm
  LLM_FLAG="--use-llm"
fi

if [ "$MODE" = "smoke" ]; then
  python - <<'PY'
import pandas as pd
for f in ["announcements", "prices", "southbound"]:
    df = pd.read_parquet(f"data/raw/{f}.parquet")
    print(f"{f:14s} rows={len(df):>8,}  cols={list(df.columns)[:6]}")
a = pd.read_parquet("data/raw/announcements.parquet")
print("Chinese titles matched:", f"{a['title_zh'].notna().mean():.0%}")
print("Scrapers OK. Next: scripts/run_live.sh full")
PY
  exit 0
fi

echo "== research run";                    hkalpha run $LLM_FLAG
scripts/publish_results.sh live
echo "Done. Results: reports/live/summary.md  (figures copied to docs/figures for the README)"
