#!/usr/bin/env bash
# Copy the latest report figures into docs/figures so the README shows them.
# Usage: scripts/publish_results.sh [live|synthetic]   (default: live)
set -euo pipefail
src="reports/${1:-live}"
for f in car_paths equity ic_decay; do cp "$src/$f.png" docs/figures/; done
echo "Copied figures from $src. Now update the tables in README.md from $src/summary.md."
