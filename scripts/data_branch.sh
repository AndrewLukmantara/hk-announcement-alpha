#!/usr/bin/env bash
# Keep collected data on an orphan `data` branch so main stays code-only.
#   scripts/data_branch.sh restore      # copy data branch -> data/raw, data/cache
#   scripts/data_branch.sh save "msg"   # copy data/raw, data/cache -> data branch, commit, push
set -euo pipefail
STORE=/tmp/hkalpha-data
mkdir -p data/raw data/cache
if [ ! -d "$STORE" ]; then
  if git fetch -q origin data 2>/dev/null; then
    git worktree add -q "$STORE" origin/data --detach
  else
    git worktree add -q --detach "$STORE"
    (cd "$STORE" && git checkout -q --orphan data && git rm -rqf . && rm -rf ./*)
  fi
fi
case "$1" in
  restore)
    cp -r "$STORE"/raw/. data/raw/ 2>/dev/null || true
    cp -r "$STORE"/cache/. data/cache/ 2>/dev/null || true
    echo "restored: $(ls data/raw | tr '\n' ' ')" ;;
  save)
    mkdir -p "$STORE/raw" "$STORE/cache"
    cp data/raw/*.parquet "$STORE/raw/" 2>/dev/null || true
    cp data/cache/* "$STORE/cache/" 2>/dev/null || true
    cd "$STORE"
    git -c user.name=github-actions -c user.email=github-actions@users.noreply.github.com add -A
    if git diff --cached --quiet; then echo "no new data"; exit 0; fi
    git -c user.name=github-actions -c user.email=github-actions@users.noreply.github.com commit -qm "${2:-data update}"
    git push -q origin HEAD:data
    echo "saved to data branch" ;;
esac
