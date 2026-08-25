#!/bin/bash
# 急拉高空方 Q8/Q10 → disposal-signals 網站 自動同步
# 用法: sync_surge_short_to_site.sh [morning|afternoon]  (只影響commit訊息, 邏輯相同)
MODE="${1:-manual}"
REPO=$HOME/disposal/repo
LIVE=$HOME/stock/live
LOG=$HOME/disposal/logs/surge_sync_$(date +%Y%m%d_%H%M).log
mkdir -p "$HOME/disposal/logs"

{
  echo "=== $(date) 開始同步(${MODE}) ==="
  cd "$REPO"
  echo "=== git pull ==="
  git pull --no-rebase --no-edit origin master

  mkdir -p "$REPO/data"
  cp "$LIVE/q_filters_today.json" "$REPO/data/急拉高_今日選股.json" 2>/dev/null || echo "q_filters_today.json 不存在, 略過"
  cp "$LIVE/logs/q_surge_trades_Q8.csv" "$REPO/data/q_surge_trades_Q8.csv" 2>/dev/null || echo "Q8交易CSV尚不存在, 略過"
  cp "$LIVE/logs/q_surge_trades_Q10.csv" "$REPO/data/q_surge_trades_Q10.csv" 2>/dev/null || echo "Q10交易CSV尚不存在, 略過"
  cp "$LIVE/logs/q_surge_trades_Q10PB.csv" "$REPO/data/q_surge_trades_Q10PB.csv" 2>/dev/null || echo "Q10PB交易CSV尚不存在, 略過"

  git add data/急拉高_今日選股.json data/q_surge_trades_Q8.csv data/q_surge_trades_Q10.csv data/q_surge_trades_Q10PB.csv 2>/dev/null
  if git diff --cached --quiet; then
    echo "無變更, 跳過commit"
  else
    git commit -m "auto: 急拉高空方 ${MODE} $(date +%Y-%m-%d\ %H:%M)"
    git push origin master && echo "push 成功"
  fi
  echo "=== $(date) 完成 ==="
} >> "$LOG" 2>&1
find "$HOME/disposal/logs" -name 'surge_sync_*.log' -mtime +30 -delete
