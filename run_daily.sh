#!/bin/bash
REPO=$HOME/disposal/repo
LOG=$HOME/disposal/logs/daily_$(date +%Y%m%d_%H%M).log
mkdir -p $HOME/disposal/logs
{
  echo "=== $(date) 開始每日更新 ==="
  cd $REPO
  echo "=== git pull ==="
  git pull --rebase origin master
  echo "=== 更新訊號資料 ==="
  $HOME/disposal/venv/bin/python update_signals.py
  echo "=== git commit & push ==="
  git add data/signals.csv data/history.csv data/backtest_grid.csv data/meta.json data/signals_5min.csv data/history_5min.csv data/signals_tail20.csv data/history_tail20.csv
  if git diff --cached --quiet; then
    echo '無新資料，跳過 commit'
  else
    git commit -m "auto: $(date +%Y-%m-%d) 每日更新"
    git push origin master && echo 'push 成功'
  fi
  echo "=== $(date) 完成 ==="
} >> "$LOG" 2>&1
find $HOME/disposal/logs -name '*.log' -mtime +30 -delete
