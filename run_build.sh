#!/bin/bash
LOG=$HOME/disposal/logs/build_$(date +%Y%m%d).log
mkdir -p $HOME/disposal/logs
{
  echo "=== $(date) 開始重建處置資料 ==="
  cd $HOME/disposal
  $HOME/disposal/venv/bin/python build_disposal_data.py
  echo "=== $(date) 完成 ==="
} >> "$LOG" 2>&1
