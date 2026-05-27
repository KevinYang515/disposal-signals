#!/bin/bash
# 期貨日盤收盤後自動執行：更新訊號 → commit → push
# cron: 0 6 * * 1-5  /bin/bash ~/disposal-signals/auto_update.sh >> ~/disposal-signals/auto_update.log 2>&1

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$LOG_TIME] 開始更新..."
cd "$REPO_DIR"

# 更新訊號資料
python3 update_signals.py

# 若有變更才 commit & push
if ! git diff --quiet data/; then
    DATA_DATE=$(python3 -c "import json; m=json.load(open('data/meta.json')); print(m['data_date'])")
    git add data/
    git commit -m "data: 自動更新訊號 ${DATA_DATE}"
    git push
    echo "[$LOG_TIME] push 完成 ✅"
else
    echo "[$LOG_TIME] 資料無變更，跳過 push"
fi
