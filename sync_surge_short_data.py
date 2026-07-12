# -*- coding: utf-8 -*-
r"""
每日收盤後執行：把VM上的Q8/Q10今日選股+模擬交易記錄拉到網站data資料夾，
拉完後手動 git add/commit/push 讓streamlit.app自動重新部署。
"""
import subprocess
from pathlib import Path

SSH_KEY = r"C:\Users\USER\.ssh\google_compute_engine"
VM = "kevin850515123456789@35.212.129.240"
DATA_DIR = Path(__file__).parent / "data"

files = [
    ("~/stock/live/q_filters_today.json", DATA_DIR / "急拉高_今日選股.json"),
    ("~/stock/live/logs/q_surge_trades_Q8.csv", DATA_DIR / "q_surge_trades_Q8.csv"),
    ("~/stock/live/logs/q_surge_trades_Q10.csv", DATA_DIR / "q_surge_trades_Q10.csv"),
]

for remote, local in files:
    print(f"拉取 {remote} -> {local}", flush=True)
    r = subprocess.run(["scp", "-i", SSH_KEY, f"{VM}:{remote}", str(local)])
    if r.returncode != 0:
        print(f"  失敗或檔案尚不存在 (可能今日還沒有交易記錄), 略過", flush=True)

print("\n完成。請檢查 data/ 底下檔案後手動 git add/commit/push 讓網站更新。", flush=True)
