# -*- coding: utf-8 -*-
"""Publish today's five Stage1 candidate snapshots into disposal-signals.

This deliberately makes a local Git commit only.  The coordinator remains
responsible for pushing that commit to GitHub / Streamlit Community Cloud.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


SOURCE_DIR = Path(r"E:\stock\live_stage1_data")
REPOSITORY = Path(r"E:\stock_recovery\github_repos\disposal-signals")
DESTINATION_DIR = REPOSITORY / "data" / "live_candidates"
FILENAMES = (
    "citycenter_candidates_today.json",
    "unicenter_candidates_today.json",
    "strategyh_candidates_today.json",
    "flipbranch_candidates_today.json",
    "fubon_candidates_today.json",
)
TAIPEI_TZ = timezone(timedelta(hours=8))


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY), *args],
        check=True,
        text=True,
        encoding="utf-8",
    )


def validate_snapshot(path: Path, expected_trade_date: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"無法讀取候選檔 {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise RuntimeError(f"候選檔 {path.name} 不符合預期 JSON 結構。")
    if payload.get("trade_date") != expected_trade_date:
        raise RuntimeError(
            f"候選檔 {path.name} 的 trade_date={payload.get('trade_date')!r}，"
            f"不是今天 {expected_trade_date}；拒絕發布舊資料。"
        )


def main() -> int:
    if not (REPOSITORY / ".git").is_dir():
        raise RuntimeError(f"找不到 disposal-signals Git repo: {REPOSITORY}")

    expected_trade_date = datetime.now(TAIPEI_TZ).date().isoformat()
    sources = [SOURCE_DIR / filename for filename in FILENAMES]
    for source in sources:
        if not source.is_file():
            raise RuntimeError(f"找不到 Stage1 候選檔: {source}")
        validate_snapshot(source, expected_trade_date)

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    destinations = []
    for source in sources:
        destination = DESTINATION_DIR / source.name
        shutil.copy2(source, destination)
        destinations.append(destination)
        print(f"已發布 {source.name}")

    relative_paths = [str(path.relative_to(REPOSITORY)) for path in destinations]
    run_git("add", "--", *relative_paths)
    changed = subprocess.run(
        ["git", "-C", str(REPOSITORY), "diff", "--cached", "--quiet", "--", *relative_paths],
        text=True,
        encoding="utf-8",
    )
    if changed.returncode == 0:
        print("今日候選資料沒有變更，不建立重複提交。")
        return 0
    if changed.returncode != 1:
        raise RuntimeError("無法檢查候選資料的 Git 暫存差異。")

    run_git("commit", "-m", f"更新今日候選資料（{expected_trade_date}）")
    print("今日候選資料已建立本機 Git 提交；未推送。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"發布失敗：{error}", file=sys.stderr)
        sys.exit(1)
