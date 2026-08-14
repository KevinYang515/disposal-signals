# -*- coding: utf-8 -*-
"""Display the Stage1 candidate snapshots published to this repository."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(page_title="今日候選", page_icon="📋", layout="wide")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "live_candidates"
DATA_SCHEMA_VERSION = "2026-08-14-live-candidates-v2"

STRATEGIES = (
    {
        "label": "凱基－城中（City-GA）",
        "filename": "citycenter_candidates_today.json",
        "columns": (
            ("code", "股票代號"),
            ("prev_close", "前一日收盤"),
            ("net_amt_wan", "淨買超（萬元）"),
            ("cz_influence_pct", "城中影響度（%）"),
        ),
    },
    {
        "label": "統一城中（UniCenter）",
        "filename": "unicenter_candidates_today.json",
        "columns": (
            ("code", "股票代號"),
            ("prev_close", "前一日收盤"),
            ("net_amt_wan", "淨買超（萬元）"),
            ("d0_turnover", "D0 成交金額"),
            ("uc_influence_pct", "統一城中影響度（%）"),
        ),
    },
    {
        "label": "StrategyH",
        "filename": "strategyh_candidates_today.json",
        "columns": (
            ("code", "股票代號"),
            ("prev_close", "前一日收盤"),
            ("style_buyers", "風格買方數"),
            ("aggregate_ratio_pct", "彙總比率（%）"),
        ),
    },
    {
        "label": "FlipBranch",
        "filename": "flipbranch_candidates_today.json",
        "columns": (
            ("code", "股票代號"),
            ("prev_close", "前一日收盤"),
            ("ranking_value", "排名值"),
            ("ranking_field", "排名欄位"),
            ("dominant_broker", "主導分點"),
        ),
    },
    {
        "label": "富邦（bare-parent）",
        "caption": "僅供研究資訊參考；富邦不是實盤／模擬交易機器人，候選出現不代表任何 bot 會下單。",
        "filename": "fubon_candidates_today.json",
        "columns": (
            ("code", "股票代號"),
            ("prev_close", "前一日收盤"),
            ("net_amt_wan", "淨買超（萬元）"),
            ("influence_pct", "富邦影響度（%）"),
        ),
    },
)


@st.cache_data(ttl=3600)
def load_candidate_payload(filename: str, _cache_bust: str = DATA_SCHEMA_VERSION) -> dict[str, Any]:
    """Load one repository-published Stage1 snapshot, with a clear schema check."""
    path = DATA_DIR / filename
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{filename} 的內容不是 JSON 物件。")
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"{filename} 的 candidates 必須是清單。")
    return payload


def status_text(payload: dict[str, Any]) -> tuple[bool, str]:
    """Return the plain-language enabled state and any known daily-gate reason."""
    active = bool(payload.get("active", False))
    if active:
        return True, "啟用"

    before_gate = payload.get("candidate_count_before_daily_gate")
    minimum = payload.get("min_candidate_count")
    if before_gate is not None and minimum is not None:
        try:
            gate_failed = float(before_gate) < float(minimum)
        except (TypeError, ValueError):
            gate_failed = False
        if gate_failed:
            return False, f"未啟用：每日候選數 {before_gate}，未達最低 {minimum} 檔"
    if payload.get("candidates"):
        return False, "未啟用：策略當日未啟用"
    return False, "未啟用：篩選後無候選"


def candidate_table(payload: dict[str, Any], columns: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """Create a stable, strategy-specific display table from a candidate list."""
    candidates = payload.get("candidates", [])
    if not candidates:
        return pd.DataFrame()

    frame = pd.DataFrame(candidates)
    ordered_keys = [key for key, _ in columns if key in frame.columns]
    extra_keys = [key for key in frame.columns if key not in ordered_keys]
    frame = frame[ordered_keys + extra_keys]
    labels = {key: label for key, label in columns}
    return frame.rename(columns=labels)


def float_column_formats(table: pd.DataFrame) -> dict[str, str]:
    """Match the event-detail tables' two-decimal display convention."""
    return {
        column: "{:.2f}"
        for column in table.columns
        if pd.api.types.is_float_dtype(table[column])
    }


st.title("📋 今日候選")
st.caption("四個 live Stage1 策略與一個研究資訊快照的下一個交易日候選清單。")

payloads: dict[str, dict[str, Any]] = {}
load_errors: dict[str, tuple[str, str]] = {}
for strategy in STRATEGIES:
    filename = strategy["filename"]
    try:
        payloads[filename] = load_candidate_payload(filename)
    except FileNotFoundError:
        load_errors[filename] = ("warning", "尚未發布今日候選資料。")
    except (json.JSONDecodeError, ValueError) as error:
        load_errors[filename] = ("error", f"候選資料格式無法讀取：{error}")

active_count = sum(bool(payload.get("active", False)) for payload in payloads.values())
candidate_count = sum(len(payload.get("candidates", [])) for payload in payloads.values())
summary_active, summary_candidates = st.columns(2)
summary_active.metric("今日啟用策略", f"{active_count} / {len(STRATEGIES)}")
summary_candidates.metric("今日候選總數", f"{candidate_count} 檔")
if load_errors:
    st.caption(f"摘要目前僅納入成功讀取的 {len(payloads)} / {len(STRATEGIES)} 份資料快照。")

st.divider()

for strategy in STRATEGIES:
    filename = strategy["filename"]
    payload = payloads.get(filename)
    load_error = load_errors.get(filename)

    with st.container(border=True):
        heading, badge = st.columns([6, 2])
        with heading:
            st.subheader(strategy["label"])
            if caption := strategy.get("caption"):
                st.caption(caption)
        with badge:
            if load_error:
                if load_error[0] == "warning":
                    st.warning("資料未發布")
                else:
                    st.error("資料格式錯誤")
            elif bool(payload.get("active", False)):
                st.success("啟用中")
            else:
                st.info("未啟用")

        if load_error:
            if load_error[0] == "warning":
                st.warning(load_error[1])
            else:
                st.error(load_error[1])
            continue

        active, state = status_text(payload)
        trade_date = payload.get("trade_date", "-")
        signal_date = payload.get("signal_date", "-")
        st.caption(f"交易日：{trade_date}　｜　訊號日：{signal_date}　｜　狀態說明：{state}")
        st.caption(f"上次更新：{payload.get('generated_at', '-')}")

        table = candidate_table(payload, strategy["columns"])
        if table.empty:
            st.info("今日無候選")
        else:
            st.dataframe(
                table.style.format(float_column_formats(table), na_rep="-"),
                width="stretch",
                hide_index=True,
            )

st.caption("此頁反映 Stage1 產生當下的候選清單；候選股仍可能因 live engine 的進場反彈或借券可用性檢查而不實際進場。")
