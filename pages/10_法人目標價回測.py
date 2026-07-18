"""
法人目標價事件研究（D0）— 券商評等/目標價調整事件，開盤買、收盤賣。
資料由 D:\\stock\\tw-quant-research 產生（OMU 專案），透過
tools/_export_for_disposal_signals_site.py 匯出快照，非即時同步。
狀態：研究/紙上模擬（testing），非正式交易訊號。
"""

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="法人目標價事件研究",
    page_icon="🎯",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "broker_target")

UP_COLOR = "#f87171"
DOWN_COLOR = "#4ade80"


@st.cache_data(ttl=3600)
def load_data():
    ev = pd.read_csv(
        os.path.join(DATA_DIR, "events.csv"),
        parse_dates=["event_date"],
        dtype={"ticker": str},
        encoding="utf-8-sig",
    )
    with open(os.path.join(DATA_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return ev, meta


try:
    events, meta = load_data()
except FileNotFoundError:
    st.error(
        "找不到資料檔，請先在 tw-quant-research 執行 "
        "`tools/_export_for_disposal_signals_site.py`"
    )
    st.stop()

st.title("🎯 法人目標價事件研究（D0）")
st.caption(
    f"上市＋上櫃個股・券商評等／目標價調整事件・"
    f"樣本 {events['event_date'].min().date()} ~ {events['event_date'].max().date()}・"
    f"共 {len(events):,} 個 ticker-day・最後更新 {meta['generated_at'][:16].replace('T', ' ')}"
)

st.warning(
    "⚠️ **研究階段（testing），非交易訊號**。原始 OCR 整列正確率 293/301 = 97.34%，"
    "低於專案自訂 99% 品質門檻；已知的 ticker/公司撞號 bug 已修復，並經人工逐筆確認過歷史候選。"
    "策略門檻（下方 v1 候選標記）為回測研究結果，樣本仍偏小（探索期+驗證期共約 50-90 筆），"
    "**不構成買賣建議，不做放空研究**（放空假說已在擴充樣本上被推翻）。"
    "報酬未計入實際成交摩擦（如漲停鎖死、流動性），僅供歸因與紙上模擬追蹤用。"
    f"\n\n{meta['notes']}"
)

# ── 側欄篩選 ─────────────────────────────────────────────
with st.sidebar:
    st.subheader("篩選")
    dt_min, dt_max = events["event_date"].min().date(), events["event_date"].max().date()
    dt_range = st.slider("日期範圍", dt_min, dt_max, (dt_min, dt_max))
    only_v1 = st.checkbox("只看 v1 研究候選 ⭐", value=False)
    only_positive = st.checkbox("只看偏多事件", value=True)
    exclude_pending = st.checkbox("排除資料待確認列（data_pending）", value=True)
    industries = sorted(events["industry"].dropna().unique().tolist())
    ind_sel = st.multiselect("產業", industries, default=[])
    st.divider()
    st.caption(
        "v1 候選條件：偏多事件、Potential 中位數 ≥20%、前20交易日內無其他報告（新資訊）、"
        "目標價調整中位數 ≥15%、前5日累積報酬 <5%、開盤可成交。"
        "來源：`HANDOFF_2026-07-18_CODEX_UPDATE.md`。"
    )
    st.caption("D0淨報酬＝開盤買、收盤賣，扣 0.60% 假設來回成本（未計滑價）。")


def apply_filters(df):
    d = df[
        (df["event_date"].dt.date >= dt_range[0]) & (df["event_date"].dt.date <= dt_range[1])
    ]
    if only_v1:
        d = d[d["v1_candidate"] == True]
    if only_positive:
        d = d[d["positive_event_flag"] == True]
    if exclude_pending:
        d = d[d["data_pending_flag"] != True]
    if ind_sel:
        d = d[d["industry"].isin(ind_sel)]
    return d


filtered = apply_filters(events)

# ── 摘要指標 ─────────────────────────────────────────────
st.subheader("摘要（依目前篩選條件即時計算）")

v1_all = events[(events["v1_candidate"] == True) & (events["data_pending_flag"] != True)]
scored = filtered[filtered["open_to_d0_close_pct"].notna()]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("符合篩選筆數", f"{len(filtered):,}")
c2.metric("其中 v1 候選（全樣本）", f"{len(v1_all):,}")
if len(scored):
    c3.metric("篩選後 D0 毛報酬均值", f"{scored['open_to_d0_close_pct'].mean():.2f}%")
    c4.metric("篩選後 D0 淨報酬均值", f"{scored['net_d0_return_pct'].mean():.2f}%")
    win = (scored["net_d0_return_pct"] > 0).mean() * 100
    c5.metric("淨報酬勝率", f"{win:.1f}%")
else:
    c3.metric("篩選後 D0 毛報酬均值", "—")
    c4.metric("篩選後 D0 淨報酬均值", "—")
    c5.metric("淨報酬勝率", "—")

st.divider()

# ── 每日等權籃子淨報酬（v1候選） ────────────────────────
st.subheader("v1 候選：每日等權籃子淨報酬")
basket_src = events[
    (events["v1_candidate"] == True)
    & (events["data_pending_flag"] != True)
    & (events["net_d0_return_pct"].notna())
]
if len(basket_src):
    daily = (
        basket_src.groupby("event_date")["net_d0_return_pct"]
        .mean()
        .reset_index()
        .sort_values("event_date")
    )
    daily["cum_pct"] = daily["net_d0_return_pct"].cumsum()
    base = alt.Chart(daily).encode(x=alt.X("event_date:T", title="事件日"))
    bar = base.mark_bar().encode(
        y=alt.Y("net_d0_return_pct:Q", title="當日籃子淨報酬 (%)"),
        color=alt.condition(
            "datum.net_d0_return_pct >= 0", alt.value(UP_COLOR), alt.value(DOWN_COLOR)
        ),
        tooltip=["event_date:T", alt.Tooltip("net_d0_return_pct:Q", format=".2f")],
    )
    line = base.mark_line(color="#60a5fa").encode(
        y=alt.Y("cum_pct:Q", title="累積淨報酬 (%)"),
        tooltip=[alt.Tooltip("cum_pct:Q", format=".2f")],
    )
    st.altair_chart(
        alt.layer(bar, line).resolve_scale(y="independent").properties(height=320),
        use_container_width=True,
    )
    st.caption(f"共 {daily.shape[0]} 個有 v1 候選訊號的交易日；長條=當日等權淨報酬，藍線=累積淨報酬（右軸）。")
else:
    st.info("目前沒有已結算的 v1 候選事件可畫圖。")

st.divider()

# ── 事件明細表 ───────────────────────────────────────────
st.subheader("事件明細")

display_cols = {
    "event_date": "事件日",
    "ticker": "代號",
    "company": "公司",
    "industry": "產業",
    "brokers": "券商",
    "positive_event_flag": "偏多",
    "upgrade_event_flag": "評等上調",
    "median_potential_pct": "Potential中位數%",
    "median_target_change_pct": "目標價調整中位數%",
    "prior_report_broker_events_20td": "前20日報告數",
    "pre_return_5d_pct": "前5日報酬%",
    "event_open": "開盤",
    "event_close": "收盤",
    "open_to_d0_close_pct": "D0毛報酬%",
    "net_d0_return_pct": "D0淨報酬%",
    "v1_candidate": "v1候選",
}
show = filtered[list(display_cols.keys())].rename(columns=display_cols).copy()
show["事件日"] = show["事件日"].dt.date
for c in ["Potential中位數%", "目標價調整中位數%", "前5日報酬%", "開盤", "收盤", "D0毛報酬%", "D0淨報酬%"]:
    show[c] = show[c].round(2)

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    height=520,
)
st.caption(
    "「v1候選」為研究標記（研究/紙上模擬用途），不是即時下單訊號；"
    "「前20日報告數」=0 代表該事件是該股票近期第一份法人報告（新資訊）。"
)
