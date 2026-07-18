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

# v1 研究候選門檻（見 HANDOFF_2026-07-18_CODEX_UPDATE.md），作為篩選面板的「一鍵套用」預設值
V1_PRESET = dict(
    potential_min=20.0,
    target_change_min=15.0,
    prior20_max=0,
    pre5_max=5.0,
    positive_only=True,
    upgrade_only=False,
    tradable_only=True,
    exclude_pending=True,
)


# 手動bump：資料schema變了但函式原始碼沒變時，強制cache失效（見「9_漲跌停事件研究」同款寫法）
DATA_VERSION = "v3-d0_d8_market_board"


@st.cache_data(ttl=3600)
def load_data(_version):
    ev = pd.read_csv(
        os.path.join(DATA_DIR, "events.csv"),
        parse_dates=["event_date"],
        dtype={"ticker": str},
        encoding="utf-8-sig",
    )
    with open(os.path.join(DATA_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    brokers_all = sorted(set(b for row in ev["brokers"].dropna() for b in row.split(";")))
    industries_all = sorted(ev["industry"].dropna().unique().tolist())
    return ev, meta, brokers_all, industries_all


try:
    events, meta, brokers_all, industries_all = load_data(DATA_VERSION)
except FileNotFoundError:
    st.error(
        "找不到資料檔，請先在 tw-quant-research 執行 "
        "`tools/_export_for_disposal_signals_site.py`"
    )
    st.stop()

st.title("🎯 法人目標價事件研究（D0-D8）")
st.caption(
    f"上市＋上櫃個股・券商評等／目標價調整事件・"
    f"樣本 {events['event_date'].min().date()} ~ {events['event_date'].max().date()}・"
    f"共 {len(events):,} 個 ticker-day・最後更新 {meta['generated_at'][:16].replace('T', ' ')}"
)

st.warning(
    "⚠️ **研究階段（testing），非交易訊號**。原始 OCR 整列正確率 293/301 = 97.34%，"
    "低於專案自訂 99% 品質門檻；已知的 ticker/公司撞號 bug 已修復，並經人工逐筆確認過歷史候選。"
    "下方所有門檻皆可自行調整組合，樣本數會即時顯示——**樣本數低於約 20-30 筆時統計非常不穩定，"
    "請勿只憑一個好看的小樣本就當成規則**。不做放空研究（放空假說已在擴充樣本上被推翻）。"
    "報酬未計入實際成交摩擦（如漲停鎖死、流動性），僅供歸因與紙上模擬追蹤用。"
)

if "filters_reset_token" not in st.session_state:
    st.session_state["filters_reset_token"] = 0


def _apply_preset(preset: dict):
    st.session_state["f_potential_min"] = preset["potential_min"]
    st.session_state["f_target_change_min"] = preset["target_change_min"]
    st.session_state["f_prior20_max"] = preset["prior20_max"]
    st.session_state["f_pre5_max"] = preset["pre5_max"]
    st.session_state["f_positive_only"] = preset["positive_only"]
    st.session_state["f_upgrade_only"] = preset["upgrade_only"]
    st.session_state["f_tradable_only"] = preset["tradable_only"]
    st.session_state["f_exclude_pending"] = preset["exclude_pending"]
    st.session_state["f_brokers"] = []
    st.session_state["f_industries"] = []


def _reset_all():
    _apply_preset(
        dict(
            potential_min=float(events["median_potential_pct"].min()),
            target_change_min=float(events["median_target_change_pct"].min()),
            prior20_max=int(events["prior_report_broker_events_20td"].max()),
            pre5_max=float(events["pre_return_5d_pct"].max()),
            positive_only=False,
            upgrade_only=False,
            tradable_only=False,
            exclude_pending=True,
        )
    )


with st.sidebar:
    st.subheader("自訂篩選")
    b1, b2 = st.columns(2)
    if b1.button("套用 v1 預設值 ⭐", width="stretch"):
        _apply_preset(V1_PRESET)
    if b2.button("重設全部", width="stretch"):
        _reset_all()

    dt_min, dt_max = events["event_date"].min().date(), events["event_date"].max().date()
    dt_range = st.slider("日期範圍", dt_min, dt_max, (dt_min, dt_max))

    st.caption("以下每個因子門檻都可獨立調整，右側表格/圖表即時反映組合結果")

    # 注意：widget 一旦有 key，就不能再同時傳 value=，否則 Streamlit 會噴例外。
    # 改用 setdefault 先確保 session_state 有初始值，widget 只靠 key 讀寫。
    pot_lo, pot_hi = float(events["median_potential_pct"].min()), float(events["median_potential_pct"].max())
    st.session_state.setdefault("f_potential_min", 0.0)
    potential_min = st.slider(
        "Potential 中位數 ≥ (%)", min_value=float(min(pot_lo, 0)), max_value=round(pot_hi, 0),
        step=1.0, key="f_potential_min",
    )

    tc_lo, tc_hi = float(events["median_target_change_pct"].min()), float(events["median_target_change_pct"].max())
    st.session_state.setdefault("f_target_change_min", tc_lo)
    target_change_min = st.slider(
        "目標價調整中位數 ≥ (%)", min_value=round(tc_lo, 0), max_value=round(tc_hi, 0),
        step=1.0, key="f_target_change_min",
    )

    p20_max_data = int(events["prior_report_broker_events_20td"].max())
    st.session_state.setdefault("f_prior20_max", p20_max_data)
    prior20_max = st.slider(
        "前20交易日內報告數 ≤", min_value=0, max_value=p20_max_data,
        step=1, key="f_prior20_max",
        help="0＝該事件是近期第一份法人報告（新資訊）",
    )

    pre5_lo, pre5_hi = float(events["pre_return_5d_pct"].min()), float(events["pre_return_5d_pct"].max())
    st.session_state.setdefault("f_pre5_max", pre5_hi)
    pre5_max = st.slider(
        "前5日累積報酬 < (%)", min_value=round(pre5_lo, 0), max_value=round(pre5_hi, 0),
        step=1.0, key="f_pre5_max",
        help="已大漲的股票排除，屬「減碼/避開」條件，不是放空訊號",
    )

    st.session_state.setdefault("f_positive_only", True)
    st.session_state.setdefault("f_upgrade_only", False)
    st.session_state.setdefault("f_tradable_only", True)
    st.session_state.setdefault("f_exclude_pending", True)
    positive_only = st.checkbox("只看偏多事件", key="f_positive_only")
    upgrade_only = st.checkbox("只看評等上調", key="f_upgrade_only")
    tradable_only = st.checkbox("只看開盤可成交", key="f_tradable_only")
    exclude_pending = st.checkbox("排除資料待確認列（data_pending）", key="f_exclude_pending")

    st.session_state.setdefault("f_brokers", [])
    st.session_state.setdefault("f_industries", [])
    broker_sel = st.multiselect("券商（符合任一即可）", brokers_all, key="f_brokers")
    industry_sel = st.multiselect("產業（符合任一即可）", industries_all, key="f_industries")

    market_boards = sorted(events["market_board"].dropna().unique().tolist())
    st.session_state.setdefault("f_market", [])
    market_sel = st.multiselect("市場別（上市sii／上櫃otc）", market_boards, key="f_market")

    st.divider()
    cost_pct = st.number_input("假設來回成本 (%)", min_value=0.0, max_value=3.0, value=0.60, step=0.05,
                                help="淨報酬＝毛報酬－此成本（單次買賣，不隨持有天數累加）；未計滑價")
    st.divider()
    max_h = max(int(c.replace("open_to_d", "").replace("_close_pct", ""))
                 for c in events.columns if c.startswith("open_to_d") and c.endswith("_close_pct"))
    st.session_state.setdefault("f_horizon", 0)
    horizon = st.select_slider(
        "持有天數（事件日開盤進場，第N日收盤出場）", options=list(range(0, max_h + 1)),
        key="f_horizon", format_func=lambda n: f"D{n}",
    )
    st.caption("D1-D8 為獨立計算的延伸（見 `_compute_multi_horizon_returns.py`），"
               "方法已與 pipeline 原生 D0/D1/D3/D5 對照驗證誤差為 0，但未經 OCR 複核以外的額外人工複核。")


def apply_filters(df):
    d = df[(df["event_date"].dt.date >= dt_range[0]) & (df["event_date"].dt.date <= dt_range[1])]
    d = d[d["median_potential_pct"] >= potential_min]
    d = d[d["median_target_change_pct"] >= target_change_min]
    d = d[d["prior_report_broker_events_20td"] <= prior20_max]
    d = d[d["pre_return_5d_pct"] < pre5_max]
    if positive_only:
        d = d[d["positive_event_flag"] == True]
    if upgrade_only:
        d = d[d["upgrade_event_flag"] == True]
    if tradable_only:
        d = d[d["main_open_entry_tradable_flag"] == True]
    if exclude_pending:
        d = d[d["data_pending_flag"] != True]
    if broker_sel:
        pattern = "|".join(pd.Series(broker_sel).str.replace(r"([.^$*+?{}\[\]\\|()])", r"\\\1", regex=True))
        d = d[d["brokers"].fillna("").str.contains(pattern, regex=True)]
    if industry_sel:
        d = d[d["industry"].isin(industry_sel)]
    if market_sel:
        d = d[d["market_board"].isin(market_sel)]
    return d


filtered = apply_filters(events).copy()

GROSS_COL = f"open_to_d{horizon}_close_pct"
NET_COL = f"net_d{horizon}_return_pct"
MKT_ADJ_COL = f"market_adjusted_open_to_d{horizon}_close_pct"
filtered[NET_COL] = filtered[GROSS_COL] - cost_pct

is_v1_default = all(
    st.session_state.get(k) == v
    for k, v in {
        "f_potential_min": V1_PRESET["potential_min"],
        "f_target_change_min": V1_PRESET["target_change_min"],
        "f_prior20_max": V1_PRESET["prior20_max"],
        "f_pre5_max": V1_PRESET["pre5_max"],
    }.items()
)

# ── 摘要指標 ─────────────────────────────────────────────
st.subheader(f"摘要：D{horizon}（依左側篩選條件即時計算）")
if is_v1_default:
    st.caption("目前套用的是 v1 研究預設值 ⭐")

scored = filtered[filtered[GROSS_COL].notna()]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("符合篩選筆數", f"{len(filtered):,}")
if len(scored):
    c2.metric(f"D{horizon} 毛報酬均值", f"{scored[GROSS_COL].mean():.2f}%")
    c3.metric(f"D{horizon} 淨報酬均值", f"{scored[NET_COL].mean():.2f}%")
    win = (scored[NET_COL] > 0).mean() * 100
    c4.metric("淨報酬勝率", f"{win:.1f}%")
    mkt_scored = filtered[filtered[MKT_ADJ_COL].notna()]
    c5.metric(
        "超額報酬均值（扣加權指數）",
        f"{mkt_scored[MKT_ADJ_COL].mean():.2f}%" if len(mkt_scored) else "—",
        help="個股毛報酬 － 同期加權指數(TAIEX)開盤到收盤報酬。上櫃股嚴格說該扣櫃買指數，"
             "但finlab目前查無櫃買開盤指數資料集，暫時統一用加權指數計算，上櫃股的超額報酬解讀請保守。",
    )
else:
    c2.metric(f"D{horizon} 毛報酬均值", "—")
    c3.metric(f"D{horizon} 淨報酬均值", "—")
    c4.metric("淨報酬勝率", "—")
    c5.metric("超額報酬均值（扣加權指數）", "—")

if 0 < len(scored) < 30:
    st.info(f"⚠️ 目前有結算報酬的樣本只有 {len(scored)} 筆，統計量不穩定，僅供參考。")

st.divider()

# ── 持有天數輪廓（D0-D8，不受右側單一持有天數選擇影響） ──
st.subheader("持有天數輪廓：報酬如何隨持有天數變化（依目前篩選條件）")
profile_rows = []
for n in range(0, max_h + 1):
    g, m = f"open_to_d{n}_close_pct", f"market_adjusted_open_to_d{n}_close_pct"
    gs = filtered[filtered[g].notna()][g]
    ms = filtered[filtered[m].notna()][m] if m in filtered.columns else pd.Series(dtype=float)
    profile_rows.append({
        "持有天數": n,
        "毛報酬均值%": gs.mean() if len(gs) else None,
        "淨報酬均值%": (gs.mean() - cost_pct) if len(gs) else None,
        "超額報酬均值%": ms.mean() if len(ms) else None,
        "樣本數": len(gs),
    })
profile = pd.DataFrame(profile_rows)
if profile["樣本數"].max() > 0:
    prof_long = profile.melt(
        id_vars=["持有天數", "樣本數"], value_vars=["毛報酬均值%", "淨報酬均值%", "超額報酬均值%"],
        var_name="指標", value_name="報酬%",
    )
    chart = alt.Chart(prof_long).mark_line(point=True).encode(
        x=alt.X("持有天數:O", title="持有天數 (D0-D8)"),
        y=alt.Y("報酬%:Q", title="平均報酬 (%)"),
        color=alt.Color("指標:N", title=None),
        tooltip=["持有天數:O", "指標:N", alt.Tooltip("報酬%:Q", format=".2f"), "樣本數:Q"],
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")
    st.dataframe(profile.round(2), width="stretch", hide_index=True)
    st.caption("各持有天數樣本數可能不同（越晚結算的天數，近期事件還沒到那天，會被排除為pending），比較長天期時留意樣本是否偏向較舊事件。")
else:
    st.info("目前篩選條件下沒有已結算的事件可畫輪廓圖。")

st.divider()

# ── 每日等權籃子淨報酬（目前篩選條件與所選持有天數） ──────
st.subheader(f"每日等權籃子淨報酬：D{horizon}（依目前篩選條件）")
basket_src = filtered[filtered[NET_COL].notna()]
if len(basket_src):
    daily = (
        basket_src.groupby("event_date")[NET_COL]
        .mean()
        .reset_index()
        .sort_values("event_date")
    )
    daily["cum_pct"] = daily[NET_COL].cumsum()
    base = alt.Chart(daily).encode(x=alt.X("event_date:T", title="事件日"))
    bar = base.mark_bar().encode(
        y=alt.Y(f"{NET_COL}:Q", title="當日籃子淨報酬 (%)"),
        color=alt.condition(
            f"datum.{NET_COL} >= 0", alt.value(UP_COLOR), alt.value(DOWN_COLOR)
        ),
        tooltip=["event_date:T", alt.Tooltip(f"{NET_COL}:Q", format=".2f")],
    )
    line = base.mark_line(color="#60a5fa").encode(
        y=alt.Y("cum_pct:Q", title="累積淨報酬 (%)"),
        tooltip=[alt.Tooltip("cum_pct:Q", format=".2f")],
    )
    st.altair_chart(
        alt.layer(bar, line).resolve_scale(y="independent").properties(height=320),
        width="stretch",
    )
    st.caption(f"共 {daily.shape[0]} 個有訊號的交易日；長條=當日等權淨報酬，藍線=累積淨報酬（右軸）。")
else:
    st.info("目前篩選條件下沒有已結算的事件可畫圖，試著放寬左側門檻。")

st.divider()

# ── 事件明細表 ───────────────────────────────────────────
st.subheader(f"事件明細：D{horizon}")

display_cols = {
    "event_date": "事件日",
    "ticker": "代號",
    "company": "公司",
    "industry": "產業",
    "market_board": "市場別",
    "brokers": "券商",
    "positive_event_flag": "偏多",
    "upgrade_event_flag": "評等上調",
    "median_potential_pct": "Potential中位數%",
    "median_target_change_pct": "目標價調整中位數%",
    "prior_report_broker_events_20td": "前20日報告數",
    "pre_return_5d_pct": "前5日報酬%",
    "event_open": "開盤",
    "event_close": "收盤",
    GROSS_COL: f"D{horizon}毛報酬%",
    NET_COL: f"D{horizon}淨報酬%",
    MKT_ADJ_COL: f"D{horizon}超額報酬%",
    "v1_candidate": "v1候選",
}
show = filtered[list(display_cols.keys())].rename(columns=display_cols).copy()
show["事件日"] = show["事件日"].dt.date
numeric_cols = ["Potential中位數%", "目標價調整中位數%", "前5日報酬%", "開盤", "收盤",
                f"D{horizon}毛報酬%", f"D{horizon}淨報酬%", f"D{horizon}超額報酬%"]
for c in numeric_cols:
    show[c] = show[c].round(2)
show = show.sort_values("事件日", ascending=False)

st.dataframe(
    show,
    width="stretch",
    hide_index=True,
    height=520,
)

csv_bytes = show.to_csv(index=False).encode("utf-8-sig")
st.download_button("下載目前篩選結果 (CSV)", csv_bytes, file_name="broker_target_filtered.csv", mime="text/csv")

st.caption(
    "「v1候選」欄位標記的是研究預設門檻是否全部通過（不受左側篩選影響），方便對照；"
    "「前20日報告數」=0 代表該事件是該股票近期第一份法人報告（新資訊）。"
)
