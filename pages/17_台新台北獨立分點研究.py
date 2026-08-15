# -*- coding: utf-8 -*-
r"""台新-台北 strict-lock/liquid 隔日沖放空事件監看頁。

狀態：研究／個人監看（testing），非驗證完成的正式交易訊號。
資料與結論來源：new_branch_discovery_20260813.md。
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="台新-台北獨立分點研究", page_icon="🔬", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EVENTS_FILE = os.path.join(DATA_DIR, "taishin_taipei_events.csv")


def as_bool(series: pd.Series) -> pd.Series:
    """Read current bool CSVs and older string-shaped snapshots safely."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _parse_spark(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def synced_pct_input(label, default, key):
    """Slider + number_input synced via session_state for mobile-friendly entry."""
    slider_key, num_key = f"{key}_slider", f"{key}_num"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default

    def _from_slider():
        st.session_state[num_key] = st.session_state[slider_key]

    def _from_num():
        st.session_state[slider_key] = st.session_state[num_key]

    st.slider(label, 0.0, 10.0, step=0.5, key=slider_key, on_change=_from_slider)
    if num_key not in st.session_state:
        st.session_state[num_key] = st.session_state[slider_key]
    st.number_input("或直接輸入數字（%，適合手機操作）", 0.0, 10.0, step=0.5, key=num_key, on_change=_from_num)
    return st.session_state[slider_key]


@st.cache_data(ttl=3600)
def load_events(_cache_bust: str = "2026-08-15-offline-branch-context-v1"):
    df = pd.read_csv(EVENTS_FILE, parse_dates=["d0", "d1"], dtype={"code": str})
    df["code"] = df["code"].str.zfill(4)
    names_fp = os.path.join(DATA_DIR, "stock_names.csv")
    if os.path.exists(names_fp):
        names = pd.read_csv(names_fp, dtype={"code": str})
        names["code"] = names["code"].str.zfill(4)
        df = df.merge(names, on="code", how="left")
        df["name"] = df["name"].fillna("")
    else:
        df["name"] = ""
    for column, default in [
        ("d0_disposal", False), ("d1_disposal", False), ("d1_disposal_type", ""),
        ("day_trade_short_suspended_d1", False), ("d1_intraday_close", ""), ("has_intraday", False),
        ("entry_is_ceiling", False), ("entry_frozen", False), ("entry_disposal", False),
        ("entry_disposal_type", ""), ("day_trade_short_suspended_entry", False),
    ]:
        if column not in df:
            df[column] = default
    for column, fallback in [
        ("entry_day", "d1"), ("entry_open", "d1_open"), ("entry_high", "d1_high"),
        ("entry_low", "d1_low"), ("entry_close", "d1_close"),
    ]:
        if column not in df:
            df[column] = df[fallback]
    df["entry_day"] = pd.to_datetime(df["entry_day"], errors="coerce").fillna(df["d1"])
    for column in ["d0_locked", "d1_frozen", "censored", "success", "d0_disposal", "d1_disposal",
                   "day_trade_short_suspended_d1", "has_intraday", "entry_is_ceiling", "entry_frozen",
                   "entry_disposal", "day_trade_short_suspended_entry"]:
        df[column] = as_bool(df[column])
    df["d1_intraday_spark"] = df["d1_intraday_close"].apply(_parse_spark)
    branch_context_defaults = {
        "city_ga_net_amt_wan": 0.0, "city_ga_influence_pct": 0.0,
        "unicenter_city_net_amt_wan": 0.0, "unicenter_city_influence_pct": 0.0,
        "fubon_net_amt_wan": 0.0, "fubon_influence_pct": 0.0,
        "taishin_taipei_net_amt_wan": 0.0, "taishin_taipei_influence_pct": 0.0,
        "d0_top3_net_buy_branches": "", "top3_available": False,
        "v1_candidate_d1": False, "other_branch_active_count": 0,
    }
    for column, default in branch_context_defaults.items():
        if column not in df:
            df[column] = default
    df["d0_top3_net_buy_branches"] = df["d0_top3_net_buy_branches"].fillna("")
    for column in ["top3_available", "v1_candidate_d1"]:
        df[column] = as_bool(df[column])
    df["other_branch_active_count"] = df["other_branch_active_count"].fillna(0).astype(int)
    return df


def gap_bucket(gap):
    if gap < -5:
        return "<-5%"
    if gap < -2:
        return "-5~-2%"
    if gap < 0:
        return "-2~0%"
    if gap < 1:
        return "0~1%"
    if gap < 3:
        return "1~3%"
    if gap < 5:
        return "3~5%"
    if gap < 7:
        return "5~7%"
    if gap < 9:
        return "7~9%"
    if gap < 9.5:
        return "9~9.5%"
    return "≥9.5%"


def portfolio_metrics(frame: pd.DataFrame, return_column: str):
    """Equal-weight same actual-entry positions, then compound entry dates."""
    daily = frame.dropna(subset=[return_column]).groupby("entry_day")[return_column].mean().sort_index()
    if daily.empty:
        return np.nan, np.nan, np.nan, daily
    wealth = (1.0 + daily / 100.0).cumprod()
    period_return = (wealth.iloc[-1] - 1.0) * 100.0
    drawdown = (wealth / wealth.cummax() - 1.0) * 100.0
    mdd = drawdown.min()
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252) if len(daily) > 1 and daily.std(ddof=1) else np.nan
    return period_return, sharpe, mdd, daily


st.title("🔬 台新-台北獨立分點｜隔日沖放空研究監看")
st.warning(
    "⚠️ **狀態：研究／個人監看（testing），非正式交易訊號。** 台新-台北可用歷史僅 "
    "2026-04-07～2026-08-07、約 4 個月，遠短於城中 GA／富邦的多年資料。以相同嚴格鎖漲停、"
    "高流動性條件的對照測試，放空報酬只高 **+0.20 個百分點**，p=0.627，沒有顯著優勢；"
    "把可用時間切成前後半，優勢反而從前半的 +0.47 個百分點弱化成後半的 -0.16 個百分點，"
    "沒有隨資料累積而確認。D1 日報酬又已與城中 GA／富邦重疊（相關係數 0.34／0.48，"
    "各 60 個重疊日期），不是獨立訊號。**本頁只供 Kevin 個人監看／查閱，不是已驗證或推薦策略；"
    "城中 GA 與富邦頁的驗證地位不適用於本頁。**完整依據見 `new_branch_discovery_20260813.md`。"
)

events = load_events()
events["年份"] = events["d0"].dt.year.astype(str)
events["gap_bucket"] = events["gap_pct"].apply(gap_bucket)
events["streak_capped"] = events["lock_streak"].clip(upper=4)
with np.errstate(divide="ignore", invalid="ignore"):
    events["entry_gap_pct"] = (
        (events["entry_open"].to_numpy(dtype=float) - events["d0_close"].to_numpy(dtype=float))
        / events["d0_close"].to_numpy(dtype=float) * 100.0
    )
ceiling_entries = int(events["entry_is_ceiling"].sum())
if ceiling_entries:
    st.caption(
        f"⚠️ 進場天花板價修正：{ceiling_entries}/{len(events)} 筆 D1 開盤等於依 D0 收盤計算的漲停價，"
        "已改在第一個開盤脫離當日漲停、可交易的日期進場；下方 KPI 與報酬均已使用修正後的實際進場日。"
    )
st.caption(
    "D0 母體：只取精確分點名稱「台新-台北」的 strict-lock/liquid 事件，250 筆；"
    "不包含裸名稱「台新」或「台新證券」。本 CSV 的 strict-lock/liquid D0 日期為 "
    f"{events['d0'].min().date()}～{events['d0'].max().date()}；分點原始可用歷史到 2026-08-07。"
)

# 預設不套用 City-GA 的 gap／市值／大盤因子：那些規則沒有在台新-台北這個未驗證分點上測過。
sel_years = st.multiselect(
    "年份", sorted(events["年份"].unique().tolist(), reverse=True),
    default=sorted(events["年份"].unique().tolist(), reverse=True),
)
st.caption("⚠️ 處置分盤期間、當沖限制的事件一律硬性排除（無法切換）——理由見下方展開。")
with st.expander("📖 硬性排除規則說明", expanded=False):
    st.caption(
        "⚠️ 實際進場日處於處置分盤集合競價期間的事件一律排除，不提供切換選項：分盤集合競價沒有連續撮合，"
        "無法按本頁的開盤放空／收盤回補方式執行。"
    )
    st.caption(
        "⚠️ 實際進場日命中本地已知「停止先賣後買」限制的事件一律排除，不提供切換選項：這是當沖放空可能無法合法執行的問題，"
        "不是依報酬好壞挑選。其餘借券／融券額度與費率資料仍不足，未被視為可交易保證。"
    )

st.markdown("#### 🎚️ 進場緩衝設定")
entry_buffer_pct = synced_pct_input(
    "進場緩衝：實際進場當天開盤漲幅（相對 D0 收盤）需低於多少%才進場（10=不啟用，因單日漲停就是10%，不會有更高的）",
    10.0,
    "entry_gap_buffer_17",
)
events["entry_buffer_excluded"] = events["entry_gap_pct"] >= entry_buffer_pct
buffer_excluded_count = int(events["entry_buffer_excluded"].sum())
st.caption(
    f"目前設定下排除 {buffer_excluded_count}/{len(events)} 筆事件"
    f"（{buffer_excluded_count / len(events):.1%}）；用修正後實際進場當天的開盤價相對 D0 收盤的漲幅%計算，"
    "不符合者只排除、不再尋找下一個進場日。"
)

exclude_target_v1 = st.checkbox(
    "排除當日同時符合法人目標價V1候選的事件",
    value=False,
    key="exclude_target_v1_17",
)
st.caption(
    "探索性工具：前一輪全母體測試（`target_price_interaction_20260813.md`）發現，"
    "法人目標價 V1 候選重疊對城中GA或富邦的放空報酬都**沒有** holdout-confirmed 的驗證效果；"
    "此切換是依 Kevin 要求提供查閱，不是已證實的篩選條件。D1-exact 對齊。"
)

base_mask = (
    events["年份"].isin(sel_years)
    & ~events["entry_disposal"]
    & ~events["day_trade_short_suspended_entry"]
    & ~events["entry_buffer_excluded"]
)
if exclude_target_v1:
    mask = base_mask & ~events["v1_candidate_d1"]
else:
    mask = base_mask
    # Default-off must retain the exact pre-toggle population.
    if not events.index[mask].equals(events.index[base_mask]):
        raise RuntimeError("V1 排除切換預設關閉時改變了台新-台北事件母體。")
view = events.loc[mask].copy()
show_ceiling_only = st.checkbox("只看曾因漲停天花板而延後進場的事件", value=False)
if show_ceiling_only:
    view = view.loc[view["entry_is_ceiling"]].copy()

st.divider()
st.markdown("#### 🎚️ 停損／停利情境設定")
st.caption(
    "設定「實際進場日股價比 D0 收盤漲多少%停損」「實際進場日股價比 D0 收盤跌多少%停利」，以實際進場日最高／最低價估算是否觸及；"
    "同日兩者都觸及時保守採停損優先。實際進場日整日無成交、無法回補的事件視為截尾，不納入 KPI。"
    "**這裡只提供情境查閱，台新-台北沒有任何已驗證的停損／停利建議；預設 0%／0% 是單純持有到收盤。**"
)
col_s1, col_s2 = st.columns(2)
with col_s1:
    stop_pct = synced_pct_input("停損：實際進場日股價比 D0 收盤漲多少% 出場（0=不停損）", 0.0, "stop_17")
with col_s2:
    tp_pct = synced_pct_input("停利：實際進場日股價比 D0 收盤跌多少% 出場（0=不停利）", 0.0, "tp_17")

entry = view["entry_open"].to_numpy(dtype=float)
d0_close = view["d0_close"].to_numpy(dtype=float)
entry_high = view["entry_high"].to_numpy(dtype=float)
entry_low = view["entry_low"].to_numpy(dtype=float)
entry_close = view["entry_close"].to_numpy(dtype=float)
base_ret = view["short_ret_open_to_close_pct"].to_numpy(dtype=float)
with np.errstate(invalid="ignore"):
    stop_price = d0_close * (1.0 + stop_pct / 100.0)
    tp_price = d0_close * (1.0 - tp_pct / 100.0)
    hit_stop = (stop_pct > 0) & (entry_high >= stop_price)
    hit_tp = (tp_pct > 0) & (entry_low <= tp_price)
    exit_price = np.where(hit_tp, tp_price, entry_close)
    exit_price = np.where(hit_stop, stop_price, exit_price)
    sim_ret = (entry - exit_price) / entry * 100.0
    sim_ret = np.where(view["entry_frozen"].to_numpy(dtype=bool), base_ret, sim_ret)
    sim_ret = np.where(view["censored"].to_numpy(dtype=bool) | ~np.isfinite(entry), np.nan, sim_ret)
view["sim_ret"] = sim_ret

st.divider()
settled = view.loc[~view["censored"]].copy()
rets = settled["sim_ret"].dropna()
if rets.empty:
    st.warning("目前篩選條件下無已結算資料")
else:
    period_return, sharpe, mdd, daily = portfolio_metrics(settled, "sim_ret")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("已結算筆數", f"{len(rets)} 筆", delta=f"{len(view)} 筆通過篩選", delta_color="off")
    c2.metric("期間報酬", f"{period_return:+.2f}%")
    c3.metric("勝率", f"{(rets > 0).mean() * 100:.2f}%")
    c4.metric("日組合 Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "-")
    c5.metric("最大回撤", f"{mdd:.2f}%" if pd.notna(mdd) else "-")
    st.caption(
        "期間報酬／Sharpe／最大回撤：同一實際進場日的多筆事件先等權，再依實際進場日複利；"
        "不含交易成本、滑價或實際借券額度。這些是目前短歷史的描述數字，不是推薦依據。"
    )
    if stop_pct > 0 or tp_pct > 0:
        st.caption(f"⚙️ 以上 KPI 已套用停損 {stop_pct:.1f}%／停利 {tp_pct:.1f}% 的情境設定。")
    if entry_buffer_pct < 10.0:
        st.caption(f"⚙️ 以上 KPI 已套用進場緩衝：開盤漲幅≥{entry_buffer_pct:.1f}%者已完全排除。")

st.divider()
gap_bins = ["<-5%", "-5~-2%", "-2~0%", "0~1%", "1~3%", "3~5%", "5~7%", "7~9%", "9~9.5%", "≥9.5%"]
with st.expander("📊 依 D1 開盤跳空區間的描述性統計", expanded=True):
    rows = []
    for gap in gap_bins:
        sub = view.loc[(view["gap_bucket"] == gap) & ~view["censored"]]
        if sub.empty:
            continue
        _, sub_sharpe, sub_mdd, _ = portfolio_metrics(sub.assign(sim_ret=sub["short_ret_open_to_close_pct"]), "sim_ret")
        rows.append({
            "跳空區間": gap, "筆數": len(sub),
            "勝率(%)": (sub["short_ret_open_to_close_pct"] > 0).mean() * 100.0,
            "平均放空報酬(%)": sub["short_ret_open_to_close_pct"].mean(),
            "日組合Sharpe": sub_sharpe, "最大回撤(%)": sub_mdd,
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index("跳空區間").style.format(
                {"勝率(%)": "{:.2f}", "平均放空報酬(%)": "{:+.2f}", "日組合Sharpe": "{:.2f}",
                 "最大回撤(%)": "{:.2f}"}, na_rep="-"),
            use_container_width=True,
        )
    st.caption("僅供描述性查閱；沒有將任何跳空區間設為預設濾網或推薦條件。")

with st.expander(
    f"🎚️ 停損／停利效果對照（目前設定：停損{stop_pct:.1f}%／停利{tp_pct:.1f}%，與上方KPI同一組設定）"
):
    sim_settled = view.loc[~view["censored"]].copy()
    base_rets = sim_settled["short_ret_open_to_close_pct"].dropna()
    scenario_rets = sim_settled["sim_ret"].dropna()
    if base_rets.empty or scenario_rets.empty:
        st.warning("目前篩選條件下無足夠已結算資料可比較。")
    else:
        base_period, base_sharpe, base_mdd, _ = portfolio_metrics(
            sim_settled.assign(_scenario=sim_settled["short_ret_open_to_close_pct"]), "_scenario"
        )
        scenario_period, scenario_sharpe, scenario_mdd, _ = portfolio_metrics(sim_settled, "sim_ret")
        comparison = pd.DataFrame(
            {
                "原本（持有到實際進場日收盤）": [
                    f"{base_period:+.2f}%",
                    f"{(base_rets > 0).mean() * 100:.2f}%",
                    f"{base_rets.mean():+.2f}%",
                    f"{base_sharpe:.2f}" if pd.notna(base_sharpe) else "-",
                    f"{base_mdd:.2f}%" if pd.notna(base_mdd) else "-",
                    f"{base_rets.min():+.2f}%",
                ],
                "套用停損／停利後（=上方KPI）": [
                    f"{scenario_period:+.2f}%",
                    f"{(scenario_rets > 0).mean() * 100:.2f}%",
                    f"{scenario_rets.mean():+.2f}%",
                    f"{scenario_sharpe:.2f}" if pd.notna(scenario_sharpe) else "-",
                    f"{scenario_mdd:.2f}%" if pd.notna(scenario_mdd) else "-",
                    f"{scenario_rets.min():+.2f}%",
                ],
            },
            index=["期間報酬", "勝率", "平均放空報酬", "日組合Sharpe", "最大回撤", "最慘單筆"],
        )
        st.dataframe(comparison, use_container_width=True)
        st.caption(
            f"已結算 {len(sim_settled)} 筆；此表只比較目前滑桿的情境，不代表台新-台北已有可驗證的停損／停利規則。"
        )

with st.expander("📊 依連續鎖漲停天數的描述性統計（短歷史，請勿視為濾網）"):
    rows = []
    for streak in [1, 2, 3, 4]:
        sub = view.loc[(view["streak_capped"] == streak) & ~view["censored"]]
        if sub.empty:
            continue
        rows.append({
            "連鎖天數": f"{streak}{'+' if streak == 4 else ''}",
            "筆數": len(sub),
            "勝率(%)": (sub["short_ret_open_to_close_pct"] > 0).mean() * 100.0,
            "平均放空報酬(%)": sub["short_ret_open_to_close_pct"].mean(),
            "最慘單筆(%)": sub["short_ret_open_to_close_pct"].min(),
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index("連鎖天數").style.format(
                {"勝率(%)": "{:.2f}", "平均放空報酬(%)": "{:+.2f}", "最慘單筆(%)": "{:+.2f}"},
                na_rep="-",
            ),
            use_container_width=True,
        )
    st.caption("僅呈現目前約四個月資料的分布；沒有從中推導預設排除條件，也不移植其他分點的連鎖規則。")

with st.expander("📅 可用期間的月份分布（僅約 4 個月，請勿過度解讀）"):
    monthly = settled.assign(月份=settled["d0"].dt.strftime("%Y-%m")).groupby("月份")["short_ret_open_to_close_pct"].agg(
        筆數="size", 勝率=lambda x: (x > 0).mean() * 100.0, 平均放空報酬="mean"
    )
    st.dataframe(monthly.style.format({"勝率": "{:.2f}", "平均放空報酬": "{:+.2f}"}, na_rep="-"), use_container_width=True)

st.subheader(f"📋 完整逐筆歷史紀錄（{len(view)} 筆，已套用硬排除、年份與進場緩衝篩選）")
st.caption(
    "「其他已發現分點」均為同一個 D0、同一檔股票的本機分點成交快取批次計算；"
    "淨買超為萬元、影響力依 D0 成交金額計算。沒有該分點活動時顯示 0。"
)
show_cols = [
    "d0", "code", "name", "market", "d1", "entry_day", "net_amt_wan", "influence_pct", "gap_pct", "lock_streak",
    "city_ga_net_amt_wan", "city_ga_influence_pct", "unicenter_city_net_amt_wan", "unicenter_city_influence_pct",
    "fubon_net_amt_wan", "fubon_influence_pct", "d0_top3_net_buy_branches", "v1_candidate_d1",
    "d1_open", "d1_high", "d1_low", "d1_close", "entry_open", "entry_close", "entry_gap_pct", "d1_frozen", "censored", "short_ret_open_to_close_pct", "sim_ret",
    "short_mae_pct",
]
show = view[show_cols].sort_values(["d0", "code"], ascending=False).copy()
show["d0"] = show["d0"].dt.strftime("%Y-%m-%d")
show["d1"] = show["d1"].dt.strftime("%Y-%m-%d")
show["entry_day"] = show["entry_day"].dt.strftime("%Y-%m-%d")
show.columns = [
    "D0訊號日", "代號", "名稱", "市場", "D1原始日", "實際進場日", "買超金額(萬)", "影響力%", "跳空%", "連鎖天數",
    "城中GA淨買超(萬)", "城中GA影響力%", "統一城中淨買超(萬)", "統一城中影響力%",
    "富邦證券淨買超(萬)", "富邦證券影響力%", "D0前三大買超分點", "D1同時為V1候選",
    "D1開盤", "D1最高", "D1最低", "D1收盤", "實際開盤", "實際收盤", "進場開盤漲幅%", "D1鎖死", "截尾", "原始放空報酬%", "情境放空報酬%", "最大不利波動%",
]

# Only show real cached minute bars.  If none are available, omit the column rather than fabricate a D1 path.
has_sparkline = bool(view["has_intraday"].any())
if has_sparkline:
    show["D1走勢"] = view.loc[show.index, "d1_intraday_spark"].tolist()


def color_return(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if value > 5:
        return "color: #26c281; font-weight: 700"
    if value > 0:
        return "color: #2ecc71"
    if value > -5:
        return "color: #e67e22"
    return "color: #e74c3c; font-weight: 700"


column_config = {
    "D0前三大買超分點": st.column_config.TextColumn(
        "D0前三大買超分點", width="large",
        help="同一 D0／同一股票的全部正淨買分點，依淨買金額排序；不是只限本頁四個訊號分點。",
    ),
}
if has_sparkline:
    column_config["D1走勢"] = st.column_config.LineChartColumn(
        "D1走勢（分K收盤）", width="medium", help="真實歷史分鐘 K 收盤價；缺資料事件留空，非模擬。"
    )
st.dataframe(
    show.style.map(color_return, subset=["情境放空報酬%"]).format(
        {"買超金額(萬)": "{:,.2f}", "影響力%": "{:.2f}", "跳空%": "{:+.2f}",
         "城中GA淨買超(萬)": "{:,.2f}", "城中GA影響力%": "{:+.2f}",
         "統一城中淨買超(萬)": "{:,.2f}", "統一城中影響力%": "{:+.2f}",
         "富邦證券淨買超(萬)": "{:,.2f}", "富邦證券影響力%": "{:+.2f}",
         "D1開盤": "{:.2f}", "D1最高": "{:.2f}", "D1最低": "{:.2f}", "D1收盤": "{:.2f}",
         "實際開盤": "{:.2f}", "實際收盤": "{:.2f}", "進場開盤漲幅%": "{:+.2f}", "原始放空報酬%": "{:+.2f}", "情境放空報酬%": "{:+.2f}",
         "最大不利波動%": "{:.2f}"}, na_rep="-"),
    use_container_width=True, height=520, column_config=column_config,
)
if has_sparkline:
    coverage = view["has_intraday"].mean() * 100.0
    st.caption(f"D1走勢為真實歷史分鐘 K 收盤價，資料覆蓋率約 {coverage:.2f}%；沒有抓到分鐘 K 的事件留空。")
else:
    st.caption("本批事件沒有可用的真實歷史分鐘 K，因此已省略 D1走勢欄位，未以日線或推估資料替代。")

download = show.drop(columns=["D1走勢"], errors="ignore")
st.download_button(
    "📥 下載此表 CSV", download.to_csv(index=False, encoding="utf-8-sig"),
    "taishin_taipei_events_filtered.csv", "text/csv", key="dl_taishin_taipei_events",
)

st.divider()
st.markdown("**累積報酬走勢**（同一實際進場日多筆等權、依實際進場日排序；套用目前停損／停利情境）")
if not rets.empty:
    _, _, _, daily = portfolio_metrics(settled, "sim_ret")
    if len(daily) >= 2:
        cumulative = ((daily / 100.0 + 1.0).cumprod() - 1.0) * 100.0
        st.line_chart(pd.DataFrame({"累積報酬(%)": cumulative.values}), height=250)
    else:
        st.caption("目前篩選條件下已結算日期不足，無法繪製累積報酬走勢。")

st.markdown("---")
st.caption(
    "研究結論與母體稽核請見 `E:\\stock\\reports\\new_branch_discovery_20260813.md`："
    "pattern match-rate 15.77%（全股票 2,181 筆）／38.0%（strict-lock-liquid 250 筆），"
    "只表示後續賣超型態的命中比例，不是隔日放空有實際報酬優勢的證明。"
)
