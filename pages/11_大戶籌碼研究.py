"""Large-holder accumulation research dashboard (research only, not execution)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="大戶籌碼研究", page_icon="🐋", layout="wide")

DATA = Path(__file__).resolve().parents[1] / "data" / "whale_research"
DATA_VERSION = "2026-07-19-v4"
STRATEGY_LABELS = {
    "dip_10_20d_above_ma240": "多頭回檔承接", "dip_5_10d_above_ma240": "淺幅回檔承接",
    "ma20_reclaim_uptrend": "月線重新站回", "ma20_support_uptrend": "月線支撐",
    "ma60_reclaim_uptrend": "季線重新站回", "ma60_support_uptrend": "季線支撐",
    "near_ma20_pm5": "月線附近", "near_ma60_pm5": "季線附近",
    "low_1m": "1個月相對低位", "low_3m": "3個月相對低位", "low_6m": "6個月相對低位",
    "pullback_20_30": "中期回檔", "breakout_20d": "20日突破", "breakout_60d": "60日突破",
}


@st.cache_data(ttl=3600)
def load_csv(name, version):
    return pd.read_csv(DATA / name)


@st.cache_data(ttl=3600)
def load_meta(version):
    with open(DATA / "meta.json", encoding="utf-8") as f:
        return json.load(f)


try:
    meta = load_meta(DATA_VERSION)
    latest = load_csv("latest_signals.csv", DATA_VERSION)
    recommended = load_csv("recommended_candidates.csv", DATA_VERSION)
    composite = load_csv("composite_candidate_validation.csv", DATA_VERSION)
    path_quality = load_csv("path_quality_validation.csv", DATA_VERSION)
    exits = load_csv("exit_rule_summary.csv", DATA_VERSION)
    whale_bands = load_csv("attribution_whale_bands.csv", DATA_VERSION)
    concentration = load_csv("attribution_concentration_relative_change.csv", DATA_VERSION)
    weekly = load_csv("weekly_whale_movements.csv", DATA_VERSION)
    history = load_csv("strategy_backtest_history.csv", DATA_VERSION)
    jump_history = load_csv("jump_path_history.csv", DATA_VERSION)
    auto_strategies = load_csv("whale_walkforward_robust_candidates.csv", DATA_VERSION)
except FileNotFoundError:
    st.error("找不到大戶研究資料。請先執行 export_whale_research_to_site.py。")
    st.stop()

if "strategy_name" not in latest.columns:
    latest["strategy_name"] = latest["strategy"].map(STRATEGY_LABELS).fillna(latest["strategy"])
if "strategy_description" not in latest.columns:
    latest["strategy_description"] = "-"
if "company_name" not in latest.columns:
    latest["company_name"] = "名稱待補"

st.title("🐋 大戶籌碼研究")
st.caption(
    f"最新訊號資料日：{meta['signal_date']}｜可回測資料範圍：{meta['research_signal_start']} 至 {meta['research_complete_backtest_end']}｜匯出時間：{meta['generated_at']}｜"
    "週籌碼資料採下一交易日收盤進場的保守假設。"
)
st.warning("此頁已排除 ETF／ETN（00xx 代號）。此頁是研究與觀察工具，不是買賣建議；最新籌碼資料可能落後當日行情，請先確認資料日期。")
strategy_labels = latest.drop_duplicates("strategy").set_index("strategy")["strategy_name"].to_dict()


def round_numbers(df):
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].round(2)
    return out


RETURN_COLUMNS = {
    "events": "樣本數", "signal_dates": "訊號日數",
    "mean_20d_pct": "20日平均報酬(%)", "mean_net_20d_pct": "20日平均淨報酬(%)",
    "median_20d_pct": "20日中位數報酬(%)", "win_rate_20d_pct": "20日勝率(%)",
    "mean_60d_pct": "60日平均報酬(%)", "mean_net_60d_pct": "60日平均淨報酬(%)",
    "median_60d_pct": "60日中位數報酬(%)", "win_rate_60d_pct": "60日勝率(%)",
    "mean_120d_pct": "120日平均報酬(%)", "mean_net_120d_pct": "120日平均淨報酬(%)",
    "median_120d_pct": "120日中位數報酬(%)", "win_rate_120d_pct": "120日勝率(%)",
}

tab_signal, tab_weekly, tab_history, tab_validation, tab_factor, tab_jump, tab_auto, tab_method = st.tabs([
    "最新候選", "每週大戶變動", "策略歷史回測", "跨期驗證", "因子歸因", "單雙週跳升研究", "自動策略搜尋", "策略定義",
])

with tab_signal:
    st.subheader("目前符合主要候選模型")
    st.caption("條件：20 日回檔 10–20%、仍在 240 日線上、起始大戶比例 <40%、4 週絕對增加 ≥1pp 且相對增加 ≥5%。")
    st.info("進場不是四週前：『4週前大戶(%)』只用來計算目前已確認的籌碼變化。回測的進場基準是資料日後的下一個交易日收盤。")
    c1, c2, c3 = st.columns(3)
    c1.metric("最新資料日", meta["signal_date"])
    c2.metric("全部型態訊號", f"{meta['latest_rows']:,}")
    c3.metric("主要候選", f"{meta['recommended_rows']:,}")

    def display_frame(df):
        cols = [
            "signal_date", "entry_date", "stock_id", "company_name", "strategy_name", "strategy_description", "signal_close", "entry_close", "whale_base_4w_pct", "whale_current_pct",
            "whale_change_4w_pp", "whale_relative_change_pct", "whale_up_weeks_4",
            "whale_pullback_from_peak_pp", "selected_lot_bucket", "prior_20d_return_pct",
            "distance_ma20_pct", "daily_value_20d_m", "prev_day_limit_down",
            "prev_day_single_limit_down",
        ]
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()
        for col in out.select_dtypes(include="number").columns:
            out[col] = out[col].round(2)
        out = out.rename(columns={
            "signal_date": "訊號資料日", "entry_date": "回測進場日", "stock_id": "代號", "company_name": "股票名稱", "strategy_name": "價格型態", "strategy_description": "型態說明",
            "signal_close": "資料日收盤價", "entry_close": "回測進場價",
            "whale_base_4w_pct": "4週前大戶(%)", "whale_current_pct": "目前大戶(%)",
            "whale_change_4w_pp": "4週變化(pp)", "whale_relative_change_pct": "相對變化(%)",
            "whale_up_weeks_4": "上升週數", "whale_pullback_from_peak_pp": "距峰值回落(pp)",
            "selected_lot_bucket": "動態級距(張)", "prior_20d_return_pct": "近20日漲跌(%)",
            "distance_ma20_pct": "距月線(%)", "daily_value_20d": "20日均成交額",
            "daily_value_20d_m": "20日均成交額(百萬元)",
        })
        for col in ["prev_day_limit_down", "prev_day_single_limit_down"]:
            if col in out.columns:
                out[col] = out[col].map({True: "是", False: "否"}).fillna("否")
        out = out.rename(columns={
            "prev_day_limit_down": "訊號日前一交易日是否跌停",
            "prev_day_single_limit_down": "訊號日前一交易日是否為單獨跌停",
        })
        return out

    if recommended.empty:
        st.info("最新一期沒有符合主要候選模型的標的。可於下方調整探索條件。")
    else:
        st.dataframe(display_frame(recommended), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("探索最新籌碼訊號")
    a, b, c, d = st.columns(4)
    strategy_options = ["全部"] + sorted(latest["strategy"].dropna().unique().tolist())
    strategy = a.selectbox("價格型態", strategy_options,
                           format_func=lambda x: "全部" if x == "全部" else strategy_labels.get(x, x))
    base_max = b.slider("4週前大戶比例上限", 0, 100, 100)
    abs_min = c.slider("4週絕對增加至少(pp)", -5.0, 10.0, 0.5, 0.5)
    rel_min = d.slider("相對增加至少(%)", -20.0, 50.0, 0.0, 1.0)
    exclude_prev_limit = st.checkbox("排除訊號日前一交易日跌停", value=False)
    exclude_single_limit = st.checkbox("排除訊號日前一交易日單獨跌停", value=False)
    view = latest.copy()
    if strategy != "全部":
        view = view[view["strategy"] == strategy]
    if exclude_prev_limit and "prev_day_limit_down" in view.columns:
        view = view[~view["prev_day_limit_down"].fillna(False)]
    if exclude_single_limit and "prev_day_single_limit_down" in view.columns:
        view = view[~view["prev_day_single_limit_down"].fillna(False)]
    view = view[
        (view["whale_base_4w_pct"] <= base_max)
        & (view["whale_change_4w_pp"] >= abs_min)
        & (view["whale_relative_change_pct"] >= rel_min)
    ].sort_values(["whale_relative_change_pct", "whale_change_4w_pp"], ascending=False)
    st.caption(f"符合條件：{len(view):,} 筆；同一股票可能同時符合多種型態。")
    st.dataframe(display_frame(view), use_container_width=True, hide_index=True, height=500)
    st.download_button("下載目前篩選結果 CSV", display_frame(view).to_csv(index=False).encode("utf-8-sig"),
                       "whale_research_candidates.csv", "text/csv")

with tab_weekly:
    st.subheader("每週大戶持股變動排行")
    st.caption("依當週股價動態選擇對應大戶級距；已排除 ETF／ETN。每週保留上升與下降各前 200 名，方便找出明顯籌碼變化。")
    weekly["資料日期"] = pd.to_datetime(weekly["資料日期"])
    w1, w2, w3, w4 = st.columns(4)
    week_dates = sorted(weekly["資料日期"].dt.date.unique(), reverse=True)
    selected_week = w1.selectbox("選擇資料日期", week_dates, format_func=lambda x: x.strftime("%Y-%m-%d"))
    direction = w2.radio("持股變動方向", ["上升", "下降"], horizontal=True)
    threshold = w3.slider("變動門檻（百分點）", 0.0, 10.0, 0.5, 0.1)
    top_n = w4.slider("顯示前幾名", 20, 200, 100, 10)
    week_view = weekly[weekly["資料日期"].dt.date == selected_week].copy()
    change_col = "本週變化(百分點)"
    if direction == "上升":
        week_view = week_view[week_view[change_col] >= threshold].sort_values(change_col, ascending=False)
    else:
        week_view = week_view[week_view[change_col] <= -threshold].sort_values(change_col)
    week_view = week_view.head(top_n)
    week_view["資料日期"] = week_view["資料日期"].dt.strftime("%Y-%m-%d")
    st.caption(f"符合條件：{len(week_view):,} 筆")
    st.dataframe(round_numbers(week_view), use_container_width=True, hide_index=True, height=520)
    st.download_button("下載本週清單 CSV", week_view.to_csv(index=False).encode("utf-8-sig"),
                       f"weekly_whale_{selected_week}.csv", "text/csv")

with tab_history:
    st.subheader("策略歷史回測紀錄")
    st.caption("可調整策略與籌碼條件，查看每筆歷史訊號的資料日、下一交易日收盤進場價及後續報酬。僅包含已走完 120 個交易日的完整樣本。")
    history["signal_date"] = pd.to_datetime(history["signal_date"])
    history["entry_date"] = pd.to_datetime(history["entry_date"])
    h1, h2, h3, h4 = st.columns(4)
    h_strategy_options = sorted(history["strategy"].unique().tolist())
    h_strategy = h1.selectbox("回測策略", h_strategy_options, format_func=lambda x: strategy_labels.get(x, x), key="history_strategy")
    min_abs = h2.slider("大戶4週絕對增加至少（pp）", -5.0, 10.0, 0.0, 0.5, key="history_abs")
    min_rel = h3.slider("大戶4週相對增加至少（%）", -20.0, 50.0, -20.0, 1.0, key="history_rel")
    max_base = h4.slider("4週前大戶比例上限（%）", 0.0, 100.0, 100.0, 1.0, key="history_base")
    min_date, max_date = history["signal_date"].min().date(), history["signal_date"].max().date()
    date_range = st.date_input("訊號資料日範圍", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="history_dates")
    exclude_hist_limit = st.checkbox("回測排除訊號日前一交易日跌停", value=False, key="history_limit")
    exclude_hist_single = st.checkbox("回測排除訊號日前一交易日單獨跌停", value=False, key="history_single_limit")
    hist_view = history[
        (history["strategy"] == h_strategy)
        & (history["whale_change_4w_pp"] >= min_abs)
        & (history["whale_relative_change_pct"] >= min_rel)
        & (history["whale_base_4w_pct"] <= max_base)
    ].copy()
    if exclude_hist_limit and "prev_day_limit_down" in hist_view.columns:
        hist_view = hist_view[~hist_view["prev_day_limit_down"].fillna(False)]
    if exclude_hist_single and "prev_day_single_limit_down" in hist_view.columns:
        hist_view = hist_view[~hist_view["prev_day_single_limit_down"].fillna(False)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        hist_view = hist_view[hist_view["signal_date"].dt.date.between(date_range[0], date_range[1])]
    returns = hist_view[["return_20d", "return_60d", "return_120d"]] * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("歷史訊號數", f"{len(hist_view):,}")
    k2.metric("20日平均報酬", f"{returns['return_20d'].mean():.2f}%" if len(hist_view) else "-")
    k3.metric("60日平均報酬", f"{returns['return_60d'].mean():.2f}%" if len(hist_view) else "-")
    k4.metric("120日平均報酬", f"{returns['return_120d'].mean():.2f}%" if len(hist_view) else "-")
    hist_show = hist_view.rename(columns={
        "signal_date": "訊號資料日", "entry_date": "回測進場日", "stock_id": "代號", "策略名稱": "策略",
        "signal_close": "資料日收盤價", "entry_close": "回測進場價", "whale_base_4w_pct": "4週前大戶(%)",
        "whale_current_pct": "目前大戶(%)", "whale_change_4w_pp": "4週變化(pp)",
        "whale_relative_change_pct": "相對變化(%)", "whale_up_weeks_4": "上升週數",
        "whale_pullback_from_peak_pp": "距峰值回落(pp)", "prior_20d_return_pct": "近20日漲跌(%)",
        "return_20d": "20日報酬(%)", "return_60d": "60日報酬(%)", "return_120d": "120日報酬(%)",
    })
    for col in ["20日報酬(%)", "60日報酬(%)", "120日報酬(%)"]:
        hist_show[col] = hist_show[col] * 100
    history_cols = ["訊號資料日", "回測進場日", "代號", "股票名稱", "策略", "資料日收盤價", "回測進場價",
                    "4週前大戶(%)", "目前大戶(%)", "4週變化(pp)", "相對變化(%)", "近20日漲跌(%)",
                    "20日報酬(%)", "60日報酬(%)", "120日報酬(%)"]
    st.dataframe(round_numbers(hist_show[[c for c in history_cols if c in hist_show]]), use_container_width=True, hide_index=True, height=520)

with tab_validation:
    st.subheader("候選策略跨期驗證")
    st.caption("報酬為未扣成本平均報酬；同時列出每個訊號日等權平均，以降低訊號群聚的影響。")
    condition_names = {
        "dip10_20_uptrend_low_base_relative_accumulation": "多頭回檔承接＋低起始集中度＋相對籌碼累積",
        "ma20_reclaim_uptrend_low_base_relative_accumulation": "月線重新站回＋低起始集中度＋相對籌碼累積",
        "dip10_20_uptrend: strict net>=1pp": "多頭回檔承接：4週淨增加至少1個百分點",
        "dip10_20_uptrend: flexible net>=0.5pp, 2+ up weeks, peak pullback<=1pp": "多頭回檔承接：4週淨增加至少0.5pp、至少2週增加、距峰值回落不超過1pp",
        "ma20_reclaim_uptrend: strict net>=1pp": "月線重新站回：4週淨增加至少1個百分點",
        "ma20_reclaim_uptrend: flexible net>=0.5pp, 2+ up weeks, peak pullback<=1pp": "月線重新站回：4週淨增加至少0.5pp、至少2週增加、距峰值回落不超過1pp",
    }
    comp = composite.copy()
    comp["candidate"] = comp["candidate"].map(condition_names).fillna(comp["candidate"])
    comp = comp.rename(columns={
        "candidate": "候選策略", "period": "期間", "events": "樣本數", "signal_dates": "訊號日數",
        "raw_mean_20d_pct": "20日平均(%)", "raw_mean_60d_pct": "60日平均(%)",
        "raw_mean_120d_pct": "120日平均(%)", "weekly_equal_mean_20d_pct": "每訊號日等權20日(%)",
        "weekly_equal_mean_60d_pct": "每訊號日等權60日(%)", "weekly_equal_mean_120d_pct": "每訊號日等權120日(%)",
    })
    st.dataframe(round_numbers(comp), use_container_width=True, hide_index=True)

    st.subheader("容許籌碼途中小幅回落的驗證")
    path = path_quality.copy()
    path["definition"] = path["definition"].map(condition_names).fillna(path["definition"])
    path = path.rename(columns={
        "definition": "條件", "period": "期間", "events": "樣本數", "signal_dates": "訊號日數",
        "raw_mean_20d_pct": "20日平均(%)", "raw_mean_60d_pct": "60日平均(%)",
        "raw_mean_120d_pct": "120日平均(%)", "weekly_equal_mean_20d_pct": "每訊號日等權20日(%)",
        "weekly_equal_mean_60d_pct": "每訊號日等權60日(%)", "weekly_equal_mean_120d_pct": "每訊號日等權120日(%)",
    })
    st.dataframe(round_numbers(path), use_container_width=True, hide_index=True)

    st.subheader("進出場規則比較：主要候選模型")
    st.caption("進場：籌碼資料確認後的下一交易日收盤。以下為扣除 0.60% 往返成本的收盤價模擬；不是四週前進場。")
    exit_view = exits.rename(columns={
        "rule": "出場規則", "period": "期間", "trades": "交易數", "signal_dates": "訊號日數",
        "net_mean_pct": "平均淨報酬(%)", "net_median_pct": "中位數淨報酬(%)",
        "win_rate_pct": "勝率(%)", "weekly_equal_net_mean_pct": "每訊號日等權淨報酬(%)",
        "mean_holding_days": "平均持有日數", "stop_rate_pct": "停損比例(%)",
        "take_profit_rate_pct": "停利比例(%)",
    })
    exit_view = exit_view[exit_view["期間"] == "all"].copy()
    for col in exit_view.select_dtypes(include="number").columns:
        exit_view[col] = exit_view[col].round(2)
    exit_view["期間"] = exit_view["期間"].replace({"all": "全期間"})
    st.dataframe(round_numbers(exit_view), use_container_width=True, hide_index=True)
    st.info("目前資料較支持固定持有 60–120 個交易日；固定 10% 停損會過早切掉後續大波段。這是初步結果，後續仍需加入部位重複與實際滑價測試。")

with tab_factor:
    st.subheader("不同大戶變化幅度的後續報酬")
    selected_strategy = st.selectbox("選擇型態", sorted(whale_bands["strategy"].unique().tolist()),
                                     format_func=lambda x: strategy_labels.get(x, x), key="factor_strategy")
    wb = whale_bands[whale_bands["strategy"] == selected_strategy].copy()
    chart_cols = [c for c in ["mean_20d_pct", "mean_60d_pct", "mean_120d_pct"] if c in wb]
    whale_band_labels = {
        "<-3": "4週減少超過3個百分點", "-3~-1": "4週減少1–3個百分點",
        "-1~-0.5": "4週減少0.5–1個百分點", "-0.5~0": "4週小幅減少0–0.5個百分點",
        "0~0.5": "4週小幅增加0–0.5個百分點", "0.5~1": "4週增加0.5–1個百分點",
        "1~2": "4週增加1–2個百分點", "2~3": "4週增加2–3個百分點",
        "3~5": "4週增加3–5個百分點", ">=5": "4週增加至少5個百分點",
    }
    chart = wb.assign(大戶變化=wb["whale_band"].map(whale_band_labels)).set_index("大戶變化")[chart_cols].rename(columns={
        "mean_20d_pct": "20日", "mean_60d_pct": "60日", "mean_120d_pct": "120日",
    })
    st.bar_chart(chart)
    wb_view = wb.assign(大戶變化=wb["whale_band"].map(whale_band_labels)).drop(columns=["strategy", "whale_band"])
    wb_view = wb_view.rename(columns={"大戶變化": "大戶4週變化區間", **RETURN_COLUMNS})
    st.dataframe(round_numbers(wb_view), use_container_width=True, hide_index=True)

    st.subheader("初始集中度 × 相對大戶變化")
    con = concentration[concentration["strategy"] == selected_strategy]
    base_labels = {"<40%": "起始大戶比例低於40%", "40~60%": "起始大戶比例40–60%",
                   "60~80%": "起始大戶比例60–80%", ">=80%": "起始大戶比例至少80%"}
    relative_labels = {"<-5%": "相對減少超過5%", "-5~-1%": "相對減少1–5%",
                       "-1~0%": "相對小幅減少0–1%", "0~1%": "相對小幅增加0–1%",
                       "1~5%": "相對增加1–5%", "5~10%": "相對增加5–10%", ">=10%": "相對增加至少10%"}
    base_options = sorted(con["base_concentration_band"].dropna().unique().tolist())
    base_band = st.selectbox("初始大戶集中度", base_options, format_func=lambda x: base_labels.get(x, x), key="base_band")
    con_view = con[con["base_concentration_band"] == base_band].copy()
    con_view["相對大戶變化"] = con_view["relative_change_band"].map(relative_labels)
    con_view = con_view.drop(columns=["strategy", "base_concentration_band", "relative_change_band"])
    con_view = con_view.rename(columns={"相對大戶變化": "相對大戶變化區間", **RETURN_COLUMNS})
    st.dataframe(round_numbers(con_view), use_container_width=True, hide_index=True)

with tab_jump:
    st.subheader("單週／兩週／四週大戶跳升：歷史互動研究")
    st.caption("可自行調整大戶跳升與『跳升前未流失』條件。這是歷史研究，不代表保證的交易建議；訊號日後以次一交易日進場計算。")
    jump_history["signal_date"] = pd.to_datetime(jump_history["signal_date"])
    j1, j2, j3, j4 = st.columns(4)
    jump_type = j1.selectbox("大戶路徑", ["單週大幅跳升", "兩週累積跳升", "四週淨累積"], key="jump_type")
    jump_strategy = j2.selectbox("價格結構", ["全部"] + sorted(jump_history["strategy"].unique().tolist()),
                                 format_func=lambda x: "全部" if x == "全部" else strategy_labels.get(x, x), key="jump_strategy")
    jump_min = j3.slider("大戶增加至少（百分點）", 0.0, 8.0, 1.0, 0.25, key="jump_min")
    base_cap = j4.slider("跳升前大戶比例上限（%）", 0.0, 100.0, 40.0, 1.0, key="jump_base")
    allowed_loss = st.slider("跳升前3週相對4週前最多允許流失（百分點）", 0.0, 5.0, 0.0, 0.25, key="jump_loss")
    jump_view = jump_history.copy()
    if jump_strategy != "全部":
        jump_view = jump_view[jump_view["strategy"] == jump_strategy]
    if jump_type == "單週大幅跳升":
        jump_view = jump_view[(jump_view["whale_lag1_pct"] <= base_cap)
                              & (jump_view["單週大戶變化(百分點)"] >= jump_min)
                              & (jump_view["跳升前3週最低大戶比例(%)"] >= jump_view["whale_lag4_pct"] - allowed_loss)]
    elif jump_type == "兩週累積跳升":
        jump_view = jump_view[(jump_view["whale_lag2_pct"] <= base_cap)
                              & (jump_view["兩週大戶變化(百分點)"] >= jump_min)
                              & (jump_view["單週大戶變化(百分點)"] >= -allowed_loss)]
    else:
        jump_view = jump_view[(jump_view["whale_lag4_pct"] <= base_cap)
                              & (jump_view["四週大戶變化(百分點)"] >= jump_min)]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("歷史訊號數", f"{len(jump_view):,}")
    for metric, col, label in [(k2, "excess_0050_5d", "5日平均超額"),
                               (k3, "excess_0050_10d", "10日平均超額"),
                               (k4, "return_20d", "20日平均報酬")]:
        value = jump_view[col].mean() * 100 if len(jump_view) and col in jump_view else None
        metric.metric(label, f"{value:.2f}%" if value is not None and pd.notna(value) else "-")
    jump_show = jump_view.rename(columns={
        "signal_date": "訊號資料日", "entry_date": "回測進場日", "stock_id": "代號", "strategy": "價格結構代碼",
        "價格型態": "價格結構", "signal_close": "訊號收盤價", "entry_close": "回測進場價",
        "whale_lag4_pct": "4週前大戶比例(%)", "whale_lag3_pct": "3週前大戶比例(%)",
        "whale_lag2_pct": "2週前大戶比例(%)", "whale_lag1_pct": "1週前大戶比例(%)",
        "whale_current_pct": "本週大戶比例(%)", "selected_lot_bucket": "固定大戶級距(張)",
        "prior_20d_return_pct": "訊號前20日股價報酬(%)", "return_5d": "進場後5日報酬(%)",
        "return_10d": "進場後10日報酬(%)", "return_20d": "進場後20日報酬(%)",
        "excess_0050_5d": "5日相對0050超額(%)", "excess_0050_10d": "10日相對0050超額(%)",
    }).copy()
    pct_cols = [c for c in ["進場後5日報酬(%)", "進場後10日報酬(%)", "進場後20日報酬(%)",
                              "5日相對0050超額(%)", "10日相對0050超額(%)"] if c in jump_show]
    for col in pct_cols:
        jump_show[col] = jump_show[col] * 100
    jump_cols = ["訊號資料日", "回測進場日", "代號", "股票名稱", "價格結構", "訊號收盤價", "回測進場價",
                 "4週前大戶比例(%)", "跳升前3週最低大戶比例(%)", "單週大戶變化(百分點)",
                 "兩週大戶變化(百分點)", "四週大戶變化(百分點)", "單週相對變化(%)", "兩週相對變化(%)",
                 "進場後5日報酬(%)", "5日相對0050超額(%)", "進場後10日報酬(%)",
                 "10日相對0050超額(%)", "進場後20日報酬(%)"]
    st.dataframe(round_numbers(jump_show[[c for c in jump_cols if c in jump_show]]), use_container_width=True,
                 hide_index=True, height=520)
    st.download_button("下載目前篩選的歷史資料 CSV", round_numbers(jump_show).to_csv(index=False).encode("utf-8-sig"),
                       "whale_jump_path_history.csv", "text/csv")

with tab_auto:
    st.subheader("自動策略搜尋：跨期驗證後的候選")
    st.caption("先只用 2017–2021 排名；再以 2022–2023 篩選。2024–2025 僅作未參與選模的最終檢驗，即使結果不佳也保留顯示。這可降低過度擬合，但不代表未來保證有效。")
    auto_view = auto_strategies.copy()
    auto_view["價格結構"] = auto_view["價格結構"].map(STRATEGY_LABELS).fillna(auto_view["價格結構"])
    auto_view = auto_view.rename(columns={
        "增加門檻(百分點)": "大戶增加門檻(pp)", "原始比例上限(%)": "原始大戶比例上限(%)",
        "跳升前允許流失(百分點)": "跳升前允許流失(pp)",
        "訓練期2017–2021_n": "訓練樣本數", "訓練期2017–2021_weekly": "訓練期每週等權超額(%)",
        "驗證期2022–2023_n": "第一驗證樣本數", "驗證期2022–2023_weekly": "第一驗證每週等權超額(%)",
        "最終驗證2024–2025_n": "最終驗證樣本數", "最終驗證2024–2025_weekly": "最終驗證每週等權超額(%)",
        "最終驗證2024–2025_median": "最終驗證中位超額(%)", "最終驗證2024–2025_win": "最終驗證勝過0050比例(%)",
    })
    auto_cols = ["價格結構", "大戶路徑", "大戶增加門檻(pp)", "原始大戶比例上限(%)", "跳升前允許流失(pp)", "穩健分數",
                 "訓練樣本數", "訓練期每週等權超額(%)", "第一驗證樣本數", "第一驗證每週等權超額(%)",
                 "最終驗證樣本數", "最終驗證每週等權超額(%)", "最終驗證中位超額(%)", "最終驗證勝過0050比例(%)"]
    st.dataframe(round_numbers(auto_view[[c for c in auto_cols if c in auto_view]]), use_container_width=True, hide_index=True)
    st.download_button("下載自動搜尋候選 CSV", round_numbers(auto_view).to_csv(index=False).encode("utf-8-sig"),
                       "whale_walkforward_robust_candidates.csv", "text/csv")

with tab_method:
    st.markdown("""
### 動態大戶級距

依股價選擇最接近約 8,000 萬元持有市值的 100／200／400／600／800／1,000 張級距；
因此高價股不會因固定張數門檻而失真。

### 核心因子

- 絕對變化：4 週大戶持股比例增加的百分點。
- 相對變化：絕對變化 ÷ 4 週前大戶比例，避免將 90%→90.5% 與 40%→45% 視為相同訊號。
- 籌碼路徑品質：允許過程小幅回落，但衡量淨增加、上升週數及距近期峰值的回落。

### 重要限制

- 報酬未扣交易成本；候選策略尚未加入不重複持倉與完整部位管理。
- 回測事件可能重疊，頁面同時提供每訊號日等權平均作為較保守的檢查。
- 資料日不是即時交易訊號，使用前應確認最新可得日期。
""")
