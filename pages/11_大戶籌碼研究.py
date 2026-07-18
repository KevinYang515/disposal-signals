"""Large-holder accumulation research dashboard (research only, not execution)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="大戶籌碼研究", page_icon="🐋", layout="wide")

DATA = Path(__file__).resolve().parents[1] / "data" / "whale_research"
DATA_VERSION = "2026-07-19-v2"
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

tab_signal, tab_validation, tab_factor, tab_method = st.tabs([
    "最新候選", "跨期驗證", "因子歸因", "策略定義",
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
            "distance_ma20_pct", "daily_value_20d_m",
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
    view = latest.copy()
    if strategy != "全部":
        view = view[view["strategy"] == strategy]
    view = view[
        (view["whale_base_4w_pct"] <= base_max)
        & (view["whale_change_4w_pp"] >= abs_min)
        & (view["whale_relative_change_pct"] >= rel_min)
    ].sort_values(["whale_relative_change_pct", "whale_change_4w_pp"], ascending=False)
    st.caption(f"符合條件：{len(view):,} 筆；同一股票可能同時符合多種型態。")
    st.dataframe(display_frame(view), use_container_width=True, hide_index=True, height=500)
    st.download_button("下載目前篩選結果 CSV", display_frame(view).to_csv(index=False).encode("utf-8-sig"),
                       "whale_research_candidates.csv", "text/csv")

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
