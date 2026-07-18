"""Large-holder accumulation research dashboard (research only, not execution)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="大戶籌碼研究", page_icon="🐋", layout="wide")

DATA = Path(__file__).resolve().parents[1] / "data" / "whale_research"


@st.cache_data(ttl=3600)
def load_csv(name):
    return pd.read_csv(DATA / name)


@st.cache_data(ttl=3600)
def load_meta():
    with open(DATA / "meta.json", encoding="utf-8") as f:
        return json.load(f)


try:
    meta = load_meta()
    latest = load_csv("latest_signals.csv")
    recommended = load_csv("recommended_candidates.csv")
    composite = load_csv("composite_candidate_validation.csv")
    path_quality = load_csv("path_quality_validation.csv")
    whale_bands = load_csv("attribution_whale_bands.csv")
    concentration = load_csv("attribution_concentration_relative_change.csv")
except FileNotFoundError:
    st.error("找不到大戶研究資料。請先執行 export_whale_research_to_site.py。")
    st.stop()

st.title("🐋 大戶籌碼研究")
st.caption(
    f"研究資料日期：{meta['signal_date']}｜匯出時間：{meta['generated_at']}｜"
    "週籌碼資料採下一交易日收盤進場的保守假設。"
)
st.warning("此頁是研究與觀察工具，不是買賣建議；最新籌碼資料可能落後當日行情，請先確認資料日期。")

tab_signal, tab_validation, tab_factor, tab_method = st.tabs([
    "最新候選", "跨期驗證", "因子歸因", "策略定義",
])

with tab_signal:
    st.subheader("目前符合主要候選模型")
    st.caption("條件：20 日回檔 10–20%、仍在 240 日線上、起始大戶比例 <40%、4 週絕對增加 ≥1pp 且相對增加 ≥5%。")
    c1, c2, c3 = st.columns(3)
    c1.metric("最新資料日", meta["signal_date"])
    c2.metric("全部型態訊號", f"{meta['latest_rows']:,}")
    c3.metric("主要候選", f"{meta['recommended_rows']:,}")

    def display_frame(df):
        cols = [
            "stock_id", "strategy", "signal_close", "whale_base_4w_pct", "whale_current_pct",
            "whale_change_4w_pp", "whale_relative_change_pct", "whale_up_weeks_4",
            "whale_pullback_from_peak_pp", "selected_lot_bucket", "prior_20d_return_pct",
            "distance_ma20_pct", "daily_value_20d",
        ]
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()
        out = out.rename(columns={
            "stock_id": "代號", "strategy": "型態", "signal_close": "訊號收盤價",
            "whale_base_4w_pct": "4週前大戶(%)", "whale_current_pct": "目前大戶(%)",
            "whale_change_4w_pp": "4週變化(pp)", "whale_relative_change_pct": "相對變化(%)",
            "whale_up_weeks_4": "上升週數", "whale_pullback_from_peak_pp": "距峰值回落(pp)",
            "selected_lot_bucket": "動態級距(張)", "prior_20d_return_pct": "近20日漲跌(%)",
            "distance_ma20_pct": "距月線(%)", "daily_value_20d": "20日均成交額",
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
    strategy = a.selectbox("價格型態", strategy_options)
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
    comp = composite.rename(columns={
        "candidate": "候選策略", "period": "期間", "events": "樣本數", "signal_dates": "訊號日數",
        "raw_mean_20d_pct": "20日平均(%)", "raw_mean_60d_pct": "60日平均(%)",
        "raw_mean_120d_pct": "120日平均(%)", "weekly_equal_mean_60d_pct": "60日等權平均(%)",
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.subheader("容許籌碼途中小幅回落的驗證")
    path = path_quality.rename(columns={
        "definition": "條件", "period": "期間", "events": "樣本數", "signal_dates": "訊號日數",
        "raw_mean_20d_pct": "20日平均(%)", "raw_mean_60d_pct": "60日平均(%)",
        "raw_mean_120d_pct": "120日平均(%)", "weekly_equal_mean_60d_pct": "60日等權平均(%)",
    })
    st.dataframe(path, use_container_width=True, hide_index=True)

with tab_factor:
    st.subheader("不同大戶變化幅度的後續報酬")
    selected_strategy = st.selectbox("選擇型態", sorted(whale_bands["strategy"].unique().tolist()), key="factor_strategy")
    wb = whale_bands[whale_bands["strategy"] == selected_strategy].copy()
    chart_cols = [c for c in ["mean_20d_pct", "mean_60d_pct", "mean_120d_pct"] if c in wb]
    chart = wb.set_index("whale_band")[chart_cols].rename(columns={
        "mean_20d_pct": "20日", "mean_60d_pct": "60日", "mean_120d_pct": "120日",
    })
    st.bar_chart(chart)
    st.dataframe(wb, use_container_width=True, hide_index=True)

    st.subheader("初始集中度 × 相對大戶變化")
    con = concentration[concentration["strategy"] == selected_strategy]
    base_options = sorted(con["base_concentration_band"].dropna().unique().tolist())
    base_band = st.selectbox("初始集中度", base_options, key="base_band")
    st.dataframe(con[con["base_concentration_band"] == base_band], use_container_width=True, hide_index=True)

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
