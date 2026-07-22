"""
爆量反彈策略 - 每日觀察名單
用已驗證的4因子規則(市場寬度/20日回檔/融資5日變化/爆量)掃描最新一個交易日資料。
資料由 D:\\stock\\stock\\burst_ml_ga_study\\daily_scan.py 產生，目前為手動執行，尚未排程。
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="爆量反彈-明日觀察", page_icon="🔎", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "data" / "burst_ml_ga"

UP_COLOR = "#f87171"
DOWN_COLOR = "#4ade80"


@st.cache_data(ttl=1800)
def load_daily():
    qualified = pd.read_csv(DATA_DIR / "daily_watch_qualified.csv")
    watch = pd.read_csv(DATA_DIR / "daily_watch_nearmiss.csv")
    meta = pd.read_csv(DATA_DIR / "daily_watch_meta.csv").iloc[0]
    return qualified, watch, meta


st.title("🔎 爆量反彈策略：每日觀察名單")

try:
    qualified, watch, meta = load_daily()
except FileNotFoundError:
    st.error("找不到每日掃描資料，請先執行 `daily_scan.py` 產生 data/burst_ml_ga/daily_watch_*.csv")
    st.stop()

st.caption(
    f"掃描日期：{meta['scan_date']}　產生時間：{meta['generated_at']} (台灣時間)　"
    f"當天全市場同時跌停家數：**{int(meta['breadth_severe_today'])}**　"
    f"規則門檻：寬度≥{int(meta['rule_breadth_min'])}檔 / 20日回檔≤{meta['rule_drawdown_max']:.0f}% / "
    f"融資5日變化≤{meta['rule_margin_max']:.0f} / 量比≥{meta['rule_vol_min']:.1f}倍"
)

st.warning(
    "⚠️ **這份清單目前是手動產生的研究工具，還沒有排程自動每日更新**——上面「產生時間」如果不是今天，"
    "代表資料是舊的，先確認時間再參考。核心規則詳見「💥爆量反彈策略」tab。"
    "**市場寬度是必要條件，不是加分項**：如果當天全市場跌停家數遠低於規則門檻，"
    "即使某檔股票其他3個條件都符合，歷史上也沒有可靠的正期望值——這種情況下面的"
    "「觀察名單」只代表「差臨門一腳」，不是「可以進場」，除非市場寬度真的跟上來。"
)

tab_q, tab_w = st.tabs([f"✅ 完全符合訊號 ({len(qualified)})", f"👀 觀察名單/差1個條件 ({len(watch)})"])


def render_table(df):
    if df.empty:
        st.info("目前沒有符合的股票。")
        return
    disp = df.rename(columns={
        "code": "代號", "name": "名稱", "industry": "產業", "close_px": "收盤價",
        "ret_pct": "當日漲跌%", "drawdown_20d": "20日回檔%", "vol_ratio": "量比",
        "margin_chg5": "融資5日變化", "breadth_severe": "當天全市場跌停家數",
        "ok_breadth": "寬度✓", "ok_drawdown": "回檔✓", "ok_margin": "融資✓", "ok_volume": "量比✓",
        "n_ok": "符合條件數",
    })

    def color_ret(v):
        if pd.isna(v):
            return ""
        return f"color: {UP_COLOR}" if v > 0 else (f"color: {DOWN_COLOR}" if v < 0 else "")

    st.dataframe(
        disp.style.map(color_ret, subset=["當日漲跌%"]).format({
            "收盤價": "{:.2f}", "當日漲跌%": "{:+.2f}%", "20日回檔%": "{:+.2f}%",
            "量比": "{:.2f}", "融資5日變化": "{:+.2f}", "當天全市場跌停家數": "{:.0f}",
            "符合條件數": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True, height=420)


with tab_q:
    st.caption("4個條件(市場寬度/20日回檔/融資5日變化/量比)全部符合——歷史上驗證有效的完整訊號。")
    render_table(qualified)

with tab_w:
    st.caption("符合3個條件、通常是差市場寬度這一項——先觀察，等寬度真的跟上來再考慮，不要單獨依賴這份清單進場。")
    render_table(watch)

st.divider()
st.caption(
    "資料/規則來源：D:\\stock\\stock\\burst_ml_ga_study\\daily_scan.py，"
    "完整回測與驗證見「💥爆量反彈策略」tab（含市場寬度為何是必要條件、可成交性校正等）。"
)
