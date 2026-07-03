"""
急拉高空方策略回測
策略來源：童大的交易生活「急拉高策略」
核心邏輯：爆量急拉後推不動 → 多單套牢 → 尾盤失望性殺盤
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="急拉高空方", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")

DATA_DIR   = Path(__file__).parent.parent / "data"
SIGNAL_LIB = DATA_DIR / "急拉高_信號庫_HONEST.parquet"
SWEEP_CSV  = DATA_DIR / "急拉高參數掃描結果_HONEST.csv"

# ── 側欄參數 ─────────────────────────────────────────────
with st.sidebar:
    st.title("策略參數")
    surge_thr = st.selectbox("急拉幅度（前收）", [0.03, 0.05, 0.07, 0.10],
                             index=1, format_func=lambda x: f">= {x:.0%}")
    entry_pb  = st.selectbox("進場回落", [0.01, 0.015, 0.02, 0.025, 0.03],
                             index=0, format_func=lambda x: f"{x:.1%}")
    stop_lvl  = st.selectbox("停損距離", [0.003, 0.005, 0.01],
                             index=1, format_func=lambda x: f"{x:.1%}")
    vol_mult  = st.selectbox("量能倍率", [1.0, 1.5, 2.0],
                             index=1, format_func=lambda x: f"> 均量 {x:.1f}x")
    sig_win   = st.selectbox("信號窗口", [60, 90, 120],
                             index=0, format_func=lambda x: f"前 {x} 分鐘")
    st.divider()
    st.caption("資料期間：2023-01 ~ 2026-06\n758 支個股 1-min K\n"
               "2026-07-03 修正：量能改用可知窗口量(非全天量)、"
               "進場改用停損市價單滑1檔(非精準觸價)、加處置股/可空性過濾")

# ── 讀取信號庫 ────────────────────────────────────────────
@st.cache_data
def load_lib():
    return pd.read_parquet(SIGNAL_LIB)

@st.cache_data
def load_sweep():
    return pd.read_csv(SWEEP_CSV)

if not SIGNAL_LIB.exists():
    st.error(f"找不到信號庫：{SIGNAL_LIB}\n請先執行 `策略_急拉高參數掃描_v2.py`")
    st.stop()

lib  = load_lib()
lib["date"] = pd.to_datetime(lib["date"].astype(str))
lib["year"] = lib["date"].dt.year

# ── 套用參數 ──────────────────────────────────────────────
ep_col     = f"entry_min_{entry_pb}"
fill_col   = f"entry_fill_{entry_pb}"
exp_col    = f"exit_price_{entry_pb}_{stop_lvl}"
stop_col   = f"stopped_{entry_pb}_{stop_lvl}"
volr_col   = f"vol_ratio_{sig_win}"

mask = (
    (lib["surge_pct"]  >= surge_thr) &
    (lib[volr_col]     >= vol_mult) &
    (lib["surge_min"]  <= 9*60 + sig_win) &
    (lib[ep_col].notna())
)
sub = lib[mask].copy()
sub["entry_price"] = sub[fill_col]   # 停損市價單, 滑1檔的誠實成交價 (非精準觸價幻覺)
sub["exit_price"]  = sub[exp_col]
sub["ret"]         = (sub["entry_price"] - sub["exit_price"]) / sub["entry_price"]
sub["win"]         = sub["ret"] > 0

# ── 頂部指標 ─────────────────────────────────────────────
st.title("急拉高空方策略回測")
st.caption("爆量急拉後推不動 → 回落 1% 進場做空 → 13:25 強制平倉")

if sub.empty:
    st.warning("目前參數組合無觸發交易，請放寬條件。")
    st.stop()

n      = len(sub)
wr     = sub["win"].mean()
avg_r  = sub["ret"].mean()
gw     = sub.loc[sub["win"], "ret"].sum()
gl     = abs(sub.loc[~sub["win"], "ret"].sum())
pf     = gw / gl if gl > 0 else float("inf")
stop_r = sub[stop_col].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("交易次數", f"{n:,}")
col2.metric("勝率", f"{wr:.1%}")
col3.metric("Profit Factor", f"{pf:.2f}")
col4.metric("均報酬/筆", f"{avg_r:.2%}")
col5.metric("停損觸發率", f"{stop_r:.1%}")

# ── 年度分解 ──────────────────────────────────────────────
st.subheader("年度穩健性")
yr = (sub.groupby("year")
        .agg(次數=("ret","count"), 勝率=("win","mean"),
             均報酬=("ret","mean"), 停損率=(stop_col,"mean"))
        .reset_index())

bar_wr = alt.Chart(yr).mark_bar(color="#4a90d9").encode(
    x=alt.X("year:O", title="年度"),
    y=alt.Y("勝率:Q", title="勝率", scale=alt.Scale(domain=[0,1]),
            axis=alt.Axis(format=".0%")),
    tooltip=[alt.Tooltip("year:O", title="年"),
             alt.Tooltip("勝率:Q", format=".1%"),
             alt.Tooltip("次數:Q"),
             alt.Tooltip("均報酬:Q", format=".2%")]
).properties(height=200, title="年度勝率")

bar_ret = alt.Chart(yr).mark_bar(color="#27ae60").encode(
    x=alt.X("year:O", title="年度"),
    y=alt.Y("均報酬:Q", title="均報酬", axis=alt.Axis(format=".1%")),
    tooltip=[alt.Tooltip("year:O", title="年"),
             alt.Tooltip("均報酬:Q", format=".2%"),
             alt.Tooltip("次數:Q")]
).properties(height=200, title="年度均報酬")

c1, c2 = st.columns(2)
c1.altair_chart(bar_wr,  use_container_width=True)
c2.altair_chart(bar_ret, use_container_width=True)

# ── 急拉幅度分層 ──────────────────────────────────────────
st.subheader("急拉幅度分層")
sub["surge_bucket"] = pd.cut(
    sub["surge_pct"],
    bins=[surge_thr, 0.07, 0.10, 0.15, 1.0],
    labels=[f"{surge_thr:.0%}–7%", "7–10%", "10–15%", ">15%"],
    right=True
)
sg = (sub.groupby("surge_bucket", observed=True)
        .agg(次數=("ret","count"), 勝率=("win","mean"), 均報酬=("ret","mean"))
        .reset_index())

bar_sg = alt.Chart(sg).mark_bar().encode(
    x=alt.X("surge_bucket:O", title="急拉幅度"),
    y=alt.Y("勝率:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0,1])),
    color=alt.Color("勝率:Q", scale=alt.Scale(scheme="blues"), legend=None),
    tooltip=[alt.Tooltip("surge_bucket:O", title="急拉幅度"),
             alt.Tooltip("勝率:Q", format=".1%"),
             alt.Tooltip("均報酬:Q", format=".2%"),
             alt.Tooltip("次數:Q")]
).properties(height=250, title="各急拉幅度勝率")

st.altair_chart(bar_sg, use_container_width=True)

# ── 累積權益曲線 ──────────────────────────────────────────
st.subheader("累積報酬曲線")
sub_sorted = sub.sort_values("date").reset_index(drop=True)
sub_sorted["cum_ret"] = (1 + sub_sorted["ret"]).cumprod() - 1
sub_sorted["trade_no"] = sub_sorted.index + 1

line = alt.Chart(sub_sorted).mark_line(color="#e74c3c").encode(
    x=alt.X("trade_no:Q", title="累計交易筆數"),
    y=alt.Y("cum_ret:Q", title="累計報酬", axis=alt.Axis(format=".0%")),
    tooltip=[alt.Tooltip("date:T", title="日期"),
             alt.Tooltip("ticker:N", title="股票"),
             alt.Tooltip("cum_ret:Q", format=".1%", title="累計報酬")]
).properties(height=300)

zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
    color="gray", strokeDash=[4,4]
).encode(y="y:Q")

st.altair_chart(line + zero, use_container_width=True)

# ── 參數掃描熱力圖 ────────────────────────────────────────
st.subheader("參數掃描 — Profit Factor 熱力圖")
st.caption("固定量能倍率 = 1.5x、信號窗口 = 60min")
if SWEEP_CSV.exists():
    sweep = load_sweep()
    heat_data = sweep[
        (sweep["量能倍率"] == 1.5) & (sweep["信號窗口"] == 60)
    ].copy()
    heat_data["PF_num"] = pd.to_numeric(
        heat_data["PF"].astype(str).str.replace("inf","99"), errors="coerce"
    ).clip(upper=12)

    hmap = alt.Chart(heat_data).mark_rect().encode(
        x=alt.X("進場回落:O", title="進場回落"),
        y=alt.Y("急拉幅度:O", title="急拉幅度"),
        color=alt.Color("PF_num:Q", scale=alt.Scale(scheme="viridis"),
                        legend=alt.Legend(title="PF")),
        tooltip=[alt.Tooltip("急拉幅度:N"), alt.Tooltip("進場回落:N"),
                 alt.Tooltip("停損距離:N"), alt.Tooltip("樣本數:Q"),
                 alt.Tooltip("勝率:N"), alt.Tooltip("PF:N")]
    ).properties(height=250)

    text = alt.Chart(heat_data).mark_text(fontSize=11).encode(
        x="進場回落:O",
        y="急拉幅度:O",
        text=alt.Text("PF:N"),
        color=alt.condition(
            alt.datum.PF_num > 6, alt.value("black"), alt.value("white")
        )
    )
    st.altair_chart(hmap + text, use_container_width=True)

# ── 近期明細 ─────────────────────────────────────────────
st.subheader("近期交易明細")
recent = (sub[["date","ticker","surge_pct","surge_high","entry_price",
               "exit_price","ret","win",stop_col]]
          .sort_values("date", ascending=False)
          .head(50)
          .copy())
recent.columns = ["日期","股票","急拉幅度","急拉高點","進場","出場","報酬","勝","停損"]
recent["日期"]   = recent["日期"].dt.strftime("%Y-%m-%d")
recent["急拉幅度"] = recent["急拉幅度"].map("{:.1%}".format)
recent["報酬"]   = recent["報酬"].map("{:.2%}".format)

def color_ret(val):
    try:
        v = float(val.replace("%",""))
        return "color:#27ae60;font-weight:700" if v > 0 else "color:#e74c3c"
    except:
        return ""

st.dataframe(
    recent.style.map(color_ret, subset=["報酬"]),
    use_container_width=True, height=400
)

# ── 策略說明 ─────────────────────────────────────────────
with st.expander("策略說明"):
    st.markdown("""
**來源**：童大的交易生活 — 急拉高策略
**類型**：個股當沖空方（日內波）

**信號條件**
- 前 N 分鐘（預設 60min）出現急拉：最高價 > 前收 × (1 + 急拉幅度)
- 該 N 分鐘窗口內成交量 > 20日同窗口均量 × 量能倍率（**非全天量**，
  避免決策時間點不可知的未來資訊）

**進場**：從急拉高點回落 X% 觸發停損市價單做空，成交價估算為觸價
再滑 1 檔（假設當日成交量足夠承接部位；量太薄的股票滑價可能更大）

**出場**
- 停損：高點 × (1 + 停損距離) 觸發後，以下一根 1 分鐘 K 的開盤價
  市價回補（模擬 1 分鐘輪詢延遲的真實滑價，而非精準停損價成交）
- 收盤：13:25 強制平倉

**可交易性過濾**：已排除處置股期間、非資券標的（無法融券/現股當沖先賣）

**核心邏輯**
爆量急拉代表主力可能在出貨。股價推不動後，追高的多方開始套牢。
到下午若仍未過高，當沖多方被迫在尾盤砍倉，形成下殺賣壓。

**成本估算（永豐 2折）**
手續費 0.057% + 交易稅 0.15% ≒ 0.207%/筆

**⚠️ 2026-07-03 修正說明**：舊版信號庫用「當日全天量」篩選（決策時
無法得知的未來資訊）且假設進場精準成交在觸價（現實中限價單會在
高點附近就被動成交，或停損市價單有滑價）。本頁已改用誠實成交模型
重新計算，數字比舊版保守，但更接近真實可執行的績效。**投組層級的
高 Sharpe 尚未經過 tick 級試撮驗證，正式上真金前建議先用 XQ 驗證。**
""")
