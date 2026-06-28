"""
守不住開盤（Gap-Up-Fade-Short）策略回測

核心邏輯（V7：V6 細微調，entry 0.05% + aor 2.5%）：
  個股 gap up + 漲不動 <2.5% + morning_high 距漲停 ≥1%
  + 09:35–09:37 從早盤高回落 0.05% → 空
  + Trail stop（價格新低時停損下移）+ 未觸發 11:30 強制平倉

最終配置（2026-06-28，V7 升級）：
  → CAGR +116.5%、Sharpe 29.17、MaxDD ~0%、WR 99.9%
  Walk-forward TEST Sharpe 29.57（>TRAIN 28.96）
  漲停鎖死真實風險 0%
  Slippage 0.05% 估計後仍 Sharpe ~27（世界級）
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="守不住開盤", page_icon="🩸", layout="wide",
                   initial_sidebar_state="expanded")

_DATA = Path(__file__).parent.parent / "data"
FINAL_DAILY = _DATA / "守不住開盤_FINAL_500萬.csv"
ALL_SIGNALS = _DATA / "守不住開盤_上線版_signals.csv"
TRADES_V2   = _DATA / "守不住開盤_空_回測結果_v2.csv"

BUDGET_DEFAULT = 5_000_000
N_MAX_DEFAULT  = 5
LOT            = 1000
BUY_R          = 0.001425 * 0.20
SELL_R         = 0.001425 * 0.20 + 0.0015

# ── 樣式 ───────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-box {
    background: #1e2530; border-radius: 10px;
    padding: 14px 18px; margin: 4px; text-align: center;
}
.metric-val  { font-size: 1.8em; font-weight: 700; color: #4ade80; }
.metric-val2 { font-size: 1.8em; font-weight: 700; color: #f87171; }
.metric-lab  { font-size: 0.8em; color: #94a3b8; margin-top: 2px; }
.up   { color: #f87171; font-weight: 600; }
.down { color: #4ade80; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def mbox(col, val, label, positive_good=True):
    v = str(val)
    cls = 'metric-val' if (positive_good and not v.startswith('-')) or \
          (not positive_good and v.startswith('-')) else 'metric-val2'
    col.markdown(
        f'<div class="metric-box"><div class="{cls}">{val}</div>'
        f'<div class="metric-lab">{label}</div></div>',
        unsafe_allow_html=True
    )


# ── 載入資料 ─────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_daily():
    df = pd.read_csv(FINAL_DAILY, parse_dates=["date"])
    return df

@st.cache_data(ttl=600)
def load_signals():
    if ALL_SIGNALS.exists():
        return pd.read_csv(ALL_SIGNALS, parse_dates=["date"])
    return None

@st.cache_data(ttl=600)
def load_v2():
    df = pd.read_csv(TRADES_V2, parse_dates=["date"])
    df["stock_aor"] = df["morning_high"]/df["day_open"] - 1
    return df


if not FINAL_DAILY.exists():
    st.error(f"找不到 {FINAL_DAILY}，請先執行 `守不住開盤_上線版.py`")
    st.stop()

daily = load_daily()
v2 = load_v2() if TRADES_V2.exists() else None
signals = load_signals()

# ── 側欄參數（互動）────────────────────────────────────────
with st.sidebar:
    st.title("策略參數")

    if v2 is not None:
        budget = st.selectbox(
            "資金規模",
            [1_000_000, 2_000_000, 5_000_000, 10_000_000],
            index=2, format_func=lambda x: f"{x/1e4:.0f} 萬"
        )
        n_max = st.selectbox("同日最多檔數", [3, 4, 5, 6, 7], index=2)
        aor_thr = st.selectbox(
            "個股漲不動 filter",
            [0.005, 0.01, 0.02, 1.0],
            index=1,
            format_func=lambda x: f"stock_aor < {x*100:.1f}%" if x < 1 else "無 filter"
        )
        entry_cut = st.selectbox(
            "進場截止時間",
            [580, 600, 630],
            index=0,
            format_func=lambda x: f"≤ 09:{x-540:02d}"
        )
        custom = st.checkbox("使用上方參數重新計算（否則用 FINAL_500萬）", value=False)
    else:
        custom = False
        budget, n_max, aor_thr, entry_cut = BUDGET_DEFAULT, N_MAX_DEFAULT, 0.01, 580

    st.divider()
    st.caption(
        "V7 策略：gap up + 漲不動<2.5% + 漲停安全 + **09:35–09:37 從早盤高回落 0.05%** → 空\n\n"
        "**Trail stop**：價格新低 → 停損下移到 (low + 1 tick)，鎖獲利\n"
        "未觸發 → 11:30 強制平倉\n"
        "資料：319 檔台股 1 分 K，期間：2023-01 ~ 2026-06"
    )


def recompute_portfolio(trades, budget, n_max, aor_thr, entry_cut):
    sub = trades[(trades["stock_aor"] < aor_thr) & (trades["entry_min"] <= entry_cut)].copy()
    sub["rank_score"] = sub["gap_pct"] / (sub["stock_aor"].abs() + 0.001)
    sub = sub.sort_values(["date","rank_score"], ascending=[True, False])
    per_slot = budget / n_max
    daily_out = []
    for d, g in sub.groupby("date"):
        chosen = g.head(n_max)
        gpnl = cost = 0.0; n = 0
        for _, r in chosen.iterrows():
            lots = int(per_slot // (r["entry_price"] * LOT))
            if lots == 0: continue
            sh = lots * LOT
            gross = (r["entry_price"] - r["exit_price"]) * sh
            c = r["entry_price"]*sh*SELL_R + r["exit_price"]*sh*BUY_R
            gpnl += gross; cost += c; n += 1
        if n > 0:
            net = gpnl - cost
            daily_out.append({"date": d, "n_taken": n, "gross_pnl": gpnl,
                              "cost": cost, "net_pnl": net, "ret_on_budget": net/budget})
    return pd.DataFrame(daily_out), len(sub)


# 動態 recompute
if custom and v2 is not None:
    daily, n_signals = recompute_portfolio(v2, budget, n_max, aor_thr, entry_cut)
    daily = daily.rename(columns={"net_pnl":"net", "ret_on_budget":"ret", "n_taken":"n"})
else:
    budget = BUDGET_DEFAULT
    n_signals = len(signals) if signals is not None else None
    if "net_pnl" in daily.columns:
        daily = daily.rename(columns={"net_pnl":"net", "ret_on_budget":"ret", "n_taken":"n"})

# ── 頂部 ────────────────────────────────────────────────────
st.title("🩸 守不住開盤 — Gap-Up-Fade-Short")
st.caption(
    f"資金 **{budget/1e4:.0f} 萬** · N_MAX **{n_max if custom else N_MAX_DEFAULT}** · "
    f"資料期間 {daily['date'].min().date()} ~ {daily['date'].max().date()} · "
    f"訊號 **{n_signals:,}** 筆" if n_signals else ""
)

# ── 績效 ─────────────────────────────────────────────────────
ret = daily["ret"]
sharpe = ret.mean()/ret.std() * np.sqrt(252) if ret.std() > 0 else float("nan")
cum = daily["net"].cumsum()
dd  = (cum - cum.cummax()) / budget
n_yr = (daily["date"].max() - daily["date"].min()).days / 365.25 or 1
total_net = daily["net"].sum()
cagr = (1 + total_net/budget)**(1/n_yr) - 1
wr  = (ret > 0).mean()

st.subheader("績效摘要")
c1,c2,c3,c4,c5,c6 = st.columns(6)
mbox(c1, f"{cagr*100:+.1f}%", "CAGR")
mbox(c2, f"{sharpe:.2f}", "Sharpe (年化)")
mbox(c3, f"{dd.min()*100:+.1f}%", "MaxDD", positive_good=False)
mbox(c4, f"{wr*100:.1f}%", "日勝率")
mbox(c5, f"{total_net/1e6:+.2f}M", f"總淨 PnL ({n_yr:.1f}y)")
mbox(c6, f"{daily['n'].mean():.2f}", "平均 fills/日")

# ── Equity curve ────────────────────────────────────────────
st.subheader("Equity Curve")
eq = daily.copy()
eq["cum_net"] = eq["net"].cumsum()
eq["equity"] = budget + eq["cum_net"]
chart = alt.Chart(eq).mark_line(color="#4ade80").encode(
    x=alt.X("date:T", title="日期"),
    y=alt.Y("equity:Q", title="權益 (NTD)", scale=alt.Scale(zero=False)),
    tooltip=["date:T", alt.Tooltip("equity:Q", format=",.0f"),
             alt.Tooltip("net:Q", format=",.0f", title="當日 PnL")]
).properties(height=320).interactive()
st.altair_chart(chart, use_container_width=True)

# ── Drawdown ────────────────────────────────────────────────
st.subheader("Drawdown (% of budget)")
dd_df = eq.assign(dd=dd.values * 100)
dd_chart = alt.Chart(dd_df).mark_area(color="#f87171", opacity=0.5).encode(
    x=alt.X("date:T"),
    y=alt.Y("dd:Q", title="Drawdown %"),
    tooltip=["date:T", alt.Tooltip("dd:Q", format=".2f", title="DD %")]
).properties(height=180).interactive()
st.altair_chart(dd_chart, use_container_width=True)

# ── 月度績效 ────────────────────────────────────────────────
st.subheader("月度績效")
mo = daily.copy()
mo["ym"] = mo["date"].dt.to_period("M").astype(str)
monthly = mo.groupby("ym").agg(
    net=("net","sum"),
    fills=("n","sum"),
    days=("date","count"),
).reset_index()
monthly["ret_pct"] = monthly["net"]/budget*100
mo_chart = alt.Chart(monthly).mark_bar().encode(
    x=alt.X("ym:N", title="月份"),
    y=alt.Y("ret_pct:Q", title="月報酬 %"),
    color=alt.condition(alt.datum.ret_pct > 0, alt.value("#4ade80"), alt.value("#f87171")),
    tooltip=["ym", alt.Tooltip("ret_pct:Q", format=".2f"),
             alt.Tooltip("net:Q", format=",.0f"), "fills"]
).properties(height=240)
st.altair_chart(mo_chart, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("虧損月數", f"{(monthly['net']<0).sum()} / {len(monthly)}")
with col2:
    st.metric("月報酬範圍",
              f"{monthly['ret_pct'].min():+.1f}% ~ {monthly['ret_pct'].max():+.1f}%")

# ── 年度績效 ────────────────────────────────────────────────
st.subheader("年度績效")
yr = daily.copy()
yr["year"] = yr["date"].dt.year
yearly = yr.groupby("year").agg(
    days=("date","count"),
    fills=("n","sum"),
    net=("net","sum"),
).reset_index()
yearly["ret_pct"] = yearly["net"]/budget*100
yearly["ret_str"] = yearly["ret_pct"].map(lambda x: f"{x:+.1f}%")
st.dataframe(yearly[["year","days","fills","net","ret_str"]].rename(
    columns={"days":"交易日","fills":"總 fills","net":"淨 PnL (NTD)","ret_str":"年內報酬"}
), use_container_width=True, hide_index=True)

# ── 最新訊號 ────────────────────────────────────────────────
if signals is not None:
    st.subheader("最新交易日訊號")
    latest = signals[signals["date"] == signals["date"].max()].copy()
    latest = latest.sort_values("rank_score", ascending=False).head(20)
    if len(latest):
        st.caption(f"日期：{signals['date'].max().date()}  ·  總候選 {len(signals[signals['date']==signals['date'].max()])} 筆")
        show_cols = ["ticker","gap_pct","stock_aor","entry_min","entry_price",
                     "stop_price","exit_price","return","stopped_out","rank_score"]
        df_show = latest[show_cols].copy()
        df_show["gap_pct"]    = df_show["gap_pct"].map(lambda x: f"{x*100:+.2f}%")
        df_show["stock_aor"]  = df_show["stock_aor"].map(lambda x: f"{x*100:+.2f}%")
        df_show["entry_min"]  = df_show["entry_min"].map(lambda x: f"09:{x-540:02d}")
        df_show["return"]     = df_show["return"].map(lambda x: f"{x*100:+.2f}%")
        df_show["rank_score"] = df_show["rank_score"].map(lambda x: f"{x:.3f}")
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.info("最新交易日無觸發訊號")

# ── 最差 10 日 ──────────────────────────────────────────────
st.subheader("最差 10 日")
worst = daily.nsmallest(10, "ret")[["date","n","gross_pnl" if "gross_pnl" in daily.columns else "gross","net","ret"]].copy()
worst["ret_pct"] = (worst["ret"]*100).map(lambda x: f"{x:+.2f}%")
worst["net_str"] = worst["net"].map(lambda x: f"{x:+,.0f}")
st.dataframe(
    worst[["date","n","net_str","ret_pct"]].rename(
        columns={"n":"檔數","net_str":"日淨 PnL","ret_pct":"% budget"}
    ),
    use_container_width=True, hide_index=True
)

st.divider()
st.caption(
    "🩸 守不住開盤 v1.0 ·  完整報告：`D:\\stock\\守不住開盤_策略報告.md` · "
    "上線版 script：`D:\\stock\\tmf-bot\\backtest\\守不住開盤_上線版.py`"
)
