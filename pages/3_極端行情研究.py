"""
極端行情研究
台灣加權指數單日大跌後，各持有期間的正報酬機率與報酬分佈。
資料來源：Yahoo Finance ^TWII（2000 年至今）
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="極端行情研究", page_icon="📉", layout="wide")

st.title("📉 極端行情研究")
st.caption("台灣加權指數單日大跌後走勢分析 · 資料來源：Yahoo Finance ^TWII（2000 年至今）· 快取 1 小時")

HORIZONS = [1, 3, 5, 7, 10, 21]


@st.cache_data(ttl=3600)
def load_crash_study(threshold: float):
    try:
        import yfinance as yf
        twii = yf.download("^TWII", start="2000-01-01", progress=False, auto_adjust=True)
        if twii.empty:
            return pd.DataFrame(), pd.DataFrame()
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = twii.columns.get_level_values(0)
        close = twii["Close"].dropna()
        daily_ret = close.pct_change()
        crash_days = daily_ret[daily_ret <= threshold].index

        records = []
        for date in crash_days:
            loc = close.index.get_loc(date)
            row = {
                "日期": date.strftime("%Y-%m-%d"),
                "當日跌幅(%)": round(daily_ret[date] * 100, 2),
                "收盤價": int(round(close[date], 0)),
            }
            for h in HORIZONS:
                future_loc = loc + h
                if future_loc < len(close):
                    fwd = (close.iloc[future_loc] - close[date]) / close[date]
                    row[f"+{h}d(%)"] = round(fwd * 100, 2)
                else:
                    row[f"+{h}d(%)"] = None
            records.append(row)

        if not records:
            return pd.DataFrame(), pd.DataFrame()

        events_df = pd.DataFrame(records).sort_values("日期", ascending=False).reset_index(drop=True)

        stats_rows = []
        for h in HORIZONS:
            col = f"+{h}d(%)"
            vals = events_df[col].dropna()
            if len(vals) == 0:
                continue
            stats_rows.append({
                "持有期間": f"+{h} 交易日",
                "樣本數": int(len(vals)),
                "上漲次數": int((vals > 0).sum()),
                "上漲機率": f"{(vals > 0).mean()*100:.2f}%",
                "平均報酬": f"{vals.mean():.2f}%",
                "中位數": f"{vals.median():.2f}%",
                "最差": f"{vals.min():.2f}%",
                "最佳": f"{vals.max():.2f}%",
                "_mean": vals.mean(),
                "_winrate": (vals > 0).mean() * 100,
            })
        stats_df = pd.DataFrame(stats_rows)
        return events_df, stats_df

    except Exception as e:
        st.error(f"資料載入失敗：{e}")
        return pd.DataFrame(), pd.DataFrame()


# ── 控制列 ───────────────────────────────────────────────────────────────
col_ctrl, col_info = st.columns([1, 3])
with col_ctrl:
    threshold_pct = st.selectbox(
        "觸發條件（單日跌幅）",
        options=[-3, -4, -5, -6, -7, -8, -10],
        index=2,
        format_func=lambda x: f"≤ {x}%",
    )
threshold = threshold_pct / 100.0

events_df, stats_df = load_crash_study(threshold)

if events_df.empty:
    st.warning("資料載入失敗，請稍後重試")
    st.stop()

n_events = len(events_df)
last_event_date = events_df["日期"].iloc[0]
last_drop = events_df["當日跌幅(%)"].iloc[0]

with col_info:
    st.info(
        f"歷史上共 **{n_events}** 次單日跌幅 ≤ {threshold_pct}%　｜　"
        f"最近一次：**{last_event_date}**（{last_drop:+.2f}%）"
    )

# ── 統計摘要 ─────────────────────────────────────────────────────────────
st.markdown("### 各持有期間統計摘要")

display_stats = stats_df.drop(columns=["_mean", "_winrate"], errors="ignore")

def color_prob(val):
    try:
        p = float(str(val).replace("%", ""))
        if p >= 60:
            return "color: #26c281; font-weight: bold"
        elif p <= 40:
            return "color: #e74c3c; font-weight: bold"
    except Exception:
        pass
    return ""

def color_return(val):
    try:
        v = float(str(val).replace("%", ""))
        if v > 0:
            return "color: #26c281"
        elif v < 0:
            return "color: #e74c3c"
    except Exception:
        pass
    return ""

styled = display_stats.style \
    .map(color_prob, subset=["上漲機率"]) \
    .map(color_return, subset=["平均報酬", "中位數", "最差", "最佳"])
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── 平均報酬走勢圖 ────────────────────────────────────────────────────────
st.markdown("### 平均報酬隨持有天數變化")
if "_mean" in stats_df.columns:
    chart_df = stats_df[["持有期間", "_mean"]].rename(
        columns={"_mean": "平均報酬(%)", "持有期間": "期間"}
    ).set_index("期間")
    st.bar_chart(chart_df, height=280)

# ── 報酬分佈（全部 6 期間，2×3 格）────────────────────────────────────────
st.markdown("### 報酬分佈（各持有期間）")
row1 = st.columns(3)
row2 = st.columns(3)
dist_grid = row1 + row2

for i, h in enumerate(HORIZONS):
    col_name = f"+{h}d(%)"
    if col_name not in events_df.columns:
        continue
    vals = events_df[col_name].dropna()
    if vals.empty:
        continue
    bins = pd.cut(vals, bins=12)
    hist = vals.groupby(bins, observed=True).count()
    hist.index = [f"{b.left:.2f}~{b.right:.2f}" for b in hist.index]
    win_rate = (vals > 0).mean() * 100
    mean_val = vals.mean()
    with dist_grid[i]:
        st.markdown(
            f"**+{h} 交易日**　"
            f"<span style='color:{'#26c281' if win_rate>=50 else '#e74c3c'}'>"
            f"上漲 {win_rate:.2f}%</span>　均值 {mean_val:+.2f}%",
            unsafe_allow_html=True,
        )
        st.bar_chart(hist, height=200)

# ── 完整歷史事件表 ─────────────────────────────────────────────────────────
st.markdown("### 完整歷史事件紀錄")
st.caption("按跌幅日期由近到遠排列；綠底 = 正報酬，紅底 = 負報酬")

fwd_cols = [f"+{h}d(%)" for h in HORIZONS if f"+{h}d(%)" in events_df.columns]

def color_fwd(val):
    try:
        v = float(val)
        if v > 0:
            return "background-color: #0d2b1a"
        elif v < 0:
            return "background-color: #2b0d0d"
    except Exception:
        pass
    return ""

fmt_events = {"當日跌幅(%)": "{:+.2f}", "收盤價": "{:,.0f}"}
fmt_events.update({c: "{:+.2f}" for c in fwd_cols})
st.dataframe(
    events_df.style.map(color_fwd, subset=fwd_cols).format(fmt_events, na_rep="-"),
    use_container_width=True,
    hide_index=True,
    height=500,
)

# ── 關鍵結論 ──────────────────────────────────────────────────────────────
st.markdown("### 關鍵結論")

def get_row(label):
    r = stats_df[stats_df["持有期間"] == label]
    return r.iloc[0] if not r.empty else None

r1  = get_row("+1 交易日")
r5  = get_row("+5 交易日")
r10 = get_row("+10 交易日")
r21 = get_row("+21 交易日")

lines = []
if r1 is not None:
    p = float(r1["上漲機率"].replace("%", ""))
    lines.append(
        f"- **隔日（+1d）**：上漲機率 **{r1['上漲機率']}**，平均 **{r1['平均報酬']}**"
        + ("　→ 統計上傾向繼續跌，不宜追殺" if p < 50 else "　→ 統計上有反彈傾向")
    )
if r5 is not None:
    lines.append(f"- **一週（+5d）**：上漲機率 **{r5['上漲機率']}**，平均 **{r5['平均報酬']}**，中位數 **{r5['中位數']}**")
if r10 is not None:
    lines.append(f"- **兩週（+10d）**：上漲機率 **{r10['上漲機率']}**，平均 **{r10['平均報酬']}**")
if r21 is not None:
    p21 = float(r21["上漲機率"].replace("%", ""))
    lines.append(
        f"- **一個月（+21d）**：上漲機率 **{r21['上漲機率']}**，平均 **{r21['平均報酬']}**"
        + ("　→ 長線多半回穩" if p21 >= 55 else "")
    )

if lines:
    st.markdown("\n".join(lines))

st.markdown(
    f"> **操作參考**：跌幅 ≤ {threshold_pct}% 的極端事件後，"
    "隔日往往仍有賣壓（恐慌未消化）；1～2 週後若無新利空，"
    "歷史顯示多數案例出現明顯反彈。現貨部位是否調節，"
    "建議參考隔日成交量與止跌訊號，而非單純依據跌幅決定。"
)
