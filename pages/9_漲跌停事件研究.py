"""
漲停 / 跌停 事件研究
當日收盤漲跌幅 >=9.5% 視為漲停/跌停，之後隔天到 20 個交易日的報酬與勝率統計。
資料由 D:\\stock\\stock\\limitup_limitdown_study\\event_study.py 產生 (2015-01-01起)。
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="漲跌停事件研究", page_icon="🔺", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "data" / "limitup_study"

UP_COLOR = "#f87171"    # 紅 = 漲停（台股慣例：紅漲綠跌）
DOWN_COLOR = "#4ade80"  # 綠 = 跌停

HORIZON_COLS = ["gap_next_pct", "fwd1_pct", "fwd2_pct", "fwd3_pct", "fwd5_pct", "fwd10_pct", "fwd20_pct"]
HORIZON_LABELS = ["隔日跳空", "+1日", "+2日", "+3日", "+5日", "+10日", "+20日"]

CAP_ORDER = ["A_大型(>=500億)", "B_中型(100-500億)", "C_中小型(50-100億)", "D_小型(<50億)"]
PRICE_ORDER = ["A_<20元", "B_20-50元", "C_50-100元", "D_100-200元", "E_200-500元", "F_>=500元"]


DATA_VERSION = "v2-in_disposal"  # 手動bump：資料schema變了但函式原始碼沒變時，強制cache失效


@st.cache_data(ttl=3600)
def load_events(_version):
    up = pd.read_csv(DATA_DIR / "events_up.csv", parse_dates=["date"], dtype={"code": str})
    down = pd.read_csv(DATA_DIR / "events_down.csv", parse_dates=["date"], dtype={"code": str})
    for df in (up, down):
        if "in_disposal" not in df.columns:
            df["in_disposal"] = False
        else:
            df["in_disposal"] = df["in_disposal"].astype(bool)
    return up, down


try:
    up_all, down_all = load_events(DATA_VERSION)
except FileNotFoundError:
    st.error("找不到事件資料，請確認 data/limitup_study/events_up.csv 與 events_down.csv 存在")
    st.stop()

st.title("🔺 漲停 / 跌停 事件研究")
st.caption(
    f"上市＋上櫃個股・收盤漲跌幅 ≥9.5% 視為漲停/跌停・"
    f"樣本 {up_all['date'].min().date()} ~ {up_all['date'].max().date()}・"
    f"漲停 {len(up_all):,} 筆／跌停 {len(down_all):,} 筆・裸報酬未扣費"
)

st.warning(
    "⚠️ **倖存者偏誤**：樣本只涵蓋目前仍在上市/上櫃名單的公司，曾下市/下櫃個股的歷史事件"
    "被結構性排除。跌停研究「20日後多半反彈」的結論可能因此偏樂觀，連續跌停3天以上的解讀"
    "務必保守。報酬未計手續費、證交稅，且未必能在漲跌停價位實際成交（尤其鎖死時對手單很少）。"
    "本頁為描述性事件研究，非投資建議。"
)

with st.sidebar:
    st.subheader("篩選")
    yr_min = int(up_all["date"].dt.year.min())
    yr_max = int(up_all["date"].dt.year.max())
    yr_range = st.slider("年份範圍", yr_min, yr_max, (yr_min, yr_max))
    only_locked = st.checkbox("只看鎖死事件（收盤=當日最高/最低價）", value=False)
    exclude_disposal = st.checkbox("排除處置期間的漲跌停", value=False,
                                    help="處置股改人工管制撮合(5分/20分)，成交機制跟一般盤中不同，"
                                         "統計特性也不太一樣（漲停通常更強、跌停初期更弱但長線彈更兇）")
    st.divider()
    st.caption(
        "方法：市值＝事件當天收盤價×**目前**股本（無歷史股本序列，早期事件分類會略失真）。"
        "「鎖死」＝收盤價等於當日最高價(漲停)/最低價(跌停)，代表全日封住、盤中沒有對手單。"
        "「處置期間」＝事件當天該股票正處於任一次處置(第一次/第二次)公告的起訖區間內。"
    )


def apply_filters(df):
    d = df[(df["date"].dt.year >= yr_range[0]) & (df["date"].dt.year <= yr_range[1])]
    if only_locked:
        d = d[d["locked"] == True]
    if exclude_disposal:
        d = d[d["in_disposal"] == False]
    return d


up = apply_filters(up_all)
down = apply_filters(down_all)


def horizon_stats(df):
    rows = []
    for col, lbl in zip(HORIZON_COLS, HORIZON_LABELS):
        v = df[col].dropna()
        if len(v) == 0:
            continue
        rows.append({"期間": lbl, "平均報酬%": v.mean(), "勝率%": (v > 0).mean() * 100, "樣本數": len(v)})
    return pd.DataFrame(rows)


def group_summary(df, group_col, order=None, min_n=5):
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for gname, g in df.groupby(group_col):
        if len(g) < min_n:
            continue
        row = {"分組": gname, "n": len(g)}
        for col, short in [("gap_next_pct", "gap"), ("fwd1_pct", "f1"), ("fwd5_pct", "f5"), ("fwd20_pct", "f20")]:
            v = g[col].dropna()
            row[f"{short}_mean"] = v.mean() if len(v) else np.nan
            row[f"{short}_win"] = (v > 0).mean() * 100 if len(v) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if order is not None and not out.empty:
        out["_ord"] = out["分組"].apply(lambda x: order.index(x) if x in order else 999)
        out = out.sort_values("_ord").drop(columns="_ord")
    return out


def style_group_table(df):
    if df.empty:
        st.info("篩選後樣本不足")
        return
    disp = df.rename(columns={
        "n": "n", "gap_mean": "隔日跳空", "gap_win": "跳空勝率",
        "f1_mean": "+1日", "f1_win": "+1日勝率",
        "f5_mean": "+5日", "f5_win": "+5日勝率",
        "f20_mean": "+20日", "f20_win": "+20日勝率",
    })
    mean_cols = ["隔日跳空", "+1日", "+5日", "+20日"]
    win_cols = ["跳空勝率", "+1日勝率", "+5日勝率", "+20日勝率"]

    def color_ret(v):
        if pd.isna(v):
            return ""
        return f"color: {UP_COLOR}" if v > 0 else (f"color: {DOWN_COLOR}" if v < 0 else "")

    fmt = {c: "{:+.2f}%" for c in mean_cols}
    fmt.update({c: "{:.2f}%" for c in win_cols})
    fmt["n"] = "{:,}"
    styled = disp.style.map(color_ret, subset=mean_cols).format(fmt, na_rep="—")
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_direction(df, df_full, color, direction_label):
    if df.empty:
        st.info("篩選後沒有資料")
        return
    n = len(df)
    fwd20 = df["fwd20_pct"].dropna()
    gap = df["gap_next_pct"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("事件數", f"{n:,}")
    c2.metric("隔日跳空平均", f"{gap.mean():+.2f}%")
    c3.metric("+20日平均報酬", f"{fwd20.mean():+.2f}%")
    c4.metric("+20日勝率", f"{(fwd20 > 0).mean() * 100:.2f}%")

    stats = horizon_stats(df)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 平均報酬隨時間演變")
        chart = alt.Chart(stats).mark_bar(color=color, size=26).encode(
            x=alt.X("期間:N", sort=HORIZON_LABELS, title=None),
            y=alt.Y("平均報酬%:Q", title="平均報酬 (%)"),
            tooltip=[alt.Tooltip("期間:N"), alt.Tooltip("平均報酬%:Q", format="+.2f"),
                     alt.Tooltip("樣本數:Q", format=",")],
        ).properties(height=260)
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#94a3b8").encode(y="y:Q")
        st.altair_chart(chart + zero, use_container_width=True)
    with col_b:
        st.markdown("##### 勝率隨時間演變")
        base = alt.Chart(stats).mark_bar(color=color, size=26).encode(
            x=alt.X("期間:N", sort=HORIZON_LABELS, title=None),
            y=alt.Y("勝率%:Q", scale=alt.Scale(domain=[0, 100]), title="正報酬機率 (%)"),
            tooltip=[alt.Tooltip("期間:N"), alt.Tooltip("勝率%:Q", format=".2f")],
        ).properties(height=260)
        rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(y="y:Q")
        st.altair_chart(base + rule, use_container_width=True)

    st.markdown("##### 鎖死 vs 未鎖死")
    st.caption("沒鎖死＝收盤未達當日最高/最低價，代表尾盤已有反向對手單進場")
    style_group_table(group_summary(df, "locked").assign(
        分組=lambda d: d["分組"].map({True: "鎖死", False: "未鎖死"})))

    st.markdown("##### 處置期間 vs 非處置期間")
    st.caption("處置期間改人工管制撮合(5分/20分一次)，成交機制跟一般盤中不同")
    style_group_table(group_summary(df, "in_disposal").assign(
        分組=lambda d: d["分組"].map({True: "處置期間", False: "非處置期間"})))

    st.markdown("##### 連續天數效應")
    st.caption("streak = 連續第幾天漲停/跌停（1=首日）")
    df2 = df.copy()
    df2["streak_bucket"] = df2["streak"].apply(lambda s: str(int(s)) if s <= 4 else "5+")
    style_group_table(group_summary(df2, "streak_bucket", order=["1", "2", "3", "4", "5+"]))

    st.markdown("##### 市值分組")
    style_group_table(group_summary(df, "cap_bucket", order=CAP_ORDER))

    st.markdown("##### 股價分組")
    style_group_table(group_summary(df, "price_bucket", order=PRICE_ORDER))

    st.markdown("##### 產業別（樣本數 ≥ 200 才列出）")
    ind = group_summary(df, "industry", min_n=200).sort_values("n", ascending=False)
    style_group_table(ind)

    st.markdown("##### 市場別")
    style_group_table(group_summary(df, "market"))

    with st.expander("🔍 查特定股票的歷史事件"):
        code_q = st.text_input(f"輸入股票代號（{direction_label}）", key=f"code_{direction_label}")
        if code_q:
            sub = df_full[df_full["code"] == code_q.strip()].sort_values("date", ascending=False)
            if sub.empty:
                st.caption("查無事件")
            else:
                show_cols = ["date", "event_ret_pct", "streak", "locked", "close_px", "cap_bucket", "industry",
                             "gap_next_pct", "fwd1_pct", "fwd5_pct", "fwd10_pct", "fwd20_pct"]
                st.dataframe(sub[show_cols].round(2), use_container_width=True, hide_index=True)


tab_up, tab_down = st.tabs(["📈 漲停", "📉 跌停"])
with tab_up:
    render_direction(up, up_all, UP_COLOR, "漲停")
with tab_down:
    render_direction(down, down_all, DOWN_COLOR, "跌停")

st.divider()
st.caption(
    "原始回測程式：D:\\stock\\stock\\limitup_limitdown_study\\event_study.py　"
    "逐筆事件與分組彙總 CSV 存於同目錄。"
)
