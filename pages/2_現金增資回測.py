"""
現金增資事件研究 — 除權日前後股價表現
資料來源: FinLab OTC dividend 表 + price:收盤價
預計算 CSV 由 stock/cash_increase_compute.py 產生
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
from datetime import datetime

st.set_page_config(
    page_title="現金增資回測",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e2530;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 4px;
}
.pos { color: #26c281; font-weight: 700; }
.neg { color: #e74c3c; font-weight: 700; }
.neu { color: #95a5a6; }
h1, h2, h3 { color: #e8ecf0; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    avg_win  = pd.read_csv(f"{DATA_DIR}/cash_increase_avg_window.csv")
    summary  = pd.read_csv(f"{DATA_DIR}/cash_increase_summary.csv")
    horizon  = pd.read_csv(f"{DATA_DIR}/cash_increase_horizon.csv",
                           parse_dates=["ex_date"])
    year_df  = pd.read_csv(f"{DATA_DIR}/cash_increase_year.csv")
    events   = pd.read_csv(f"{DATA_DIR}/cash_increase_events.csv",
                           parse_dates=["ex_date"])
    with open(f"{DATA_DIR}/cash_increase_meta.json") as f:
        meta = json.load(f)
    return avg_win, summary, horizon, year_df, events, meta


try:
    avg_win, summary, horizon, year_df, events, meta = load_all()
except FileNotFoundError:
    st.error("找不到預計算資料，請先在本地執行 `python3 stock/cash_increase_compute.py`")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────
st.title("💰 現金增資回測分析")
st.caption(
    f"資料: FinLab 上櫃除權日  ·  "
    f"共 {meta['n_events']} 筆事件  ·  "
    f"{meta['date_start']} ~ {meta['date_end']}  ·  "
    f"更新: {meta['updated_at'][:10]}"
)
st.info(
    "📌 事件日 (Day 0) = **除權日**（除權日當天含機械性稀釋壓力）。"
    "Day -1 為除權前收盤（基準價）。正報酬代表在此基礎上繼續上漲。",
    icon="ℹ️"
)

# ── Filters ──────────────────────────────────────────────────────────────
with st.expander("篩選條件", expanded=False):
    col1, col2, col3 = st.columns(3)
    year_min, year_max = int(horizon["ex_date"].dt.year.min()), int(horizon["ex_date"].dt.year.max())
    sel_years = col1.slider("年份範圍", year_min, year_max, (year_min, year_max))

    disc_min = float(horizon["discount_pct"].quantile(0.05)) if "discount_pct" in horizon.columns else -1.0
    disc_max = float(horizon["discount_pct"].quantile(0.95)) if "discount_pct" in horizon.columns else 0.0
    sel_disc = col2.slider("認購折扣範圍 (vs 前日收盤)", -1.0, 0.0, (disc_min, 0.0), step=0.05,
                           format="%.0f%%", help="負值=認購價低於市價；越負=折扣越深")

    min_events = col3.number_input("最少樣本數", min_value=5, max_value=200, value=10)

# Apply year filter to horizon
h_mask = (
    (horizon["ex_date"].dt.year >= sel_years[0]) &
    (horizon["ex_date"].dt.year <= sel_years[1])
)
if "discount_pct" in horizon.columns:
    h_mask &= (horizon["discount_pct"] >= sel_disc[0]) & (horizon["discount_pct"] <= sel_disc[1])
h_filtered = horizon[h_mask].copy()
n_ev = len(h_filtered)

# Re-compute summary from filtered data
ret_cols = sorted([c for c in h_filtered.columns if c.startswith("ret_")])
abn_cols = sorted([c for c in h_filtered.columns if c.startswith("abn_")])
horizon_vals = [int(c.replace("ret_", "").replace("d", "")) for c in ret_cols]

sum_rows = []
for r_col, a_col, h in zip(ret_cols, abn_cols, horizon_vals):
    s = h_filtered[r_col].dropna()
    a = h_filtered[a_col].dropna()
    if len(s) < min_events:
        continue
    sum_rows.append({
        "horizon": h, "n": len(s),
        "mean_ret": s.mean(), "median_ret": s.median(),
        "win_rate": (s > 0).mean(),
        "mean_abn": a.mean(), "abn_wr": (a > 0).mean(),
        "std_ret": s.std(),
        "pct25": s.quantile(0.25), "pct75": s.quantile(0.75),
    })
f_summary = pd.DataFrame(sum_rows)

# ── KPI Row ───────────────────────────────────────────────────────────────
st.divider()
kpis = st.columns(5)
kpis[0].metric("篩選事件數", f"{n_ev:,}")

def kpi_val(h_val, col="mean_ret"):
    row = f_summary[f_summary["horizon"] == h_val]
    if row.empty:
        return None
    return row.iloc[0][col]

for col_ui, h_val, label in zip(kpis[1:], [5, 10, 20, 60], ["+5日", "+10日", "+20日", "+60日"]):
    v = kpi_val(h_val)
    wr = kpi_val(h_val, "win_rate")
    if v is not None:
        col_ui.metric(f"平均報酬 {label}", f"{v*100:.1f}%", f"勝率 {wr*100:.0f}%",
                      delta_color="normal" if v >= 0 else "inverse")
    else:
        col_ui.metric(f"平均報酬 {label}", "—")

st.divider()

# ── Row 1: CAR curve + Horizon bar ───────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("除權日前後平均累積報酬 (CAR)")

    # Re-compute avg window from filtered data if needed
    # Use avg_win (pre-computed) if year filter is full range, else recompute
    win_full = pd.read_csv(f"{DATA_DIR}/cash_increase_window.csv",
                            parse_dates=["ex_date"])
    win_mask = (
        (win_full["ex_date"].dt.year >= sel_years[0]) &
        (win_full["ex_date"].dt.year <= sel_years[1])
    )
    win_f = win_full[win_mask]
    car_df = win_f.groupby("offset")[["cum_ret", "mkt_ret", "abnormal"]].mean().reset_index()
    car_df = car_df.sort_values("offset")

    import altair as alt

    car_long = pd.melt(car_df, id_vars=["offset"],
                       value_vars=["cum_ret", "mkt_ret", "abnormal"],
                       var_name="series", value_name="return")
    label_map = {"cum_ret": "個股累積報酬", "mkt_ret": "市場累積報酬", "abnormal": "超額報酬 (CAR)"}
    car_long["series"] = car_long["series"].map(label_map)

    color_map = {
        "個股累積報酬": "#4a90d9",
        "市場累積報酬": "#95a5a6",
        "超額報酬 (CAR)": "#e74c3c",
    }

    chart = (
        alt.Chart(car_long)
        .mark_line()
        .encode(
            x=alt.X("offset:Q", title="交易日 (0=除權日)", scale=alt.Scale(domain=[-30, 60])),
            y=alt.Y("return:Q", title="累積報酬", axis=alt.Axis(format=".1%")),
            color=alt.Color("series:N", scale=alt.Scale(
                domain=list(color_map.keys()),
                range=list(color_map.values())
            )),
            tooltip=["offset", alt.Tooltip("return:Q", format=".2%"), "series"],
        )
        .properties(height=360)
    )
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="white", strokeDash=[4, 4], opacity=0.3
    ).encode(y="y:Q")
    event_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="#f6c90e", strokeDash=[4, 4], opacity=0.6
    ).encode(x="x:Q")

    st.altair_chart(chart + zero_line + event_line, use_container_width=True)
    st.caption("基準日 = 除權日前一日收盤價。0=除權日。")

with col_right:
    st.subheader("各期間平均報酬 & 勝率")
    if not f_summary.empty:
        bar_df = f_summary[f_summary["horizon"] > 0].copy()
        bar_df["label"] = bar_df["horizon"].apply(lambda h: f"+{h}d")
        bar_df["color"] = bar_df["mean_ret"].apply(lambda v: "#26c281" if v >= 0 else "#e74c3c")

        bar_chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("label:N", sort=None, title="持有期間"),
                y=alt.Y("mean_ret:Q", title="平均報酬", axis=alt.Axis(format=".1%")),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("label:N", title="期間"),
                    alt.Tooltip("mean_ret:Q", format=".2%", title="平均報酬"),
                    alt.Tooltip("median_ret:Q", format=".2%", title="中位數報酬"),
                    alt.Tooltip("win_rate:Q", format=".0%", title="勝率"),
                    alt.Tooltip("n:Q", title="樣本數"),
                ],
            )
            .properties(height=180)
        )
        wr_chart = (
            alt.Chart(bar_df)
            .mark_bar(color="#4a90d9")
            .encode(
                x=alt.X("label:N", sort=None, title=""),
                y=alt.Y("win_rate:Q", title="勝率", axis=alt.Axis(format=".0%"),
                        scale=alt.Scale(domain=[0, 1])),
                tooltip=["label:N", alt.Tooltip("win_rate:Q", format=".0%")],
            )
            .properties(height=150)
        )
        fifty_line = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
            color="white", strokeDash=[4, 4], opacity=0.4
        ).encode(y="y:Q")

        st.altair_chart(bar_chart, use_container_width=True)
        st.altair_chart(wr_chart + fifty_line, use_container_width=True)
    else:
        st.warning("篩選後樣本數不足")

# ── Row 2: Distribution + Year heatmap ───────────────────────────────────
st.divider()
col_a, col_b = st.columns([2, 3])

with col_a:
    st.subheader("報酬分佈")
    sel_horizon = st.selectbox(
        "選擇期間",
        options=[h for h in [1, 5, 10, 20, 30, 60] if f"ret_{h:+d}d" in h_filtered.columns],
        format_func=lambda h: f"+{h} 交易日",
        index=2
    )
    col_name = f"ret_{sel_horizon:+d}d"
    dist_data = h_filtered[col_name].dropna()
    # Clip extreme outliers for display
    lo, hi = dist_data.quantile(0.02), dist_data.quantile(0.98)
    dist_clipped = dist_data.clip(lo, hi)
    dist_df = dist_clipped.to_frame("ret_val")

    hist = (
        alt.Chart(dist_df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("ret_val:Q", bin=alt.Bin(maxbins=40), title=f"+{sel_horizon}日報酬",
                    axis=alt.Axis(format=".0%")),
            y=alt.Y("count():Q", title="頻次"),
            color=alt.condition(
                alt.datum.ret_val >= 0,
                alt.value("#26c281"),
                alt.value("#e74c3c")
            )
        )
        .properties(height=300)
    )
    mean_line = alt.Chart(pd.DataFrame({"x": [dist_data.mean()]})).mark_rule(
        color="#f6c90e", strokeDash=[5, 3]
    ).encode(x="x:Q")

    st.altair_chart(hist + mean_line, use_container_width=True)

    mean_v = dist_data.mean()
    med_v  = dist_data.median()
    wr_v   = (dist_data > 0).mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("平均", f"{mean_v*100:.1f}%")
    c2.metric("中位數", f"{med_v*100:.1f}%")
    c3.metric("勝率", f"{wr_v*100:.0f}%")

with col_b:
    st.subheader("年度 × 期間 平均報酬熱力圖")
    yr_pivot = year_df[year_df["horizon"] > 0].pivot_table(
        index="year", columns="horizon", values="mean_ret"
    )
    if not yr_pivot.empty:
        # Melt for altair
        yr_melt = yr_pivot.reset_index().melt(
            id_vars="year", var_name="horizon", value_name="mean_ret"
        ).dropna()
        yr_melt["label"] = yr_melt["mean_ret"].apply(lambda v: f"{v*100:.1f}%")
        yr_melt["horizon_label"] = yr_melt["horizon"].apply(lambda h: f"+{h}d")

        heatmap = (
            alt.Chart(yr_melt)
            .mark_rect()
            .encode(
                x=alt.X("horizon_label:N",
                         sort=[f"+{h}d" for h in sorted(yr_melt["horizon"].unique())],
                         title="持有期間"),
                y=alt.Y("year:O", title="年份"),
                color=alt.Color("mean_ret:Q",
                                scale=alt.Scale(scheme="rdylgn", domainMid=0),
                                title="平均報酬"),
                tooltip=[
                    alt.Tooltip("year:O"),
                    alt.Tooltip("horizon_label:N", title="期間"),
                    alt.Tooltip("mean_ret:Q", format=".1%", title="平均報酬"),
                ],
            )
        )
        text_layer = (
            alt.Chart(yr_melt)
            .mark_text(fontSize=9)
            .encode(
                x=alt.X("horizon_label:N",
                         sort=[f"+{h}d" for h in sorted(yr_melt["horizon"].unique())]),
                y=alt.Y("year:O"),
                text=alt.Text("label:N"),
                color=alt.condition(
                    "datum.mean_ret < -0.05 | datum.mean_ret > 0.05",
                    alt.value("white"), alt.value("#333")
                )
            )
        )
        st.altair_chart((heatmap + text_layer).properties(height=400),
                        use_container_width=True)

# ── Row 3: Discount effect ────────────────────────────────────────────────
if "discount_pct" in h_filtered.columns and h_filtered["discount_pct"].notna().sum() > 10:
    st.divider()
    st.subheader("認購折扣 vs 後續報酬")

    disc_data = h_filtered[["discount_pct", "ret_+20d", "ret_+60d"]].dropna()
    if len(disc_data) > 20:
        col_d1, col_d2 = st.columns(2)
        for col_ui, ret_col, label in zip([col_d1, col_d2],
                                           ["ret_+20d", "ret_+60d"],
                                           ["+20日報酬", "+60日報酬"]):
            if ret_col not in disc_data.columns:
                continue
            scatter_df = disc_data[["discount_pct", ret_col]].rename(
                columns={ret_col: "ret"})
            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(opacity=0.5, size=40)
                .encode(
                    x=alt.X("discount_pct:Q", title="認購折扣 (vs 前日收盤)",
                             axis=alt.Axis(format=".0%")),
                    y=alt.Y("ret:Q", title=label, axis=alt.Axis(format=".0%")),
                    color=alt.condition(
                        alt.datum.ret >= 0,
                        alt.value("#26c281"), alt.value("#e74c3c")
                    ),
                    tooltip=[
                        alt.Tooltip("discount_pct:Q", format=".1%", title="折扣"),
                        alt.Tooltip("ret:Q", format=".1%", title=label),
                    ]
                )
                .properties(height=280, title=f"認購折扣 vs {label}")
            )
            col_ui.altair_chart(scatter, use_container_width=True)

# ── Row 4: Event table ────────────────────────────────────────────────────
st.divider()
st.subheader("個別事件明細")

display_cols = ["ticker", "ex_date", "sub_price", "ref_price", "discount_pct",
                "ret_+1d", "ret_+5d", "ret_+10d", "ret_+20d", "ret_+60d"]
disp_df = h_filtered[[c for c in display_cols if c in h_filtered.columns]].copy()
disp_df = disp_df.sort_values("ex_date", ascending=False)

# Format
pct_cols = [c for c in disp_df.columns if c.startswith("ret_") or c == "discount_pct"]
for c in pct_cols:
    disp_df[c] = disp_df[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
if "ex_date" in disp_df.columns:
    disp_df["ex_date"] = disp_df["ex_date"].dt.strftime("%Y-%m-%d")
if "sub_price" in disp_df.columns:
    disp_df["sub_price"] = disp_df["sub_price"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
if "ref_price" in disp_df.columns:
    disp_df["ref_price"] = disp_df["ref_price"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—")

disp_df.columns = [
    c.replace("ret_", "").replace("discount_pct", "折扣").replace("ex_date", "除權日")
     .replace("ticker", "股票").replace("sub_price", "認購價").replace("ref_price", "前日收")
    for c in disp_df.columns
]

st.dataframe(disp_df, use_container_width=True, height=400)
