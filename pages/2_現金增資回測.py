"""
現金增資事件研究 — 除權日前後股價表現
資料來源: FinLab OTC dividend 表 + price:收盤價
預計算 CSV 由 stock/cash_increase_compute.py 產生
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
import altair as alt

st.set_page_config(
    page_title="現金增資回測",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.pos { color: #26c281; font-weight: 700; }
.neg { color: #e74c3c; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FINE_DAYS = list(range(-10, 11))          # -10 ~ +10
SUMMARY_HORIZONS = [-20, 5, 10, 20, 30, 60]

# ── Load ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all():
    avg_win  = pd.read_csv(f"{DATA_DIR}/cash_increase_avg_window.csv")
    horizon  = pd.read_csv(f"{DATA_DIR}/cash_increase_horizon.csv",
                           parse_dates=["ex_date"])
    year_df  = pd.read_csv(f"{DATA_DIR}/cash_increase_year.csv")
    win_full = pd.read_csv(f"{DATA_DIR}/cash_increase_window.csv",
                           parse_dates=["ex_date"])
    with open(f"{DATA_DIR}/cash_increase_meta.json") as f:
        meta = json.load(f)
    return avg_win, horizon, year_df, win_full, meta

try:
    avg_win, horizon, year_df, win_full, meta = load_all()
except FileNotFoundError:
    st.error("找不到預計算資料，請先執行 `python3 stock/cash_increase_compute.py`")
    st.stop()

# ── Sidebar filter ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("篩選條件")
    yr_min = int(horizon["ex_date"].dt.year.min())
    yr_max = int(horizon["ex_date"].dt.year.max())
    sel_years = st.slider("年份範圍", yr_min, yr_max, (yr_min, yr_max))

    disc_vals = horizon["discount_pct"].dropna()
    disc_lo = float(disc_vals.quantile(0.05))
    disc_hi = float(disc_vals.quantile(0.95))
    sel_disc = st.slider(
        "認購折扣範圍 (vs 前日收盤)",
        min_value=-1.0, max_value=0.3,
        value=(disc_lo, disc_hi),
        step=0.01,
        format="%.2f",
        help="負值=認購價低於市價"
    )

# Apply filters
mask = (
    (horizon["ex_date"].dt.year >= sel_years[0]) &
    (horizon["ex_date"].dt.year <= sel_years[1]) &
    (horizon["discount_pct"].fillna(-999) >= sel_disc[0]) &
    (horizon["discount_pct"].fillna(999)  <= sel_disc[1])
)
h = horizon[mask].copy()
n_ev = len(h)

win_mask = (
    (win_full["ex_date"].dt.year >= sel_years[0]) &
    (win_full["ex_date"].dt.year <= sel_years[1])
)
win = win_full[win_mask].copy()

# ── Header ────────────────────────────────────────────────────────────────
st.title("💰 現金增資回測分析")
st.caption(
    f"資料: FinLab 上櫃除權日  ·  {meta['n_events']} 筆事件  ·  "
    f"{meta['date_start']} ~ {meta['date_end']}  ·  更新: {meta['updated_at'][:10]}"
)
st.info(
    "**Day 0 = 除權日**（含機械性稀釋壓力）。Day -1 為基準日（除權前收盤）。"
    "正報酬代表股價高於基準日。",
    icon="ℹ️"
)

# ── KPIs ──────────────────────────────────────────────────────────────────
def get_kpi(df, col):
    s = df[col].dropna()
    return s.mean() if len(s) > 0 else None

k0, k1, k2, k3, k4 = st.columns(5)
k0.metric("篩選事件數", f"{n_ev:,}")
for col_ui, day, label in zip([k1,k2,k3,k4], [5,10,20,60], ["+5日","+10日","+20日","+60日"]):
    col_name = f"ret_{day:+d}d"
    if col_name in h.columns:
        v  = get_kpi(h, col_name)
        wr = (h[col_name].dropna() > 0).mean()
        if v is not None:
            col_ui.metric(f"均報酬 {label}", f"{v*100:.1f}%", f"勝率 {wr*100:.0f}%",
                          delta_color="normal" if v >= 0 else "inverse")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 價格走勢", "📊 日別細節 -10~+10", "💲 認購價分析", "📋 歷史紀錄"]
)

# ── Tab 1: CAR Curve + Horizon summary ───────────────────────────────────
with tab1:
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.subheader("除權日前後平均累積報酬 (CAR)")
        car = win.groupby("offset")[["cum_ret","mkt_ret","abnormal"]].mean().reset_index()
        car = car.sort_values("offset")
        car_long = car.melt(id_vars="offset",
                            value_vars=["cum_ret","mkt_ret","abnormal"],
                            var_name="series", value_name="ret")
        label_map = {"cum_ret":"個股累積報酬","mkt_ret":"市場累積報酬","abnormal":"超額報酬(CAR)"}
        car_long["series"] = car_long["series"].map(label_map)

        chart = (
            alt.Chart(car_long)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("offset:Q", title="交易日 (0=除權日)",
                         scale=alt.Scale(domain=[-30, 60])),
                y=alt.Y("ret:Q", title="累積報酬",
                         axis=alt.Axis(format=".1%")),
                color=alt.Color("series:N",
                    scale=alt.Scale(
                        domain=["個股累積報酬","市場累積報酬","超額報酬(CAR)"],
                        range=["#4a90d9","#95a5a6","#e74c3c"]
                    ),
                    legend=alt.Legend(orient="top")
                ),
                tooltip=[
                    alt.Tooltip("offset:Q", title="Day"),
                    alt.Tooltip("ret:Q", format=".2%", title="報酬"),
                    "series:N"
                ]
            )
            .properties(height=360)
        )
        zero  = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(
            color="white", strokeDash=[4,4], opacity=0.3).encode(y="y:Q")
        event = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(
            color="#f6c90e", strokeDash=[4,4], opacity=0.6).encode(x="x:Q")
        st.altair_chart(chart + zero + event, use_container_width=True)

    with c_right:
        st.subheader("各期間平均報酬 & 勝率")
        sum_rows = []
        for day in SUMMARY_HORIZONS:
            col_r = f"ret_{day:+d}d"
            col_a = f"abn_{day:+d}d"
            if col_r not in h.columns:
                continue
            s = h[col_r].dropna()
            a = h[col_a].dropna() if col_a in h.columns else pd.Series([], dtype=float)
            if len(s) < 5:
                continue
            sum_rows.append({
                "label": f"+{day}d",
                "day": day,
                "mean_ret": s.mean(),
                "win_rate": (s > 0).mean(),
                "n": len(s),
            })
        f_sum = pd.DataFrame(sum_rows)

        if not f_sum.empty:
            bar = (
                alt.Chart(f_sum)
                .mark_bar()
                .encode(
                    x=alt.X("label:N", sort=None, title="期間"),
                    y=alt.Y("mean_ret:Q", title="平均報酬",
                             axis=alt.Axis(format=".1%")),
                    color=alt.Color("mean_ret:Q",
                        scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("label:N", title="期間"),
                        alt.Tooltip("mean_ret:Q", format=".2%", title="平均報酬"),
                        alt.Tooltip("win_rate:Q", format=".0%", title="勝率"),
                        alt.Tooltip("n:Q", title="樣本數"),
                    ]
                )
                .properties(height=185)
            )
            wr_bar = (
                alt.Chart(f_sum)
                .mark_bar(color="#4a90d9")
                .encode(
                    x=alt.X("label:N", sort=None, title=""),
                    y=alt.Y("win_rate:Q", title="勝率",
                             axis=alt.Axis(format=".0%"),
                             scale=alt.Scale(domain=[0,1])),
                    tooltip=[alt.Tooltip("win_rate:Q", format=".0%")]
                )
                .properties(height=145)
            )
            fifty = alt.Chart(pd.DataFrame({"y":[0.5]})).mark_rule(
                color="white", strokeDash=[4,4], opacity=0.4).encode(y="y:Q")
            st.altair_chart(bar, use_container_width=True)
            st.altair_chart(wr_bar + fifty, use_container_width=True)

    # Year heatmap
    st.subheader("年度 × 期間 平均報酬熱力圖")
    yr_f = year_df[
        (year_df["year"] >= sel_years[0]) &
        (year_df["year"] <= sel_years[1]) &
        (year_df["horizon"] > 0)
    ].copy()
    yr_f = yr_f[yr_f["horizon"].isin(SUMMARY_HORIZONS + [1, 10])]
    yr_f["hlabel"] = yr_f["horizon"].apply(lambda x: f"+{x}d")
    yr_f["text"]   = yr_f["mean_ret"].apply(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")

    if not yr_f.empty:
        hmap = (
            alt.Chart(yr_f)
            .mark_rect()
            .encode(
                x=alt.X("hlabel:N",
                         sort=[f"+{d}d" for d in sorted(yr_f["horizon"].unique())],
                         title="持有期間"),
                y=alt.Y("year:O", title="年份"),
                color=alt.Color("mean_ret:Q",
                    scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                    title="平均報酬"),
                tooltip=[
                    alt.Tooltip("year:O"),
                    alt.Tooltip("hlabel:N", title="期間"),
                    alt.Tooltip("mean_ret:Q", format=".1%", title="平均報酬"),
                    alt.Tooltip("n:Q", title="樣本數"),
                ]
            )
        )
        text_layer = (
            alt.Chart(yr_f)
            .mark_text(fontSize=9, color="white")
            .encode(
                x=alt.X("hlabel:N",
                         sort=[f"+{d}d" for d in sorted(yr_f["horizon"].unique())]),
                y=alt.Y("year:O"),
                text=alt.Text("text:N"),
            )
        )
        st.altair_chart((hmap + text_layer).properties(height=420),
                        use_container_width=True)


# ── Tab 2: Day-by-day -10 to +10 ─────────────────────────────────────────
with tab2:
    st.subheader("除權日前後 -10 ~ +10 每日報酬明細")

    # Build per-day stats from horizon CSV
    day_rows = []
    for day in range(-10, 11):
        col_r = f"ret_{day:+d}d"
        col_a = f"abn_{day:+d}d"
        if col_r not in h.columns:
            continue
        s = h[col_r].dropna()
        a = h[col_a].dropna() if col_a in h.columns else pd.Series([], dtype=float)
        if len(s) == 0:
            continue
        day_rows.append({
            "day": day,
            "n": len(s),
            "mean_ret": s.mean(),
            "median_ret": s.median(),
            "win_rate": (s > 0).mean(),
            "mean_abn": a.mean() if len(a) > 0 else np.nan,
            "std": s.std(),
            "p25": s.quantile(0.25),
            "p75": s.quantile(0.75),
        })
    day_df = pd.DataFrame(day_rows)

    # Chart: mean_ret bar with ±1 std band
    bar_day = (
        alt.Chart(day_df)
        .mark_bar(width=18)
        .encode(
            x=alt.X("day:Q", title="交易日 (0=除權日)",
                     scale=alt.Scale(domain=[-10.5, 10.5]),
                     axis=alt.Axis(tickCount=21)),
            y=alt.Y("mean_ret:Q", title="平均累積報酬",
                     axis=alt.Axis(format=".1%")),
            color=alt.condition(
                alt.datum.mean_ret >= 0,
                alt.value("#26c281"),
                alt.value("#e74c3c")
            ),
            tooltip=[
                alt.Tooltip("day:Q", title="Day"),
                alt.Tooltip("mean_ret:Q", format=".2%", title="平均報酬"),
                alt.Tooltip("median_ret:Q", format=".2%", title="中位數"),
                alt.Tooltip("win_rate:Q", format=".0%", title="勝率"),
                alt.Tooltip("n:Q", title="樣本數"),
            ]
        )
        .properties(height=280, title="每日平均累積報酬 (基準 = 除權前一日收盤)")
    )
    zero_rule = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(
        color="white", strokeDash=[4,4], opacity=0.4).encode(y="y:Q")
    ev_rule = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(
        color="#f6c90e", opacity=0.5).encode(x="x:Q")
    st.altair_chart(bar_day + zero_rule + ev_rule, use_container_width=True)

    # Win rate line
    wr_day = (
        alt.Chart(day_df)
        .mark_line(point=True, color="#4a90d9", strokeWidth=2)
        .encode(
            x=alt.X("day:Q", title="交易日 (0=除權日)",
                     scale=alt.Scale(domain=[-10.5, 10.5])),
            y=alt.Y("win_rate:Q", title="勝率",
                     axis=alt.Axis(format=".0%"),
                     scale=alt.Scale(domain=[0,1])),
            tooltip=[
                alt.Tooltip("day:Q", title="Day"),
                alt.Tooltip("win_rate:Q", format=".0%", title="勝率"),
            ]
        )
        .properties(height=200, title="每日勝率")
    )
    fifty_r = alt.Chart(pd.DataFrame({"y":[0.5]})).mark_rule(
        color="white", strokeDash=[4,4], opacity=0.4).encode(y="y:Q")
    st.altair_chart(wr_day + fifty_r + ev_rule, use_container_width=True)

    # Table
    st.subheader("日別統計表")
    tbl = day_df.copy()
    tbl["mean_ret"]   = tbl["mean_ret"].apply(lambda x: f"{x*100:.2f}%")
    tbl["median_ret"] = tbl["median_ret"].apply(lambda x: f"{x*100:.2f}%")
    tbl["win_rate"]   = tbl["win_rate"].apply(lambda x: f"{x*100:.1f}%")
    tbl["mean_abn"]   = tbl["mean_abn"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    tbl["std"]        = tbl["std"].apply(lambda x: f"{x*100:.2f}%")
    tbl["p25"]        = tbl["p25"].apply(lambda x: f"{x*100:.2f}%")
    tbl["p75"]        = tbl["p75"].apply(lambda x: f"{x*100:.2f}%")
    tbl.columns       = ["Day","樣本","平均報酬","中位數","勝率","超額報酬(vs市場)","標準差","25%","75%"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ── Tab 3: Subscription Price Analysis ───────────────────────────────────
with tab3:
    st.subheader("股價 vs 認購價分析")

    disc_data = h[h["discount_pct"].notna()].copy()
    st.caption(f"共 {len(disc_data)} 筆有認購價資料的事件")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**認購折扣分佈**（認購價 / 除權前收盤 - 1）")
        dd = disc_data["discount_pct"].clip(-1, 0.5)
        dist_df = dd.to_frame("disc")
        hist_disc = (
            alt.Chart(dist_df)
            .mark_bar(opacity=0.8, color="#4a90d9")
            .encode(
                x=alt.X("disc:Q", bin=alt.Bin(maxbins=40),
                         title="折扣比例", axis=alt.Axis(format=".0%")),
                y=alt.Y("count():Q", title="頻次"),
            )
            .properties(height=280)
        )
        mean_disc = alt.Chart(pd.DataFrame({"x":[disc_data["discount_pct"].mean()]})).mark_rule(
            color="#f6c90e", strokeDash=[5,3]).encode(x="x:Q")
        st.altair_chart(hist_disc + mean_disc, use_container_width=True)
        st.metric("平均折扣", f"{disc_data['discount_pct'].mean()*100:.1f}%")
        st.metric("中位數折扣", f"{disc_data['discount_pct'].median()*100:.1f}%")

    with col_b:
        st.markdown("**折扣深度 vs +20日報酬**")
        if "ret_+20d" in disc_data.columns:
            sc_df = disc_data[["discount_pct","ret_+20d","ret_+60d"]].dropna()
            scatter = (
                alt.Chart(sc_df)
                .mark_circle(opacity=0.5, size=45)
                .encode(
                    x=alt.X("discount_pct:Q", title="認購折扣",
                             axis=alt.Axis(format=".0%")),
                    y=alt.Y("ret_+20d:Q", title="+20日報酬",
                             axis=alt.Axis(format=".0%")),
                    color=alt.Color("ret_+20d:Q",
                        scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("discount_pct:Q", format=".1%", title="折扣"),
                        alt.Tooltip("ret_+20d:Q", format=".1%", title="+20日"),
                        alt.Tooltip("ret_+60d:Q", format=".1%", title="+60日"),
                    ]
                )
                .properties(height=280)
            )
            h_rule = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(
                color="white", strokeDash=[4,4], opacity=0.3).encode(y="y:Q")
            v_rule = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(
                color="white", strokeDash=[4,4], opacity=0.3).encode(x="x:Q")
            st.altair_chart(scatter + h_rule + v_rule, use_container_width=True)

    # Discount bucket analysis
    st.subheader("折扣分組 → 後續各期報酬")
    disc_data["disc_bucket"] = pd.cut(
        disc_data["discount_pct"],
        bins=[-1, -0.4, -0.3, -0.2, -0.1, 0, 0.5],
        labels=["<-40%", "-40~-30%", "-30~-20%", "-20~-10%", "-10~0%", ">0%"]
    )
    bucket_rows = []
    for bkt, grp in disc_data.groupby("disc_bucket", observed=True):
        row = {"折扣區間": str(bkt), "樣本數": len(grp)}
        for day in [1, 5, 10, 20, 60]:
            col_r = f"ret_{day:+d}d"
            if col_r in grp.columns:
                s = grp[col_r].dropna()
                row[f"+{day}d均報"] = f"{s.mean()*100:.1f}%" if len(s) > 0 else "—"
                row[f"+{day}d勝率"] = f"{(s>0).mean()*100:.0f}%" if len(s) > 0 else "—"
        bucket_rows.append(row)
    bkt_df = pd.DataFrame(bucket_rows)
    st.dataframe(bkt_df, use_container_width=True, hide_index=True)


# ── Tab 4: Historical Records ─────────────────────────────────────────────
with tab4:
    st.subheader("歷史回測紀錄")

    disp_cols = (
        ["ticker", "ex_date", "sub_price", "ref_price", "discount_pct"]
        + ["ret_-20d"]
        + [f"ret_{d:+d}d" for d in range(-10, 11) if f"ret_{d:+d}d" in h.columns]
        + ["ret_+20d", "ret_+30d", "ret_+60d"]
    )
    disp_cols = [c for c in disp_cols if c in h.columns]
    raw = h[disp_cols].copy().sort_values("ex_date", ascending=False)

    # Format
    raw["ex_date"] = raw["ex_date"].dt.strftime("%Y-%m-%d")
    for c in raw.columns:
        if c.startswith("ret_") or c == "discount_pct":
            raw[c] = raw[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    for c in ["sub_price", "ref_price"]:
        if c in raw.columns:
            raw[c] = raw[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    rename = {
        "ticker": "股票代號", "ex_date": "除權日",
        "sub_price": "認購價", "ref_price": "前日收盤",
        "discount_pct": "折扣%",
    }
    rename["ret_-20d"] = "Day-20"
    for d in range(-10, 11):
        rename[f"ret_{d:+d}d"] = f"Day{d:+d}"
    rename["ret_+20d"] = "Day+20"
    rename["ret_+30d"] = "Day+30"
    rename["ret_+60d"] = "Day+60"
    raw = raw.rename(columns=rename)

    # Download button
    st.download_button(
        "⬇️ 下載 CSV",
        data=raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="cash_increase_backtest.csv",
        mime="text/csv"
    )

    st.dataframe(raw, use_container_width=True, height=560)
