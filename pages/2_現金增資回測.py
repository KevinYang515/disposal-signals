"""
現金增資事件研究 v2
- 公告日 (T=0=交易所認購條件公告) 和 除權日 (T=0=除權) 兩種錨點
- 資料: FinLab dividend_announcement，上市+上櫃+興櫃 2006-2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
import altair as alt

st.set_page_config(page_title="現金增資回測", page_icon="💰", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
.badge-ann { background:#2c3e50; color:#f6c90e; padding:2px 8px;
             border-radius:4px; font-size:0.8em; font-weight:700; }
.badge-exr { background:#2c3e50; color:#4a90d9; padding:2px 8px;
             border-radius:4px; font-size:0.8em; font-weight:700; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FINE_DAYS        = list(range(-10, 11))
SUMMARY_HORIZONS = [-20, -10, 1, 5, 10, 20, 30, 60]


@st.cache_data(ttl=3600)
def load_all():
    events  = pd.read_csv(f"{DATA_DIR}/cash_increase_events.csv",
                          parse_dates=["ann_date","exr_date"])
    win_ann = pd.read_csv(f"{DATA_DIR}/cash_increase_window_ann.csv",
                          parse_dates=["ann_date"])
    win_exr = pd.read_csv(f"{DATA_DIR}/cash_increase_window_exr.csv",
                          parse_dates=["ann_date"])
    avg_ann = pd.read_csv(f"{DATA_DIR}/cash_increase_avg_ann.csv")
    avg_exr = pd.read_csv(f"{DATA_DIR}/cash_increase_avg_exr.csv")
    hor_ann = pd.read_csv(f"{DATA_DIR}/cash_increase_horizon_ann.csv",
                          parse_dates=["ann_date","exr_date"])
    hor_exr = pd.read_csv(f"{DATA_DIR}/cash_increase_horizon_exr.csv",
                          parse_dates=["ann_date","exr_date"])
    summary = pd.read_csv(f"{DATA_DIR}/cash_increase_summary.csv")
    year_df = pd.read_csv(f"{DATA_DIR}/cash_increase_year.csv")
    with open(f"{DATA_DIR}/cash_increase_meta.json") as f:
        meta = json.load(f)
    # 計算市值代理並分級（大>500億 / 中100~500億 / 小<100億）
    def add_size_band(df):
        mktcap = df["ref_price"] * df["shares"] / (df["sub_ratio"] / 100) / 1e8
        df = df.copy()
        df["mktcap_b"] = mktcap
        df["size_band"] = pd.cut(
            mktcap,
            bins=[0, 100, 500, float("inf")],
            labels=["小型 <100億", "中型 100~500億", "大型 >500億"],
        )
        return df
    hor_ann = add_size_band(hor_ann)
    hor_exr = add_size_band(hor_exr)

    # 把 event_id → size_band / market 對應表 join 進 window 資料
    if "event_id" in hor_ann.columns:
        id_map = hor_ann.set_index("event_id")[["market", "size_band"]]
        for win_df in [win_ann, win_exr]:
            if "event_id" in win_df.columns:
                extra = id_map.reindex(win_df["event_id"].values)
                extra.index = win_df.index
                for col in ["market", "size_band"]:
                    if col not in win_df.columns:
                        win_df[col] = extra[col].values
    return events, win_ann, win_exr, avg_ann, avg_exr, hor_ann, hor_exr, summary, year_df, meta

try:
    events, win_ann, win_exr, avg_ann, avg_exr, hor_ann, hor_exr, summary, year_df, meta = load_all()
except FileNotFoundError:
    st.error("找不到預計算資料，請先執行 `python3 stock/cash_increase_compute.py`")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("篩選條件")
    yr_min = int(events["ann_date"].dt.year.min())
    yr_max = int(events["ann_date"].dt.year.max())
    default_yr_start = min(2024, yr_max)
    sel_years = st.slider("年份範圍", yr_min, yr_max, (default_yr_start, yr_max))

    SIZE_OPTS = ["全部", "大型 >500億", "中型 100~500億", "小型 <100億"]
    sel_size = st.selectbox("市值規模", SIZE_OPTS, index=1,
                            help="以 認購前日收盤 × 現有股數 估算市值。\n"
                                 "2024+ 大型: ~40筆  中型: ~177筆  小型: ~72筆")

    disc_vals = hor_ann["discount_pct"].dropna()
    d_lo = float(disc_vals.quantile(0.03))
    d_hi = float(disc_vals.quantile(0.97))
    sel_disc = st.slider("認購折扣範圍 (vs 前日收盤)", -1.0, 0.5, (d_lo, d_hi), step=0.01)

    with st.expander("進階：市場別篩選"):
        mkt_opts = ["全部", "sii（上市）", "otc（上櫃）", "rotc（興櫃）"]
        sel_mkt_disp = st.selectbox("市場別", mkt_opts, index=0)
        sel_mkt = sel_mkt_disp.split("（")[0] if sel_mkt_disp != "全部" else "全部"


def apply_filters(df, date_col="ann_date"):
    m = (df[date_col].dt.year >= sel_years[0]) & (df[date_col].dt.year <= sel_years[1])
    if sel_size != "全部" and "size_band" in df.columns:
        m &= df["size_band"] == sel_size
    if sel_mkt != "全部" and "market" in df.columns:
        m &= df["market"] == sel_mkt
    if "discount_pct" in df.columns:
        m &= (df["discount_pct"].fillna(-999) >= sel_disc[0]) & \
             (df["discount_pct"].fillna(999)  <= sel_disc[1])
    return df[m].copy()


h_ann = apply_filters(hor_ann)
h_exr = apply_filters(hor_exr)
ev_f  = apply_filters(events)


def apply_window_filter(win_df):
    m = (win_df["ann_date"].dt.year >= sel_years[0]) & \
        (win_df["ann_date"].dt.year <= sel_years[1])
    if sel_size != "全部" and "size_band" in win_df.columns:
        m &= win_df["size_band"] == sel_size
    if sel_mkt != "全部" and "market" in win_df.columns:
        m &= win_df["market"] == sel_mkt
    return win_df[m]

wa_f = apply_window_filter(win_ann)
we_f = apply_window_filter(win_exr)


# ── Header ────────────────────────────────────────────────────────────────
st.title("💰 現金增資回測分析")
st.caption(
    f"FinLab · 上市+上櫃+興櫃 · {meta['n_events']} 筆事件 · "
    f"{meta['date_start']} ~ {meta['date_end']} · 更新 {meta['updated_at'][:10]}"
)

st.info(
    "**兩種 T=0：**  \n"
    "🟡 **公告日** = 交易所公告認購條件（距除權約 7-10 天，此時認購價確定）  \n"
    "🔵 **除權日** = 股票去除認購權利（機械性稀釋壓力）  \n"
    "⚠️ **董事會決議日（你看到的新聞日，如華通 2026/5/7）**不在此資料中，"
    "通常比公告日早 2-3 個月，是最早可交易的訊號。",
    icon="ℹ️"
)

# ── 樣本數警示 ────────────────────────────────────────────────────────────
n_filtered = len(h_ann)
if n_filtered < 30:
    st.error(f"⚠️ 目前篩選後僅剩 **{n_filtered}** 筆，樣本太少，統計結果不可信賴。建議放寬年份或市場別篩選。")
elif n_filtered < 80:
    st.warning(f"⚠️ 篩選後共 **{n_filtered}** 筆，統計結果僅供參考，建議搭配更長時間區間交叉驗證。")

# ── KPIs ─────────────────────────────────────────────────────────────────
k0, k1, k2, k3, k4, k5 = st.columns(6)
k0.metric("篩選事件數", f"{n_filtered:,}")

for col_ui, df_h, day, anchor in [
    (k1, h_ann, 5,  "📢+5d"),
    (k2, h_ann, 20, "📢+20d"),
    (k3, h_exr, 5,  "📋+5d"),
    (k4, h_exr, 20, "📋+20d"),
    (k5, h_exr, 60, "📋+60d"),
]:
    col_name = f"ret_{day:+d}d"
    if col_name in df_h.columns:
        s  = df_h[col_name].dropna()
        v  = s.mean()
        wr = (s > 0).mean()
        col_ui.metric(anchor, f"{v*100:.1f}%", f"勝率 {wr*100:.0f}%",
                      delta_color="normal" if v >= 0 else "inverse")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 CAR 曲線對比", "📊 日別細節 -10~+10", "💲 認購價分析", "📋 歷史紀錄"]
)


# ── Tab 1: CAR curves comparison ─────────────────────────────────────────
with tab1:
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.subheader("公告日 vs 除權日 CAR 曲線")

        avg_a = wa_f.groupby("offset")[["cum_ret","abnormal"]].mean().reset_index()
        avg_e = we_f.groupby("offset")[["cum_ret","abnormal"]].mean().reset_index()
        avg_a["anchor"] = "公告日"
        avg_e["anchor"] = "除權日"
        combined = pd.concat([avg_a, avg_e])
        combined_long = combined.melt(
            id_vars=["offset","anchor"], value_vars=["cum_ret","abnormal"],
            var_name="type", value_name="ret"
        )
        combined_long["series"] = combined_long["anchor"] + " " + combined_long["type"].map(
            {"cum_ret":"個股累積報酬", "abnormal":"超額報酬"})

        color_scale = alt.Scale(
            domain=["公告日 個股累積報酬","公告日 超額報酬",
                    "除權日 個股累積報酬","除權日 超額報酬"],
            range=["#f6c90e","#e67e22","#4a90d9","#e74c3c"]
        )
        car_chart = (
            alt.Chart(combined_long)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("offset:Q", title="交易日 (0=各自錨點)",
                         scale=alt.Scale(domain=[-30,60])),
                y=alt.Y("ret:Q", title="累積報酬",
                         axis=alt.Axis(format=".1%")),
                color=alt.Color("series:N", scale=color_scale,
                                legend=alt.Legend(orient="top", columns=2)),
                strokeDash=alt.StrokeDash("type:N",
                    scale=alt.Scale(domain=["cum_ret","abnormal"],
                                    range=[[1,0],[4,2]]),
                    legend=None),
                tooltip=["series:N","offset:Q",
                         alt.Tooltip("ret:Q", format=".2%")]
            )
            .properties(height=360)
        )
        zero  = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(
            color="white",strokeDash=[3,3],opacity=0.3).encode(y="y:Q")
        ev_r  = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(
            color="gray",opacity=0.5).encode(x="x:Q")
        st.altair_chart(car_chart + zero + ev_r, use_container_width=True)

        st.caption(
            "🟡實線=公告日個股報酬 　🟠虛線=公告日超額報酬  \n"
            "🔵實線=除權日個股報酬 　🔴虛線=除權日超額報酬"
        )

    with c_right:
        st.subheader("各期間報酬比較")

        sel_anchor = st.radio("錨點", ["公告日 🟡", "除權日 🔵"],
                               horizontal=True, label_visibility="collapsed")
        use_h = h_ann if "公告" in sel_anchor else h_exr

        bar_rows = []
        for day in [1, 5, 10, 20, 30, 60]:
            col_r = f"ret_{day:+d}d"
            if col_r not in use_h.columns:
                continue
            s = use_h[col_r].dropna()
            if len(s) < 5:
                continue
            bar_rows.append({
                "label": f"+{day}d",
                "mean_ret": s.mean(),
                "win_rate": (s > 0).mean(),
                "n": len(s),
            })
        bar_df = pd.DataFrame(bar_rows)

        if not bar_df.empty:
            ret_bar = (
                alt.Chart(bar_df)
                .mark_bar()
                .encode(
                    x=alt.X("label:N", sort=None, title="期間"),
                    y=alt.Y("mean_ret:Q", title="平均報酬",
                             axis=alt.Axis(format=".1%")),
                    color=alt.Color("mean_ret:Q",
                        scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                        legend=None),
                    tooltip=[
                        alt.Tooltip("label:N"),
                        alt.Tooltip("mean_ret:Q", format=".2%", title="平均報酬"),
                        alt.Tooltip("win_rate:Q", format=".0%", title="勝率"),
                        alt.Tooltip("n:Q", title="樣本數"),
                    ]
                )
                .properties(height=185)
            )
            wr_bar = (
                alt.Chart(bar_df)
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
                color="white",strokeDash=[4,4],opacity=0.4).encode(y="y:Q")
            st.altair_chart(ret_bar, use_container_width=True)
            st.altair_chart(wr_bar + fifty, use_container_width=True)

    # Year heatmap
    st.subheader("年度 × 期間 熱力圖")
    sel_anchor_yr = st.radio("錨點", ["公告日", "除權日"], horizontal=True, key="yr_anchor")
    yr_f = year_df[
        (year_df["anchor"] == sel_anchor_yr) &
        (year_df["year"] >= sel_years[0]) &
        (year_df["year"] <= sel_years[1]) &
        (year_df["horizon"].isin([1, 5, 10, 20, 30, 60]))
    ].copy()
    yr_f["hlabel"] = yr_f["horizon"].apply(lambda x: f"+{x}d")
    yr_f["text"]   = yr_f["mean_ret"].apply(
        lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")

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
                tooltip=[alt.Tooltip("year:O"), alt.Tooltip("hlabel:N", title="期間"),
                         alt.Tooltip("mean_ret:Q", format=".1%"), alt.Tooltip("n:Q")]
            )
        )
        txt = (
            alt.Chart(yr_f)
            .mark_text(fontSize=9, color="white")
            .encode(
                x=alt.X("hlabel:N",
                         sort=[f"+{d}d" for d in sorted(yr_f["horizon"].unique())]),
                y=alt.Y("year:O"),
                text=alt.Text("text:N"),
            )
        )
        st.altair_chart((hmap + txt).properties(height=420), use_container_width=True)


# ── Tab 2: Day -10 to +10 ─────────────────────────────────────────────────
with tab2:
    st.subheader("除權日 / 公告日 前後逐日報酬 (-10 ~ +10)")
    sel_anc2 = st.radio("錨點", ["公告日 🟡", "除權日 🔵"], horizontal=True, key="tab2anc")
    use_h2 = h_ann if "公告" in sel_anc2 else h_exr
    color_anc = "#f6c90e" if "公告" in sel_anc2 else "#4a90d9"

    day_rows = []
    for day in range(-10, 11):
        col_r = f"ret_{day:+d}d"
        col_a = f"abn_{day:+d}d"
        if col_r not in use_h2.columns:
            continue
        s = use_h2[col_r].dropna()
        a = use_h2[col_a].dropna() if col_a in use_h2.columns else pd.Series(dtype=float)
        if len(s) == 0:
            continue
        day_rows.append({
            "day": day, "n": len(s),
            "mean_ret": s.mean(), "median_ret": s.median(),
            "win_rate": (s > 0).mean(),
            "mean_abn": a.mean() if len(a) > 0 else np.nan,
            "std": s.std(),
            "p25": s.quantile(0.25),
            "p75": s.quantile(0.75),
        })
    day_df = pd.DataFrame(day_rows)

    bar_day = (
        alt.Chart(day_df)
        .mark_bar(width=20)
        .encode(
            x=alt.X("day:Q", title="交易日 (0=錨點日)",
                     scale=alt.Scale(domain=[-10.5,10.5]),
                     axis=alt.Axis(tickCount=21)),
            y=alt.Y("mean_ret:Q", title="平均累積報酬",
                     axis=alt.Axis(format=".1%")),
            color=alt.condition(
                alt.datum.mean_ret >= 0,
                alt.value("#26c281"), alt.value("#e74c3c")
            ),
            tooltip=[
                alt.Tooltip("day:Q", title="Day"),
                alt.Tooltip("mean_ret:Q", format=".2%", title="平均報酬"),
                alt.Tooltip("median_ret:Q", format=".2%", title="中位數"),
                alt.Tooltip("win_rate:Q", format=".0%", title="勝率"),
                alt.Tooltip("n:Q", title="樣本"),
            ]
        )
        .properties(height=260)
    )
    wr_line = (
        alt.Chart(day_df)
        .mark_line(point=True, strokeWidth=2, color=color_anc)
        .encode(
            x=alt.X("day:Q", scale=alt.Scale(domain=[-10.5,10.5])),
            y=alt.Y("win_rate:Q", title="勝率",
                     axis=alt.Axis(format=".0%"),
                     scale=alt.Scale(domain=[0,1])),
            tooltip=[alt.Tooltip("day:Q"), alt.Tooltip("win_rate:Q", format=".0%")]
        )
        .properties(height=200)
    )
    zero_r  = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="white",strokeDash=[4,4],opacity=0.3).encode(y="y:Q")
    ev_rule = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(color="gray",opacity=0.5).encode(x="x:Q")
    fifty_r = alt.Chart(pd.DataFrame({"y":[0.5]})).mark_rule(color="white",strokeDash=[4,4],opacity=0.3).encode(y="y:Q")

    st.altair_chart(bar_day + zero_r + ev_rule, use_container_width=True)
    st.altair_chart(wr_line + fifty_r + ev_rule, use_container_width=True)

    # Table
    tbl = day_df.copy()
    for col in ["mean_ret","median_ret","mean_abn","std","p25","p75"]:
        tbl[col] = tbl[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")
    tbl["win_rate"] = tbl["win_rate"].apply(lambda x: f"{x*100:.1f}%")
    tbl.columns = ["Day","樣本","平均報酬","中位數","勝率","超額報酬","標準差","P25","P75"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ── Tab 3: Subscription Price ─────────────────────────────────────────────
with tab3:
    st.subheader("認購價 vs 股價分析")
    disc = h_ann[h_ann["discount_pct"].notna()].copy()
    st.caption(f"{len(disc)} 筆有認購價資料")

    ca, cb = st.columns(2)

    with ca:
        st.markdown("**認購折扣分佈**（認購價 / 前日收盤 - 1）")
        dd = disc["discount_pct"].clip(-1, 0.5).to_frame("disc")
        hist_d = (
            alt.Chart(dd)
            .mark_bar(opacity=0.8, color="#4a90d9")
            .encode(
                x=alt.X("disc:Q", bin=alt.Bin(maxbins=50),
                         title="折扣", axis=alt.Axis(format=".0%")),
                y=alt.Y("count():Q", title="頻次"),
            )
            .properties(height=260)
        )
        mean_r = alt.Chart(pd.DataFrame({"x":[disc["discount_pct"].mean()]})).mark_rule(
            color="#f6c90e",strokeDash=[5,3]).encode(x="x:Q")
        st.altair_chart(hist_d + mean_r, use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("平均折扣",  f"{disc['discount_pct'].mean()*100:.1f}%")
        c2.metric("中位數折扣", f"{disc['discount_pct'].median()*100:.1f}%")
        c3.metric(">0（溢價）", f"{(disc['discount_pct']>0).mean()*100:.0f}%")

    with cb:
        st.markdown("**折扣深度 vs +20日報酬**")
        if "ret_+20d" in disc.columns:
            sc = disc[["discount_pct","ret_+20d","ret_+60d"]].dropna()
            scatter = (
                alt.Chart(sc)
                .mark_circle(opacity=0.45, size=40)
                .encode(
                    x=alt.X("discount_pct:Q", title="認購折扣",
                             axis=alt.Axis(format=".0%")),
                    y=alt.Y("ret_+20d:Q", title="+20日報酬",
                             axis=alt.Axis(format=".0%")),
                    color=alt.Color("ret_+20d:Q",
                        scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                        legend=None),
                    tooltip=[
                        alt.Tooltip("discount_pct:Q", format=".1%", title="折扣"),
                        alt.Tooltip("ret_+20d:Q", format=".1%", title="+20日"),
                        alt.Tooltip("ret_+60d:Q", format=".1%", title="+60日"),
                    ]
                )
                .properties(height=260)
            )
            hr = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="white",strokeDash=[3,3],opacity=0.3).encode(y="y:Q")
            vr = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(color="white",strokeDash=[3,3],opacity=0.3).encode(x="x:Q")
            st.altair_chart(scatter + hr + vr, use_container_width=True)

    st.subheader("折扣分組 → 後續各期均報 / 勝率")
    disc["bucket"] = pd.cut(
        disc["discount_pct"],
        bins=[-1.1,-0.4,-0.3,-0.2,-0.1,-0.05,0,0.5],
        labels=["<-40%","-40~-30%","-30~-20%","-20~-10%","-10~-5%","-5~0%",">0%"]
    )
    bkt_rows = []
    for bkt, grp in disc.groupby("bucket", observed=True):
        row = {"折扣區間": str(bkt), "樣本": len(grp)}
        for day in [1, 5, 10, 20, 60]:
            c = f"ret_{day:+d}d"
            if c in grp.columns:
                s = grp[c].dropna()
                row[f"+{day}d均報"] = f"{s.mean()*100:.1f}%" if len(s)>0 else "—"
                row[f"+{day}d勝率"] = f"{(s>0).mean()*100:.0f}%" if len(s)>0 else "—"
        bkt_rows.append(row)
    st.dataframe(pd.DataFrame(bkt_rows), use_container_width=True, hide_index=True)


# ── Tab 4: Historical Records ─────────────────────────────────────────────
with tab4:
    st.subheader("歷史紀錄 — 完整現金增資事件清單")
    st.caption("含公司名稱、市場別、公告日、除權日、認購價、各期間報酬")

    # Select anchor for return columns
    sel_anc4 = st.radio("報酬錨點", ["公告日 🟡", "除權日 🔵"], horizontal=True, key="rec_anchor")
    use_h4 = h_ann if "公告" in sel_anc4 else h_exr

    # Build display DF
    ret_days = [-20, -10, -5] + list(range(-3, 11)) + [20, 30, 60]
    ret_cols = [f"ret_{d:+d}d" for d in ret_days if f"ret_{d:+d}d" in use_h4.columns]

    base_cols = ["ticker","name","market","ann_date","exr_date",
                 "sub_price","ref_price","discount_pct","sub_ratio","gap_days","shares"]
    disp_cols = [c for c in base_cols if c in use_h4.columns] + ret_cols
    raw = use_h4[disp_cols].copy().sort_values("ann_date", ascending=False)

    # Format
    raw["ann_date"] = raw["ann_date"].dt.strftime("%Y-%m-%d")
    raw["exr_date"] = raw["exr_date"].dt.strftime("%Y-%m-%d") if "exr_date" in raw else "—"
    for c in ret_cols + (["discount_pct"] if "discount_pct" in raw.columns else []):
        raw[c] = raw[c].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    for c in ["sub_price","ref_price"]:
        if c in raw.columns:
            raw[c] = raw[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    if "sub_ratio" in raw.columns:
        raw["sub_ratio"] = raw["sub_ratio"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
    if "shares" in raw.columns:
        raw["shares"] = raw["shares"].apply(lambda x: f"{int(x/1000):,}張" if pd.notna(x) else "—")

    rename = {
        "ticker":"代號","name":"公司名稱","market":"市場","ann_date":"公告日",
        "exr_date":"除權日","sub_price":"認購價","ref_price":"前日收",
        "discount_pct":"折扣","sub_ratio":"認股比率%","gap_days":"公告→除權天",
        "shares":"增資規模",
    }
    for d in ret_days:
        rename[f"ret_{d:+d}d"] = f"Day{d:+d}"
    raw = raw.rename(columns=rename)

    st.download_button(
        "⬇️ 下載 CSV",
        data=raw.to_csv(index=False).encode("utf-8-sig"),
        file_name="cash_increase_history.csv",
        mime="text/csv"
    )
    st.dataframe(raw, use_container_width=True, height=580)
