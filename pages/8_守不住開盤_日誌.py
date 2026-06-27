"""
守不住開盤 — 每日交易日誌

顯示任一交易日的：
  選股清單 / 進場時間 & 價格 / 停損防守 / 出場時間 & 價格 / 損益
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="守不住開盤 日誌", page_icon="📋", layout="wide",
                   initial_sidebar_state="expanded")

SIGNALS_CSV = Path(__file__).parent.parent / "data" / "守不住開盤_上線版_signals.csv"
BUDGET      = 5_000_000
N_MAX       = 5
LOT         = 1000
BUY_R       = 0.001425 * 0.20
SELL_R      = 0.001425 * 0.20 + 0.0015
EXIT_TIME   = "13:25"

st.markdown("""
<style>
.trade-header { font-size: 1.05em; font-weight: 700; color: #94a3b8; margin: 12px 0 4px; }
.badge-green { background:#16a34a; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; }
.badge-red   { background:#dc2626; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; }
.badge-gray  { background:#475569; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; }
.kv { display:flex; gap:6px; align-items:baseline; }
.kv-k { color:#94a3b8; font-size:0.8em; min-width:72px; }
.kv-v { color:#e2e8f0; font-size:0.95em; font-weight:600; }
.card { background:#1e2530; border-radius:10px; padding:14px 18px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_signals():
    df = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])
    df["stock_aor"] = df["morning_high"] / df["day_open"] - 1
    return df


def fmt_min(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def calc_pnl(row, per_slot):
    lots = int(per_slot // (row["entry_price"] * LOT))
    if lots == 0:
        return 0, 0, 0
    sh = lots * LOT
    gross = (row["entry_price"] - row["exit_price"]) * sh
    cost  = row["entry_price"] * sh * SELL_R + row["exit_price"] * sh * BUY_R
    net   = gross - cost
    return net, lots, sh


if not SIGNALS_CSV.exists():
    st.error(f"找不到 {SIGNALS_CSV}，請先執行 `守不住開盤_上線版.py`")
    st.stop()

all_sig = load_signals()

# ── 側欄 ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📋 日誌設定")

    available_dates = sorted(all_sig["date"].dt.date.unique(), reverse=True)
    sel_date = st.selectbox(
        "交易日期",
        available_dates,
        index=0,
        format_func=lambda d: str(d)
    )

    st.divider()
    budget  = st.selectbox("資金規模", [1_000_000, 2_000_000, 5_000_000, 10_000_000],
                           index=2, format_func=lambda x: f"{x/1e4:.0f} 萬")
    n_max   = st.selectbox("N_MAX", [3, 4, 5, 6, 7], index=2)
    aor_thr = st.selectbox("漲不動 filter", [0.005, 0.01, 0.02, 1.0], index=1,
                           format_func=lambda x: f"< {x*100:.1f}%" if x < 1 else "無")
    st.divider()
    st.caption("進場：09:30–09:40 跌破 open×0.998\n停損：max(morning_high×1.005, open×1.003)\n出場：13:25 強制回補")

# ── 取當日資料 ──────────────────────────────────────────────
day_sig = all_sig[all_sig["date"].dt.date == sel_date].copy()
day_sig = day_sig[day_sig["stock_aor"] < aor_thr]
day_sig = day_sig.sort_values("rank_score", ascending=False)
selected = day_sig.head(n_max).copy()

per_slot = budget / n_max

# 計算 PnL
pnl_rows = []
for _, r in selected.iterrows():
    net, lots, sh = calc_pnl(r, per_slot)
    pnl_rows.append({"net": net, "lots": lots, "shares": sh})
pnl_df = pd.DataFrame(pnl_rows, index=selected.index)
selected = pd.concat([selected, pnl_df], axis=1)

# ── 頂部標題 ────────────────────────────────────────────────
st.title(f"🩸 守不住開盤 — 每日交易日誌")
st.subheader(f"📅 {sel_date}")

n_fills = (selected["lots"] > 0).sum()
total_net = selected["net"].sum()
wr = (selected["net"] > 0).mean() if n_fills > 0 else float("nan")
stopped = selected["stopped_out"].sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("候選數", f"{len(day_sig)}")
col2.metric("實際成交", f"{n_fills} 檔")
col3.metric("日淨 PnL", f"{total_net:+,.0f}", delta=f"{total_net/budget*100:+.2f}%")
col4.metric("勝率", f"{wr*100:.0f}%" if not np.isnan(wr) else "—")
col5.metric("停損觸發", f"{int(stopped)} 次")

st.divider()

# ── 個股卡片 ────────────────────────────────────────────────
st.subheader("📌 選入個股詳情")

if selected.empty:
    st.info(f"當日（{sel_date}）無符合條件的訊號")
else:
    for rank, (_, r) in enumerate(selected.iterrows(), 1):
        net = r["net"]; stopped_out = r["stopped_out"]
        tag = ("badge-red" if stopped_out else ("badge-green" if net > 0 else "badge-red"))
        tag_txt = "停損" if stopped_out else ("獲利" if net > 0 else "虧損")
        stop_dist = (r["stop_price"] / r["entry_price"] - 1) * 100

        with st.container():
            h1, h2 = st.columns([3, 1])
            h1.markdown(
                f"**#{rank}  {r['ticker']}**  "
                f"&nbsp;<span class='{tag}'>{tag_txt}</span> "
                f"&nbsp;<span class='badge-gray'>排序 {r['rank_score']:.2f}</span>",
                unsafe_allow_html=True
            )
            h2.markdown(
                f"<div style='text-align:right; font-size:1.3em; font-weight:700; "
                f"color:{'#4ade80' if net>0 else '#f87171'}'>{net:+,.0f}</div>",
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("**開盤資訊**")
                st.markdown(f"跳空幅度：`{r['gap_pct']*100:+.2f}%`")
                st.markdown(f"開盤價：`{r['day_open']:.2f}`")
                st.markdown(f"早盤最高：`{r['morning_high']:.2f}`")
                st.markdown(f"漲不動幅：`{r['stock_aor']*100:+.2f}%`")

            with c2:
                st.markdown("**進場**")
                st.markdown(f"時間：`{fmt_min(int(r['entry_min']))}`")
                st.markdown(f"觸價：`{r['entry_price']:.2f}`")
                st.markdown(f"口數：`{int(r['lots'])} 張 ({int(r['shares']):,} 股)`")

            with c3:
                st.markdown("**停損防守**")
                st.markdown(f"停損價：`{r['stop_price']:.2f}`")
                st.markdown(f"停損距離：`+{stop_dist:.2f}%`")
                st.markdown(f"防守方向：空頭需空頭反向突破")
                if stopped_out:
                    st.markdown(f":red[⚠ 本日觸及停損]")

            with c4:
                st.markdown("**出場**")
                st.markdown(f"時間：`{EXIT_TIME if not stopped_out else '停損觸發'}`")
                st.markdown(f"出場價：`{r['exit_price']:.2f}`")
                ret_pct = r["return"] * 100
                arrow = "▲" if ret_pct > 0 else "▼"
                color = "green" if ret_pct > 0 else "red"
                st.markdown(f"單筆報酬：<span style='color:{'#4ade80' if ret_pct>0 else '#f87171'}'>`{arrow} {abs(ret_pct):.2f}%`</span>",
                            unsafe_allow_html=True)

            # 迷你進出場價格視覺
            prices = {
                "開盤": r["day_open"],
                "早盤高": r["morning_high"],
                "進場(空)": r["entry_price"],
                "出場": r["exit_price"],
                "停損": r["stop_price"],
            }
            price_df = pd.DataFrame(list(prices.items()), columns=["label", "price"])
            price_df["color"] = price_df["label"].map({
                "開盤": "#94a3b8",
                "早盤高": "#fbbf24",
                "進場(空)": "#60a5fa",
                "出場": "#4ade80" if not stopped_out else "#f87171",
                "停損": "#f87171",
            })
            bar = alt.Chart(price_df).mark_bar(size=20).encode(
                x=alt.X("label:N", sort=list(prices.keys()), title=None,
                         axis=alt.Axis(labelColor="#94a3b8")),
                y=alt.Y("price:Q", scale=alt.Scale(
                    domain=[min(prices.values())*0.998, max(prices.values())*1.002]),
                    title="價格"),
                color=alt.Color("color:N", scale=None),
                tooltip=["label", alt.Tooltip("price:Q", format=".2f")]
            ).properties(height=140)
            st.altair_chart(bar, use_container_width=True)
            st.divider()

# ── 彙總表格 ────────────────────────────────────────────────
st.subheader("📊 當日彙總表")

if not selected.empty:
    show = selected[["ticker","gap_pct","stock_aor","rank_score",
                      "entry_min","entry_price","stop_price","exit_price",
                      "return","stopped_out","lots","net"]].copy()
    show["gap_pct"]    = show["gap_pct"].map(lambda x: f"{x*100:+.2f}%")
    show["stock_aor"]  = show["stock_aor"].map(lambda x: f"{x*100:+.2f}%")
    show["rank_score"] = show["rank_score"].map(lambda x: f"{x:.2f}")
    show["entry_min"]  = show["entry_min"].map(lambda x: fmt_min(int(x)))
    show["return"]     = show["return"].map(lambda x: f"{x*100:+.2f}%")
    show["stopped_out"]= show["stopped_out"].map(lambda x: "⚠ 停損" if x else "✓ 正常")
    show["lots"]       = show["lots"].map(lambda x: f"{int(x)} 張")
    show["net"]        = show["net"].map(lambda x: f"{x:+,.0f}")
    show = show.rename(columns={
        "ticker": "股票",
        "gap_pct": "跳空%",
        "stock_aor": "漲不動%",
        "rank_score": "排序分",
        "entry_min": "進場時間",
        "entry_price": "進場價",
        "stop_price": "停損價",
        "exit_price": "出場價",
        "return": "單筆報酬",
        "stopped_out": "狀態",
        "lots": "口數",
        "net": "淨 PnL (元)",
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

# ── 候選但未選入 ────────────────────────────────────────────
excluded = day_sig.iloc[n_max:] if len(day_sig) > n_max else pd.DataFrame()
if not excluded.empty:
    with st.expander(f"📋 候選但未選入（{len(excluded)} 檔）"):
        ex_show = excluded[["ticker","gap_pct","stock_aor","rank_score","entry_min","entry_price"]].copy()
        ex_show["gap_pct"]   = ex_show["gap_pct"].map(lambda x: f"{x*100:+.2f}%")
        ex_show["stock_aor"] = ex_show["stock_aor"].map(lambda x: f"{x*100:+.2f}%")
        ex_show["rank_score"]= ex_show["rank_score"].map(lambda x: f"{x:.2f}")
        ex_show["entry_min"] = ex_show["entry_min"].map(lambda x: fmt_min(int(x)))
        st.dataframe(ex_show.rename(columns={
            "ticker": "股票", "gap_pct": "跳空%", "stock_aor": "漲不動%",
            "rank_score": "排序分", "entry_min": "進場時間", "entry_price": "進場價"
        }), use_container_width=True, hide_index=True)

# ── PnL 條形圖 ───────────────────────────────────────────────
if not selected.empty and n_fills > 0:
    st.subheader("💰 個股損益")
    pnl_chart_df = selected[selected["lots"] > 0][["ticker","net"]].copy()
    pnl_chart_df["color"] = pnl_chart_df["net"].apply(lambda x: "#4ade80" if x > 0 else "#f87171")
    bar2 = alt.Chart(pnl_chart_df).mark_bar().encode(
        x=alt.X("ticker:N", title="股票", sort="-y"),
        y=alt.Y("net:Q", title="淨 PnL (元)"),
        color=alt.Color("color:N", scale=None),
        tooltip=["ticker", alt.Tooltip("net:Q", format=",.0f", title="淨 PnL")]
    ).properties(height=220)
    st.altair_chart(bar2, use_container_width=True)

st.divider()
st.caption(
    f"🩸 守不住開盤 日誌 v1.0 · 資料：{SIGNALS_CSV.name} · "
    f"今日候選共 {len(day_sig)} 檔（漲不動 < {aor_thr*100:.1f}%），選入 top-{n_max}"
)
