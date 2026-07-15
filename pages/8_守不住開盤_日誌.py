"""
守不住開盤 — 每日交易日誌

顯示任一交易日的：
  選股清單 / 進場時間 & 價格 / 停損防守 / 出場時間 & 價格 / 損益
  + 股票名稱 / 漲停近接警示（停損本身 V3b 後自動 clip 安全）
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path

st.set_page_config(page_title="守不住開盤 日誌", page_icon="📋", layout="wide",
                   initial_sidebar_state="expanded")

_DATA       = Path(__file__).parent.parent / "data"
SIGNALS_CSV = _DATA / "守不住開盤_上線版_signals.csv"
NAMES_CSV   = _DATA / "stock_names.csv"
LOT         = 1000
BUY_R       = 0.001425 * 0.20
SELL_R      = 0.001425 * 0.20 + 0.0015
EXIT_TIME   = "11:30"

st.markdown("""
<style>
.badge-green  { background:#16a34a; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; }
.badge-red    { background:#dc2626; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; }
.badge-gray   { background:#475569; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; }
.badge-orange { background:#d97706; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; }
.badge-danger { background:#7c2d12; color:#fef2f2; border-radius:6px; padding:2px 8px; font-size:0.82em; font-weight:600; border: 1px solid #dc2626; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_signals():
    df = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])
    df["stock_aor"] = df["morning_high"] / df["day_open"] - 1
    # 漲停風險計算
    df["prev_close"]    = df["day_open"] / (1 + df["gap_pct"])
    df["limit_price"]   = df["prev_close"] * 1.1
    df["gap_to_limit"]  = (df["limit_price"] / df["day_open"] - 1)   # 開盤到漲停還差多少
    df["stop_vs_limit"] = df["stop_price"] / df["limit_price"] - 1   # >0 代表停損超出漲停
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


@st.cache_data(ttl=3600)
def load_names():
    if NAMES_CSV.exists():
        m = pd.read_csv(NAMES_CSV, dtype=str)
        m["ticker"] = m["ticker"].str.strip()
        return dict(zip(m["ticker"], m["name"]))
    return {}


def fmt_min(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def calc_pnl(row, per_slot):
    lots = int(per_slot // (row["entry_price"] * LOT))
    if lots == 0:
        return 0, 0, 0
    sh = lots * LOT
    gross = (row["entry_price"] - row["exit_price"]) * sh
    cost  = row["entry_price"] * sh * SELL_R + row["exit_price"] * sh * BUY_R
    return gross - cost, lots, sh


def limit_badge(row):
    """回傳 (html badge, 說明文字)，None 表示無風險

    V3b 已自動 clip 停損 ≤ 漲停 - 1 tick，danger 不會觸發。
    Orange 是「進場後若被鎖漲停，部位無法出場」的潛在風險（與停損價無關）。
    """
    if row["stop_vs_limit"] > 0:
        # V3b 後不會發生（保留邏輯防止資料異常）
        return "badge-danger", f"⚠ 異常：停損({row['stop_price']:.2f}) 超出漲停({row['limit_price']:.2f})"
    if row["gap_to_limit"] < 0.02:
        return "badge-orange", (
            f"開盤距漲停僅 {row['gap_to_limit']*100:.2f}%。停損價({row['stop_price']:.2f}) "
            f"低於漲停({row['limit_price']:.2f}) {abs(row['stop_vs_limit'])*100:.2f}%，**停損本身安全**；"
            f"但若進場後被鎖漲停，部位卡到收盤前無法出場（強制承受最大損失）"
        )
    return None, None


if not SIGNALS_CSV.exists():
    st.error(f"找不到 {SIGNALS_CSV}，請先執行 `守不住開盤_上線版.py`")
    st.stop()

all_sig = load_signals()
names   = load_names()

# ── 側欄 ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📋 日誌設定")
    available_dates = sorted(all_sig["date"].dt.date.unique(), reverse=True)
    sel_date = st.selectbox("交易日期", available_dates, index=0,
                            format_func=lambda d: str(d))
    st.divider()
    budget  = st.selectbox("資金規模", [1_000_000, 2_000_000, 5_000_000, 10_000_000],
                           index=2, format_func=lambda x: f"{x/1e4:.0f} 萬")
    n_max   = st.selectbox("N_MAX", [3, 4, 5, 6, 7], index=2)
    aor_thr = st.selectbox("漲不動 filter", [0.005, 0.01, 0.02, 1.0], index=1,
                           format_func=lambda x: f"< {x*100:.2f}%" if x < 1 else "無")
    st.divider()
    st.caption(
        "**E4 配置（2026-07-01 最終版）**: 758 支中型股 universe\n\n"
        "- **Gap 0.5-10%**（全開）\n"
        "- **市值 < 500 億元**（排除大型股）\n"
        "- **進場窗 09:35-11:00**\n"
        "- **09:35 開棒 Open >= 觸價**（過濾已破股票）\n"
        "- Trail: running_low + 1 tick\n"
        "- Exit: Trail 或 11:30 強平\n\n"
        "**全期 Sharpe 4.17 / CAGR +10.60% / MaxDD -1.50%**\n"
        "TEST ≥2025 Sharpe 4.88, 2025 Sharpe +5.00"
    )

# ── 取當日資料 ──────────────────────────────────────────────
day_sig  = all_sig[all_sig["date"].dt.date == sel_date].copy()
day_sig  = day_sig[day_sig["stock_aor"] < aor_thr]
day_sig  = day_sig.sort_values("rank_score", ascending=False)
selected = day_sig.head(n_max).copy()

per_slot = budget / n_max

pnl_rows = []
for _, r in selected.iterrows():
    net, lots, sh = calc_pnl(r, per_slot)
    pnl_rows.append({"net": net, "lots": lots, "shares": sh})
if pnl_rows:
    pnl_df = pd.DataFrame(pnl_rows, index=selected.index)
    selected = pd.concat([selected, pnl_df], axis=1)
else:
    selected["net"] = selected["lots"] = selected["shares"] = 0

# ── 頂部 ────────────────────────────────────────────────────
st.title("🩸 守不住開盤 — 每日交易日誌")

st.success(
    "✅ **本頁已升級為 E4 訊號（2026-07-01 最終版）**\n\n"
    "從 758 支 TWSE 中型股（<500 億市值）universe 中找到的可執行訊號。"
    "共 **2,640 筆**，2023-2026 H1 全期實際可成交。\n\n"
    "- 全期 **Sharpe 4.17**, CAGR **+10.60%**, MaxDD **-1.50%**\n"
    "- Walk-forward **TEST ≥2025 Sharpe 4.88** (OOS 甚至更強)\n"
    "- **每天 ~3.3 fills**（500 萬本金每月平均 +3.7 萬）\n"
    "- 2025 Sharpe **+5.00**（regime problem 完全解決）\n\n"
    "🟢 **從無法上線的 bug 版 → 真正可執行的 Sharpe 4+ 策略**\n"
    "詳細分析見 `REPORT_守不住開盤_崩潰分析.md`"
)

st.subheader(f"📅 {sel_date}")

n_fills   = int((selected["lots"] > 0).sum()) if "lots" in selected else 0
total_net = float(selected["net"].sum()) if "net" in selected else 0.0
wr        = float((selected["net"] > 0).mean()) if n_fills > 0 else float("nan")
stopped   = int(selected["stopped_out"].sum()) if not selected.empty else 0
n_limit_risk = int((selected["stop_vs_limit"] > 0).sum()) if not selected.empty else 0

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("候選數", f"{len(day_sig)}")
c2.metric("實際成交", f"{n_fills} 檔")
c3.metric("日淨 PnL", f"{total_net:+,.0f}", delta=f"{total_net/budget*100:+.2f}%")
c4.metric("勝率", f"{wr*100:.2f}%" if not np.isnan(wr) else "—")
c5.metric("停損觸發", f"{stopped} 次")
c6.metric("停損超漲停", f"{n_limit_risk} 檔",
          delta="V3b 已自動 clip" if n_limit_risk == 0 else "需注意",
          delta_color="inverse")

if n_limit_risk > 0:
    st.warning(
        f"⚠ 本日選入個股中有 **{n_limit_risk}** 檔的停損價格高於漲停價，"
        "若股價反彈鎖漲停，停損單將無法成交，須以市價或撤單處理。",
        icon="🚨"
    )

st.divider()

# ── 個股卡片 ────────────────────────────────────────────────
st.subheader("📌 選入個股詳情")

if selected.empty:
    st.info(f"當日（{sel_date}）無符合條件的訊號")
else:
    for rank, (_, r) in enumerate(selected.iterrows(), 1):
        net = float(r.get("net", 0))
        lots = int(r.get("lots", 0))
        stopped_out = bool(r["stopped_out"])
        ticker = str(r["ticker"])
        name   = names.get(ticker, "")
        stop_dist = (r["stop_price"] / r["entry_price"] - 1) * 100
        lbadge, lmsg = limit_badge(r)

        # V4: stopped_out 可能是 trail stop 鎖獲利（不是虧損）
        is_loss = net < 0
        status_cls = "badge-green" if not is_loss else "badge-red"
        if stopped_out:
            status_txt = "Trail 鎖獲利" if not is_loss else "真實停損"
        else:
            status_txt = "時間出場-獲利" if not is_loss else "時間出場-虧損"

        with st.container():
            h1, h2 = st.columns([4, 1])
            limit_html = f"&nbsp;<span class='{lbadge}'>{'🔴 停損超漲停' if r['stop_vs_limit']>0 else '🟡 開盤接近漲停'}</span>" if lbadge else ""
            h1.markdown(
                f"**#{rank}&nbsp; {ticker} {name}**&nbsp;&nbsp;"
                f"<span class='{status_cls}'>{status_txt}</span>&nbsp;"
                f"<span class='badge-gray'>排序 {r['rank_score']:.2f}</span>"
                f"{limit_html}",
                unsafe_allow_html=True
            )
            h2.markdown(
                f"<div style='text-align:right;font-size:1.3em;font-weight:700;"
                f"color:{'#4ade80' if net>0 else '#f87171'}'>{net:+,.0f}</div>",
                unsafe_allow_html=True
            )

            if lmsg:
                st.error(f"⚠ **漲停風險**：{lmsg}")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("**開盤資訊**")
                st.markdown(f"跳空幅度：`{r['gap_pct']*100:+.2f}%`")
                st.markdown(f"開盤價：`{r['day_open']:.2f}`")
                st.markdown(f"早盤最高：`{r['morning_high']:.2f}`")
                st.markdown(f"漲不動幅：`{r['stock_aor']*100:+.2f}%`")
                st.markdown(f"漲停價：`{r['limit_price']:.2f}` (距開盤 `{r['gap_to_limit']*100:.2f}%`)")

            with c2:
                st.markdown("**進場**")
                st.markdown(f"時間：`{fmt_min(int(r['entry_min']))}`")
                st.markdown(f"觸價：`{r['entry_price']:.2f}`")
                st.markdown(f"口數：`{lots} 張 ({lots*LOT:,} 股)`")

            with c3:
                st.markdown("**停損防守**")
                st.markdown(f"停損價：`{r['stop_price']:.2f}`")
                st.markdown(f"停損距離（多方）：`+{stop_dist:.2f}%`")
                if r["stop_vs_limit"] > 0:
                    st.markdown(f":red[🔴 停損超出漲停 {r['stop_vs_limit']*100:+.2f}%（V3b 後不應發生）]")
                elif r["gap_to_limit"] < 0.02:
                    st.markdown(
                        f":orange[🟡 開盤距漲停僅 {r['gap_to_limit']*100:.2f}%]  "
                        f"（停損價低於漲停 {abs(r['stop_vs_limit'])*100:.2f}% 仍安全，"
                        f"但進場後若鎖漲停，部位卡到收盤承擔最大損失）"
                    )
                if stopped_out:
                    # V4 下，"stopped" 大多是 trail stop 鎖獲利出場（非虧損停損）
                    is_profit = float(r["exit_price"]) < float(r["entry_price"])
                    if is_profit:
                        st.markdown(":green[✓ Trail stop 觸發 — 鎖獲利出場（非虧損）]")
                    else:
                        st.markdown(":red[⚠ 真實停損觸發（exit > entry）]")

            with c4:
                st.markdown("**出場**")
                st.markdown(f"時間：`{'11:30 時間出場' if not stopped_out else 'Trail/停損觸發'}`")
                st.markdown(f"出場價：`{r['exit_price']:.2f}`")
                ret_pct = float(r["return"]) * 100
                arrow = "▲" if ret_pct > 0 else "▼"
                st.markdown(
                    f"單筆報酬：<span style='color:{'#4ade80' if ret_pct>0 else '#f87171'}'>"
                    f"`{arrow} {abs(ret_pct):.2f}%`</span>",
                    unsafe_allow_html=True
                )

            # 價格視覺 bar
            prices = {
                "開盤": r["day_open"],
                "早盤高": r["morning_high"],
                "進場(空)": r["entry_price"],
                "出場": r["exit_price"],
                "停損": r["stop_price"],
                "漲停": r["limit_price"],
            }
            colors = {
                "開盤": "#94a3b8",
                "早盤高": "#fbbf24",
                "進場(空)": "#60a5fa",
                "出場": "#4ade80" if not stopped_out else "#f87171",
                "停損": "#f87171",
                "漲停": "#7c3aed",
            }
            price_df = pd.DataFrame(
                [{"label": k, "price": v, "color": colors[k]} for k, v in prices.items()]
            )
            bar = alt.Chart(price_df).mark_bar(size=20).encode(
                x=alt.X("label:N", sort=list(prices.keys()), title=None,
                         axis=alt.Axis(labelColor="#94a3b8")),
                y=alt.Y("price:Q",
                         scale=alt.Scale(domain=[min(prices.values())*0.997,
                                                  max(prices.values())*1.003]),
                         title="價格"),
                color=alt.Color("color:N", scale=None),
                tooltip=["label", alt.Tooltip("price:Q", format=".2f")]
            ).properties(height=140)
            st.altair_chart(bar, use_container_width=True)
            st.divider()

# ── 彙總表格 ────────────────────────────────────────────────
st.subheader("📊 當日彙總表")

if not selected.empty:
    show = selected[[
        "ticker", "gap_pct", "stock_aor", "rank_score",
        "entry_min", "entry_price", "stop_price", "limit_price", "gap_to_limit",
        "exit_price", "return", "stopped_out", "lots", "net"
    ]].copy()
    show.insert(1, "名稱", show["ticker"].map(lambda t: names.get(str(t), "")))
    show["gap_pct"]      = show["gap_pct"].map(lambda x: f"{x*100:+.2f}%")
    show["stock_aor"]    = show["stock_aor"].map(lambda x: f"{x*100:+.2f}%")
    show["rank_score"]   = show["rank_score"].map(lambda x: f"{x:.2f}")
    show["entry_min"]    = show["entry_min"].map(lambda x: fmt_min(int(x)))
    show["gap_to_limit"] = show["gap_to_limit"].map(lambda x: f"{x*100:.2f}%")
    show["return"]       = show["return"].map(lambda x: f"{x*100:+.2f}%")
    # V4+ Trail Stop: stopped_out 大多是 trail 鎖獲利（exit < entry, net > 0）
    def _status(row):
        stopped = bool(row["stopped_out"])
        is_loss = float(row["exit_price"]) > float(row["entry_price"])
        if stopped:
            return "⚠ 真實停損" if is_loss else "✓ Trail 鎖獲利"
        return "⏱ 時間出場" if not is_loss else "⏱ 時間-虧損"
    show["stopped_out"] = selected.apply(_status, axis=1)
    show["lots"]         = show["lots"].map(lambda x: f"{int(x)} 張")
    show["net"]          = show["net"].map(lambda x: f"{x:+,.0f}")
    # 漲停風險欄
    show["漲停風險"] = selected["stop_vs_limit"].map(
        lambda x: "🔴 停損超漲停" if x > 0 else ("🟡 接近漲停" if x > -0.02 else "✓")
    )
    show = show.rename(columns={
        "ticker": "代號", "gap_pct": "跳空%", "stock_aor": "漲不動%",
        "rank_score": "排序分", "entry_min": "進場時間", "entry_price": "進場價",
        "stop_price": "停損價", "limit_price": "漲停價", "gap_to_limit": "距漲停",
        "exit_price": "出場價", "return": "單筆報酬", "stopped_out": "狀態",
        "lots": "口數", "net": "淨 PnL (元)",
    })
    st.dataframe(show, use_container_width=True, hide_index=True)

# ── 候選但未選入 ────────────────────────────────────────────
excluded = day_sig.iloc[n_max:] if len(day_sig) > n_max else pd.DataFrame()
if not excluded.empty:
    with st.expander(f"📋 候選但未選入（{len(excluded)} 檔）"):
        ex = excluded[["ticker", "gap_pct", "stock_aor", "rank_score",
                        "entry_min", "entry_price", "gap_to_limit"]].copy()
        ex.insert(1, "名稱", ex["ticker"].map(lambda t: names.get(str(t), "")))
        ex["gap_pct"]      = ex["gap_pct"].map(lambda x: f"{x*100:+.2f}%")
        ex["stock_aor"]    = ex["stock_aor"].map(lambda x: f"{x*100:+.2f}%")
        ex["rank_score"]   = ex["rank_score"].map(lambda x: f"{x:.2f}")
        ex["entry_min"]    = ex["entry_min"].map(lambda x: fmt_min(int(x)))
        ex["gap_to_limit"] = ex["gap_to_limit"].map(lambda x: f"{x*100:.2f}%")
        ex["漲停風險"] = excluded["stop_vs_limit"].map(
            lambda x: "🔴 停損超漲停" if x > 0 else ("🟡 接近漲停" if x > -0.02 else "✓")
        )
        st.dataframe(ex.rename(columns={
            "ticker": "代號", "gap_pct": "跳空%", "stock_aor": "漲不動%",
            "rank_score": "排序分", "entry_min": "進場時間",
            "entry_price": "進場價", "gap_to_limit": "距漲停",
        }), use_container_width=True, hide_index=True)

# ── PnL 條形圖 ───────────────────────────────────────────────
if not selected.empty and n_fills > 0:
    st.subheader("💰 個股損益")
    pcdf = selected[selected["lots"] > 0][["ticker", "net"]].copy()
    pcdf["label"] = pcdf["ticker"].map(lambda t: f"{t} {names.get(str(t), '')}")
    pcdf["color"] = pcdf["net"].apply(lambda x: "#4ade80" if x > 0 else "#f87171")
    bar2 = alt.Chart(pcdf).mark_bar().encode(
        x=alt.X("label:N", title="股票", sort="-y"),
        y=alt.Y("net:Q", title="淨 PnL (元)"),
        color=alt.Color("color:N", scale=None),
        tooltip=["label", alt.Tooltip("net:Q", format=",.0f", title="淨 PnL")]
    ).properties(height=220)
    st.altair_chart(bar2, use_container_width=True)

st.divider()
st.caption(
    f"🩸 守不住開盤 日誌 v8.0 (V8 aor 3.0%) · "
    f"今日候選 {len(day_sig)} 檔（漲不動 < {aor_thr*100:.2f}%），選入 top-{n_max}  |  "
    "Trail stop + 11:30 強制平倉 + 漲停安全 + Sharpe 29.28"
)
