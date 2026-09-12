# -*- coding: utf-8 -*-
r"""分點買賣超排行查詢：自選分點、自選截止日、自選往前幾個交易日、自選金額門檻，
看該分點在這段期間對各股票的估計淨買賣超金額排行。

資料：`data/broker_flow_recent.parquet`（近 45 個交易日、單日 |估計淨買賣超金額| >= 500 萬、
排除 ETF/槓桿反向、只收錄「我們研究過的主力大戶分點」約 20 個的逐日「分點×股票」明細），
由 `E:\stock\scripts\bigmoney_watchlist_daily.py` 每個交易日晚上重算。
估計金額 = 當日淨買賣超張數 × 當日收盤價 / 10（單位：萬元）；真實成交價散落盤中，這是概估。
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="分點買賣超排行", page_icon="🔎", layout="wide")

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "broker_flow_recent.parquet")


@st.cache_data(ttl=600)
def load_flow(_v: str = "roster19-influence-float"):
    """_v 只用來讓精選名單改版時強制失效舊快取。"""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df


flow = load_flow()

st.title("🔎 分點買賣超排行查詢")

if flow.empty:
    st.error("找不到 `data/broker_flow_recent.parquet`，資料尚未產生。")
    st.stop()

dmin, dmax = flow["date"].min(), flow["date"].max()
st.caption(
    f"資料範圍 **{dmin.date()} ~ {dmax.date()}**（近 {flow['date'].nunique()} 個交易日）｜"
    f"只收錄我們研究過的主力大戶分點 {flow['broker'].nunique()} 個、已排除 ETF/槓桿反向｜"
    "單日 |估計淨買賣超金額| ≥ 500 萬才收錄（可在下方再拉高門檻）｜"
    "金額為估計值（淨張數 × 當日收盤價 ÷ 10，單位：萬元）。分點揭露資料通常落後 1-2 個交易日。"
)

brokers = sorted(flow["broker"].unique().tolist())
default_ix = brokers.index("凱基-站前") if "凱基-站前" in brokers else 0
dates_desc = sorted(flow["date"].dt.date.unique().tolist(), reverse=True)

c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
with c1:
    sel_broker = st.selectbox("分點（可直接打字搜尋）", brokers, index=default_ix)
with c2:
    sel_end = st.selectbox("截止日期", dates_desc, index=0)
with c3:
    lookback = st.number_input("往前幾個交易日（含當日）", min_value=1, max_value=flow["date"].nunique(), value=5, step=1)
with c4:
    min_wan = st.number_input("期間累計金額門檻（萬）", min_value=0, max_value=200000, value=0, step=500,
                              help="只顯示期間淨買超或淨賣超累計金額絕對值 ≥ 此值的股票；0 = 不過濾")

# 決定日期視窗
all_days = sorted(flow["date"].dt.normalize().unique())
end_ts = pd.Timestamp(sel_end)
days_upto = [d for d in all_days if d <= end_ts]
window_days = days_upto[-int(lookback):]
if not window_days:
    st.warning("所選截止日在資料範圍之前，沒有資料。")
    st.stop()
win_start = window_days[0]

view = flow[(flow["broker"] == sel_broker) & (flow["date"] >= win_start) & (flow["date"] <= end_ts)]
if view.empty:
    st.info(f"{sel_broker} 在 {win_start.date()} ~ {sel_end} 這段期間沒有 ≥ 500 萬的買賣超紀錄。")
    st.stop()

agg = (
    view.groupby(["stock_id", "name"], as_index=False)
    .agg(淨買賣超金額_萬=("net_wan", "sum"), 淨張數=("net_lots", "sum"), 出現天數=("date", "nunique"),
         _turnover_sum=("turnover_wan", "sum"), _shares_out=("shares_out", "max"))
    .sort_values("淨買賣超金額_萬", ascending=False)
)
agg["淨買賣超金額_萬"] = agg["淨買賣超金額_萬"].round(1)
# 期間影響力% = 期間淨買賣超金額加總 / 期間全市場成交金額加總；佔股本比% = 期間淨股數 / 已發行股數
agg["期間影響力_pct"] = np.where(agg["_turnover_sum"] > 0, agg["淨買賣超金額_萬"] / agg["_turnover_sum"] * 100, np.nan).round(2)
agg["期間佔股本比_pct"] = np.where(
    agg["_shares_out"] > 0, agg["淨張數"] * 1000 / agg["_shares_out"] * 100, np.nan
).round(4)
agg = agg.drop(columns=["_turnover_sum", "_shares_out"])
if min_wan > 0:
    agg = agg[agg["淨買賣超金額_萬"].abs() >= min_wan]

buy_side = agg[agg["淨買賣超金額_萬"] > 0]
sell_side = agg[agg["淨買賣超金額_萬"] < 0].sort_values("淨買賣超金額_萬")

m1, m2, m3, m4 = st.columns(4)
m1.metric("期間", f"{win_start.date()} ~ {sel_end}", delta=f"{len(window_days)} 個交易日", delta_color="off")
m2.metric("淨買超總額", f"{buy_side['淨買賣超金額_萬'].sum():,.0f} 萬")
m3.metric("淨賣超總額", f"{sell_side['淨買賣超金額_萬'].sum():,.0f} 萬")
m4.metric("涉及股票數", f"{len(agg)} 檔")

st.caption(
    "📎 **期間影響力%** 與 **期間佔股本比%** 是兩個輔助參考欄位，不是驗證過的訊號——"
    "分別全市場嚴謹回測過『佔當日成交量影響力』(PLAYBOOK §6i) 跟『佔已發行股數比例』(§6o)，"
    "兩者單獨拿來當進場門檻都測不出可重複優勢。放在這裡純粹當背景資訊：同樣的金額，"
    "對小型/低量股票的影響力或持股比例會比大型股高很多，看這兩欄有助於判斷這筆買賣的相對份量。"
)

fmt_cols = {"淨買賣超金額_萬": "{:,.0f}", "淨張數": "{:,.0f}", "期間影響力_pct": "{:+.2f}%", "期間佔股本比_pct": "{:+.3f}%"}

st.divider()
col_b, col_s = st.columns(2)
with col_b:
    st.markdown(f"#### 🟥 淨買超前 25（{sel_broker}）")
    show_b = buy_side.head(25).rename(columns={"stock_id": "代號", "name": "名稱"})
    st.dataframe(
        show_b.style.format(fmt_cols, na_rep="-"),
        use_container_width=True, height=560, hide_index=True,
    )
with col_s:
    st.markdown(f"#### 🟩 淨賣超前 25（{sel_broker}）")
    show_s = sell_side.head(25).rename(columns={"stock_id": "代號", "name": "名稱"})
    st.dataframe(
        show_s.style.format(fmt_cols, na_rep="-"),
        use_container_width=True, height=560, hide_index=True,
    )

st.divider()
st.markdown("#### 📋 完整明細（該分點該期間所有 ≥ 500 萬紀錄，含當日影響力%／佔股本比%）")
detail = view[["date", "broker", "stock_id", "name", "net_lots", "net_wan", "influence_pct", "float_pct"]].sort_values(
    ["date", "net_wan"], ascending=[False, False]
).rename(columns={
    "date": "日期", "broker": "分點", "stock_id": "代號", "name": "名稱",
    "net_lots": "淨張數", "net_wan": "估計淨買賣超金額_萬",
    "influence_pct": "當日影響力_pct", "float_pct": "當日佔股本比_pct",
}).copy()
detail["日期"] = detail["日期"].dt.strftime("%Y-%m-%d")
st.dataframe(
    detail.style.format({"估計淨買賣超金額_萬": "{:,.0f}", "淨張數": "{:,.0f}",
                          "當日影響力_pct": "{:+.2f}%", "當日佔股本比_pct": "{:+.3f}%"}, na_rep="-"),
    use_container_width=True, height=420, hide_index=True,
)
st.download_button(
    "📥 下載此分點此期間明細 CSV",
    detail.to_csv(index=False, encoding="utf-8-sig"),
    f"{sel_broker}_{win_start.date()}_{sel_end}.csv", "text/csv", key="dl_flow",
)
