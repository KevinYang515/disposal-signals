# -*- coding: utf-8 -*-
r"""NH20 突破監控：13:20 盤中檢查今天哪些觀察股符合突破條件。

狀態：個人研究監看（未實單驗證）。
資料：收盤後產生觀察名單；盤中抓即時快照後更新。
"""

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NH20 突破監控", page_icon="📈", layout="wide")
D = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nh20")


@st.cache_data(ttl=60)
def rd(name):
    p = os.path.join(D, name)
    try:
        return pd.read_csv(p, dtype={"代號": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data(ttl=60)
def rj(name):
    p = os.path.join(D, name)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


meta, im = rj("nh20_meta.json"), rj("nh20_intraday_meta.json")
st.title("📈 NH20 突破監控")
st.caption("規則：收盤創 20 日新高（首日）＋突破前 60 日成交密集區上緣＋半年線不下彎＋突破日量 ≥ 1.5 倍 20 日均量；"
           "加權報酬指數在季線上才進場；收盤漲停不買；尾盤（13:20–13:25）判斷、收盤買；"
           "出場＝收盤跌破進場後最大量 K 棒低點 → 隔日開盤賣。主卡＝6 檔、電子、20 日均額 ≥ 1 億。")

c1, c2, c3, c4 = st.columns(4)
c1.metric("名單資料日（前一交易日收盤）", meta.get("data_date", "—"))
c2.metric("盤中快照時間", im.get("snapshot_at", "尚未更新"))
gate = meta.get("gate")
c3.metric("大盤濾網", "可新進場" if gate else "不新進場",
          f"{meta.get('taiex', 0):,.0f} vs 季線 {meta.get('taiex_ma60', 0):,.0f}" if meta else None)
c4.metric("快照時點已成交比例（歷史中位數）", f"{im.get('vol_share', 0) * 100:.1f}%" if im else "—")
if im and meta and not str(im.get("snapshot_at", "")).startswith(pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")):
    st.warning("盤中快照不是今天的資料，請以名單資料為準。")

card = st.radio("卡別", ["主卡", "10檔卡", "全部"], horizontal=True)
f = (lambda d: d if card == "全部" or d.empty else d[d["卡別"] == card])

intra = f(rd("nh20_intraday.csv"))
st.subheader("① 盤中狀態")
if intra.empty:
    st.info("今天尚未有盤中快照。")
else:
    order = {"✅ 符合": 0, "🟡 價到量不足": 1, "🟡 接近（差<1%）": 2, "⛔ 漲停（不買）": 3, "—": 4}
    intra = intra.assign(_o=intra["狀態"].map(order).fillna(9)).sort_values(["_o", "距觸發pct"], ascending=[True, False]).drop(columns="_o")
    k1, k2, k3 = st.columns(3)
    k1.metric("✅ 符合", int((intra["狀態"] == "✅ 符合").sum()))
    k2.metric("🟡 價到量不足 / 接近", int(intra["狀態"].str.startswith("🟡").sum()))
    k3.metric("⛔ 漲停", int((intra["狀態"] == "⛔ 漲停（不買）").sum()))
    show_all = st.checkbox("顯示全部觀察股（含未接近）", value=False)
    view = intra if show_all else intra[intra["狀態"] != "—"]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("「目前應達張」＝20 日均量 × 1.5 × 快照時點歷史已成交比例。現價離觸發價 < 0.5% 的，收盤可能翻盤。")

lu = rd("nh20_limitup_nextopen.csv")
st.subheader("② 昨日收盤漲停的突破（10 檔卡：今日開盤買，開盤鎖漲停則跳過）")
if lu.empty:
    st.write("無")
else:
    st.dataframe(lu, use_container_width=True, hide_index=True)

st.subheader("③ 訊號追蹤（9/1 起每個訊號獨立追蹤，非實際持股；實際只持有名額內的檔數）")
pi = f(rd("nh20_intraday_positions.csv"))
pos = pi if not pi.empty else f(rd("nh20_positions.csv"))
if not pos.empty:
    st.dataframe(pos, use_container_width=True, hide_index=True)

with st.expander("完整觀察名單（收盤後產生）"):
    w = f(rd("nh20_watch.csv"))
    if w.empty:
        st.write("無")
    else:
        st.dataframe(w.sort_values("需漲pct"), use_container_width=True, hide_index=True)
