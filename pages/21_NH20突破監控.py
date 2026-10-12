# -*- coding: utf-8 -*-
r"""NH20 突破監控：每天 13:15／13:21 檢查觀察股是否符合「帶量突破」條件。

狀態：個人研究監看（未實單驗證）。
資料：收盤後產生觀察名單；盤中抓即時快照後更新。
"""

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NH20 突破監控", page_icon="📈", layout="wide")
D = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nh20")
NC = st.column_config.NumberColumn
TC = st.column_config.TextColumn


@st.cache_data(ttl=60)
def rd(name):
    try:
        return pd.read_csv(os.path.join(D, name), dtype={"代號": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


@st.cache_data(ttl=60)
def rj(name):
    try:
        with open(os.path.join(D, name), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def table(df, cols, cfg, height=None):
    if df.empty:
        st.write("目前沒有。")
        return
    cols = [c for c in cols if c in df.columns]
    kw = dict(use_container_width=True, hide_index=True, column_config=cfg)
    if height:
        kw["height"] = height
    st.dataframe(df[cols], **kw)


meta, im = rj("nh20_meta.json"), rj("nh20_intraday_meta.json")
today = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
fresh = bool(im) and str(im.get("snapshot_at", "")).startswith(today)

# ---------------- 標題與今日狀態 ----------------
st.title("📈 NH20 突破監控")
st.markdown("找出**今天收盤可能帶量突破**的股票。每天 **13:15、13:21** 用即時報價更新一次，"
            "**13:24 左右做決定、收盤前買進**。")

c1, c2, c3 = st.columns(3)
gate = meta.get("gate")
c1.metric("大盤", "✅ 可以買新股" if gate else "⛔ 暫停買新股",
          f"加權 {meta.get('taiex', 0):,.0f}／季線 {meta.get('taiex_ma60', 0):,.0f}" if meta else None, delta_color="off")
c2.metric("盤中最後更新", im.get("snapshot_at", "")[11:16] if fresh else "今天尚未更新",
          None if fresh else "每天 13:15、13:21 更新", delta_color="off")
c3.metric("名單依據", f"{meta.get('data_date', '—')} 收盤")
if gate is False:
    st.error("大盤在季線下：今天**不買新股**，只處理手上持股的出場。")

card_lbl = st.radio("看哪一組", ["主卡（最多 6 檔・電子・成交額 ≥ 1 億）", "10 檔卡（不限產業）", "全部"], horizontal=True)
card = {"主": "主卡", "1": "10檔卡"}.get(card_lbl[0], "全部")


def f(d):
    if card == "全部" or d.empty or "卡別" not in d.columns:
        return d
    return d[d["卡別"] == card]


tab1, tab2, tab3, tab4 = st.tabs(["🎯 今天的買點", "📋 持股與出場", "🗂 完整觀察名單", "📖 規則說明"])

# ---------------- 1. 今天的買點 ----------------
with tab1:
    intra = f(rd("nh20_intraday.csv"))
    if not fresh or intra.empty:
        st.info("盤中資料 13:15 後才會出現。先看下面「最接近突破」的股票，盤中留意它們。")
        w = f(rd("nh20_watch.csv"))
        if not w.empty:
            w = w.sort_values("需漲pct").head(20)
        table(w, ["代號", "名稱", "昨收", "觸發價", "需漲pct", "需量張1320"], {
            "昨收": NC("昨收", format="%.2f"), "觸發價": NC("收盤要高於", format="%.2f"),
            "需漲pct": NC("還要漲", format="%.1f%%"), "需量張1320": NC("13:20 成交量要 ≥（張）", format="%d")})
    else:
        ok = intra[intra["狀態"] == "✅ 符合"].sort_values("距觸發pct")
        near = intra[intra["狀態"].str.startswith("🟡")].sort_values("距觸發pct", ascending=False)
        lim = intra[intra["狀態"] == "⛔ 漲停（不買）"]
        k1, k2, k3 = st.columns(3)
        k1.metric("✅ 符合，可買", len(ok))
        k2.metric("🟡 差一點", len(near))
        k3.metric("⛔ 漲停買不到", len(lim))
        cfg = {"現價": NC("現價", format="%.2f"), "觸發價": NC("收盤要高於", format="%.2f"),
               "距觸發pct": NC("高於觸發價", format="%+.1f%%"), "漲幅pct": NC("今日漲幅", format="%+.1f%%"),
               "量達成率": st.column_config.ProgressColumn("成交量達成（滿格=2倍）", format="%.2f", min_value=0, max_value=2),
               "累計量張": NC("目前成交（張）", format="%d"), "目前應達張": NC("現在應達（張）", format="%d"),
               "收盤前提醒": TC("提醒")}
        cols = ["代號", "名稱", "現價", "觸發價", "距觸發pct", "漲幅pct", "量達成率", "累計量張", "目前應達張", "收盤前提醒"]
        st.markdown("#### ✅ 符合條件（價格站上、量也夠）")
        table(ok.assign(量達成率=ok["量達成率"].clip(upper=2)), cols, cfg)
        if not ok.empty:
            st.caption("「成交量達成」1.00 代表剛好達標。「提醒」有字的，價格只比觸發價高一點點，收盤前可能跌回去，13:24 再確認一次。")
        st.markdown("#### 🟡 差一點（價格到了但量不夠，或價格還差不到 1%）")
        table(near.assign(量達成率=near["量達成率"].clip(upper=2)), ["狀態"] + cols, cfg)
        if not lim.empty:
            st.markdown("#### ⛔ 已經漲停（規則上不追）")
            table(lim, ["代號", "名稱", "現價", "漲幅pct"], cfg)
    lu = rd("nh20_limitup_nextopen.csv")
    if not lu.empty and card != "主卡":
        st.markdown("#### 🌅 昨天收盤漲停的突破（10 檔卡：今天開盤買；開盤就鎖漲停則跳過）")
        table(lu, ["代號", "名稱", "收盤", "防守價"], {"收盤": NC("昨收", format="%.2f"), "防守價": NC("跌破就賣", format="%.2f")})

# ---------------- 2. 持股與出場 ----------------
with tab2:
    st.warning("這裡列的是 9/1 以來**每個訊號都買**的追蹤結果，不是實際持股。"
               "實際上主卡最多只持有 6 檔，請對照你自己的持股看防守價。")
    pi = f(rd("nh20_intraday_positions.csv")) if fresh else pd.DataFrame()
    pos = pi if not pi.empty else f(rd("nh20_positions.csv"))
    cfg = {"買價": NC("買價", format="%.2f"), "防守價": NC("跌破就賣", format="%.2f"), "現價": NC("現價", format="%.2f"),
           "距防守pct": NC("離防守還有", format="%.1f%%"), "報酬pct": NC("目前報酬", format="%+.1f%%"), "買進日": TC("買進日")}
    cols = ["代號", "名稱", "買進日", "買價", "現價", "報酬pct", "防守價", "距防守pct"]
    if pos.empty:
        st.write("目前沒有。")
    else:
        stt = pos["狀態"].astype(str) if "狀態" in pos.columns else pd.Series("", index=pos.index)
        sell, warn = pos[stt.str.contains("賣")], pos[stt.str.contains("⚠️")]
        hold = pos.drop(sell.index.union(warn.index))
        st.markdown("#### 🔴 今天開盤要賣（昨天收盤已跌破防守價）")
        table(sell, cols, cfg)
        if not warn.empty:
            st.markdown("#### ⚠️ 現價已低於防守價（以收盤為準，收盤仍低於就明天開盤賣）")
            table(warn, cols, cfg)
        st.markdown("#### 🟢 持有中（離防守越近的排越前面）")
        if "距防守pct" in hold.columns:
            hold = hold.sort_values("距防守pct")
        table(hold, cols, cfg, height=420)

# ---------------- 3. 完整觀察名單 ----------------
with tab3:
    st.caption("昨天收盤後算好的全部候選。今天收盤只要「高於觸發價」且「成交量夠」，就是買進訊號。")
    w = f(rd("nh20_watch.csv"))
    if not w.empty:
        w = w.sort_values("需漲pct")
    table(w, ["代號", "名稱", "產業", "昨收", "觸發價", "需漲pct", "均量張", "需量張1320", "需量張1324"], {
        "昨收": NC("昨收", format="%.2f"), "觸發價": NC("收盤要高於", format="%.2f"), "需漲pct": NC("還要漲", format="%.1f%%"),
        "均量張": NC("20 日均量（張）", format="%d"), "需量張1320": NC("13:20 要 ≥（張）", format="%d"),
        "需量張1324": NC("13:24 要 ≥（張）", format="%d")}, height=500)

# ---------------- 4. 規則說明 ----------------
with tab4:
    st.markdown("""
**買進（收盤時全部成立）**
1. **突破**：收盤價高於「前 20 天最高價」，也高於「前 60 天成交最密集的價格區間上緣」（套牢區），而且是第一天站上。
2. **帶量**：當天成交量 ≥ 20 日均量的 1.5 倍（13:20 看時約要 1.41 倍，13:24 約 1.43 倍）。
3. **長期趨勢沒有往下**：半年線（180 日均線）沒有下彎。
4. **大盤**：加權報酬指數在季線（60 日均線）之上才買新股。
5. **收盤漲停不買**（買不到）。10 檔卡例外：昨天收盤漲停的突破，今天開盤買。
6. **主卡**另外限定：電子股、20 日平均成交額 ≥ 1 億、最多 6 檔，滿了就不買新的。

**賣出**
- 進場後，找出「量最大的那根 K 棒」，它的最低價就是防守價；之後如果出現更大量的 K 棒，防守價就換成那根的低點。
- **收盤跌破防守價 → 隔天開盤賣**。沒跌破就一直抱，沒有時間限制。

**為什麼這樣設計**：突破套牢區，代表之前買的人解套、賣壓變小；突破後追進最多的人的成本（最大量 K 棒低點）被跌破，代表他們被套住、賣壓出現。

⚠️ 這是個人研究用的監看頁面，回測結果不代表未來，也還沒有實單驗證。
""")
