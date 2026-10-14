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
st.markdown("找出**今天收盤可能帶量突破**的股票。盤中 **09:30、10:30、11:30、12:30（參考）與 13:15、13:21（決策）** 用即時報價更新，"
            "**13:24 左右做決定、收盤前買進**。")

c1, c2, c3 = st.columns(3)
gate = meta.get("gate")
c1.metric("大盤", "✅ 可以買新股" if gate else "⛔ 暫停買新股",
          f"加權 {meta.get('taiex', 0):,.0f}／季線 {meta.get('taiex_ma60', 0):,.0f}" if meta else None, delta_color="off")
c2.metric("盤中最後更新", im.get("snapshot_at", "")[11:16] if fresh else "今天尚未更新",
          None if fresh else "09:30 起每小時、13:15、13:21 更新", delta_color="off")
c3.metric("名單依據", f"{meta.get('data_date', '—')} 收盤")
if gate is False:
    st.error("大盤在季線下：今天**不買新股**，只處理手上持股的出場。")

card_lbl = st.radio("看哪一組", ["主卡（最多 6 檔・電子・成交額 ≥ 1 億）", "10 檔卡（不限產業）", "全部"], horizontal=True)
card = {"主": "主卡", "1": "10檔卡"}.get(card_lbl[0], "全部")


def f(d):
    if card == "全部" or d.empty or "卡別" not in d.columns:
        return d
    return d[d["卡別"] == card]


tab1, tab2, tab5, tab3, tab4 = st.tabs(["🎯 今天的買點", "📋 持股與出場", "📜 歷史紀錄", "🗂 完整觀察名單", "📖 規則說明"])

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
        if im.get("snapshot_at", "")[11:16] < "13:00":
            st.warning(f"⏰ 這是早盤 {im.get('snapshot_at', '')[11:16]} 的參考快照：早盤突破常在收盤前跌回，"
                       "**規則上仍以 13:20 後的狀態為準**（量的門檻已依時間換算）。")
        ok = intra[intra["狀態"] == "✅ 符合"].sort_values("距觸發pct")
        near = intra[intra["狀態"].str.startswith("🟡")].sort_values("距觸發pct", ascending=False)
        lim = intra[intra["狀態"] == "⛔ 漲停（不買）"]
        zf = intra[intra["狀態"].str.startswith("⚠️")]
        k1, k2, k3 = st.columns(3)
        k1.metric("✅ 符合，可買", len(ok))
        k2.metric("🟡 差一點", len(near))
        k3.metric("⛔ 漲停買不到", len(lim))
        cfg = {"現價": NC("現價", format="%.2f"), "觸發價": NC("收盤要高於", format="%.2f"),
               "距觸發pct": NC("高於觸發價", format="%+.1f%%"), "漲幅pct": NC("今日漲幅", format="%+.1f%%"),
               "量達成率": st.column_config.ProgressColumn("成交量達成（滿格=2倍）", format="%.2f", min_value=0, max_value=2),
               "累計量張": NC("目前成交（張）", format="%d"), "目前應達張": NC("現在應達（張）", format="%d"),
               "收盤前提醒": TC("提醒"), "套牢提醒": TC("套牢／長上影")}
        cols = ["代號", "名稱", "現價", "觸發價", "距觸發pct", "漲幅pct", "量達成率", "累計量張", "目前應達張", "套牢提醒", "收盤前提醒"]
        st.markdown("#### ✅ 符合條件（價格站上、量也夠）")
        table(ok.assign(量達成率=ok["量達成率"].clip(upper=2)), cols, cfg)
        if not ok.empty:
            st.caption("「成交量達成」1.00 代表剛好達標。「提醒」有字的，價格只比觸發價高一點點，收盤前可能跌回去，13:24 再確認一次。")
        st.markdown("#### 🟡 差一點（價格到了但量不夠，或價格還差不到 1%）")
        table(near.assign(量達成率=near["量達成率"].clip(upper=2)), ["狀態"] + cols, cfg)
        if not zf.empty:
            st.markdown("#### ⚠️ 價量達標，但撞上套牢區或爆量長上影（主卡規則上不買）")
            table(zf.assign(量達成率=zf["量達成率"].clip(upper=2)), cols, cfg)
            st.caption("上方 5% 內有一大塊過去 120 天的成交密集區（套牢區），或今天爆量（≥2 倍）卻收長上影（上影佔當天高低差一半以上）。"
                       "2020 年後回測，這類突破勝率約 23%、平均幾乎不賺。10 檔卡只做提醒、不排除。")
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

# ---------------- 5. 歷史紀錄 ----------------
with tab5:
    st.caption("2019 年起每一個符合規則的訊號（每個都買、各自計算，不受名額限制）。尚未出場的以最新收盤價計算。"
               "報酬已扣手續費與交易稅。用下面的條件篩選，找出哪些時期、產業、量能的訊號比較好。")
    h = f(rd("nh20_history.csv"))
    if h.empty:
        st.write("尚無資料。")
    else:
        h["訊號日"] = pd.to_datetime(h["訊號日"])
        h["年"] = h["訊號日"].dt.year
        h["年月"] = h["訊號日"].dt.strftime("%Y-%m")
        mode = st.radio("選擇期間的方式", ["依年／月", "依日期區間"], horizontal=True)
        if mode == "依年／月":
            ys = sorted(h["年"].unique(), reverse=True)
            a1, a2 = st.columns(2)
            ysel = a1.multiselect("年", ys, default=ys[:1])
            ms = sorted(h.loc[h["年"].isin(ysel), "年月"].unique(), reverse=True) if ysel else []
            msel = a2.multiselect("月（不選＝整年）", ms)
            hh = h[h["年"].isin(ysel)] if ysel else h.iloc[0:0]
            if msel:
                hh = hh[hh["年月"].isin(msel)]
        else:
            d0, d1 = h["訊號日"].min().date(), h["訊號日"].max().date()
            rng = st.date_input("訊號日區間", (max(d0, (h["訊號日"].max() - pd.Timedelta(days=30)).date()), d1), min_value=d0, max_value=d1)
            if isinstance(rng, (tuple, list)) and len(rng) == 2:
                hh = h[(h["訊號日"].dt.date >= rng[0]) & (h["訊號日"].dt.date <= rng[1])]
            else:
                hh = h[h["訊號日"].dt.date == (rng[0] if isinstance(rng, (tuple, list)) else rng)]
        with st.expander("更多篩選（產業、量比、代號）"):
            b1, b2, b3 = st.columns(3)
            inds = sorted(hh["產業"].dropna().unique())
            isel = b1.multiselect("產業", inds)
            vr = b2.slider("突破日量比（相對 20 日均量）", 1.5, 10.0, (1.5, 10.0), 0.1)
            code = b3.text_input("代號或名稱包含")
            zsel = st.radio("套牢區／長上影旗標", ["全部", "只看有旗標", "只看沒旗標"], horizontal=True)
        if isel:
            hh = hh[hh["產業"].isin(isel)]
        hh = hh[(hh["量比"] >= vr[0]) & ((hh["量比"] <= vr[1]) | (vr[1] >= 10.0))]
        if code:
            hh = hh[hh["代號"].str.contains(code) | hh["名稱"].astype(str).str.contains(code)]
        if "套牢旗標" in hh.columns and zsel != "全部":
            has = hh["套牢旗標"].fillna("").astype(str).str.len() > 0
            hh = hh[has] if zsel == "只看有旗標" else hh[~has]
        if hh.empty:
            st.write("這個條件下沒有訊號。")
        else:
            r = hh["報酬pct"]
            win, loss = r[r > 0], r[r <= 0]
            k = st.columns(5)
            k[0].metric("訊號筆數", len(hh))
            k[1].metric("勝率", f"{(r > 0).mean() * 100:.0f}%")
            k[2].metric("平均每筆報酬", f"{r.mean():+.1f}%")
            k[3].metric("賺賠比", f"{win.mean() / abs(loss.mean()):.1f}" if len(win) and len(loss) and loss.mean() != 0 else "—")
            k[4].metric("平均持有", f"{hh['持有天數'].mean():.0f} 天")
            k2 = st.columns(5)
            k2[0].metric("平均獲利", f"{win.mean():+.1f}%" if len(win) else "—")
            k2[1].metric("平均虧損", f"{loss.mean():+.1f}%" if len(loss) else "—")
            k2[2].metric("最大單筆獲利", f"{r.max():+.1f}%")
            k2[3].metric("最大單筆虧損", f"{r.min():+.1f}%")
            k2[4].metric("仍持有", int((hh["狀態"] != "已出場").sum()))
            st.markdown("#### 每月統計")
            g = hh.groupby("年月")["報酬pct"]
            mt = pd.DataFrame({"筆數": g.size(), "勝率": g.apply(lambda x: (x > 0).mean() * 100), "平均報酬": g.mean(),
                               "最好一筆": g.max(), "最差一筆": g.min()}).reset_index().sort_values("年月", ascending=False)
            st.dataframe(mt, use_container_width=True, hide_index=True, column_config={
                "勝率": NC("勝率", format="%.0f%%"), "平均報酬": NC("平均每筆報酬", format="%+.1f%%"),
                "最好一筆": NC("最好一筆", format="%+.1f%%"), "最差一筆": NC("最差一筆", format="%+.1f%%")})
            st.markdown("#### 累積報酬（每筆投入相同金額、依訊號日累加）")
            cum = hh.sort_values("訊號日").groupby("訊號日")["報酬pct"].sum().cumsum().rename("累積報酬（%，每筆一單位）")
            st.line_chart(cum)
            st.markdown("#### 逐筆明細")
            show = hh.sort_values("訊號日", ascending=False).assign(訊號日=lambda x: x["訊號日"].dt.strftime("%Y-%m-%d"))
            cols = ["訊號日", "卡別", "代號", "名稱", "產業", "買價", "觸發價", "量比", "出場日", "賣價", "報酬pct", "持有天數", "最大漲幅pct", "最大跌幅pct", "狀態", "套牢旗標"]
            cols = [c for c in cols if c in show.columns]
            st.dataframe(show[cols], use_container_width=True, hide_index=True, height=450, column_config={
                "買價": NC("買價", format="%.2f"), "觸發價": NC("觸發價", format="%.2f"), "量比": NC("量比", format="%.1f×"),
                "賣價": NC("賣價／現價", format="%.2f"), "報酬pct": NC("報酬", format="%+.1f%%"), "持有天數": NC("持有天數", format="%d"),
                "最大漲幅pct": NC("期間最大漲幅", format="%+.1f%%"), "最大跌幅pct": NC("期間最大跌幅", format="%+.1f%%")})
            st.download_button("下載這份明細（CSV）", show[cols].to_csv(index=False).encode("utf-8-sig"), "nh20_history_filtered.csv", "text/csv")

# ---------------- 3. 完整觀察名單 ----------------
with tab3:
    st.caption("昨天收盤後算好的全部候選。今天收盤只要「高於觸發價」且「成交量夠」，就是買進訊號。")
    w = f(rd("nh20_watch.csv"))
    if not w.empty:
        w = w.sort_values("需漲pct")
    table(w, ["代號", "名稱", "產業", "昨收", "觸發價", "需漲pct", "均量張", "需量張1320", "需量張1324", "上方套牢區"], {
        "上方套牢區": TC("觸發價上方 10% 內的套牢區（佔量）"),
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
7. **主卡不買「撞上套牢區」的突破**（2026-10-14 起）：收盤價上方 5% 內有一大塊過去 120 天的成交密集區（佔量 ≥ 10%），或突破當天爆量（≥ 2 倍）卻收長上影（上影 ≥ 當天高低差一半）。10 檔卡只標示提醒。

**第 7 條規則的回測比較（主卡）**

| 年度 | 組合報酬：不採用 | 組合報酬：採用 | 採用較好的模擬比例 | 被排除的訊號：筆數／勝率／平均 | 保留的訊號：勝率／平均 |
|---|---|---|---|---|---|
| 2024 | +100% | **+121%** | 73% | 33 筆／21%／-2.2% | 31%／+13.9% |
| 2025 | +104% | **+107%** | 100% | 30 筆／33%／+7.1% | 45%／+23.6% |
| 2026 至今 | +41% | **+81%** | 100% | 34 筆／18%／-0.5% | 39%／+6.1% |

組合報酬＝每年 1 月 1 日空手開始、最多 6 檔、30 次隨機挑股的中位數；訊號統計＝每個訊號都買、各自計算。

**賣出**
- 進場後，找出「量最大的那根 K 棒」，它的最低價就是防守價；之後如果出現更大量的 K 棒，防守價就換成那根的低點。
- **收盤跌破防守價 → 隔天開盤賣**。沒跌破就一直抱，沒有時間限制。

**為什麼這樣設計**：突破套牢區，代表之前買的人解套、賣壓變小；突破後追進最多的人的成本（最大量 K 棒低點）被跌破，代表他們被套住、賣壓出現。

⚠️ 這是個人研究用的監看頁面，回測結果不代表未來，也還沒有實單驗證。
""")
