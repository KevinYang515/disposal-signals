# -*- coding: utf-8 -*-
r"""大戶波段跟單監控：已驗證的少數幾個分點/共振訊號，追蹤牠們最近在買什麼。

狀態：研究／個人監看，只收錄有「固定訊號日＋固定持有期＋逐事件扣大盤基準」嚴謹回測支撐的兩條規則。
資料與結論來源：E:\stock\broker_flow_bigmoney_branches_20260908\PLAYBOOK.md（§6h起）。
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="大戶波段跟單監控", page_icon="🐋", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TODAY_FILE = os.path.join(DATA_DIR, "bigmoney_today_watchlist.csv")
OBS_FILE = os.path.join(DATA_DIR, "bigmoney_observation_watchlist.csv")
SOLO_FILE = os.path.join(DATA_DIR, "bigmoney_solo_zhanqian_events.csv")
CONV_FILE = os.path.join(DATA_DIR, "bigmoney_convergence_events.csv")
META_FILE = os.path.join(DATA_DIR, "bigmoney_watchlist_meta.json")


def sharpe_pf(s):
    """(夏普值, 賺賠比)，跟站內 app.py 同一個定義：Sharpe=mean/std（不年化），PF=均獲利/|均虧損|。"""
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan, np.nan
    sharpe = round(s.mean() / s.std(), 2) if s.std() > 0 else np.nan
    wins, loss = s[s > 0], s[s <= 0]
    pf = round(wins.mean() / abs(loss.mean()), 2) if len(loss) > 0 and loss.mean() != 0 else np.nan
    return sharpe, pf


@st.cache_data(ttl=3600)
def load_meta():
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_today():
    if not os.path.exists(TODAY_FILE):
        return pd.DataFrame()
    df = pd.read_csv(TODAY_FILE, dtype={"stock_id": str})
    df.columns = ["股票代號", "公司簡稱", "觸發路徑", "說明"]
    df["公司簡稱"] = df["公司簡稱"].fillna("")
    return df


@st.cache_data(ttl=3600)
def load_observation():
    if not os.path.exists(OBS_FILE):
        return pd.DataFrame()
    df = pd.read_csv(OBS_FILE, dtype={"股票代號": str})
    df["公司簡稱"] = df["公司簡稱"].fillna("")
    return df


@st.cache_data(ttl=3600)
def load_track_record():
    solo = pd.read_csv(SOLO_FILE, dtype={"stock_id": str}) if os.path.exists(SOLO_FILE) else pd.DataFrame()
    conv = pd.read_csv(CONV_FILE, dtype={"stock_id": str}) if os.path.exists(CONV_FILE) else pd.DataFrame()
    if not solo.empty:
        solo = solo.rename(columns={"company_name": "公司簡稱"})
        solo["訊號類型"] = "路徑一：凱基-站前 單獨衝到近2年新建倉20億+"
        solo["年份"] = pd.to_datetime(solo["event_date"]).dt.year.astype(str)
    if not conv.empty:
        conv_cols = conv.columns.tolist()
        conv = conv.rename(columns={conv_cols[2]: "公司簡稱", conv_cols[3]: "參與分點清單", conv_cols[4]: "參與分點數"})
        conv3 = conv[conv["參與分點數"] >= 3].copy()
        conv3["訊號類型"] = "路徑二：≥3個已驗證分點10個交易日內同時買超"
        conv3["年份"] = pd.to_datetime(conv3["event_date"]).dt.year.astype(str)
        conv = conv3
    return solo, conv


meta = load_meta()

st.title("🐋 大戶波段跟單監控")
st.warning(
    "⚠️ **狀態：研究／個人監看，非正式交易建議。** 這個研究線的核心教訓是：市面上看起來「重壓=準」的分點訊號，"
    "多數用寬鬆方法測會虛高——這一頁只保留兩條用「固定訊號日＋固定10/20/60日持有期＋逐事件扣同期大盤等權重指數報酬」"
    "嚴謹回測過的規則，連曾經被當作標竿引用的凱基-信義、國泰-板橋單獨訊號，重測後其實跟丟銅板一樣（60日勝率約50%），"
    "已經從名單移除。**這類訊號需要抱60個交易日（約3個月）才會顯現效果，經確認不是隔日沖分點的短線出貨行為，"
    "不適合當短線進出依據。**"
)
if meta.get("data_asof"):
    st.caption(
        f"📅 分點交易資料涵蓋至 **{meta['data_asof']}**（券商分點揭露資料常落後1-2個交易日，不是今天）；"
        f"本頁最後產生時間 {meta.get('generated_at', '-')}。"
    )

st.divider()
st.header("📋 目前觀察名單：這些大戶最近在買什麼")

today = load_today()
if today.empty:
    st.info("目前沒有任何股票觸發任一條規則。")
else:
    path_counts = today["觸發路徑"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("觸發股票數", f"{today['股票代號'].nunique()} 檔")
    c2.metric("路徑一觸發數", f"{path_counts.get('路徑一:單分點高門檻', 0)} 檔")
    c3.metric("路徑二觸發數", f"{path_counts.get('路徑二:3+分點共振', 0)} 檔")

    sel_path = st.multiselect(
        "篩選觸發路徑", today["觸發路徑"].unique().tolist(),
        default=today["觸發路徑"].unique().tolist(),
    )
    view_today = today[today["觸發路徑"].isin(sel_path)].sort_values(["觸發路徑", "股票代號"])
    st.dataframe(view_today, use_container_width=True, height=420, hide_index=True)
    st.download_button(
        "📥 下載目前觀察名單 CSV", view_today.to_csv(index=False, encoding="utf-8-sig"),
        "bigmoney_today_watchlist.csv", "text/csv", key="dl_today",
    )

st.caption(
    "路徑一＝凱基-站前近2年新建倉金額衝到自己驗證過的門檻（20億）；"
    "路徑二＝凱基-信義／站前／士林、國泰-板橋、港商野村、新加坡商瑞銀這6個名單裡，"
    "有≥3個在近10個交易日內同時買超各自的驗證門檻。任一路徑觸發即列入，不代表兩者強度相同——"
    "下面的歷史回測分別列出兩條路徑各自的實際表現，請對照著看。"
)

st.divider()
with st.expander("👀 觀察名單（弱訊號／未驗證，僅供參考）：這些次要大戶最近在買什麼", expanded=False):
    st.caption(
        "**國票-安和／港商野村／群益金鼎／凱基-竹科**：用同一套嚴謹方法測過，有真實但很溫和的優勢"
        "（近2年新建倉衝到高門檻時，60日扣大盤勝率約 50-60%、中位數約 +0~+3%，遠不如凱基-站前的 67%／+9%），"
        "還不到能單獨當進場訊號的程度。群益金鼎為裸名稱（總公司彙總帳戶），訊號較雜；"
        "凱基-竹科規模只有凱基-站前的約1/10，門檻要拉到 3 億以上才轉正。\n\n"
        "**台新-五權西**：法人級大分點（近5個月總買進約 312 億、單一股票押到 46-57 億），"
        "但 2026-04 台新併元富造成分點代號斷點、可用歷史只有 5 個月，無法做嚴謹回測。"
        "操作型態像有紀律的「賺大賠小」——大立光（留倉 92%）、創意（留倉 86%）抱得很緊，"
        "其他多是波段進出甚至獲利了結（聯發科已從 2,100 張減到 600 張）。**列此純粹是從 2026-09 起開始"
        "累積它的樣本外紀錄，還不是驗證過的訊號。**\n\n"
        "以下都只是「近20個交易日淨買超金額前15名」，不套用任何門檻，僅列非 ETF 個股。"
    )
    obs = load_observation()
    if obs.empty:
        st.info("目前沒有觀察名單資料。")
    else:
        sel_obs = st.multiselect("篩選分點", obs["分點"].unique().tolist(),
                                 default=obs["分點"].unique().tolist(), key="obs_branch")
        st.dataframe(obs[obs["分點"].isin(sel_obs)], use_container_width=True, height=420, hide_index=True)

st.divider()
st.header("📊 兩條規則的歷史回測實績")
st.caption("以下用同一套方法：訊號成立當天為基準，固定60個交易日後的個股報酬，扣掉同期全市場等權重指數報酬（超額報酬）。")

solo_events, conv_events = load_track_record()
tab1, tab2 = st.tabs(["路徑一：凱基-站前 單獨訊號", "路徑二：≥3分點共振"])

for tab, events, label in [(tab1, solo_events, "路徑一"), (tab2, conv_events, "路徑二")]:
    with tab:
        if events.empty:
            st.info("目前沒有歷史事件資料。")
            continue
        years = sorted(events["年份"].unique().tolist(), reverse=True)
        sel_years = st.multiselect("年份", years, default=years, key=f"years_{label}")
        view = events[events["年份"].isin(sel_years)]
        settled = view.dropna(subset=["excess_60d"])
        if settled.empty:
            st.warning("目前篩選條件下沒有已結算（滿60個交易日）的事件。")
        else:
            rets = settled["excess_60d"] * 100.0
            wins, loss = rets[rets > 0], rets[rets <= 0]
            sharpe, pf = sharpe_pf(rets)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("總筆數", f"{len(view)} 筆", delta=f"{len(settled)} 已結算(滿60日)", delta_color="off")
            c2.metric("勝率(扣大盤)", f"{(rets > 0).mean() * 100:.2f}%")
            c3.metric("期望超額報酬", f"{rets.mean():+.2f}%")
            c4.metric("夏普值", f"{sharpe:.2f}" if pd.notna(sharpe) else "-")
            c5.metric("賺賠比", f"{pf:.2f}" if pd.notna(pf) else "-")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("獲利筆", len(wins))
            d2.metric("虧損筆", len(loss))
            d3.metric("均獲利", f"{wins.mean():+.2f}%" if len(wins) else "-")
            d4.metric("均虧損", f"{loss.mean():+.2f}%" if len(loss) else "-")
            st.caption("超額報酬中位數：" + f"{rets.median():+.2f}%")

        st.markdown("##### 逐筆歷史事件")
        show_cols = ["event_date", "stock_id", "公司簡稱"]
        if "broker" in view.columns:
            show_cols.append("broker")
        if "參與分點清單" in view.columns:
            show_cols += ["參與分點清單", "參與分點數"]
        show_cols += ["ret_60d", "excess_60d"]
        show_cols = [c for c in show_cols if c in view.columns]
        show = view[show_cols].sort_values("event_date", ascending=False).copy()
        for pct_col in ["ret_60d", "excess_60d"]:
            if pct_col in show.columns:
                show[pct_col] = (show[pct_col] * 100).round(2)
        rename_map = {
            "event_date": "訊號日", "stock_id": "代號", "broker": "分點",
            "ret_60d": "60日原始報酬%", "excess_60d": "60日超額報酬%",
        }
        show = show.rename(columns=rename_map)
        st.dataframe(show, use_container_width=True, height=360, hide_index=True)
        st.download_button(
            f"📥 下載{label}歷史事件 CSV", show.to_csv(index=False, encoding="utf-8-sig"),
            f"bigmoney_{label}_events.csv", "text/csv", key=f"dl_{label}",
        )

st.divider()
st.markdown(
    "**方法論摘要**（完整過程與踩過的坑見 `E:\\stock\\broker_flow_bigmoney_branches_20260908\\PLAYBOOK.md`）：\n"
    "- 曾經測過凱基-三多、凱基-興隆、元大-松江、台新-五權西、台新-西松、國泰-館前、7家外資法人、"
    "以及第二輪全市場篩選出的147個廣泛型候選、6個新指名分點（兆豐-民生/永豐金-新店/港商野村/康和/群益金鼎/富邦-南屯）——"
    "**除了本頁列出的兩條規則，其餘全部沒有通過嚴謹回測，不建議跟單。**\n"
    "- 已個別確認凱基-站前、凱基-士林、國泰-板橋不是隔日沖分點（買進後3日內出貨比例僅11-21%，遠低於真正隔日沖分點的80%以上），"
    "所以用「持有」而非「當沖」的邏輯詮釋牠們的訊號是合理的。\n"
    "- 已測試縮短持有期（3日/5日），結果一致變差不是變好——這類訊號本來就需要抱到60個交易日左右才會顯現效果。"
)
