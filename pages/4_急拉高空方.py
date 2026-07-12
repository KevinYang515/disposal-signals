"""
急拉高空方策略 — Q8/Q10 最終版 (2026-07-12定案)
核心邏輯：09:00-09:15爆量急拉0~7% → 流動性前30%+市值最小8%/10%動態分位篩選
         → 09:30進場放空 → 前高+1tick緊停損 → 13:25強制平倉
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
from pathlib import Path

st.set_page_config(page_title="急拉高空方 Q8/Q10", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")

DATA_DIR   = Path(__file__).parent.parent / "data"
Q8_PARQUET  = DATA_DIR / "急拉高_Q8版.parquet"
Q10_PARQUET = DATA_DIR / "急拉高_Q10版.parquet"
TODAY_JSON  = DATA_DIR / "急拉高_今日選股.json"
NAMES_CSV   = DATA_DIR / "stock_names.csv"

HEADLINE = {
    "Q8": dict(n=1665, sharpe=3.10, cum=2300.5, mdd=-21.7, avg2026=0.245,
               desc="流動性前30%(70分位) + 市值最小8%動態分位 + 前日OTC漲幅>=3%跳過",
               note="3.5年全期表現最佳，但2026單獨檢驗較Q10脆弱(排除最強5天即轉負)"),
    "Q10": dict(n=2084, sharpe=2.99, cum=1981.0, mdd=-20.0, avg2026=0.327,
                desc="流動性前30%(70分位) + 市值最小10%動態分位 + 前日OTC漲幅>=3%跳過",
                note="2026單獨檢驗較穩健(排除最強10天才打平)，是較保守的替代選項"),
}

with st.sidebar:
    st.title("版本切換")
    variant = st.radio("策略版本", ["Q8", "Q10"], index=0,
                        help="Q8=長期最優，Q10=近期較穩健。目前兩版同時在模擬帳戶跑，尚未二選一。")
    st.divider()
    st.caption(
        "**2026-07-12 定案**：從758檔(選股偏誤)擴大到995檔universe，"
        "做流動性x市值聯合網格搜索找到比先前K/N版更優的組合。\n\n"
        "09:00-09:15訊號窗口急拉0~7%+量能>=20日均量2倍 → "
        "09:30直接進場 → 停損=09:00-09:30累積高點+1tick → 13:25強平"
    )

@st.cache_data(ttl=300)
def load_trades(path):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["ret"] = df["net_realistic"]
    df["win"] = df["ret"] > 0
    return df

@st.cache_data(ttl=300)
def load_today():
    if not TODAY_JSON.exists():
        return None
    with open(TODAY_JSON, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_names():
    if NAMES_CSV.exists():
        m = pd.read_csv(NAMES_CSV, dtype=str)
        m["ticker"] = m["ticker"].str.strip()
        return dict(zip(m["ticker"], m["name"]))
    return {}

@st.cache_data(ttl=300)
def load_live_trades(variant):
    p = DATA_DIR / f"q_surge_trades_{variant}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

names = load_names()
h = HEADLINE[variant]
parquet_path = Q8_PARQUET if variant == "Q8" else Q10_PARQUET

st.title(f"急拉高空方 — {variant}版（2026-07-12定案）")
st.caption("爆量急拉後推不動 → 流動性/市值/大盤regime三層濾網 → 09:30進場做空 → 13:25強制平倉")

st.success(
    f"✅ **{variant}版**：{h['desc']}\n\n"
    f"3.5年(2023-01~2026-07) 全期 Sharpe **{h['sharpe']}**、累積報酬 **+{h['cum']:.1f}%**、"
    f"MaxDD **{h['mdd']:.1f}%**、2026單獨均報酬 **+{h['avg2026']:.3f}%/筆**\n\n"
    f"ℹ️ {h['note']}"
)

if not parquet_path.exists():
    st.error(f"找不到 {parquet_path}")
    st.stop()

sub = load_trades(parquet_path)

# ── 頂部指標 ─────────────────────────────────────────────
n = len(sub)
wr = sub["win"].mean()
avg_r = sub["ret"].mean()
gw = sub.loc[sub["win"], "ret"].sum()
gl = abs(sub.loc[~sub["win"], "ret"].sum())
pf = gw / gl if gl > 0 else float("inf")
n_days = sub["date"].nunique()
per_day = n / max(sub["date"].dt.year.nunique() * 250, 1)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("交易次數", f"{n:,}")
col2.metric("訊號天數", f"{n_days:,}")
col3.metric("勝率", f"{wr:.1%}")
col4.metric("Profit Factor", f"{pf:.2f}")
col5.metric("均報酬/筆", f"{avg_r:.2%}")
col6.metric("每日候選(約)", f"{n/n_days:.2f}" if n_days else "—")

st.divider()

# ── 今日選股 ─────────────────────────────────────────────
st.subheader("📌 今日候選池")
today = load_today()
if today is None:
    st.info("尚無今日選股資料")
else:
    cand = today.get("candidates", {})
    cap_max = 0.08 if variant == "Q8" else 0.10
    pool = {k: v for k, v in cand.items()
            if v.get("cap_pct_rank") is not None and v["cap_pct_rank"] < cap_max}
    c1, c2, c3 = st.columns(3)
    c1.metric("計算日期(資料截至)", today.get("計算日期", "—"))
    c2.metric(f"{variant}候選池大小", f"{len(pool)} 檔")
    otc_skip = today.get("otc_skip_today", False)
    c3.metric("前日OTC濾網", "🔴 今日跳過" if otc_skip else "🟢 正常",
              delta=f"前日OTC {today.get('prev_otc_ret', 0)*100:+.2f}%", delta_color="off")
    if otc_skip:
        st.warning("前一日OTC(櫃買指數)漲幅>=3%，今日Q8/Q10皆不進場（regime濾網觸發）")
    if pool:
        pool_df = pd.DataFrame(pool).T.reset_index().rename(columns={"index": "ticker"})
        pool_df["名稱"] = pool_df["ticker"].map(lambda t: names.get(str(t), ""))
        pool_df = pool_df.sort_values("cap_pct_rank")[["ticker", "名稱", "cap_yi", "cap_pct_rank"]]
        pool_df.columns = ["代號", "名稱", "市值(億)", "市值百分位"]
        pool_df["市值百分位"] = pool_df["市值百分位"].map("{:.1%}".format)
        st.caption(f"符合流動性+市值資格的觀察名單（實際是否進場，仍要看今日09:00-09:15是否急拉0~7%+爆量）")
        st.dataframe(pool_df, use_container_width=True, hide_index=True)
    else:
        st.caption("今日無股票通過流動性+市值資格門檻")

st.divider()

# ── 模擬單交易狀況 ────────────────────────────────────────
st.subheader("💰 模擬帳戶交易狀況")
live = load_live_trades(variant)
if live.empty:
    st.info(
        f"{variant}版模擬單尚無成交記錄。VM已排入cron，開盤日 08:55(TWN)自動啟動、"
        "09:30進場、13:25平倉，完成後這裡會顯示當日進出場明細。"
    )
else:
    live["date"] = pd.to_datetime(live["date"])
    total_live_pnl = live["ret_pct"].sum() if "ret_pct" in live else np.nan
    live_wr = (live["ret_pct"] > 0).mean() if "ret_pct" in live else np.nan
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("累計模擬交易", f"{len(live)} 筆")
    lc2.metric("累計報酬率(單純加總)", f"{total_live_pnl:.2%}" if pd.notna(total_live_pnl) else "—")
    lc3.metric("模擬勝率", f"{live_wr:.1%}" if pd.notna(live_wr) else "—")
    show_live = live.sort_values("date", ascending=False).copy()
    show_live["名稱"] = show_live["ticker"].map(lambda t: names.get(str(t), ""))
    st.dataframe(show_live, use_container_width=True, hide_index=True)

st.divider()

# ── 年度分解 ──────────────────────────────────────────────
st.subheader("年度穩健性")
yr = (sub.groupby("year")
        .agg(次數=("ret", "count"), 勝率=("win", "mean"), 均報酬=("ret", "mean"))
        .reset_index())

bar_wr = alt.Chart(yr).mark_bar(color="#4a90d9").encode(
    x=alt.X("year:O", title="年度"),
    y=alt.Y("勝率:Q", title="勝率", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
    tooltip=[alt.Tooltip("year:O", title="年"), alt.Tooltip("勝率:Q", format=".1%"),
             alt.Tooltip("次數:Q"), alt.Tooltip("均報酬:Q", format=".2%")]
).properties(height=200, title="年度勝率")

bar_ret = alt.Chart(yr).mark_bar(color="#27ae60").encode(
    x=alt.X("year:O", title="年度"),
    y=alt.Y("均報酬:Q", title="均報酬", axis=alt.Axis(format=".1%")),
    tooltip=[alt.Tooltip("year:O", title="年"), alt.Tooltip("均報酬:Q", format=".2%"),
             alt.Tooltip("次數:Q")]
).properties(height=200, title="年度均報酬")

c1, c2 = st.columns(2)
c1.altair_chart(bar_wr, use_container_width=True)
c2.altair_chart(bar_ret, use_container_width=True)

# ── 累積報酬曲線 (逐筆複利近似, 非投組加權) ──────────────────
st.subheader("累積報酬曲線（逐筆複利近似）")
sub_sorted = sub.sort_values("date").reset_index(drop=True)
sub_sorted["cum_ret"] = (1 + sub_sorted["ret"]).cumprod() - 1
sub_sorted["trade_no"] = sub_sorted.index + 1

line = alt.Chart(sub_sorted).mark_line(color="#e74c3c").encode(
    x=alt.X("trade_no:Q", title="累計交易筆數"),
    y=alt.Y("cum_ret:Q", title="累計報酬", axis=alt.Axis(format=".0%")),
    tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("ticker:N", title="股票"),
             alt.Tooltip("cum_ret:Q", format=".1%", title="累計報酬")]
).properties(height=300)
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
st.altair_chart(line + zero, use_container_width=True)
st.caption("此曲線為逐筆複利近似（非每日N檔投組加權），headline的Sharpe/累積報酬/MaxDD是用投組層級算法，數字以上方指標卡為準。")

# ── 2026 逐月/逐日損益 ────────────────────────────────────
st.subheader("2026 逐月／逐日損益")
sub_2026 = sub[sub["year"] == 2026].copy()
if sub_2026.empty:
    st.info("2026年目前無交易紀錄")
else:
    sub_2026["month"] = sub_2026["date"].dt.strftime("%Y-%m")
    monthly = (sub_2026.groupby("month")
               .agg(次數=("ret", "count"), 勝率=("win", "mean"),
                    月報酬合計=("ret", "sum"), 均報酬=("ret", "mean"))
               .reset_index())

    bar_month = alt.Chart(monthly).mark_bar().encode(
        x=alt.X("month:O", title="月份"),
        y=alt.Y("月報酬合計:Q", title="當月報酬合計(逐筆加總)", axis=alt.Axis(format=".0%")),
        color=alt.condition(alt.datum.月報酬合計 > 0, alt.value("#27ae60"), alt.value("#e74c3c")),
        tooltip=[alt.Tooltip("month:O", title="月份"), alt.Tooltip("次數:Q", title="交易次數"),
                 alt.Tooltip("勝率:Q", format=".1%"), alt.Tooltip("月報酬合計:Q", format=".2%")]
    ).properties(height=250, title="2026 逐月報酬合計（逐筆簡單加總，非投組加權）")
    st.altair_chart(bar_month, use_container_width=True)

    show_m = monthly.copy()
    show_m["勝率"] = show_m["勝率"].map("{:.1%}".format)
    show_m["月報酬合計"] = show_m["月報酬合計"].map("{:.2%}".format)
    show_m["均報酬"] = show_m["均報酬"].map("{:.2%}".format)
    show_m.columns = ["月份", "交易次數", "勝率", "月報酬合計", "均報酬/筆"]
    st.dataframe(show_m, use_container_width=True, hide_index=True)

    with st.expander("展開查看2026逐日損益"):
        daily = (sub_2026.groupby(sub_2026["date"].dt.strftime("%Y-%m-%d"))
                 .agg(次數=("ret", "count"), 當日報酬合計=("ret", "sum"))
                 .reset_index().rename(columns={"date": "日期"}))
        daily["當日報酬合計"] = daily["當日報酬合計"].map("{:.2%}".format)
        st.dataframe(daily, use_container_width=True, hide_index=True, height=300)
    st.caption("報酬合計為當日/當月所有交易筆數的單純加總（非資金加權），僅供觀察走勢用，非真實資金曲線。")

# ── 近期交易明細 ─────────────────────────────────────────
st.subheader("近期交易明細（回測）")
recent = sub.sort_values("date", ascending=False).head(50).copy()
recent["名稱"] = recent["ticker"].map(lambda t: names.get(str(t), ""))
recent = recent[["date", "ticker", "名稱", "surge_pct", "entry", "ret", "win", "stopped", "locked", "cap_pct_rank_yearly"]]
recent.columns = ["日期", "股票", "名稱", "急拉幅度", "進場價", "報酬", "勝", "停損", "鎖漲停", "市值百分位"]
recent["日期"] = recent["日期"].dt.strftime("%Y-%m-%d")
recent["急拉幅度"] = recent["急拉幅度"].map("{:.1%}".format)
recent["進場價"] = recent["進場價"].map("{:.2f}".format)
recent["報酬"] = recent["報酬"].map("{:.2%}".format)
recent["市值百分位"] = recent["市值百分位"].map("{:.1%}".format)

def color_ret(val):
    try:
        v = float(val.replace("%", ""))
        return "color:#27ae60;font-weight:700" if v > 0 else "color:#e74c3c"
    except Exception:
        return ""

st.dataframe(recent.style.map(color_ret, subset=["報酬"]), use_container_width=True, height=400)

# ── 策略說明 ─────────────────────────────────────────────
with st.expander("策略說明 / 方法論"):
    st.markdown(f"""
**訊號條件**
- 09:00-09:15訊號窗口出現急拉：窗口最高價相對前收介於 0%~7%
- 該窗口成交量 >= 20日同窗口均量 x 2.0倍

**濾網（{variant}版）**
- 流動性：個股20日均成交金額 / 全市場20日均成交金額，取前30%(70分位)，固定門檻(從3.5年歷史算出)
- 市值：個股市值(收盤價x發行股數)在「當年度、已通過流動性篩選的候選股」中的百分位排名，
  取最小{'8%' if variant=='Q8' else '10%'}（**動態相對分位，非固定金額**，避免大盤/市值隨時間成長造成的漂移偏誤）
- 大盤regime：前一日櫃買指數(OTC)漲幅 >= 3% 時，當天整天跳過不進場

**進場**：09:30後直接以市價附近成交（非等回落觸發）

**停損**：09:00-09:30累積最高點 + 1 tick（貼齊高點的緊停損）

**出場**：13:25強制平倉（現股當沖回補期限），若鎖漲停無法回補則按-4.18%估算強迫成本

**⚠️ 為何有停損還是會被鎖漲停**：停損價=進場前累積高點+1 tick，是股價的名目價位。若進場前
股價已經衝到很接近10%漲停（例如09:30前已來到9.8%），則「高點+1 tick」換算後可能**超過
漲停價本身**——但股價依法規不可能交易超過漲停，這個停損就永遠碰不到，等於完全失去防護，
股價一路鎖到漲停收盤。這類交易(`locked=True`且`stopped=False`)才會用-4.18%估算強迫成本，
若已經正常觸發停損出場則不會套用此估算。

**為何用OTC而非加權指數(TAIEX)判斷大盤regime**：候選股以中小型股為主，跟櫃買指數連動性
明顯優於加權指數（同樣邏輯下相關係數強7倍），加權指數常被少數權值股(如台積電)牽動，
不能反映中小型股真實的強弱狀態。

**{variant}版 vs 另一版本的取捨**：Q8長期(3.5年)表現最優，但2026年單獨檢驗顯示較脆弱；
Q10相對保守、近期穩健度較好。兩版本目前都同時在Shioaji模擬帳戶跑，尚未正式二選一，
用實盤資料持續比較中。

**成本估算（永豐2折）**：手續費0.057% + 交易稅0.15% ≒ 0.207%/筆（已內含於net_realistic）

**⚠️ 全部為回測+模擬單結果，尚未動用真金。**
""")
