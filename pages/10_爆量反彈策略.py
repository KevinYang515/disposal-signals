"""
爆量反彈 / 高檔爆量出貨 策略研究
延伸自「漲跌停事件研究」，測試「連續跌停+爆量」能不能拿來搶反彈，
以及是否一定要跌停、反向高檔爆量是不是出貨訊號、市值/流動性/產業穩不穩健。
資料由 D:\\stock\\stock\\limitup_limitdown_study\\burst_rebound_study.py 產生 (2015-01-01起)。
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
from pathlib import Path

st.set_page_config(page_title="爆量反彈策略", page_icon="💥", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "data" / "burst_study"

UP_COLOR = "#f87171"    # 紅 = 正報酬（台股慣例：紅漲綠跌）
DOWN_COLOR = "#4ade80"  # 綠 = 負報酬
NEUTRAL_COLOR = "#60a5fa"

HORIZON_ORDER = [1, 3, 5, 10, 20]
HORIZON_LABELS = {1: "+1日", 3: "+3日", 5: "+5日", 10: "+10日", 20: "+20日"}


ML_DATA_DIR = Path(__file__).parent.parent / "data" / "burst_ml_ga"


@st.cache_data(ttl=3600)
def load_all():
    d = {}
    for name in ["core_signals", "core_horizon_stats", "control_horizon_stats", "core_year_stats",
                 "core_by_breadth", "core_episodes", "core_by_cap", "core_by_industry", "core_by_liquidity",
                 "general_pullback_grid", "general_pullback_by_breadth",
                 "reverse_top_grid", "reverse_top_by_breadth", "strategy_grid"]:
        d[name] = pd.read_csv(DATA_DIR / f"burst_{name}.csv")
    d["core_signals"]["date"] = pd.to_datetime(d["core_signals"]["date"])
    d["core_signals"]["code"] = d["core_signals"]["code"].astype(str)
    d["core_episodes"]["start"] = pd.to_datetime(d["core_episodes"]["start"])
    d["core_episodes"]["end"] = pd.to_datetime(d["core_episodes"]["end"])
    return d


@st.cache_data(ttl=3600)
def load_ml():
    d = {}
    for name in ["final_signals", "final_horizon_stats", "shap_importance", "univariate_quantiles"]:
        d[name] = pd.read_csv(ML_DATA_DIR / f"ga_{name}.csv")
    d["validate_yearly"] = pd.read_csv(ML_DATA_DIR / "validate_yearly.csv")
    d["validate_liquidity"] = pd.read_csv(ML_DATA_DIR / "validate_liquidity.csv")
    d["margin_level_quantiles"] = pd.read_csv(ML_DATA_DIR / "margin_level_quantiles.csv")
    d["margin_level_vs_change_summary"] = pd.read_csv(ML_DATA_DIR / "margin_level_vs_change_summary.csv")
    with open(ML_DATA_DIR / "ga_best_rule.json", encoding="utf-8") as f:
        d["rule"] = json.load(f)
    d["final_signals"]["date"] = pd.to_datetime(d["final_signals"]["date"])
    d["final_signals"]["code"] = d["final_signals"]["code"].astype(str)
    return d


try:
    data = load_all()
except FileNotFoundError:
    st.error("找不到資料，請確認 data/burst_study/ 目錄下的 burst_*.csv 存在")
    st.stop()

try:
    ml_data = load_ml()
    ML_AVAILABLE = True
except FileNotFoundError:
    ML_AVAILABLE = False

core_signals = data["core_signals"]
core_horizon = data["core_horizon_stats"]
control_horizon = data["control_horizon_stats"]
year_stats = data["core_year_stats"]
by_breadth = data["core_by_breadth"]
episodes = data["core_episodes"]
by_cap = data["core_by_cap"]
by_industry = data["core_by_industry"]
by_liquidity = data["core_by_liquidity"]
general_grid = data["general_pullback_grid"]
general_breadth = data["general_pullback_by_breadth"]
reverse_grid = data["reverse_top_grid"]
reverse_breadth = data["reverse_top_by_breadth"]

st.title("💥 爆量反彈策略：連續跌停後爆量，能不能搶反彈？")
st.caption(
    f"樣本 2015-01-01 起・訊號＝個股連續跌停第2-3天 ＋ 當日成交量≥20日均量2倍(爆量)・"
    f"共 {len(core_signals):,} 筆訊號・買在訊號日收盤，賣在N個交易日後收盤・裸報酬未扣費"
)

st.warning(
    "⚠️ **這不是一個可以無腦自動下單的日常策略**。往下看會發現：訊號的報酬幾乎全部來自"
    "「當天全市場同時有 30 檔以上跌停」的系統性恐慌事件（11年只發生約20次），"
    "單一個股孤立跌停+爆量，統計上**沒有正期望值、甚至是負的**。"
    "本頁為描述性事件研究，非投資建議，跌停鎖死期間也未必能真的買到。"
)

tab_overview, tab_breadth, tab_q1, tab_q2, tab_q3, tab_ml, tab_history = st.tabs(
    ["📊 核心數據", "🌪️ 關鍵發現：市場寬度", "❓一定要跌停嗎", "🔄 反向：高檔爆量出貨？",
     "🏭 市值/流動性/產業", "🤖 ML+GA挖掘版", "📜 歷史事件"]
)

# ============================================================
# Tab 1: 核心數據
# ============================================================
with tab_overview:
    st.markdown("### 策略定義")
    st.markdown(
        "- **進場**：個股當日收盤跌停（跌幅≥9.5%），且是**連續跌停的第2或第3天**（streak 2-3），"
        "且當日成交量 ≥ 20日均量的 **2倍**（爆量）→ 收盤買進\n"
        "- **出場**：固定持有 N 個交易日後收盤賣出（本頁測 1/3/5/10/20 日）\n"
        "- **為什麼是第2-3天而不是首日**：上一輪研究([[漲跌停事件研究]])已驗證跌停首日爆量"
        "反而是壞消息(恐慌出貨潮延續)；連跌4天以上樣本太小、且很可能是走向下市的個股(倖存者偏誤最重)，"
        "第2-3天是「賣壓已經釋放一部分、但還沒惡化到基本面問題」的甜蜜點"
    )

    st.markdown("### 各持有期間報酬 / 勝率 / Sharpe / 賺賠比")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**訊號組（連續跌停2-3天+爆量）**")
        disp1 = core_horizon.copy()
        disp1["期間"] = disp1["horizon"].map(HORIZON_LABELS)
        disp1 = disp1[["期間", "n", "mean", "win", "sharpe", "pf"]].rename(
            columns={"n": "樣本數", "mean": "平均報酬%", "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
        st.dataframe(
            disp1.style.format({"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}",
                                 "樣本數": "{:,}"}),
            use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**對照組（連續跌停2-3天但未爆量）**")
        disp2 = control_horizon.copy()
        disp2["期間"] = disp2["horizon"].map(HORIZON_LABELS)
        disp2 = disp2[["期間", "n", "mean", "win", "sharpe", "pf"]].rename(
            columns={"n": "樣本數", "mean": "平均報酬%", "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
        st.dataframe(
            disp2.style.format({"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}",
                                 "樣本數": "{:,}"}),
            use_container_width=True, hide_index=True)

    st.caption(
        "**注意 +1日仍是負的**（訊號組 -0.52%，勝率45%）：爆量那天的反彈不是隔天秒噴，"
        "至少要抱到 +5日 以後才會明顯轉正，操作上要有等幾天的心理準備。"
    )

    chart_df = core_horizon.copy()
    chart_df["期間"] = chart_df["horizon"].map(HORIZON_LABELS)
    chart = alt.Chart(chart_df).mark_bar(color=NEUTRAL_COLOR, size=30).encode(
        x=alt.X("期間:N", sort=[HORIZON_LABELS[h] for h in HORIZON_ORDER], title=None),
        y=alt.Y("mean:Q", title="平均報酬 (%)"),
        tooltip=[alt.Tooltip("期間:N"), alt.Tooltip("mean:Q", format="+.2f"), alt.Tooltip("n:Q", format=",")],
    ).properties(height=260, title="訊號組平均報酬隨持有天數演變")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#94a3b8").encode(y="y:Q")
    st.altair_chart(chart + zero, use_container_width=True)

    st.markdown("### 閾值敏感度（爆量倍數 × 連續天數規則）")
    grid = data["strategy_grid"]
    grid_disp = grid[grid["horizon"] == 20].copy()
    grid_disp = grid_disp[["threshold", "streak_rule", "n", "mean", "win", "sharpe", "pf"]].rename(
        columns={"threshold": "爆量倍數", "streak_rule": "連續天數規則", "n": "樣本數",
                 "mean": "+20日平均報酬%", "win": "+20日勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
    st.dataframe(
        grid_disp.style.format({"+20日平均報酬%": "{:+.2f}%", "+20日勝率%": "{:.2f}%",
                                 "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:,}"}),
        use_container_width=True, hide_index=True)
    st.caption("streak3+（連跌3天以上才進場）Sharpe 普遍比 streak2-3 更高，但樣本數也少一截，兩者互為取捨。")

# ============================================================
# Tab 2: 市場寬度 -- 關鍵發現
# ============================================================
with tab_breadth:
    st.markdown("### 🌪️ 全站最重要的一張表")
    st.markdown(
        "把訊號依「**當天全市場同時有幾檔股票跌停**」分組，結果天差地遠：\n\n"
        "- 只有你買的這檔孤立跌停（1-3檔／4-10檔／11-30檔同時跌停）→ **各期間平均報酬幾乎都是負的**\n"
        "- 當天全市場 **30檔以上同時跌停**（系統性恐慌/股災）→ 報酬與勝率大幅跳升\n\n"
        "換句話說：「連續跌停+爆量」本身不是一個穩定的個股訊號，它真正在偵測的是"
        "**「市場正在集體投降、恐慌性拋售已經到極限」**——這是總經/市場層級的事件，不是個股層級的。"
    )

    horizon_pick = st.selectbox("選期間看寬度分組結果", HORIZON_ORDER, index=3,
                                 format_func=lambda h: HORIZON_LABELS[h], key="breadth_h")
    bd = by_breadth[by_breadth["horizon"] == horizon_pick].copy()
    order = ["1-3檔(單一個股)", "4-10檔", "11-30檔", "30+檔(全面性恐慌)"]
    bd["breadth_bucket"] = pd.Categorical(bd["breadth_bucket"], categories=order, ordered=True)
    bd = bd.sort_values("breadth_bucket")
    bd_disp = bd.rename(columns={"breadth_bucket": "當天全市場跌停家數", "n": "樣本數",
                                  "mean": "平均報酬%", "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
    bd_disp = bd_disp[["當天全市場跌停家數", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]]

    def color_ret(v):
        if pd.isna(v):
            return ""
        return f"color: {UP_COLOR}" if v > 0 else (f"color: {DOWN_COLOR}" if v < 0 else "")

    styled = bd_disp.style.map(color_ret, subset=["平均報酬%"]).format(
        {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:,}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    chart = alt.Chart(bd_disp).mark_bar(size=40).encode(
        x=alt.X("當天全市場跌停家數:N", sort=order, title=None),
        y=alt.Y("平均報酬%:Q", title="平均報酬 (%)"),
        color=alt.condition("datum['平均報酬%'] > 0", alt.value(UP_COLOR), alt.value(DOWN_COLOR)),
        tooltip=["當天全市場跌停家數", alt.Tooltip("平均報酬%:Q", format="+.2f"), alt.Tooltip("樣本數:Q", format=",")],
    ).properties(height=280)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#94a3b8").encode(y="y:Q")
    st.altair_chart(chart + zero, use_container_width=True)

    st.markdown("### 危機事件清單（30檔以上同時跌停，逐次列出）")
    st.caption(
        "把連續發生(間隔≤10天)的日期合併成同一次「危機事件」，11年來約20次。"
        f"其中 {(episodes['fwd20_mean'] > 0).sum()}/{len(episodes)} 次事件後續 +20日平均報酬為正，"
        "但也有像 2015-07~08、2016-11、2018-02、2018-10、2024-03 這種事後沒有反彈甚至續跌的案例——"
        "**不是每次股災都會反彈，勝率是七成五左右，不是保證**。"
    )
    ep_disp = episodes.copy()
    ep_disp["start"] = ep_disp["start"].dt.strftime("%Y-%m-%d")
    ep_disp["end"] = ep_disp["end"].dt.strftime("%Y-%m-%d")
    ep_disp = ep_disp.rename(columns={"start": "開始", "end": "結束", "n_events": "個股數",
                                       "fwd10_mean": "+10日均報酬%", "fwd10_win": "+10日勝率%",
                                       "fwd20_mean": "+20日均報酬%", "fwd20_win": "+20日勝率%"})
    st.dataframe(
        ep_disp.style.map(color_ret, subset=["+10日均報酬%", "+20日均報酬%"]).format(
            {"+10日均報酬%": "{:+.2f}%", "+10日勝率%": "{:.2f}%", "+20日均報酬%": "{:+.2f}%", "+20日勝率%": "{:.2f}%",
             "個股數": "{:,}"}, na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption(
        "注意 2025-04(關稅衝擊)這一次事件就佔了全部訊號的一半以上(555/1044筆)，"
        "單一事件的極端結果會嚴重拉高整體平均值，解讀報酬數字時要意識到這個集中度風險，"
        "20次獨立事件才是比較能代表「這招平均而言值不值得做」的樣本數，不是1044筆。"
    )

    st.markdown("### 逐年 +10日 穩健度（未依寬度篩選前的訊號組全部）")
    yr_disp = year_stats.rename(columns={"year": "年份", "n": "樣本數", "mean": "平均報酬%",
                                          "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
    st.dataframe(
        yr_disp.style.map(color_ret, subset=["平均報酬%"]).format(
            {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:.0f}",
             "年份": "{:.0f}"}, na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption("2016/2017/2019/2022 是負報酬年——都是沒有系統性恐慌事件、只有零星個股跌停的年份，與上面的寬度發現互相印證。")

# ============================================================
# Tab 3: Q1 一定要跌停嗎
# ============================================================
with tab_q1:
    st.markdown("### 不要求觸及跌停，改用「回檔幅度+爆量」測一次")
    st.markdown(
        "訊號改成：收盤價相對**20日內高點**回檔達到某個幅度，且當日爆量(≥2倍均量)，"
        "**不要求真的跌停鎖死**。同一段回檔只算第一次觸發(避免同一次下跌被重複計算)。"
    )
    hcol = st.radio("看哪個持有期間", [5, 10, 20], index=2, horizontal=True, format_func=lambda h: HORIZON_LABELS[h],
                     key="q1_h")
    gg = general_grid[general_grid["horizon"] == hcol].copy()
    gg_disp = gg.rename(columns={"drawdown_th": "回檔幅度門檻%", "n": "樣本數", "mean": "平均報酬%",
                                  "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
    gg_disp = gg_disp[["回檔幅度門檻%", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]]
    st.dataframe(
        gg_disp.style.map(color_ret, subset=["平均報酬%"]).format(
            {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:,}"}),
        use_container_width=True, hide_index=True)

    st.markdown("#### 結論：不用跌停也有效，但效果明顯比較弱")
    st.markdown(
        "- 回檔 -20% + 爆量（強度大約等於連跌2次跌停）：+20日平均 +8.80%／勝率68.6%／Sharpe 0.38\n"
        "- 對照組核心策略（真的跌停+連續2-3天+爆量）：+20日平均 +11.99%／勝率78.3%／Sharpe 0.65\n"
        "- **不需要精確踩到跌停線，「已經跌一大段+爆量」這個廣義邏輯本身就有效**，"
        "而且樣本數多超過10倍（跌停版1000多筆 vs 廣義版1萬多筆），交易機會多很多\n"
        "- 但「跌停鎖死」這個機制本身會再加成一次——它代表賣單在跌停價位强制排隊、"
        "隔天集中反映，比單純陰跌更接近「賣壓一次性出清」，所以效果比同等回檔幅度但沒鎖死的情況更強"
    )

    st.markdown("### 同樣檢查：廣義版本是否也是「市場寬度」在主導？")
    hcol2 = st.radio("看哪個持有期間", [10, 20], index=1, horizontal=True, format_func=lambda h: HORIZON_LABELS[h],
                      key="q1_h2")
    gb = general_breadth[general_breadth["horizon"] == hcol2].copy()
    order2 = ["1-3檔", "4-10檔", "10+檔(全面性)"]
    gb["breadth_bucket"] = pd.Categorical(gb["breadth_bucket"], categories=order2, ordered=True)
    gb = gb.sort_values("breadth_bucket")
    gb_disp = gb.rename(columns={"breadth_bucket": "當天全市場同時觸發家數", "n": "樣本數",
                                  "mean": "平均報酬%", "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
    st.dataframe(
        gb_disp[["當天全市場同時觸發家數", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.map(
            color_ret, subset=["平均報酬%"]).format(
            {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:,}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        "答案是肯定的：不要求跌停的廣義版本，一樣是「市場寬度」在決定成敗——"
        "全市場10檔以上同時觸發回檔+爆量時 Sharpe 明顯優於孤立個股事件。"
        "跌停與否只是強度調整，市場寬度才是核心因子，這點兩種定義下一致。"
    )

# ============================================================
# Tab 4: Q2 反向 -- 高檔爆量出貨？
# ============================================================
with tab_q2:
    st.markdown("### 連續漲停後爆量，是出貨訊號嗎？")
    st.markdown(
        "鏡射測試：個股連續漲停第2-3天 + 當日爆量(≥2倍均量)，之後的報酬是不是比「未爆量」組更差"
        "（驗證「爆量=籌碼在高檔換手/出貨，後續容易回落」的常見說法）？"
    )
    hcol3 = st.radio("看哪個持有期間", [5, 10, 20], index=2, horizontal=True, format_func=lambda h: HORIZON_LABELS[h],
                      key="q2_h")
    rg = reverse_grid[reverse_grid["horizon"] == hcol3].copy()
    rg_disp = rg.rename(columns={"streak_rule": "連續天數規則", "burst": "是否爆量", "n": "樣本數",
                                  "mean": "平均報酬%", "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
    st.dataframe(
        rg_disp[["連續天數規則", "是否爆量", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.map(
            color_ret, subset=["平均報酬%"]).format(
            {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:,}"}),
        use_container_width=True, hide_index=True)

    st.markdown("#### 結論：資料不支持「高檔爆量=出貨反轉」")
    st.markdown(
        "- 不管是連續漲停第1天還是第2-3天，**爆量組的後續報酬跟勝率都比未爆量組更好，不是更差**"
        "（例如首日爆量+20日 +6.75%／勝率55.4% vs 未爆量僅+3.49%／勝率45.7%）\n"
        "- 高檔爆量在台股歷史上比較像是「動能確認/買盤湧入」的訊號，不是「派發出貨」的訊號——"
        "跟直覺相反，但這是統計上一致的結果\n"
        "- 再檢查「當天全市場同時很多檔漲停(全面亢奮)」是否才是真正的反轉/過熱訊號："
    )
    hcol4 = st.radio("看哪個持有期間", [10, 20], index=1, horizontal=True, format_func=lambda h: HORIZON_LABELS[h],
                      key="q2_h2")
    rb = reverse_breadth[reverse_breadth["horizon"] == hcol4].copy()
    order3 = ["<=10檔", "11-30檔", "30+檔(全面亢奮)"]
    rb["breadth_bucket"] = pd.Categorical(rb["breadth_bucket"], categories=order3, ordered=True)
    rb = rb.sort_values("breadth_bucket")
    rb_disp = rb.rename(columns={"breadth_bucket": "當天全市場同時漲停家數", "n": "樣本數",
                                  "mean": "平均報酬%", "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
    st.dataframe(
        rb_disp[["當天全市場同時漲停家數", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.map(
            color_ret, subset=["平均報酬%"]).format(
            {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:,}"}),
        use_container_width=True, hide_index=True)
    st.markdown(
        "**就連「全市場30檔以上同時漲停」(市場全面亢奮/多頭噴出)也沒有出現反轉——後續報酬依然是正的**，"
        "只是幅度比孤立個股稍弱一點點。與跌停側的巨大不對稱形成鮮明對比：市場恐慌有清楚的均值回歸，"
        "市場亢奮沒有對稱的均值回歸。**不建議只憑「高檔爆量」或「集體漲停」當作減碼/放空理由**，"
        "這在歷史資料中站不住腳；真正的出貨訊號應該回到[[漲跌停事件研究]]裡「漲停沒鎖在最高價」"
        "(尾盤有人在高點倒貨)這個已驗證有效的訊號。"
    )

# ============================================================
# Tab 5: 市值/流動性/產業穩健度
# ============================================================
with tab_q3:
    st.markdown("### 核心策略(連續跌停2-3天+爆量, +10日)分組穩健度")
    st.caption(
        "⚠️ 這個分組結果高度受上面「20次危機事件」的產業/規模組成影響"
        "（例如某次股災剛好電子股跌得多，產業分組就會偏向電子股較強），"
        "不是獨立於市場寬度效應之外的另一個因子，解讀時要跟 Tab2 的寬度結論一起看。"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**依市值**")
        cap_order = ["A_大型(>=500億)", "B_中型(100-500億)", "C_中小型(50-100億)", "D_小型(<50億)"]
        cd = by_cap.copy()
        cd["cap_bucket"] = pd.Categorical(cd["cap_bucket"], categories=cap_order, ordered=True)
        cd = cd.sort_values("cap_bucket").rename(
            columns={"cap_bucket": "市值分組", "n": "樣本數", "mean": "平均報酬%", "win": "勝率%",
                     "sharpe": "Sharpe", "pf": "賺賠比"})
        st.dataframe(
            cd[["市值分組", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.format(
                {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:.0f}"}),
            use_container_width=True, hide_index=True)
        st.caption("大型股 Sharpe (0.68) 明顯優於小型股 (0.37)——恐慌時大型股跌停較罕見，一旦發生訊號品質反而更高。")

    with c2:
        st.markdown("**依事件當日流動性(成交金額)**")
        liq_order = ["<1000萬(極低)", "1000萬-5000萬", "5000萬-2億", ">2億(高)"]
        ld = by_liquidity.copy()
        ld["liquidity_bucket"] = pd.Categorical(ld["liquidity_bucket"], categories=liq_order, ordered=True)
        ld = ld.sort_values("liquidity_bucket").rename(
            columns={"liquidity_bucket": "流動性分組", "n": "樣本數", "mean": "平均報酬%", "win": "勝率%",
                     "sharpe": "Sharpe", "pf": "賺賠比"})
        st.dataframe(
            ld[["流動性分組", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.format(
                {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:.0f}"}),
            use_container_width=True, hide_index=True)
        st.caption("沒有單調關係，極低流動性組仍是正報酬——但要注意這是「事件當天」成交金額，不代表隔天真的能無滑價買到。")

    st.markdown("**依產業（樣本數 ≥ 15）**")
    ind = by_industry.sort_values("n", ascending=False).rename(
        columns={"industry": "產業", "n": "樣本數", "mean": "平均報酬%", "win": "勝率%",
                 "sharpe": "Sharpe", "pf": "賺賠比"})
    st.dataframe(
        ind[["產業", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.format(
            {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:.0f}"}),
        use_container_width=True, hide_index=True)
    st.caption("半導體/通訊業/其他電子偏強，塑膠工業接近打平——電子權值股在歷次系統性股災中本來就佔比最高，可能只是反映台股結構而非產業本身的獨立因子。")

# ============================================================
# Tab: ML+GA挖掘版
# ============================================================
with tab_ml:
    if not ML_AVAILABLE:
        st.info("ML+GA 挖掘資料尚未產生，請先執行 D:\\stock\\stock\\burst_ml_ga_study\\ 底下的流程。")
    else:
        rule = ml_data["rule"]["with_position"]
        final_sig = ml_data["final_signals"]
        st.markdown("### 用 LightGBM+SHAP 找因子，GA(deap) 優化門檻")
        st.markdown(
            "為了回答「一定要跌停鎖死嗎？低檔爆量算不算？」，這裡把事件母體放寬到"
            "**不限跌停**：只要「當日收黑≥3% + 成交量≥1.5倍均量」就算候選"
            f"（{len(final_sig):,} 筆候選在此規則下被選中），"
            "讓機器學習從一堆候選特徵裡自己找出真正重要的，而不是人工預設「一定要跌停」。"
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### SHAP 特徵重要性排序")
            shap_df = ml_data["shap_importance"].copy()
            shap_df["feature_cn"] = shap_df["feature"].map({
                "breadth_severe": "市場寬度(跌停家數)", "breadth_moderate": "市場寬度(跌5%+家數)",
                "breadth_up_severe": "市場寬度(漲停家數)", "ret_20d": "20日累積報酬",
                "price_position_252": "52週檔位(低檔位置)", "log_liquidity": "流動性(對數)",
                "is_limit_down": "是否跌停", "industry": "產業", "margin_chg5": "融資使用率5日變化",
                "drawdown_20d": "20日回檔幅度", "ret_5d": "5日累積報酬", "log_mktcap": "市值(對數)",
                "streak_down_days": "連續收黑天數", "near_day_low": "收盤貼近當日低點",
                "vol_ratio": "爆量倍數", "market": "市場別",
            }).fillna(shap_df["feature"])
            chart = alt.Chart(shap_df.head(10)).mark_bar(color=NEUTRAL_COLOR).encode(
                x=alt.X("mean_abs_shap:Q", title="平均|SHAP值|(重要性)"),
                y=alt.Y("feature_cn:N", sort="-x", title=None),
                tooltip=["feature_cn", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)
        with c2:
            st.markdown("#### 關鍵單變量檢查（回答「一定要低檔嗎」）")
            uq = ml_data["univariate_quantiles"]
            pos_q = uq[uq["feature"] == "price_position_252"]
            dd_q = uq[uq["feature"] == "drawdown_20d"]
            st.caption("52週檔位分5組（q0=最接近52週低點的「低檔」～q4=接近52週高點）：")
            st.dataframe(
                pos_q[["q", "mean_fwd10", "win%", "n"]].rename(
                    columns={"q": "分組(0=最低檔)", "mean_fwd10": "+10日均報酬%", "win%": "勝率%", "n": "樣本數"}
                ).style.format({"+10日均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "樣本數": "{:,}"}),
                use_container_width=True, hide_index=True)
            st.caption("20日回檔幅度分5組（q0=跌最深）：")
            st.dataframe(
                dd_q[["q", "mean_fwd10", "win%", "n"]].rename(
                    columns={"q": "分組(0=跌最深)", "mean_fwd10": "+10日均報酬%", "win%": "勝率%", "n": "樣本數"}
                ).style.format({"+10日均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "樣本數": "{:,}"}),
                use_container_width=True, hide_index=True)

        st.markdown("#### 結論：市場寬度依然是王者，「低檔」不是必要條件")
        st.markdown(
            "- **市場寬度**（同一天全市場多少檔跌停/跌5%以上）在 SHAP 排名第1、2名，跟先前純統計研究的發現完全一致\n"
            "- **52週檔位(低不低)排名中段**，單變量檢查也顯示：最低檔那組(q0)平均報酬確實略高，"
            "但差距不大（+1.24% vs 其他組+0.4~1.0%），**不是決定性因子**\n"
            "- 用GA做消融測試(ablation)——拿掉「必須在低檔」這個限制重新優化，"
            f"TRAIN t-統計量幾乎不變（{ml_data['rule']['train_tstat_no_position']:.1f} vs "
            f"{ml_data['rule']['train_tstat_with_position']:.1f}），**代表低檔位置對這個策略沒有增量貢獻**——"
            "它已經被「20日回檔幅度」跟「市場寬度」間接涵蓋了，不需要額外要求股價一定要在52週新低附近\n"
            "- **新發現的因子：融資使用率5日變化**——這支股票的融資使用率最近5天正在明顯下降(去槓桿/斷頭)的那組，"
            "後續報酬也比較好，訊號背後有合理的故事：融資戶已經先被洗出場，賣壓提前釋放"
        )

        st.markdown("### 融資：水位重要，還是變化量重要？")
        st.markdown(
            "個股層級的「整戶維持率」(斷頭門檻，通常130%)是券商帳戶層級的私有資料，"
            "沒有公開的逐股資料源可以取得(永豐/富邦API也查不到，見下方討論)。"
            "能拿到最接近的替代指標是「融資使用率」(該股融資餘額占融資限額的比例)，"
            "測了兩種用法看哪個才是真正有效的："
        )
        mlq = ml_data["margin_level_quantiles"]
        st.caption("融資使用率「水位」分5組（q0=水位最低～q4=水位最高）：")
        st.dataframe(
            mlq[["q", "mean_fwd10", "win%", "n"]].rename(
                columns={"q": "分組", "mean_fwd10": "+10日均報酬%", "win%": "勝率%", "n": "樣本數"}
            ).style.format({"+10日均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "樣本數": "{:,}"}),
            use_container_width=True, hide_index=True)
        st.markdown(
            "水位越高反而報酬略差(q4最高水位組僅+0.60%勝率48%，比q0低水位組還差)，"
            "GA也把「水位門檻」收斂到完全不設限。**真正有貢獻的是「5日內融資餘額下降」這個變化量**"
            "(已經用在上面的最終規則裡)——不是「這支股票原本槓桿重不重」重要，"
            "是「槓桿部位正在被強制/主動清洗」重要，這是動態訊號、不是靜態體質。"
        )

        st.markdown("### GA 最終規則與報酬")
        st.markdown(
            f"- 20日內從高點回檔 ≤ **{rule['drawdown_20d_max']:.1f}%**\n"
            f"- 當日成交量 ≥ **{rule['vol_ratio_min']:.2f}倍**均量（比人工版寬鬆，不用到2倍）\n"
            f"- 當天全市場同時 ≥ **{rule['breadth_severe_min']:.0f}檔**跌停（市場壓力門檻，比人工版「30+檔」寬鬆很多）\n"
            f"- 融資使用率5日變化 ≤ **{rule['margin_chg5_max']:.2f}**（沒有還在加碼融資）\n"
            "- **不要求連續跌停、不要求52週低點**——這是跟人工版最大的不同\n\n"
            f"TRAIN(2015~2023) t-統計量高達 **{rule['train_tstat_with_position']:.1f}**，"
            "但這個數字本身會嚴重高估信心水準——訊號在同一次崩盤日高度相關(不是統計獨立樣本)，"
            "不能直接套 sqrt(n) 解讀成「非常顯著」，實際可信度要看下面的逐年穩健度。"
        )
        hz = ml_data["final_horizon_stats"].rename(columns={"horizon": "持有天數", "n": "樣本數", "mean": "平均報酬%",
                                                              "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
        st.dataframe(
            hz[["持有天數", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.format(
                {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}", "樣本數": "{:,}"}),
            use_container_width=True, hide_index=True)

        st.markdown("### 三層驗證")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**逐年穩健度（+10日）**")
            yr = ml_data["validate_yearly"].rename(columns={"year": "年份", "n": "樣本數", "mean": "平均報酬%",
                                                              "win": "勝率%", "sharpe": "Sharpe", "pf": "賺賠比"})
            st.dataframe(
                yr.style.format({"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}",
                                  "樣本數": "{:.0f}", "年份": "{:.0f}"}, na_rep="—"),
                use_container_width=True, hide_index=True)
            neg = (yr["平均報酬%"] < 0).sum()
            st.caption(f"{len(yr)}年中只有 {neg} 年平均為負（2019年樣本數僅4筆不列入判斷）——"
                       "比先前純跌停版本(11年5年負)明顯更穩健，這是放寬到不限跌停之後樣本更分散帶來的好處。")
        with v2:
            st.markdown("**流動性分組（+10日）**")
            liq = ml_data["validate_liquidity"].rename(columns={"liq_bucket": "流動性分組", "n": "樣本數",
                                                                  "mean": "平均報酬%", "win": "勝率%",
                                                                  "sharpe": "Sharpe", "pf": "賺賠比"})
            st.dataframe(
                liq[["流動性分組", "樣本數", "平均報酬%", "勝率%", "Sharpe", "賺賠比"]].style.format(
                    {"平均報酬%": "{:+.2f}%", "勝率%": "{:.2f}%", "Sharpe": "{:.2f}", "賺賠比": "{:.2f}",
                     "樣本數": "{:.0f}"}),
                use_container_width=True, hide_index=True)
            st.caption("四個流動性分組報酬都差不多，不是靠殭屍股/極端小型股撐出來的假訊號。")

        st.warning(
            "⚠️ **凍結進出場誠實檢查**：約8.6%的訊號日、6.7%的出場日該股票整天鎖死無實際換手(open==close)。"
            "已用「延伸到真正解鎖那天的開盤價」重新估算，修正後 +10日均報酬僅從10.54%微調到10.65%，"
            "**不是靠鎖死無法成交的假訊號撐出來的**——但實際下單時遇到鎖死當天仍要有應變計畫(隔天再補、分批)。"
        )

        st.markdown("### 歷史訊號明細（ML+GA規則）")
        show_cols_ml = ["date", "code", "drawdown_20d", "vol_ratio", "breadth_severe", "margin_chg5",
                         "price_position_252", "is_limit_down", "fwd5_pct", "fwd10_pct", "fwd20_pct"]
        ml_disp = final_sig[show_cols_ml].sort_values("date", ascending=False).rename(columns={
            "date": "日期", "code": "代號", "drawdown_20d": "20日回檔%", "vol_ratio": "量比",
            "breadth_severe": "當天全市場跌停家數", "margin_chg5": "融資使用率5日變化",
            "price_position_252": "52週檔位(0=低)", "is_limit_down": "當天是否跌停",
            "fwd5_pct": "+5日%", "fwd10_pct": "+10日%", "fwd20_pct": "+20日%",
        })
        st.dataframe(
            ml_disp.style.format({
                "20日回檔%": "{:+.2f}%", "量比": "{:.2f}", "當天全市場跌停家數": "{:.0f}",
                "融資使用率5日變化": "{:+.2f}", "52週檔位(0=低)": "{:.2f}", "當天是否跌停": "{:.0f}",
                "+5日%": "{:+.2f}%", "+10日%": "{:+.2f}%", "+20日%": "{:+.2f}%",
            }, na_rep="—"),
            use_container_width=True, hide_index=True, height=380)

# ============================================================
# Tab: 歷史事件
# ============================================================
with tab_history:
    st.markdown("### 逐筆訊號明細")
    yr_min, yr_max = int(core_signals["date"].dt.year.min()), int(core_signals["date"].dt.year.max())
    yr_range = st.slider("年份範圍", yr_min, yr_max, (yr_min, yr_max), key="hist_yr")
    only_high_breadth = st.checkbox("只看「全市場30檔以上同時跌停」的訊號", value=False)

    hd = core_signals[(core_signals["date"].dt.year >= yr_range[0]) & (core_signals["date"].dt.year <= yr_range[1])]
    if only_high_breadth:
        hd = hd[hd["breadth_bucket"] == "30+檔(全面性恐慌)"]

    show_cols = ["date", "code", "streak", "vol_ratio", "event_ret_pct", "close_px", "cap_bucket",
                 "industry", "breadth_down", "fwd1_pct", "fwd5_pct", "fwd10_pct", "fwd20_pct"]
    hd_disp = hd[show_cols].sort_values("date", ascending=False).rename(columns={
        "date": "日期", "code": "代號", "streak": "連續第幾天", "vol_ratio": "量比",
        "event_ret_pct": "當日跌幅%", "close_px": "收盤價", "cap_bucket": "市值分組", "industry": "產業",
        "breadth_down": "當天全市場跌停家數", "fwd1_pct": "+1日%", "fwd5_pct": "+5日%",
        "fwd10_pct": "+10日%", "fwd20_pct": "+20日%",
    })
    st.dataframe(
        hd_disp.style.format({
            "量比": "{:.2f}", "當日跌幅%": "{:+.2f}%", "收盤價": "{:.2f}", "當天全市場跌停家數": "{:.0f}",
            "+1日%": "{:+.2f}%", "+5日%": "{:+.2f}%", "+10日%": "{:+.2f}%", "+20日%": "{:+.2f}%",
        }, na_rep="—"),
        use_container_width=True, hide_index=True, height=420)

    with st.expander("🔍 查特定股票的歷史事件"):
        code_q = st.text_input("輸入股票代號", key="burst_code_lookup")
        if code_q:
            sub = core_signals[core_signals["code"] == code_q.strip()].sort_values("date", ascending=False)
            if sub.empty:
                st.caption("查無事件（該股從未觸發過連續跌停2-3天+爆量的訊號）")
            else:
                st.dataframe(sub[show_cols].rename(columns={
                    "date": "日期", "code": "代號", "streak": "連續第幾天", "vol_ratio": "量比",
                    "event_ret_pct": "當日跌幅%", "close_px": "收盤價", "cap_bucket": "市值分組", "industry": "產業",
                    "breadth_down": "當天全市場跌停家數", "fwd1_pct": "+1日%", "fwd5_pct": "+5日%",
                    "fwd10_pct": "+10日%", "fwd20_pct": "+20日%",
                }).style.format({"量比": "{:.2f}", "當日跌幅%": "{:+.2f}%", "收盤價": "{:.2f}",
                                  "當天全市場跌停家數": "{:.0f}", "+1日%": "{:+.2f}%", "+5日%": "{:+.2f}%",
                                  "+10日%": "{:+.2f}%", "+20日%": "{:+.2f}%"}, na_rep="—"),
                             use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "原始回測程式：D:\\stock\\stock\\limitup_limitdown_study\\burst_rebound_study.py　"
    "延伸自 [[漲跌停事件研究]] 事件庫。"
)
