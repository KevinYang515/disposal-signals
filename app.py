"""
處置股橡皮筋訊號系統
Streamlit webapp — 讀 data/ 下的 CSV，由 update_signals.py 每日更新
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os

st.set_page_config(
    page_title='處置股橡皮筋訊號',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='collapsed',
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# ── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e2530;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 4px;
}
.grade-green  { color: #26c281; font-weight: 700; }
.grade-yellow { color: #f6c90e; font-weight: 700; }
.grade-red    { color: #e74c3c; font-weight: 700; }
.grade-grey   { color: #95a5a6; }
.stDataFrame  { font-size: 13px; }
thead tr th   { background: #1e2530 !important; }
</style>
""", unsafe_allow_html=True)

# ── 讀資料 ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_signals():
    p = f'{DATA_DIR}/signals.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(ttl=300)
def load_history():
    p = f'{DATA_DIR}/history.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(ttl=3600)
def load_grid():
    p = f'{DATA_DIR}/backtest_grid.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_data(ttl=300)
def load_meta():
    p = f'{DATA_DIR}/meta.json'
    if not os.path.exists(p):
        return {'updated_at': '尚未更新'}
    with open(p) as f:
        return json.load(f)

@st.cache_data(ttl=300)
def load_signals_5min():
    p = f'{DATA_DIR}/signals_5min.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p, dtype={'代號': str})

@st.cache_data(ttl=300)
def load_history_5min():
    p = f'{DATA_DIR}/history_5min.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p, dtype={'代號': str})

@st.cache_data(ttl=300)
def load_signals_tail20():
    p = f'{DATA_DIR}/signals_tail20.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p, dtype={'代號': str})

@st.cache_data(ttl=300)
def load_history_tail20():
    p = f'{DATA_DIR}/history_tail20.csv'
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p, dtype={'代號': str})

@st.cache_data(ttl=3600)
def load_exit_timing():
    s = f'{DATA_DIR}/exit_timing_summary.csv'
    d = f'{DATA_DIR}/exit_timing_d1strat.csv'
    if not os.path.exists(s) or not os.path.exists(d):
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(s), pd.read_csv(d)

# ── Header ──────────────────────────────────────────────────────────────
meta = load_meta()
st.title('📈 處置股橡皮筋訊號系統')
data_date = meta.get('data_date', '')
updated_at = meta.get('updated_at', '-')
st.caption(f"腳本更新：{updated_at}　｜　價格資料截至：{data_date}　｜　策略：漲多處置 × 大+中型 × 20分鐘撮合")

# 若價格資料超過 1 個交易日未更新，顯示警告
if data_date:
    from datetime import date
    try:
        lag = (date.today() - pd.Timestamp(data_date).date()).days
        if lag >= 2:
            st.warning(f"價格資料截至 {data_date}，距今 {lag} 天。今D幾估算可能偏差，請更新 finlab_db 後重新執行 update_signals.py")
    except Exception:
        pass

def fmt_pct(v):
    try:
        return f'{float(v):+.2f}%'
    except:
        return '-'

def sharpe_pf(s):
    """(夏普值, 賺賠比) for a return series. Sharpe = mean/std; PF = avg_win/|avg_loss|."""
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan, np.nan
    sharpe = round(s.mean() / s.std(), 2) if s.std() > 0 else np.nan
    wins = s[s > 0]; loss = s[s <= 0]
    pf = round(wins.mean() / abs(loss.mean()), 2) if len(loss) > 0 and loss.mean() != 0 else np.nan
    return sharpe, pf

tab_lab, tab1, tab_5m, tab_t20, tab_cmp, tab2, tab3, tab4, tab5, tab_exit, tab6, tab_cz = st.tabs(['🧪 策略研發', '🔔 今日訊號', '⚡ 5分盤動能', '🚪 出關動能', '⚖️ 策略比較', '📜 歷史回測紀錄', '🔬 自訂策略', '📊 進場網格回測', '📖 策略說明', '📤 持有出清時機', '⚙️ 使用方式', '🏙️ 城中GA研究'])

# ════════════════════════════════════════════════════════
# TAB 1：今日訊號
# ════════════════════════════════════════════════════════
with tab1:
    sig = load_signals()

    if sig.empty:
        st.warning('尚無資料，請先執行 update_signals.py')
        st.stop()

    # 評級顏色對應
    GRADE_COLOR = {
        '✅ 主力訊號':       '#26c281',
        '⚠️ 漲多但大戶減碼': '#f6c90e',
        '🟡 觀察中':         '#f39c12',
        '⬜ 待觀察':         '#95a5a6',
        '❌ 避開':           '#e74c3c',
    }

    # KPI 列
    total     = len(sig)
    main_sig  = len(sig[sig['評級'] == '✅ 主力訊號'])
    watching  = len(sig[sig['評級'].str.contains('觀察')])
    avoid     = len(sig[sig['評級'] == '❌ 避開'])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('📋 追蹤中', total)
    c2.metric('✅ 主力訊號', main_sig)
    c3.metric('🟡 觀察中', watching)
    c4.metric('❌ 避開', avoid)

    st.divider()

    # 過濾選項
    col_f1, col_f2, col_f3 = st.columns([2, 2, 4])
    with col_f1:
        filter_grade = st.multiselect(
            '評級篩選',
            options=['✅ 主力訊號', '⚠️ 漲多但大戶減碼', '🟡 觀察中', '⬜ 待觀察', '❌ 避開'],
            default=['✅ 主力訊號', '⚠️ 漲多但大戶減碼', '🟡 觀察中'],
        )
    with col_f2:
        filter_cap = st.multiselect('規模', options=['大', '中', '小'], default=['大', '中'], key='tab1_cap')
        st.caption('大 >500億 ｜ 中 100~500億 ｜ 小 <100億')

    view = sig.copy()
    if filter_grade:
        view = view[view['評級'].isin(filter_grade)]
    if filter_cap:
        view = view[view['規模'].isin(filter_cap)]

    # D 列動態顯示（只顯示有資料的 D）
    d_cols = []
    for n in range(1, 9):
        col = f'D{n}%'
        if col in view.columns and view[col].notna().any():
            d_cols.append(col)

    display_cols = ['評級', '買進訊號', '代號', '名稱', '規模', '處置原因', '近20日漲幅', '大戶(%)',
                    '起始日', '今D幾', '出關日', '今日漲跌'] + d_cols

    display_cols = [c for c in display_cols if c in view.columns]

    disp = view[display_cols].copy()
    for c in d_cols + ['近20日漲幅', '大戶(%)', '今日漲跌']:
        if c in disp.columns:
            disp[c] = disp[c].apply(fmt_pct)

    # 顏色函數（針對已格式化的字串）
    def color_grade(val):
        color = GRADE_COLOR.get(str(val), '')
        return f'color: {color}; font-weight: bold;' if color else ''

    def color_ret_str(val):
        try:
            v = float(str(val).replace('%', ''))
            if v < -10: return 'color: #e74c3c; font-weight:700'
            if v < -5:  return 'color: #e67e22; font-weight:700'
            if v > 5:   return 'color: #26c281'
        except:
            pass
        return ''

    def color_entry_signal(val):
        return 'color:#26c281;font-weight:700' if str(val).startswith('D') else ''

    styled = disp.style.map(color_grade, subset=['評級'])
    if '買進訊號' in disp.columns:
        styled = styled.map(color_entry_signal, subset=['買進訊號'])
    color_cols = d_cols + (['今日漲跌'] if '今日漲跌' in disp.columns else [])
    if color_cols:
        styled = styled.map(color_ret_str, subset=color_cols)

    st.dataframe(styled, use_container_width=True, height=500)

    # ── 進場預覽 / 觸發分析 ──────────────────────────────────────────────
    if '距觸發(%)' in view.columns and '觸發價' in view.columns:
        with st.expander('📍 進場預覽 / 觸發分析（依缺口排序）'):
            # 用規模篩選但不過濾評級，顯示全部漲多處置的觀察狀況
            prev_base = sig[sig['規模'].isin(filter_cap)].copy() if filter_cap else sig.copy()
            prev_base = prev_base[prev_base['處置原因'] == '漲多處置']

            def fmt_preview(row):
                entry = str(row.get('買進訊號', ''))
                if entry.startswith('D'):
                    return f'✅ 已觸發 ({entry})'
                gap = row.get('距觸發(%)', np.nan)
                if pd.isna(gap):
                    return '-'
                elif gap >= -2:
                    return f'🔥 再跌 {abs(gap):.2f}%'
                elif gap >= -5:
                    return f'🟡 再跌 {abs(gap):.2f}%'
                else:
                    return f'⬜ 再跌 {abs(gap):.2f}%'

            prev_base['明日預覽'] = prev_base.apply(fmt_preview, axis=1)

            # 顯示欄位
            prev_cols = [c for c in ['代號', '名稱', '評級', '買進訊號', '今D幾', '出關日',
                                     '今日漲跌', '目前損益(%)', '觸發價', '距觸發(%)', '明日預覽']
                         if c in prev_base.columns]
            prev_show = prev_base[prev_cols].copy()

            # 格式化數字
            for c in ['今日漲跌', '目前損益(%)', '距觸發(%)']:
                if c in prev_show.columns:
                    prev_show[c] = prev_show[c].apply(fmt_pct)
            if '觸發價' in prev_show.columns:
                prev_show['觸發價'] = prev_show['觸發價'].apply(
                    lambda v: f'{v:.2f}' if pd.notna(v) else '-')

            # 排序：已觸發 > 距觸發接近的（用 sort_key 合成）
            sort_df = prev_base[['買進訊號', '距觸發(%)']].copy()
            sort_df['_triggered'] = sort_df['買進訊號'].apply(lambda v: 0 if str(v).startswith('D') else 1)
            sort_df['_gap_fill']  = sort_df['距觸發(%)'].fillna(-999)
            sort_df = sort_df.sort_values(['_triggered', '_gap_fill'], ascending=[True, False])
            prev_show = prev_show.loc[sort_df.index]

            def color_preview_col(val):
                s = str(val)
                if '✅' in s: return 'color:#26c281;font-weight:700'
                if '🔥' in s: return 'color:#e67e22;font-weight:700'
                if '🟡' in s: return 'color:#f6c90e'
                return 'color:#95a5a6'

            def color_gap_col(val):
                try:
                    v = float(str(val).replace('%', ''))
                    if v >= 0:   return 'color:#26c281;font-weight:700'
                    if v >= -2:  return 'color:#e67e22;font-weight:700'
                    if v >= -5:  return 'color:#f6c90e'
                    return 'color:#95a5a6'
                except:
                    return ''

            styled_prev = prev_show.style.map(color_preview_col, subset=['明日預覽'])
            if '距觸發(%)' in prev_show.columns:
                styled_prev = styled_prev.map(color_gap_col, subset=['距觸發(%)'])
            if '評級' in prev_show.columns:
                styled_prev = styled_prev.map(color_grade, subset=['評級'])
            for c in ['今日漲跌', '目前損益(%)']:
                if c in prev_show.columns:
                    styled_prev = styled_prev.map(color_ret_str, subset=[c])

            st.dataframe(styled_prev, use_container_width=True)
            st.caption('已觸發 = 距觸發 ≥ 0%（目前收盤已低於觸發價）｜🔥 = 距觸發 < 2%（高度警戒）｜觸發條件：D3~D8 任意天累積跌幅 < -5%')

    # 說明
    st.markdown("""
**評級說明：**
✅ 主力訊號 = 漲多處置 + 任意 Dn 累積跌幅 < -5%（歷史勝率 85%+）
⚠️ 漲多但大戶減碼 > -1.5%，大戶在出貨，謹慎
🟡 觀察中 = 所有 Dn 介於 -3%~-5%，橡皮筋還在壓縮中
⬜ 待觀察 = 漲多處置但尚未跌 3%，暫時觀望
❌ 避開 = 跌深處置 + 近20日漲幅 < 0（真實賣壓，非機制壓縮）
""")

    # 進場時序提醒
    with st.expander('📅 進場時序說明'):
        st.markdown("""
| 時間點 | 動作 |
|---|---|
| 每日收盤後 | 看今日是第幾天（D幾），確認累積跌幅 |
| **任意 Dn 收盤 < -5%** | ✅ **當天收盤買進**（D3~D6 皆有效） |
| 累積跌幅 -3%~-5% | 🟡 繼續觀察，等待跌破 -5% |
| 累積跌幅 < -10% | 加重部位，橡皮筋壓縮更深 |
| 累積跌幅 < -15% | 極度壓縮，D5/D6 仍可進場 |
| **T+1 出關日 開盤賣出** | ✅ **最佳出場點**（85% 勝率 +15%） |
| T+1 收盤才賣 | 較差：勝率降至 79%，均報 +13.5% |
| T+2 開盤才賣 | 相近但多一天風險：86%、+15% |
""")

# ════════════════════════════════════════════════════════
# TAB 5分盤動能（第一次處置）
# ════════════════════════════════════════════════════════
with tab_5m:
    sig5 = load_signals_5min()
    hist5 = load_history_5min()

    st.subheader('⚡ 5分盤動能策略（第一次處置）')
    st.markdown("""
5分盤跟 20分盤是**完全不同的邏輯**：20分盤買「深跌反彈」，5分盤買「**動能延續**」。
26% 的 5分盤事件會惡化成二次處置（續飆到被關 20分盤），這群平均 **+10.6%（勝率 70%）**；沒惡化的只有 +0.1%。

#### 📌 操作 SOP（白話版）

| 步驟 | 動作 |
|---|---|
| **① 處置公告日**（進處置前一天） | 看這檔**今天漲多少**：漲 **2% ~ 9%**（強勢但沒漲停）→ 列入觀察。漲停(>9%)或漲不到2% → 放棄。處置原因含**「當沖比重」條款 → 放棄**（此類統計為負，訊號欄會標 🚫） |
| **② D1 = 第一個處置日開盤** | 看**開盤價有沒有比昨天收盤高**：開平盤或開低 → 準備買。開高（跳空上漲）→ 放棄，因為你會買貴 |
| **③ D1 收盤前** | 13:25 那一撮出來後，**用市價（或漲停限價）掛進 13:30 尾撮**——成交價就是收盤價，跟回測完全一致。**不要掛低一個 tick 撿便宜**（tick 實測成交率只 10%，且買到的都是走弱的）。若收盤漲停鎖死買不到 → 放棄 |
| **④ 抱著不動** | 約 11 個交易日，中間震盪**不要停損**（見下方風險說明） |
| **⑤ 出關日或前一天** | **最後處置日收盤 或 出關第一天收盤 賣出**（兩者績效相同），不要抱更久 |

⚠️ 前一日**漲停（>9%）反而不買**——漲停隔天易被出貨，統計上明顯較差。
""")

    with st.expander('📅 買賣時點結論（25 組進出場網格回測）'):
        st.markdown("""
| 問題 | 結論 |
|---|---|
| **何時買** | ✅ **D1（第一個處置日）收盤買**——全網格最強（勝率 68%、t=5.4）。等收盤才確認「未跳空 + 全日走勢」，也避免追開盤 |
| D1 開盤買可以嗎 | 可以但較差（平均少 ~0.8%）：弱開後盤中通常續跌，收盤買更便宜 |
| 晚點進場呢 | D2、D3 進場報酬單調遞減（+6.8% → +6.4% → +5.7%），**越早越好，過了 D3 不建議追** |
| **何時賣** | ✅ **最後處置日收盤 或 出關D1 收盤**，兩者統計等效（+6.8% vs +7.1%）。出關後只剩 +0.2~0.5% 殘餘漂移且勝率下降，**不要抱過出關D2** |
| 要篩市值嗎 | **不用**（與 20分盤不同）：小/中/大型在 ⭐訊號內全部 5/5 年正（+6.2 / +8.5 / +7.7%） |
| 要篩產業嗎 | **不用**：半導體最強（+6.4%）、光電最弱（+1.2%）但樣本太小（n=20~41）不顯著，篩了反而過擬 |
| 前10日漲幅有用嗎 | **沒用**（rank IC ≈ 0）：能進 5分盤處置本身就保證了近期大漲，漲多的「程度」不帶額外資訊 |
| 出關時進了二次處置要續抱嗎 | **不要（接力判死）**：接力那些筆單筆多賺 +3.8%，但多佔 11 天資金（0.35%/天 < 策略平均 0.64%/天），組合年化 Sharpe 2.08→1.92。**出關就賣，二次處置讓 20分盤策略照它自己的規則接手** |
| 要看大盤臉色嗎 | **不用**：OTC 20MA 下方期間的訊號反而更強（+10.42% vs +6.33%）——處置動能是個股籌碼行為，不是大盤 beta |
| 處置條款有差嗎 | **有**：「10日6次注意」觸發的最強（+11.18%、勝率 77.27%）；**「當沖比重」條款是毒（-1.15%，全樣本 -1.43%）→ 已自動排除** |
| 有季節性嗎 | 沒有：12 個月全數為正（+1.3% ~ +18.4%），四季 +5.0 ~ +9.1%，不用挑月份 |
| 收盤買得到嗎（tick 實測） | **買得到**：抽 10 檔 D1 實測，尾撮（13:30）金額中位 962 萬、最薄 49 萬 → **單筆金額控制在該股尾撮量能 1/3 內**，用市價掛進尾撮 = 成交在收盤價，與回測一致 |
| 掛低一個 tick 呢 | **不要**：實測成交率只 10%（掛 13:25 價也只 50%）——動能股尾撮常走高（平均 +0.37%），掛低價只買得到走弱的，是逆選擇 |
""")

    if hist5.empty:
        st.warning('尚無 5分盤資料，請先執行 update_signals.py')
    else:
        h = hist5.copy()
        h['策略報酬(%)'] = pd.to_numeric(h['策略報酬(%)'], errors='coerce')
        hit = h[h['符合因子'] == '✅']

        # ⭐ = 完整買進訊號（前日2~9% + D1未跳空上漲），已排除 D1 收盤漲停買不到的
        star = h[h.get('加強訊號', pd.Series(dtype=str)).eq('⭐')] if '加強訊號' in h.columns else pd.DataFrame()

        # ── KPI（完整買進訊號，2022~至今全歷史）──
        kpi = star if len(star) >= 20 else hit
        kpi_name = '買進訊號' if len(star) >= 20 else '符合因子'
        r_kpi = kpi['策略報酬(%)'].dropna()
        sharpe, pf = sharpe_pf(r_kpi)
        yr_mean = kpi.groupby('年份')['策略報酬(%)'].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f'歷史樣本（{kpi_name}）', f"{len(r_kpi)} 筆")
        c2.metric('勝率', f"{(r_kpi > 0).mean()*100:.2f}%")
        c3.metric('平均報酬', f"{r_kpi.mean():+.2f}%")
        c4.metric('中位數', f"{r_kpi.median():+.2f}%")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric('夏普值（每筆）', f"{sharpe:.2f}" if pd.notna(sharpe) else '-')
        c6.metric('賺賠比', f"{pf:.2f}" if pd.notna(pf) else '-')
        c7.metric('正報酬年數', f"{(yr_mean > 0).sum()}/{len(yr_mean)}")
        c8.metric('最差單筆', f"{r_kpi.min():+.2f}%")
        st.caption(f'統計期間 2022 ~ 至今**全歷史**（非單一年度），已扣成本 0.357%（手續費2折雙邊 + 證交稅0.3%）。'
                   f'D1 收盤漲停買不到的事件已從統計中排除。處置股買進需預收全額款券。')
        st.info('📐 **夏普值口徑說明**：上方「夏普值（每筆）」是單筆報酬分布的 mean/std，跟一般說的年化夏普不同口徑。'
                '以**日權益曲線**回測（每日資金等權分配到在倉部位）：**年化 Sharpe 2.08**、逐年 +41% / +122% / +210% / +218% / +122% 全數為正；'
                '保守倉位版（單檔上限 20% 資金）年化 Sharpe 2.09、MaxDD -22.08%、4.5 年總報酬 +859%。')
        st.success('🧪 **樣本外驗證通過**：同一訊號套用到 **2016~2021**（六年、從未參與任何參數優化）：'
                   'n=126、**+6.14%/筆、勝率 57.94%、6/6 年正**、最差年 +0.33%——'
                   '對比 2022+ 樣本內 +7.06%，衰減很小，策略非過度擬合。')

        # ── 今日訊號 ──
        st.markdown('#### 🔔 目前處置中（5分盤）')
        if sig5.empty:
            st.info('目前沒有進行中的 5分盤處置')
        else:
            s5 = sig5.copy()
            sig_col = '訊號' if '訊號' in s5.columns else None
            if sig_col:
                s5[sig_col] = s5[sig_col].fillna('—')

            def hl_hit(row):
                v = row.get('訊號', '')
                if isinstance(v, str) and v.startswith('🟢'):
                    return ['background-color: #1a4a35'] * len(row)
                if isinstance(v, str) and v.startswith('🟡'):
                    return ['background-color: #3d3517'] * len(row)
                return [''] * len(row)

            st.dataframe(
                s5.style.apply(hl_hit, axis=1)
                  .format({**{c: '{:+.2f}' for c in ['前日漲幅(%)', 'D1跳空(%)', '目前損益(%)', '今日漲跌'] if c in s5.columns},
                            **({'進場價(D1收)': '{:.2f}'} if '進場價(D1收)' in s5.columns else {})}, na_rep='-'),
                use_container_width=True, hide_index=True)
            st.markdown("""
**訊號欄說明**：
🟢 **買進** = 前一日漲2~9% ✚ D1開盤沒跳空上漲，兩個條件都成立 → D1收盤買（若已過D1則顯示歷史觸發狀態）
🟡 **等D1開盤確認** = 前一日漲幅符合，但還沒到D1開盤，明天開盤見真章
🔒 = 條件都符合但D1收盤漲停鎖死，實際買不到 → 放棄
🚫 = 處置由「當沖比重」條款觸發，此類統計為負 → 不買
❌ = D1開高（跳空上漲），會買貴 → 放棄
— = 前一日漲幅不符合（不到2%或超過9%）
""")

        # ── 逐年績效 ──
        st.markdown('#### 📊 逐年績效（買進訊號 vs 全部5分盤）— 證明不是只靠某一年')
        src = star if len(star) >= 20 else hit
        ytbl = pd.DataFrame({
            '買進訊號 n': src.groupby('年份')['策略報酬(%)'].size(),
            '買進訊號 平均%': src.groupby('年份')['策略報酬(%)'].mean().round(2),
            '買進訊號 勝率%': src.groupby('年份')['策略報酬(%)'].apply(lambda x: round((x > 0).mean()*100, 2)),
            '買進訊號 夏普': src.groupby('年份')['策略報酬(%)'].apply(lambda x: sharpe_pf(x)[0]),
            '買進訊號 賺賠比': src.groupby('年份')['策略報酬(%)'].apply(lambda x: sharpe_pf(x)[1]),
            '全部 平均%': h.groupby('年份')['策略報酬(%)'].mean().round(2),
        })
        st.dataframe(ytbl.style.format({'買進訊號 平均%': '{:+.2f}', '全部 平均%': '{:+.2f}',
                                        '買進訊號 勝率%': '{:.2f}', '買進訊號 夏普': '{:.2f}',
                                        '買進訊號 賺賠比': '{:.2f}'}, na_rep='-'),
                     use_container_width=True)

        # ── 前日漲幅分組對照 ──
        st.markdown('#### 🧲 為什麼是 2~9%？前日漲幅分組對照')
        h['g'] = pd.cut(h['前日漲幅(%)'], bins=[-100, 0, 2, 9, 100],
                        labels=['前日下跌', '0~2%（溫和）', '2~9%（強但未漲停）', '>9%（漲停）'])
        btbl = h.groupby('g', observed=True)['策略報酬(%)'].agg(
            n='size',
            勝率=lambda x: round((x > 0).mean()*100, 2),
            平均=lambda x: round(x.mean(), 2),
            中位=lambda x: round(x.median(), 2),
            賺賠比=lambda x: sharpe_pf(x)[1])
        st.dataframe(btbl.style.format({'勝率': '{:.2f}', '平均': '{:+.2f}', '中位': '{:+.2f}',
                                        '賺賠比': '{:.2f}'}, na_rep='-'),
                     use_container_width=True)
        st.caption('獲利峰值在「強但未鎖死」：還有續航力、又沒吸引到漲停隔日的出貨賣壓。')

        # ── D1 跳空分組（加強訊號依據）──
        if 'D1跳空(%)' in hit.columns and hit['D1跳空(%)'].notna().sum() > 50:
            st.markdown('#### ⭐ 為什麼要等 D1 弱開？符合因子內的 D1 跳空分組')
            hb = hit.copy()
            hb['gq'] = pd.cut(hb['D1跳空(%)'], bins=[-100, -2, -0.5, 0.5, 2, 100],
                              labels=['跳空<-2%', '-2~-0.5%', '-0.5~0.5%', '0.5~2%', '跳空>2%'])
            gtbl = hb.groupby('gq', observed=True)['策略報酬(%)'].agg(
                n='size',
                勝率=lambda x: round((x > 0).mean()*100, 2),
                平均=lambda x: round(x.mean(), 2),
                中位=lambda x: round(x.median(), 2))
            st.dataframe(gtbl.style.format({'勝率': '{:.2f}', '平均': '{:+.2f}', '中位': '{:+.2f}'},
                                           na_rep='-'), use_container_width=True)
            st.caption('單調遞減：D1 跳空越高越差。高開 = 搶跑買盤墊高進場成本；弱開才有處置恐慌折價可賺。')

        # ── 歷史回測紀錄（樣式對齊 20分盤歷史頁）──
        st.divider()
        st.markdown('#### 📜 歷史回測紀錄')
        hf = h.drop(columns=['g'], errors='ignore').copy()
        hf['結果'] = hf['策略報酬(%)'].apply(
            lambda v: f'✅ 獲利 {v:+.2f}%' if pd.notna(v) and v > 0
            else (f'❌ 虧損 {v:+.2f}%' if pd.notna(v) else '—'))

        f1, f2, f3 = st.columns([2, 4, 2])
        with f1:
            yr_sel5 = st.selectbox('年份', ['全部'] + sorted(hf['年份'].astype(str).unique().tolist(), reverse=True), key='h5_yr')
        with f2:
            SIG_OPTS = ['🟢 買進訊號（前日2~9% + D1未跳空）', '符合前日2~9%（不看D1跳空）', '全部5分盤']
            sig_sel5 = st.selectbox('訊號條件', SIG_OPTS, index=0, key='h5_sig')
        with f3:
            cap_sel5 = st.multiselect('規模', ['大', '中', '小'], default=['大', '中', '小'], key='h5_cap')

        v5 = hf.copy()
        if yr_sel5 != '全部':
            v5 = v5[v5['年份'].astype(str) == yr_sel5]
        if sig_sel5 == SIG_OPTS[0] and '加強訊號' in v5.columns:
            v5 = v5[v5['加強訊號'] == '⭐']
        elif sig_sel5 == SIG_OPTS[1]:
            v5 = v5[v5['符合因子'] == '✅']
        if cap_sel5:
            v5 = v5[v5['規模'].isin(cap_sel5)]

        if len(v5) == 0:
            st.warning('目前篩選條件無資料')
        else:
            r5 = v5['策略報酬(%)'].dropna()
            sp5, pf5 = sharpe_pf(r5)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric('總筆數', f'{len(r5)} 筆')
            k2.metric('勝率', f'{(r5 > 0).mean()*100:.2f}%')
            k3.metric('期望報酬', f'{r5.mean():+.2f}%')
            k4.metric('夏普值', f'{sp5:.2f}' if pd.notna(sp5) else '-')
            k5.metric('賺賠比', f'{pf5:.2f}' if pd.notna(pf5) else '-')
            k6, k7, k8, k9, k10 = st.columns(5)
            k6.metric('獲利筆', int((r5 > 0).sum()))
            k7.metric('虧損筆', int((r5 <= 0).sum()))
            k8.metric('均獲利', f'{r5[r5 > 0].mean():+.2f}%' if (r5 > 0).any() else '-')
            k9.metric('均虧損', f'{r5[r5 <= 0].mean():+.2f}%' if (r5 <= 0).any() else '-')
            k10.metric('最大虧損', f'{r5.min():+.2f}%')

            def _c_res5(val):
                if str(val).startswith('✅'): return 'color: #26c281; font-weight: 700'
                if str(val).startswith('❌'): return 'color: #e74c3c; font-weight: 700'
                return ''

            def _c_ret5(val):
                try:
                    v = float(str(val).replace('%', ''))
                    if v > 10:  return 'color: #26c281; font-weight: 700'
                    if v > 0:   return 'color: #2ecc71'
                    if v > -5:  return 'color: #e67e22'
                    return 'color: #e74c3c; font-weight: 700'
                except:
                    return ''

            disp5 = v5.sort_values('起始日', ascending=False).reset_index(drop=True)
            pct_cols5 = ['前日漲幅(%)', 'D1跳空(%)', 'D1%', 'D3%', 'D5%', 'D8%', '策略報酬(%)']
            for c in pct_cols5:
                if c in disp5.columns:
                    disp5[c] = disp5[c].apply(fmt_pct)
            st.dataframe(
                disp5.style
                    .map(_c_res5, subset=['結果'])
                    .map(_c_ret5, subset=[c for c in ['策略報酬(%)', 'D1%', 'D3%', 'D5%', 'D8%'] if c in disp5.columns]),
                use_container_width=True, hide_index=True, height=520)
            st.caption('策略報酬 = D1收盤買 → 出關D1收盤賣，已扣成本 0.357%；D1%~D8% = 相對處置前一日收盤的累積漲跌')
            st.download_button('📥 下載此表 CSV', disp5.to_csv(index=False, encoding='utf-8-sig'),
                               'history_5min.csv', 'text/csv', key='dl_h5')

            # 累積報酬走勢
            st.markdown('**累積報酬走勢**（假設每筆等權重）')
            cum5 = (v5.sort_values('起始日')['策略報酬(%)'].dropna() / 100 + 1).cumprod() - 1
            cum5.index = range(len(cum5))
            st.line_chart(pd.DataFrame({'累積報酬(%)': (cum5 * 100).values}))

        st.markdown("""
##### 💰 倉位建議（依 4.5 年日權益回測）

歷史上平均**同時在倉 2.7 檔**（中位 2 檔、前 10% 忙碌期 6 檔、極端 16 檔），69% 的日子有部位。
單檔投入上限對績效的影響：

| 單檔上限（佔策略資金） | 年化 Sharpe | MaxDD | 年化報酬 | 適合 |
|---|---|---|---|---|
| **10%** | 2.01 | **-14.77%** | +35.3% | 保守：最平穩，資金忙碌期也裝得下 6+ 檔 |
| **20%（建議）** | 2.09 | -22.08% | +65.3% | 標準：報酬/回撤平衡點 |
| 33% | 2.11 | -22.08% | +92.7% | 積極：同時 >3 檔時會裝不下，需捨棄訊號 |

**執行規則：**
1. 每檔進場投入固定比例（建議 20%），**絕不單檔梭哈**——約 12% 的訊號會賠 10% 以上（最差 -26.54%），單筆倉位是唯一風控手段
2. 訊號多到裝不下時，優先選**前日漲幅較高**的（band 內動能強度）
3. 中途**不停損**（回測證明停損全數傷績效）、不加碼、抱到出關
4. 剩餘資金保持現金或放低風險部位，不要拿去追不符合訊號的處置股

##### ⚠️ 風險提醒（輸家歸因結論，2026-07 完整分析）
- **約 12% 的訊號會大賠（<-10%，平均 -16.87%）**，且**無法預先識別**：20 個籌碼/技術因子
  （外資、投信、融資、大戶、散戶、量能、波動…）在大賠組 vs 其他組全部無顯著差異（p 值全 > 0.2）
- **不要停損**：測過 7 組停損規則（D2~D8 觸發 -8% ~ -15%），**全部讓平均報酬變差**
  （+6.71% 掉到 +4.53 ~ +6.03%）。原因：贏家在 D3~D4 的中位數也是 -1.5 ~ -5%，
  停損砍掉的常是後來的大贏家——這個策略的本質是「抱過震盪，換 32% 機率的惡化大獎」
- **風控唯一手段 = 倉位**：單筆投入固定小額（例如總資金 5~10%），靠 152 筆的大數法則賺期望值
- 歷史 p5 = -17.05%、最差單筆 -26.54%（金寶 2026-01）
- 頻率約每月 2~3 筆；獲利集中在處置後期（D3 出場 ≈ 0%、出關 +6.6%），中途下車等於白做
""")

# ════════════════════════════════════════════════════════
# TAB：出關動能（20分盤尾段）
# ════════════════════════════════════════════════════════
with tab_t20:
    sig_t = load_signals_tail20()
    hist_t = load_history_tail20()

    st.subheader('🚪 出關動能策略（20分盤尾段）')
    st.markdown("""
只吃 20分盤處置的**最後一段**：出關日在處置公告時就寫死了（**事先可知，無未來函數**），
統計上**出關當天本身就是大漲日**——解禁後流動性回來、被 20 分鐘撮合憋住的買盤一次進場。
**只持有 3 個交易日**，資金效率約 1.31%/天（5分盤策略抱滿是 0.69%/天）。

#### 📌 操作 SOP（白話版）

| 步驟 | 動作 |
|---|---|
| **① 找標的** | 所有**漲多進 20分盤處置**的股票都適用，**不需要任何其他濾網**（市值/漲幅/條款都不用看） |
| **② 買進日 = 出關日往前第 3 個交易日** | 該日**收盤買**（20分盤用市價掛進 13:30 尾撮，成交價=收盤價）。當天收盤**漲停鎖死 → 放棄** |
| **③ 抱 2 天** | 不停損、不加碼 |
| **④ 出關當天（恢復正常交易第一天）收盤賣** | 已恢復逐筆交易，流動性正常，尾盤市價賣出即可 |

⚠️ 進場那 3 天還在處置期 = 買進需**全額預收款券**。
""")

    if hist_t.empty:
        st.warning('尚無出關動能資料，請先執行 update_signals.py')
    else:
        ht = hist_t.copy()
        ht['策略報酬(%)'] = pd.to_numeric(ht['策略報酬(%)'], errors='coerce')
        ok_t = ht[ht['策略報酬(%)'].notna()]
        r_t = ok_t['策略報酬(%)']
        sp_t, pf_t = sharpe_pf(r_t)
        yr_t = ok_t.groupby('年份')['策略報酬(%)'].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('歷史樣本（2022~）', f'{len(r_t)} 筆')
        c2.metric('勝率', f'{(r_t > 0).mean()*100:.2f}%')
        c3.metric('平均報酬', f'{r_t.mean():+.2f}%')
        c4.metric('中位數', f'{r_t.median():+.2f}%')
        c5, c6, c7, c8 = st.columns(4)
        c5.metric('夏普值（每筆）', f'{sp_t:.2f}' if pd.notna(sp_t) else '-')
        c6.metric('賺賠比', f'{pf_t:.2f}' if pd.notna(pf_t) else '-')
        c7.metric('正報酬年數', f'{(yr_t > 0).sum()}/{len(yr_t)}')
        c8.metric('最差單筆', f'{r_t.min():+.2f}%')
        st.caption('已扣成本 0.357%（手續費2折雙邊 + 證交稅0.3%）。買進日收盤漲停買不到的已排除（表中標 🔒）。')
        st.info('📐 **組合層級（日權益、每日等權在倉、單檔上限 20% 資金）**：2022+ 年化 Sharpe 2.58、'
                'MaxDD -19.61%、年化報酬 +85.9%；上限 33% 版 Sharpe 2.60、MaxDD -30.71%、年化 +132.4%。')
        st.success('🧪 **樣本外驗證通過**：同一規則套用 **2016~2021**（六年、未參與任何優化）：'
                   'n=425、**+2.87%/筆、勝率 57.41%、t=4.88、6/6 年正**、最差年 +0.46%；'
                   '組合層級（上限20%）年化 Sharpe 1.71、0 負年。11 年裡 10 年正，唯一弱年 2022 約打平。')

        # ── 今日訊號 ──
        st.markdown('#### 🔔 目前處置中（20分盤出關倒數）')
        if sig_t.empty:
            st.info('目前沒有進行中的 20分盤漲多處置')
        else:
            s_t = sig_t.copy()

            def hl_t20(row):
                v = str(row.get('訊號', ''))
                if v.startswith('🟢'):
                    return ['background-color: #1a4a35'] * len(row)
                if v.startswith('🔴'):
                    return ['background-color: #4a2a1a'] * len(row)
                if v.startswith('🔵'):
                    return ['background-color: #1a2a4a'] * len(row)
                return [''] * len(row)

            st.dataframe(
                s_t.style.apply(hl_t20, axis=1)
                   .format({**{c: '{:+.2f}' for c in ['目前損益(%)', '今日漲跌'] if c in s_t.columns},
                             **({'進場價': '{:.2f}'} if '進場價' in s_t.columns else {})}, na_rep='-'),
                use_container_width=True, hide_index=True)
            st.markdown("""
**訊號欄說明**：
🟢 **今日收盤買進** = 今天就是出關前第 3 個交易日 → 收盤前用市價掛進尾撮
🔵 **持有中** = 已過買進日，抱著等出關
🔴 **今日收盤賣出** = 今天是出關日（恢復正常交易第一天）→ 尾盤賣出
🔒 = 買進日收盤漲停鎖死買不到 → 放棄這筆
🟡 = 還沒到買進日，顯示預估買進日（遇臨時休市會順延，以每日更新為準）
**深跌單 ✅** = 這檔「歷史回測紀錄」分頁的深跌策略也有進場（兩策略同檔，合計倉位要控管，見下方重疊說明）
""")

        # ── 逐年績效 ──
        st.markdown('#### 📊 逐年績效 — 證明不是只靠某一年')
        ytbl_t = pd.DataFrame({
            'n': ok_t.groupby('年份')['策略報酬(%)'].size(),
            '平均%': ok_t.groupby('年份')['策略報酬(%)'].mean().round(2),
            '勝率%': ok_t.groupby('年份')['策略報酬(%)'].apply(lambda x: round((x > 0).mean()*100, 2)),
            '夏普(每筆)': ok_t.groupby('年份')['策略報酬(%)'].apply(lambda x: sharpe_pf(x)[0]),
            '賺賠比': ok_t.groupby('年份')['策略報酬(%)'].apply(lambda x: sharpe_pf(x)[1]),
        })
        st.dataframe(ytbl_t.style.format({'平均%': '{:+.2f}', '勝率%': '{:.2f}',
                                          '夏普(每筆)': '{:.2f}', '賺賠比': '{:.2f}'}, na_rep='-'),
                     use_container_width=True)

        # ── 與深跌策略的重疊 ──
        with st.expander('🔀 跟「歷史回測紀錄」的深跌策略重疊嗎？（重要）'):
            st.markdown("""
會重疊：**53.8% 的出關動能交易，深跌策略也在同一檔股票上有部位**（它買深跌後會抱到出關 T+1）。

| 子集 | n | 平均/筆 | 勝率 | 正年數 |
|---|---|---|---|---|
| 重疊（深跌單也進場） | 261 | **+5.24%** | 64.37% | 4/5 |
| 不重疊（深跌單沒進場） | 224 | +2.44% | 58.48% | 4/5 |

兩個子集都是正的，所以**這是獨立成立的策略**，不是深跌策略的影子。但注意：
- 重疊組更強（深跌後反彈的動能會一路延續到出關）——如果你兩個策略都跑，重疊時**等於同一檔加倍押注最後 3 天**
- **建議規則：同一檔股票兩策略合計不超過總資金 20%**。深跌單已滿倉位 → 出關動能這筆跳過或減半
- 兩策略出場只差一天（出關收盤 vs 出關 T+1 開盤），重疊部位會同時解除，回撤會同步
""")

        # ── 歷史回測紀錄 ──
        st.divider()
        st.markdown('#### 📜 歷史回測紀錄')
        hf_t = ht.copy()
        hf_t['結果'] = hf_t.apply(
            lambda r: '🔒 買不到' if str(r.get('訊號', '')).startswith('🔒')
            else (f"✅ 獲利 {r['策略報酬(%)']:+.2f}%" if pd.notna(r['策略報酬(%)']) and r['策略報酬(%)'] > 0
                  else (f"❌ 虧損 {r['策略報酬(%)']:+.2f}%" if pd.notna(r['策略報酬(%)']) else '—')), axis=1)

        tf1, tf2, tf3 = st.columns([2, 3, 3])
        with tf1:
            yr_sel_t = st.selectbox('年份', ['全部'] + sorted(hf_t['年份'].astype(str).unique().tolist(), reverse=True), key='ht_yr')
        with tf2:
            ovl_sel = st.selectbox('深跌單重疊', ['全部', '只看重疊（深跌單✅）', '只看不重疊'], key='ht_ovl')
        with tf3:
            cap_sel_t = st.multiselect('規模', ['大', '中', '小'], default=['大', '中', '小'], key='ht_cap')

        v_t = hf_t.copy()
        if yr_sel_t != '全部':
            v_t = v_t[v_t['年份'].astype(str) == yr_sel_t]
        if ovl_sel == '只看重疊（深跌單✅）':
            v_t = v_t[v_t['深跌單'] == '✅']
        elif ovl_sel == '只看不重疊':
            v_t = v_t[v_t['深跌單'] != '✅']
        if cap_sel_t:
            v_t = v_t[v_t['規模'].isin(cap_sel_t)]

        if len(v_t) == 0:
            st.warning('目前篩選條件無資料')
        else:
            rv_t = v_t['策略報酬(%)'].dropna()
            sp_v, pf_v = sharpe_pf(rv_t)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric('總筆數', f'{len(rv_t)} 筆')
            m2.metric('勝率', f'{(rv_t > 0).mean()*100:.2f}%' if len(rv_t) else '-')
            m3.metric('期望報酬', f'{rv_t.mean():+.2f}%' if len(rv_t) else '-')
            m4.metric('夏普值', f'{sp_v:.2f}' if pd.notna(sp_v) else '-')
            m5.metric('賺賠比', f'{pf_v:.2f}' if pd.notna(pf_v) else '-')
            m6, m7, m8, m9, m10 = st.columns(5)
            m6.metric('獲利筆', int((rv_t > 0).sum()))
            m7.metric('虧損筆', int((rv_t <= 0).sum()))
            m8.metric('均獲利', f'{rv_t[rv_t > 0].mean():+.2f}%' if (rv_t > 0).any() else '-')
            m9.metric('均虧損', f'{rv_t[rv_t <= 0].mean():+.2f}%' if (rv_t <= 0).any() else '-')
            m10.metric('最大虧損', f'{rv_t.min():+.2f}%' if len(rv_t) else '-')

            def _c_res_t(val):
                if str(val).startswith('✅'): return 'color: #26c281; font-weight: 700'
                if str(val).startswith('❌'): return 'color: #e74c3c; font-weight: 700'
                return ''

            disp_t = v_t.sort_values('起始日', ascending=False).reset_index(drop=True)
            disp_t['策略報酬(%)'] = disp_t['策略報酬(%)'].apply(fmt_pct)
            price_cols_t = [c for c in ['進場價', '出場價'] if c in disp_t.columns]
            st.dataframe(
                disp_t.style.map(_c_res_t, subset=['結果'])
                    .format({c: '{:.2f}' for c in price_cols_t}, na_rep='-'),
                use_container_width=True, hide_index=True, height=520)
            st.caption('策略報酬 = 出關前第3個交易日收盤買 → 出關當天收盤賣（持有3天），已扣成本 0.357%')
            st.download_button('📥 下載此表 CSV', disp_t.to_csv(index=False, encoding='utf-8-sig'),
                               'history_tail20.csv', 'text/csv', key='dl_ht')

            st.markdown('**累積報酬走勢**（假設每筆等權重）')
            cum_t = (v_t.sort_values('起始日')['策略報酬(%)'].dropna() / 100 + 1).cumprod() - 1
            cum_t.index = range(len(cum_t))
            st.line_chart(pd.DataFrame({'累積報酬(%)': (cum_t * 100).values}))

        st.markdown("""
##### 💰 倉位建議（依 2016~2026 日權益回測）

歷史上平均**同時在倉 2.21 檔**（中位 1 檔、p90 4 檔、極端 16 檔），約 60% 的日子有部位。

| 單檔上限（佔策略資金） | 年化 Sharpe (2022+) | MaxDD | 年化報酬 | 備註 |
|---|---|---|---|---|
| **20%（建議）** | 2.58 | **-19.61%** | +85.9% | 樣本外(2016-2021) Sharpe 1.71、0 負年 |
| 33% | 2.60 | -30.71% | +132.4% | 回撤明顯放大，訊號擠的時候裝不下 |

**執行規則：**
1. 每筆固定投入策略資金的 20%，訊號多到裝不下時優先選**深跌單也有進場的**（重疊組 +5.24% > 不重疊 +2.44%）
2. 同一檔股票若深跌策略已持有 → 兩策略**合計**不超過 20%（不要同檔加倍押）
3. 買進日收盤漲停買不到就放棄，不追隔天
4. 只持有 3 天、不停損不加碼；出關當天收盤一定賣，不留戀

##### ⚠️ 風險提醒
- **2022 年約打平**（+0.8%~+2.0%）：空頭年動能弱，這策略吃的是市場過熱的尾巴，熊市時期望值趨近零（但沒有大虧）
- 進場那 3 天標的仍在處置中，**需全額預收款券**，資金排程要跟 5分盤策略一起算
- 單筆虧損尾部存在（最差可 -15% 以上），單檔倉位上限是唯一風控
- 本策略 2026-07 上線，上線後績效尚未累積，前 3 個月建議以最小單位驗證成交品質
""")

# ════════════════════════════════════════════════════════
# TAB 策略比較：三策略（⭐187 / 橡皮筋 / tail20）資金效率、勝率、賺賠比對照
# ════════════════════════════════════════════════════════
with tab_cmp:
    st.subheader('⚖️ 三策略比較：⭐187 5分盤動能 ／ 橡皮筋 20分盤深跌 ／ tail20 出關動能')
    st.caption('研究快照 2026-07-22（同一套重建方法：D:/stock/disposal-signals/research/capital_efficiency_analysis.py、mdd_forensics_analysis.py），2022年起，單檔上限20%資金。目前實盤僅使用橡皮筋。')

    cmp_rows = [
        {'策略': '⭐187 5分盤動能', 'n': 201, '平均持有天數': 10.26, '平均報酬/筆(%)': 7.03, '中位數報酬/筆(%)': 4.37,
         '勝率(%)': 68.66, '賺賠比': 3.49, '資金效率(%/天)': 0.694, '年化交易頻率(筆/年)': 44.1, 'Sharpe': 2.09, 'MaxDD(%)': -22.08, '2022+負年數': 0},
        {'策略': '橡皮筋 20分盤深跌', 'n': 292, '平均持有天數': 8.23, '平均報酬/筆(%)': 8.70, '中位數報酬/筆(%)': 7.24,
         '勝率(%)': 71.92, '賺賠比': 4.15, '資金效率(%/天)': 1.078, '年化交易頻率(筆/年)': 64.1, 'Sharpe': 1.97, 'MaxDD(%)': -21.28, '2022+負年數': 1},
        {'策略': 'tail20 出關動能', 'n': 497, '平均持有天數': 3.00, '平均報酬/筆(%)': 3.67, '中位數報酬/筆(%)': 3.13,
         '勝率(%)': 60.97, '賺賠比': 2.31, '資金效率(%/天)': 1.224, '年化交易頻率(筆/年)': 109.1, 'Sharpe': 2.33, 'MaxDD(%)': -30.22, '2022+負年數': 1},
    ]
    cmp_df = pd.DataFrame(cmp_rows).set_index('策略')

    def _cmp_best(col, higher_is_better=True):
        def _f(s):
            best = s.max() if higher_is_better else s.min()
            return ['background-color:#1a5c38;color:white;font-weight:700' if v == best else '' for v in s]
        return _f

    styled_cmp = (
        cmp_df.style
        .apply(_cmp_best('勝率(%)'), subset=['勝率(%)'])
        .apply(_cmp_best('賺賠比'), subset=['賺賠比'])
        .apply(_cmp_best('資金效率(%/天)'), subset=['資金效率(%/天)'])
        .apply(_cmp_best('Sharpe'), subset=['Sharpe'])
        .apply(_cmp_best('MaxDD(%)', higher_is_better=True), subset=['MaxDD(%)'])  # MaxDD 越接近0(越大)越好
        .format({
            '平均持有天數': '{:.2f}', '平均報酬/筆(%)': '{:+.2f}', '中位數報酬/筆(%)': '{:+.2f}',
            '勝率(%)': '{:.2f}', '賺賠比': '{:.2f}', '資金效率(%/天)': '{:.2f}',
            '年化交易頻率(筆/年)': '{:.2f}', 'Sharpe': '{:.2f}', 'MaxDD(%)': '{:.2f}',
        })
    )
    st.dataframe(styled_cmp, use_container_width=True)
    st.caption('綠底 = 該欄位表現最好的策略。資金效率 = 逐筆「淨報酬 ÷ 持有天數」取平均，衡量每天資金週轉的報酬率，不是總報酬。')

    st.divider()
    st.markdown("""
**怎麼解讀：**
- **單筆品質最好：橡皮筋**（勝率71.9%、賺賠比4.15，三者最高，也是目前唯一實盤使用的策略）
- **資金效率最好：tail20**（每天資金週轉報酬1.224%，年交易頻率109.1筆/年也最高，資金周轉最快）
- **⭐187 風險最低**（2022年起0負年）且**與另外兩個策略幾乎獨立**（三策略重疊分析：⭐187與橡皮筋/tail20重疊事件都很少，橡皮筋與tail20則重疊62%）——是分散風險效益最高的候選
- 三策略維持**各自獨立資金運用**，不做動態合併資金池；若同時使用橡皮筋+tail20要注意兩者訊號重疊，同一檔股票可能被兩個策略同時選到
- 本表為一次性研究快照，尚未接上每日自動更新；待確認要納入的策略後，再排入「🔔 今日訊號」實際上線
""")


# ════════════════════════════════════════════════════════
# TAB 2：歷史回測紀錄
# ════════════════════════════════════════════════════════
with tab2:
    hist = load_history()

    if hist.empty:
        st.warning('尚無歷史資料，請先執行 update_signals.py')
    else:
        st.subheader('漲多處置 × 歷史交易紀錄')

        # ── 確保規模欄格式一致（舊版 CSV 可能是'大型股'）──
        if '規模' in hist.columns:
            hist['規模'] = hist['規模'].apply(lambda v: '大' if '大' in str(v) else ('中' if '中' in str(v) else '小'))

        # ── 確保 Dn組別 欄位存在（舊版 CSV 相容）──
        if 'Dn組別' not in hist.columns:
            if 'D3組別' in hist.columns:
                hist.rename(columns={'D3組別': 'Dn組別'}, inplace=True)
                hist['Dn組別'] = hist['Dn組別'].str.replace('D3 ', 'Dn ', regex=False)
            elif 'D3累積(%)' in hist.columns:
                def _dng(v):
                    try:
                        v = float(v)
                        if v < -5:  return 'Dn < -5%'
                        if v < 0:   return 'Dn -5%~0%'
                        return 'Dn ≥ 0%'
                    except: return 'Dn無資料'
                hist['Dn組別'] = hist['D3累積(%)'].apply(_dng)

        # ── 確保 處置類型 欄位存在（舊版 CSV 相容）──
        if '處置類型' not in hist.columns:
            hist['處置類型'] = '20分鐘'

        # ── 篩選器 ──
        hist['年份'] = hist['起始日'].str[:4]
        DN_OPTIONS = ['Dn < -5%', 'Dn -5%~0%', 'Dn ≥ 0%', '全部漲多']

        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 5, 2, 2])
        with col_f1:
            sel_year = st.selectbox('年份', ['全部'] + sorted(hist['年份'].unique().tolist(), reverse=True))
        with col_f2:
            sel_dn = st.multiselect(
                'Dn最深跌幅條件（D1~D8 任意最大跌幅）',
                options=DN_OPTIONS,
                default=['Dn < -5%'],
            )
        with col_f3:
            sel_cap = st.multiselect('規模', ['大', '中', '小'], default=['大', '中'], key='tab2_cap')
            st.caption('大 >500億 ｜ 中 100~500億 ｜ 小 <100億')
        with col_f4:
            sel_disp = st.multiselect('處置類型', ['20分鐘', '5分鐘'], default=['20分鐘'], key='tab2_disp')
            st.caption('20分鐘=第二次處置\n5分鐘=第一次處置')

        view_h = hist.copy()
        if sel_year != '全部':
            view_h = view_h[view_h['年份'] == sel_year]
        if sel_dn and '全部漲多' not in sel_dn:
            view_h = view_h[view_h['Dn組別'].isin(sel_dn)]
        if sel_cap:
            view_h = view_h[view_h['規模'].isin(sel_cap)]
        if sel_disp:
            view_h = view_h[view_h['處置類型'].isin(sel_disp)]

        # ── KPI（依篩選結果動態更新）──
        if len(view_h) == 0:
            st.warning('目前篩選條件無資料')
        else:
            wins    = view_h[view_h['結果'].str.startswith('✅')]
            loss    = view_h[view_h['結果'].str.startswith('❌')]
            settled = len(wins) + len(loss)
            wr      = len(wins) / settled * 100 if settled > 0 else 0.0
            avg_r   = view_h['出關報酬(%)'].dropna().mean()
            sharpe, pf = sharpe_pf(view_h['出關報酬(%)'])

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('總筆數', f'{len(view_h)} 筆', delta=f'{settled} 已出關', delta_color='off')
            c2.metric('勝率', f'{wr:.2f}%')
            c3.metric('期望報酬', f'{avg_r:+.2f}%' if pd.notna(avg_r) else '-')
            c4.metric('夏普值', f'{sharpe:.2f}' if pd.notna(sharpe) else '-')
            c5.metric('賺賠比', f'{pf:.2f}' if pd.notna(pf) else '-')
            d1, d2, d3 = st.columns([1, 1, 3])
            d1.metric('獲利筆', len(wins))
            d2.metric('虧損筆', len(loss))

        st.divider()

        # ── 出場時間點比較 ──
        with st.expander('⏰ 出場時間點比較（依目前篩選條件）'):
            exit_timing_info = [
                ('出關報酬(%)',  'T+1 開盤（現行策略）'),
            ] + [(f'T+{k}收盤(%)', f'T+{k} 收盤') for k in range(1, 11)]
            et_rows = []
            for ecol, elabel in exit_timing_info:
                if ecol not in view_h.columns:
                    continue
                es = view_h[ecol].dropna()
                if len(es) == 0:
                    continue
                ew = (es > 0).sum(); el = (es <= 0).sum()
                sp, pf_et = sharpe_pf(es)
                et_rows.append({
                    '出場時間點':  elabel,
                    '筆數':        len(es),
                    '勝率(%)':     round(ew / (ew + el) * 100, 2) if (ew + el) > 0 else 0,
                    '期望報酬(%)':  round(es.mean(), 2),
                    '夏普值':      round(sp, 2) if pd.notna(sp) else sp,
                    '賺賠比':      round(pf_et, 2) if pd.notna(pf_et) else pf_et,
                    '均獲利(%)':   round(es[es > 0].mean(), 2) if ew > 0 else 0,
                    '均虧損(%)':   round(es[es <= 0].mean(), 2) if el > 0 else 0,
                    '最大獲利(%)': round(es.max(), 2),
                    '最大虧損(%)': round(es.min(), 2),
                })
            if et_rows:
                etdf = pd.DataFrame(et_rows).set_index('出場時間點')
                def _et_wr(v):
                    if v >= 85: return 'background-color:#1a5c38;color:white;font-weight:700'
                    if v >= 75: return 'background-color:#26c281;color:black;font-weight:700'
                    if v >= 65: return 'background-color:#f6c90e;color:black'
                    return 'background-color:#c0392b;color:white'
                def _et_ret(v):
                    if v >= 15: return 'background-color:#1a5c38;color:white;font-weight:700'
                    if v >= 8:  return 'background-color:#26c281;color:black;font-weight:700'
                    if v >= 0:  return 'background-color:#f6c90e;color:black'
                    return 'background-color:#c0392b;color:white'
                def _et_sp(v):
                    try:
                        if v >= 1.0: return 'color:#26c281;font-weight:700'
                        if v >= 0.5: return 'color:#2ecc71'
                        if v >= 0:   return 'color:#f6c90e'
                        return 'color:#e74c3c'
                    except: return ''
                st.dataframe(
                    etdf.style
                        .map(_et_wr,  subset=['勝率(%)'])
                        .map(_et_ret, subset=['期望報酬(%)'])
                        .map(_et_sp,  subset=['夏普值'])
                        .format({'勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}',
                                 '夏普值': '{:.2f}', '賺賠比': '{:.2f}',
                                 '均獲利(%)': '{:+.2f}', '均虧損(%)': '{:+.2f}',
                                 '最大獲利(%)': '{:+.2f}', '最大虧損(%)': '{:+.2f}'},
                                na_rep='-'),
                    use_container_width=True, hide_index=False
                )
                st.caption('T+1/T+2/T+3 報酬基準：從買進日收盤 → 各出場時間點收/開盤（T+1 開盤為現行策略）')
                st.download_button('📥 下載此表 CSV', etdf.reset_index().to_csv(index=False, encoding='utf-8-sig'),
                                   'exit_timing_tab2.csv', 'text/csv', key='dl_tab2_exit')

        # ── 比較組（D3篩選效果）──
        cmp_stats = meta.get('cmp_stats', [])
        if cmp_stats:
            with st.expander('📊 各組勝率比較（全部資料）'):
                cmp_df = pd.DataFrame(cmp_stats)[['label', 'n', 'wr', 'ret']]
                cmp_df.columns = ['條件', '筆數', '勝率(%)', '期望報酬(%)']
                def color_cmp_wr(val):
                    try:
                        v = float(val)
                        if v >= 85: return 'color: #26c281; font-weight: 700'
                        if v >= 75: return 'color: #f6c90e'
                        return 'color: #e74c3c'
                    except: return ''
                st.dataframe(
                    cmp_df.style.map(color_cmp_wr, subset=['勝率(%)']).format(
                        {'勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}'}),
                    hide_index=True, use_container_width=True
                )

        # ── 主表 ──
        def color_result(val):
            if str(val).startswith('✅'): return 'color: #26c281; font-weight: 700'
            if str(val).startswith('❌'): return 'color: #e74c3c; font-weight: 700'
            return ''

        def color_ret_h(val):
            try:
                v = float(str(val).replace('%', ''))
                if v > 10:  return 'color: #26c281; font-weight: 700'
                if v > 0:   return 'color: #2ecc71'
                if v > -5:  return 'color: #e67e22'
                return 'color: #e74c3c; font-weight: 700'
            except:
                return ''

        if len(view_h) > 0:
            disp = view_h.drop(columns=['年份', 'Dn組別', 'D3累積(%)'], errors='ignore').reset_index(drop=True)
            # 所有百分比欄位都轉成 2 位小數字串（含 D1~D10 累積/報酬、出關後D1~D5，避免顯示原始浮點數過長）
            ret_cols = ['近20日漲幅'] + [c for c in disp.columns if c.endswith('(%)')]
            for c in ret_cols:
                if c in disp.columns:
                    disp[c] = disp[c].apply(fmt_pct)

            exit_cols = [c for c in ['出關報酬(%)'] + [f'T+{k}收盤(%)' for k in range(1, 11)]
                         if c in disp.columns]
            styled_h = (
                disp.style
                .map(color_result, subset=['結果'])
                .map(color_ret_h,  subset=exit_cols)
            )
            st.dataframe(styled_h, use_container_width=True, height=520)
            st.caption('買進日 = 首次 Dn < -5% 的交易日；出關報酬 = T+1 開盤賣出；T+1收盤/T+2/T+3 = 若繼續持有的報酬（基準為買進日收盤）')

        # 累積報酬走勢
        st.divider()
        st.markdown('**累積報酬走勢**（假設每筆等權重）')
        cum = (view_h.sort_values('起始日')['出關報酬(%)'].dropna() / 100 + 1).cumprod() - 1
        cum.index = range(len(cum))
        chart_df = pd.DataFrame({'累積報酬(%)': (cum * 100).values})
        st.line_chart(chart_df, height=250)


# ════════════════════════════════════════════════════════
# TAB 3：自訂策略回測
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader('🔬 自訂策略回測')
    st.caption('自由組合進場日與跌幅門檻，查看任意策略的歷史勝率與報酬')

    hist_c = load_history()
    if hist_c.empty or 'D1累積(%)' not in hist_c.columns:
        st.warning('缺少 D1~D10 欄位，請重新執行 update_signals.py 後再使用此功能' if not hist_c.empty else '尚無歷史資料')
    else:
        if '規模' in hist_c.columns:
            hist_c['規模'] = hist_c['規模'].apply(lambda v: '大' if '大' in str(v) else ('中' if '中' in str(v) else '小'))
        if '處置類型' not in hist_c.columns:
            hist_c['處置類型'] = '20分鐘'
        hist_c['_year'] = hist_c['起始日'].astype(str).str[:4]

        kpi_area = st.container()   # 全寬佔位，稍後填入 KPI

        ctrl_col, main_col = st.columns([1, 3])
        with ctrl_col:
            mode_c = st.radio('進場模式', ['固定日進場', '範圍進場'])
            sel_cap_c = st.multiselect('規模', ['大', '中', '小'], default=['大', '中'], key='tab_c_cap')
            st.caption('大 >500億\n中 100~500億\n小 <100億')
            sel_disp_c = st.multiselect('處置類型', ['20分鐘', '5分鐘'], default=['20分鐘'], key='tab_c_disp')
            st.caption('20分鐘=第二次處置（主策略）\n5分鐘=第一次處置')
            year_opts_c = ['全部'] + sorted(hist_c['_year'].dropna().unique().tolist(), reverse=True)
            sel_year_c = st.selectbox('年份', year_opts_c, key='tab_c_year')
            threshold_c = st.slider('門檻：Dn累積跌幅 <', -25, 0, -5, 1,
                                    format='%d%%', key='tab_c_thr')
            exit_mode_c = st.selectbox('出場時間點',
                                       ['T+1 開盤'] + [f'T+{k} 收盤' for k in range(1, 11)],
                                       key='tab_c_exit')
            EXIT_COL_MAP = {f'T+{k} 收盤': f'T+{k}收盤(%)' for k in range(1, 11)}
            if mode_c == '固定日進場':
                entry_day_c = st.selectbox('進場日', [f'D{n}' for n in range(1, 11)], index=2, key='tab_c_day')
                entry_n_c = int(entry_day_c[1:])
            else:
                range_days_c = st.slider('進場日範圍', 1, 10, (3, 8), key='tab_c_range')
                range_start_c, range_end_c = range_days_c

        base_c = hist_c.copy()
        if sel_cap_c:
            base_c = base_c[base_c['規模'].isin(sel_cap_c)]
        if sel_disp_c:
            base_c = base_c[base_c['處置類型'].isin(sel_disp_c)]
        if sel_year_c != '全部':
            base_c = base_c[base_c['_year'] == sel_year_c]

        # ── 計算符合條件的進場紀錄 ──
        eligible = pd.DataFrame()
        if mode_c == '固定日進場':
            cum_c = f'D{entry_n_c}累積(%)'
            ret_c = f'D{entry_n_c}報酬(%)'
            if cum_c in base_c.columns:
                eligible = base_c[base_c[cum_c] < threshold_c].copy()
                eligible['_entry_n']   = entry_n_c
                eligible['_entry_cum'] = eligible[cum_c]
                eligible['_entry_ret'] = eligible[ret_c] if ret_c in eligible.columns else np.nan
        else:
            days_r     = [n for n in range(range_start_c, range_end_c + 1)
                          if f'D{n}累積(%)' in base_c.columns]
            cum_cols_r = [f'D{n}累積(%)' for n in days_r]
            ret_cols_r = [f'D{n}報酬(%)' for n in days_r]
            if days_r:
                cum_mat  = base_c[cum_cols_r].values.astype(float)
                has_all_ret = all(c in base_c.columns for c in ret_cols_r)
                ret_mat  = base_c[ret_cols_r].values.astype(float) if has_all_ret else np.full_like(cum_mat, np.nan)
                cum_fill = np.where(np.isnan(cum_mat), np.inf, cum_mat)
                below    = cum_fill < threshold_c
                has_entry_mask = below.any(axis=1)
                if has_entry_mask.any():
                    first_idx = np.argmax(below, axis=1)
                    ri = np.arange(len(cum_mat))
                    eligible = base_c.copy()
                    eligible['_entry_n']   = np.where(has_entry_mask, np.array(days_r)[first_idx], np.nan)
                    eligible['_entry_cum'] = np.where(has_entry_mask, cum_mat[ri, first_idx], np.nan)
                    eligible['_entry_ret'] = np.where(has_entry_mask, ret_mat[ri, first_idx], np.nan)
                    eligible = eligible[eligible['_entry_n'].notna()].copy()

        # 在 exit adjustment 前先存 T+1 開盤報酬，供出場比較表使用
        if not eligible.empty:
            eligible['_ret_t1open'] = eligible['_entry_ret'].copy()

        # ── 出場時間點調整 ──
        # T+k收盤(%) = T+k_close / P_orig_entry - 1（原始策略進場日為基準）
        # 自訂進場 D{n}: T+k_close / P_Dn - 1
        #   = (1 + T+k收/100) * (1 + 買進時累積/100) / (1 + D{n}累積/100) - 1
        if not eligible.empty and exit_mode_c != 'T+1 開盤':
            ecol = EXIT_COL_MAP[exit_mode_c]
            if ecol in eligible.columns and '_entry_cum' in eligible.columns and '買進時累積(%)' in eligible.columns:
                exit_factor  = 1 + eligible[ecol] / 100
                orig_factor  = 1 + eligible['買進時累積(%)'] / 100
                entry_factor = 1 + eligible['_entry_cum'] / 100
                eligible['_entry_ret'] = np.where(
                    entry_factor > 0,
                    (exit_factor * orig_factor / entry_factor - 1) * 100,
                    np.nan
                )
            else:
                eligible['_entry_ret'] = np.nan

        # ── 全寬 KPI（填入 kpi_area，視覺上在 columns 上方）──
        with kpi_area:
            if eligible.empty:
                st.info('目前條件無符合紀錄（可嘗試調寬門檻或擴大進場日範圍）')
            else:
                settled_c = eligible[eligible['_entry_ret'].notna()]
                wins_c    = settled_c[settled_c['_entry_ret'] > 0]
                loss_c    = settled_c[settled_c['_entry_ret'] < 0]
                wr_c      = len(wins_c) / len(settled_c) * 100 if len(settled_c) > 0 else 0
                avg_r_c   = settled_c['_entry_ret'].mean() if len(settled_c) > 0 else np.nan
                avg_w_c   = wins_c['_entry_ret'].mean() if len(wins_c) > 0 else np.nan
                avg_l_c   = loss_c['_entry_ret'].mean() if len(loss_c) > 0 else np.nan
                sharpe_c, pf_c = sharpe_pf(settled_c['_entry_ret'])

                kc1, kc2, kc3, kc4, kc5 = st.columns(5)
                kc1.metric('符合條件', f'{len(eligible)} 筆', delta=f'{len(settled_c)} 已出關', delta_color='off')
                kc2.metric('勝率', f'{wr_c:.2f}%')
                kc3.metric('期望報酬', f'{avg_r_c:+.2f}%' if pd.notna(avg_r_c) else '-')
                kc4.metric('夏普值', f'{sharpe_c:.2f}' if pd.notna(sharpe_c) else '-')
                kc5.metric('賺賠比', f'{pf_c:.2f}' if pd.notna(pf_c) else '-')
                kd1, kd2, kd3 = st.columns([1, 1, 3])
                kd1.metric('均獲利', f'{avg_w_c:+.2f}%' if pd.notna(avg_w_c) else '-')
                kd2.metric('均虧損', f'{avg_l_c:+.2f}%' if pd.notna(avg_l_c) else '-')
                st.divider()

        with main_col:
            if not eligible.empty:
                # ── 各天進場對比表 ──
                year_label_c = f'，{sel_year_c}年' if sel_year_c != '全部' else ''
                st.markdown(f'**各天進場效果對比**（門檻 {threshold_c}%，出場：{exit_mode_c}，同規模篩選{year_label_c}）')
                cmp_rows_c = []
                for n in range(1, 11):
                    cc = f'D{n}累積(%)'
                    if cc not in base_c.columns:
                        continue
                    sub_n = base_c[base_c[cc] < threshold_c].copy()
                    if len(sub_n) < 3:
                        continue
                    if exit_mode_c == 'T+1 開盤':
                        rc = f'D{n}報酬(%)'
                        if rc not in sub_n.columns:
                            continue
                        s = sub_n[rc].dropna()
                    else:
                        ecol_n = EXIT_COL_MAP[exit_mode_c]
                        if ecol_n not in sub_n.columns or '買進時累積(%)' not in sub_n.columns:
                            continue
                        ef   = 1 + sub_n[ecol_n] / 100
                        orig = 1 + sub_n['買進時累積(%)'] / 100
                        nf   = 1 + sub_n[cc] / 100
                        adj  = np.where(nf > 0, (ef * orig / nf - 1) * 100, np.nan)
                        s    = pd.Series(adj).dropna()
                    if len(s) < 3:
                        continue
                    sp_n, pf_n = sharpe_pf(s)
                    cmp_rows_c.append({'進場日': f'D{n}', '樣本N': len(s),
                                       '勝率(%)': round((s > 0).mean() * 100, 2),
                                       '期望報酬(%)': round(s.mean(), 2),
                                       '夏普值': sp_n,
                                       '賺賠比': pf_n,
                                       '均獲利(%)': round(s[s > 0].mean(), 2) if (s > 0).any() else 0,
                                       '均虧損(%)': round(s[s < 0].mean(), 2) if (s < 0).any() else 0})
                def _wr_clr(v):
                    if v >= 85: return 'background-color:#1a5c38;color:white;font-weight:700'
                    if v >= 75: return 'background-color:#26c281;color:black;font-weight:700'
                    if v >= 65: return 'background-color:#f6c90e;color:black'
                    return 'background-color:#c0392b;color:white'

                def _ret_clr(v):
                    if v >= 15: return 'background-color:#1a5c38;color:white;font-weight:700'
                    if v >= 8:  return 'background-color:#26c281;color:black;font-weight:700'
                    if v >= 3:  return 'background-color:#f6c90e;color:black'
                    return 'background-color:#c0392b;color:white'

                def _sp_clr(v):
                    try:
                        if v >= 1.0: return 'color:#26c281;font-weight:700'
                        if v >= 0.5: return 'color:#2ecc71'
                        if v >= 0:   return 'color:#f6c90e'
                        return 'color:#e74c3c'
                    except: return ''

                if cmp_rows_c:
                    cdf = pd.DataFrame(cmp_rows_c).set_index('進場日')
                    st.dataframe(
                        cdf.style
                            .map(_wr_clr,  subset=['勝率(%)'])
                            .map(_ret_clr, subset=['期望報酬(%)'])
                            .map(_sp_clr,  subset=['夏普值'])
                            .format({'勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}',
                                     '夏普值': '{:.2f}', '賺賠比': '{:.2f}',
                                     '均獲利(%)': '{:+.2f}', '均虧損(%)': '{:+.2f}'},
                                    na_rep='-'),
                        use_container_width=True
                    )
                    st.download_button('📥 下載此表 CSV', cdf.reset_index().to_csv(index=False, encoding='utf-8-sig'),
                                       'entry_day_compare.csv', 'text/csv', key='dl_tab3_entry')

                # ── 各出場時間點效果對比 ──────────────────────────────────
                entry_label = (f'D{entry_n_c} 進場' if mode_c == '固定日進場'
                               else f'D{range_start_c}~D{range_end_c} 首觸發')
                st.markdown(f'**各出場時間點效果對比**（{entry_label}，門檻 {threshold_c}%，同規模篩選）')
                exit_cmp_rows = []

                # T+1 開盤（已算好，直接取）
                if '_ret_t1open' in eligible.columns:
                    s_e = eligible['_ret_t1open'].dropna()
                    if len(s_e) > 0:
                        ew_e = (s_e > 0).sum(); el_e = (s_e <= 0).sum()
                        sp_e, pf_e = sharpe_pf(s_e)
                        exit_cmp_rows.append({
                            '出場時間點': 'T+1 開盤 ★',
                            '筆數': len(s_e),
                            '勝率(%)':   round(ew_e / (ew_e + el_e) * 100, 2) if (ew_e + el_e) > 0 else 0,
                            '期望報酬(%)': round(s_e.mean(), 2),
                            '夏普值': sp_e, '賺賠比': pf_e,
                            '均獲利(%)': round(s_e[s_e > 0].mean(), 2) if ew_e > 0 else 0,
                            '均虧損(%)': round(s_e[s_e <= 0].mean(), 2) if el_e > 0 else 0,
                        })

                # T+1 ~ T+10 收盤（套用相同的基準轉換公式）
                for k in range(1, 11):
                    ecol_k = f'T+{k}收盤(%)'
                    if ecol_k not in eligible.columns or '買進時累積(%)' not in eligible.columns:
                        continue
                    ef_k   = 1 + eligible[ecol_k] / 100
                    orig_k = 1 + eligible['買進時累積(%)'] / 100
                    ent_k  = 1 + eligible['_entry_cum'] / 100
                    adj_k  = pd.Series(
                        np.where(ent_k > 0, (ef_k * orig_k / ent_k - 1) * 100, np.nan),
                        index=eligible.index).dropna()
                    if len(adj_k) == 0:
                        continue
                    ew_e = (adj_k > 0).sum(); el_e = (adj_k <= 0).sum()
                    sp_e, pf_e = sharpe_pf(adj_k)
                    exit_cmp_rows.append({
                        '出場時間點': f'T+{k} 收盤',
                        '筆數': len(adj_k),
                        '勝率(%)':   round(ew_e / (ew_e + el_e) * 100, 2) if (ew_e + el_e) > 0 else 0,
                        '期望報酬(%)': round(adj_k.mean(), 2),
                        '夏普值': sp_e, '賺賠比': pf_e,
                        '均獲利(%)': round(adj_k[adj_k > 0].mean(), 2) if ew_e > 0 else 0,
                        '均虧損(%)': round(adj_k[adj_k <= 0].mean(), 2) if el_e > 0 else 0,
                    })

                if exit_cmp_rows:
                    ecdf = pd.DataFrame(exit_cmp_rows).set_index('出場時間點')
                    st.dataframe(
                        ecdf.style
                            .map(_wr_clr,  subset=['勝率(%)'])
                            .map(_ret_clr, subset=['期望報酬(%)'])
                            .map(_sp_clr,  subset=['夏普值'])
                            .format({'勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}',
                                     '夏普值': '{:.2f}', '賺賠比': '{:.2f}',
                                     '均獲利(%)': '{:+.2f}', '均虧損(%)': '{:+.2f}'},
                                    na_rep='-'),
                        use_container_width=True
                    )
                    st.caption('★ = 現行策略（T+1 開盤）｜目前選擇出場：' + exit_mode_c)
                    st.download_button('📥 下載此表 CSV', ecdf.reset_index().to_csv(index=False, encoding='utf-8-sig'),
                                       'exit_timing_compare.csv', 'text/csv', key='dl_tab3_exit')

                st.divider()

                # ── 明細表 ──
                st.markdown('**符合條件明細**')
                detail_c = eligible.copy()
                if mode_c == '固定日進場':
                    detail_c['買進日'] = f'D{entry_n_c}'
                else:
                    detail_c['買進日'] = detail_c['_entry_n'].apply(
                        lambda v: f'D{int(v)}' if pd.notna(v) else '-')
                detail_c['買進累積(%)'] = detail_c['_entry_cum'].round(2)
                detail_c['模擬報酬(%)'] = detail_c['_entry_ret'].round(2)
                detail_c['結果'] = detail_c['_entry_ret'].apply(
                    lambda v: f'✅ {v:+.2f}%' if pd.notna(v) and v > 0
                              else (f'❌ {v:+.2f}%' if pd.notna(v) else '-'))

                show_dc_cols = [c for c in ['代號', '名稱', '規模', '起始日', '買進日',
                                            '近20日漲幅', '大戶(%)', '買進累積(%)', '模擬報酬(%)', '結果']
                                if c in detail_c.columns]
                show_dc = detail_c[show_dc_cols].sort_values('起始日', ascending=False)
                num_dc  = [c for c in ['近20日漲幅', '大戶(%)', '買進累積(%)', '模擬報酬(%)']
                           if c in show_dc.columns]

                def _res_clr_c(val):
                    if str(val).startswith('✅'): return 'color:#26c281;font-weight:700'
                    if str(val).startswith('❌'): return 'color:#e74c3c;font-weight:700'
                    return ''

                st.dataframe(
                    show_dc.style
                        .format({c: '{:+.2f}%' for c in num_dc}, na_rep='-')
                        .map(_res_clr_c, subset=['結果']),
                    use_container_width=True, height=420
                )

                st.divider()

                # ── 出關後停利分析 ──────────────────────────────────────────
                with st.expander('📈 出關後停利分析（從 T+1 開盤持有）'):
                    post_cols_available = [f'出關後D{n}(%)' for n in range(1, 6)
                                           if f'出關後D{n}(%)' in eligible.columns]
                    if not post_cols_available:
                        st.info('缺少出關後資料，請重新執行 update_signals.py')
                    else:
                        post_settled = eligible[eligible['出關報酬(%)'].notna()].copy()

                        # ── 出關後各天走勢統計 ──
                        st.markdown('**出關後各天收盤走勢**（基準：T+1 開盤 = 0%）')
                        pr_rows = []
                        for n in range(1, 6):
                            pc = f'出關後D{n}(%)'
                            if pc not in post_settled.columns:
                                continue
                            s = post_settled[pc].dropna()
                            if len(s) == 0:
                                continue
                            pr_rows.append({
                                '出關後': f'D{n} 收盤',
                                '樣本N':    len(s),
                                '均漲跌(%)': round(s.mean(), 2),
                                '中位數(%)': round(s.median(), 2),
                                '上漲率(%)': round((s > 0).mean() * 100, 2),
                                '最大漲幅(%)': round(s.max(), 2),
                                '最大跌幅(%)': round(s.min(), 2),
                            })
                        if pr_rows:
                            prdf = pd.DataFrame(pr_rows).set_index('出關後')
                            def _pr_ret(v):
                                if v >= 5:  return 'color:#26c281;font-weight:700'
                                if v >= 0:  return 'color:#2ecc71'
                                if v >= -3: return 'color:#e67e22'
                                return 'color:#e74c3c;font-weight:700'
                            st.dataframe(
                                prdf.style
                                    .map(_pr_ret, subset=['均漲跌(%)', '中位數(%)'])
                                    .format({'均漲跌(%)': '{:+.2f}', '中位數(%)': '{:+.2f}',
                                             '上漲率(%)': '{:.2f}',
                                             '最大漲幅(%)': '{:+.2f}', '最大跌幅(%)': '{:+.2f}'}),
                                use_container_width=True
                            )

                        st.divider()

                        # ── 停利模擬 ──
                        st.markdown('**停利模擬**（T+1 開盤後，收盤漲達目標則出，否則持至 D5 收盤）')
                        tp_target = st.slider('停利目標（從 T+1 開盤）', 1, 30, 10, 1,
                                              format='%d%%', key='tab_c_tp')

                        hit_days, hit_rets, miss_rets = [], [], []
                        for _, r in post_settled.iterrows():
                            hit = False
                            for n in range(1, 6):
                                v = r.get(f'出關後D{n}(%)', np.nan)
                                if pd.notna(v) and v >= tp_target:
                                    hit_days.append(n)
                                    hit_rets.append(v)
                                    hit = True
                                    break
                            if not hit:
                                v5 = r.get('出關後D5(%)', np.nan)
                                miss_rets.append(v5 if pd.notna(v5) else np.nan)

                        total_n = len(post_settled)
                        hit_n   = len(hit_days)
                        miss_n  = len(miss_rets)

                        all_tp = hit_rets + [v for v in miss_rets if not np.isnan(v)]
                        sp_tp, pf_tp = sharpe_pf(all_tp)

                        tp1, tp2, tp3, tp4, tp5, tp6 = st.columns(6)
                        tp1.metric('觸發停利', f'{hit_n} 筆 ({hit_n/total_n*100:.2f}%)')
                        tp2.metric('觸發者期望報酬', f'{np.mean(hit_rets):+.2f}%' if hit_rets else '-')
                        tp3.metric('未觸發(持至D5)均', f'{np.nanmean(miss_rets):+.2f}%' if miss_rets else '-')
                        tp4.metric('整體期望報酬', f'{np.mean(all_tp):+.2f}%' if all_tp else '-',
                                   delta='vs T+1開盤直出 0%')
                        tp5.metric('夏普值', f'{sp_tp:.2f}' if pd.notna(sp_tp) else '-')
                        tp6.metric('賺賠比', f'{pf_tp:.2f}' if pd.notna(pf_tp) else '-')

                        if hit_days:
                            from collections import Counter
                            day_cnt = Counter(hit_days)
                            day_df = pd.DataFrame([
                                {'出關後第幾天觸發': f'D{d}', '觸發筆數': day_cnt[d]}
                                for d in sorted(day_cnt)
                            ])
                            st.caption('停利觸發天分佈')
                            st.dataframe(day_df, hide_index=True, use_container_width=False)

# ════════════════════════════════════════════════════════
# TAB 4：進場網格回測
# ════════════════════════════════════════════════════════
with tab4:
    grid = load_grid()

    if grid.empty:
        st.warning('尚無回測資料，請先執行 update_signals.py')
    else:
        st.subheader('漲多處置 × 大+中型 × 20分鐘｜進場日 × 跌幅門檻 網格')
        st.caption('每格：從 Dn 收盤進場 → T+1 出關日開盤賣出的歷史統計')

        metric = st.radio('顯示指標', ['勝率(%)', '期望報酬(%)'], horizontal=True)

        # 轉成 pivot
        val_col = 'wr' if metric == '勝率(%)' else 'ret'
        pivot = grid.pivot(index='day', columns='threshold', values=val_col)
        pivot.index = [f'D{d}' for d in pivot.index]
        pivot.columns = [f'{t:+d}%' for t in pivot.columns]

        # N pivot for annotation
        n_pivot = grid.pivot(index='day', columns='threshold', values='N')
        n_pivot.index = [f'D{d}' for d in n_pivot.index]
        n_pivot.columns = [f'{t:+d}%' for t in n_pivot.columns]

        # 顯示為帶顏色的 dataframe
        def color_cell(val):
            if pd.isna(val): return ''
            if metric == '勝率(%)':
                if val >= 90: return 'background-color: #1a5c38; color: white; font-weight:700'
                if val >= 80: return 'background-color: #26c281; color: black; font-weight:700'
                if val >= 70: return 'background-color: #f6c90e; color: black'
                return 'background-color: #c0392b; color: white'
            else:
                if val >= 15: return 'background-color: #1a5c38; color: white; font-weight:700'
                if val >= 10: return 'background-color: #26c281; color: black; font-weight:700'
                if val >= 5:  return 'background-color: #f6c90e; color: black'
                return 'background-color: #c0392b; color: white'

        fmt = '{:.2f}%' if metric == '勝率(%)' else '{:+.2f}%'
        st.dataframe(
            pivot.style.map(color_cell).format(fmt, na_rep='-'),
            use_container_width=True
        )

        st.caption('（括號內為樣本數）')
        st.dataframe(n_pivot.style.format('{:.0f}', na_rep='-'), use_container_width=True)

        st.markdown("""
**關鍵發現：**
- **D3 < -10%**：91% 勝率，+18.5%　← 最強訊號
- **D5/D6 < -15%**：100%/93% 勝率　← 橡皮筋極度壓縮
- **D1/D2 過濾反效果**：越早的跌深代表真實賣壓，非機制造成
""")

        with st.expander('完整數字'):
            show = grid[['label', 'N', 'wr', 'ret', 'wl']].copy()
            show.columns = ['條件', 'N', '勝率(%)', '期望報酬(%)', '賺賠比']
            st.dataframe(show.style.format({'勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}', '賺賠比': '{:.2f}'}, na_rep='-'), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 5：策略說明
# ════════════════════════════════════════════════════════
with tab5:
    st.subheader('📖 橡皮筋效應策略說明')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 核心機制

**20分鐘撮合制度**（第二次處置）每 20 分鐘才撮合一次委託，
導致急著賣出的人只能接受極差的價格，造成**人工賣壓 → 股價被壓低**。

```
漲多進處置（近20日漲幅 > 0）
    ↓
20分撮合 → 窒息量 + 股價人工壓低
    ↓
任意 Dn 累積跌幅 > 5% = 橡皮筋壓縮
（D3 最常見，D4/D5/D6 也有效）
    ↓
出關後 → 人工壓力消失 → 橡皮筋彈回
    ↓
我們賺的是「壓力解除」那段反彈
```

### 為什麼只看漲多處置？

- **漲多處置**：股票先漲（近3日漲幅超閾值）再被打壓 → 底層有支撐 → 橡皮筋有力
- **跌深處置**：近3日已在跌（超過閾值）→ 真實賣壓 → 出關後繼續跌
- 注意：**處置原因是我們自算的**，依據「進處置前3日累積報酬」是漲是跌分類，非 TWSE 官方標籤
""")

    with col2:
        st.markdown("""
### 市值規模定義

| 規模 | 市值門檻 | 說明 |
|------|---------|------|
| 大型股 | > 500 億元 | 流動性高，訊號穩定性最佳 |
| 中型股 | 100 ~ 500 億元 | 流動性適中，佔歷史樣本最多 |
| 小型股 | < 100 億元 | 流動性較低，橡皮筋效應可能更強但雜訊也更多，建議謹慎使用 |

> 預設篩選「大+中」為策略驗證主體；小型股可在篩選器中手動加入觀察。

### 回測統計（漲多 × 大+中 × 20分）

| 進場條件 | 勝率 | 期望報酬 |
|---|---|---|
| 無過濾（任意進場）| 75% | +9% |
| 任意 Dn < -5% | **85%** | **+15%** |
| 任意 Dn < -10% | **91%** | **+19%** |
| D5/D6 < -15% | **93~100%** | **+18~19%** |

### 輔助判斷

| 因子 | 意義 |
|---|---|
| 近20日漲幅 | 進場前20日漲幅，越高橡皮筋越強 |
| 大戶(%) | 大戶持股變動，< -1.5% 要謹慎 |
| 處置原因 | 漲多 > 震盪 > 跌深 |

### 大盤不影響策略

實測三種大盤環境（強/中/弱）勝率均在 75-90%，
橡皮筋效應是**制度性**的，不依賴市場行情。
""")

# ════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════
# TAB LAB：策略研發（實驗區，不影響主策略）
# ════════════════════════════════════════════════════════
with tab_lab:
    st.subheader('🧪 策略研發實驗區')
    st.caption('此頁為探索新方向用，不影響主策略。確認有效後再考慮整合。')

    hist_lab = load_history()
    if hist_lab.empty or '出關後D1(%)' not in hist_lab.columns:
        st.warning('缺少出關後欄位，請更新 history.csv')
    else:
        if '處置類型' not in hist_lab.columns:
            hist_lab['處置類型'] = '20分鐘'
        if '規模' in hist_lab.columns:
            hist_lab['規模'] = hist_lab['規模'].apply(lambda v: '大' if '大' in str(v) else ('中' if '中' in str(v) else '小'))

        # ── 研發項目一：出關後繼續持有 ──────────────────────────
        st.markdown('### 📐 研發方向三：出關後延長持有')
        st.caption('主策略在 T+1 開盤出場，此分析探索「出場後繼續持有 N 天」是否有額外優勢')

        lab_base = hist_lab[
            (hist_lab['處置類型'] == '20分鐘') &
            (hist_lab['規模'].isin(['大', '中'])) &
            (hist_lab['結果'].astype(str).str.startswith('✅') | hist_lab['結果'].astype(str).str.startswith('❌'))
        ].copy()

        post_cols = [f'出關後D{n}(%)' for n in range(1, 6)]
        available_post = [c for c in post_cols if c in lab_base.columns]

        if not available_post:
            st.warning('缺少 出關後D1~D5 欄位')
        else:
            orig_s = pd.to_numeric(lab_base['出關報酬(%)'], errors='coerce').dropna()

            lc1, lc2 = st.columns(2)

            with lc1:
                st.markdown('**出關後獨立報酬（從 T+1 開盤算起）**')
                post_rows = []
                for c in available_post:
                    s = pd.to_numeric(lab_base[c], errors='coerce').dropna()
                    if len(s) < 5: continue
                    sp, pf = sharpe_pf(s)
                    post_rows.append({
                        '持有期': c.replace('出關後', '').replace('(%)', ''),
                        '樣本N': len(s),
                        '勝率(%)': round((s > 0).mean() * 100, 2),
                        '期望報酬(%)': round(s.mean(), 2),
                        '均獲利(%)': round(s[s > 0].mean(), 2) if (s > 0).any() else 0,
                        '均虧損(%)': round(s[s < 0].mean(), 2) if (s < 0).any() else 0,
                    })
                if post_rows:
                    pdf = pd.DataFrame(post_rows).set_index('持有期')
                    st.dataframe(pdf.style.format({
                        '勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}',
                        '均獲利(%)': '{:+.2f}', '均虧損(%)': '{:+.2f}'
                    }, na_rep='-'), use_container_width=True)

            with lc2:
                st.markdown('**延長持有總報酬（原策略 + 繼續持有 N 天）**')
                ext_rows = [{'出場時機': 'T+1 開盤（原策略）',
                              '樣本N': len(orig_s),
                              '勝率(%)': round((orig_s > 0).mean() * 100, 2),
                              '期望報酬(%)': round(orig_s.mean(), 2)}]
                for c in available_post:
                    merged = lab_base.copy()
                    merged['_orig'] = pd.to_numeric(merged['出關報酬(%)'], errors='coerce')
                    merged['_post'] = pd.to_numeric(merged[c], errors='coerce')
                    merged = merged.dropna(subset=['_orig', '_post'])
                    total = ((1 + merged['_orig'] / 100) * (1 + merged['_post'] / 100) - 1) * 100
                    label = '延長至 ' + c.replace('出關後', '').replace('(%)', '')
                    ext_rows.append({'出場時機': label,
                                     '樣本N': len(total),
                                     '勝率(%)': round((total > 0).mean() * 100, 2),
                                     '期望報酬(%)': round(total.mean(), 2)})
                edf = pd.DataFrame(ext_rows).set_index('出場時機')
                st.dataframe(edf.style.format({
                    '勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}'
                }, na_rep='-'), use_container_width=True)

            st.info('💡 結論：T+1 開盤出場勝率最高（80%+）。延長持有期望報酬略升但勝率明顯下降，建議維持原策略。')

        st.divider()
        st.markdown('### 📐 研發方向一：輔助篩選因子')
        st.caption('探索「近20日漲幅」與「大戶持股變化」是否能作為額外篩選條件，提升期望報酬。')

        lab1_base = hist_lab[
            (hist_lab['處置類型'] == '20分鐘') &
            (hist_lab['規模'].isin(['大', '中'])) &
            (hist_lab['結果'].astype(str).str.startswith('✅') | hist_lab['結果'].astype(str).str.startswith('❌'))
        ].copy()
        lab1_base['_ret'] = pd.to_numeric(lab1_base['出關報酬(%)'], errors='coerce')
        lab1_base['_rise'] = pd.to_numeric(lab1_base['近20日漲幅'], errors='coerce')
        lab1_base['_whale'] = pd.to_numeric(lab1_base['大戶(%)'], errors='coerce')

        l1a, l1b = st.columns(2)

        # ── 近20日漲幅篩選 ──
        with l1a:
            st.markdown('**近20日漲幅門檻篩選**')
            st.caption(f'樣本中位數：{lab1_base["_rise"].median():.2f}%（多數股票處置前已大漲）')
            rise_rows = [{'漲幅門檻': '不篩選（全部）',
                          '樣本N': len(lab1_base['_ret'].dropna()),
                          '勝率(%)': round((lab1_base['_ret'].dropna() > 0).mean() * 100, 2),
                          '期望報酬(%)': round(lab1_base['_ret'].dropna().mean(), 2)}]
            for th in [20, 30, 40, 50, 70, 100]:
                sub = lab1_base[lab1_base['_rise'] >= th]['_ret'].dropna()
                if len(sub) < 5: continue
                rise_rows.append({'漲幅門檻': f'>= {th}%', '樣本N': len(sub),
                                  '勝率(%)': round((sub > 0).mean() * 100, 2),
                                  '期望報酬(%)': round(sub.mean(), 2)})
            rdf = pd.DataFrame(rise_rows).set_index('漲幅門檻')
            st.dataframe(rdf.style.format({
                '勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}'
            }, na_rep='-'), use_container_width=True)
            st.caption('💡 近20日漲幅篩選對結果影響有限，不建議作為主要濾網。')

        # ── 大戶持股變化篩選 ──
        with l1b:
            st.markdown('**大戶持股變化篩選**')
            st.caption('大戶(%)為進場時大戶持股變化；正值=大戶增持，負值=大戶減持')
            whale_rows = [{'大戶條件': '不篩選（全部）',
                           '樣本N': len(lab1_base['_ret'].dropna()),
                           '勝率(%)': round((lab1_base['_ret'].dropna() > 0).mean() * 100, 2),
                           '期望報酬(%)': round(lab1_base['_ret'].dropna().mean(), 2)}]
            # 大戶減持（負值，多數情況）
            for th in [-1, -2, -3, -5]:
                sub = lab1_base[lab1_base['_whale'] <= th]['_ret'].dropna()
                if len(sub) < 5: continue
                whale_rows.append({'大戶條件': f'<= {th}%（減持）', '樣本N': len(sub),
                                   '勝率(%)': round((sub > 0).mean() * 100, 2),
                                   '期望報酬(%)': round(sub.mean(), 2)})
            # 大戶增持（正值，較少）
            for th in [0, 1]:
                sub = lab1_base[lab1_base['_whale'] >= th]['_ret'].dropna()
                if len(sub) < 5: continue
                whale_rows.append({'大戶條件': f'>= {th}%（增持/中性）', '樣本N': len(sub),
                                   '勝率(%)': round((sub > 0).mean() * 100, 2),
                                   '期望報酬(%)': round(sub.mean(), 2)})
            wdf = pd.DataFrame(whale_rows).set_index('大戶條件')
            st.dataframe(wdf.style.format({
                '勝率(%)': '{:.2f}', '期望報酬(%)': '{:+.2f}'
            }, na_rep='-'), use_container_width=True)
            st.caption('💡 大戶增持（≥0%）時期望報酬明顯較高，但樣本僅 ~20 筆，需持續累積觀察。')

        st.divider()
        st.markdown('### 📐 研發方向二：出關後反彈（大戶增持篩選）')
        st.caption('假設：處置期間大戶逆勢增持 → 出關後補漲動能更強。進場：T+1 開盤；與主策略為不同時段的獨立交易。')

        lab2_base = hist_lab[
            (hist_lab['處置類型'] == '20分鐘') &
            (hist_lab['規模'].isin(['大', '中'])) &
            (hist_lab['結果'].astype(str).str.startswith('✅') | hist_lab['結果'].astype(str).str.startswith('❌'))
        ].copy()
        lab2_base['_whale'] = pd.to_numeric(lab2_base['大戶(%)'], errors='coerce')
        lab2_base['_深']   = pd.to_numeric(lab2_base['期間最深(%)'], errors='coerce')

        l2ctrl, l2main = st.columns([1, 3])
        with l2ctrl:
            lab2_whale_thr = st.slider('大戶篩選 ≥', -10, 5, 0, 1, format='%d%%', key='lab2_whale')
            st.caption('0% = 大戶增持/中性\n負值 = 包含更多減持案例')
            lab2_deep_thr = st.slider('期間最深 <（可選）', -40, 0, 0, 1, format='%d%%', key='lab2_deep')
            st.caption('0 = 不限深度\n-10% = 至少跌10%')

        with l2main:
            mask = lab2_base['_whale'] >= lab2_whale_thr
            if lab2_deep_thr < 0:
                mask = mask & (lab2_base['_深'] < lab2_deep_thr)
            filtered = lab2_base[mask].copy()

            st.markdown(f'**出關後持有效果**（大戶 ≥ {lab2_whale_thr}%{"，期間最深 < " + str(lab2_deep_thr) + "%" if lab2_deep_thr < 0 else ""}，樣本 {len(filtered)} 筆）')

            # 對比表：無篩選 vs 篩選後
            compare_rows = []
            for label, subset in [('全體基準', lab2_base), (f'大戶≥{lab2_whale_thr}% 篩選後', filtered)]:
                row = {'條件': label, '樣本N': len(subset)}
                for c in ['出關後D1(%)', '出關後D2(%)', '出關後D3(%)', '出關後D4(%)', '出關後D5(%)']:
                    if c not in subset.columns: continue
                    s = pd.to_numeric(subset[c], errors='coerce').dropna()
                    day = c.replace('出關後', '').replace('(%)', '')
                    row[f'{day} 勝率'] = round((s > 0).mean() * 100, 2) if len(s) >= 3 else np.nan
                    row[f'{day} 期望'] = round(s.mean(), 2) if len(s) >= 3 else np.nan
                compare_rows.append(row)

            if compare_rows:
                cdf2 = pd.DataFrame(compare_rows).set_index('條件')
                wr_cols = [c for c in cdf2.columns if '勝率' in c]
                ret_cols = [c for c in cdf2.columns if '期望' in c]

                def _wr2(v):
                    try:
                        if v >= 80: return 'background-color:#1a5c38;color:white;font-weight:700'
                        if v >= 65: return 'background-color:#26c281;color:black'
                        if v >= 50: return 'background-color:#f6c90e;color:black'
                        return 'background-color:#c0392b;color:white'
                    except: return ''

                def _ret2(v):
                    try:
                        if v >= 8:  return 'background-color:#1a5c38;color:white;font-weight:700'
                        if v >= 3:  return 'background-color:#26c281;color:black'
                        if v >= 0:  return 'background-color:#f6c90e;color:black'
                        return 'background-color:#c0392b;color:white'
                    except: return ''

                fmt = {c: '{:.2f}' for c in wr_cols}
                fmt.update({c: '{:+.2f}' for c in ret_cols})
                st.dataframe(
                    cdf2.style.map(_wr2, subset=wr_cols).map(_ret2, subset=ret_cols)
                              .format(fmt, na_rep='-'),
                    use_container_width=True
                )

            if len(filtered) < 10:
                st.warning(f'⚠️ 目前樣本僅 {len(filtered)} 筆，統計尚不穩定，持續累積中。')
            else:
                st.info(f'💡 大戶增持篩選後，出關後 D2 勝率與期望報酬顯著提升。樣本 {len(filtered)} 筆，建議累積至 50+ 再考慮實盤。')

# TAB 出清時機
# ════════════════════════════════════════════════════════
with tab_exit:
    st.subheader('📤 持有者出清時機分析')
    st.caption('漲多處置 × 20分鐘撮合，基準：進處置前一天收盤（pre-disposal close）')

    et_summary, et_strat = load_exit_timing()

    if et_summary.empty:
        st.info('資料尚未產生，請執行 update_signals.py 後重整。')
    else:
        st.markdown('#### 各時間點出清報酬（相對 pre-disposal 收盤）')

        col_a, col_b = st.columns([3, 2])

        with col_a:
            def style_exit_table(df):
                def color_mean(val):
                    try:
                        v = float(val)
                        if v > 2:   return 'color:#26c281;font-weight:700'
                        if v > 0:   return 'color:#7dcea0'
                        if v > -2:  return 'color:#e59866'
                        return 'color:#e74c3c;font-weight:700'
                    except:
                        return ''
                fmt_et = {c: '{:+.2f}' for c in ['平均報酬(%)','中位數(%)'] if c in df.columns}
                fmt_et.update({c: '{:.2f}' for c in ['上漲機率(%)','跌>3%機率(%)','跌>5%機率(%)'] if c in df.columns})
                return df.style.map(color_mean, subset=['平均報酬(%)','中位數(%)']).format(fmt_et, na_rep='-')

            display_cols = ['時間點','平均報酬(%)','中位數(%)','上漲機率(%)','跌>3%機率(%)','跌>5%機率(%)']
            st.dataframe(
                style_exit_table(et_summary[display_cols]),
                use_container_width=True,
                hide_index=True,
            )

        with col_b:
            st.markdown("""
**重點結論**

- **D1 收盤**：平均 -2.7%，70% 機率下跌
- **D3 / D5**：略優於 D1，但差異不大
- **持到出關**：平均 +4.9%，反超 pre-disposal

> 出清最佳順序：
> **D0（今日，正常交易）> D1 第一盤 > 持到出關**
>
> 若今天還沒賣：D1 第一盤比 D1 收盤好
> 若能撐過整個處置期：出關比 D1 賣好 7~8%
""")

        st.divider()
        st.markdown('#### D1 跌幅分層 → D3 後續走勢')
        st.caption('D1 跌很深，D3 有機會回來嗎？')

        if not et_strat.empty:
            def style_strat(df):
                def color_d3(val):
                    try:
                        v = float(val)
                        if v > 0:   return 'color:#26c281'
                        if v > -3:  return 'color:#e59866'
                        return 'color:#e74c3c'
                    except:
                        return ''
                fmt_strat = {c: '{:+.2f}' for c in ['D3均值(%)'] if c in df.columns}
                fmt_strat.update({c: '{:.2f}' for c in ['佔比(%)','D3比D1好機率(%)'] if c in df.columns})
                return df.style.map(color_d3, subset=['D3均值(%)']).format(fmt_strat, na_rep='-')
            st.dataframe(
                style_strat(et_strat),
                use_container_width=True,
                hide_index=True,
            )
            st.caption('D3比D1好機率：不論 D1 跌多少，D3 回升機率都只有 50~60%，沒有明顯時機優勢——D1 跌深後等 D3 是賭局，不是策略。')

        st.divider()
        st.markdown("""
#### 實務建議

| 情境 | 建議 |
|------|------|
| 今天盤中知道明天進處置 | 今天盤中找相對強勢時賣（D0，正常流動性） |
| 收盤後才知道 | 明天 D1 第一盤賣（9:20 撮合），不等 D1 收盤 |
| 想扛過整個處置期 | 出關 Day1 開盤賣，歷史平均 +4.9%，但需要承受 D1~D5 帳面虧損 |
| 分批策略 | D1 第一盤先出一半鎖定，剩一半等出關賣 |
""")

# TAB 6：使用方式
# ════════════════════════════════════════════════════════
with tab6:
    st.subheader('⚙️ 資料更新方式')
    st.markdown("""
### 每日操作流程

**1. 台股收盤後（14:35 以後）執行更新腳本：**
```bash
cd ~/stock/webapp
python update_signals.py
```

**2. 推上 GitHub（Streamlit Cloud 自動重新部署）：**
```bash
git add data/
git commit -m "update signals $(date +%Y-%m-%d)"
git push
```

**3. 或設定自動排程（每天 15:00 自動執行）：**
```bash
# 編輯 crontab
crontab -e

# 加入這行
0 15 * * 1-5 cd ~/stock/webapp && python update_signals.py && git add data/ && git commit -m "auto update" && git push
```

---

### 訊號解讀

1. 看 **今日訊號** 頁面
2. 找 **✅ 主力訊號**（漲多 + 任意 Dn < -5%）
3. 確認**今D幾**和當天的累積跌幅 Dn%
   - 累積跌幅 < -10% → 加重部位
   - 累積跌幅 -5%~-10% → 標準部位
   - D5/D6 才跌破也有效，不限定必須是 D3
4. **T+1 出關日開盤賣出**（最佳時機），不要提前跑
   - T+1 收盤才賣：勝率下降、均報少 1.5%
   - 提早在 T 收盤賣：勝率只有 76%，少賺 3%
5. 注意**大戶(%)** < -1.5% 的標的降低部位或跳過

---

### 風險提示

- 歷史樣本 84% 集中於 2026 年，策略仍在驗證期
- 建議每筆部位控制在總資金 **2-3%**
- 跌深處置（❌）標的避免進場
""")

# ════════════════════════════════════════════════════════
# TAB CZ：城中GA研究總覽
# ════════════════════════════════════════════════════════
with tab_cz:
    st.subheader('🏙️ 凱基-城中GA 隔日沖放空策略｜研究總覽')
    st.caption('本頁彙整 2026-08-04 一整晚的因子挖掘與回測結果，供之後接手研究時快速回顧。所有數字皆基於歷史樣本，非即時訊號，本頁不自動更新。')

    st.markdown("""
### 核心策略

**D0(訊號日)嚴格鎖漲停 + 凱基-城中當日淨買超金額/影響力雙門檻達標 → D1(隔日)開盤放空、收盤回補。**

固定絕對金額門檻(458.689萬/1.93204%)已證實是用2023年後市場規模校準，對2018-2022年不公平(金額中位數從2018年83萬成長到2026年4,730萬)。改用**動態門檻**：TRAIN(2021-2023)自身分布40百分位校準 → `net_amt_wan ≥ 263.568萬` 且 `cz_influence_pct ≥ 0.86301%`，TRAIN樣本從103筆回升到177筆。正式建議：季度滾動36個月窗口重新校準，不要永久鎖死單一數字。
""")

    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### ✅ 已驗證：進場/風控規則

**1. Gap甜蜜點：D1開盤跳空 1%～9.5%**
| Gap區間 | 平均報酬 | t值 |
|---|---|---|
| <0%開低 | 弱 | — |
| **1%-9%(甜蜜點)** | **TEST +1.49%** | **t=9.78** |
| 9.5%-10% | -2.05% | t=-3.04(轉負) |

**2. ⭐避開gap≥9.5%——最重要的防嘎漲停規則**
D1當天直接鎖死開不了倉的145筆事件中，81.4%的gap都≥9.5%。避開這個門檻能防住九成以上的鎖死災難案例，效果遠勝任何盤中停損機制。

**3. 09:15檢查點(觀察用，不是自動出場鍵)**
| 09:15狀態 | n | 勝率 | 平均報酬 |
|---|---|---|---|
| 仍低於開盤 | 1029 | 78.3% | +3.50%(t=23) |
| 已收上開盤 | 949 | 33.6% | -1.88%(t=-11) |

⚠️ 實測過「9:15一收回開盤就無條件出場」反而讓報酬變差(甜蜜點組+1.577%→+0.727%)，因為當下的價格常常只是暫時衝高、之後會回落。9:15應該當「訊號強弱參考」，不是機械出場鍵。

**4. lock_streak：跟FlipBranch規律相反！**
| lock_streak | 勝率 |
|---|---|
| =1 | 61.0% |
| ≥2 | 64.2% |
| ≥3 | 73.7% |

城中GA連續鎖漲停天數越多、隔日勝率越高——跟FlipBranch「連鎖越多越差」剛好相反，不要把兩策略的濾網邏輯互相套用。

**5. 盤中停損/停利：測過沒有用**
2%~8%停損、2%~7%停利全部測過，**沒有一組贏過「不設限、單純持有到收盤」**。真正的尾端風險是D1鎖死(上面第2點)，不是盤中價格波動，停損停利機制對此無能為力。
""")

    with col2:
        st.markdown("""
### ✅ 已驗證：承接／低接與做多方向

**1. D1扛住賣壓 → D2收盤偏強(跨TRAIN/TEST顯著)**
若D1收盤不低於D0收盤(=城中賣壓被承接)：
- TRAIN：D1收→D2收 +1.72%(p=0.02)
- TEST：D1收→D2收 +0.70%(p=0.0035)

持股遇到城中賣壓但D1收平/收紅，資料支持**不用因此恐慌減碼**。但不保證D2盤中不會被砸，兩者可能同時成立。

**2. D1承接強度 → D2再度鎖漲停機率(單調遞增)**
| D1承接強度 | D2再鎖機率 |
|---|---|
| 弱承接(0-6%) | 7.5% |
| 中承接(6-9.9%) | 19.0% |
| 強承接(9.9-10%) | 22.3% |
| **極強承接(前10%)** | **25.0%** |

實例：8046南電 07-31鎖漲停→08-03城中賣17萬股但收漲3.6%(弱承接)→08-04再衝高至1045近漲停(+9.7%)。

**3. ⚠️但「D2開盤買進」做多策略判死**
承接的漲幅主要發生在**隔夜跳空**(D1收→D2開，TEST均值+1.0%)，等你D2開盤才進場，這段漲幅已經被吃掉，剩下的**盤中段(D2開→D2收)反而是負的(-1.17%)**。承接現象是真的，但用「隔日開盤買進」的方式接不到這個edge。

**4. 台新台北 + 凱基城中同時進D0前3大買方**
n=**僅6筆**：勝率83.3%、平均+3.33%(比一般情況的56.6%/+0.85%更好)。**方向支持「兩者同時出現時開盤空的把握度較高」，但樣本量嚴重不足，不能當規則使用**，需持續累積案例。
""")

    st.markdown('---')
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
### 🏆 分點層級：獨立驗證結果

| 分點 | TRAIN | TEST | 判斷 |
|---|---|---|---|
| 凱基-城中(既有) | t=1.36(弱) | t=6.46 | 可用(既有策略) |
| **富邦證券** | t=6.14 | t=5.86 | **可用，最扎實** |
| 統一-城中 | t=3.25 | t=7.44 | 可用 |
| **富邦-嘉義** | t=5.09 | t=5.32 | **可用，新發現** |
| 台新-台北(接元富) | t=3.34 | t=1.33(不顯著) | 判死 |
| 凱基台北 | t=4.23 | t=1.96(邊緣) | 效果太弱不算真發現 |

**⭐重大身分破解**：「台新-台北」＝2026-04-06併購前的「元富」(地址逐字比對驗證)。接上完整歷史後才有真正TRAIN樣本，結果TEST期不顯著，判死。

**嘉義/虎尾地域聚合：判死**，整個地域加總訊號比富邦-嘉義單一分點還弱，且拆開「富邦-嘉義有無參與」發現訊號完全由這一家分點主導(有參與:+0.58%~+0.83%顯著；沒參與:反而顯著負-0.41%~-0.87%)，證實不是地域性現象，是單一分點的個別能力。
""")
    with col4:
        st.markdown("""
### ❌ 已測試、判死或需更多資料的方向

- 是否首次買這檔股票(首次組表現略好但樣本不足n=14/13)
- 連續買超天數(不要求鎖漲停版本，樣本不足或方向不穩)
- 分點共現(265高flip分點)：**負向訊號**，越多知名分點同時買超，隔日放空表現反而越差(獨大組TEST成功率78% vs 共現組54%)——「多方合力」代表真買盤，不是多方倒貨
- ML(LightGBM+SHAP)+GA系統性因子挖掘：TEST預測分數與實際報酬相關係數僅-0.012(等於雜訊)，系統性挖掘後現行兩門檻已是資料能榨出的極限
- 鎖漲停時間點因子(開盤多快鎖死)：僅30筆分鐘K可配對、無TRAIN樣本，資料不足以驗證(不是判死，是擱置)
- 盤中量能爆量訊號：★方法論陷阱，用「當天最終總量」當分母是偷看未來；改用「9:15量/D0全天量」重測後，訊號幾乎消失(≥60%那組平均仍是+1.46%正值)

### 📌 今日(08-04)即時案例對照

6檔候選(2301/3443/2454/2345/2327/6213)實際結果：2454表現最佳(+5.62%，無負向旗標)；2301最差(-7.92%，有共現旗標**且**gap僅1.74%在甜蜜點內卻被外資巨量買盤(台灣摩根/摩根大通等百萬股級)完全蓋過──**證實即使規則都對，仍有真實虧損可能，這是策略固有的殘餘風險，非操作失誤**。價位vs分鐘K時間反推發現：城中與台新台北在2301於同一分鐘(09:07-09:08)一起從賣轉買，顯示兩者操作技法/時間窗高度相似。
""")

    st.markdown('---')
    st.caption('研究過程與完整方法論見 D:\\stock 下 memory 系統 project_隔日沖分點戰術.md 第66節，以及各 CODEX_*.md 報告。本頁為2026-08-04單日整理，未來新發現需手動更新此頁。')
