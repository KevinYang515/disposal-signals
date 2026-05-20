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

tab1, tab2, tab3, tab4, tab5 = st.tabs(['🔔 今日訊號', '📜 歷史回測紀錄', '📊 進場網格回測', '📖 策略說明', '⚙️ 使用方式'])

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
        filter_cap = st.multiselect('規模', options=['大', '中'], default=['大', '中'], key='tab1_cap')

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

    display_cols = ['評級', '代號', '名稱', '規模', '處置原因', '近20日漲幅', '大戶(%)',
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

    styled = disp.style.map(color_grade, subset=['評級'])
    color_cols = d_cols + (['今日漲跌'] if '今日漲跌' in disp.columns else [])
    if color_cols:
        styled = styled.map(color_ret_str, subset=color_cols)

    st.dataframe(styled, use_container_width=True, height=500)

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
# TAB 2：歷史回測紀錄
# ════════════════════════════════════════════════════════
with tab2:
    hist = load_history()

    if hist.empty:
        st.warning('尚無歷史資料，請先執行 update_signals.py')
    else:
        st.subheader('漲多處置 × 大+中型 × 20分鐘　歷史交易紀錄')

        # ── 確保規模欄格式一致（舊版 CSV 可能是'大型股'）──
        if '規模' in hist.columns:
            hist['規模'] = hist['規模'].apply(lambda v: '大' if '大' in str(v) else '中')

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

        # ── 篩選器 ──
        hist['年份'] = hist['起始日'].str[:4]
        DN_OPTIONS = ['Dn < -5%', 'Dn -5%~0%', 'Dn ≥ 0%', '全部漲多']

        col_f1, col_f2, col_f3 = st.columns([2, 5, 2])
        with col_f1:
            sel_year = st.selectbox('年份', ['全部'] + sorted(hist['年份'].unique().tolist(), reverse=True))
        with col_f2:
            sel_dn = st.multiselect(
                'Dn最深跌幅條件（D1~D8 任意最大跌幅）',
                options=DN_OPTIONS,
                default=['Dn < -5%'],
            )
        with col_f3:
            sel_cap = st.multiselect('規模', ['大', '中'], default=['大', '中'], key='tab2_cap')

        view_h = hist.copy()
        if sel_year != '全部':
            view_h = view_h[view_h['年份'] == sel_year]
        if sel_dn and '全部漲多' not in sel_dn:
            view_h = view_h[view_h['Dn組別'].isin(sel_dn)]
        if sel_cap:
            view_h = view_h[view_h['規模'].isin(sel_cap)]

        # ── KPI（依篩選結果動態更新）──
        if len(view_h) == 0:
            st.warning('目前篩選條件無資料')
        else:
            wins  = view_h[view_h['結果'].str.startswith('✅')]
            loss  = view_h[view_h['結果'].str.startswith('❌')]
            wr    = len(wins) / len(view_h) * 100
            avg_r = view_h['出關報酬(%)'].mean()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('總筆數', len(view_h))
            c2.metric('勝率', f'{wr:.1f}%')
            c3.metric('期望報酬', f'{avg_r:+.2f}%')
            c4.metric('獲利筆', len(wins))
            c5.metric('虧損筆', len(loss))

        st.divider()

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
                        {'勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}'}),
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
            ret_cols = ['近20日漲幅', '買進時累積(%)', '期間最深(%)', '大戶(%)', '出關報酬(%)', 'T+1收盤(%)', 'T+2收盤(%)', 'T+3收盤(%)']
            # 最深日 是字串欄位，不需格式化
            for c in ret_cols:
                if c in disp.columns:
                    disp[c] = disp[c].apply(fmt_pct)

            exit_cols = [c for c in ['出關報酬(%)', 'T+1收盤(%)', 'T+2收盤(%)', 'T+3收盤(%)'] if c in disp.columns]
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
# TAB 3：進場網格回測
# ════════════════════════════════════════════════════════
with tab3:
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
            st.dataframe(show.style.format({'勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}', '賺賠比': '{:.2f}'}, na_rep='-'), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4：策略說明
# ════════════════════════════════════════════════════════
with tab4:
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
# TAB 5：使用方式
# ════════════════════════════════════════════════════════
with tab5:
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
