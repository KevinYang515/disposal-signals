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
st.caption(f"資料更新：{meta.get('updated_at', '-')}　｜　策略：漲多處置 × 大+中型 × 20分鐘撮合 × D3累積跌幅 < -5%")

tab1, tab2, tab3, tab4 = st.tabs(['🔔 今日訊號', '📊 進場網格回測', '📖 策略說明', '⚙️ 使用方式'])

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
        filter_cap = st.multiselect('規模', options=['大', '中'], default=['大', '中'])

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

    display_cols = ['評級', '代號', '名稱', '規模', '處置原因', 'prerun', '大戶(%)',
                    '起始日', '今D幾', '出關日'] + d_cols

    display_cols = [c for c in display_cols if c in view.columns]

    def color_grade(val):
        color = GRADE_COLOR.get(str(val), '')
        return f'color: {color}; font-weight: bold;' if color else ''

    def color_ret(val):
        try:
            v = float(val)
            if v < -10: return 'color: #e74c3c; font-weight:700'
            if v < -5:  return 'color: #e67e22; font-weight:700'
            if v > 5:   return 'color: #26c281'
        except:
            pass
        return ''

    styled = (
        view[display_cols]
        .style
        .map(color_grade, subset=['評級'])
        .map(color_ret,   subset=d_cols)
        .format({c: '{:+.1f}%' for c in d_cols if c in view.columns}, na_rep='-')
        .format({'prerun': '{:+.0f}%', '大戶(%)': '{:+.2f}%'}, na_rep='-')
    )

    st.dataframe(styled, use_container_width=True, height=500)

    # 說明
    st.markdown("""
**評級說明：**
✅ 主力訊號 = 漲多處置 + D3累積 < -5%（歷史勝率 85%+）
⚠️ 漲多但大戶減碼 > -1.5%，謹慎
🟡 觀察中 = D3 尚未達 -5%，等後續發展
❌ 避開 = 跌深處置 + 進場前已在跌（prerun < 0）
""")

    # 進場時序提醒
    with st.expander('📅 進場時序說明'):
        st.markdown("""
| 時間點 | 動作 |
|---|---|
| 每日收盤後 | 看今日是第幾天（D幾），確認累積跌幅 |
| **D3 收盤 < -5%** | ✅ **當天收盤買進**（主力進場點） |
| D3 收盤 -3%~-5% | 等 D4，若 D4 < -5% 再買 |
| D5/D6 < -15% | 橡皮筋極度壓縮，仍可考慮補進 |
| T+1 出關日 開盤 | 賣出（處置結束後第一個交易日） |
""")

# ════════════════════════════════════════════════════════
# TAB 2：進場網格回測
# ════════════════════════════════════════════════════════
with tab2:
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

        fmt = '{:.1f}%' if metric == '勝率(%)' else '{:+.1f}%'
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
# TAB 3：策略說明
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader('📖 橡皮筋效應策略說明')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 核心機制

**20分鐘撮合制度**（第二次處置）每 20 分鐘才撮合一次委託，
導致急著賣出的人只能接受極差的價格，造成**人工賣壓 → 股價被壓低**。

```
漲多進處置（prerun > 0）
    ↓
20分撮合 → 窒息量 + 股價人工壓低
    ↓
D3/D4 累積跌幅 > 5% = 橡皮筋壓縮
    ↓
出關後 → 人工壓力消失 → 橡皮筋彈回
    ↓
我們賺的是「壓力解除」那段反彈
```

### 為什麼只看漲多處置？

- **漲多處置**：股票先漲再被打壓 → 底層有支撐 → 橡皮筋有力
- **跌深處置**：本來就在跌 → 真實賣壓 → 出關後繼續跌
""")

    with col2:
        st.markdown("""
### 回測統計（漲多 × 大+中 × 20分）

| 進場條件 | 勝率 | 期望報酬 |
|---|---|---|
| 無過濾（任意進場）| 75% | +9% |
| D3 < -5% | **85%** | **+15%** |
| D3 < -10% | **91%** | **+19%** |
| D5 < -15% | **100%** | **+18%** |
| D6 < -15% | **93%** | **+19%** |

### 輔助判斷

| 因子 | 意義 |
|---|---|
| prerun | 進場前20日漲幅，越高橡皮筋越強 |
| 大戶(%) | 大戶持股變動，< -1.5% 要謹慎 |
| 處置原因 | 漲多 > 震盪 > 跌深 |

### 大盤不影響策略

實測三種大盤環境（強/中/弱）勝率均在 75-90%，
橡皮筋效應是**制度性**的，不依賴市場行情。
""")

# ════════════════════════════════════════════════════════
# TAB 4：使用方式
# ════════════════════════════════════════════════════════
with tab4:
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
2. 找 **✅ 主力訊號**（漲多 + D3 < -5%）
3. 確認**今D幾**和 **D3%**
   - D3% < -10% → 加重部位
   - D3% -5%~-10% → 標準部位
4. **出關日開盤賣出**，不要提前跑
5. 注意**大戶(%)** < -1.5% 的標的降低部位或跳過

---

### 風險提示

- 歷史樣本 84% 集中於 2026 年，策略仍在驗證期
- 建議每筆部位控制在總資金 **2-3%**
- 跌深處置（❌）標的避免進場
""")
