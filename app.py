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
    sharpe = round(s.mean() / s.std(), 3) if s.std() > 0 else np.nan
    wins = s[s > 0]; loss = s[s <= 0]
    pf = round(wins.mean() / abs(loss.mean()), 3) if len(loss) > 0 and loss.mean() != 0 else np.nan
    return sharpe, pf

tab_lab, tab1, tab2, tab3, tab4, tab5, tab_exit, tab6 = st.tabs(['🧪 策略研發', '🔔 今日訊號', '📜 歷史回測紀錄', '🔬 自訂策略', '📊 進場網格回測', '📖 策略說明', '📤 持有出清時機', '⚙️ 使用方式'])

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
                    return f'🔥 再跌 {abs(gap):.1f}%'
                elif gap >= -5:
                    return f'🟡 再跌 {abs(gap):.1f}%'
                else:
                    return f'⬜ 再跌 {abs(gap):.1f}%'

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
                    lambda v: f'{v:.1f}' if pd.notna(v) else '-')

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
            c2.metric('勝率', f'{wr:.1f}%')
            c3.metric('期望報酬', f'{avg_r:+.2f}%' if pd.notna(avg_r) else '-')
            c4.metric('夏普值', f'{sharpe:.3f}' if pd.notna(sharpe) else '-')
            c5.metric('賺賠比', f'{pf:.3f}' if pd.notna(pf) else '-')
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
                    '勝率(%)':     round(ew / (ew + el) * 100, 1) if (ew + el) > 0 else 0,
                    '期望報酬(%)':  round(es.mean(), 2),
                    '夏普值':      sp,
                    '賺賠比':      pf_et,
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
                        .format({'勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}',
                                 '夏普值': '{:.3f}', '賺賠比': '{:.3f}',
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
            ret_cols = (['近20日漲幅', '買進時累積(%)', '期間最深(%)', '大戶(%)', '出關報酬(%)'] +
                        [f'T+{k}收盤(%)' for k in range(1, 11)])
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
                kc2.metric('勝率', f'{wr_c:.1f}%')
                kc3.metric('期望報酬', f'{avg_r_c:+.2f}%' if pd.notna(avg_r_c) else '-')
                kc4.metric('夏普值', f'{sharpe_c:.3f}' if pd.notna(sharpe_c) else '-')
                kc5.metric('賺賠比', f'{pf_c:.3f}' if pd.notna(pf_c) else '-')
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
                                       '勝率(%)': round((s > 0).mean() * 100, 1),
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
                            .format({'勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}',
                                     '夏普值': '{:.3f}', '賺賠比': '{:.3f}',
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
                            '勝率(%)':   round(ew_e / (ew_e + el_e) * 100, 1) if (ew_e + el_e) > 0 else 0,
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
                        '勝率(%)':   round(ew_e / (ew_e + el_e) * 100, 1) if (ew_e + el_e) > 0 else 0,
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
                            .format({'勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}',
                                     '夏普值': '{:.3f}', '賺賠比': '{:.3f}',
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
                                '上漲率(%)': round((s > 0).mean() * 100, 1),
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
                                             '上漲率(%)': '{:.1f}',
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
                        tp1.metric('觸發停利', f'{hit_n} 筆 ({hit_n/total_n*100:.0f}%)')
                        tp2.metric('觸發者期望報酬', f'{np.mean(hit_rets):+.2f}%' if hit_rets else '-')
                        tp3.metric('未觸發(持至D5)均', f'{np.nanmean(miss_rets):+.2f}%' if miss_rets else '-')
                        tp4.metric('整體期望報酬', f'{np.mean(all_tp):+.2f}%' if all_tp else '-',
                                   delta='vs T+1開盤直出 0%')
                        tp5.metric('夏普值', f'{sp_tp:.3f}' if pd.notna(sp_tp) else '-')
                        tp6.metric('賺賠比', f'{pf_tp:.3f}' if pd.notna(pf_tp) else '-')

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
            st.dataframe(show.style.format({'勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}', '賺賠比': '{:.2f}'}, na_rep='-'), use_container_width=True)

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
                        '勝率(%)': round((s > 0).mean() * 100, 1),
                        '期望報酬(%)': round(s.mean(), 2),
                        '均獲利(%)': round(s[s > 0].mean(), 2) if (s > 0).any() else 0,
                        '均虧損(%)': round(s[s < 0].mean(), 2) if (s < 0).any() else 0,
                    })
                if post_rows:
                    pdf = pd.DataFrame(post_rows).set_index('持有期')
                    st.dataframe(pdf.style.format({
                        '勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}',
                        '均獲利(%)': '{:+.2f}', '均虧損(%)': '{:+.2f}'
                    }, na_rep='-'), use_container_width=True)

            with lc2:
                st.markdown('**延長持有總報酬（原策略 + 繼續持有 N 天）**')
                ext_rows = [{'出場時機': 'T+1 開盤（原策略）',
                              '樣本N': len(orig_s),
                              '勝率(%)': round((orig_s > 0).mean() * 100, 1),
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
                                     '勝率(%)': round((total > 0).mean() * 100, 1),
                                     '期望報酬(%)': round(total.mean(), 2)})
                edf = pd.DataFrame(ext_rows).set_index('出場時機')
                st.dataframe(edf.style.format({
                    '勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}'
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
            st.caption(f'樣本中位數：{lab1_base["_rise"].median():.0f}%（多數股票處置前已大漲）')
            rise_rows = [{'漲幅門檻': '不篩選（全部）',
                          '樣本N': len(lab1_base['_ret'].dropna()),
                          '勝率(%)': round((lab1_base['_ret'].dropna() > 0).mean() * 100, 1),
                          '期望報酬(%)': round(lab1_base['_ret'].dropna().mean(), 2)}]
            for th in [20, 30, 40, 50, 70, 100]:
                sub = lab1_base[lab1_base['_rise'] >= th]['_ret'].dropna()
                if len(sub) < 5: continue
                rise_rows.append({'漲幅門檻': f'>= {th}%', '樣本N': len(sub),
                                  '勝率(%)': round((sub > 0).mean() * 100, 1),
                                  '期望報酬(%)': round(sub.mean(), 2)})
            rdf = pd.DataFrame(rise_rows).set_index('漲幅門檻')
            st.dataframe(rdf.style.format({
                '勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}'
            }, na_rep='-'), use_container_width=True)
            st.caption('💡 近20日漲幅篩選對結果影響有限，不建議作為主要濾網。')

        # ── 大戶持股變化篩選 ──
        with l1b:
            st.markdown('**大戶持股變化篩選**')
            st.caption('大戶(%)為進場時大戶持股變化；正值=大戶增持，負值=大戶減持')
            whale_rows = [{'大戶條件': '不篩選（全部）',
                           '樣本N': len(lab1_base['_ret'].dropna()),
                           '勝率(%)': round((lab1_base['_ret'].dropna() > 0).mean() * 100, 1),
                           '期望報酬(%)': round(lab1_base['_ret'].dropna().mean(), 2)}]
            # 大戶減持（負值，多數情況）
            for th in [-1, -2, -3, -5]:
                sub = lab1_base[lab1_base['_whale'] <= th]['_ret'].dropna()
                if len(sub) < 5: continue
                whale_rows.append({'大戶條件': f'<= {th}%（減持）', '樣本N': len(sub),
                                   '勝率(%)': round((sub > 0).mean() * 100, 1),
                                   '期望報酬(%)': round(sub.mean(), 2)})
            # 大戶增持（正值，較少）
            for th in [0, 1]:
                sub = lab1_base[lab1_base['_whale'] >= th]['_ret'].dropna()
                if len(sub) < 5: continue
                whale_rows.append({'大戶條件': f'>= {th}%（增持/中性）', '樣本N': len(sub),
                                   '勝率(%)': round((sub > 0).mean() * 100, 1),
                                   '期望報酬(%)': round(sub.mean(), 2)})
            wdf = pd.DataFrame(whale_rows).set_index('大戶條件')
            st.dataframe(wdf.style.format({
                '勝率(%)': '{:.1f}', '期望報酬(%)': '{:+.2f}'
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
                    row[f'{day} 勝率'] = round((s > 0).mean() * 100, 1) if len(s) >= 3 else np.nan
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

                fmt = {c: '{:.1f}' for c in wr_cols}
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
                return df.style.applymap(color_mean, subset=['平均報酬(%)','中位數(%)'])

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
                return df.style.applymap(color_d3, subset=['D3均值(%)'])
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
