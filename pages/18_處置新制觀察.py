# -*- coding: utf-8 -*-
"""2026-08-10 處置新制(撮合統一改2分鐘、期間縮為5或7個營業日)：獨立觀察頁面。

完全獨立於舊制頁面（首頁今日訊號、page8/15/16/17 等）——資料來源
(data/newregime_signals.csv、data/newregime_history.csv)、評級與統計
都是這個頁面專用的，不會反過來影響舊制頁面的任何數字。

新制下第一次與第二次(含)以上處置撮合頻率相同(都是2分鐘)，但收款門檻文字
逐字比對後跟舊制5分鐘(第一次)/20分鐘(第二次+)完全相同，只有撮合頻率變快，
所以本頁比照舊站「5分鐘/20分鐘」的分法，拆成「第一次(對照舊5分)」與
「第二次+(對照舊20分)」兩組獨立呈現，避免混在一起看不出差異。
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='處置新制觀察', page_icon='🆕', layout='wide')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def sharpe_pf(s):
    """(夏普值, 賺賠比)，定義與 app.py／page16 的 sharpe_pf() 完全一致。"""
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan, np.nan
    sharpe = s.mean() / s.std() if s.std() > 0 else np.nan
    wins = s[s > 0]
    loss = s[s <= 0]
    pf = wins.mean() / abs(loss.mean()) if len(loss) > 0 and loss.mean() != 0 else np.nan
    return sharpe, pf


def safe_read_csv(path, **kwargs):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


GRADE_COLOR = {
    '✅ 主力訊號':       '#26c281',
    '⚠️ 漲多但大戶減碼': '#f6c90e',
    '🟡 觀察中':         '#f39c12',
    '⬜ 待觀察':         '#95a5a6',
    '❌ 避開':           '#e74c3c',
}


def color_grade(val):
    color = GRADE_COLOR.get(str(val), '')
    return f'color: {color}; font-weight: bold;' if color else ''


st.title('🆕 處置新制觀察（2026-08-10起，2分鐘撮合）')
st.warning(
    '⚠️ **本頁資料尚未驗證，評級只是沿用舊制的分類方式，不代表已證實有效**。'
    '新制撮合頻率(2分鐘)比舊制快很多(舊5分鐘→2分鐘、舊20分鐘→2分鐘)，處置期間也縮短為5(或當沖加重7)個營業日，'
    '核心「橡皮筋壓縮」機制是否仍然成立需要更多完整出關的樣本才能判斷。'
    '本頁完全獨立於首頁與其他舊制頁面的資料與統計，觀察本頁不會改變舊制頁面任何數字。',
    icon='⚠️',
)

sig = safe_read_csv(f'{DATA_DIR}/newregime_signals.csv', dtype={'代號': str})
hist = safe_read_csv(f'{DATA_DIR}/newregime_history.csv', dtype={'代號': str})

st.caption(
    '分組依據：新制下第一次與第二次(含)以上處置的「收款門檻」文字跟舊制5分鐘(第一次)/20分鐘(第二次+)逐字相同，'
    '只有撮合頻率變快——第一次仍是單筆10張或累積30張以上才全額收款，第二次+是所有投資人全額收款。'
    '因此比照舊站的「5分鐘/20分鐘」分法，拆成以下兩組獨立呈現。'
)

TAB_DEFS = [
    ('第一次', '📗 第一次處置（對照舊5分鐘）'),
    ('第二次+', '📕 第二次(含)以上處置（對照舊20分鐘）'),
]

tabs = st.tabs([label for _, label in TAB_DEFS])

for (key, label), tab in zip(TAB_DEFS, tabs):
    with tab:
        sub_sig = sig[sig['處置次別'] == key] if len(sig) else sig
        sub_hist = hist[hist['處置次別'] == key] if len(hist) else hist

        st.subheader(f'🔔 今日訊號（{key}）')
        if sub_sig.empty:
            st.info('目前沒有符合條件的處置中股票。')
        else:
            display_cols = ['評級', '買進訊號', '代號', '名稱', '規模', '處置原因', '近20日漲幅', '大戶(%)',
                             '起始日', '今D幾', '出關日', '觸發價', '距觸發(%)', '目前損益(%)', '今日漲跌']
            d_cols = [c for c in [f'D{n}%' for n in range(1, 9)]
                      if c in sub_sig.columns and sub_sig[c].notna().any()]
            display_cols = [c for c in display_cols + d_cols if c in sub_sig.columns]
            disp = sub_sig[display_cols].copy()
            for c in d_cols + ['近20日漲幅', '大戶(%)', '今日漲跌', '目前損益(%)', '距觸發(%)']:
                if c in disp.columns:
                    disp[c] = disp[c].apply(lambda v: f'{v:+.2f}%' if pd.notna(v) else '-')
            if '觸發價' in disp.columns:
                disp['觸發價'] = disp['觸發價'].apply(lambda v: f'{v:.2f}' if pd.notna(v) else '-')
            styled = disp.style.map(color_grade, subset=['評級']) if '評級' in disp.columns else disp
            st.dataframe(styled, use_container_width=True, height=min(400, 60 + 35 * len(disp)))

            # ── 進場預覽 / 觸發分析（依缺口排序）── 跟首頁Tab1同樣的邏輯，只是資料源換成新制 ──
            if '距觸發(%)' in sub_sig.columns and '觸發價' in sub_sig.columns:
                with st.expander(f'📍 進場預覽 / 觸發分析（{key}，依缺口排序）', expanded=True):
                    prev_base = sub_sig[sub_sig['處置原因'] == '漲多處置'].copy()
                    if prev_base.empty:
                        st.info('目前沒有漲多處置的股票可預覽。')
                    else:
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
                        prev_cols = [c for c in ['代號', '名稱', '評級', '買進訊號', '今D幾', '出關日',
                                                 '今日漲跌', '目前損益(%)', '觸發價', '距觸發(%)', '明日預覽']
                                     if c in prev_base.columns]
                        prev_show = prev_base[prev_cols].copy()
                        for c in ['今日漲跌', '目前損益(%)', '距觸發(%)']:
                            if c in prev_show.columns:
                                prev_show[c] = prev_show[c].apply(lambda v: f'{v:+.2f}%' if pd.notna(v) else '-')
                        if '觸發價' in prev_show.columns:
                            prev_show['觸發價'] = prev_show['觸發價'].apply(lambda v: f'{v:.2f}' if pd.notna(v) else '-')

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
                            except Exception:
                                return ''

                        styled_prev = prev_show.style.map(color_preview_col, subset=['明日預覽'])
                        if '距觸發(%)' in prev_show.columns:
                            styled_prev = styled_prev.map(color_gap_col, subset=['距觸發(%)'])
                        if '評級' in prev_show.columns:
                            styled_prev = styled_prev.map(color_grade, subset=['評級'])
                        st.dataframe(styled_prev, use_container_width=True)
                        st.caption(
                            f'已觸發 = 距觸發 ≥ 0%（目前收盤已低於觸發價）｜🔥 = 距觸發 < 2%（高度警戒）｜'
                            f'觸發條件：D3~D5 任意天累積跌幅 < -5%（{key}，比照舊制邏輯但窗口縮到5天近似）'
                        )

        st.divider()
        st.subheader(f'📜 累積至今結果（{key}，已完整出關者）')
        if sub_hist.empty:
            st.info('目前還沒有已完整出關的樣本。')
        else:
            settled = sub_hist[sub_hist['出關報酬(%)'].notna()]
            if settled.empty:
                st.info(f'{key} 目前有 {len(sub_hist)} 筆處置事件，但尚無已完整出關並可計算報酬的樣本（處置期間本來就短，還在陸續出關中）。')
            else:
                rets = settled['出關報酬(%)']
                triggered = settled[settled['買進日'] != '-']
                trig_rets = triggered['出關報酬(%)']
                sharpe, pf = sharpe_pf(trig_rets) if len(trig_rets) else (np.nan, np.nan)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric('已出關筆數', len(settled), delta=f'{len(triggered)} 筆有觸發D3~D8<-5%進場訊號', delta_color='off')
                c2.metric('觸發後勝率', f'{(trig_rets > 0).mean() * 100:.1f}%' if len(trig_rets) else '-')
                c3.metric('觸發後期望報酬', f'{trig_rets.mean():+.2f}%' if len(trig_rets) else '-')
                c4.metric('夏普值', f'{sharpe:.2f}' if pd.notna(sharpe) else '-')
                c5.metric('賺賠比', f'{pf:.2f}' if pd.notna(pf) else '-')
                st.caption(
                    f'全部{key}已出關（不論是否觸發-5%進場條件）平均報酬：{rets.mean():+.2f}%，勝率 {(rets > 0).mean()*100:.1f}%'
                    f'（n={len(settled)}）。樣本數還很小，以上數字僅供觀察趨勢，不是可信賴的統計結果。'
                )
                show_cols = ['起始日', '出關日', '代號', '名稱', '規模', 'Dn組別', '買進日',
                             '買進時累積(%)', '最深日', '期間最深(%)', '出關報酬(%)', '結果']
                show_cols = [c for c in show_cols if c in settled.columns]
                st.dataframe(settled[show_cols].sort_values('起始日', ascending=False),
                             use_container_width=True, height=min(400, 60 + 35 * len(settled)))

st.divider()
with st.expander('📖 本頁的已知近似與侷限（務必先讀再解讀數字）'):
    st.markdown("""
- **5天 vs 7天不分**：新制處置期間一般是5個營業日，若同時因當沖比重過高遭加重處置則是7個營業日；本頁資料沒有欄位可以精確分辨，統一用5天近似，可能讓少數7天案例的「今D幾」「出關日」顯示提前。
- **進場窗口沿用舊制D3~D8**：這是直接套用舊制邏輯的假設，新制期間縮短後這個窗口是否還合理，尚未驗證。
- **評級只是描述性分類**：套用跟舊制頁面相同的規則(grade())來標「✅主力訊號/🟡觀察中」等，純粹方便閱讀，**不代表這些評級在新制下已被證實有效**。
- **樣本數極少**：新制上路才幾天，已完整出關的樣本數只有個位數到十幾筆，任何統計數字都可能只是巧合，需要更長時間累積才能判斷。
- 完整規則變更細節與研究進度，見 PLAYBOOK.md「研究 Roadmap」章節。
""")
