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


def round_to_tick(price):
    """台股法定升降單位，自選窗口重算的價格是用百分比反推(D0收盤×(1+報酬%))，
    直接四捨五入到小數點兩位會出現528.02這種不存在的價格，要對齊到合法跳動單位。
    （固定預設窗口(D1~D4)的欄位不受影響，那是直接讀真實收盤價，本來就是合法跳動價。）"""
    if price is None or pd.isna(price) or price <= 0:
        return price
    if price < 10:
        tick = 0.01
    elif price < 50:
        tick = 0.05
    elif price < 100:
        tick = 0.1
    elif price < 500:
        tick = 0.5
    elif price < 1000:
        tick = 1
    else:
        tick = 5
    return round(round(price / tick) * tick, 2)


def color_signal(val):
    s = str(val)
    if '🟢' in s: return 'color:#26c281;font-weight:700'
    if '🔵' in s: return 'color:#2980b9;font-weight:700'
    if '🟡' in s: return 'color:#f6c90e'
    if '🔒' in s or '❌' in s: return 'color:#e74c3c'
    return 'color:#95a5a6'


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
old_hist = safe_read_csv(f'{DATA_DIR}/history.csv', dtype={'代號': str})  # 舊制基準，用來對照新制是否走勢相似

OLD_TYPE_MAP = {'第一次': '5分鐘', '第二次+': '20分鐘'}

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
        sub_sig = (sig[sig['處置次別'] == key] if len(sig) else sig).copy()
        sub_hist = (hist[hist['處置次別'] == key] if len(hist) else hist).copy()

        use_alt = False
        if key == '第一次':
            entry_mode = st.radio(
                '第一次處置的進場邏輯',
                ['5分盤動能（建議，已驗證）', '-5%回檔（套用第二次+規則，僅供比較，第一次未驗證過此規則）'],
                horizontal=True, key=f'entry_mode_{key}',
            )
            use_alt = entry_mode.startswith('-5%')
            if use_alt:
                st.info('目前顯示的是「若第一次也套用第二次+的-5%回檔規則」會怎樣，純供比較，不是建議規則——第一次已驗證的規則是5分盤動能。', icon='ℹ️')
            for base_col, alt_col in [
                ('買進訊號', '買進訊號(-5%版)'), ('觸發價', '觸發價(-5%版)'), ('距觸發(%)', '距觸發(-5%版)(%)'),
                ('目前損益(%)', '目前損益(-5%版)(%)'),
            ]:
                if use_alt and alt_col in sub_sig.columns:
                    sub_sig[base_col] = sub_sig[alt_col]
            for base_col, alt_col in [
                ('買進日', '買進日(-5%版)'), ('買進時累積(%)', '買進時累積(-5%版)(%)'),
                ('出關報酬(%)', '出關報酬(-5%版)(%)'), ('結果', '結果(-5%版)'),
            ]:
                if use_alt and alt_col in sub_hist.columns:
                    sub_hist[base_col] = sub_hist[alt_col]

        # 第二次+：可自選進場窗口（D幾~D幾），今日訊號跟累積至今結果都會跟著重算。
        # 2026-08-30 起預設改為D1~D4（原本D3~D5）：disposal_entry_day_full_test研究
        # 發現D1單日表現目前為止最好，且D3~D5幾乎都是D1早就觸發過的同一批股票，改用
        # 比較早、比較便宜的進場點；D1~D4排除D5是因為D5獨立測試時明顯最弱，跟D1~D5
        # 在目前資料下實測完全相同(D5從未單獨觸發過)。**這是Kevin在樣本仍很小的情況下
        # 主動決定要換的，不是本系列一貫的「驗證足夠才換」標準，務必在畫面上誠實揭露
        # 這一點，不要包裝成已驗證的規則。**
        if key == '第二次+':
            # 2026-09-02：觸發基準預設改用「處置前5日高點」(峰值版)，理由見
            # disposal_peak3_reference_test研究(單筆品質不變、訊號量多25%、資金限制
            # 模擬N=3/5/10六種情境CAGR全勝)。Kevin要求把舊的D0收盤基準留著當對照，
            # 不要直接刪掉，用切換選項讓兩邊都能觀察。
            basis_mode = st.radio(
                '第二次+的-5%回檔基準',
                ['峰值基準（處置前5日高點，預設，2026-09-02起）', 'D0收盤基準（舊版本，僅供對照觀察）'],
                horizontal=True, key=f'basis_mode_{key}',
            )
            use_old_basis = basis_mode.startswith('D0收盤基準')
            low_suffix = '' if use_old_basis else '(峰值版)'
            if use_old_basis:
                st.info('目前顯示的是舊版「D0收盤」基準，僅供跟峰值版對照——網站正式預設已改用峰值基準，理由見PLAYBOOK.md與disposal_peak3_reference_test研究。', icon='ℹ️')
                for base_col, alt_col in [('觸發價', '觸發價(D0收盤版)'), ('距觸發(%)', '距觸發(D0收盤版)(%)')]:
                    if alt_col in sub_sig.columns:
                        sub_sig[base_col] = sub_sig[alt_col]

            wc1, wc2 = st.columns(2)
            with wc1:
                win_start = st.number_input('進場窗口起始 D', min_value=1, max_value=5, value=1, step=1, key=f'win_start_{key}')
            with wc2:
                win_end = st.number_input('進場窗口結束 D', min_value=1, max_value=5, value=4, step=1, key=f'win_end_{key}')
            if win_start > win_end:
                win_start, win_end = win_end, win_start
                st.warning('起始日不能大於結束日，已自動對調。')
            st.caption('⚠️ 目前預設是D1~D4，2026-08-30由Kevin主動決定換掉原本的D3~D5——樣本量還很小(D1目前只有13筆已結算)、沒有訓練/驗證期切分，不是這系列一貫要求的驗證標準，屬於觀察中的規則，非正式驗證結論。')
            if (win_start, win_end) != (1, 4):
                st.info(
                    f'目前顯示的是自選窗口 D{win_start}~D{win_end}，不是網站預設的D1~D4——'
                    f'僅供比較觀察。詳見PLAYBOOK.md「開放問題」與'
                    f'disposal_entry_day_full_test研究（新制樣本仍很小，任何窗口的統計數字都可能只是巧合）。',
                    icon='ℹ️',
                )
            window_days = list(range(win_start, win_end + 1))

            def _recompute_sig(row):
                # 2026-09-02起：觸發判斷預設改用「處置前5日高點」基準(峰值版欄位)，
                # 可用上面的basis_mode切回D0收盤版對照——見update_signals.py同一處
                # 註解(disposal_peak3_reference_test研究)。進場價/損益重建一律用
                # D0收盤版D{n}%換算，不受選用哪個觸發基準影響。
                trig_n = None
                for n in window_days:
                    lv = row.get(f'LowD{n}%{low_suffix}')
                    if pd.notna(lv) and lv < -5:
                        trig_n = n
                        break
                if trig_n is None:
                    return pd.Series({'買進訊號': '', '觸發方式': '', '目前損益(%)': np.nan})
                close_v = row.get(f'D{trig_n}%')
                close_v_cls = close_v if use_old_basis else row.get(f'D{trig_n}%{low_suffix}')
                ttype = '收盤跌破(A)' if pd.notna(close_v_cls) and close_v_cls < -5 else '僅盤中觸及(C)'
                cur_cum = np.nan
                for n in range(5, 0, -1):
                    v = row.get(f'D{n}%')
                    if pd.notna(v):
                        cur_cum = v
                        break
                pl = np.nan
                if pd.notna(cur_cum) and pd.notna(close_v):
                    ef = 1 + close_v / 100
                    cf = 1 + cur_cum / 100
                    if ef > 0:
                        pl = round((cf / ef - 1) * 100, 2)
                return pd.Series({'買進訊號': f'D{trig_n}', '觸發方式': ttype, '目前損益(%)': pl})

            if not sub_sig.empty:
                _r = sub_sig.apply(_recompute_sig, axis=1)
                for c in ['買進訊號', '觸發方式', '目前損益(%)']:
                    sub_sig[c] = _r[c]

            def _recompute_hist(row):
                d0 = row.get('D0收盤價')
                # 2026-09-02起：觸發判斷預設用峰值版欄位(處置前5日高點基準)，
                # 可用basis_mode切回D0收盤版對照；買進價/出關價重建一律用
                # D0收盤版D{n}%/出關開盤%換算，不受選用哪個觸發基準影響。
                trig_n = None
                for n in window_days:
                    lv = row.get(f'LowD{n}%{low_suffix}')
                    if pd.notna(lv) and lv < -5:
                        trig_n = n
                        break
                if trig_n is None:
                    return pd.Series({'買進日': '-', '買進價': np.nan, '買進時累積(%)': np.nan,
                                       '出關價': np.nan, '出關報酬(%)': np.nan, '結果': '-'})
                close_v = row.get(f'D{trig_n}%')
                exit_v = row.get('出關開盤(相對D0)%')
                # 價格先對齊跳動單位，報酬%一定要用對齊後的真實價格重算，
                # 不能沿用對齊前的百分比——不然價格跟報酬會對不起來(Kevin抓到的問題)。
                entry_price = round_to_tick(d0 * (1 + close_v / 100)) if pd.notna(d0) and pd.notna(close_v) else np.nan
                exit_price  = round_to_tick(d0 * (1 + exit_v / 100)) if pd.notna(d0) and pd.notna(exit_v) else np.nan
                ret = np.nan
                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                    ret = round((exit_price / entry_price - 1) * 100, 2)
                result = (f'✅ {ret:+.2f}%' if pd.notna(ret) and ret > 0
                          else (f'❌ {ret:+.2f}%' if pd.notna(ret) else '-'))
                return pd.Series({'買進日': f'D{trig_n}', '買進價': entry_price, '買進時累積(%)': close_v,
                                   '出關價': exit_price, '出關報酬(%)': ret, '結果': result})

            if not sub_hist.empty:
                _rh = sub_hist.apply(_recompute_hist, axis=1)
                for c in ['買進日', '買進價', '買進時累積(%)', '出關價', '出關報酬(%)', '結果']:
                    sub_hist[c] = _rh[c]

        st.subheader(f'🔔 今日訊號（{key}）')
        if sub_sig.empty:
            st.info('目前沒有符合條件的處置中股票。')
        else:
            display_cols = ['評級', '訊號', '買進訊號', '觸發方式', '代號', '名稱', '規模', '處置原因', '近20日漲幅', '大戶(%)',
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
            if '訊號' in disp.columns:
                styled = styled.map(color_signal, subset=['訊號'])
            st.dataframe(styled, use_container_width=True, height=min(400, 60 + 35 * len(disp)))

            # ── 進場預覽 / 觸發分析（依缺口排序）── 跟首頁Tab1同樣的邏輯，只是資料源換成新制 ──
            if '距觸發(%)' in sub_sig.columns and '觸發價' in sub_sig.columns:
                with st.expander(f'📍 進場預覽 / 觸發分析（{key}，依缺口排序）', expanded=True):
                    prev_base = sub_sig[sub_sig['處置原因'] == '漲多處置'].copy()
                    if prev_base.empty:
                        st.info('目前沒有漲多處置的股票可預覽。')
                    else:
                        # 第一次(5分盤動能，非alt)是「D1開盤不能高於觸發價」，距觸發(%)正值=目前
                        # 還沒開高(安全)、負值=已經開高(失格)——跟第二次+「還要再跌多少」的方向相反，
                        # 不能沿用同一套「再跌X%」文字，否則會誤導成方向相反的意思。
                        is_first_primary = (key == '第一次') and not use_alt

                        def fmt_preview(row):
                            entry = str(row.get('買進訊號', ''))
                            if entry.startswith('D'):
                                return f'✅ 已觸發 ({entry})'
                            gap = row.get('距觸發(%)', np.nan)
                            if pd.isna(gap):
                                return '-'
                            if is_first_primary:
                                if gap < 0:
                                    return f'❌ 已開高失格 {abs(gap):.2f}%'
                                elif gap <= 1:
                                    return f'🔥 距開高失格僅 {gap:.2f}%'
                                else:
                                    return f'🟢 距開高失格還有 {gap:.2f}%'
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
                            if '✅' in s or '🟢' in s: return 'color:#26c281;font-weight:700'
                            if '❌' in s: return 'color:#e74c3c;font-weight:700'
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
                        if is_first_primary:
                            st.caption(
                                '觸發條件（5分盤動能）：D0(處置前一天)漲2~9%，且D1開盤不高於D0收盤（觸發價=D0收盤）、'
                                'D1收盤沒鎖漲停 → D1收盤買進。距觸發(%)為正代表目前還沒開高（安全），轉負代表已經開高、訊號作廢。'
                            )
                        elif key == '第一次':
                            st.caption(
                                f'已觸發 = 距觸發 ≥ 0%（目前收盤已低於觸發價）｜🔥 = 距觸發 < 2%（高度警戒）｜'
                                f'觸發條件：D1~D4 任意天收盤累積跌幅 < -5%（套用第二次+規則於第一次，僅供比較）'
                            )
                        else:
                            st.caption(
                                f'觸發條件：D1~D4 任意天「最低價」相對D0收盤跌破-5%（不只看收盤；2026-08-30起窗口由D3~D5'
                                f'改為D1~D4，因新制真實資料顯示D1單日表現最好——樣本仍很小，屬觀察中的規則，非正式驗證結論，'
                                f'詳見disposal_entry_day_full_test研究）'
                            )

        st.divider()
        st.subheader(f'⚖️ 策略對照：新制 vs 舊制基準（{key} vs 舊制{OLD_TYPE_MAP[key]}）')
        if key == '第一次':
            if use_alt:
                st.caption('用「D1~D4(新制)/D3~D8(舊制) 任一天跌破-5%進場、出關日開盤出場」規則比較（套用第二次+規則於第一次，僅供比較）。')
            else:
                st.caption('用「5分盤動能」規則比較：D0漲2~9%、D1不跳空高開、D1收盤沒鎖漲停 → D1收盤買進、出關日開盤出場——這是第一次已驗證的規則，新舊制用同一套規則比較表現是否相似。')
        else:
            st.caption('用「D1~D4(新制)/D3~D8(舊制) 任一天跌破-5%進場、出關日開盤出場」規則比較——舊制窗口是已驗證的基準；新制窗口2026-08-30起改為D1~D4，樣本仍很小，屬觀察中的規則。')

        old_pool = (old_hist[old_hist['處置類型'] == OLD_TYPE_MAP[key]] if len(old_hist) else pd.DataFrame()).copy()
        if key == '第一次' and use_alt and len(old_pool):
            for base_col, alt_col in [('買進日', '買進日(-5%版)'), ('出關報酬(%)', '出關報酬(-5%版)(%)')]:
                if alt_col in old_pool.columns:
                    old_pool[base_col] = old_pool[alt_col]
        old_triggered = old_pool[old_pool['買進日'] != '-'] if len(old_pool) else pd.DataFrame()
        new_triggered = sub_hist[(sub_hist['出關報酬(%)'].notna()) & (sub_hist['買進日'] != '-')] if len(sub_hist) else pd.DataFrame()

        comp_rows = []
        if len(old_triggered):
            r = old_triggered['出關報酬(%)']
            comp_rows.append({'版本': f'舊制{OLD_TYPE_MAP[key]}（歷史全樣本）', '已出關且觸發筆數': len(r),
                               '勝率': f'{(r > 0).mean()*100:.1f}%', '期望報酬': f'{r.mean():+.2f}%'})
        if len(new_triggered):
            r = new_triggered['出關報酬(%)']
            comp_rows.append({'版本': f'新制{key}（累積至今）', '已出關且觸發筆數': len(r),
                               '勝率': f'{(r > 0).mean()*100:.1f}%', '期望報酬': f'{r.mean():+.2f}%'})
        else:
            comp_rows.append({'版本': f'新制{key}（累積至今）', '已出關且觸發筆數': 0,
                               '勝率': '-', '期望報酬': '尚無已出關且觸發的樣本'})
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # ── D1~D5 平均走勢對照（用「今日訊號」裡進行中的事件，不用等出關，樣本比完整出關對照多）──
        new_path_src = sub_sig  # 這個tab目前進行中(尚未出關)的事件，已經是 sig[處置次別==key]
        if len(new_path_src) and 'D1%' in new_path_src.columns and len(old_pool):
            path_rows = []
            for n in range(1, 6):
                new_v = new_path_src[f'D{n}%'].mean() if f'D{n}%' in new_path_src.columns else np.nan
                old_v = old_pool[f'D{n}累積(%)'].mean() if f'D{n}累積(%)' in old_pool.columns else np.nan
                path_rows.append({'天數': f'D{n}', f'新制{key}平均累積(進行中,n={len(new_path_src)})': f'{new_v:+.2f}%' if pd.notna(new_v) else '-',
                                   f'舊制{OLD_TYPE_MAP[key]}平均累積(歷史全樣本,n={len(old_pool)})': f'{old_v:+.2f}%' if pd.notna(old_v) else '-'})
            st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)
            st.caption(
                '新制那欄只算「今日訊號」裡目前還在進行中(尚未出關)的事件，樣本會隨事件陸續出關而變動，不是固定母體；'
                '已出關的完整結果請看下面「累積至今結果」。這裡純粹是看早期走勢形狀是否相似，不是嚴謹的統計對照。'
            )
        else:
            st.caption('資料不足，暫時無法比較D1~D5平均走勢。')

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
                rule_label = ('觸發D1~D4<-5%進場訊號' if key != '第一次' or use_alt else '觸發5分盤動能進場訊號')
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric('已出關筆數', len(settled), delta=f'{len(triggered)} 筆有{rule_label}', delta_color='off')
                c2.metric('觸發後勝率', f'{(trig_rets > 0).mean() * 100:.1f}%' if len(trig_rets) else '-')
                c3.metric('觸發後期望報酬', f'{trig_rets.mean():+.2f}%' if len(trig_rets) else '-')
                c4.metric('夏普值', f'{sharpe:.2f}' if pd.notna(sharpe) else '-')
                c5.metric('賺賠比', f'{pf:.2f}' if pd.notna(pf) else '-')
                st.caption(
                    f'全部{key}已出關（不論是否觸發進場條件）平均報酬：{rets.mean():+.2f}%，勝率 {(rets > 0).mean()*100:.1f}%'
                    f'（n={len(settled)}）。樣本數還很小，以上數字僅供觀察趨勢，不是可信賴的統計結果。'
                )
                show_cols = ['起始日', '出關日', '代號', '名稱', '規模', 'Dn組別', '買進日', '觸發方式',
                             '買進價', '買進時累積(%)', '最深日', '期間最深(%)', '出關價', '出關報酬(%)', '結果']
                show_cols += [f'T+{k}收盤(%)' for k in range(1, 11)]
                show_cols = [c for c in show_cols if c in settled.columns]
                show_disp = settled[show_cols].sort_values('起始日', ascending=False).copy()
                for c in [f'T+{k}收盤(%)' for k in range(1, 11)]:
                    if c in show_disp.columns:
                        show_disp[c] = show_disp[c].apply(lambda v: f'{v:+.2f}%' if pd.notna(v) else '-')
                st.dataframe(show_disp, use_container_width=True, height=min(400, 60 + 35 * len(settled)))
                st.caption('T+1~T+10收盤(%)：若出關日沒有照規則賣出、繼續抱著，之後10個交易日的報酬（基準是買進日收盤價）——比照舊制Tab2「歷史回測紀錄」同樣的欄位，方便對照是否該提早或延後出場。')

                # D1~D5各自獨立表現：不管現在選的窗口是哪一段，這張表固定顯示每一天
                # 單獨進場(當天最低價跌破-5%就當天收盤買)的表現，方便一眼比較哪天最好。
                if key == '第二次+':
                    st.markdown('##### 📊 D1~D5 各自獨立表現（不受上面窗口選擇影響，固定顯示每一天）')
                    day_rows = []
                    for n in range(1, 6):
                        low_col, close_col = f'LowD{n}%{low_suffix}', f'D{n}%'
                        if low_col not in sub_hist.columns or close_col not in sub_hist.columns:
                            continue
                        trig = sub_hist[sub_hist[low_col] < -5].copy()
                        if trig.empty:
                            day_rows.append({'進場日': f'D{n}', '觸發數': 0, '已出關': 0, '勝率': '-', '平均報酬': '-'})
                            continue
                        d0v = trig['D0收盤價']
                        entry_p = [round_to_tick(d0*(1+c/100)) if pd.notna(d0) and pd.notna(c) else np.nan
                                   for d0, c in zip(d0v, trig[close_col])]
                        exit_p  = [round_to_tick(d0*(1+e/100)) if pd.notna(d0) and pd.notna(e) else np.nan
                                   for d0, e in zip(d0v, trig['出關開盤(相對D0)%'])]
                        ret = [round((xp/ep-1)*100, 2) if pd.notna(ep) and pd.notna(xp) and ep > 0 else np.nan
                               for ep, xp in zip(entry_p, exit_p)]
                        ret_s = pd.Series(ret).dropna()
                        day_rows.append({
                            '進場日': f'D{n}', '觸發數': len(trig), '已出關': len(ret_s),
                            '勝率': f'{(ret_s > 0).mean()*100:.1f}%' if len(ret_s) else '-',
                            '平均報酬': f'{ret_s.mean():+.2f}%' if len(ret_s) else '-',
                        })
                    st.dataframe(pd.DataFrame(day_rows), use_container_width=True, hide_index=True)
                    st.caption('每一天都獨立計算「當天最低價跌破-5%就當天收盤買進」的表現，不是「第一次觸發」，同一事件可能在好幾天都算進去——樣本還很小，僅供觀察趨勢。')

                # 出關時間點比較：跟現行「出關日開盤賣」比，若延後到出關日收盤、或再抱1~10天，
                # 表現會怎樣，比照舊制Tab2「⏰出場時間點比較」的呈現方式。
                st.markdown('##### ⏰ 出關時間點比較')
                exit_rows = [{
                    '出場時間點': 'T+1 開盤（現行策略）',
                    '筆數': len(trig_rets),
                    '勝率': f'{(trig_rets > 0).mean()*100:.1f}%' if len(trig_rets) else '-',
                    '平均報酬': f'{trig_rets.mean():+.2f}%' if len(trig_rets) else '-',
                }]
                for k in range(1, 11):
                    col = f'T+{k}收盤(%)'
                    if col not in triggered.columns:
                        continue
                    s = triggered[col].dropna()
                    exit_rows.append({
                        '出場時間點': f'T+{k} 收盤',
                        '筆數': len(s),
                        '勝率': f'{(s > 0).mean()*100:.1f}%' if len(s) else '-',
                        '平均報酬': f'{s.mean():+.2f}%' if len(s) else '-',
                    })
                st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)
                st.caption('T+1開盤是現行規則（出關日開盤賣出）；T+1~T+10收盤是「如果沒賣、繼續抱著」的對照，基準都是買進日收盤價。比照舊制Tab2的呈現方式。')

st.divider()
with st.expander('📖 本頁的已知近似與侷限（務必先讀再解讀數字）'):
    st.markdown("""
- **5天 vs 7天不分**：新制處置期間一般是5個營業日，若同時因當沖比重過高遭加重處置則是7個營業日；本頁資料沒有欄位可以精確分辨，統一用5天近似，可能讓少數7天案例的「今D幾」「出關日」顯示提前。
- **進場窗口預設為D1~D4**（不是舊制20分鐘的D3~D8，也不是2026-08-26~08-30間曾經用過的D3~D5）：2026-08-30由Kevin主動決定換成D1~D4，依據是新制真實資料(僅十幾筆已結算樣本)顯示D1單日表現目前最好——**這是樣本很小情況下的主動決策，不是這系列一貫「驗證足夠才換」的標準流程**，屬於觀察中的規則。可以用頁面上的窗口選擇器切回D3~D5或其他範圍比較。
- **評級只是描述性分類**：套用跟舊制頁面相同的規則(grade())來標「✅主力訊號/🟡觀察中」等，純粹方便閱讀，**不代表這些評級在新制下已被證實有效**。
- **樣本數極少**：新制上路才幾天，已完整出關的樣本數只有個位數到十幾筆，任何統計數字都可能只是巧合，需要更長時間累積才能判斷。
- 完整規則變更細節與研究進度，見 PLAYBOOK.md「研究 Roadmap」章節。
""")
