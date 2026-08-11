# -*- coding: utf-8 -*-
"""分點兩兩配對訊號：逐筆歷史紀錄。2026-08-10 pairwise 分析找到的三組組合。"""
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='分點組合訊號逐筆紀錄', page_icon='🔗', layout='wide')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def sharpe_pf(returns):
    r = returns.dropna()
    if len(r) < 2:
        return np.nan, np.nan
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std(ddof=1) else np.nan
    gains = r[r > 0].sum()
    losses = -r[r <= 0].sum()
    pf = gains / losses if losses > 0 else np.nan
    return sharpe, pf


@st.cache_data(ttl=3600)
def load_events(_cache_bust: str = '2026-08-10-v1'):
    fp = os.path.join(DATA_DIR, 'branch_pair_events_20260810.csv')
    df = pd.read_csv(fp, parse_dates=['d0', 'd1'])
    df['code'] = df['code'].astype(str)
    names_fp = os.path.join(DATA_DIR, 'stock_names.csv')
    if os.path.exists(names_fp):
        names = pd.read_csv(names_fp, dtype={'code': str})
        df = df.merge(names, on='code', how='left')
        df['name'] = df['name'].fillna('')
    else:
        df['name'] = ''
    return df


st.title('🔗 分點組合訊號｜逐筆歷史紀錄')
st.warning(
    '⚠️ **誠實揭露，先讀這段**：這三組配對訊號在2018~2026-05-05期間的證據都很扎實(t值4~6，'
    'p<0.0001)，2026-05-06起(近期)方向也一致為正、且都贏過兩邊分點各自單獨表現，但**近期資料的'
    '正式統計顯著性，會隨著「跟多少組配對一起做多重比較校正」而變動**——同一組配對，在較窄的比較'
    '族群下能通過p<0.05，換成較寬的比較族群(例如跟19.8萬組全配對一起校正)就通不過。這不是訊號忽強'
    '忽弱，是近期資料量(目前僅約3個月)本身還不夠大，任何嚴格校正都撐不住小樣本。**這些是「有希望、'
    '方向一致，但還沒正式證實」的觀察名單，不是已驗證可上線的訊號**，需要更多時間累積近期資料才能'
    '真正確認。完整方法論與四種比較族群校正結果見 `E:\\stock\\reports\\branch_pair_*_20260810.md`'
    '（本機研究報告，不在此網站上）。'
)
st.caption(
    'D0嚴格鎖漲停 + 兩個分點同天皆為正淨買超 → D1開盤放空，觸及停損天花板(D0收盤算出的隔日漲停×0.98)'
    '提前出場，否則D1收盤回補。三組配對：①統一-城中×富邦/富邦證券(裸名稱) ②統一(母公司裸名稱，可能'
    '是自營盤)×統一-城中——統一自己單獨買時報酬是負的(-0.14%)，只有跟統一城中同天一起買才轉正，機制'
    '解讀是「統一扮演確認訊號角色，篩選出統一城中訊號裡比較真的子集」③統一-城中×凱基-城中(城中GA本尊，'
    'Kevin原始假設)。資料截至本機finlab_db分點快取最新可用日(2026-08-07)。'
)

events = load_events()

st.divider()

# ── 篩選器 ──────────────────────────────────────────────
col_f1, col_f2 = st.columns(2)
with col_f1:
    pair_options = events['pair'].unique().tolist()
    sel_pair = st.selectbox('分點組合', pair_options, index=0)
with col_f2:
    sel_period = st.multiselect('期間', ['in_sample', 'out_of_sample'],
                                 default=['in_sample', 'out_of_sample'],
                                 format_func=lambda x: '2018~2026-05-05期間' if x == 'in_sample' else '2026-05-06起(近期)')

col_f3, col_f4 = st.columns(2)
with col_f3:
    exclude_frozen = st.checkbox('排除D1鎖死(frozen)事件', value=True,
                                  help='D1當天開高低收完全相同，代表整天鎖住無法真實成交')
with col_f4:
    exclude_abs20 = st.checkbox('排除|報酬|≥20%極端值', value=True,
                                 help='與本站其他頁面一致的離群值排除規則，避免少數極端事件主導平均數')

mask = events['pair'] == sel_pair
mask &= events['period'].isin(sel_period)
if exclude_frozen:
    mask &= ~events['frozen']
if exclude_abs20:
    mask &= ~events['abs20_excluded']
view = events[mask].copy()

st.divider()

# ── KPI ──────────────────────────────────────────────────
if len(view) == 0:
    st.warning('目前篩選條件下無資料')
else:
    wr = view['success'].mean() * 100
    avg_r = view['short_ret_pct'].mean()
    med_r = view['short_ret_pct'].median()
    sharpe, pf = sharpe_pf(view['short_ret_pct'])
    stop_rate = view['stopped'].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('筆數', f'{len(view)} 筆')
    c2.metric('勝率', f'{wr:.2f}%')
    c3.metric('平均放空報酬', f'{avg_r:+.2f}%', delta=f'中位數 {med_r:+.2f}%', delta_color='off')
    c4.metric('Sharpe(近似, 事件層級非日組合)', f'{sharpe:.2f}' if pd.notna(sharpe) else '-')
    c5.metric('觸停損天花板比例', f'{stop_rate:.2f}%')

st.divider()

# ── 各期間績效對照 ──────────────────────────────────
with st.expander('📊 各期間績效對照（此組合，未套用上方篩選器的期間限制）', expanded=True):
    rows = []
    for p in ['in_sample', 'out_of_sample']:
        sub = events[(events['pair'] == sel_pair) & (events['period'] == p)]
        if exclude_frozen:
            sub = sub[~sub['frozen']]
        if exclude_abs20:
            sub = sub[~sub['abs20_excluded']]
        if len(sub) == 0:
            continue
        sh, _ = sharpe_pf(sub['short_ret_pct'])
        rows.append({
            '期間': '2018~2026-05-05期間' if p == 'in_sample' else '2026-05-06起(近期)',
            '筆數': len(sub),
            '勝率(%)': round(sub['success'].mean() * 100, 2),
            '平均報酬(%)': round(sub['short_ret_pct'].mean(), 2),
            '中位數報酬(%)': round(sub['short_ret_pct'].median(), 2),
            'Sharpe(近似)': round(sh, 2) if pd.notna(sh) else None,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index('期間').style.format(
            {'勝率(%)': '{:.2f}', '平均報酬(%)': '{:+.2f}', '中位數報酬(%)': '{:+.2f}', 'Sharpe(近似)': '{:.2f}'},
            na_rep='-',
        ), use_container_width=True)
    st.caption('近期資料筆數天生較少（統計力較弱），這是為何正式顯著性對校正方式敏感的直接原因，不代表訊號變弱。')

# ── 完整逐筆歷史紀錄 ──────────────────────────────────────
st.subheader(f'📋 完整逐筆歷史紀錄（{len(view)} 筆，{sel_pair}）')

show_cols = ['d0', 'code', 'name', 'd1', 'period', 'i_net_amt_wan', 'i_influence_pct',
             'j_net_amt_wan', 'j_influence_pct', 'd1_open', 'd1_high', 'd1_close',
             'stop_price', 'stopped', 'frozen', 'short_ret_pct', 'success']
show = view[show_cols].sort_values('d0', ascending=False).copy()
show['d0'] = show['d0'].dt.strftime('%Y-%m-%d')
show['d1'] = show['d1'].dt.strftime('%Y-%m-%d')
show['period'] = show['period'].map({'in_sample': '2018~2026-05-05', 'out_of_sample': '2026-05-06起'})
branch_i_label = view['branch_i'].iloc[0] if len(view) else 'i'
branch_j_label = view['branch_j'].iloc[0] if len(view) else 'j'
show.columns = ['D0訊號日', '代號', '名稱', 'D1進場日', '期間', f'{branch_i_label}買超(萬)',
                f'{branch_i_label}影響力%', f'{branch_j_label}買超(萬)', f'{branch_j_label}影響力%',
                '開盤', '最高', '收盤', '停損天花板價', '觸停損', 'D1鎖死', '放空報酬%', '成功']


def color_success(val):
    if val is True:
        return 'color: #26c281; font-weight: 700'
    if val is False:
        return 'color: #e74c3c; font-weight: 700'
    return ''


def color_ret(val):
    try:
        v = float(val)
        if v > 5:
            return 'color: #26c281; font-weight: 700'
        if v > 0:
            return 'color: #2ecc71'
        if v > -5:
            return 'color: #e67e22'
        return 'color: #e74c3c; font-weight: 700'
    except Exception:
        return ''


show_format = {
    f'{branch_i_label}買超(萬)': '{:,.2f}', f'{branch_i_label}影響力%': '{:.2f}',
    f'{branch_j_label}買超(萬)': '{:,.2f}', f'{branch_j_label}影響力%': '{:.2f}',
    '開盤': '{:.2f}', '最高': '{:.2f}', '收盤': '{:.2f}', '停損天花板價': '{:.2f}',
    '放空報酬%': '{:+.2f}',
}
st.dataframe(
    show.style
        .map(color_success, subset=['成功'])
        .map(color_ret, subset=['放空報酬%'])
        .format(show_format, na_rep='-'),
    use_container_width=True, height=520,
)
st.caption('買超(萬)/影響力%欄位若空白，代表該分點在這筆事件當天的原始交易列在快取中找不到精確金額（通常是資料join邊界問題，不影響報酬計算本身，報酬只依賴D1的OHLC價格）。')
st.download_button(
    '📥 下載此表 CSV',
    show.to_csv(index=False, encoding='utf-8-sig'),
    f'branch_pair_events_{sel_pair}.csv'.replace('/', '_').replace(' ', '_'), 'text/csv', key='dl_pair_events',
)

st.divider()
st.markdown('**累積報酬走勢**（假設每筆等權重，依D0訊號日排序）')
cum_src = view.sort_values('d0')['short_ret_pct'].dropna()
if len(cum_src) >= 2:
    cum = (cum_src / 100 + 1).cumprod() - 1
    cum.index = range(len(cum))
    st.line_chart(pd.DataFrame({'累積報酬(%)': (cum * 100).values}), height=250)
else:
    st.caption('目前篩選條件下筆數不足，無法繪製累積報酬走勢。')

st.markdown('---')
st.subheader('🔎 三組配對橫向比較（全期間合併，未篩選frozen/極端值）')
compare_rows = []
for p in events['pair'].unique():
    sub = events[events['pair'] == p]
    sh, _ = sharpe_pf(sub['short_ret_pct'])
    compare_rows.append({
        '組合': p, '總筆數': len(sub),
        '勝率(%)': round(sub['success'].mean() * 100, 2),
        '平均報酬(%)': round(sub['short_ret_pct'].mean(), 2),
    })
st.dataframe(pd.DataFrame(compare_rows).set_index('組合').style.format(
    {'勝率(%)': '{:.2f}', '平均報酬(%)': '{:+.2f}'}), use_container_width=True)

st.caption(
    '本頁是三組配對訊號的可篩選、可下載逐筆事件明細；富邦證券(裸名稱)獨立分點的單獨規則（非配對）'
    '請見 **pages/16「富邦證券獨立分點研究」**，兩頁互為補充、不重複。方法論、程式碼與四種比較族群'
    '(369/2,514/2,517/4,278組)的完整校正結果只存在本機研究報告，未上傳此網站。'
)
