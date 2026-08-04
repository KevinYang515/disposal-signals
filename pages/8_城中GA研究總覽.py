# -*- coding: utf-8 -*-
"""凱基-城中GA 隔日沖放空策略：歷史回測紀錄與因子挖掘總覽。"""
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='城中GA研究總覽', page_icon='🏙️', layout='wide')

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
def load_events():
    fp = os.path.join(DATA_DIR, 'citycenter_ga_events.csv')
    df = pd.read_csv(fp, parse_dates=['d0', 'd1'])
    df['code'] = df['code'].astype(str)
    names_fp = os.path.join(DATA_DIR, 'stock_names.csv')
    names = pd.read_csv(names_fp, dtype={'code': str})
    df = df.merge(names, on='code', how='left')
    df['name'] = df['name'].fillna('')
    return df


st.title('🏙️ 凱基-城中GA 隔日沖放空策略｜歷史回測紀錄')
st.caption('D0嚴格鎖漲停 + 凱基-城中淨買超金額/影響力雙門檻達標 → D1開盤放空、收盤回補。動態門檻：net_amt_wan≥263.568萬 且 cz_influence_pct≥0.86301%（TRAIN 2021-2023自身分布40百分位校準，取代舊版固定絕對金額門檻）。資料截至2026-07-24，2026-08-04整理，即時候選/模擬單請見「城中GA放空策略」頁。')

events = load_events()
events['censored'] = events['censored'].astype(bool)
events['success'] = events['success'].astype(bool)
events['period'] = np.where(events['d0'] < pd.Timestamp('2024-01-01'), 'TRAIN 2021-2023', 'TEST 2024-2026')

# ── 篩選器 ──────────────────────────────────────────────
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    sel_period = st.multiselect('期間', ['TRAIN 2021-2023', 'TEST 2024-2026'], default=['TRAIN 2021-2023', 'TEST 2024-2026'])
with col_f2:
    sel_market = st.multiselect('市場', sorted(events['market'].dropna().unique().tolist()), default=sorted(events['market'].dropna().unique().tolist()))
with col_f3:
    gap_bins = ['<0%', '0-1%', '1-9%(甜蜜點)', '9-9.5%', '≥9.5%(避開)']
    sel_gap = st.multiselect('D1開盤跳空區間', gap_bins, default=gap_bins)
with col_f4:
    sel_streak = st.multiselect('連續鎖漲停天數', [1, 2, 3, 4], default=[1, 2, 3, 4])


def gap_bucket(g):
    if g < 0:
        return '<0%'
    if g < 1:
        return '0-1%'
    if g < 9:
        return '1-9%(甜蜜點)'
    if g < 9.5:
        return '9-9.5%'
    return '≥9.5%(避開)'


events['gap_bucket'] = events['gap_pct'].apply(gap_bucket)
events['streak_capped'] = events['lock_streak'].clip(upper=4)

view = events[
    events['period'].isin(sel_period)
    & events['market'].isin(sel_market)
    & events['gap_bucket'].isin(sel_gap)
    & events['streak_capped'].isin(sel_streak)
].copy()

st.divider()

# ── KPI ──────────────────────────────────────────────────
settled = view[~view['censored']]
if len(settled) == 0:
    st.warning('目前篩選條件下無已結算資料')
else:
    wr = settled['success'].mean() * 100
    avg_r = settled['short_ret_open_to_close_pct'].mean()
    sharpe, pf = sharpe_pf(settled['short_ret_open_to_close_pct'])
    frozen_rate = view['d1_frozen'].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('已結算筆數', f'{len(settled)} 筆', delta=f'{len(view)} 篩選出', delta_color='off')
    c2.metric('勝率', f'{wr:.2f}%')
    c3.metric('平均放空報酬', f'{avg_r:+.2f}%')
    c4.metric('日組合Sharpe(近似)', f'{sharpe:.2f}' if pd.notna(sharpe) else '-')
    c5.metric('D1鎖死率', f'{frozen_rate:.2f}%', help='D1當天直接鎖漲停開不了倉的比例；≥9.5%跳空區間佔了81.4%的鎖死案例')

st.divider()

# ── Gap × 是否鎖漲停 分組表 ─────────────────────────────
with st.expander('📊 依「開盤跳空區間」分組（目前篩選條件下）', expanded=True):
    rows = []
    for g in gap_bins:
        sub = view[(view['gap_bucket'] == g) & (~view['censored'])]
        if len(sub) == 0:
            continue
        sh, pf_g = sharpe_pf(sub['short_ret_open_to_close_pct'])
        rows.append({
            '跳空區間': g,
            '筆數': len(sub),
            '勝率(%)': round(sub['success'].mean() * 100, 2),
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 3),
            '最慘單筆(%)': round(sub['short_ret_open_to_close_pct'].min(), 2),
            'Sharpe(近似)': round(sh, 2) if pd.notna(sh) else None,
            'D1鎖死率(%)': round(view[view['gap_bucket'] == g]['d1_frozen'].mean() * 100, 1),
        })
    if rows:
        gdf = pd.DataFrame(rows).set_index('跳空區間')

        def _wr_color(v):
            if v >= 70:
                return 'background-color:#1a5c38;color:white;font-weight:700'
            if v >= 55:
                return 'background-color:#26c281;color:black;font-weight:700'
            if v >= 45:
                return 'background-color:#f6c90e;color:black'
            return 'background-color:#c0392b;color:white'

        st.dataframe(
            gdf.style.map(_wr_color, subset=['勝率(%)']).format(
                {'平均報酬(%)': '{:+.3f}', '最慘單筆(%)': '{:+.2f}', 'D1鎖死率(%)': '{:.1f}'}, na_rep='-'
            ),
            use_container_width=True,
        )
        st.caption('⭐ 9.5%以上是關鍵懸崖：D1鎖死率從9%以下的個位數暴增，最慘單筆也急遽惡化。這是目前找到最有效的防嘎漲停規則。')

# ── lock_streak 分組表 ──────────────────────────────────
with st.expander('📊 依「連續鎖漲停天數」分組（跟FlipBranch規律相反）'):
    rows = []
    for s in [1, 2, 3, 4]:
        sub = view[(view['streak_capped'] == s) & (~view['censored'])]
        if len(sub) < 5:
            continue
        rows.append({
            'lock_streak': f'{s}{"+" if s == 4 else ""}',
            '筆數': len(sub),
            '勝率(%)': round(sub['success'].mean() * 100, 2),
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 3),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index('lock_streak').style.format({'平均報酬(%)': '{:+.3f}'}), use_container_width=True)
    st.caption('城中GA連續鎖漲停天數越多、隔日勝率越高（streak≥3達73.7%），跟FlipBranch「連鎖越多越差」剛好相反，不要混用兩策略的濾網邏輯。')

# ── 逐年穩定性 ──────────────────────────────────────────
with st.expander('📅 逐年穩定性（目前篩選條件下）'):
    yr_rows = []
    for yr, sub in view[~view['censored']].groupby(view['d0'].dt.year):
        if len(sub) < 5:
            continue
        sh, _ = sharpe_pf(sub['short_ret_open_to_close_pct'])
        yr_rows.append({
            '年份': yr, '筆數': len(sub),
            '勝率(%)': round(sub['success'].mean() * 100, 2),
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 3),
            'Sharpe(近似)': round(sh, 2) if pd.notna(sh) else None,
        })
    if yr_rows:
        st.dataframe(pd.DataFrame(yr_rows).set_index('年份'), use_container_width=True)

# ── 完整逐筆明細（主表，直接顯示，比照「歷史回測紀錄」頁樣式）──
st.subheader(f'📋 完整逐筆歷史紀錄（{len(view)} 筆，依目前篩選條件）')

show_cols = ['d0', 'code', 'name', 'market', 'd1', 'gap_pct', 'lock_streak', 'net_amt_wan',
             'cz_influence_pct', 'd1_frozen', 'censored', 'short_ret_open_to_close_pct',
             'short_mae_pct', 'success']
show = view[show_cols].sort_values('d0', ascending=False).copy()
show['d0'] = show['d0'].dt.strftime('%Y-%m-%d')
show['d1'] = show['d1'].dt.strftime('%Y-%m-%d')
show.columns = ['D0訊號日', '代號', '名稱', '市場', 'D1進場日', '跳空%', '連鎖天數', '買超金額(萬)',
                 '影響力%', 'D1鎖死', '截尾', '放空報酬%', '最大不利波動%', '成功']


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


st.dataframe(
    show.style
        .map(color_success, subset=['成功'])
        .map(color_ret, subset=['放空報酬%'])
        .format({'跳空%': '{:+.2f}', '買超金額(萬)': '{:,.0f}', '影響力%': '{:.2f}',
                 '放空報酬%': '{:+.2f}', '最大不利波動%': '{:.2f}'}, na_rep='-'),
    use_container_width=True, height=520,
)
st.download_button('📥 下載此表 CSV', show.to_csv(index=False, encoding='utf-8-sig'),
                    'citycenter_ga_events_filtered.csv', 'text/csv', key='dl_events_full')

st.divider()
st.markdown('**累積報酬走勢**（假設每筆等權重，依D0訊號日排序）')
cum_src = view[~view['censored']].sort_values('d0')['short_ret_open_to_close_pct'].dropna()
if len(cum_src) >= 2:
    cum = (cum_src / 100 + 1).cumprod() - 1
    cum.index = range(len(cum))
    st.line_chart(pd.DataFrame({'累積報酬(%)': (cum * 100).values}), height=250)
else:
    st.caption('目前篩選條件下已結算筆數不足，無法繪製累積報酬走勢。')

st.markdown('---')
st.subheader('🔎 其他研究結論（非本頁資料表涵蓋範圍）')
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**承接／低接與做多方向**
- D1收盤守住D0收盤 → D1收→D2收顯著偏強（TRAIN+1.72%/p=0.02、TEST+0.70%/p=0.0035），持股遇賣壓但收平/收紅不用因此恐慌減碼
- D1承接強度越強，D2再度鎖漲停機率越高（弱承接7.5% → 極強承接25.0%，單調遞增）
- ⚠️但「D2開盤買進」做多策略判死：承接的漲幅主要在隔夜跳空(+1.0%)，D2盤中段反而是負的(-1.17%)
- 台新台北+凱基城中同時進D0前3大買方：n僅6筆，勝率83.3%/+3.33%，方向支持但樣本嚴重不足

**分點層級**
- 富邦證券、統一-城中、富邦-嘉義：獨立驗證可用；台新-台北(已確認即2026-04-06併購前的元富)、凱基台北：判死或效果太弱
- 富邦證券也複製了城中「D1自己淨賣才準」的機制(D1淨賣勝率63-64% vs 淨買23-24%)，證實非城中獨有現象
- 嘉義/虎尾地域聚合：判死，訊號完全由富邦-嘉義單一分點主導，不是地域性現象
""")
with col2:
    st.markdown("""
**已測試、判死或需更多資料**
- 是否首次買這檔股票、連續買超天數(不要求鎖漲停版)：樣本不足或方向不穩
- 分點共現(265高flip分點)：**負向訊號**——多個知名分點同時買超代表真買盤，隔日放空表現反而更差
- ML(LightGBM+SHAP)+GA系統性挖掘：TEST預測分數與實際報酬相關係數僅-0.012，現行兩門檻已是資料極限
- 技術性鎖漲停比例因子：Spearman相關幾乎是0，沒有支持證據
- 09:15檢查點應作觀察用，「一收回開盤就無條件出場」實測反而讓報酬變差
- 盤中停損/停利(2-8%)：全部測過，沒有一組贏過持有到收盤

**2026-08-04即時案例**：2454表現最佳(+5.62%，無負向旗標)；2301最差(-7.92%，gap僅1.74%在甜蜜點內卻被外資巨量買盤蓋過)——證實即使規則都對仍有真實虧損可能，這是策略固有的殘餘風險。
""")

st.caption('完整方法論見 D:\\stock 下 memory 系統 project_隔日沖分點戰術.md 第66節，以及各 CODEX_*.md 報告。')
