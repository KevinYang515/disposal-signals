# -*- coding: utf-8 -*-
"""富邦證券(裸名稱/母公司)獨立分點隔日沖策略：歷史回測紀錄與因子挖掘總覽。"""
import json
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='富邦證券獨立分點研究', page_icon='🧭', layout='wide')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def sharpe_pf(s):
    """(夏普值, 賺賠比) for a return series. Sharpe = mean/std; PF = avg_win/|avg_loss|.

    與站台首頁 app.py 的 sharpe_pf() 定義完全一致，方便跨頁比較數字。
    """
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan, np.nan
    sharpe = s.mean() / s.std() if s.std() > 0 else np.nan
    wins = s[s > 0]
    loss = s[s <= 0]
    pf = wins.mean() / abs(loss.mean()) if len(loss) > 0 and loss.mean() != 0 else np.nan
    return sharpe, pf


def _parse_spark(v):
    if not isinstance(v, str) or not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        return None


def synced_pct_input(label, default, key):
    """Slider + number_input synced via session_state — sliders are fiddly on mobile, this gives a typeable alternative."""
    slider_key, num_key = f'{key}_slider', f'{key}_num'
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default

    def _from_slider():
        st.session_state[num_key] = st.session_state[slider_key]

    def _from_num():
        st.session_state[slider_key] = st.session_state[num_key]

    st.slider(label, 0.0, 10.0, step=0.5, key=slider_key, on_change=_from_slider)
    if num_key not in st.session_state:
        st.session_state[num_key] = st.session_state[slider_key]
    st.number_input('或直接輸入數字（%，適合手機操作）', 0.0, 10.0, step=0.5, key=num_key, on_change=_from_num)
    return st.session_state[slider_key]


@st.cache_data(ttl=3600)
def load_events(_cache_bust: str = '2026-08-15-offline-branch-context-v1'):
    fp = os.path.join(DATA_DIR, 'fubon_branch_events.csv')
    df = pd.read_csv(fp, parse_dates=['d0', 'd1'])
    df['code'] = df['code'].astype(str).str.zfill(4)
    names_fp = os.path.join(DATA_DIR, 'stock_names.csv')
    if os.path.exists(names_fp):
        names = pd.read_csv(names_fp, dtype={'code': str})
        names['code'] = names['code'].str.zfill(4)
        df = df.merge(names, on='code', how='left')
        df['name'] = df['name'].fillna('')
    else:
        df['name'] = ''
    if 'd1_intraday_close' not in df.columns:
        df['d1_intraday_close'] = ''
    df['d1_intraday_spark'] = df['d1_intraday_close'].apply(_parse_spark)
    for col, default in [('d0_disposal', False), ('d1_disposal', False), ('d1_disposal_type', ''),
                          ('mktcap_billion', float('nan')), ('taiex_2day_mom_pct', float('nan')),
                          ('day_trade_short_suspended_d1', False)]:
        if col not in df.columns:
            df[col] = default
    df['day_trade_short_suspended_d1'] = df['day_trade_short_suspended_d1'].fillna(False).astype(bool)
    branch_context_defaults = {
        'city_ga_net_amt_wan': 0.0, 'city_ga_influence_pct': 0.0,
        'unicenter_city_net_amt_wan': 0.0, 'unicenter_city_influence_pct': 0.0,
        'fubon_net_amt_wan': 0.0, 'fubon_influence_pct': 0.0,
        'taishin_taipei_net_amt_wan': 0.0, 'taishin_taipei_influence_pct': 0.0,
        'd0_top3_net_buy_branches': '', 'top3_available': False,
        'v1_candidate_d1': False, 'other_branch_active_count': 0,
    }
    for col, default in branch_context_defaults.items():
        if col not in df.columns:
            df[col] = default
    df['d0_top3_net_buy_branches'] = df['d0_top3_net_buy_branches'].fillna('')
    for col in ['top3_available', 'v1_candidate_d1']:
        df[col] = df[col].fillna(False).astype(bool)
    df['other_branch_active_count'] = df['other_branch_active_count'].fillna(0).astype(int)
    return df


st.title('🧭 富邦證券(裸名稱)獨立分點｜隔日沖放空策略歷史回測')
st.caption(
    '此為富邦證券（母公司代號9600）當日鎖漲停淨買超門檻規則，與 pages/13「富邦證券連板隔日沖」的'
    '連續鎖漲停解鎖動能規則是**不同策略，請勿混用**——13是連續鎖漲停後解鎖隔天的動能規則，本頁是'
    '單日嚴格鎖漲停 + 富邦證券(裸名稱，排除所有富邦-XX分行)當日淨買超金額/影響力雙門檻達標，'
    'D1開盤放空、收盤回補。現行門檻（沿用2026-08-10找回的歷史規則，特徵百分位映射自城中GA自身'
    '分布，未用報酬資訊最佳化）：net_amt_wan≥1,705.13萬 且 influence_pct≥2.70%。'
    '資料範圍2018-01-02～2026-08-07，2026-08-11整理。'
)
st.info(
    '⚠️ **結算慣例與2026-08-10復原報告的差異，已揭露**：本頁採用**單純D1開盤放空/D1收盤回補**'
    '結算（與pages/8、門檻比較系列現行慣例一致），不是舊報告記載的「D1若open==close則往後找第一個'
    '可成交日回補（最多20個交易日）」展延式結算。因此本頁彙總數字（勝率/平均報酬）會與舊報告的'
    '數字略有出入，屬於預期且已揭露的差異，不是資料錯誤——重建時已用舊報告的稽核數字'
    '（母體8,757筆/候選4,087筆）做過sanity check，本次重建得到母體8,954筆/候選4,235筆，量級一致、'
    '判定事件選取邏輯正確。完整說明見`E:\\stock\\reports\\fubon_branch_events_build_20260811.md`。'
)
st.caption('提醒：2026-05-06之後的資料因累積時間較短，筆數較少，統計結果雜訊會比較大，解讀時請留意。')

events = load_events()
events['censored'] = events['censored'].astype(bool)
events['success'] = events['success'].astype(bool)
events['d0_disposal'] = events['d0_disposal'].astype(bool)
events['d1_disposal'] = events['d1_disposal'].astype(bool)
if 'has_intraday' not in events.columns:
    events['has_intraday'] = False
events['has_intraday'] = events['has_intraday'].astype(bool)
events['年份'] = events['d0'].dt.year.astype(str)

# ── 篩選器 ──────────────────────────────────────────────
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    sel_years = st.multiselect('年份', sorted(events['年份'].unique().tolist(), reverse=True),
                                default=sorted(events['年份'].unique().tolist(), reverse=True))
with col_f2:
    sel_market = st.multiselect('市場', sorted(events['market'].dropna().unique().tolist()), default=sorted(events['market'].dropna().unique().tolist()))
with col_f3:
    gap_bins = ['<-5%', '-5~-2%', '-2~0%', '0~1%', '1~3%', '3~5%', '5~7%', '7~9%', '9~9.5%', '≥9.5%(避開)']
    sel_gap = st.multiselect('D1開盤跳空區間', gap_bins, default=gap_bins)
with col_f4:
    sel_streak = st.multiselect('連續鎖漲停天數', [1, 2, 3, 4], default=[1, 2, 3, 4])

st.caption(
    '⚠️ D1落在處置分盤集合競價期間的事件一律排除，不提供切換選項：分盤集合競價沒有連續撮合，'
    '等於無法照這個策略的方式實際執行，不是單純「表現比較差」的統計選擇（沿用城中GA頁已驗證的'
    '同一機制性理由；富邦分點策略本身尚未針對處置期間污染單獨驗證，此為保守預設）。'
)
st.caption(
    '⚠️ D1當天若命中本地已知的「停止先賣後買」(day-trade-short)限制，一律排除，同樣不提供切換選項：'
    '當沖放空這個動作本身在該日可能無法合法執行，屬於能不能做這筆交易的問題，不是報酬好壞的統計選擇。'
    '2026-08-12稽核(`borrow_availability_risk_20260812.md`)：符合此旗標的事件佔全樣本3.66%(155/4,233)。'
    '**注意**：完整的「借券/融券額度是否足夠」尚無法驗證——本機資料庫目前沒有可借券賣出股數、借券限額、'
    '借券費率等即時額度資料，只有這個已知的當沖限制旗標可用，不代表其餘事件借券一定沒問題，只是本頁能排除的部分。'
)

col_f7, col_f8 = st.columns(2)
with col_f7:
    mktcap_max_default = int(events['mktcap_billion'].max()) + 100 if pd.notna(events['mktcap_billion'].max()) else 5000
    mktcap_max = st.slider(
        'D0市值上限（億元）', min_value=0, max_value=mktcap_max_default,
        value=mktcap_max_default, step=50,
        help='本策略尚未像城中GA一樣單獨驗證市值切分的最佳門檻，僅供互動探索（三策略因子篩選council '
             '曾指出富邦候選在市值Q4組較弱，但未换算成具體門檻值）。預設不過濾，自行拖動查看差異。'
    )
with col_f8:
    taiex_mode = st.radio(
        '大盤(加權指數)D0前2日累積漲幅',
        ['全部納入', '排除過熱天(≥2.6%)', '只看過熱天(≥2.6%)'],
        index=0,
        horizontal=True,
        help='沿用城中GA頁驗證過的2.6%門檻定義，但本策略尚未針對此因子單獨驗證，預設不過濾，'
             '僅供互動探索。'
    )

exclude_target_v1 = st.checkbox(
    '排除當日同時符合法人目標價V1候選的事件',
    value=False,
    key='exclude_target_v1_16',
)
st.caption(
    '探索性工具：前一輪全母體測試（`target_price_interaction_20260813.md`）發現，'
    '法人目標價 V1 候選重疊對城中GA或富邦的放空報酬都**沒有** holdout-confirmed 的驗證效果；'
    '此切換是依 Kevin 要求提供查閱，不是已證實的篩選條件。D1-exact 對齊。'
)


def gap_bucket(g):
    if g < -5:
        return '<-5%'
    if g < -2:
        return '-5~-2%'
    if g < 0:
        return '-2~0%'
    if g < 1:
        return '0~1%'
    if g < 3:
        return '1~3%'
    if g < 5:
        return '3~5%'
    if g < 7:
        return '5~7%'
    if g < 9:
        return '7~9%'
    if g < 9.5:
        return '9~9.5%'
    return '≥9.5%(避開)'


events['gap_bucket'] = events['gap_pct'].apply(gap_bucket)
events['streak_capped'] = events['lock_streak'].clip(upper=4)

base_mask = (
    events['market'].isin(sel_market)
    & events['gap_bucket'].isin(sel_gap)
    & events['streak_capped'].isin(sel_streak)
)
base_mask &= events['年份'].isin(sel_years)
base_mask &= ~events['d1_disposal']
base_mask &= ~events['day_trade_short_suspended_d1']
base_mask &= events['mktcap_billion'].isna() | (events['mktcap_billion'] <= mktcap_max)
if taiex_mode == '排除過熱天(≥2.6%)':
    base_mask &= events['taiex_2day_mom_pct'].isna() | (events['taiex_2day_mom_pct'] <= 2.6)
elif taiex_mode == '只看過熱天(≥2.6%)':
    base_mask &= events['taiex_2day_mom_pct'] > 2.6

if exclude_target_v1:
    mask = base_mask & ~events['v1_candidate_d1']
else:
    mask = base_mask
    # Default-off must retain the exact pre-toggle population.
    if not events.index[mask].equals(events.index[base_mask]):
        raise RuntimeError('V1 排除切換預設關閉時改變了富邦事件母體。')

view = events[mask].copy()

st.divider()

# ── 停損/停利設定（套用到下方KPI與所有統計）──────────────────
st.markdown('#### 🎚️ 停損／停利設定')
st.caption(
    '設定「D1股價比D0收盤漲多少%停損」「D1股價比D0收盤跌多少%停利」（標準的漲跌幅%報價方式），用D1當天的最高價/最低價估算是否會被觸及（並非逐筆tick，未計入手續費/滑價）。'
    '若同一天停損價與停利價都被觸及，保守假設停損先發生。D1當天整日無成交（開盤=最高=最低=收盤，實際上無法回補）的事件視為censored，直接排除於統計之外，不做延伸到解鎖日開盤的回補假設。'
    '**下方KPI已套用這裡的設定**。'
    '⚠️ 本專案先前針對城中GA/統一城中/富邦三個已驗證策略做過42組（7種停損%×6種停利%）網格掃描，'
    '樣本內選出最佳組合後在完全沒碰過的樣本外資料驗證（見`exit_grid_scan_with_risk_20260810.md`）：'
    '**拿掉停損（漲多少%就回補）這件事，三個策略樣本外都確定是對的**——加停損不只犧牲平均報酬，連風險'
    '(最大回撤)都一起變差，所以下面預設停損=0%。**停利（跌多少%提早回補）則因策略而異**：富邦是唯一一個'
    '停利=7%在樣本外真的贏過單純持有到收盤的（樣本外平均+0.92% vs +0.79%，回撤也更小），所以這裡預設幫你'
    '設成7%；城中GA/統一城中做同樣測試則沒有穩定的樣本外證據支持停利有幫助，那兩頁預設維持0%（單純持有到收盤）。'
    '你隨時可以自行拖動滑桿覆蓋這個預設值。'
)
if st.button('🎯 套用目前研究建議的最佳參數', key='apply_recommended_16'):
    st.session_state['stop_16_slider'] = 0.0
    st.session_state['stop_16_num'] = 0.0
    st.session_state['tp_16_slider'] = 7.0
    st.session_state['tp_16_num'] = 7.0
    st.rerun()
st.caption('研究建議依據：`exit_grid_scan_with_risk_20260810.md`（富邦樣本外：停損 0%、停利 7%）。')
col_s1, col_s2 = st.columns(2)
with col_s1:
    stop_pct = synced_pct_input('停損：D1股價比D0收盤漲多少% 出場（0=不停損）', 0.0, 'stop_16')
with col_s2:
    tp_pct = synced_pct_input('停利：D1股價比D0收盤跌多少% 出場（0=不停利，網格掃描建議值7%）', 7.0, 'tp_16')

entry = view['d1_open'].to_numpy(dtype=float)
d0_close_arr = view['d0_close'].to_numpy(dtype=float)
close_arr = view['d1_close'].to_numpy(dtype=float)
high_arr = view['d1_high'].to_numpy(dtype=float)
low_arr = view['d1_low'].to_numpy(dtype=float)
frozen_arr = view['d1_frozen'].to_numpy(dtype=bool)
censored_arr = view['censored'].to_numpy(dtype=bool)
base_ret_arr = view['short_ret_open_to_close_pct'].to_numpy(dtype=float)

with np.errstate(invalid='ignore'):
    stop_price = d0_close_arr * (1 + stop_pct / 100)
    tp_price = d0_close_arr * (1 - tp_pct / 100)
    hit_stop = (stop_pct > 0) & (high_arr >= stop_price)
    hit_tp = (tp_pct > 0) & (low_arr <= tp_price)

    exit_price = close_arr.copy()
    exit_price = np.where(hit_tp, tp_price, exit_price)
    exit_price = np.where(hit_stop, stop_price, exit_price)  # 停損優先，覆蓋停利

    sim_ret = (entry - exit_price) / entry * 100
    sim_ret = np.where(frozen_arr, base_ret_arr, sim_ret)
    sim_ret = np.where(censored_arr | np.isnan(entry), np.nan, sim_ret)

view['sim_ret'] = sim_ret

st.divider()

# ── KPI（比照站台首頁app.py「歷史回測紀錄」頁的同一套指標；套用上方停損/停利設定）──
settled = view[~view['censored']]
rets = settled['sim_ret'].dropna()
if len(rets) == 0:
    st.warning('目前篩選條件下無已結算資料')
else:
    sharpe, pf = sharpe_pf(rets)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    frozen_rate = view['d1_frozen'].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('總筆數', f'{len(view)} 筆', delta=f'{len(rets)} 已結算', delta_color='off')
    c2.metric('勝率', f'{(rets > 0).mean() * 100:.2f}%')
    c3.metric('期望報酬', f'{rets.mean():+.2f}%')
    c4.metric('夏普值', f'{sharpe:.2f}' if pd.notna(sharpe) else '-')
    c5.metric('賺賠比', f'{pf:.2f}' if pd.notna(pf) else '-')

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric('獲利筆', len(wins))
    d2.metric('虧損筆', len(losses))
    d3.metric('均獲利', f'{wins.mean():+.2f}%' if len(wins) else '-')
    d4.metric('均虧損', f'{losses.mean():+.2f}%' if len(losses) else '-')
    d5.metric('最大虧損', f'{rets.min():+.2f}%' if len(rets) else '-')
    st.caption(f'D1鎖死率：{frozen_rate:.2f}%（D1當天直接鎖漲停、實際上無法建倉的比例，已排除在上述統計外）')
    if stop_pct > 0 or tp_pct > 0:
        st.caption(f'⚙️ 以上KPI已套用停損{stop_pct:.1f}% / 停利{tp_pct:.1f}%（非單純持有到收盤）；逐筆明細另列相同情境的放空報酬，原始報酬也一併保留供對照。')

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
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 2),
            '最慘單筆(%)': round(sub['short_ret_open_to_close_pct'].min(), 2),
            'Sharpe(近似)': round(sh, 2) if pd.notna(sh) else None,
            'D1鎖死率(%)': round(view[view['gap_bucket'] == g]['d1_frozen'].mean() * 100, 2),
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
                {'勝率(%)': '{:.2f}', '平均報酬(%)': '{:+.2f}', '最慘單筆(%)': '{:+.2f}',
                 'Sharpe(近似)': '{:.2f}', 'D1鎖死率(%)': '{:.2f}'}, na_rep='-'
            ),
            use_container_width=True,
        )
        st.caption('跳空區間分組為描述性統計，本頁尚未像城中GA頁一樣針對此因子做過9.5%懸崖式的獨立驗證，僅供互動探索參考。')

# ── 停損/停利效果對照（沿用上方同一組滑桿，不重複設定）──────
with st.expander(f'🎚️ 停損／停利效果對照（目前設定：停損{stop_pct:.1f}% / 停利{tp_pct:.1f}%，與上方KPI區塊同一組設定）'):
    sim_settled = view[~view['censored']]
    if len(sim_settled) == 0:
        st.warning('目前篩選條件下無已結算資料')
    else:
        base_wr = sim_settled['success'].mean() * 100
        base_avg = sim_settled['short_ret_open_to_close_pct'].mean()
        base_sh, _ = sharpe_pf(sim_settled['short_ret_open_to_close_pct'])
        base_worst = sim_settled['short_ret_open_to_close_pct'].min()

        sim_rets_here = sim_settled['sim_ret'].dropna()
        sim_wr = (sim_rets_here > 0).mean() * 100
        sim_avg = sim_rets_here.mean()
        sim_sh, _ = sharpe_pf(sim_rets_here)
        sim_worst = sim_rets_here.min() if len(sim_rets_here) else np.nan

        cmp_df = pd.DataFrame({
            '原本（持有到收盤）': [f'{base_wr:.2f}%', f'{base_avg:+.2f}%',
                                    f'{base_sh:.2f}' if pd.notna(base_sh) else '-', f'{base_worst:+.2f}%'],
            '套用停損/停利後（=上方KPI）': [f'{sim_wr:.2f}%', f'{sim_avg:+.2f}%',
                          f'{sim_sh:.2f}' if pd.notna(sim_sh) else '-',
                          f'{sim_worst:+.2f}%' if pd.notna(sim_worst) else '-'],
        }, index=['勝率', '平均報酬', 'Sharpe(近似)', '最慘單筆'])
        st.dataframe(cmp_df, use_container_width=True)
        st.caption(f'已結算 {len(sim_settled)} 筆（D1整日無成交的censored事件已排除）。要調整停損/停利數值，請拖動上方「停損／停利設定」的滑桿——這裡跟頂部KPI共用同一組設定，不是獨立的模擬。')

# ── lock_streak 分組表 ──────────────────────────────────
with st.expander('📊 依「連續鎖漲停天數」分組'):
    rows = []
    for s in [1, 2, 3, 4]:
        sub = view[(view['streak_capped'] == s) & (~view['censored'])]
        if len(sub) < 5:
            continue
        rows.append({
            'lock_streak': f'{s}{"+" if s == 4 else ""}',
            '筆數': len(sub),
            '勝率(%)': round(sub['success'].mean() * 100, 2),
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 2),
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index('lock_streak').style.format(
                {'勝率(%)': '{:.2f}', '平均報酬(%)': '{:+.2f}'}
            ),
            use_container_width=True,
        )
    st.caption('連續鎖漲停天數與報酬的關係，本頁尚未針對此策略單獨驗證方向性，僅供互動探索（不要直接套用城中GA頁「越多越好」的結論，兩者是不同分點/不同規則）。')

# ── 逐年穩定性 ──────────────────────────────────────────
with st.expander('📅 逐年穩定性（目前篩選條件下）'):
    yr_rows = []
    for yr, sub in view[~view['censored']].groupby(view['d0'].dt.year):
        if len(sub) < 5:
            continue
        sh, _ = sharpe_pf(sub['short_ret_open_to_close_pct'])
        t_yr = (
            sub['short_ret_open_to_close_pct'].mean()
            / (sub['short_ret_open_to_close_pct'].std(ddof=1) / np.sqrt(len(sub)))
            if len(sub) > 1 else np.nan
        )
        yr_rows.append({
            '年份': yr, '筆數': len(sub),
            '勝率(%)': round(sub['success'].mean() * 100, 2),
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 2),
            't值': round(t_yr, 2) if pd.notna(t_yr) else None,
            'Sharpe(近似)': round(sh, 2) if pd.notna(sh) else None,
        })
    if yr_rows:
        st.dataframe(
            pd.DataFrame(yr_rows).set_index('年份').style.format(
                {'勝率(%)': '{:.2f}', '平均報酬(%)': '{:+.2f}', 't值': '{:.2f}', 'Sharpe(近似)': '{:.2f}'},
                na_rep='-',
            ),
            use_container_width=True,
        )

# ── 完整逐筆明細（主表）──────────────────────────────────
st.subheader(f'📋 完整逐筆歷史紀錄（{len(view)} 筆，依目前篩選條件）')
st.caption(
    '「其他已發現分點」均為同一個 D0、同一檔股票的本機分點成交快取批次計算；'
    '淨買超為萬元、影響力依 D0 成交金額計算。沒有該分點活動時顯示 0。'
)

show_cols = ['d0', 'code', 'name', 'market', 'd1', 'gap_pct', 'lock_streak', 'net_amt_wan',
             'influence_pct', 'city_ga_net_amt_wan', 'city_ga_influence_pct',
             'unicenter_city_net_amt_wan', 'unicenter_city_influence_pct',
             'taishin_taipei_net_amt_wan', 'taishin_taipei_influence_pct',
             'd0_top3_net_buy_branches', 'v1_candidate_d1',
             'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_frozen',
             'censored', 'short_ret_open_to_close_pct', 'sim_ret', 'short_mae_pct', 'success']
show = view[show_cols].sort_values('d0', ascending=False).copy()
show['d0'] = show['d0'].dt.strftime('%Y-%m-%d')
show['d1'] = show['d1'].dt.strftime('%Y-%m-%d')
show.columns = ['D0訊號日', '代號', '名稱', '市場', 'D1進場日', '跳空%', '連鎖天數', '買超金額(萬)',
                  '影響力%', '城中GA淨買超(萬)', '城中GA影響力%', '統一城中淨買超(萬)', '統一城中影響力%',
                  '台新台北淨買超(萬)', '台新台北影響力%', 'D0前三大買超分點', 'D1同時為V1候選',
                  '開盤', '最高', '最低', '收盤', 'D1鎖死', '截尾', '原始放空報酬%', '情境放空報酬%',
                  '最大不利波動%', '成功']
show['D1走勢'] = view.loc[show.index, 'd1_intraday_spark'].tolist()
intraday_coverage = view['has_intraday'].mean() * 100 if len(view) else 0.0


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
        .map(color_ret, subset=['情境放空報酬%'])
        .format({'跳空%': '{:+.2f}', '買超金額(萬)': '{:,.2f}', '影響力%': '{:.2f}',
                 '城中GA淨買超(萬)': '{:,.2f}', '城中GA影響力%': '{:+.2f}',
                 '統一城中淨買超(萬)': '{:,.2f}', '統一城中影響力%': '{:+.2f}',
                 '台新台北淨買超(萬)': '{:,.2f}', '台新台北影響力%': '{:+.2f}',
                  '開盤': '{:.2f}', '最高': '{:.2f}', '最低': '{:.2f}', '收盤': '{:.2f}',
                  '原始放空報酬%': '{:+.2f}', '情境放空報酬%': '{:+.2f}',
                  '最大不利波動%': '{:.2f}'}, na_rep='-'),
    use_container_width=True, height=520,
    column_config={
        'D1走勢': st.column_config.LineChartColumn(
            'D1走勢（分K收盤）', width='medium',
            help='D1當天每分鐘收盤價走勢（真實Shioaji歷史分K，2026-08-11回填）；沒有抓到分K資料的事件留空。',
        ),
        'D0前三大買超分點': st.column_config.TextColumn(
            'D0前三大買超分點', width='large',
            help='同一 D0／同一股票的全部正淨買分點，依淨買金額排序；不是只限本頁四個訊號分點。',
        ),
    },
)
st.caption(f'開盤/最高/最低/收盤為D1當天日線價。「D1走勢」為D1當天逐分鐘收盤價，涵蓋率約{intraday_coverage:.2f}%（真實歷史分K，非模擬；缺資料事件留空，非100%覆蓋）。')
st.download_button(
    '📥 下載此表 CSV',
    show.drop(columns=['D1走勢']).to_csv(index=False, encoding='utf-8-sig'),
    'fubon_branch_events_filtered.csv', 'text/csv', key='dl_events_full',
)

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
**統一-城中訊號重新驗證（VM實際上線規則）**
- 規則：任何正買超（無金額門檻）＋嚴格鎖漲停＋D1觸及停損天花板（D0收盤算出的隔日漲停×0.98）出場，
  否則D1收盤出場。這是VM上`unicenter_live.py`實際使用的邏輯，不是本頁的富邦裸名稱規則。
- 對照凱基-台北：2026-05-06前 n=455，平均報酬差+0.59%，p<0.0001；2026-05-06起(近期) n=284，
  平均報酬差+1.17%，p<0.0001（未做多重比較校正）。兩個期間方向一致、較長期間顯著，是目前VM上
  實際部署（模擬單）的規則。
- 完整逐筆pairwise組合事件資料（統一-城中×富邦(裸)、統一×統一-城中、統一-城中×凱基-城中）
  請見 **pages/15「分點組合訊號逐筆紀錄」**，本頁不重複列出這些數字，避免兩頁資料互相打架。

**元大證券：觀察清單，暫不建議部署**
- 兩輪深度搜尋（含D槽原始備份）都找不到歷史上真實使用過的元大獨立分點規則。
- 現做的新建構規則（比照富邦方法論，用城中GA百分位映射門檻，非報酬最佳化）結果明顯偏弱且持續
  惡化：2018-2023期間 t=11.50 → 2024-2026期間 t=2.88 → 2026-05起(近期，近3個月) t=1.97
  （臨界顯著）；2025整年顯著虧損。
- 狀態：**觀察清單（pocket list），暫不建議部署**——規則本身無歷史依據、近年在惡化、對出場機制
  敏感度高，與本頁富邦規則（有找回的歷史依據、各期間方向一致）性質不同，不應混為一談。
""")
with col2:
    st.markdown("""
**分點層級補充脈絡**
- 富邦證券也複製了城中GA「D1自己淨賣才準」的機制（D1淨賣勝率明顯高於D1淨買），證實此現象非城中
  GA獨有。
- 富邦證券獨立分點策略與城中GA策略近期(2026年，D0>=2026-05-06起算的近期窗口)相關性明顯上升
  （全期Pearson相關約0.16，近期跳升到約0.53，同天同時有部位比例從全期23.6%跳到近期89.2%）——
  在做資金配置/部位規模決策時，不建議把兩者當成互相獨立的部位處理。

**目前狀態**
- 本頁富邦規則：**尚未部署到任何實盤/模擬單**，屬於研究階段結果（各期間方向一致、皆顯著為正，
  是四個「目前有效」分點/策略之一，但尚未上線）。
- 若要重跑驗證：建置腳本`E:\\stock\\scripts\\run_build_fubon_branch_events.py`，方法論說明
  `E:\\stock\\reports\\fubon_branch_events_build_20260811.md`。
""")

st.caption('完整方法論見本機 memory 系統 project_隔日沖分點戰術.md，以及 E:\\stock\\reports\\ 下相關報告。')
