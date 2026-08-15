# -*- coding: utf-8 -*-
"""凱基-城中GA 隔日沖放空策略：歷史回測紀錄與因子挖掘總覽。"""
import json
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='城中GA研究總覽', page_icon='🏙️', layout='wide')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ADDON_DATA_FILE = os.path.join(DATA_DIR, 'citycenter_ga_0915_addon_events.csv')
ADDON_TOGGLE_KEY = 'city_0915_addon_enabled'
CITY_GAP_EXECUTABLE_MAX = 9.5
CITY_GAP_VALIDATED_MIN = 5.024636961142823
GAP_MODE_ALL_EXECUTABLE = 'All executable gaps (<9.5%)'
GAP_MODE_VALIDATED = 'Validated gap: 5.02% to <9.5%'
GAP_MODE_HELP = '2026-08-12 City-GA study: IS-selected D1 gap >=5.02% and <9.5%; untouched holdout n=78, mean +2.34%, +1.31pp vs fixed remaining groups, Welch p=0.047. Gaps >=9.5% are mechanically excluded as frequently locked/unfillable.'


def sharpe_pf(returns):
    r = returns.dropna()
    if len(r) < 2:
        return np.nan, np.nan
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std(ddof=1) else np.nan
    gains = r[r > 0].sum()
    losses = -r[r <= 0].sum()
    pf = gains / losses if losses > 0 else np.nan
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
    fp = os.path.join(DATA_DIR, 'citycenter_ga_events.csv')
    df = pd.read_csv(fp, parse_dates=['d0', 'd1'])
    df['code'] = df['code'].astype(str)
    names_fp = os.path.join(DATA_DIR, 'stock_names.csv')
    names = pd.read_csv(names_fp, dtype={'code': str})
    df = df.merge(names, on='code', how='left')
    df['name'] = df['name'].fillna('')
    # 09:15 加碼資料由 scripts/build_citycenter_ga_0915_addon_data.py 從真實分點成交
    # 明細與真實分鐘 K 線逐筆 join 產生；Cloud 只讀這個小型、可版本控管的產物。
    addon_columns = [
        'city_d0_buy_rank_top3', 'unicenter_city_d0_cobuy', 'fubon_chiayi_d0_cobuy',
        'bonus_signal_count', 'd1_0915_close', 'has_d1_0915_close', 'addon_eligible',
        'addon_ret_0915_to_close_pct',
    ]
    if os.path.exists(ADDON_DATA_FILE):
        addon = pd.read_csv(ADDON_DATA_FILE, parse_dates=['d0', 'd1'], dtype={'code': str})
        if addon.duplicated(['d0', 'code', 'd1']).any():
            raise ValueError('citycenter_ga_0915_addon_events.csv 有重複事件鍵，拒絕混入 KPI')
        df = df.merge(addon, on=['d0', 'code', 'd1'], how='left', validate='one_to_one')
    else:
        # 防禦舊部署：缺少資料時不會虛構訊號，toggle 也會在頁面下方停用。
        for col in addon_columns:
            df[col] = np.nan
    for col in ['city_d0_buy_rank_top3', 'unicenter_city_d0_cobuy', 'fubon_chiayi_d0_cobuy',
                'has_d1_0915_close', 'addon_eligible']:
        df[col] = df[col].fillna(False).astype(bool)
    if 'bonus_signal_count' not in df.columns:
        df['bonus_signal_count'] = 0
    df['bonus_signal_count'] = df['bonus_signal_count'].fillna(0).astype(int)
    if 'd1_intraday_close' not in df.columns:
        df['d1_intraday_close'] = ''
    df['d1_intraday_spark'] = df['d1_intraday_close'].apply(_parse_spark)
    # 2026-08-07新增: 防禦性欄位補齊——Streamlit Cloud曾在CSV已更新後仍持續回報
    # KeyError('d1_disposal')，懷疑是舊worker的cache_data在redeploy後沒有確實失效。
    # 除了下面靠_cache_bust參數強制失效既有cache外，這裡再加一層防禦：即使真的讀到
    # 沒有這三欄的舊CSV，也用安全預設值補上，讓頁面不會直接crash。
    for col, default in [('d0_disposal', False), ('d1_disposal', False), ('d1_disposal_type', ''),
                          ('mktcap_billion', float('nan')), ('mktcap_pct_rank', float('nan')),
                          ('taiex_2day_mom_pct', float('nan')), ('day_trade_short_suspended_d1', False)]:
        if col not in df.columns:
            df[col] = default
    df['mktcap_pct_rank'] = pd.to_numeric(df['mktcap_pct_rank'], errors='coerce')
    df['day_trade_short_suspended_d1'] = df['day_trade_short_suspended_d1'].fillna(False).astype(bool)
    # This context is built offline into the committed CSV.  Older snapshots
    # remain deployable with inert defaults instead of triggering local reads.
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


st.title('🏙️ 凱基-城中GA 隔日沖放空策略｜歷史回測紀錄')
st.caption('D0嚴格鎖漲停 + 凱基-城中淨買超金額/影響力雙門檻達標 → D1開盤放空、收盤回補。現行門檻（2026-08-10更新，較舊門檻更嚴格）：net_amt_wan≥458.69萬 且 cz_influence_pct≥1.93%。資料範圍2018-01-09～2026-08-07，2026-08-11整理；結算採D1開盤放空/收盤回補（非延伸多日回補），與最新門檻比較報告一致。即時候選/模擬單請見「城中GA放空策略」頁。')

events = load_events()
events['censored'] = events['censored'].astype(bool)
events['success'] = events['success'].astype(bool)
if 'has_intraday' not in events.columns:
    events['has_intraday'] = False
events['has_intraday'] = events['has_intraday'].astype(bool)
events['年份'] = events['d0'].dt.year.astype(str)
if ADDON_TOGGLE_KEY not in st.session_state:
    # 以 2025 作樣本內、2026 作保留樣本的同一事件重跑皆顯示：等權加碼會稀釋報酬，預設不加碼。
    st.session_state[ADDON_TOGGLE_KEY] = False
MKT_CAP_QUINTILES = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
DEFAULT_MKT_CAP_QUINTILES = ['Q1', 'Q2', 'Q3', 'Q4']
if 'mktcap_quintiles_8' not in st.session_state:
    st.session_state['mktcap_quintiles_8'] = DEFAULT_MKT_CAP_QUINTILES.copy()
if 'gap_mode_8' not in st.session_state:
    st.session_state['gap_mode_8'] = GAP_MODE_ALL_EXECUTABLE
if 'taiex_mode_8' not in st.session_state:
    st.session_state['taiex_mode_8'] = '排除過熱天(建議)'

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

gap_mode = st.radio(
    'D1 gap research filter',
    [GAP_MODE_ALL_EXECUTABLE, GAP_MODE_VALIDATED],
    key='gap_mode_8',
    horizontal=True,
    help=GAP_MODE_HELP,
)

st.caption('⚠️ 處置分盤期間、當沖限制的事件一律硬性排除（無法切換）——理由見下方展開。')
with st.expander('📖 硬性排除規則說明', expanded=False):
    st.caption(
        '⚠️ D1落在處置分盤集合競價期間的事件一律排除，不提供切換選項：分盤集合競價沒有連續撮合，'
        '現行盤中監控/09:15加碼/停損邏輯在這段期間不成立，等於無法照這個策略的方式實際執行，'
        '不是單純「表現比較差」的統計選擇。2026-08-07驗證(CODEX_處置期間污染評估_REPORT.md)：'
        '這類事件近期regime平均報酬-1.53%，明顯劣於正常組+0.85%(Welch p<0.01)，方向也支持排除。'
    )
    st.caption(
        '⚠️ D1當天若命中本地已知的「停止先賣後買」(day-trade-short)限制，一律排除，同樣不提供切換選項：'
        '這代表當沖放空這個動作本身在該日可能無法合法執行，屬於能不能做這筆交易的問題，不是報酬好壞的統計選擇。'
        '2026-08-12稽核(`borrow_availability_risk_20260812.md`)：符合此旗標的事件佔全樣本3.25%(50/1,540)。'
        '**注意**：完整的「借券/融券額度是否足夠」尚無法驗證——本機資料庫目前沒有可借券賣出股數、借券限額、'
        '借券費率等即時額度資料，只有這個已知的當沖限制旗標可用，不代表其餘事件借券一定沒問題，只是本頁能排除的部分。'
    )

def apply_recommended_parameters():
    st.session_state['mktcap_quintiles_8'] = DEFAULT_MKT_CAP_QUINTILES.copy()
    st.session_state['taiex_mode_8'] = '排除過熱天(建議)'
    st.session_state['stop_8_slider'] = 0.0
    st.session_state['stop_8_num'] = 0.0
    st.session_state['tp_8_slider'] = 0.0
    st.session_state['tp_8_num'] = 0.0
    st.session_state[ADDON_TOGGLE_KEY] = False
    st.session_state['gap_mode_8'] = GAP_MODE_VALIDATED


st.button('🎯 套用目前研究建議的最佳參數', key='apply_recommended_8', on_click=apply_recommended_parameters)
st.caption('研究建議依據：`exit_grid_scan_with_risk_20260810.md`、`CODEX_0915加碼規則正式驗證_REPORT.md`，並以本頁現行事件資料重跑 2025 樣本內／2026 保留樣本。')

col_f7, col_f8 = st.columns(2)
with col_f7:
    sel_mktcap_quintiles = st.multiselect(
        'D0市值百分位分組', MKT_CAP_QUINTILES,
        key='mktcap_quintiles_8',
        help='逐筆以 D0 當日全市場可交易股票的市值百分位分成五組：Q1 最小、Q5 最大。'
    )
    st.caption(
        'Q5 的固定邊界是當日市值第 80.00 百分位；預設排除 Q5。2026-08-11 研究的 IS Q1–Q4 '
        '切點為 83.08%，接近但不等同於 80.00%；其 holdout 相對優勢 p=0.609，未證明回測較佳。'
        '改用百分位／分組是為避免固定金額隨市場成長而漂移，不是績效宣稱。'
    )
with col_f8:
    taiex_mode = st.radio(
        '大盤(加權指數)D0前2日累積漲幅',
        ['排除過熱天(建議)', '全部納入', '只看過熱天'],
        key='taiex_mode_8',
        horizontal=True,
        help='2026-08-07驗證：D0收盤時往前2個交易日的加權指數累積漲幅按五分位切分，'
             '最強20%那組近期regime優勢幾乎消失甚至轉負(mean=-0.05%,t=-0.15)，對照最強'
             'Q2組(+1.51%,t=4.71)。門檻2.6%，取全期(2.35%)與近期regime(2.69%)80百分位'
             '之間。現行live候選腳本已用同一門檻直接停發當天全部候選。'
    )

exclude_target_v1 = st.checkbox(
    '排除當日同時符合法人目標價V1候選的事件',
    value=False,
    key='exclude_target_v1_8',
)
st.caption(
    '探索性工具：法人目標價 V1 候選重疊沒有 holdout-confirmed 的驗證效果；'
    '此切換僅供查閱，使用 D1-exact 對齊。'
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


def mktcap_quintile(percentile):
    """Q1 is the smallest 20% of the same-D0 investable universe."""
    if not np.isfinite(percentile):
        return np.nan
    return f'Q{min(int(np.ceil(percentile / 20.0)), 5)}'


events['gap_bucket'] = events['gap_pct'].apply(gap_bucket)
events['streak_capped'] = events['lock_streak'].clip(upper=4)
events['mktcap_quintile'] = events['mktcap_pct_rank'].apply(mktcap_quintile)

base_mask = (
    events['年份'].isin(sel_years)
    & events['market'].isin(sel_market)
    & events['gap_bucket'].isin(sel_gap)
    & events['streak_capped'].isin(sel_streak)
)
# >=9.5% is an executability gate, not a return-picked weak bucket.
base_mask &= events['gap_pct'] < CITY_GAP_EXECUTABLE_MAX
if gap_mode == GAP_MODE_VALIDATED:
    base_mask &= events['gap_pct'] >= CITY_GAP_VALIDATED_MIN
base_mask &= ~events['d1_disposal']
base_mask &= ~events['day_trade_short_suspended_d1']
base_mask &= events['mktcap_quintile'].isin(sel_mktcap_quintiles)
if taiex_mode == '排除過熱天(建議)':
    base_mask &= events['taiex_2day_mom_pct'] <= 2.6
elif taiex_mode == '只看過熱天':
    base_mask &= events['taiex_2day_mom_pct'] > 2.6

if exclude_target_v1:
    mask = base_mask & ~events['v1_candidate_d1']
else:
    mask = base_mask
    if not events.index[mask].equals(events.index[base_mask]):
        raise RuntimeError('V1 排除切換預設關閉時改變了城中GA事件母體。')

view = events[mask].copy()

st.divider()

# ── 停損/停利設定（套用到下方KPI與所有統計）──────────────────
st.markdown('#### 🎚️ 停損／停利設定')
st.caption('下方KPI已套用這裡的設定（預設0%/0%＝單純持有到收盤，樣本外驗證最紮實的做法）。')
with st.expander('📖 設定說明與樣本外驗證依據', expanded=False):
    st.caption(
        '設定「D1股價比D0收盤漲多少%停損」「D1股價比D0收盤跌多少%停利」（標準的漲跌幅%報價方式），用D1當天的最高價/最低價估算是否會被觸及（並非逐筆tick，未計入手續費/滑價）。'
        '若同一天停損價與停利價都被觸及，保守假設停損先發生。D1當天整日無成交（開盤=最高=最低=收盤，實際上無法回補）的事件視為censored，直接排除於統計之外，不做延伸到解鎖日開盤的回補假設。'
        '⚠️ 本專案先前針對城中GA/統一城中/富邦三個已驗證策略做過42組（7種停損%×6種停利%）網格掃描，'
        '樣本內選出最佳組合後在完全沒碰過的樣本外資料驗證（見`E:\\stock\\reports\\exit_grid_scan_with_risk_20260810.md`）：'
        '**拿掉停損這件事，三個策略樣本外都確定是對的**（加停損不只犧牲平均報酬，連最大回撤都一起變差）；'
        '**但城中GA的停利，樣本外沒有穩定證據支持有幫助**（樣本外平均+1.32% vs 單純持有到收盤+1.34%，幾乎打平，'
        '跟富邦那頁停利=7%樣本外真的贏的情況不同），所以這裡預設維持0%/0%（單純持有到收盤，就是目前驗證最紮實的做法）。'
        '你隨時可以自行拖動滑桿覆蓋這個預設值做互動驗證。'
    )
col_s1, col_s2 = st.columns(2)
with col_s1:
    stop_pct = synced_pct_input('停損：D1股價比D0收盤漲多少% 出場（0=不停損）', 0.0, 'stop_8')
with col_s2:
    tp_pct = synced_pct_input('停利：D1股價比D0收盤跌多少% 出場（0=不停利）', 0.0, 'tp_8')

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
addon_eligible_arr = view['addon_eligible'].to_numpy(dtype=bool)
addon_ret_arr = view['addon_ret_0915_to_close_pct'].to_numpy(dtype=float)
with np.errstate(invalid='ignore'):
    view['sim_ret_with_0915_addon'] = np.where(
        addon_eligible_arr & np.isfinite(sim_ret) & np.isfinite(addon_ret_arr),
        (sim_ret + addon_ret_arr) / 2.0,
        sim_ret,
    )
addon_enabled = st.session_state[ADDON_TOGGLE_KEY]
view['detail_ret'] = view['sim_ret_with_0915_addon'] if addon_enabled else view['sim_ret']

st.divider()

# ── KPI（套用上方停損/停利設定；預設0/0時等同持有到收盤）──────
settled = view[~view['censored']]
if len(settled) == 0:
    st.warning('目前篩選條件下無已結算資料')
else:
    kpi_ret_col = 'sim_ret_with_0915_addon' if addon_enabled else 'sim_ret'
    sim_settled_ret = settled[kpi_ret_col].dropna()
    wr = (sim_settled_ret > 0).mean() * 100
    avg_r = sim_settled_ret.mean()
    sharpe, pf = sharpe_pf(sim_settled_ret)
    frozen_rate = view['d1_frozen'].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('已結算筆數', f'{len(settled)} 筆', delta=f'{len(view)} 篩選出', delta_color='off')
    c2.metric('勝率', f'{wr:.2f}%')
    c3.metric('平均放空報酬', f'{avg_r:+.2f}%')
    c4.metric('日組合Sharpe(近似)', f'{sharpe:.2f}' if pd.notna(sharpe) else '-')
    c5.metric('D1鎖死率', f'{frozen_rate:.2f}%', help='D1當天直接鎖漲停開不了倉的比例；≥9.5%跳空區間佔了81.4%的鎖死案例')
    if stop_pct > 0 or tp_pct > 0 or addon_enabled:
        kpi_note = f'停損{stop_pct:.1f}% / 停利{tp_pct:.1f}%'
        if addon_enabled:
            add_count = int((settled['addon_eligible'] & settled[kpi_ret_col].notna()).sum())
            kpi_note += f'；09:15加碼 {add_count} 筆（原始部位與加碼部位各半資金）'
        st.caption(f'⚙️ 以上KPI已套用{kpi_note}；逐筆明細另列相同情境的放空報酬，原始報酬也一併保留供對照。')

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
        st.caption('⭐ 9.5%以上是關鍵懸崖：D1鎖死率從9%以下的個位數暴增，最慘單筆也急遽惡化。這是目前找到最有效的防嘎漲停規則。跳空1~9%整體是甜蜜點，細分後可看到區間內部仍有差異；負跳空（開低）樣本數較少，僅供參考。')

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
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 2),
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index('lock_streak').style.format(
                {'勝率(%)': '{:.2f}', '平均報酬(%)': '{:+.2f}'}
            ),
            use_container_width=True,
        )
    st.caption('城中GA連續鎖漲停天數越多、隔日勝率越高（實際數字依上方篩選條件即時計算），跟FlipBranch「連鎖越多越差」剛好相反，不要混用兩策略的濾網邏輯。')

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
            '平均報酬(%)': round(sub['short_ret_open_to_close_pct'].mean(), 2),
            'Sharpe(近似)': round(sh, 2) if pd.notna(sh) else None,
        })
    if yr_rows:
        st.dataframe(
            pd.DataFrame(yr_rows).set_index('年份').style.format(
                {'勝率(%)': '{:.2f}', '平均報酬(%)': '{:+.2f}', 'Sharpe(近似)': '{:.2f}'}, na_rep='-'
            ),
            use_container_width=True,
        )

# ── 完整逐筆明細（主表，直接顯示，比照「歷史回測紀錄」頁樣式）──
st.subheader(f'📋 完整逐筆歷史紀錄（{len(view)} 筆，依目前篩選條件）')

show_cols = ['d0', 'code', 'name', 'market', 'd1', 'gap_pct', 'lock_streak', 'net_amt_wan',
             'cz_influence_pct', 'unicenter_city_net_amt_wan', 'unicenter_city_influence_pct',
             'fubon_net_amt_wan', 'fubon_influence_pct', 'taishin_taipei_net_amt_wan',
             'taishin_taipei_influence_pct', 'd0_top3_net_buy_branches', 'v1_candidate_d1',
             'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_frozen', 'censored',
             'short_ret_open_to_close_pct', 'detail_ret', 'short_mae_pct', 'success']
show = view[show_cols].sort_values('d0', ascending=False).copy()
show['d0'] = show['d0'].dt.strftime('%Y-%m-%d')
show['d1'] = show['d1'].dt.strftime('%Y-%m-%d')
show.columns = ['D0訊號日', '代號', '名稱', '市場', 'D1進場日', '跳空%', '連鎖天數', '買超金額(萬)',
                 '影響力%', '統一城中淨買超(萬)', '統一城中影響力%', '富邦證券淨買超(萬)', '富邦證券影響力%',
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
                 '統一城中淨買超(萬)': '{:,.2f}', '統一城中影響力%': '{:+.2f}',
                 '富邦證券淨買超(萬)': '{:,.2f}', '富邦證券影響力%': '{:+.2f}',
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
            help='同一 D0／同一股票的全部正淨買分點，依淨買金額排序；不是只限已知訊號分點。',
        ),
    },
)
st.caption(f'開盤/最高/最低/收盤為D1當天日線價。「D1走勢」為D1當天逐分鐘收盤價，涵蓋率約{intraday_coverage:.2f}%（真實歷史分K，非模擬；缺資料事件留空，非100%覆蓋）。')
st.download_button(
    '📥 下載此表 CSV',
    show.drop(columns=['D1走勢']).to_csv(index=False, encoding='utf-8-sig'),
    'citycenter_ga_events_filtered.csv', 'text/csv', key='dl_events_full',
)

st.divider()
st.markdown('**累積報酬走勢**（假設每筆等權重，依D0訊號日排序）')
cum_src = view[~view['censored']].sort_values('d0')['detail_ret'].dropna()
if len(cum_src) >= 2:
    cum = (cum_src / 100 + 1).cumprod() - 1
    cum.index = range(len(cum))
    st.line_chart(pd.DataFrame({'累積報酬(%)': (cum * 100).values}), height=250)
else:
    st.caption('目前篩選條件下已結算筆數不足，無法繪製累積報酬走勢。')

st.markdown('---')
st.subheader('⭐ 09:15加碼規則（2026-08-05驗證通過，唯一可用的checkpoint延伸規則）')
st.caption(
    'D0至少1個bonus訊號成立（城中D0買超排名前三 / 統一-城中D0共買 / 富邦-嘉義D0共買）'
    '且D1的09:15收盤價低於D1開盤價 → 用「額外資金」在09:15價位加開一筆新空單、抱到收盤，'
    '完全不動用/不稀釋原本D1開盤就進場的部位，純粹增量。僅限近期regime(2025-2026)，需持續累積樣本；'
    '台新台北同日共買樣本仍太小(2048筆裡僅32筆)，只作觀察不進正式規則。'
)
addon_data_coverage = events['has_d1_0915_close'].mean() * 100
addon_toggle_supported = events['has_d1_0915_close'].any()
st.checkbox(
    'KPI 模式：加碼（未勾選＝不加碼；原始／加碼部位各 50% 資金）',
    key=ADDON_TOGGLE_KEY,
    disabled=not addon_toggle_supported,
    help='僅當 bonus≥1 且真實 D1 09:15 收盤低於 D1 開盤時，才以 09:15 價放空並持有到收盤。原始開盤部位完全不變。',
)
if addon_toggle_supported:
    st.caption(
        f'KPI 加碼資料覆蓋 {addon_data_coverage:.2f}%（真實分鐘 K 線）；沒有精確 09:15 K 線的事件保留原始部位報酬，'
        '不把缺資料誤判成「條件不成立」。現行嚴格 City-GA 事件以 2025 作樣本內、2026 作保留樣本重跑，'
        '平均報酬：2025 不加碼 +2.14% vs 加碼 +1.45%；2026 不加碼 +1.65% vs 加碼 +0.69%，故研究建議預設「不加碼」。'
    )
else:
    st.warning('目前部署缺少可驗證的 09:15 分鐘 K 線資料，已停用 KPI 加碼，避免用推估資料模擬。')
bonus_rule_df = pd.DataFrame([
    {'bonus訊號數': '0（不該加碼）', 'n': 117, '加碼筆均報酬%': -0.493, '勝率%': 47.0, 't值': -1.155},
    {'bonus訊號數': '1個', 'n': 419, '加碼筆均報酬%': 0.494, '勝率%': 60.9, 't值': 2.173},
    {'bonus訊號數': '2個', 'n': 143, '加碼筆均報酬%': 1.105, '勝率%': 67.1, 't值': 2.886},
    {'bonus訊號數': '3個', 'n': 11, '加碼筆均報酬%': 3.037, '勝率%': 81.8, 't值': 4.565},
]).set_index('bonus訊號數')
st.dataframe(
    bonus_rule_df.style.format({'加碼筆均報酬%': '{:+.2f}', '勝率%': '{:.2f}', 't值': '{:.2f}'}),
    use_container_width=True,
)
st.caption('近期regime(2025-2026)，樣本數皆為n≥100之穩固樣本(3個bonus除外，n=11僅供參考)。此表為加碼腿本身的原始研究統計；上方 toggle 則以現行嚴格 City-GA 事件逐筆計算「原始腿／加碼腿」等權合併。資料來源：CODEX_0915加碼規則正式驗證_REPORT.md。')

st.markdown('---')
st.subheader('🔎 其他研究結論（非本頁資料表涵蓋範圍）')
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**承接／低接與做多方向**
- D1收盤守住D0收盤 → D1收→D2收顯著偏強（TRAIN+1.72%/p=0.02、TEST+0.70%/p<0.01），持股遇賣壓但收平/收紅不用因此恐慌減碼
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
- ML(LightGBM+SHAP)+GA系統性挖掘：TEST預測分數與實際報酬相關係數僅-0.01，現行兩門檻已是資料極限
- 技術性鎖漲停比例因子：Spearman相關幾乎是0，沒有支持證據
- 09:15檢查點拿來當「原本部位」的進場/出場觸發規則(延後進場/提早出場/分批建倉)全部判死，唯一驗證有效的是上方「額外加碼」規則，不可拿來改變原本開盤進場、收盤出場的做法
- 盤中停損/停利(2-8%)：全部測過，沒有一組贏過持有到收盤
- 承接力道做多策略(D1自己收紅→抱D2-D5)：判死，D2普遍負值，D4/D5轉正但t值不顯著、勝率未過45%
- 城中D0排名前三/占比：全期(2021-2026)判死(TRAIN/TEST方向相反)，但近期regime(2025-2026)顯著轉正(p<0.01)——城中規模近年快速成長，早期資料會污染判讀，這也是09:15加碼規則採用bonus訊號的依據
- ML+GA第二輪(納入排名/占比/checkpoint等新特徵)：GA找到的門檻TRAIN(2025)+1.76%漂亮但TEST(2026)崩到-0.07%，過擬合判死，維持09:15加碼的簡單規則即可

**2026-08-04即時案例**：2454表現最佳(+5.62%，無負向旗標)；2301最差(-7.92%，gap僅1.74%在甜蜜點內卻被外資巨量買盤蓋過)——證實即使規則都對仍有真實虧損可能，這是策略固有的殘餘風險。
""")

st.caption('完整方法論見 D:\\stock 下 memory 系統 project_隔日沖分點戰術.md 第66節，以及各 CODEX_*.md 報告。')
