"""
隔日沖回測  — 策略：F1+F2+F3 + 今日漲幅排名 + 隔日開盤賣（排除漲停）
資料由 overnight_bt/generate_trade_log.py 產生，每日收盤後重跑更新。
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
import altair as alt

st.set_page_config(
    page_title='隔日沖回測',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown("""
<style>
.metric-box {
    background: #1e2530; border-radius: 10px;
    padding: 14px 18px; margin: 4px; text-align: center;
}
.metric-val  { font-size: 1.8em; font-weight: 700; color: #4ade80; }
.metric-val2 { font-size: 1.8em; font-weight: 700; color: #f87171; }
.metric-lab  { font-size: 0.8em; color: #94a3b8; margin-top: 2px; }
.cand-card {
    background: #1e2530; border-radius: 8px;
    padding: 12px 16px; margin: 4px 0;
    border-left: 3px solid #4ade80;
}
.cand-stock { font-size: 1.1em; font-weight: 700; color: #f1f5f9; }
.cand-meta  { font-size: 0.85em; color: #94a3b8; margin-top: 4px; }
.pnl-box {
    background: #1a2535; border-radius: 10px;
    padding: 16px 20px; margin: 6px 0;
    border: 1px solid #2d3748;
}
.pnl-title { font-size: 0.85em; color: #94a3b8; }
.pnl-val   { font-size: 1.5em; font-weight: 700; color: #4ade80; margin-top: 4px; }
.up   { color: #f87171; font-weight: 600; }
.down { color: #4ade80; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

@st.cache_data(ttl=600)
def load_data():
    trades = pd.read_csv(
        os.path.join(DATA_DIR, 'overnight_trades.csv'),
        parse_dates=['signal_date', 'exit_date'],
    )
    cands = pd.read_csv(
        os.path.join(DATA_DIR, 'overnight_candidates.csv'),
        parse_dates=['signal_date'],
    )
    with open(os.path.join(DATA_DIR, 'overnight_stats.json'), encoding='utf-8') as f:
        stats = json.load(f)
    return trades, cands, stats

try:
    trades, cands, stats = load_data()
except FileNotFoundError:
    st.error('找不到資料檔，請先執行 overnight_bt/generate_trade_log.py')
    st.stop()

# 成本（從 stats 讀取，由 generate_trade_log.py 寫入）
COST_DISC_PCT = stats.get('cost_disc_pct', 0.357) / 100   # 手續費 2折
COST_STD_PCT  = stats.get('cost_std_pct',  0.585) / 100   # 標準費率

# avg_bps 已是淨值（generate_trade_log 扣費後計算）
net_yf_pct    = stats['avg_bps'] / 100          # 淨日均報酬（2折）
avg_gross_pct = stats.get('avg_gross_bps', stats['avg_bps'] + COST_DISC_PCT * 10000) / 100

# ── 標題 ────────────────────────────────────────────────
st.title('⚡ 隔日沖回測')
st.caption(
    f"策略：**{stats['strategy']}**　　"
    f"資料區間：{stats['period']}　　"
    f"最後更新：{stats['latest_date']}"
)

# ── 績效指標 ────────────────────────────────────────────
st.subheader('績效摘要')

def mbox(col, val, label, positive_good=True):
    v = str(val)
    cls = 'metric-val' if (positive_good and not v.startswith('-')) or \
          (not positive_good and v.startswith('-')) else 'metric-val2'
    col.markdown(
        f'<div class="metric-box"><div class="{cls}">{val}</div>'
        f'<div class="metric-lab">{label}</div></div>',
        unsafe_allow_html=True
    )

c1, c2, c3, c4, c5, c6 = st.columns(6)
mbox(c1, stats['sharpe'],                       'Sharpe（扣費後）')
mbox(c2, f"{float(stats['win_rate_pct']):.2f}%", '勝率（扣費後）')
mbox(c3, f"{avg_gross_pct:.2f}%",               '日均毛報酬')
mbox(c4, f"{net_yf_pct:.2f}%",                  '日均淨報酬（2折）')
mbox(c5, f"{float(stats['max_dd_pct']):.2f}%",  '最大回撤（扣費後）', positive_good=False)
mbox(c6, f"{stats['n_trades']}",                f'總交易筆數 ({stats["n_days"]}天)')

st.caption(
    f"手續費：2折 {COST_DISC_PCT*100:.2f}% | 標準 {COST_STD_PCT*100:.2f}%（買+賣，含 0.3% 證交稅）"
    f"　　Sharpe / 勝率 / DD 均已扣費計算"
)

st.divider()

# ── 資金試算 ────────────────────────────────────────────
st.subheader('💰 資金試算')
st.caption('2 檔等金額部位，每檔 50% 本金（手續費兩折，扣費後淨報酬）')

cap_col, note_col = st.columns([1, 2])
with cap_col:
    capital = st.number_input(
        '投入本金（元）', min_value=100_000, max_value=50_000_000,
        value=1_000_000, step=100_000, format='%d'
    )
with note_col:
    st.markdown(f"""
每檔部位 **{capital // 2:,.0f} 元**（2 檔各 50%）

> ⚠️ 下列試算為回測歷史均值，實際損益受市況、滑價影響而不同。
> 未計入尾盤買進的滑價成本（估計額外 5–10 bps）。
""")

per_trade_nt  = capital / 2
daily_net_nt  = capital * net_yf_pct / 100
monthly_net   = daily_net_nt * 21
annual_factor = (1 + net_yf_pct / 100) ** 252 - 1
annual_net_nt = capital * annual_factor

b1, b2, b3, b4 = st.columns(4)
b1.markdown(
    f'<div class="pnl-box"><div class="pnl-title">每日平均淨損益</div>'
    f'<div class="pnl-val">+{daily_net_nt:,.0f} 元</div></div>',
    unsafe_allow_html=True
)
b2.markdown(
    f'<div class="pnl-box"><div class="pnl-title">每月平均淨損益（21交易日）</div>'
    f'<div class="pnl-val">+{monthly_net:,.0f} 元</div></div>',
    unsafe_allow_html=True
)
b3.markdown(
    f'<div class="pnl-box"><div class="pnl-title">年化報酬（複利）</div>'
    f'<div class="pnl-val">+{annual_factor*100:.2f}%</div></div>',
    unsafe_allow_html=True
)
b4.markdown(
    f'<div class="pnl-box"><div class="pnl-title">年化淨損益（估算）</div>'
    f'<div class="pnl-val">+{annual_net_nt:,.0f} 元</div></div>',
    unsafe_allow_html=True
)

st.divider()

# ── 候選股 ──────────────────────────────────────────────
st.subheader(f'📋 最新候選股  {stats["candidate_date"]}  收盤買進 → 隔日開盤賣')
st.caption('⚠️ 回測訊號，非投資建議。已排除當日漲停股（尾盤無法成交）。')

if len(cands) == 0:
    st.info('目前無候選股')
else:
    cols = st.columns(len(cands))
    for i, (_, row) in enumerate(cands.iterrows()):
        ret = row['signal_day_ret_pct']
        ret_str = f"+{ret:.2f}%" if ret > 0 else f"{ret:.2f}%"
        ret_cls = 'up' if ret > 0 else 'down'
        pos_nt  = capital // 2
        cols[i].markdown(
            f'<div class="cand-card">'
            f'<div class="cand-stock">{row["stock_id"]} {row["stock_name"]}</div>'
            f'<div class="cand-meta">'
            f'建議進場 <b style="color:#f1f5f9">{row["entry_price"]:.2f}</b> 元<br>'
            f'今日 <span class="{ret_cls}">{ret_str}</span>　'
            f'量比 <b>{row["signal_vol_ratio"]:.2f}x</b><br>'
            f'部位大小 <b>{pos_nt:,.0f}</b> 元'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.divider()

# ── 資金曲線 ────────────────────────────────────────────
st.subheader('📈 資金曲線')
daily_gross = trades.groupby('signal_date')['ret_pct'].mean().reset_index()
daily_net   = trades.groupby('signal_date')['net_ret_pct'].mean().reset_index()
daily_avg   = daily_gross.merge(daily_net, on='signal_date')
daily_avg['cumret']     = (1 + daily_avg['ret_pct'] / 100).cumprod()
daily_avg['cumret_net'] = (1 + daily_avg['net_ret_pct'] / 100).cumprod()

chart_data = daily_avg.melt(
    id_vars='signal_date',
    value_vars=['cumret', 'cumret_net'],
    var_name='series', value_name='value'
)
chart_data['series'] = chart_data['series'].map({
    'cumret'    : '毛報酬',
    'cumret_net': f'淨報酬（2折 {COST_DISC_PCT*100:.2f}%/回合）',
})

chart = (
    alt.Chart(chart_data)
    .mark_line(strokeWidth=1.5)
    .encode(
        x=alt.X('signal_date:T', title='日期'),
        y=alt.Y('value:Q', title='累積倍數', scale=alt.Scale(type='log')),
        color=alt.Color('series:N', legend=alt.Legend(orient='top-left')),
        tooltip=[
            alt.Tooltip('signal_date:T', title='日期'),
            'series:N',
            alt.Tooltip('value:Q', format='.2f', title='倍數'),
        ],
    )
    .properties(height=320)
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

st.divider()

# ── 交易明細 ────────────────────────────────────────────
st.subheader('📋 交易明細')

col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
with col_f1:
    date_min = trades['signal_date'].min().date()
    date_max = trades['signal_date'].max().date()
    date_range = st.date_input(
        '日期區間',
        value=(date_max - pd.Timedelta(days=30), date_max),
        min_value=date_min, max_value=date_max,
    )
with col_f2:
    stock_filter = st.text_input('股票代號 / 名稱', placeholder='例：2330')
with col_f3:
    result_filter = st.selectbox('損益篩選', ['全部', '獲利', '虧損'])

df_show = trades.copy()
if len(date_range) == 2:
    df_show = df_show[
        (df_show['signal_date'].dt.date >= date_range[0]) &
        (df_show['signal_date'].dt.date <= date_range[1])
    ]
if stock_filter:
    m = (df_show['stock_id'].str.contains(stock_filter, na=False) |
         df_show['stock_name'].str.contains(stock_filter, na=False))
    df_show = df_show[m]
if result_filter == '獲利':
    df_show = df_show[df_show['net_ret_pct'] > 0]
elif result_filter == '虧損':
    df_show = df_show[df_show['net_ret_pct'] <= 0]

# 加入每筆 NT$ 損益（用淨報酬計算）
df_show = df_show.copy()
df_show['損益_NT'] = (df_show['net_ret_pct'] / 100 * (capital / 2)).round(0).astype(int)

rename_map = {
    'signal_date'        : '訊號日',
    'exit_date'          : '出場日',
    'stock_id'           : '代號',
    'stock_name'         : '名稱',
    'entry_price'        : '進場收盤',
    'signal_day_ret_pct' : '訊號日漲跌',
    'signal_vol_ratio'   : '訊號日量比',
    'signal_amount_wan'  : '成交金額(萬)',
    'exit_open'          : '出場開盤',
    'exit_open_ret_pct'  : '開盤漲跌',
    'exit_close'         : '次日收盤',
    'exit_close_ret_pct' : '次日收盤漲跌',
    'exit_vol_ratio'     : '次日量比',
    'ret_pct'            : '毛損益%',
    'net_ret_pct'        : '淨損益%',
    '損益_NT'            : '淨損益(元)',
}
display_cols = [c for c in rename_map if c in df_show.columns]
df_disp = df_show[display_cols].rename(columns=rename_map).sort_values('訊號日', ascending=False)

def style_ret(val):
    if pd.isna(val):
        return ''
    try:
        return f'color: {"#f87171" if float(val) > 0 else "#4ade80" if float(val) < 0 else "#94a3b8"}'
    except:
        return ''

pct_cols = ['訊號日漲跌', '開盤漲跌', '次日收盤漲跌', '毛損益%', '淨損益%']
nt_cols  = ['淨損益(元)']

styled = (
    df_disp.style
    .map(style_ret, subset=pct_cols + nt_cols)
    .format({
        '進場收盤'     : '{:.2f}',
        '出場開盤'     : '{:.2f}',
        '次日收盤'     : '{:.2f}',
        '訊號日漲跌'   : '{:+.2f}%',
        '訊號日量比'   : '{:.2f}x',
        '開盤漲跌'     : '{:+.2f}%',
        '次日收盤漲跌' : '{:+.2f}%',
        '次日量比'     : '{:.2f}x',
        '毛損益%'      : '{:+.2f}%',
        '淨損益%'      : '{:+.2f}%',
        '淨損益(元)'   : '{:+,.0f}',
        '成交金額(萬)' : '{:,.0f}',
    }, na_rep='—')
)

st.caption(f'顯示 {len(df_show)} 筆 / 共 {len(trades)} 筆　　每筆部位：{capital//2:,.0f} 元（各 50%）　　損益欄已扣手續費 2折 + 證交稅')
st.dataframe(styled, use_container_width=True, height=480)

if len(df_show) > 0:
    gross_pct = df_show['ret_pct'].mean()
    net_pct   = df_show['net_ret_pct'].mean()
    wr        = (df_show['net_ret_pct'] > 0).mean() * 100
    total_nt  = df_show['損益_NT'].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('期間平均毛損益', f'{gross_pct:+.2f}%')
    c2.metric('期間平均淨損益（2折）', f'{net_pct:+.2f}%')
    c3.metric('勝率（扣費後）', f'{wr:.2f}%')
    c4.metric('期間累計淨損益(元)', f'{total_nt:+,.0f}')
