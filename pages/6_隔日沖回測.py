"""
隔日沖回測  — 策略：F1+F2+F3 + 今日漲幅排名 + 隔日開盤賣
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
.up   { color: #f87171; font-weight: 600; }
.down { color: #4ade80; font-weight: 600; }
.neu  { color: #94a3b8; }
.stDataFrame { font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── 路徑 ────────────────────────────────────────────────
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

# ── 標題 ────────────────────────────────────────────────
st.title('⚡ 隔日沖回測')
st.caption(
    f"策略：**{stats['strategy']}**　　"
    f"資料區間：{stats['period']}　　"
    f"最後更新：{stats['latest_date']}"
)

COST_YF  = 35.7   # 永豐隔日沖 bps
COST_STD = 58.5   # 標準費率 bps

# ── 績效指標 ────────────────────────────────────────────
st.subheader('績效摘要（未扣成本）')
c1, c2, c3, c4, c5, c6 = st.columns(6)

def mbox(col, val, label, positive_good=True):
    cls = 'metric-val' if (positive_good and str(val)[0] != '-') or \
          (not positive_good and str(val)[0] == '-') else 'metric-val2'
    col.markdown(
        f'<div class="metric-box">'
        f'<div class="{cls}">{val}</div>'
        f'<div class="metric-lab">{label}</div>'
        f'</div>', unsafe_allow_html=True
    )

mbox(c1, stats['sharpe'],          'Sharpe Ratio')
mbox(c2, f"{stats['win_rate_pct']}%",  '勝率')
mbox(c3, f"{stats['avg_bps']} bps",    '平均日報酬（毛）')
mbox(c4, f"{stats['avg_bps'] - COST_YF:.1f} bps", '永豐淨報酬')
mbox(c5, f"{stats['max_dd_pct']}%",    '最大回撤', positive_good=False)
mbox(c6, f"{stats['n_trades']}",       f'總交易筆數 ({stats["n_days"]}天)')

st.caption(
    f"💡 成本參考：永豐隔日沖 {COST_YF} bps | 標準費率 {COST_STD} bps（含 0.3% 證交稅）"
)

st.divider()

# ── 候選股 ──────────────────────────────────────────────
st.subheader(f'📋 最新候選股  {stats["candidate_date"]}  收盤買進 → 隔日開盤賣')
st.caption('⚠️ 回測訊號，非投資建議。實際執行前請自行評估風險。')

if len(cands) == 0:
    st.info('目前無候選股')
else:
    cols = st.columns(len(cands))
    for i, (_, row) in enumerate(cands.iterrows()):
        ret_str = f"+{row['signal_day_ret_pct']:.2f}%" if row['signal_day_ret_pct'] > 0 \
                  else f"{row['signal_day_ret_pct']:.2f}%"
        ret_cls = 'up' if row['signal_day_ret_pct'] > 0 else 'down'
        cols[i].markdown(
            f'<div class="cand-card">'
            f'<div class="cand-stock">{row["stock_id"]} {row["stock_name"]}</div>'
            f'<div class="cand-meta">'
            f'收盤 <b style="color:#f1f5f9">{row["entry_price"]}</b> 元　'
            f'<span class="{ret_cls}">{ret_str}</span><br>'
            f'量比 <b>{row["signal_vol_ratio"]:.1f}x</b>　'
            f'成交 <b>{int(row["signal_amount_wan"] or 0):,}</b> 萬元'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.divider()

# ── 資金曲線 ────────────────────────────────────────────
st.subheader('📈 資金曲線')
daily_avg = trades.groupby('signal_date')['ret_pct'].mean().reset_index()
daily_avg['cumret'] = (1 + daily_avg['ret_pct'] / 100).cumprod()
daily_avg['cumret_net_yf'] = (1 + (daily_avg['ret_pct'] / 100 - COST_YF / 10000)).cumprod()

chart_data = daily_avg.melt(
    id_vars='signal_date',
    value_vars=['cumret', 'cumret_net_yf'],
    var_name='series', value_name='value'
)
chart_data['series'] = chart_data['series'].map({
    'cumret'        : '毛報酬',
    'cumret_net_yf' : f'淨報酬（永豐 {COST_YF} bps）',
})

chart = (
    alt.Chart(chart_data)
    .mark_line(strokeWidth=1.5)
    .encode(
        x=alt.X('signal_date:T', title='日期'),
        y=alt.Y('value:Q', title='累積倍數', scale=alt.Scale(type='log')),
        color=alt.Color('series:N', legend=alt.Legend(orient='top-left')),
        tooltip=['signal_date:T', 'series:N',
                 alt.Tooltip('value:Q', format='.2f', title='倍數')],
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

# 篩選
df_show = trades.copy()
if len(date_range) == 2:
    df_show = df_show[
        (df_show['signal_date'].dt.date >= date_range[0]) &
        (df_show['signal_date'].dt.date <= date_range[1])
    ]
if stock_filter:
    mask = (
        df_show['stock_id'].str.contains(stock_filter, na=False) |
        df_show['stock_name'].str.contains(stock_filter, na=False)
    )
    df_show = df_show[mask]
if result_filter == '獲利':
    df_show = df_show[df_show['ret_bps'] > 0]
elif result_filter == '虧損':
    df_show = df_show[df_show['ret_bps'] <= 0]

st.caption(f'顯示 {len(df_show)} 筆 / 共 {len(trades)} 筆')

# 欄位重命名方便閱讀
rename_map = {
    'signal_date'       : '訊號日',
    'exit_date'         : '出場日',
    'stock_id'          : '代號',
    'stock_name'        : '名稱',
    'entry_price'       : '進場收盤',
    'signal_day_ret_pct': '訊號日漲跌%',
    'signal_vol_ratio'  : '訊號日量比',
    'signal_amount_wan' : '訊號日成交(萬)',
    'exit_open'         : '出場開盤',
    'exit_open_ret_pct' : '開盤漲跌%',
    'exit_close'        : '次日收盤',
    'exit_close_ret_pct': '次日收盤漲跌%',
    'exit_vol_ratio'    : '次日量比',
    'ret_bps'           : '損益(bps)',
    'ret_pct'           : '損益(%)',
}
display_cols = list(rename_map.keys())
df_display = df_show[display_cols].rename(columns=rename_map).sort_values('訊號日', ascending=False)

# 格式化數值顯示
def style_ret(val):
    if pd.isna(val):
        return ''
    color = '#f87171' if val > 0 else '#4ade80' if val < 0 else '#94a3b8'
    return f'color: {color}'

styled = (
    df_display.style
    .applymap(style_ret, subset=['訊號日漲跌%', '開盤漲跌%', '次日收盤漲跌%', '損益(%)', '損益(bps)'])
    .format({
        '進場收盤'      : '{:.2f}',
        '出場開盤'      : '{:.2f}',
        '次日收盤'      : '{:.2f}',
        '訊號日漲跌%'   : '{:+.2f}',
        '訊號日量比'    : '{:.1f}x',
        '開盤漲跌%'    : '{:+.2f}',
        '次日收盤漲跌%': '{:+.2f}',
        '次日量比'     : '{:.1f}x',
        '損益(bps)'   : '{:+.0f}',
        '損益(%)'     : '{:+.2f}',
        '訊號日成交(萬)': '{:,.0f}',
    }, na_rep='—')
)

st.dataframe(styled, use_container_width=True, height=480)

# 篩選期間小結
if len(df_show) > 0:
    period_avg = df_show['ret_bps'].mean()
    period_wr  = (df_show['ret_bps'] > 0).mean() * 100
    net_yf     = period_avg - COST_YF
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('期間平均損益', f'{period_avg:.1f} bps')
    c2.metric('永豐淨報酬', f'{net_yf:.1f} bps')
    c3.metric('勝率', f'{period_wr:.1f}%')
    c4.metric('筆數', len(df_show))
