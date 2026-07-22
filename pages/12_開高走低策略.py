"""
開高走低策略 — 全市場開高>=3%後找容易走低的股票，開盤空、收盤補
資料由 build_gapfade_data.py 產生，來源研究：D:\\stock\\open_high_fade_study\\
用LightGBM+SHAP找因子(市值/產業/投信外資買賣超/融券使用率等) + GA優化門檻，Walk-forward驗證(TRAIN<2024, TEST>=2024)。
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
import altair as alt

st.set_page_config(
    page_title='開高走低策略',
    page_icon='📉',
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
    border-left: 3px solid #f87171;
}
.cand-stock { font-size: 1.1em; font-weight: 700; color: #f1f5f9; }
.cand-meta  { font-size: 0.85em; color: #94a3b8; margin-top: 4px; }
.up   { color: #f87171; font-weight: 600; }
.down { color: #4ade80; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

@st.cache_data(ttl=600)
def load_data():
    trades = pd.read_csv(os.path.join(DATA_DIR, 'gapfade_trades.csv'), parse_dates=['signal_date'])
    daily = pd.read_csv(os.path.join(DATA_DIR, 'gapfade_daily.csv'), parse_dates=['date'])
    yearly = pd.read_csv(os.path.join(DATA_DIR, 'gapfade_yearly.csv'))
    with open(os.path.join(DATA_DIR, 'gapfade_stats.json'), encoding='utf-8') as f:
        stats = json.load(f)
    stock_names = pd.read_csv(os.path.join(DATA_DIR, 'stock_names.csv'))
    stock_names['ticker'] = stock_names['ticker'].astype(str)
    live = None
    live_path = os.path.join(DATA_DIR, 'gapfade_live_signal.json')
    if os.path.exists(live_path):
        with open(live_path, encoding='utf-8') as f:
            live = json.load(f)
    return trades, daily, yearly, stats, stock_names, live

try:
    trades, daily, yearly, stats, stock_names, live = load_data()
except FileNotFoundError:
    st.error('找不到資料檔，請先執行 build_gapfade_data.py')
    st.stop()

# ── 標題 ────────────────────────────────────────────────
st.title('📉 開高走低策略')
st.caption(
    f"策略：**{stats['strategy']}**　　"
    f"回測資料區間：{stats['period']}　　"
    f"最後更新：{stats['latest_date']}"
)

st.caption(
    '⚠️ 回測數字已扣除0.207%當沖成本（永豐2折手續費+當沖稅減半），'
    '但未計入滑價與部分低流動性股票的實際成交風險，僅供參考，非投資建議。'
)

st.divider()

# ── 今日候選股 (VM Stage2 於開盤後自動產生) ──────────────
st.subheader(f'📋 今日候選股')
if live is None:
    st.info('尚未產生今日訊號（每日09:00開盤後由VM自動更新，若當天還沒更新過會顯示這則訊息）。')
else:
    st.caption(f"更新時間：{live['generated_at']}　　候選池：{live['n_candidates_pool']}檔　　"
               f"開高≥3%符合：{live['n_signal']}檔　　需要≥{live['min_candidates_required']}檔才進場")
    if not live['triggered']:
        st.info(f"今日候選數 {live['n_signal']} < {live['min_candidates_required']}，未達門檻，今日空手觀望。")
    else:
        picks = pd.DataFrame(live['picks'])
        picks['code'] = picks['code'].astype(str)
        picks = picks.merge(stock_names, left_on='code', right_on='ticker', how='left')
        picks['name'] = picks['name'].fillna('')
        cols = st.columns(len(picks))
        for i, (_, row) in enumerate(picks.iterrows()):
            cols[i].markdown(
                f'<div class="cand-card">'
                f'<div class="cand-stock">{row["code"]} {row["name"]}</div>'
                f'<div class="cand-meta">'
                f'開盤 <b style="color:#f1f5f9">{row["open_px"]:.2f}</b> 元　'
                f'跳空 <span class="up">+{row["gap_pct"]:.2f}%</span><br>'
                f'市值 {row["mktcap_e8"]:.2f}億　{row["industry"]}<br>'
                f'配置權重 <b>{row["weight"]:.2%}</b>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

st.divider()

# ── 績效指標 ────────────────────────────────────────────
st.subheader('績效摘要（樣本外，未曾用來挑選門檻）')

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
mbox(c1, f"{stats['sharpe']:.2f}",           'Sharpe（日組合,扣費後）')
mbox(c2, f"{stats['win_rate_pct']:.2f}%",    '勝率（逐筆）')
mbox(c3, f"{stats['profit_factor']:.2f}",    '賺賠比')
mbox(c4, f"{stats['avg_net_pct']:.2f}%",     '平均每筆淨報酬')
mbox(c5, f"{stats['max_dd_pct']:.2f}%",      '最大回撤（日組合）', positive_good=False)
mbox(c6, f"{stats['n_trades']} 筆",          f'{stats["n_days"]} 個交易日有訊號')

st.caption('進場：開高後跳空愈大分數愈高＋ML模型分數加權配置5檔；出場：當日收盤；停損：8%（保險用，歷史上從未觸發）')

st.divider()

# ── 年度穩定性 ──────────────────────────────────────────
st.subheader('📅 年度穩定性')
yearly_disp = yearly.rename(columns={'year': '年份', 'n': '筆數', 'mean_net': '平均淨報酬%', 'winrate': '勝率%'})
st.dataframe(
    yearly_disp.style.format({'平均淨報酬%': '{:+.2f}%', '勝率%': '{:.2f}%'}),
    use_container_width=True, hide_index=True,
)

st.divider()

# ── 資金曲線 ────────────────────────────────────────────
st.subheader('📈 資金曲線（樣本外，等權=分數加權配置5檔）')
chart = (
    alt.Chart(daily)
    .mark_line(strokeWidth=1.5, color='#4ade80')
    .encode(
        x=alt.X('date:T', title='日期'),
        y=alt.Y('cum_net:Q', title='累積倍數', scale=alt.Scale(type='log')),
        tooltip=[
            alt.Tooltip('date:T', title='日期'),
            alt.Tooltip('daily_net_pct:Q', format='+.2f', title='當日淨報酬%'),
            alt.Tooltip('cum_net:Q', format='.2f', title='累積倍數'),
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
    m = (df_show['stock_id'].astype(str).str.contains(stock_filter, na=False) |
         df_show['stock_name'].astype(str).str.contains(stock_filter, na=False))
    df_show = df_show[m]
if result_filter == '獲利':
    df_show = df_show[df_show['net_ret_pct'] > 0]
elif result_filter == '虧損':
    df_show = df_show[df_show['net_ret_pct'] <= 0]

rename_map = {
    'signal_date': '訊號日',
    'stock_id': '代號',
    'stock_name': '名稱',
    'industry': '產業',
    'gap_pct': '開盤跳空%',
    'mktcap_yi': '市值(億)',
    'weight': '配置權重',
    'gross_ret_pct': '毛損益%',
    'net_ret_pct': '淨損益%',
}
display_cols = [c for c in rename_map if c in df_show.columns]
df_disp = df_show[display_cols].rename(columns=rename_map).sort_values('訊號日', ascending=False)

def style_ret(val):
    if pd.isna(val):
        return ''
    try:
        return f'color: {"#4ade80" if float(val) > 0 else "#f87171" if float(val) < 0 else "#94a3b8"}'
    except:
        return ''

pct_cols = ['毛損益%', '淨損益%']

styled = (
    df_disp.style
    .map(style_ret, subset=pct_cols)
    .format({
        '開盤跳空%': '{:+.2f}%',
        '市值(億)': '{:.2f}',
        '配置權重': '{:.2%}',
        '毛損益%': '{:+.2f}%',
        '淨損益%': '{:+.2f}%',
    }, na_rep='—')
)

st.caption(f'顯示 {len(df_show)} 筆 / 共 {len(trades)} 筆　　空方策略：淨損益為正代表股票如預期走低')
st.dataframe(styled, use_container_width=True, height=480)

if len(df_show) > 0:
    net_pct = df_show['net_ret_pct'].mean()
    wr = (df_show['net_ret_pct'] > 0).mean() * 100
    c1, c2 = st.columns(2)
    c1.metric('期間平均淨損益', f'{net_pct:+.2f}%')
    c2.metric('期間勝率', f'{wr:.2f}%')

st.divider()
st.caption(
    '📖 策略說明：全市場個股開高(gap>=3%)後，用LightGBM模型評估市值、產業、投信/外資買賣超、'
    '融券使用率等因子，篩出隔天最不容易被支撐、容易從開盤價走低的股票放空。'
    '排除前一天鎖漲停的股票（該母體交給另一套分點集中度策略處理）。'
    '每日候選數需達7檔以上才進場，用來控制單日集中度風險（未達門檻的日子空手觀望）。'
)
