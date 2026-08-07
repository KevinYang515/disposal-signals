# -*- coding: utf-8 -*-
"""富邦證券(9600)連續鎖漲停隔日沖：獨立事件研究總覽。"""
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='富邦證券連板隔日沖', page_icon='🏦', layout='wide')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


@st.cache_data(ttl=3600)
def load_data():
    events = pd.read_csv(os.path.join(DATA_DIR, 'fubon9600_events.csv'),
                          parse_dates=['lock_start', 'lock_end', 'd1', 'entry_day'])
    events['code'] = events['code'].astype(str)
    yearly = pd.read_csv(os.path.join(DATA_DIR, 'fubon9600_yearly.csv'))
    return events, yearly


st.title('🏦 富邦證券(9600)連續鎖漲停隔日沖｜獨立事件研究')
st.caption(
    '2026-08-07發現：股票結束連續鎖漲停後，富邦證券(母公司代號9600，非任何分店)常見「鎖板期間累積買超、'
    '解鎖隔天集中賣超」的模式。母體：全市場連續鎖漲停區間結束後的下一交易日，開盤放空、收盤回補。'
    '獨立於「凱基-城中GA」策略之外(扣除重疊事件後結論不變)，可平行部署，不建議與城中GA混用同一資金池'
    '(混合後Sharpe反而比富邦單獨略低，見下方比較)。'
)

events, yearly = load_data()
events['censored'] = events['censored'].astype(bool)
events['entry_is_ceiling'] = events['entry_is_ceiling'].astype(bool)
events['tp2_hit'] = events['tp2_hit'].astype(bool)

st.divider()

# ── 兩個已修正的方法論陷阱 ──────────────────────────────────────
st.subheader('⚠️ 已修正的兩個統計陷阱')
c1, c2 = st.columns(2)
with c1:
    st.markdown('**陷阱1：進場天花板價不可執行**')
    n_ceiling = int(events['entry_is_ceiling'].sum())
    st.markdown(
        f'- {n_ceiling}/{len(events)}筆({n_ceiling/len(events)*100:.1f}%)事件D1開盤價精確等於當日漲停天花板\n'
        '- 修正前：這批100%獲利、均值+7.85%（不合理）\n'
        '- 修正後（延遲進場到真正解鎖那天）：勝率51.0%、均值-0.16%\n'
        '- 已在下方所有統計套用修正'
    )
with c2:
    st.markdown('**陷阱2：同日多事件被當獨立樣本重複計算**')
    st.markdown(
        '- 3950筆事件僅來自822檔股票，最高單一股票出現30次\n'
        '- 89%事件跟其他事件同一天發生(最多同天29檔)\n'
        '- 逐筆當獨立樣本：t=27（不合理的高）\n'
        '- 改「同一天合併成一筆」：t=16.4（仍強但合理範圍）\n'
        '- 下方所有統計皆為「逐日」口徑，非逐筆'
    )

st.divider()

# ── 跟城中GA比較 ──────────────────────────────────────
st.subheader('📊 跟現有「凱基-城中GA」策略比較')
compare = pd.DataFrame([
    {'策略': '凱基-城中GA(現有)', '資料起點': '2021', '交易天數': 588, '平均報酬/天': '+1.36%', '年化Sharpe': 5.30},
    {'策略': '富邦證券連板隔日沖(新)', '資料起點': '2018', '交易天數': 1151, '平均報酬/天': '+1.75%', '年化Sharpe': 7.68},
])
st.dataframe(compare.set_index('策略'), use_container_width=True)
st.caption(
    '同一標準(逐日、已修正進場天花板陷阱)比較：富邦證券在報酬、機會數、統計穩定度、資料長度都優於城中GA。'
    '兩策略同一天都有訊號的天數達529天(佔城中GA天數90%)，但混合成同一資金池的組合Sharpe(7.27)'
    '反而比富邦證券單獨(7.68)略低——建議兩者平行、各自資金操作，不要混池。'
)

st.divider()

# ── TP/停損回測 ──────────────────────────────────────
st.subheader('🎯 停利/停損模擬')
tp_pct = st.slider('停利門檻（盤中觸及即出場，%）', 0.0, 10.0, 2.0, 0.5,
                    help='0代表不設停利(完全比照開盤空/收盤補)。2026-08-07回測：TP=2%時Sharpe最高(8.80)，'
                         '任何停損設定都會讓結果變差，不建議加停損。')
if tp_pct > 0:
    tp_price = events['entry_o'] * (1 - tp_pct / 100)
    hit = events['entry_l'] <= tp_price
    exit_price = np.where(hit, tp_price, events['entry_c'])
else:
    exit_price = events['entry_c']
sim_ret = (events['entry_o'] - exit_price) / events['entry_o'] * 100
sim_df = events[~events['censored']].copy()
sim_df['sim_ret'] = sim_ret[~events['censored']]
daily_sim = sim_df.groupby('d1')['sim_ret'].mean()
sharpe_sim = daily_sim.mean() / daily_sim.std() * np.sqrt(252) if daily_sim.std() else np.nan

m1, m2, m3, m4 = st.columns(4)
m1.metric('交易天數', f'{len(daily_sim):,}')
m2.metric('平均報酬/天', f'{daily_sim.mean():+.2f}%')
m3.metric('勝率', f'{(daily_sim > 0).mean() * 100:.1f}%')
m4.metric('年化Sharpe', f'{sharpe_sim:.2f}')

grid = pd.DataFrame([
    {'停利門檻': '不設停利', '平均報酬/天': '+1.76%', '勝率': '69.2%', 'Sharpe': 7.80},
    {'停利門檻': '1.0%', '平均報酬/天': '+0.56%', '勝率': '85.7%', 'Sharpe': 7.66},
    {'停利門檻': '1.5%', '平均報酬/天': '+0.78%', '勝率': '83.2%', 'Sharpe': 8.30},
    {'停利門檻': '2.0%（建議）', '平均報酬/天': '+0.99%', '勝率': '82.3%', 'Sharpe': 8.80},
    {'停利門檻': '2.5%', '平均報酬/天': '+1.11%', '勝率': '78.7%', 'Sharpe': 8.56},
    {'停利門檻': '3.0%', '平均報酬/天': '+1.22%', '勝率': '76.9%', 'Sharpe': 8.57},
    {'停利門檻': '任何停損(5~12%)', '平均報酬/天': '全部變差', '勝率': '全部變差', 'Sharpe': '全部變差'},
])
st.dataframe(grid.set_index('停利門檻'), use_container_width=True)

st.divider()

# ── 逐年表現 ──────────────────────────────────────
st.subheader('📅 逐年表現（已修正兩個陷阱，逐日口徑）')
yearly_disp = yearly.rename(columns={
    'year': '年份', 'n_days_base': '交易天數', 'mean_base': '均值(無停利)', 'win_base': '勝率(無停利)%',
    't_base': 't值(無停利)', 'mean_tp2': '均值(2%停利)', 'win_tp2': '勝率(2%停利)%', 't_tp2': 't值(2%停利)',
}).set_index('年份')
st.dataframe(yearly_disp, use_container_width=True)
st.caption('2018~2026每一年都是正報酬(2019樣本僅13天僅供參考)；加上2%停利後每年勝率都在7成以上、t值多數維持甚至更強。')

st.divider()

# ── 逐筆歷史紀錄 ──────────────────────────────────────
st.subheader('📋 逐筆歷史紀錄')
col1, col2 = st.columns(2)
with col1:
    sel_year = st.multiselect('年份', sorted(events['year'].unique().tolist()),
                               default=sorted(events['year'].unique().tolist()))
with col2:
    show_censored = st.checkbox('包含無法回補的截尾事件', value=False)

view = events[events['year'].isin(sel_year)]
if not show_censored:
    view = view[~view['censored']]

show_cols = ['code', 'name', 'lock_start', 'lock_end', 'lock_days', 'entry_day', 'entry_open',
             'entry_o', 'entry_h', 'entry_l', 'entry_c', 'base_ret', 'tp2_ret', 'tp2_hit',
             'entry_is_ceiling', 'citycenter_overlap']
show_cols = [c for c in show_cols if c in view.columns]
view_disp = view[show_cols].copy()
for c in ['lock_start', 'lock_end', 'entry_day']:
    if c in view_disp.columns:
        view_disp[c] = view_disp[c].dt.strftime('%Y-%m-%d')
st.dataframe(
    view_disp.sort_values('entry_day', ascending=False),
    use_container_width=True, height=500,
    column_config={
        'base_ret': st.column_config.NumberColumn('無停利報酬%', format='%.2f'),
        'tp2_ret': st.column_config.NumberColumn('2%停利報酬%', format='%.2f'),
        'entry_is_ceiling': st.column_config.CheckboxColumn('進場曾修正'),
        'citycenter_overlap': st.column_config.CheckboxColumn('與城中GA重疊'),
    },
)

st.divider()
st.subheader('⚠️ 仍需人工確認')
st.markdown(
    '- 只有交易所日OHLC，沒有分鐘K，所以停利模擬假設「當天只要摸到價位就能成交」，實際滑價、成交量、'
    '借券可得性都還沒考慮。\n'
    '- 手法分析(是哪個時間點/什麼價位倒貨)需要即時買賣日誌，但該系統只能查「最新一個交易日」，無法回溯'
    '2018-2026整段歷史，這次沒有辦法做。\n'
    '- 尚未在真實環境模擬單驗證，建議先用模擬單觀察一段時間再考慮實際部署。'
)
