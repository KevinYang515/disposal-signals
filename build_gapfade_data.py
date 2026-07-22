"""
開高走低策略 — 產生 Streamlit 頁面用的資料檔
來源：D:\\stock\\open_high_fade_study\\v2_final_rule_hits_excl_locked.parquet
規則：全市場開高>=3% + 市值<=76億等濾網 + 排除前一天鎖漲停 + 候選數>=7才進場 + top5用ML分數加權 + 8%停損(保險用)
只呈現樣本外(2024-01-01起)結果，避免用訓練期挑門檻的資料混淆使用者判斷。
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

SRC = Path('D:/stock/open_high_fade_study/v2_final_rule_hits_excl_locked.parquet')
OUT_DIR = Path('D:/stock/disposal-signals/data')

ROUND_TRIP_COST = 0.207
OOS_START = '2024-01-01'

rule = pd.read_parquet(SRC)
rule['date'] = pd.to_datetime(rule['date'])
base = rule[rule['gap_pct'] >= 3.0].copy()
base['signal_n'] = base.groupby('date')['symbol'].transform('size')
d = base[base['signal_n'] >= 7].copy()
d['rank'] = d.groupby('date')['pred_score'].rank(method='first', ascending=False)
top5 = d[d['rank'] <= 5].copy()
top5['w'] = top5.groupby('date')['pred_score'].transform(lambda s: s - s.min() + 0.01)
top5['w'] = top5.groupby('date')['w'].transform(lambda s: s / s.sum())

oos = top5[top5['date'] >= OOS_START].copy()

# --- trades csv ---
stock_names = pd.read_csv('data/stock_names.csv')
stock_names['ticker'] = stock_names['ticker'].astype(str)
oos = oos.merge(stock_names, left_on='symbol', right_on='ticker', how='left')
oos['name'] = oos['name'].fillna('')

trades = oos[['date', 'symbol', 'name', 'gap_pct', 'mktcap_e8', 'industry', 'w', 'fade_from_open_pct_honest', 'net']].copy()
trades = trades.rename(columns={
    'date': 'signal_date', 'symbol': 'stock_id', 'name': 'stock_name',
    'gap_pct': 'gap_pct', 'mktcap_e8': 'mktcap_yi', 'industry': 'industry',
    'w': 'weight', 'fade_from_open_pct_honest': 'gross_ret_pct', 'net': 'net_ret_pct',
})
trades = trades.sort_values('signal_date')
trades.to_csv(OUT_DIR / 'gapfade_trades.csv', index=False, encoding='utf-8-sig')
print(f'saved gapfade_trades.csv  n={len(trades)}')

# --- daily portfolio series for equity curve / stats ---
daily = (oos['w'] * oos['net']).groupby(oos['date']).sum()
daily_gross = (oos['w'] * oos['fade_from_open_pct_honest']).groupby(oos['date']).sum()
equity = (1 + daily / 100).cumprod()
maxdd = (equity / equity.cummax() - 1).min()
sharpe = daily.mean() / daily.std() * np.sqrt(252)
ann_ret = equity.iloc[-1] ** (252 / len(daily)) - 1
win_rate = (oos['net'] > 0).mean() * 100

daily_df = pd.DataFrame({
    'date': daily.index, 'daily_net_pct': daily.values, 'daily_gross_pct': daily_gross.reindex(daily.index).values,
    'cum_net': equity.values,
})
daily_df.to_csv(OUT_DIR / 'gapfade_daily.csv', index=False, encoding='utf-8-sig')
print(f'saved gapfade_daily.csv  n={len(daily_df)}')

wins = oos[oos['net'] > 0]['net']
losses = oos[oos['net'] < 0]['net']
profit_factor = wins.sum() / abs(losses.sum())

stats = {
    'period': f'{oos["date"].min().date()} ~ {oos["date"].max().date()} (樣本外)',
    'n_trades': int(len(oos)),
    'n_days': int(oos['date'].nunique()),
    'sharpe': round(float(sharpe), 2),
    'ann_ret_pct': round(float(ann_ret * 100), 1),
    'max_dd_pct': round(float(maxdd * 100), 2),
    'win_rate_pct': round(float(win_rate), 2),
    'profit_factor': round(float(profit_factor), 2),
    'avg_net_pct': round(float(oos['net'].mean()), 3),
    'avg_gross_pct': round(float(oos['fade_from_open_pct_honest'].mean()), 3),
    'cost_pct': ROUND_TRIP_COST,
    'latest_date': str(oos['date'].max().date()),
    'strategy': '全市場開高>=3% | 市值<=76億 | 排除鎖漲停 | 候選>=7才進場 | top5 ML分數加權 | 8%停損(保險)',
    'has_live_signal': False,
}
with open(OUT_DIR / 'gapfade_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print('saved gapfade_stats.json')
print(json.dumps(stats, ensure_ascii=False, indent=2))

# --- 年度統計 ---
oos['year'] = oos['date'].dt.year
yearly = oos.groupby('year').agg(
    n=('net', 'size'),
    mean_net=('net', 'mean'),
    winrate=('net', lambda s: (s > 0).mean() * 100),
).round(3)
yearly.to_csv(OUT_DIR / 'gapfade_yearly.csv', encoding='utf-8-sig')
print(f'saved gapfade_yearly.csv')
print(yearly.to_string())
