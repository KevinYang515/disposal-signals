"""
update_signals.py
每天收盤後執行一次，更新 data/signals.csv 和 data/backtest_grid.csv
用法：python update_signals.py
"""

import pandas as pd
import numpy as np
import warnings, os, json
from datetime import datetime
warnings.filterwarnings("ignore")

V2_CSV   = os.path.join(os.path.dirname(__file__), '../disposal_data_v2.csv')
PRICE_F  = os.path.expanduser('~/finlab_db/price#收盤價.feather')
OPEN_F   = os.path.expanduser('~/finlab_db/price#開盤價.feather')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT_DIR, exist_ok=True)

ENTRY_COL = '買進D3_出關D1賣出(%)'

# ── 讀資料 ──────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(V2_CSV)
    df['股票代號']  = df['股票代號'].astype(str).str.zfill(4)
    df['處置起始日'] = pd.to_datetime(df['處置起始日'])
    for col in ['大戶持股變動(%)', '入場前20日漲幅(%)', 'D3收盤報酬(%)',
                'd4r', 'D5收盤報酬(%)', ENTRY_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_price():
    price = pd.DataFrame(pd.read_feather(PRICE_F))
    price['date'] = pd.to_datetime(price['date'])
    price = price.set_index('date').sort_index()

    open_p = pd.DataFrame(pd.read_feather(OPEN_F))
    open_p['date'] = pd.to_datetime(open_p['date'])
    open_p = open_p.set_index('date').sort_index()
    return price, open_p

# ── 工具函數 ────────────────────────────────────────────────────────────
def cumret(price, idx, sid, sd, nd):
    pos = idx.searchsorted(sd)
    if pos < 1 or pos + nd > len(idx) or sid not in price.columns:
        return np.nan
    p0 = price[sid].iloc[pos - 1]
    pn = price[sid].iloc[pos + nd - 1]
    if p0 <= 0 or np.isnan(p0) or np.isnan(pn):
        return np.nan
    return round((pn / p0 - 1) * 100, 2)

def exit_date(idx, sd):
    pos = idx.searchsorted(sd)
    if pos + 10 < len(idx):
        return idx[pos + 10]
    return pd.NaT

def trading_day_n(idx, sd):
    today = pd.Timestamp(datetime.today().date())
    count = int(((idx >= sd) & (idx <= today)).sum())
    return min(count, 10)

def grade(row):
    reason  = row.get('處置原因', '')
    prerun  = row.get('入場前20日漲幅(%)', np.nan)
    d3r     = row.get('D3收盤報酬(%)', np.nan)
    whale   = row.get('大戶持股變動(%)', np.nan)

    if reason == '漲多處置' and not np.isnan(d3r) and d3r < -5:
        if not np.isnan(whale) and whale < -1.5:
            return '⚠️ 漲多但大戶減碼'
        return '✅ 主力訊號'
    if reason == '跌深處置' and not np.isnan(prerun) and prerun < 0:
        return '❌ 避開'
    if reason == '漲多處置' and not np.isnan(d3r) and d3r < -3:
        return '🟡 觀察中'
    if reason == '漲多處置':
        return '⬜ 待觀察'
    return '🟡 觀察中'

# ── 產生訊號表 ───────────────────────────────────────────────────────────
def build_signals(df, price, open_p):
    idx = price.index
    today = pd.Timestamp(datetime.today().date())
    cutoff = today - pd.Timedelta(days=20)

    active = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)'])) &
                (df['處置類型'] == '20分鐘') &
                (df['處置起始日'] >= cutoff)].copy()

    rows = []
    for _, row in active.iterrows():
        sid = row['股票代號']
        sd  = row['處置起始日']
        nd  = trading_day_n(idx, sd)
        ex  = exit_date(idx, sd)

        d = {
            '代號':   sid,
            '名稱':   row['股票名稱'],
            '規模':   '大' if '大型' in row['市值規模'] else '中',
            '處置原因': row.get('處置原因', ''),
            'prerun': round(row.get('入場前20日漲幅(%)', np.nan), 1),
            '大戶(%)': round(row.get('大戶持股變動(%)', np.nan), 2),
            '起始日':  sd.strftime('%m/%d'),
            '今D幾':   f'D{nd}',
            '出關日':  ex.strftime('%m/%d') if pd.notna(ex) else '?',
        }
        for n in range(1, 9):
            v = cumret(price, idx, sid, sd, n)
            d[f'D{n}%'] = v
        d['評級'] = grade({**row.to_dict(), 'D3收盤報酬(%)': d.get('D3%')})
        rows.append(d)

    return pd.DataFrame(rows).sort_values('起始日', ascending=False)

# ── 產生回測網格 ─────────────────────────────────────────────────────────
def build_backtest_grid(df, price, open_p):
    idx = price.index

    base = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)'])) &
              (df['處置類型'] == '20分鐘') &
              (df['處置原因'] == '漲多處置')].copy()

    results = []
    for _, row in base.iterrows():
        sid = row['股票代號']
        sd  = row['處置起始日']
        if sid not in price.columns:
            continue
        pos = idx.searchsorted(sd)
        if pos < 1 or pos + 11 >= len(idx):
            continue
        p0 = price[sid].iloc[pos - 1]
        if p0 <= 0 or np.isnan(p0):
            continue
        t1_open = open_p[sid].iloc[pos + 10] if sid in open_p.columns else np.nan
        if np.isnan(t1_open) or t1_open <= 0:
            continue
        rec = {}
        for n in range(1, 9):
            dn_close = price[sid].iloc[pos + n - 1]
            if dn_close > 0 and not np.isnan(dn_close):
                rec[f'd{n}_cum']   = round((dn_close / p0 - 1) * 100, 2)
                rec[f'd{n}_entry'] = round((t1_open / dn_close - 1) * 100, 2)
            else:
                rec[f'd{n}_cum']   = np.nan
                rec[f'd{n}_entry'] = np.nan
        results.append(rec)

    res = pd.DataFrame(results)
    rows = []
    for n in range(1, 9):
        for th in [-3, -5, -7, -10, -15]:
            sub = res[res[f'd{n}_cum'] < th][f'd{n}_entry'].dropna()
            if len(sub) < 5:
                continue
            wins = sub[sub > 0]
            loss = sub[sub < 0]
            wl = wins.mean() / abs(loss.mean()) if len(loss) > 0 and loss.mean() != 0 else np.nan
            rows.append({
                'day': n, 'threshold': th,
                'label': f'D{n}<{th:+d}%',
                'N': len(sub),
                'wr': round(len(wins) / len(sub) * 100, 1),
                'ret': round(sub.mean(), 2),
                'wl': round(wl, 2) if not np.isnan(wl) else None,
            })
    return pd.DataFrame(rows)

# ── 產生歷史回測紀錄 ──────────────────────────────────────────────────────
def build_history(df):
    ENTRY = '買進D3_出關D1賣出(%)'
    base = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)'])) &
              (df['處置類型'] == '20分鐘') &
              (df['處置原因'] == '漲多處置') &
              (df['D3收盤報酬(%)'].notna()) &
              (df['D3收盤報酬(%)'] < -5) &
              (df[ENTRY].notna())].copy()

    out = pd.DataFrame({
        '起始日':   base['處置起始日'].dt.strftime('%Y-%m-%d'),
        '代號':     base['股票代號'],
        '名稱':     base['股票名稱'],
        '規模':     base['市值規模'].str.extract(r'^(.+?)\(')[0],
        'prerun':   base['入場前20日漲幅(%)'].round(2),
        'D3累積(%)': base['D3收盤報酬(%)'].round(2),
        '大戶(%)':  base['大戶持股變動(%)'].round(2),
        '出關報酬(%)': base[ENTRY].round(2),
        '結果':     base[ENTRY].apply(lambda v: '✅ 獲利' if v > 0 else '❌ 虧損'),
    })
    return out.sort_values('起始日', ascending=False).reset_index(drop=True)

# ── main ────────────────────────────────────────────────────────────────
def main():
    print('載入資料...')
    df = load()
    price, open_p = load_price()

    print('產生訊號表...')
    sig = build_signals(df, price, open_p)
    sig.to_csv(f'{OUT_DIR}/signals.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'  → signals.csv ({len(sig)} 筆)')

    print('產生回測網格...')
    grid = build_backtest_grid(df, price, open_p)
    grid.to_csv(f'{OUT_DIR}/backtest_grid.csv', index=False, encoding='utf-8-sig')
    print(f'  → backtest_grid.csv ({len(grid)} 筆)')

    print('產生歷史回測紀錄...')
    hist = build_history(df)
    hist.to_csv(f'{OUT_DIR}/history.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'  → history.csv ({len(hist)} 筆)')

    # 更新時間
    meta = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M')}
    with open(f'{OUT_DIR}/meta.json', 'w') as f:
        json.dump(meta, f)
    print(f'  → meta.json')
    print('完成 ✅')

if __name__ == '__main__':
    main()
