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
WHALE_FILES = {
    100:  os.path.expanduser('~/finlab_db/etl#inventory#大於一百張佔比.feather'),
    200:  os.path.expanduser('~/finlab_db/etl#inventory#大於二百張佔比.feather'),
    400:  os.path.expanduser('~/finlab_db/etl#inventory#大於四百張佔比.feather'),
    800:  os.path.expanduser('~/finlab_db/etl#inventory#大於八百張佔比.feather'),
    1000: os.path.expanduser('~/finlab_db/etl#inventory#大於一千張佔比.feather'),
}
_WHALE_THRESHOLDS = sorted(WHALE_FILES)
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT_DIR, exist_ok=True)

ENTRY_COL = '買進D3_出關D1賣出(%)'

# ── 從 finlab 刷新價格資料 ────────────────────────────────────────────────
def refresh_finlab():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith('FINLAB_TOKEN='):
                    token = line.strip().split('=', 1)[1]
    if not token:
        print('  未找到 FINLAB_TOKEN，跳過刷新')
        return
    try:
        import finlab
        finlab.login(token)
        from finlab import data
        print('  更新 price:收盤價...')
        data.get('price:收盤價')
        print('  更新 price:開盤價...')
        data.get('price:開盤價')
        print('  更新大戶持股資料 (5個門檻)...')
        for zh in ['一百', '二百', '四百', '八百', '一千']:
            data.get(f'etl:inventory:大於{zh}張佔比')
        print('  finlab 資料刷新完成')
    except Exception as e:
        print(f'  finlab 刷新失敗（將使用現有 feather）: {e}')

# ── 讀資料 ──────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(V2_CSV)
    df['股票代號']  = df['股票代號'].astype(str).str.zfill(4)
    df['處置起始日'] = pd.to_datetime(df['處置起始日'])
    for col in ['大戶持股變動(%)', 'D3收盤報酬(%)', 'D5收盤報酬(%)']:
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

    whale_dfs = {}
    for t, path in WHALE_FILES.items():
        if os.path.exists(path):
            w = pd.DataFrame(pd.read_feather(path))
            w['date'] = pd.to_datetime(w['date'])
            whale_dfs[t] = w.set_index('date').sort_index()
    return price, open_p, whale_dfs

# ── 工具函數 ────────────────────────────────────────────────────────────
def whale_delta(whale_dfs, sid, sd, price_df=None):
    """大戶持股比例變動：依持股市值 5000萬 動態回推張數門檻"""
    threshold = 400
    if price_df is not None and sid in price_df.columns:
        pos_w = price_df.index.searchsorted(sd)
        if pos_w >= 1:
            p0 = price_df[sid].iloc[pos_w - 1]
            if pd.notna(p0) and p0 > 0:
                lots = 50_000 / p0  # 5000萬 / (每股價格 × 1000股/張)
                threshold = min(_WHALE_THRESHOLDS, key=lambda t: abs(t - lots))
    whale = whale_dfs.get(threshold)
    if whale is None:
        whale = next(iter(whale_dfs.values()), None)
    if whale is None or sid not in whale.columns:
        return np.nan
    series = whale[sid].dropna()
    before = series[series.index < sd]
    after  = series[series.index >= sd]
    if len(before) == 0 or len(after) == 0:
        return np.nan
    return round(float(after.iloc[-1]) - float(before.iloc[-1]), 2)

def prerun20(price, idx, sid, sd):
    """處置起始日前 20 個交易日的累積漲幅"""
    pos = idx.searchsorted(sd)
    if pos < 21 or sid not in price.columns:
        return np.nan
    p0 = price[sid].iloc[pos - 21]
    p1 = price[sid].iloc[pos - 1]
    if p0 <= 0 or np.isnan(p0) or np.isnan(p1):
        return np.nan
    return round((p1 / p0 - 1) * 100, 2)

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
    # feather 不夠長時，用 weekday 往後估算（不含假日）
    days_in = int(((idx >= sd) & (idx <= idx[-1])).sum())
    remaining = 11 - days_in  # 到第 11 個交易日（出關日）
    if remaining <= 0:
        return idx[-1]
    cur = idx[-1]
    cnt = 0
    while cnt < remaining:
        cur += pd.Timedelta(days=1)
        if cur.weekday() < 5:
            cnt += 1
    return cur

def trading_day_n(idx, sd):
    today = pd.Timestamp(datetime.today().date())
    last_data = idx[-1]
    # Count trading days in feather from sd to last available date
    count = int(((idx >= sd) & (idx <= last_data)).sum())
    # Add weekdays between last_data+1 and today (approximation for missing trading days)
    if today > last_data:
        extra = sum(1 for d in pd.date_range(last_data + pd.Timedelta(days=1), today)
                    if d.weekday() < 5)
        count += extra
    return min(count, 10)

def today_change(price, sid):
    if sid not in price.columns:
        return np.nan
    series = price[sid].dropna()
    if len(series) < 2:
        return np.nan
    p1, p0 = series.iloc[-1], series.iloc[-2]
    if p0 <= 0 or np.isnan(p0):
        return np.nan
    return round((p1 / p0 - 1) * 100, 2)

def grade(row):
    reason  = row.get('處置原因', '')
    prerun  = row.get('入場前20日漲幅(%)', np.nan)
    whale   = row.get('大戶持股變動(%)', np.nan)
    dn_vals = [row.get(f'D{n}%') for n in range(3, 9)]

    if reason == '漲多處置':
        any_entry = any(pd.notna(v) and v < -5 for v in dn_vals)
        any_watch = any(pd.notna(v) and v < -3 for v in dn_vals)
        if any_entry:
            if pd.notna(whale) and whale < -1.5:
                return '⚠️ 漲多但大戶減碼'
            return '✅ 主力訊號'
        if any_watch:
            return '🟡 觀察中'
        return '⬜ 待觀察'
    if reason == '跌深處置' and pd.notna(prerun) and prerun < 0:
        return '❌ 避開'
    return '🟡 觀察中'

# ── 產生訊號表 ───────────────────────────────────────────────────────────
def build_signals(df, price, open_p, whale_dfs):
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
        ex  = exit_date(idx, sd)

        # 已出關（今天超過 T+1）就跳過
        if pd.notna(ex) and today > ex:
            continue

        nd  = trading_day_n(idx, sd)

        d = {
            '代號':   sid,
            '名稱':   row['股票名稱'],
            '規模':   '大' if '大型' in row['市值規模'] else '中',
            '處置原因': row.get('處置原因', ''),
            '近20日漲幅': prerun20(price, idx, sid, sd),
            '大戶(%)': whale_delta(whale_dfs, sid, sd, price),
            '起始日':  sd.strftime('%m/%d'),
            '今D幾':   f'D{nd}',
            '出關日':  ex.strftime('%m/%d') if pd.notna(ex) else '?',
        }
        for n in range(1, 9):
            v = cumret(price, idx, sid, sd, n)
            d[f'D{n}%'] = v
        d['今日漲跌'] = today_change(price, sid)
        pr = d['近20日漲幅']
        wh = d['大戶(%)']
        d['評級'] = grade({**row.to_dict(), '入場前20日漲幅(%)': pr, '大戶持股變動(%)': wh,
                           **{f'D{n}%': d.get(f'D{n}%') for n in range(1, 9)}})
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
def build_history(df, price, open_p, whale_dfs):
    idx = price.index
    pool = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)'])) &
              (df['處置類型'] == '20分鐘') &
              (df['處置原因'] == '漲多處置')].copy()

    def compute_row(row):
        sid = row['股票代號']
        sd  = row['處置起始日']
        out = dict(entry_n=np.nan, entry_cum=np.nan, min_dn=np.nan, deepest_n=np.nan,
                   actual_ret=np.nan, _t1c=np.nan, _t2c=np.nan, _t3c=np.nan)
        for n in range(1, 11):
            out[f'_d{n}_cum'] = np.nan
            out[f'_d{n}_ret'] = np.nan
        if sid not in price.columns:
            return pd.Series(out)
        pos = idx.searchsorted(sd)
        if pos < 1 or pos + 2 >= len(idx):
            return pd.Series(out)
        p0 = price[sid].iloc[pos - 1]
        if pd.isna(p0) or p0 <= 0:
            return pd.Series(out)

        t1_open = (open_p[sid].iloc[pos + 10]
                   if sid in open_p.columns and pos + 10 < len(open_p) else np.nan)
        has_exit = pd.notna(t1_open) and t1_open > 0

        # 計算 D1~D10 各天累積報酬與出關報酬（自訂策略頁使用）
        all_rets = {}
        for n in range(1, 11):
            if pos + n - 1 < len(price):
                pn = price[sid].iloc[pos + n - 1]
                if pd.notna(pn) and pn > 0:
                    cum = round((pn / p0 - 1) * 100, 2)
                    out[f'_d{n}_cum'] = cum
                    all_rets[n] = cum
                    if has_exit:
                        out[f'_d{n}_ret'] = round((t1_open / pn - 1) * 100, 2)

        # D3~D8 用於主策略（D1/D2 跌深屬真實賣壓）
        dn_rets = {n: all_rets[n] for n in range(3, 9) if n in all_rets}
        if not dn_rets:
            return pd.Series(out)

        deepest = min(dn_rets, key=lambda n: dn_rets[n])
        out['min_dn']    = round(dn_rets[deepest], 2)
        out['deepest_n'] = deepest

        for n in range(3, 9):
            if n in dn_rets and dn_rets[n] < -5:
                out['entry_n']   = n
                out['entry_cum'] = round(dn_rets[n], 2)
                if has_exit:
                    pn = price[sid].iloc[pos + n - 1]
                    out['actual_ret'] = round((t1_open / pn - 1) * 100, 2)
                break

        ref_n = int(out['entry_n']) if pd.notna(out['entry_n']) else 3
        if ref_n in all_rets:
            p_ref = price[sid].iloc[pos + ref_n - 1]
            for off, key in [(10, '_t1c'), (11, '_t2c'), (12, '_t3c')]:
                if pos + off < len(price):
                    p = price[sid].iloc[pos + off]
                    if pd.notna(p) and p > 0:
                        out[key] = round((p / p_ref - 1) * 100, 2)

        return pd.Series(out)

    stats = pool.apply(compute_row, axis=1)
    pool  = pd.concat([pool, stats], axis=1)

    def dn_group(v):
        if pd.isna(v):  return 'Dn無資料'
        if v < -5:      return 'Dn < -5%'
        if v < 0:       return 'Dn -5%~0%'
        return 'Dn ≥ 0%'

    pool['Dn組別'] = pool['min_dn'].apply(dn_group)

    out = pd.DataFrame({
        '起始日':        pool['處置起始日'].dt.strftime('%Y-%m-%d'),
        '代號':          pool['股票代號'],
        '名稱':          pool['股票名稱'],
        '規模':          pool['市值規模'].apply(lambda v: '大' if '大型' in str(v) else '中'),
        'Dn組別':        pool['Dn組別'],
        '近20日漲幅':    pool.apply(lambda r: prerun20(price, idx, r['股票代號'], r['處置起始日']), axis=1).round(2),
        '大戶(%)':       pool.apply(lambda r: whale_delta(whale_dfs, r['股票代號'], r['處置起始日'], price), axis=1),
        '買進日':        pool['entry_n'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '買進時累積(%)': pool['entry_cum'],
        '最深日':        pool['deepest_n'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '期間最深(%)':   pool['min_dn'],
        '出關報酬(%)':   pool['actual_ret'],
        'T+1收盤(%)':    pool['_t1c'],
        'T+2收盤(%)':    pool['_t2c'],
        'T+3收盤(%)':    pool['_t3c'],
        '結果':          pool['actual_ret'].apply(
            lambda v: f'✅ {v:+.2f}%' if pd.notna(v) and v > 0
                      else (f'❌ {v:+.2f}%' if pd.notna(v) else '-')),
        # D1~D10 各天進場累積報酬與出關報酬（自訂策略回測頁使用）
        **{f'D{n}累積(%)': pool[f'_d{n}_cum'] for n in range(1, 11)},
        **{f'D{n}報酬(%)': pool[f'_d{n}_ret'] for n in range(1, 11)},
    })
    hist = out.sort_values('起始日', ascending=False).reset_index(drop=True)

    # ── 產生比較組統計 ──
    def grp_stats(sub, label):
        s = sub['actual_ret'].dropna()
        if len(s) == 0: return None
        return {
            'label': label,
            'n': len(s),
            'wr': round((s > 0).mean() * 100, 1),
            'ret': round(s.mean(), 2),
        }

    cmp_stats = [
        grp_stats(pool[pool['min_dn'] < -5],                              'Dn最深 < -5%（進場）'),
        grp_stats(pool[(pool['min_dn'] >= -5) & (pool['min_dn'] < 0)],   'Dn最深 -5%~0%（觀察中）'),
        grp_stats(pool[pool['min_dn'] >= 0],                               'Dn最深 ≥ 0%（無跌幅）'),
        grp_stats(pool,                                                    '全部漲多（不篩選Dn）'),
    ]
    cmp_stats = [x for x in cmp_stats if x]

    return hist, cmp_stats

# ── main ────────────────────────────────────────────────────────────────
def main():
    print('刷新 finlab 價格資料...')
    refresh_finlab()

    print('載入資料...')
    df = load()
    price, open_p, whale_dfs = load_price()

    print('產生訊號表...')
    sig = build_signals(df, price, open_p, whale_dfs)
    sig.to_csv(f'{OUT_DIR}/signals.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'  → signals.csv ({len(sig)} 筆)')

    print('產生回測網格...')
    grid = build_backtest_grid(df, price, open_p)
    grid.to_csv(f'{OUT_DIR}/backtest_grid.csv', index=False, encoding='utf-8-sig')
    print(f'  → backtest_grid.csv ({len(grid)} 筆)')

    print('產生歷史回測紀錄...')
    hist, cmp_stats = build_history(df, price, open_p, whale_dfs)
    hist.to_csv(f'{OUT_DIR}/history.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'  → history.csv ({len(hist)} 筆)')

    # 更新時間
    meta = {
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_date': price.index[-1].strftime('%Y-%m-%d'),
        'cmp_stats': cmp_stats,
    }
    with open(f'{OUT_DIR}/meta.json', 'w') as f:
        json.dump(meta, f)
    print(f'  → meta.json')
    print('完成 ✅')

if __name__ == '__main__':
    main()
