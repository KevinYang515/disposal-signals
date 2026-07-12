"""
update_signals.py
每天收盤後執行一次，更新 data/signals.csv 和 data/backtest_grid.csv
用法：python update_signals.py
"""

import pandas as pd
import numpy as np
import warnings, os, json
from datetime import datetime, timezone, timedelta
warnings.filterwarnings("ignore")

TW_TZ = timezone(timedelta(hours=8))

def tw_today():
    """台灣日期，不依賴主機系統時區（VM 系統時區是 UTC，00:01 TW 那次 cron
    若用 datetime.today() 會拿到還沒跨日的 UTC 日期，天數會晚一天才更新）"""
    return pd.Timestamp(datetime.now(TW_TZ).date())

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
BASIC_F  = os.path.expanduser('~/finlab_db/company_basic_info.feather')
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

def exit_date(idx, sd, t1_offset=10):
    pos = idx.searchsorted(sd)
    if pos + t1_offset < len(idx):
        return idx[pos + t1_offset]
    # feather 不夠長時，用 weekday 往後估算（不含假日）
    days_in = int(((idx >= sd) & (idx <= idx[-1])).sum())
    remaining = t1_offset + 1 - days_in  # 到第 t1_offset+1 個交易日（出關日）
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
    today = tw_today()
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

# ── 補上 v2 尚未收錄的新處置事件 ─────────────────────────────────────────
def merge_upcoming(df, price):
    """v2 每晚才重建，且只收「起始日已有收盤價」的事件。
    這裡直接從 finlab disposal_information 補上剛公告、今天/明天才開始的處置，
    讓訊號表在 D0 晚上就看得到（僅供 build_signals 使用）。"""
    try:
        from finlab import data
        disp = pd.DataFrame(data.get('disposal_information'))
    except Exception as e:
        print(f'  讀取 disposal_information 失敗，跳過新事件補充: {e}')
        return df

    d = disp.copy()
    d.columns = [c.strip() for c in d.columns]
    d['stock_id'] = d['stock_id'].astype(str).str.strip()
    d['start_date'] = pd.to_datetime(d['處置開始時間'], errors='coerce')
    d = d[d['stock_id'].str.match(r'^\d{4}$')]
    d = d[d['處置措施'].str.contains('第一次|第二次', na=False)]
    d = d[d['分時交易'].notna()]
    type_map = {5.0: '5分鐘', 20.0: '20分鐘', 25.0: '25分鐘',
                30.0: '30分鐘', 45.0: '45分鐘', 60.0: '60分鐘'}
    d['處置類型'] = d['分時交易'].map(type_map)
    today = tw_today()
    d = d[d['start_date'] >= today - pd.Timedelta(days=20)]
    d = d.drop_duplicates(subset=['stock_id', 'start_date'])

    existing = set(zip(df['股票代號'], df['處置起始日']))
    new = d[[(r.zfill(4), s) not in existing
             for r, s in zip(d['stock_id'], d['start_date'])]]
    if len(new) == 0:
        return df

    shares_map = {}
    if os.path.exists(BASIC_F):
        basic = pd.read_feather(BASIC_F)
        basic['shares'] = pd.to_numeric(
            basic['已發行普通股數或TDR原發行股數'].astype(str).str.replace(',', ''), errors='coerce')
        shares_map = dict(zip(basic['stock_id'].astype(str), basic['shares']))

    idx = price.index
    rows = []
    for _, r in new.iterrows():
        sid = r['stock_id'].zfill(4)
        pr  = prerun20(price, idx, sid, r['start_date'])
        ser = price[sid].dropna() if sid in price.columns else pd.Series(dtype=float)
        p_last = float(ser.iloc[-1]) if len(ser) else np.nan
        shares = shares_map.get(sid, np.nan)
        if pd.notna(p_last) and pd.notna(shares):
            mkt = p_last * shares
            cap = ('大型股(>500億)' if mkt >= 5e10 else
                   '中型股(100~500億)' if mkt >= 1e10 else '小型股(<100億)')
        else:
            cap = '小型股(<100億)'
        rows.append({
            '市值規模':   cap,
            '股票代號':   sid,
            '股票名稱':   r.get('證券名稱', ''),
            '處置類型':   r['處置類型'],
            '處置原因':   '跌深處置' if pd.notna(pr) and pr < 0 else '漲多處置',
            '處置起始日': r['start_date'],
        })
    add = pd.DataFrame(rows)
    names = ', '.join(add['股票代號'] + ' ' + add['股票名稱'].astype(str))
    print(f'  補上 {len(add)} 筆 v2 尚未收錄的新處置事件: {names}')
    return pd.concat([df, add], ignore_index=True)

# ── 5分盤（第一次處置）動能策略 ──────────────────────────────────────────
# 策略：漲多處置 × 5分鐘 × D1收盤買 → 出關D1收盤賣
# 因子：處置起始前一日漲 2~9%（強但未漲停）→ 2022-2026 每年皆正
COST_5MIN = 0.357  # 永豐2折手續費雙邊 + 證交稅 0.3%（非當沖）

def load_disposal_raw():
    """finlab disposal_information；線上失敗時 fallback 本機 feather"""
    try:
        from finlab import data
        disp = pd.DataFrame(data.get('disposal_information'))
    except Exception as e:
        local = os.path.expanduser('~/finlab_db/disposal_information.feather')
        if not os.path.exists(local):
            print(f'  disposal_information 無法取得: {e}')
            return None
        disp = pd.DataFrame(pd.read_feather(local))
    disp.columns = [c.strip() for c in disp.columns]
    disp['stock_id'] = disp['stock_id'].astype(str).str.strip()
    disp = disp[disp['stock_id'].str.match(r'^\d{4}$')]
    disp['start'] = pd.to_datetime(disp['處置開始時間'], errors='coerce')
    disp['end']   = pd.to_datetime(disp['處置結束時間'], errors='coerce')
    return disp

def next_trading_day(idx, after):
    """after（含當天不算）之後第一個交易日；超出資料範圍用平日估算"""
    pos = idx.searchsorted(after, side='right')
    if pos < len(idx):
        return idx[pos]
    cur = max(after, idx[-1])
    while True:
        cur += pd.Timedelta(days=1)
        if cur.weekday() < 5:
            return cur

def build_5min(df, price, open_p):
    disp = load_disposal_raw()
    if disp is None:
        return pd.DataFrame(), pd.DataFrame()

    d5 = disp[(disp['分時交易'] == 5.0) & (disp['start'] >= '2022-01-01')].copy()
    d5 = d5.drop_duplicates(subset=['stock_id', 'start'])

    cap_map = dict(zip(zip(df['股票代號'], df['處置起始日']), df['市值規模']))
    reason_map = dict(zip(zip(df['股票代號'], df['處置起始日']), df['處置原因']))

    idx = price.index
    today = tw_today()
    hist_rows, sig_rows = [], []

    for _, r in d5.iterrows():
        sid, sd, ed = r['stock_id'], r['start'], r['end']
        if sid not in price.columns or pd.isna(ed):
            continue
        pos = idx.searchsorted(sd)
        if pos < 2:
            continue
        ser = price[sid]
        p0 = ser.iloc[pos-1]                     # T-1 收盤
        p_2 = ser.iloc[pos-2]
        if pd.isna(p0) or p0 <= 0 or pd.isna(p_2) or p_2 <= 0:
            continue
        prev1 = round((p0/p_2 - 1) * 100, 2)
        hit = 2 <= prev1 <= 9

        reason = reason_map.get((sid, sd), '')
        cap = cap_map.get((sid, sd), '')
        cap_s = '大' if '大型' in cap else ('中' if '中型' in cap else '小')

        exit_d = next_trading_day(idx, ed)       # 出關D1（實際結束日的下個交易日）

        # D1 進場價（第一個處置日收盤）與 D1 開盤跳空
        e_pos = pos
        entry = ser.iloc[e_pos] if e_pos < len(idx) else np.nan
        gap = np.nan
        if e_pos < len(idx) and sid in open_p.columns:
            o1 = open_p[sid].iloc[e_pos]
            if pd.notna(o1) and o1 > 0:
                gap = round((o1/p0 - 1) * 100, 2)

        # D1 收盤近漲停（>= +9.5% vs 前日收盤）→ 收盤鎖死買不到，訊號作廢
        d1_ret = (entry/p0 - 1) * 100 if pd.notna(entry) and p0 > 0 else np.nan
        d1_locked = pd.notna(d1_ret) and d1_ret >= 9.5

        base = {
            '代號': sid, '名稱': r.get('證券名稱', ''), '規模': cap_s,
            '處置原因': reason if reason else '漲多處置',
            '起始日': sd, '出關日': exit_d,
            '前日漲幅(%)': prev1,
            '符合因子': '✅' if hit else '',
            'D1跳空(%)': gap,
            '加強訊號': ('🔒漲停買不到' if hit and pd.notna(gap) and gap <= 0 and d1_locked
                         else '⭐' if hit and pd.notna(gap) and gap <= 0 and not d1_locked
                         else ''),
        }

        if exit_d <= idx[-1]:
            # ── 已出關 → 歷史 ──
            if reason == '跌深處置':
                continue
            x_pos = idx.searchsorted(exit_d)
            px = ser.iloc[x_pos]
            if pd.isna(entry) or entry <= 0 or pd.isna(px):
                continue
            net = round((px/entry - 1) * 100 - COST_5MIN, 2)
            row = dict(base)
            row['起始日'] = sd.strftime('%Y-%m-%d')
            row['出關日'] = exit_d.strftime('%Y-%m-%d')
            row['年份'] = sd.year
            row['策略報酬(%)'] = net
            for n in [1, 3, 5, 8]:
                if pos + n - 1 < len(idx):
                    pn = ser.iloc[pos+n-1]
                    row[f'D{n}%'] = round((pn/p0-1)*100, 2) if pd.notna(pn) else np.nan
                else:
                    row[f'D{n}%'] = np.nan
            hist_rows.append(row)
        elif today <= exit_d:
            # ── 進行中 / 未開始 → 今日訊號 ──
            if reason == '跌深處置':
                continue
            nd = trading_day_n(idx, sd) if sd <= today else 0
            row = dict(base)
            row['起始日'] = sd.strftime('%m/%d')
            row['出關日'] = exit_d.strftime('%m/%d')
            row['今D幾'] = f'D{nd}' if nd else '未開始'
            # 白話訊號狀態
            if not hit:
                status = '—'
            elif nd == 0 or pd.isna(gap):
                status = '🟡 等D1開盤確認'
            elif gap > 0:
                status = '❌ D1開高，不買'
            elif d1_locked:
                status = '🔒 D1漲停買不到'
            else:
                status = '🟢 買進（D1收盤）'
            row['訊號'] = status
            row['進場價(D1收)'] = round(float(entry), 2) if nd >= 1 and pd.notna(entry) else np.nan
            last = ser.dropna().iloc[-1] if len(ser.dropna()) else np.nan
            if nd >= 1 and pd.notna(entry) and entry > 0 and pd.notna(last):
                row['目前損益(%)'] = round((last/entry - 1) * 100, 2)
            else:
                row['目前損益(%)'] = np.nan
            row['今日漲跌'] = today_change(price, sid)
            sig_rows.append(row)

    hist = pd.DataFrame(hist_rows)
    sig = pd.DataFrame(sig_rows)
    if len(sig):
        order = {'🟢 買進（D1收盤）': 0, '🟡 等D1開盤確認': 1, '🔒 D1漲停買不到': 2,
                 '❌ D1開高，不買': 3, '—': 4}
        sig['_o'] = sig['訊號'].map(order)
        sig = sig.sort_values(['_o', '起始日'], ascending=[True, False]).drop(columns=['_o'])
        sig = sig[['代號', '名稱', '規模', '訊號', '前日漲幅(%)', 'D1跳空(%)',
                   '起始日', '今D幾', '出關日', '進場價(D1收)', '目前損益(%)', '今日漲跌']]
    if len(hist):
        hist = hist[['代號', '名稱', '規模', '年份', '起始日', '出關日', '前日漲幅(%)',
                     'D1跳空(%)', '符合因子', '加強訊號', 'D1%', 'D3%', 'D5%', 'D8%', '策略報酬(%)']]
        hist = hist.sort_values('起始日', ascending=False)
    return sig, hist

# ── 產生訊號表 ───────────────────────────────────────────────────────────
def build_signals(df, price, open_p, whale_dfs):
    idx = price.index
    today = tw_today()
    cutoff = today - pd.Timedelta(days=20)

    active = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)', '小型股(<100億)'])) &
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
            '規模':   '大' if '大型' in row['市值規模'] else ('中' if '中型' in row['市值規模'] else '小'),
            '處置原因': row.get('處置原因', ''),
            '近20日漲幅': prerun20(price, idx, sid, sd),
            '大戶(%)': whale_delta(whale_dfs, sid, sd, price),
            '起始日':  sd.strftime('%m/%d'),
            '今D幾':   f'D{nd}' if sd <= today else '未開始',
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

        is_changduo = row.get('處置原因') == '漲多處置'

        # ── 買進訊號：首個 D3~D8 < -5% 的天（僅漲多處置）──
        entry_n_sig = None
        if is_changduo:
            for n in range(3, 9):
                if pd.notna(d.get(f'D{n}%')) and d[f'D{n}%'] < -5:
                    entry_n_sig = n
                    break
        d['買進訊號'] = f'D{entry_n_sig}' if entry_n_sig else ''

        # ── 當前累積報酬（最新可用的 Dn 值）──
        cur_cum = np.nan
        for n in range(nd, 0, -1):
            v = d.get(f'D{n}%')
            if pd.notna(v):
                cur_cum = v
                break
        d['當前累積(%)'] = cur_cum

        # ── 目前損益：已觸發股票從進場日到當前收盤的未實現損益 ──
        if entry_n_sig and pd.notna(cur_cum):
            entry_cum_v = d.get(f'D{entry_n_sig}%', np.nan)
            if pd.notna(entry_cum_v):
                entry_factor = 1 + entry_cum_v / 100
                cur_factor   = 1 + cur_cum / 100
                d['目前損益(%)'] = round((cur_factor / entry_factor - 1) * 100, 2) if entry_factor > 0 else np.nan
            else:
                d['目前損益(%)'] = np.nan
        else:
            d['目前損益(%)'] = np.nan

        # ── 觸發價與距觸發（僅漲多處置）──
        if is_changduo:
            pos_v = idx.searchsorted(sd)
            p0_v  = price[sid].iloc[pos_v - 1] if pos_v >= 1 and sid in price.columns else np.nan
            ser_v = price[sid].dropna()
            cur_p = float(ser_v.iloc[-1]) if len(ser_v) > 0 else np.nan
            if pd.notna(p0_v) and p0_v > 0 and pd.notna(cur_p) and cur_p > 0:
                d['觸發價']    = round(p0_v * 0.95, 1)
                d['距觸發(%)'] = round((d['觸發價'] / cur_p - 1) * 100, 2)
            else:
                d['觸發價']    = np.nan
                d['距觸發(%)'] = np.nan
        else:
            d['觸發價']    = np.nan
            d['距觸發(%)'] = np.nan

        rows.append(d)

    return pd.DataFrame(rows).sort_values('起始日', ascending=False)

# ── 產生回測網格 ─────────────────────────────────────────────────────────
def build_backtest_grid(df, price, open_p):
    idx = price.index

    base = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)', '小型股(<100億)'])) &
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
    pool = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)', '小型股(<100億)'])) &
              (df['處置類型'].isin(['5分鐘', '20分鐘'])) &
              (df['處置原因'] == '漲多處置')].copy()

    def compute_row(row):
        sid       = row['股票代號']
        sd        = row['處置起始日']
        disp_type = row.get('處置類型', '20分鐘')
        t1_offset = 5 if disp_type == '5分鐘' else 10   # 出關偏移：5分→T+5, 20分→T+10
        entry_rng = range(1, 6) if disp_type == '5分鐘' else range(3, 9)

        out = dict(entry_n=np.nan, entry_cum=np.nan, min_dn=np.nan, deepest_n=np.nan,
                   actual_ret=np.nan,
                   **{f'_t{k}c': np.nan for k in range(1, 11)})
        for n in range(1, 11):
            out[f'_d{n}_cum'] = np.nan
            out[f'_d{n}_ret'] = np.nan
        for n in range(1, 6):
            out[f'_post_d{n}'] = np.nan
        if sid not in price.columns:
            return pd.Series(out)
        pos = idx.searchsorted(sd)
        if pos < 1 or pos + 2 >= len(idx):
            return pd.Series(out)
        p0 = price[sid].iloc[pos - 1]
        if pd.isna(p0) or p0 <= 0:
            return pd.Series(out)

        t1_open = (open_p[sid].iloc[pos + t1_offset]
                   if sid in open_p.columns and pos + t1_offset < len(open_p) else np.nan)
        has_exit = pd.notna(t1_open) and t1_open > 0

        # D1~D(t1_offset) 各天累積報酬與出關報酬
        all_rets = {}
        for n in range(1, t1_offset + 1):
            if pos + n - 1 < len(price):
                pn = price[sid].iloc[pos + n - 1]
                if pd.notna(pn) and pn > 0:
                    cum = round((pn / p0 - 1) * 100, 2)
                    out[f'_d{n}_cum'] = cum
                    all_rets[n] = cum
                    if has_exit:
                        out[f'_d{n}_ret'] = round((t1_open / pn - 1) * 100, 2)

        dn_rets = {n: all_rets[n] for n in entry_rng if n in all_rets}
        if not dn_rets:
            return pd.Series(out)

        deepest = min(dn_rets, key=lambda n: dn_rets[n])
        out['min_dn']    = round(dn_rets[deepest], 2)
        out['deepest_n'] = deepest

        for n in sorted(entry_rng):
            if n in dn_rets and dn_rets[n] < -5:
                out['entry_n']   = n
                out['entry_cum'] = round(dn_rets[n], 2)
                if has_exit:
                    pn = price[sid].iloc[pos + n - 1]
                    out['actual_ret'] = round((t1_open / pn - 1) * 100, 2)
                break

        ref_n = int(out['entry_n']) if pd.notna(out['entry_n']) else (1 if disp_type == '5分鐘' else 3)
        if ref_n in all_rets:
            p_ref = price[sid].iloc[pos + ref_n - 1]
            for k in range(1, 11):
                off = (t1_offset - 1) + k  # T+1=pos+t1_offset, T+2=pos+t1_offset+1...
                if pos + off < len(price):
                    p = price[sid].iloc[pos + off]
                    if pd.notna(p) and p > 0:
                        out[f'_t{k}c'] = round((p / p_ref - 1) * 100, 2)

        # 出關後 D1~D5 收盤（基準：T+1 開盤）
        if has_exit:
            for n in range(1, 6):
                if pos + t1_offset + n < len(price):
                    p = price[sid].iloc[pos + t1_offset + n]
                    if pd.notna(p) and p > 0:
                        out[f'_post_d{n}'] = round((p / t1_open - 1) * 100, 2)

        return pd.Series(out)

    stats = pool.apply(compute_row, axis=1)
    pool  = pd.concat([pool, stats], axis=1)

    def dn_group(v):
        if pd.isna(v):  return 'Dn無資料'
        if v < -5:      return 'Dn < -5%'
        if v < 0:       return 'Dn -5%~0%'
        return 'Dn ≥ 0%'

    pool['Dn組別'] = pool['min_dn'].apply(dn_group)

    pool['_exit_date'] = pool.apply(
        lambda r: exit_date(idx, r['處置起始日'], t1_offset=5 if r['處置類型'] == '5分鐘' else 10), axis=1)

    out = pd.DataFrame({
        '起始日':        pool['處置起始日'].dt.strftime('%Y-%m-%d'),
        '出關日':        pool['_exit_date'].apply(lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else '?'),
        '處置類型':      pool['處置類型'],
        '代號':          pool['股票代號'],
        '名稱':          pool['股票名稱'],
        '規模':          pool['市值規模'].apply(lambda v: '大' if '大型' in str(v) else ('中' if '中型' in str(v) else '小')),
        'Dn組別':        pool['Dn組別'],
        '近20日漲幅':    pool.apply(lambda r: prerun20(price, idx, r['股票代號'], r['處置起始日']), axis=1).round(2),
        '大戶(%)':       pool.apply(lambda r: whale_delta(whale_dfs, r['股票代號'], r['處置起始日'], price), axis=1),
        '買進日':        pool['entry_n'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '買進時累積(%)': pool['entry_cum'],
        '最深日':        pool['deepest_n'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '期間最深(%)':   pool['min_dn'],
        '出關報酬(%)':   pool['actual_ret'],
        **{f'T+{k}收盤(%)': pool[f'_t{k}c'] for k in range(1, 11)},
        '結果':          pool['actual_ret'].apply(
            lambda v: f'✅ {v:+.2f}%' if pd.notna(v) and v > 0
                      else (f'❌ {v:+.2f}%' if pd.notna(v) else '-')),
        # D1~D10 各天進場累積報酬與出關報酬（自訂策略回測頁使用）
        **{f'D{n}累積(%)': pool[f'_d{n}_cum'] for n in range(1, 11)},
        **{f'D{n}報酬(%)': pool[f'_d{n}_ret'] for n in range(1, 11)},
        # 出關後 D1~D5 收盤報酬（基準：T+1 開盤）
        **{f'出關後D{n}(%)': pool[f'_post_d{n}'] for n in range(1, 6)},
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
    sig_df = merge_upcoming(df, price)
    sig = build_signals(sig_df, price, open_p, whale_dfs)
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

    print('產生 5分盤動能資料...')
    sig5, hist5 = build_5min(df, price, open_p)
    sig5.to_csv(f'{OUT_DIR}/signals_5min.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    hist5.to_csv(f'{OUT_DIR}/history_5min.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'  → signals_5min.csv ({len(sig5)} 筆) / history_5min.csv ({len(hist5)} 筆)')

    # 更新時間
    meta = {
        'updated_at': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M'),
        'data_date': price.index[-1].strftime('%Y-%m-%d'),
        'cmp_stats': cmp_stats,
    }
    with open(f'{OUT_DIR}/meta.json', 'w') as f:
        json.dump(meta, f)
    print(f'  → meta.json')
    print('完成')

if __name__ == '__main__':
    main()
