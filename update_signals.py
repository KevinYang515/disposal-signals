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
LOW_F    = os.path.expanduser('~/finlab_db/price#最低價.feather')
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

_LOW_CACHE = None

def load_low():
    """最低價，只用在第二次+的-5%回檔判斷（改用當天最低價觸及，不再只看收盤）。
    獨立小函式、懶載入快取，不改動load_price()既有回傳簽名，避免動到所有呼叫端。"""
    global _LOW_CACHE
    if _LOW_CACHE is None:
        low = pd.DataFrame(pd.read_feather(LOW_F))
        low['date'] = pd.to_datetime(low['date'])
        _LOW_CACHE = low.set_index('date').sort_index()
    return _LOW_CACHE


def round_to_tick(price):
    """台股法定升降單位。觸發價(D0收盤×0.95)是用百分比乘出來的價格，不對齊會出現
    像876.85這種不存在的價位（500~1000元區間法定跳動單位是1元）。pages/18裡也有
    同一份函式（自選窗口重算用），這裡是後端固定規則(D3~D5等)用，兩邊各自獨立維護，
    邏輯要保持一致。2026-08-30 Kevin抓到這個問題。"""
    if price is None or pd.isna(price) or price <= 0:
        return price
    if price < 10:
        tick = 0.01
    elif price < 50:
        tick = 0.05
    elif price < 100:
        tick = 0.1
    elif price < 500:
        tick = 0.5
    elif price < 1000:
        tick = 1
    else:
        tick = 5
    return round(round(price / tick) * tick, 2)

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
    # 2026-08-25：分時交易偶爾解析失敗留空（例如3625西勝：處置內容原文「約每10分鐘
    # 撮合一次」，分時交易卻是NaN），用原文文字補值當後備，跟build_disposal_data.py同步修正。
    if '處置內容' in d.columns:
        missing = d['分時交易'].isna()
        if missing.any():
            extracted = d.loc[missing, '處置內容'].str.extract(r'每\s*(\d+)\s*分鐘撮合')[0]
            d.loc[missing, '分時交易'] = pd.to_numeric(extracted, errors='coerce')
    d = d[d['分時交易'].notna()]
    # 直接用數字格式化，不用固定字典，避免每次遇到新搓合頻率就要手動補一筆
    d['處置類型'] = d['分時交易'].apply(lambda v: f'{v:g}分鐘')
    d['處置次別'] = d['處置措施'].apply(lambda v: '第一次' if '第一次' in str(v) else '第二次+')
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
            '處置次別':   r['處置次別'],
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

def td_count_between(idx, a, b):
    """交易日數 in (a, b]；超出價格資料範圍的未來部分用平日估算"""
    if pd.isna(b) or b <= a:
        return 0
    last = idx[-1]
    n = int(((idx > a) & (idx <= min(b, last))).sum())
    if b > last:
        cur = max(a, last)
        while cur < b:
            cur += pd.Timedelta(days=1)
            if cur.weekday() < 5 and cur <= b:
                n += 1
    return n

def est_prev_td(idx, ref, k):
    """ref 往回第 k 個交易日；ref 在未來時用平日估算銜接"""
    if ref <= idx[-1]:
        p = idx.searchsorted(ref)
        return idx[p-k] if p - k >= 0 else None
    fut, cur = [], idx[-1]
    while cur < ref:
        cur += pd.Timedelta(days=1)
        if cur.weekday() < 5:
            fut.append(cur)
    chain = list(idx[-15:]) + fut
    try:
        i = chain.index(ref)
    except ValueError:
        return None
    return chain[i-k] if i - k >= 0 else None

def classify_clause(row):
    """處置觸發條款分類：當沖比重條款統計上為負（全樣本 -1.43%），排除不買"""
    txt = str(row.get('處置條件', '')) + ' ' + str(row.get('處置內容', ''))
    daytrade = ('當沖' in txt) or ('當日沖銷' in txt)
    if '督導會報' in txt:
        base = '督導會報'
    elif ('十個營業日' in txt and '六' in txt) or '10個營業日內有6' in txt or '六次' in txt:
        base = '10日6次'
    elif '連續三' in txt or '連續3' in txt:
        base = '連續3日'
    elif '連續5' in txt or '連續五' in txt:
        base = '連續5日'
    else:
        base = '其他'
    return base + ('+當沖' if daytrade else '')

SIG5_COLS = ['代號', '名稱', '規模', '訊號', '條款', '前日漲幅(%)', 'D1跳空(%)',
             '起始日', '今D幾', '出關日', '進場價(D1收)', '目前損益(%)', '今日漲跌']
HIST5_COLS = ['代號', '名稱', '規模', '年份', '起始日', '出關日', '條款', '前日漲幅(%)',
              'D1跳空(%)', '符合因子', '加強訊號', 'D1%', 'D3%', 'D5%', 'D8%', '策略報酬(%)']

def build_5min(df, price, open_p):
    disp = load_disposal_raw()
    if disp is None:
        return pd.DataFrame(columns=SIG5_COLS), pd.DataFrame(columns=HIST5_COLS)

    d5 = disp[(disp['分時交易'] == 5.0) & (disp['start'] >= '2022-01-01')].copy()
    d5 = d5.drop_duplicates(subset=['stock_id', 'start'])
    d5['條款'] = d5.apply(classify_clause, axis=1)

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
        # 當沖比重條款觸發的處置：統計為負（全樣本 -1.43%、⭐內 -1.15%）→ 不買
        clause = r.get('條款', '')
        is_daytrade_clause = '當沖' in clause

        base = {
            '代號': sid, '名稱': r.get('證券名稱', ''), '規模': cap_s,
            '處置原因': reason if reason else '漲多處置',
            '條款': clause,
            '起始日': sd, '出關日': exit_d,
            '前日漲幅(%)': prev1,
            '符合因子': '✅' if hit else '',
            'D1跳空(%)': gap,
            '加強訊號': ('🚫當沖條款' if hit and pd.notna(gap) and gap <= 0 and is_daytrade_clause
                         else '🔒漲停買不到' if hit and pd.notna(gap) and gap <= 0 and d1_locked
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
            elif is_daytrade_clause:
                status = '🚫 當沖條款，不買'
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
                 '🚫 當沖條款，不買': 3, '❌ D1開高，不買': 4, '—': 5}
        sig['_o'] = sig['訊號'].map(order)
        sig = sig.sort_values(['_o', '起始日'], ascending=[True, False]).drop(columns=['_o'])
    # reindex（而非只在 len(sig) 時挑欄位）：0筆時也要保留正確欄位標頭，
    # 否則 to_csv 會寫出完全空白的檔案，Streamlit 端 pd.read_csv 會拋 EmptyDataError 讓整個 app 掛掉。
    sig = sig.reindex(columns=SIG5_COLS)
    if len(hist):
        hist = hist.sort_values('起始日', ascending=False)
    hist = hist.reindex(columns=HIST5_COLS)
    return sig, hist

SIG_T20_COLS = ['代號', '名稱', '規模', '訊號', '買進日', '賣出日(出關)', '深跌單',
                '進場價', '目前損益(%)', '今日漲跌']
HIST_T20_COLS = ['代號', '名稱', '規模', '年份', '起始日', '買進日', '出關日', '深跌單',
                 '訊號', '進場價', '出場價', '策略報酬(%)']

def build_tail20(df, price, hist20):
    """20分盤出關動能：出關前第3個交易日收盤買 → 出關當天(恢復正常交易首日)收盤賣
    回測 2022+ n=485 +3.94%/筆 t=7.81；OOS 2016-2021 +2.87% 6/6年正"""
    disp = load_disposal_raw()
    if disp is None:
        return pd.DataFrame(columns=SIG_T20_COLS), pd.DataFrame(columns=HIST_T20_COLS)
    d20 = disp[(disp['分時交易'] >= 20.0) & (disp['start'] >= '2022-01-01')].copy()
    d20 = d20.drop_duplicates(subset=['stock_id', 'start']).dropna(subset=['start', 'end'])

    cap_map = dict(zip(zip(df['股票代號'], df['處置起始日']), df['市值規模']))
    # 現行20分盤買深跌策略是否也進場（同事件重疊參考）
    dip_keys = set()
    if hist20 is not None and len(hist20) and '買進日' in hist20.columns:
        hh = hist20[hist20['買進日'].astype(str) != '-']
        dip_keys = set(zip(hh['代號'].astype(str).str.zfill(4),
                           pd.to_datetime(hh['起始日']).dt.strftime('%Y-%m-%d')))

    idx = price.index
    today = tw_today()
    hist_rows, sig_rows = [], []
    for _, r in d20.iterrows():
        sid, sd, ed = r['stock_id'], r['start'], r['end']
        if sid not in price.columns:
            continue
        pos = idx.searchsorted(sd)
        if pos < 22 or pos >= len(idx):
            continue
        ser = price[sid]
        p0, p20_ = ser.iloc[pos-1], ser.iloc[pos-21]
        if pd.isna(p0) or p0 <= 0 or pd.isna(p20_) or p20_ <= 0 or p0/p20_ < 1:
            continue                                   # 只做漲多處置
        cap = cap_map.get((sid, sd), '')
        cap_s = '大' if '大型' in cap else ('中' if '中型' in cap else '小')
        exit_d = next_trading_day(idx, ed)             # 出關日 = 賣出日
        dip = '✅' if (sid, sd.strftime('%Y-%m-%d')) in dip_keys else ''

        if exit_d <= idx[-1]:
            # ── 已出關 → 歷史 ──
            x_pos = idx.searchsorted(exit_d)
            e_pos = x_pos - 3
            if e_pos <= pos:
                continue                               # 處置期太短
            entry, prev_e, px = ser.iloc[e_pos], ser.iloc[e_pos-1], ser.iloc[x_pos]
            if pd.isna(entry) or entry <= 0 or pd.isna(prev_e) or prev_e <= 0 or pd.isna(px):
                continue
            locked = (entry/prev_e - 1) * 100 >= 9.5
            hist_rows.append({
                '代號': sid, '名稱': r.get('證券名稱', ''), '規模': cap_s,
                '年份': sd.year,
                '起始日': sd.strftime('%Y-%m-%d'),
                '買進日': idx[e_pos].strftime('%Y-%m-%d'),
                '出關日': exit_d.strftime('%Y-%m-%d'),
                '深跌單': dip,
                '訊號': '🔒 買進日漲停買不到' if locked else '✅',
                '進場價': round(float(entry), 2),
                '出場價': round(float(px), 2),
                '策略報酬(%)': np.nan if locked else round((px/entry - 1) * 100 - COST_5MIN, 2),
            })
        elif today <= exit_d:
            # ── 進行中 → 今日訊號 ──
            rem = td_count_between(idx, today, exit_d)   # 今天之後到出關日的交易日數
            buy_d = est_prev_td(idx, exit_d, 3)          # 買進日 = 出關日往回3個交易日
            entry, cur_pnl, locked = np.nan, np.nan, False
            if rem <= 2 and buy_d is not None and buy_d <= idx[-1]:
                e_pos = idx.searchsorted(buy_d)
                if e_pos < len(idx):
                    e_val, e_prev = ser.iloc[e_pos], ser.iloc[e_pos-1]
                    if pd.notna(e_val) and e_val > 0:
                        entry = round(float(e_val), 2)
                        if pd.notna(e_prev) and e_prev > 0:
                            locked = (e_val/e_prev - 1) * 100 >= 9.5
                        last = ser.dropna().iloc[-1] if len(ser.dropna()) else np.nan
                        if pd.notna(last):
                            cur_pnl = round((last/entry - 1) * 100, 2)
            if locked:
                status = '🔒 買進日漲停買不到'
            elif rem == 3:
                status = '🟢 今日收盤買進'
            elif rem == 0:
                status = '🔴 今日收盤賣出'
            elif rem in (1, 2):
                status = '🔵 持有中'
            else:
                status = f'🟡 等待（剩 {rem - 3} 個交易日）'
            sig_rows.append({
                '代號': sid, '名稱': r.get('證券名稱', ''), '規模': cap_s,
                '訊號': status,
                '買進日': buy_d.strftime('%m/%d') if buy_d is not None else '?',
                '賣出日(出關)': exit_d.strftime('%m/%d'),
                '深跌單': dip,
                '進場價': entry,
                '目前損益(%)': cur_pnl,
                '今日漲跌': today_change(price, sid),
            })

    hist = pd.DataFrame(hist_rows)
    sig = pd.DataFrame(sig_rows)
    if len(sig):
        order = {'🟢 今日收盤買進': 0, '🔴 今日收盤賣出': 1, '🔵 持有中': 2,
                 '🔒 買進日漲停買不到': 3}
        sig['_o'] = sig['訊號'].map(lambda v: order.get(v, 4))
        sig = sig.sort_values(['_o', '賣出日(出關)']).drop(columns=['_o'])
    # 0筆時也保留欄位標頭，避免 to_csv 寫出完全空白檔案讓 Streamlit 端 pd.read_csv 掛掉
    sig = sig.reindex(columns=SIG_T20_COLS)
    if len(hist):
        hist = hist.sort_values('起始日', ascending=False)
    hist = hist.reindex(columns=HIST_T20_COLS)
    return sig, hist

SIGNALS_COLS = ['代號', '名稱', '規模', '處置原因', '近20日漲幅', '大戶(%)', '起始日', '今D幾',
                '出關日', 'D1%', 'D2%', 'D3%', 'D4%', 'D5%', 'D6%', 'D7%', 'D8%', '今日漲跌',
                '評級', '處置次別', '買進訊號', '當前累積(%)', '目前損益(%)', '觸發價', '距觸發(%)']

# ── 產生訊號表 ───────────────────────────────────────────────────────────
def build_signals(df, price, open_p, whale_dfs):
    idx = price.index
    today = tw_today()
    cutoff = today - pd.Timedelta(days=20)

    # 2026-08-10 處置新制上路（撮合統一改2分鐘、期間縮為5或7個營業日）：Kevin 2026-08-18
    # 要求本頁繼續混合顯示新制事件（不要空白），評級標「🆕新制觀察中」不套用舊制信心度。
    # 完整拆第一次/第二次+對照的獨立統計頁另見 pages/18_處置新制觀察.py。
    active = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)', '小型股(<100億)'])) &
                (df['處置類型'].isin(['20分鐘', '2分鐘'])) &
                (df['處置起始日'] >= cutoff)].copy()

    rows = []
    for _, row in active.iterrows():
        sid = row['股票代號']
        sd  = row['處置起始日']
        is_new_regime = row['處置類型'] == '2分鐘'
        # 新制期間5(或當沖加重7)個營業日，遠短於舊制10天；沒有欄位可精確判斷5或7天，
        # 先用5天近似（多數案例），僅影響「今D幾」/「出關日」等顯示，不影響評級本身。
        ex  = exit_date(idx, sd, t1_offset=5) if is_new_regime else exit_date(idx, sd)

        # 已出關（今天超過 T+1）就跳過
        if pd.notna(ex) and today > ex:
            continue

        nd  = trading_day_n(idx, sd)

        d = {
            '代號':   sid,
            '名稱':   row['股票名稱'],
            '規模':   '大' if '大型' in row['市值規模'] else ('中' if '中型' in row['市值規模'] else '小'),
            '處置原因': row.get('處置原因', ''),
            '處置次別': row.get('處置次別', ''),
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
        if is_new_regime:
            # 收款門檻文字逐字比對，第一次=舊制5分鐘同款、第二次+=舊制20分鐘同款（僅撮合頻率變快），
            # 標示對照組方便之後比較，但仍不套用舊制信心度。
            if row.get('處置次別') == '第一次':
                d['評級'] = '🆕 新制觀察中(第一次,對照舊5分)'
            else:
                d['評級'] = '🆕 新制觀察中(第二次+,對照舊20分)'
        else:
            d['評級'] = grade({**row.to_dict(), '入場前20日漲幅(%)': pr, '大戶持股變動(%)': wh,
                               **{f'D{n}%': d.get(f'D{n}%') for n in range(1, 9)}})

        is_changduo = row.get('處置原因') == '漲多處置'

        # ── 買進訊號：首個 Dn < -5% 的天（僅漲多處置）；新制期間短，窗口只到D5近似 ──
        entry_n_sig = None
        entry_upper = 6 if is_new_regime else 9
        if is_changduo:
            for n in range(3, entry_upper):
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
                d['觸發價']    = round_to_tick(p0_v * 0.95)
                d['距觸發(%)'] = round((d['觸發價'] / cur_p - 1) * 100, 2)
            else:
                d['觸發價']    = np.nan
                d['距觸發(%)'] = np.nan
        else:
            d['觸發價']    = np.nan
            d['距觸發(%)'] = np.nan

        rows.append(d)

    sig = pd.DataFrame(rows)
    if len(sig):
        sig = sig.sort_values('起始日', ascending=False)
    # 0筆時也保留欄位標頭：舊制上路以來已經很常見（新事件全變2分鐘，20分鐘可能連續多天0筆），
    # 沒有這行 to_csv 會寫出完全空白檔案，讓 Streamlit 端 pd.read_csv 掛掉（同 build_5min/build_tail20 已修過的問題）。
    return sig.reindex(columns=SIGNALS_COLS)

NEWREGIME_SIG_COLS = ['處置次別', '評級', '訊號', '買進訊號', '觸發方式', '買進訊號(-5%版)', '代號', '名稱', '規模', '處置原因',
                      '近20日漲幅', '大戶(%)', '起始日', '今D幾', '出關日', '目前損益(%)', '目前損益(-5%版)(%)',
                      '今日漲跌', '觸發價', '距觸發(%)', '觸發價(-5%版)', '距觸發(-5%版)(%)',
                      'D1%', 'D2%', 'D3%', 'D4%', 'D5%',
                      'LowD1%', 'LowD2%', 'LowD3%', 'LowD4%', 'LowD5%', '出關開盤(相對D0)%']

# ── 2026-08-10 處置新制（2分鐘撮合）今日訊號 ──────────────────────────────
# 完全獨立於 build_signals()（舊制20分鐘頁），彼此不共用輸出檔，不影響舊制頁面。
# 撮合改2分鐘、期間縮為5(或7)個營業日，尚無完整出關樣本可驗證，評級沿用grade()
# 只是描述性分類，頁面本身要清楚標示「未驗證」，不是舊制頁面同等信心度的訊號。
def build_newregime_signals(df, price, open_p, whale_dfs):
    idx = price.index
    low_p = load_low()
    today = tw_today()
    cutoff = today - pd.Timedelta(days=20)

    active = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)', '小型股(<100億)'])) &
                (df['處置類型'] == '2分鐘') &
                (df['處置起始日'] >= cutoff)].copy()

    rows = []
    for _, row in active.iterrows():
        sid = row['股票代號']
        sd  = row['處置起始日']
        # 新制期間5(或當沖加重7)個營業日，遠短於舊制10天；沒有欄位可精確判斷5或7天，
        # 先用5天近似（多數案例），只影響「今D幾」/「出關日」等顯示。
        ex  = exit_date(idx, sd, t1_offset=5)

        if pd.notna(ex) and today > ex:
            continue

        nd = trading_day_n(idx, sd)

        d = {
            '處置次別': row.get('處置次別', ''),
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
        # 新制期間只有5(或7)個營業日，D6~D8不存在，只算到D5避免顯示出關後的一般交易日
        # 誤植成「還在處置中的Dn%」
        for n in range(1, 6):
            v = cumret(price, idx, sid, sd, n)
            d[f'D{n}%'] = v
        # LowD{n}%：當天最低價相對D0收盤的累積跌幅（觸發判斷用，跟D{n}%收盤版分開存），
        # 加上出關開盤價相對D0收盤的報酬，讓頁面可以自己選任意D幾~D幾窗口即時重算，
        # 不用每加一種窗口就要改一次後端程式（2026-08-30，Kevin要求可自選窗口）。
        pos_lw = idx.searchsorted(sd)
        p0_lw  = price[sid].iloc[pos_lw - 1] if pos_lw >= 1 and sid in price.columns else np.nan
        for n in range(1, 6):
            lp = pos_lw + n - 1
            v = np.nan
            if pd.notna(p0_lw) and p0_lw > 0 and lp < len(low_p) and sid in low_p.columns:
                ln = low_p[sid].iloc[lp]
                if pd.notna(ln):
                    v = round((ln / p0_lw - 1) * 100, 2)
            d[f'LowD{n}%'] = v
        ex_pos = pos_lw + 5
        if pd.notna(p0_lw) and p0_lw > 0 and sid in open_p.columns and ex_pos < len(open_p):
            ex_open = open_p[sid].iloc[ex_pos]
            d['出關開盤(相對D0)%'] = round((ex_open / p0_lw - 1) * 100, 2) if pd.notna(ex_open) and ex_open > 0 else np.nan
        else:
            d['出關開盤(相對D0)%'] = np.nan
        d['今日漲跌'] = today_change(price, sid)
        pr = d['近20日漲幅']
        wh = d['大戶(%)']
        d['評級'] = grade({**row.to_dict(), '入場前20日漲幅(%)': pr, '大戶持股變動(%)': wh,
                           **{f'D{n}%': d.get(f'D{n}%') for n in range(1, 6)}})

        is_changduo = row.get('處置原因') == '漲多處置'
        is_first = row.get('處置次別') == '第一次'

        # 白話狀態欄——第一次處置的規則只在D1一天內就分出勝負(觸發/開高不買/鎖漲停買不到)，
        # 但買進訊號/觸發價/距觸發(%)這些欄位空白時，看不出來是「今天還沒決定」「已經買了在
        # 持有中」還是「D1已經過了、機會沒了」，容易混淆(Kevin反映過)。比照build_5min()舊制
        # 5分盤動能頁已經在用的白話狀態寫法，搬過來給第一次用。
        d['訊號'] = ''
        entry_n_sig = None
        trigger_type_sig = ''
        if is_changduo and is_first:
            # 第一次處置沿用「5分盤動能」規則（build_5min()），不是-5%回檔：
            # D0(處置前一天)漲2~9%、D1不跳空高開、D1收盤沒鎖漲停 → D1收盤買進
            pos_e = idx.searchsorted(sd)
            p_1e  = price[sid].iloc[pos_e - 2] if pos_e >= 2 and sid in price.columns else np.nan
            p0e   = price[sid].iloc[pos_e - 1] if pos_e >= 1 and sid in price.columns else np.nan
            if pd.notna(p_1e) and p_1e > 0 and pd.notna(p0e) and p0e > 0:
                prev1 = (p0e / p_1e - 1) * 100
                hit = 2 <= prev1 <= 9
                d1_open = open_p[sid].iloc[pos_e] if sid in open_p.columns and pos_e < len(open_p) else np.nan
                gap = (d1_open / p0e - 1) * 100 if pd.notna(d1_open) else np.nan
                d1_ret = d.get('D1%')
                d1_locked = pd.notna(d1_ret) and d1_ret >= 9.5
                if hit and pd.notna(gap) and gap <= 0 and not d1_locked:
                    entry_n_sig = 1

                if not hit:
                    d['訊號'] = '❌ 不符合(D0漲幅不在2~9%)'
                elif nd < 1:
                    d['訊號'] = '⚪ 未開始'
                elif pd.isna(gap):
                    d['訊號'] = '🟡 等D1收盤資料'
                elif gap > 0:
                    d['訊號'] = '❌ D1開高，不買'
                elif d1_locked:
                    d['訊號'] = '🔒 D1漲停買不到'
                elif entry_n_sig == 1:
                    d['訊號'] = '🟢 今日D1收盤買進' if nd == 1 else '🔵 已於D1買進，持有中'
            else:
                d['訊號'] = '⚪ 資料不足'
        elif is_changduo:
            # 2026-08-26修正：觸發條件改用「當天最低價」相對D0收盤是否跌破-5%，不再只看
            # 收盤價（disposal_dip_intraday_touch研究：盤中觸及但收盤沒守住-5%的獨有子集
            # 合，驗證期n=39、勝率94.9%、平均+18.48%，比只看收盤的原規則還強）。
            # 買進價維持不變，還是當天收盤價，只有「要不要觸發」這個判斷改成看最低價。
            pos_s = idx.searchsorted(sd)
            p0_s  = price[sid].iloc[pos_s - 1] if pos_s >= 1 and sid in price.columns else np.nan
            for n in range(1, 5):
                if pd.notna(d.get(f'D{n}%')) and pd.notna(p0_s) and p0_s > 0 and sid in low_p.columns:
                    low_pos = pos_s + n - 1
                    low_n = low_p[sid].iloc[low_pos] if low_pos < len(low_p) else np.nan
                    if pd.notna(low_n):
                        low_ret = (low_n / p0_s - 1) * 100
                        if low_ret < -5:
                            entry_n_sig = n
                            # 觸發方式：收盤本身跌破-5%(A)，或收盤沒守住、只有盤中最低價
                            # 跌破(C，disposal_dip_intraday_touch驗證出更強的獨有子集合)。
                            trigger_type_sig = '收盤跌破(A)' if d[f'D{n}%'] < -5 else '僅盤中觸及(C)'
                            break
        d['買進訊號'] = f'D{entry_n_sig}' if entry_n_sig else ''
        d['觸發方式'] = trigger_type_sig if entry_n_sig else ''

        # alt：僅第一次才有值，供比較「若套用第二次+的-5%回檔規則」會是什麼訊號，
        # 純供比較，不是建議規則。
        entry_n_sig_alt = None
        if is_changduo and is_first:
            for n in range(1, 5):
                if pd.notna(d.get(f'D{n}%')) and d[f'D{n}%'] < -5:
                    entry_n_sig_alt = n
                    break
        d['買進訊號(-5%版)'] = f'D{entry_n_sig_alt}' if entry_n_sig_alt else ''

        cur_cum = np.nan
        for n in range(nd, 0, -1):
            v = d.get(f'D{n}%')
            if pd.notna(v):
                cur_cum = v
                break

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

        # alt：僅第一次才有值，若套用-5%版訊號的目前損益(%)
        if entry_n_sig_alt and pd.notna(cur_cum):
            entry_cum_v_alt = d.get(f'D{entry_n_sig_alt}%', np.nan)
            if pd.notna(entry_cum_v_alt):
                entry_factor_alt = 1 + entry_cum_v_alt / 100
                cur_factor_alt   = 1 + cur_cum / 100
                d['目前損益(-5%版)(%)'] = round((cur_factor_alt / entry_factor_alt - 1) * 100, 2) if entry_factor_alt > 0 else np.nan
            else:
                d['目前損益(-5%版)(%)'] = np.nan
        else:
            d['目前損益(-5%版)(%)'] = np.nan

        # 觸發價/距觸發(%)：第二次+是「D0收盤*0.95」（-5%回檔規則的門檻價）；
        # 第一次是「D0收盤本身」（5分盤動能規則：D1開盤只要不高於這個價，訊號才成立，
        # 不是跌到某個價位，是「不能開得比這個價高」）。距觸發(%)一律用「觸發價相對最新可得
        # 價格的距離」，跟第二次+同樣的算法、只是基準價不同。
        pos_v = idx.searchsorted(sd)
        p0_v  = price[sid].iloc[pos_v - 1] if pos_v >= 1 and sid in price.columns else np.nan
        ser_v = price[sid].dropna()
        cur_p = float(ser_v.iloc[-1]) if len(ser_v) > 0 else np.nan
        # D1(sd)當天若還沒有真實收盤價可用（資料源尚未公布今天的資料），最新可得價格
        # 就只會停在D0收盤——這時候「距觸發(%)」用cur_p算出來會巧合等於0.00%，
        # 看起來像「剛好卡在門檻上」，其實只是「還沒有今天的真實資料」，必須分開處理，
        # 不能顯示這個誤導的假精確數字。
        has_current_data = len(ser_v) > 0 and ser_v.index[-1] >= sd
        if is_changduo and not is_first:
            trigger = round_to_tick(p0_v * 0.95) if pd.notna(p0_v) and p0_v > 0 else np.nan
        elif is_changduo and is_first:
            trigger = p0_v if pd.notna(p0_v) and p0_v > 0 else np.nan
        else:
            trigger = np.nan
        d['觸發價'] = trigger if pd.notna(trigger) else np.nan
        if pd.notna(trigger) and has_current_data and pd.notna(cur_p) and cur_p > 0:
            d['距觸發(%)'] = round((trigger / cur_p - 1) * 100, 2)
        else:
            d['距觸發(%)'] = np.nan

        # alt觸發價：僅第一次才有值，若改用第二次+的-5%回檔規則，門檻價會是D0收盤*0.95
        if is_changduo and is_first and pd.notna(p0_v) and p0_v > 0:
            trigger_alt = round_to_tick(p0_v * 0.95)
            d['觸發價(-5%版)'] = trigger_alt
            d['距觸發(-5%版)(%)'] = (round((trigger_alt / cur_p - 1) * 100, 2)
                                    if has_current_data and pd.notna(cur_p) and cur_p > 0 else np.nan)
        else:
            d['觸發價(-5%版)']    = np.nan
            d['距觸發(-5%版)(%)'] = np.nan

        rows.append(d)

    sig = pd.DataFrame(rows)
    if len(sig):
        sig = sig.sort_values(['處置次別', '起始日'], ascending=[True, False])
    # 0筆時也保留欄位標頭，避免 to_csv 寫出完全空白檔案讓 Streamlit 端 pd.read_csv 掛掉
    return sig.reindex(columns=NEWREGIME_SIG_COLS)

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
                'wr': round(len(wins) / len(sub) * 100, 2),
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
                   actual_ret=np.nan, entry_n_alt=np.nan, entry_cum_alt=np.nan, actual_ret_alt=np.nan,
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

        if disp_type == '5分鐘':
            # 第一次處置(5分鐘)不是-5%回檔規則，是「5分盤動能」規則：
            # D0(處置前一天)漲2~9%、D1不跳空高開、D1收盤沒鎖漲停 → D1收盤買進、出關日開盤賣出。
            # min_dn/deepest_n上面仍算出來當診斷欄位參考，但不用來決定進場。
            p_1 = price[sid].iloc[pos - 2] if pos >= 2 else np.nan
            if pd.notna(p_1) and p_1 > 0:
                prev1 = (p0 / p_1 - 1) * 100
                hit = 2 <= prev1 <= 9
                d1_close = price[sid].iloc[pos] if pos < len(price) else np.nan
                d1_open  = open_p[sid].iloc[pos] if sid in open_p.columns and pos < len(open_p) else np.nan
                gap = (d1_open / p0 - 1) * 100 if pd.notna(d1_open) else np.nan
                d1_ret = (d1_close / p0 - 1) * 100 if pd.notna(d1_close) and p0 > 0 else np.nan
                d1_locked = pd.notna(d1_ret) and d1_ret >= 9.5
                if hit and pd.notna(gap) and gap <= 0 and not d1_locked and pd.notna(d1_close) and d1_close > 0:
                    out['entry_n']   = 1
                    out['entry_cum'] = round(d1_ret, 2)
                    if has_exit:
                        out['actual_ret'] = round((t1_open / d1_close - 1) * 100, 2)
            # alt：若第一次也套用-5%回檔規則(套用第二次+邏輯於D1~D5窗口)，純供比較，不是建議規則
            for n in sorted(entry_rng):
                if n in dn_rets and dn_rets[n] < -5:
                    out['entry_n_alt']   = n
                    out['entry_cum_alt'] = round(dn_rets[n], 2)
                    if has_exit:
                        pn = price[sid].iloc[pos + n - 1]
                        out['actual_ret_alt'] = round((t1_open / pn - 1) * 100, 2)
                    break
        else:
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
        # alt：僅5分鐘(第一次)才有值，若套用第二次+的-5%回檔規則會是什麼結果，純供比較
        '買進日(-5%版)':        pool['entry_n_alt'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '買進時累積(-5%版)(%)': pool['entry_cum_alt'],
        '出關報酬(-5%版)(%)':   pool['actual_ret_alt'],
        '結果(-5%版)':          pool['actual_ret_alt'].apply(
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
            'wr': round((s > 0).mean() * 100, 2),
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

NEWREGIME_HIST_COLS = ['起始日', '出關日', '處置次別', '代號', '名稱', '規模', 'Dn組別',
                       '近20日漲幅', '大戶(%)', 'D0收盤價', '買進日', '買進價', '買進時累積(%)', '觸發方式', '最深日',
                       '期間最深(%)', '出關價', '出關報酬(%)', '結果',
                       *[f'T+{k}收盤(%)' for k in range(1, 11)],
                       *[f'D{n}%' for n in range(1, 6)], *[f'LowD{n}%' for n in range(1, 6)],
                       '出關開盤(相對D0)%',
                       '買進日(-5%版)', '買進時累積(-5%版)(%)', '出關報酬(-5%版)(%)', '結果(-5%版)']

# ── 2026-08-10 處置新制（2分鐘撮合）歷史回測紀錄 ──────────────────────────
# 完全獨立於 build_history()（舊制頁），寫入獨立檔案不影響舊制 history.csv。
# 新舊制期間長度不同（新制5或7個營業日 vs 舊制5分鐘/20分鐘各自的天數），不可合併統計，
# 兩者的t1_offset/進場窗口是各自獨立近似，見函式內註解。
def build_newregime_history(df, price, open_p, whale_dfs):
    idx = price.index
    low_p = load_low()
    pool = df[(df['市值規模'].isin(['大型股(>500億)', '中型股(100~500億)', '小型股(<100億)'])) &
              (df['處置類型'] == '2分鐘') &
              (df['處置原因'] == '漲多處置')].copy()

    # 新制第一次/第二次+ 撮合頻率相同(2分鐘)，處置期間都是5(或7)個營業日，
    # 沒有欄位可精確分辨5或7天，統一用t1_offset=5近似（多數案例），進場窗口統一D3~D8。
    T1_OFFSET = 5
    # 2026-08-30起新預設窗口D1~D4（原本D3~D5）：新制真實資料顯示D1單日表現最好、
    # D3~D5幾乎都是D1早就觸發過的同一批股票，改用比較早、比較便宜的進場點；
    # D5排除是因為D5單獨測試時明顯最弱(見disposal_entry_day_full_test研究)，
    # D1~D4跟D1~D5在目前資料下實測完全相同(D5從未單獨觸發過)，選D1~D4當更保守版本。
    ENTRY_RNG = range(1, 5)  # D1~D4

    def compute_row(row):
        sid = row['股票代號']
        sd  = row['處置起始日']
        out = dict(entry_n=np.nan, entry_cum=np.nan, min_dn=np.nan, deepest_n=np.nan, actual_ret=np.nan,
                   entry_n_alt=np.nan, entry_cum_alt=np.nan, actual_ret_alt=np.nan, trigger_type='',
                   exit_open_rel_d0=np.nan, d0_close=np.nan, entry_price=np.nan, exit_price=np.nan,
                   **{f'_t{k}c': np.nan for k in range(1, 11)},
                   **{f'd{n}_close': np.nan for n in range(1, 6)},
                   **{f'd{n}_low': np.nan for n in range(1, 6)})
        if sid not in price.columns:
            return pd.Series(out)
        pos = idx.searchsorted(sd)
        if pos < 1 or pos + 2 >= len(idx):
            return pd.Series(out)
        p0 = price[sid].iloc[pos - 1]
        if pd.isna(p0) or p0 <= 0:
            return pd.Series(out)
        out['d0_close'] = round(float(p0), 2)

        t1_open = (open_p[sid].iloc[pos + T1_OFFSET]
                   if sid in open_p.columns and pos + T1_OFFSET < len(open_p) else np.nan)
        has_exit = pd.notna(t1_open) and t1_open > 0
        if has_exit:
            out['exit_open_rel_d0'] = round((t1_open / p0 - 1) * 100, 2)
            out['exit_price'] = round(float(t1_open), 2)

        all_rets = {}
        for n in range(1, T1_OFFSET + 1):
            if pos + n - 1 < len(price):
                pn = price[sid].iloc[pos + n - 1]
                if pd.notna(pn) and pn > 0:
                    all_rets[n] = round((pn / p0 - 1) * 100, 2)
                    out[f'd{n}_close'] = all_rets[n]
            if pos + n - 1 < len(low_p) and sid in low_p.columns:
                ln = low_p[sid].iloc[pos + n - 1]
                if pd.notna(ln) and ln > 0:
                    out[f'd{n}_low'] = round((ln / p0 - 1) * 100, 2)

        dn_rets = {n: all_rets[n] for n in ENTRY_RNG if n in all_rets}
        if dn_rets:
            deepest = min(dn_rets, key=lambda n: dn_rets[n])
            out['min_dn']    = round(dn_rets[deepest], 2)
            out['deepest_n'] = deepest

        # 第一次處置：舊制底下對應的不是「D3~D5回檔-5%」（那是第二次+/原20分鐘驗證過的規則），
        # 而是完全不同的「5分盤動能」規則（build_5min()）：D0(處置前一天)漲2~9%、
        # D1不跳空高開、D1收盤沒鎖漲停 → D1收盤買進、出關日開盤賣出。兩者進場邏輯不能共用。
        if row.get('處置次別') == '第一次':
            if pos < 2:
                return pd.Series(out)
            p_1 = price[sid].iloc[pos - 2]
            if pd.isna(p_1) or p_1 <= 0:
                return pd.Series(out)
            prev1 = (p0 / p_1 - 1) * 100
            hit = 2 <= prev1 <= 9
            d1_close = price[sid].iloc[pos] if pos < len(price) else np.nan
            d1_open  = open_p[sid].iloc[pos] if sid in open_p.columns and pos < len(open_p) else np.nan
            gap = (d1_open / p0 - 1) * 100 if pd.notna(d1_open) and p0 > 0 else np.nan
            d1_ret = (d1_close / p0 - 1) * 100 if pd.notna(d1_close) and p0 > 0 else np.nan
            d1_locked = pd.notna(d1_ret) and d1_ret >= 9.5
            if hit and pd.notna(gap) and gap <= 0 and not d1_locked and pd.notna(d1_close) and d1_close > 0:
                out['entry_n']   = 1
                out['entry_cum'] = round(d1_ret, 2)
                out['entry_price'] = round(float(d1_close), 2)
                if has_exit:
                    out['actual_ret'] = round((t1_open / d1_close - 1) * 100, 2)
            # alt：如果改用第二次+的-5%回檔規則(D3~D5任一天跌破-5%)，第一次會是什麼結果？
            # 純供比較，不是建議規則——第一次沒有這個規則的歷史驗證。
            if dn_rets:
                for n in sorted(ENTRY_RNG):
                    if n in dn_rets and dn_rets[n] < -5:
                        out['entry_n_alt']   = n
                        out['entry_cum_alt'] = round(dn_rets[n], 2)
                        if has_exit:
                            pn = price[sid].iloc[pos + n - 1]
                            out['actual_ret_alt'] = round((t1_open / pn - 1) * 100, 2)
                        break
            return pd.Series(out)

        if not dn_rets:
            return pd.Series(out)

        # 2026-08-26修正：觸發條件改用「當天最低價」相對D0收盤是否跌破-5%，不再只看收盤價
        # （disposal_dip_intraday_touch研究驗證過：盤中觸及但收盤沒守住-5%的獨有子集合，
        # 驗證期n=39、勝率94.9%、平均+18.48%，比只看收盤的原規則還強）。
        # 買進價維持不變，還是當天收盤價；只有「要不要觸發」這個判斷改成看最低價。
        for n in sorted(ENTRY_RNG):
            if n in dn_rets and pos + n - 1 < len(low_p) and sid in low_p.columns:
                low_n = low_p[sid].iloc[pos + n - 1]
                low_ret = (low_n / p0 - 1) * 100 if pd.notna(low_n) and p0 > 0 else np.nan
                if pd.notna(low_ret) and low_ret < -5:
                    out['entry_n']   = n
                    out['entry_cum'] = round(dn_rets[n], 2)
                    out['entry_price'] = round(float(price[sid].iloc[pos + n - 1]), 2)
                    # 觸發方式：收盤本身就跌破-5%(A，驗證期較強的原規則)，
                    # 還是收盤沒守住、只有盤中最低價跌破(C，disposal_dip_intraday_touch
                    # 驗證出的獨有子集合，驗證期表現比A更好：n=39/94.9%/+18.48%)。
                    out['trigger_type'] = '收盤跌破(A)' if dn_rets[n] < -5 else '僅盤中觸及(C)'
                    if has_exit:
                        pn = price[sid].iloc[pos + n - 1]
                        out['actual_ret'] = round((t1_open / pn - 1) * 100, 2)
                    break

        # T+1~T+10收盤(%)：如果沒有在出關日開盤賣、繼續抱著，之後10個交易日的報酬會是多少
        # （基準是買進日收盤價），比照舊制Tab2表格同樣的欄位，讓新制也能看到出關後續走勢。
        ref_n = int(out['entry_n']) if pd.notna(out['entry_n']) else 3
        if ref_n in dn_rets:
            p_ref = price[sid].iloc[pos + ref_n - 1]
            for k in range(1, 11):
                off = (T1_OFFSET - 1) + k  # T+1=pos+T1_OFFSET(出關日), T+2=pos+T1_OFFSET+1...
                if pos + off < len(price) and pd.notna(p_ref) and p_ref > 0:
                    p = price[sid].iloc[pos + off]
                    if pd.notna(p) and p > 0:
                        out[f'_t{k}c'] = round((p / p_ref - 1) * 100, 2)

        return pd.Series(out)

    if pool.empty:
        return pd.DataFrame(columns=NEWREGIME_HIST_COLS), []

    stats = pool.apply(compute_row, axis=1)
    pool  = pd.concat([pool, stats], axis=1)

    def dn_group(v):
        if pd.isna(v):  return 'Dn無資料'
        if v < -5:      return 'Dn < -5%'
        if v < 0:       return 'Dn -5%~0%'
        return 'Dn ≥ 0%'

    pool['Dn組別'] = pool['min_dn'].apply(dn_group)
    pool['_exit_date'] = pool.apply(lambda r: exit_date(idx, r['處置起始日'], t1_offset=T1_OFFSET), axis=1)

    out = pd.DataFrame({
        '起始日':        pool['處置起始日'].dt.strftime('%Y-%m-%d'),
        '出關日':        pool['_exit_date'].apply(lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else '?'),
        '處置次別':      pool.get('處置次別', ''),
        '代號':          pool['股票代號'],
        '名稱':          pool['股票名稱'],
        '規模':          pool['市值規模'].apply(lambda v: '大' if '大型' in str(v) else ('中' if '中型' in str(v) else '小')),
        'Dn組別':        pool['Dn組別'],
        '近20日漲幅':    pool.apply(lambda r: prerun20(price, idx, r['股票代號'], r['處置起始日']), axis=1).round(2),
        '大戶(%)':       pool.apply(lambda r: whale_delta(whale_dfs, r['股票代號'], r['處置起始日'], price), axis=1),
        'D0收盤價':      pool['d0_close'],
        '買進日':        pool['entry_n'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '買進價':        pool['entry_price'],
        '買進時累積(%)': pool['entry_cum'],
        '觸發方式':      pool['trigger_type'],
        '最深日':        pool['deepest_n'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '期間最深(%)':   pool['min_dn'],
        '出關價':        pool['exit_price'],
        '出關報酬(%)':   pool['actual_ret'],
        '結果':          pool['actual_ret'].apply(
            lambda v: f'✅ {v:+.2f}%' if pd.notna(v) and v > 0
                      else (f'❌ {v:+.2f}%' if pd.notna(v) else '-')),
        # T+1~T+10收盤(%)：出關後如果繼續抱著，之後10個交易日的報酬，比照舊制Tab2欄位
        **{f'T+{k}收盤(%)': pool[f'_t{k}c'] for k in range(1, 11)},
        # D{n}%/LowD{n}%/出關開盤(相對D0)%：給頁面自選進場窗口(D幾~D幾)即時重算用，
        # 不用每加一種窗口就要改後端程式（2026-08-30，Kevin要求可自選窗口）。
        **{f'D{n}%': pool[f'd{n}_close'] for n in range(1, 6)},
        **{f'LowD{n}%': pool[f'd{n}_low'] for n in range(1, 6)},
        '出關開盤(相對D0)%': pool['exit_open_rel_d0'],
        # alt：僅第一次處置才有值，供比較「若套用第二次+的-5%回檔規則」的結果，
        # 不是建議規則。第二次+本身沒有alt，因為-5%回檔就是它已驗證的規則。
        '買進日(-5%版)':        pool['entry_n_alt'].apply(lambda v: f'D{int(v)}' if pd.notna(v) else '-'),
        '買進時累積(-5%版)(%)': pool['entry_cum_alt'],
        '出關報酬(-5%版)(%)':   pool['actual_ret_alt'],
        '結果(-5%版)':          pool['actual_ret_alt'].apply(
            lambda v: f'✅ {v:+.2f}%' if pd.notna(v) and v > 0
                      else (f'❌ {v:+.2f}%' if pd.notna(v) else '-')),
    })
    hist = out.sort_values('起始日', ascending=False).reset_index(drop=True)

    def grp_stats(sub, label):
        s = sub['actual_ret'].dropna()
        if len(s) == 0: return None
        return {'label': label, 'n': len(s), 'wr': round((s > 0).mean() * 100, 2), 'ret': round(s.mean(), 2)}

    cmp_stats = []
    for label_prefix, sub_pool in [('第一次', pool[pool['處置次別'] == '第一次']),
                                    ('第二次+', pool[pool['處置次別'] == '第二次+'])]:
        cmp_stats.append(grp_stats(sub_pool[sub_pool['min_dn'] < -5], f'{label_prefix}：Dn最深 < -5%（進場）'))
        cmp_stats.append(grp_stats(sub_pool, f'{label_prefix}：全部漲多（不篩選Dn）'))
    cmp_stats = [x for x in cmp_stats if x]

    return hist.reindex(columns=NEWREGIME_HIST_COLS), cmp_stats

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

    print('產生出關動能資料...')
    sig_t, hist_t = build_tail20(df, price, hist)
    sig_t.to_csv(f'{OUT_DIR}/signals_tail20.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    hist_t.to_csv(f'{OUT_DIR}/history_tail20.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'  → signals_tail20.csv ({len(sig_t)} 筆) / history_tail20.csv ({len(hist_t)} 筆)')

    print('產生處置新制(2分鐘撮合)觀察資料...')
    sig_nr = build_newregime_signals(sig_df, price, open_p, whale_dfs)
    sig_nr.to_csv(f'{OUT_DIR}/newregime_signals.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    hist_nr, cmp_stats_nr = build_newregime_history(df, price, open_p, whale_dfs)
    hist_nr.to_csv(f'{OUT_DIR}/newregime_history.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'  → newregime_signals.csv ({len(sig_nr)} 筆) / newregime_history.csv ({len(hist_nr)} 筆)')

    # 更新時間
    meta = {
        'updated_at': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M'),
        'data_date': price.index[-1].strftime('%Y-%m-%d'),
        'cmp_stats': cmp_stats,
        'cmp_stats_newregime': cmp_stats_nr,
    }
    with open(f'{OUT_DIR}/meta.json', 'w') as f:
        json.dump(meta, f)
    print(f'  → meta.json')
    print('完成')

if __name__ == '__main__':
    main()
