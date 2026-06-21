"""
build_disposal_data.py
從 FinLab 全量重建 disposal_data.csv（含最新資料）

分類邏輯（處置原因）：用處置起始日前 3 個交易日的累積報酬推斷
  漲多處置   : 前3日累積報酬 > +threshold
  跌深處置   : 前3日累積報酬 < -threshold
  異常震盪/量大: 其餘（含量大但漲幅不明顯）
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

TOKEN = "iwSmg6ZUt9Mx/7fjWSm4wYVWrF/yfORaZYSX0vCCr/B9vHDUS4f7jSd6R44ti5ij#vip_m"
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'disposal_data_v2.csv')
REF_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'disposal_data_v2.csv')   # 原始 CSV，用來校正閾值


# ════════════════════════════════════════════════════════════════
# 1. 拉資料
# ════════════════════════════════════════════════════════════════
def load_all():
    import finlab
    finlab.login(TOKEN)
    from finlab import data

    print("📥 disposal_information...")
    disp = pd.DataFrame(data.get('disposal_information'))

    print("📥 price:收盤價...")
    close = data.get('price:收盤價')

    print("📥 price:開盤價...")
    open_p = data.get('price:開盤價')

    print("📥 company_basic_info...")
    basic = pd.DataFrame(data.get('company_basic_info'))

    print("📥 法人外資...")
    inst_f = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')

    print("📥 法人投信...")
    inst_t = data.get('institutional_investors_trading_summary:投信買賣超股數')

    print("📥 成交股數...")
    volume = data.get('price:成交股數')

    return disp, close, open_p, basic, inst_f, inst_t, volume


# ════════════════════════════════════════════════════════════════
# 2. 建立事件清單（第一次 + 第二次，4 碼，2022+）
# ════════════════════════════════════════════════════════════════
def build_events(disp):
    df = disp.copy()
    df.columns = [c.strip() for c in df.columns]
    df['stock_id'] = df['stock_id'].astype(str).str.strip()
    df['start_date'] = pd.to_datetime(df['處置開始時間'], errors='coerce')
    df['end_date']   = pd.to_datetime(df['處置結束時間'], errors='coerce')

    # 4 碼一般股
    df = df[df['stock_id'].str.match(r'^\d{4}$')]
    # 2022+
    df = df[df['start_date'].dt.year >= 2022]
    # 第一次或第二次
    df = df[df['處置措施'].str.contains('第一次|第二次', na=False)]
    # 分時交易有值
    df = df[df['分時交易'].notna()]

    # 對應處置類型
    type_map = {5.0: '5分鐘', 20.0: '20分鐘', 25.0: '25分鐘', 45.0: '45分鐘', 60.0: '60分鐘', 30.0: '30分鐘'}
    df['處置類型'] = df['分時交易'].map(type_map)

    df = df.drop_duplicates(subset=['stock_id', 'start_date'])
    df = df.reset_index(drop=True)
    print(f"\n✅ 篩選後事件數: {len(df)}")
    print(df['處置類型'].value_counts().to_string())
    return df


# ════════════════════════════════════════════════════════════════
# 3. 校正分類閾值（對比原始 CSV）
# ════════════════════════════════════════════════════════════════
def calibrate_threshold(events, close, ref_csv_path):
    """找最佳 3 日前報酬閾值"""
    try:
        ref = pd.read_csv(ref_csv_path)
        ref['股票代號'] = ref['股票代號'].astype(str).str.zfill(4)
        ref['處置起始日'] = pd.to_datetime(ref['處置起始日'])
        ref['_key'] = ref['股票代號'] + '_' + ref['處置起始日'].dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"無法讀取參考 CSV: {e}")
        return 0.03  # 預設閾值 3%

    # 計算每個事件的前 3 日累積報酬
    close_idx = {d: i for i, d in enumerate(close.index)}
    pre3_rets = {}

    for _, r in events.iterrows():
        sid = r['stock_id']
        sd  = r['start_date']
        if sid not in close.columns:
            continue
        if sd not in close_idx:
            continue
        ci = close_idx[sd]
        # sd 當天的前一個交易日 close = sd 的前 1 天
        # pre3 = (close[sd-1] / close[sd-4]) - 1  (sd-1 = last close before disposal)
        if ci < 4:
            continue
        c_pre1 = close.iloc[ci-1, close.columns.get_loc(sid)]  # 前1日close
        c_pre4 = close.iloc[ci-4, close.columns.get_loc(sid)]  # 前4日close（3日前）
        if pd.isna(c_pre1) or pd.isna(c_pre4) or c_pre4 <= 0:
            continue
        key = f"{sid}_{sd.strftime('%Y-%m-%d')}"
        pre3_rets[key] = (c_pre1 / c_pre4) - 1

    # 與 ref 合併
    ref_matched = ref[ref['_key'].isin(pre3_rets.keys())].copy()
    ref_matched['pre3'] = ref_matched['_key'].map(pre3_rets)
    ref_matched = ref_matched.dropna(subset=['pre3'])
    print(f"\n校正樣本: {len(ref_matched)} 筆")

    # 嘗試不同閾值
    best_acc = 0
    best_th  = 0.03
    for th in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]:
        def classify(r):
            if r['pre3'] > th:  return '漲多處置'
            if r['pre3'] < -th: return '跌深處置'
            return '異常震盪/量大'
        pred = ref_matched.apply(classify, axis=1)
        acc  = (pred == ref_matched['處置原因']).mean()
        print(f"  閾值 {th*100:.0f}%: 準確率 {acc:.1%}  (漲多={( pred=='漲多處置').sum()}, 跌深={( pred=='跌深處置').sum()}, 異常={(pred=='異常震盪/量大').sum()})")
        if acc > best_acc:
            best_acc = acc
            best_th  = th

    print(f"\n最佳閾值: {best_th*100:.0f}%  (準確率 {best_acc:.1%})")
    return best_th, pre3_rets


# ════════════════════════════════════════════════════════════════
# 4. 市值分級
# ════════════════════════════════════════════════════════════════
def classify_cap(sid, sd, close, basic_shares_map):
    if sid not in close.columns:
        return '未知'
    avail = close[sid][close[sid].index <= sd].dropna()
    if len(avail) == 0:
        return '未知'
    price  = float(avail.iloc[-1])
    shares = basic_shares_map.get(sid, np.nan)
    if np.isnan(shares) or price <= 0:
        return '未知'
    mkt = price * shares
    if mkt >= 5e10:   return '大型股(>500億)'
    if mkt >= 1e10:   return '中型股(100~500億)'
    return '小型股(<100億)'


# ════════════════════════════════════════════════════════════════
# 5. 計算所有報酬欄位
# ════════════════════════════════════════════════════════════════
def safe_ret(exit_val, entry_val):
    if pd.isna(exit_val) or pd.isna(entry_val) or entry_val <= 0:
        return np.nan
    return round((exit_val - entry_val) / entry_val * 100, 2)


def compute_row(r, close, open_p, inst_f, inst_t, volume, basic_shares_map):
    sid = r['stock_id']
    sd  = r['start_date']
    ed  = r['end_date']

    if sid not in close.columns:
        return None

    close_idx = {d: i for i, d in enumerate(close.index)}
    open_idx  = {d: i for i, d in enumerate(open_p.index)}

    if sd not in close_idx:
        return None

    ci = close_idx[sd]
    ci_e = close_idx.get(ed, None)

    # ── 市值
    cap = classify_cap(sid, sd, close, basic_shares_map)

    # ── 處置原因（近20日漲幅：>= 0% 為漲多，< 0% 為跌深）
    if ci >= 22:
        c_pre1  = close.iloc[ci-1,  close.columns.get_loc(sid)]
        c_pre21 = close.iloc[ci-21, close.columns.get_loc(sid)]
        if not pd.isna(c_pre1) and not pd.isna(c_pre21) and c_pre21 > 0:
            rise20 = (c_pre1 / c_pre21) - 1
            reason = '漲多處置' if rise20 >= 0 else '跌深處置'
        else:
            reason = '漲多處置'
    else:
        reason = '漲多處置'

    # ── close 位置輔助
    def get_close(offset):  # offset = 相對 sd 的交易日偏移
        idx = ci + offset
        if idx < 0 or idx >= len(close):
            return np.nan
        return close.iloc[idx, close.columns.get_loc(sid)]

    def get_open(offset_from_ed):  # offset 相對 ed
        if ci_e is None:
            return np.nan
        idx = ci_e + offset_from_ed
        if idx < 0 or idx >= len(open_p):
            return np.nan
        return open_p.iloc[idx, open_p.columns.get_loc(sid)] if sid in open_p.columns else np.nan

    c_pre1    = get_close(-1)    # 處置前一日close（D0）
    c_D1      = get_close(0)     # D1 close（處置起始日）
    c_D3      = get_close(2)     # D3
    c_D5      = get_close(4)     # D5
    c_ed      = get_close(ci_e - ci) if ci_e else np.nan  # 出關前夕（最後一天close）
    exit_open = get_open(1)      # 出關Day1 開盤

    # T-5~T-1 close（相對 ed）
    t_closes = {}
    if ci_e is not None:
        for t, off in [('T-5', -4), ('T-4', -3), ('T-3', -2), ('T-2', -1), ('T-1', 0)]:
            t_closes[t] = get_close(ci_e - ci + off)

    # ── 大戶持股變動 & 法人買超比（期間彙計）
    whale_delta = np.nan
    f_ratio = np.nan
    t_ratio = np.nan

    if sid in volume.columns and ci_e is not None:
        period_vol = volume[sid].iloc[ci:ci_e+1].sum()
        if period_vol > 0 and sid in inst_f.columns:
            f_ratio = round(inst_f[sid].iloc[ci:ci_e+1].sum() / period_vol * 100, 2)
        if period_vol > 0 and sid in inst_t.columns:
            t_ratio = round(inst_t[sid].iloc[ci:ci_e+1].sum() / period_vol * 100, 2)

    # 最大回撤（期間 high → low）
    max_drawdown = np.nan
    if ci_e is not None and sid in close.columns:
        period_close = close[sid].iloc[ci:ci_e+1].dropna()
        if len(period_close) > 0:
            period_max = period_close.max()
            period_min = period_close.min()
            if period_max > 0:
                max_drawdown = round((period_min - period_max) / period_max * 100, 2)

    row = {
        '市值規模':           cap,
        '股票代號':           sid,
        '股票名稱':           r.get('證券名稱', ''),
        '處置類型':           r['處置類型'],
        '處置原因':           reason,
        '處置起始日':         sd.strftime('%Y-%m-%d'),
        '大戶持股變動(%)':    whale_delta,
        '法人處置期間買超比(%)': f_ratio,
        '投信處置期間買超比(%)': t_ratio,
        'D1收盤報酬(%)':      safe_ret(c_D1, c_pre1),
        'D3收盤報酬(%)':      safe_ret(c_D3, c_pre1),
        'D5收盤報酬(%)':      safe_ret(c_D5, c_pre1),
        '出關前夕報酬(%)':    safe_ret(c_ed, c_pre1),
        '處置期間最大波動回撤(%)': max_drawdown,
        '出關Day1走勢(%)':   safe_ret(exit_open, c_ed) if not pd.isna(c_ed) else np.nan,
        '出關Day2走勢(%)':   np.nan,   # 需要出關Day2開盤，後續補
        '買進D1_出關D1賣出(%)':  safe_ret(exit_open, c_D1),
        '買進D3_出關D1賣出(%)':  safe_ret(exit_open, c_D3),
        '買進D5_出關D1賣出(%)':  safe_ret(exit_open, c_D5),
        '買進T-5_出關D1賣出(%)': safe_ret(exit_open, t_closes.get('T-5')),
        '買進T-4_出關D1賣出(%)': safe_ret(exit_open, t_closes.get('T-4')),
        '買進T-3_出關D1賣出(%)': safe_ret(exit_open, t_closes.get('T-3')),
        '買進T-2_出關D1賣出(%)': safe_ret(exit_open, t_closes.get('T-2')),
        '買進T-1_出關D1賣出(%)': safe_ret(exit_open, t_closes.get('T-1')),
    }
    return row


# ════════════════════════════════════════════════════════════════
# 6. 主流程
# ════════════════════════════════════════════════════════════════
def main():
    disp, close, open_p, basic, inst_f, inst_t, volume = load_all()

    # 市值輔助表
    basic_shares_map = {}
    basic['shares'] = pd.to_numeric(
        basic['已發行普通股數或TDR原發行股數'].astype(str).str.replace(',', ''), errors='coerce'
    )
    for _, row in basic.iterrows():
        basic_shares_map[str(row['stock_id'])] = row['shares']

    events = build_events(disp)

    # 處置原因改用近20日漲幅分類，不需要校正閾值

    # 逐事件計算
    print(f"\n🔄 計算 {len(events)} 個事件的報酬欄位...")
    records = []
    for nth, (_, r) in enumerate(events.iterrows(), 1):
        if nth % 200 == 0:
            print(f"  {nth}/{len(events)}")
        rec = compute_row(r, close, open_p, inst_f, inst_t, volume, basic_shares_map)
        if rec:
            records.append(rec)

    df_out = pd.DataFrame(records)
    print(f"\n✅ 完成: {len(df_out)} 筆")
    print("\n市值分布:")
    print(df_out['市值規模'].value_counts().to_string())
    print("\n處置類型:")
    print(df_out['處置類型'].value_counts().to_string())
    print("\n處置原因:")
    print(df_out['處置原因'].value_counts().to_string())
    print("\n年度分布:")
    print(pd.to_datetime(df_out['處置起始日']).dt.year.value_counts().sort_index().to_string())

    df_out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n💾 已儲存至 {OUT_CSV}")


if __name__ == '__main__':
    main()
