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

    # 2026-08-25發現：finlab的分時交易欄位偶爾對某些搓合頻率解析失敗留空
    # （例如3625西勝2026-08-24公告的事件，處置內容原文寫「約每10分鐘撮合一次」，
    # 但分時交易是NaN，導致整支股票被下面的notna()篩掉，完全消失不見）。
    # 用處置內容原文文字直接補值當後備，不完全依賴finlab自己的欄位。
    if '分時交易' in df.columns and '處置內容' in df.columns:
        missing = df['分時交易'].isna()
        if missing.any():
            extracted = df.loc[missing, '處置內容'].str.extract(r'每\s*(\d+)\s*分鐘撮合')[0]
            df.loc[missing, '分時交易'] = pd.to_numeric(extracted, errors='coerce')

    # 分時交易有值
    df = df[df['分時交易'].notna()]

    # 對應處置類型：直接用數字格式化，不用固定字典——固定字典每次遇到新的搓合頻率
    # （例如這次的10分鐘）就要手動補一筆，改成通用格式化可以一勞永逸。
    df['處置類型'] = df['分時交易'].apply(lambda v: f'{v:g}分鐘')

    # 處置次別（第一次/第二次+）：2026-08-10新制上路後，第一次與第二次都統一為2分鐘撮合，
    # 處置類型無法再間接代表次別，資金收款門檻（單筆10/累積30張 vs 所有投資人全額收款）
    # 仍照次別區分且文字與舊制逐字相同，故另存此欄供新舊制對照用。
    df['處置次別'] = df['處置措施'].apply(lambda v: '第一次' if '第一次' in str(v) else '第二次+')

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
        # basic_info 只有現存上市公司；下市股撈不到股本 → 當小型股保留，避免 survivorship bias
        return '小型股(<100億)'
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

    # searchsorted 統一處理三種情況：
    #   起始日=交易日 → 對到當天；起始日=停市日(颱風假) → 對到下一個交易日
    #   起始日還沒有收盤價(今天/未來才開始) → ci=len，報酬欄位留 NaN
    ci = close.index.searchsorted(sd)
    ci_e = close_idx.get(ed, None)

    # ── 市值
    cap = classify_cap(sid, sd, close, basic_shares_map)

    # ── 處置原因（近20日漲幅：>= 0% 為漲多，< 0% 為跌深）
    # 2026-08-24修正：台股正常漲跌停約±10%，20天窗口內若出現單日變動遠超此範圍
    # （減資恢復買賣參考價重設、除權息基準價調整等未還原的價格斷層），會把「漲多/跌深」
    # 判斷完全拉偏（例如5314世紀*：8/14單日收盤從61.30跳到16.20，-73.6%，把20天漲幅
    # 拖成-53.65%誤判跌深，但那不是真實市場交易造成的跌幅）。偵測到就標記異常，不套用
    # 漲多/跌深分類，避免下游策略誤把資料斷層當成真實跌深/漲多訊號。
    ANOMALY_DAILY_MOVE = 0.20  # 20%，明顯超過±10%正常漲跌停的保守門檻
    if ci >= 22:
        col = close.columns.get_loc(sid)
        window = close.iloc[ci-21:ci, col]  # 20天窗口內每天收盤（用來抓斷層，不只看頭尾）
        daily_ret = window.pct_change().dropna()
        has_anomaly_jump = (daily_ret.abs() > ANOMALY_DAILY_MOVE).any()
        c_pre1  = close.iloc[ci-1,  col]
        c_pre21 = close.iloc[ci-21, col]
        if has_anomaly_jump:
            reason = '⚠️資料異常(疑似減資/除權息斷層)'
        elif not pd.isna(c_pre1) and not pd.isna(c_pre21) and c_pre21 > 0:
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
        '處置次別':           r['處置次別'],
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
