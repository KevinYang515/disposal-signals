# -*- coding: utf-8 -*-
r"""
每日更新大盤減碼信號資料 — VM / 本機通用版
不依賴 finlab 或本機 parquet，全部使用公開 API。

執行：
  python update_market_signal.py

輸出到 data/:
  tx_oi_latest.json        — 最新 TX 外資未平倉口數（TAIFEX 官網）
  taiex_factor_history.csv — 小漲開盤日因子明細（增量更新）

VM cron 建議（台灣時間 07:30，UTC 23:30 前一天）:
  30 23 * * 0-4  cd /path/to/disposal-signals && python update_market_signal.py
"""
import sys, os, json, time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from io import StringIO

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("yfinance 未安裝，外圍因子將跳過")

OUT_DIR  = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)
HIST_CSV = OUT_DIR / "taiex_factor_history.csv"
TX_JSON  = OUT_DIR / "tx_oi_latest.json"

HEADERS  = {"User-Agent": "Mozilla/5.0 (compatible; market-signal-bot/1.0)"}


# ═══════════════════════════════════════════════════════════════════════════
# 資料抓取函式
# ═══════════════════════════════════════════════════════════════════════════

def fetch_tx_oi(date_str: str) -> int | None:
    """
    從 TAIFEX 官網抓 TX 期貨外資及陸資「多空未平倉口數淨額」。
    date_str: "YYYYMMDD"
    """
    q_date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}"
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"

    try:
        r = requests.post(url, headers=HEADERS,
                          data={"queryDate": q_date, "commodityId": "TXF"},
                          timeout=15)
        r.encoding = "utf-8"
        if "查無資料" in r.text or r.status_code != 200:
            return None

        tables = pd.read_html(StringIO(r.text), thousands=",")
        for t in tables:
            flat = t.to_string()
            if "臺股期貨" not in flat or "外資" not in flat:
                continue
            # 表格結構：col[-2] = 未平倉餘額 / 多空淨額 / 口數
            # 找「臺股期貨 + 外資」那一列
            for _, row in t.iterrows():
                vals = list(row.values)
                if len(vals) >= 14 and "臺股期貨" in str(vals[1]) and "外資" in str(vals[2]):
                    # vals[-2] = 未平倉多空淨額口數（正 = 淨多、負 = 淨空）
                    oi_val = vals[-2]
                    if isinstance(oi_val, (int, float)) and not np.isnan(float(oi_val)):
                        return int(oi_val)
    except Exception as e:
        print(f"  TAIFEX scrape error: {e}")
    return None


def fetch_taiex_ohlc_twse(date_str: str) -> dict | None:
    """
    從 TWSE FMTQIK 拿當月所有交易日的加權指數（含收盤），
    再從 MI_INDEX 拿當日 open/high/low。
    date_str: "YYYYMMDD"
    """
    result = {}

    # MI_INDEX 取當日 OHLC（大盤加權指數）
    url = (f"https://www.twse.com.tw/exchangeReport/MI_INDEX"
           f"?response=json&date={date_str}&type=IND")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        d = r.json()
        if d.get("stat") == "OK":
            for row in d.get("data9", []):
                if "發行量加權股價指數" in str(row[0]):
                    # row: [名稱, 開盤, 最高, 最低, 收盤, ...]
                    def to_f(s):
                        return float(str(s).replace(",", ""))
                    result["open"]  = to_f(row[1])
                    result["high"]  = to_f(row[2])
                    result["low"]   = to_f(row[3])
                    result["close"] = to_f(row[4])
                    break
    except Exception as e:
        print(f"  MI_INDEX error: {e}")

    return result if result else None


def fetch_taiex_monthly(year_month: str) -> pd.Series:
    """
    FMTQIK：取整個月每日加權指數收盤（用於算 gap）。
    year_month: "YYYYMM01"（任意該月日期即可）
    返回 Series，index=date, value=close
    """
    url = (f"https://www.twse.com.tw/exchangeReport/FMTQIK"
           f"?response=json&date={year_month}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        d = r.json()
        if d.get("stat") != "OK":
            return pd.Series(dtype=float)
        rows = d["data"]
        records = {}
        for row in rows:
            # 日期格式 "115/06/24"（民國）
            roc = row[0]
            y, m, dd = roc.split("/")
            greg = date(int(y) + 1911, int(m), int(dd))
            close = float(row[4].replace(",", ""))  # 大盤指數
            records[greg] = close
        return pd.Series(records, dtype=float)
    except Exception as e:
        print(f"  FMTQIK error: {e}")
        return pd.Series(dtype=float)


def fetch_twse_volume_at_910(date_str: str) -> tuple[float | None, float | None]:
    """
    抓 MI_5MINS，回傳 (cum_910_百萬元, daily_total_百萬元)。
    """
    url = (f"https://www.twse.com.tw/exchangeReport/MI_5MINS"
           f"?response=json&date={date_str}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        d = r.json()
        if d.get("stat") != "OK":
            return None, None
        rows = d["data"]
        df = pd.DataFrame(rows, columns=["time", "vol", "amt"])
        df["amt"] = pd.to_numeric(df["amt"].str.replace(",", ""), errors="coerce")
        at910 = df[df["time"] <= "09:10:00"]
        if at910.empty:
            return None, None
        return float(at910["amt"].iloc[-1]), float(df["amt"].iloc[-1])
    except Exception as e:
        print(f"  MI_5MINS error: {e}")
        return None, None


def fetch_foreign_signals(start_date: str, end_date: str) -> pd.DataFrame:
    """yfinance 抓 N225/KS11 開盤缺口、NQ 前日報酬。"""
    if not HAS_YF:
        return pd.DataFrame()

    parts = []
    for name, ticker in [("n225", "^N225"), ("ks11", "^KS11")]:
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df.sort_index()
            df[f"{name}_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
            parts.append(df[[f"{name}_gap"]])
        except Exception as e:
            print(f"  yfinance {ticker}: {e}")

    try:
        df_nq = yf.download("NQ=F", start=start_date, end=end_date,
                            auto_adjust=True, progress=False)
        if not df_nq.empty:
            if isinstance(df_nq.columns, pd.MultiIndex):
                df_nq.columns = df_nq.columns.get_level_values(0)
            df_nq.index = pd.to_datetime(df_nq.index).tz_localize(None).normalize()
            df_nq = df_nq.sort_index()
            nq_ret = df_nq["Close"].pct_change(fill_method=None)
            parts.append(nq_ret.shift(1).rename("nq_prev_ret").to_frame())
    except Exception as e:
        print(f"  yfinance NQ=F: {e}")

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# 組合歷史因子 / 增量更新
# ═══════════════════════════════════════════════════════════════════════════

def get_trading_days_needed(hist: pd.DataFrame) -> list[date]:
    """找出歷史 CSV 缺少的交易日（最多回補 60 天）。"""
    last = hist.index.max().date() if not hist.empty else date(2024, 1, 1)
    today = date.today()
    candidates = []
    d = last + timedelta(days=1)
    while d <= today:
        if d.weekday() < 5:  # 週一~週五
            candidates.append(d)
        d += timedelta(days=1)
    return candidates[-60:]  # 最多補 60 天


def compute_row(dt: date, prev_close: float | None,
                foreign_df: pd.DataFrame) -> dict | None:
    """
    對單一交易日計算所有因子，回傳 dict 或 None（非交易日/資料不足）。
    """
    date_str = dt.strftime("%Y%m%d")

    # TAIEX OHLC
    ohlc = fetch_taiex_ohlc_twse(date_str)
    if not ohlc or "open" not in ohlc:
        return None  # 非交易日或 API 無資料

    o, h, c = ohlc["open"], ohlc["high"], ohlc["close"]

    if prev_close is None or prev_close == 0:
        return None

    gap_pct       = (o - prev_close) / prev_close
    high_open_pct = (h - o) / o
    day_return    = (c - o) / o
    walked_lower  = int(c < o)
    f_rise        = int(high_open_pct < 0.003)

    # 小漲過濾（只記錄、保留全部，Streamlit 端自行過濾）
    is_small_up = 0.001 <= gap_pct <= 0.015

    # 外圍因子
    n225_gap   = float(foreign_df.loc[dt, "n225_gap"]) if dt in foreign_df.index and "n225_gap" in foreign_df.columns else np.nan
    ks11_gap   = float(foreign_df.loc[dt, "ks11_gap"]) if dt in foreign_df.index and "ks11_gap" in foreign_df.columns else np.nan
    nq_prev    = float(foreign_df.loc[dt, "nq_prev_ret"]) if dt in foreign_df.index and "nq_prev_ret" in foreign_df.columns else np.nan

    n_foreign_down = sum([
        1 if (not np.isnan(n225_gap) and n225_gap < 0) else 0,
        1 if (not np.isnan(ks11_gap) and ks11_gap < 0) else 0,
        1 if (not np.isnan(nq_prev)  and nq_prev  < 0) else 0,
    ])
    f_foreign = int(n_foreign_down >= 2) if not (np.isnan(n225_gap) and np.isnan(ks11_gap) and np.isnan(nq_prev)) else np.nan

    # 量能比（需前一個交易日的量）
    cum_910, daily_mi_total = fetch_twse_volume_at_910(date_str)
    ratio_910 = np.nan
    f_vol     = np.nan
    if cum_910 is not None and daily_mi_total is not None:
        # 找前一交易日的 MI_5MINS total
        prev_dt = dt - timedelta(days=1)
        while prev_dt.weekday() >= 5:
            prev_dt -= timedelta(days=1)
        _, prev_mi_total = fetch_twse_volume_at_910(prev_dt.strftime("%Y%m%d"))
        if prev_mi_total and prev_mi_total > 0:
            ratio_910 = cum_910 * 8 / prev_mi_total  # 同單位，百萬元
            f_vol = int(ratio_910 >= 1.2)

    # TX 外資期貨 OI（前日公告）
    prev_dt_str = (dt - timedelta(days=1))
    # 找前一個交易日
    tmp = dt - timedelta(days=1)
    while tmp.weekday() >= 5:
        tmp -= timedelta(days=1)
    tx_oi = fetch_tx_oi(tmp.strftime("%Y%m%d"))
    f_fut_short = int(tx_oi < -10000) if tx_oi is not None else np.nan

    score_3 = sum([x for x in [f_vol, f_foreign, f_rise]
                   if isinstance(x, (int, float)) and not np.isnan(x)])
    score_4 = sum([x for x in [f_vol, f_foreign, f_rise, f_fut_short]
                   if isinstance(x, (int, float)) and not np.isnan(x)])

    return dict(
        gap_pct=gap_pct, high_open_pct=high_open_pct,
        day_return=day_return, walked_lower=walked_lower,
        f_vol=f_vol, f_foreign=f_foreign, f_rise=f_rise, f_fut_short=f_fut_short,
        score_3=score_3, score_4=score_4,
        tx_oi=tx_oi if tx_oi is not None else np.nan,
        ratio_910=ratio_910, n_foreign_down=n_foreign_down,
        is_small_up=int(is_small_up),
    )


def load_or_init_history() -> pd.DataFrame:
    if HIST_CSV.exists():
        df = pd.read_csv(HIST_CSV, parse_dates=["date"], index_col="date",
                         encoding="utf-8-sig")
        return df
    return pd.DataFrame()


def update_tx_oi_json():
    """更新 tx_oi_latest.json（取今日或最近交易日）。"""
    today = date.today()
    dt = today
    if dt.weekday() >= 5:
        dt -= timedelta(days=dt.weekday() - 4)

    # 嘗試今天和最近 3 個交易日
    for offset in range(5):
        check = dt - timedelta(days=offset)
        if check.weekday() >= 5:
            continue
        tx = fetch_tx_oi(check.strftime("%Y%m%d"))
        if tx is not None:
            result = {
                "date":    str(check),
                "tx_oi":   tx,
                "updated": datetime.now().isoformat(timespec="minutes"),
            }
            TX_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                               encoding="utf-8")
            print(f"  TX OI: {check} → {tx:+,} 口")
            return result
    print("  TX OI: 抓取失敗（TAIFEX 無資料或非交易日）")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 更新大盤減碼信號資料 ===")

    # 1. TX OI
    print("\n[1] TX 外資期貨未平倉口數（TAIFEX）...")
    update_tx_oi_json()

    # 2. 歷史因子增量更新
    print("\n[2] 歷史因子表（增量補日）...")
    hist = load_or_init_history()
    days_needed = get_trading_days_needed(hist)

    if not days_needed:
        print("  已是最新，無需補日。")
    else:
        start_str = (days_needed[0] - timedelta(days=5)).strftime("%Y-%m-%d")
        end_str   = (days_needed[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  需補 {len(days_needed)} 天（{days_needed[0]} ~ {days_needed[-1]}）")

        # 外圍因子一次抓（減少 API 呼叫）
        foreign_df = fetch_foreign_signals(start_str, end_str)
        foreign_df.index = pd.to_datetime(foreign_df.index).normalize()

        # 建立前收盤 lookup：用 yfinance ^TWII 補
        taiex_close_lookup: dict[date, float] = {}
        if HAS_YF:
            try:
                df_tw = yf.download("^TWII",
                                    start=(days_needed[0] - timedelta(days=10)).strftime("%Y-%m-%d"),
                                    end=end_str,
                                    auto_adjust=True, progress=False)
                if isinstance(df_tw.columns, pd.MultiIndex):
                    df_tw.columns = df_tw.columns.get_level_values(0)
                df_tw.index = pd.to_datetime(df_tw.index).tz_localize(None).normalize()
                for idx, row in df_tw.iterrows():
                    taiex_close_lookup[idx.date()] = float(row["Close"])
            except Exception as e:
                print(f"  ^TWII yfinance: {e}")

        # 補入既有歷史最後收盤
        if not hist.empty:
            last_date = hist.index.max().date()
            # 從 CSV 抓最後一個有 close 的值（這裡我們用 gap_pct 反推比較麻煩，直接用 yfinance）
            pass

        new_rows = []
        for dt in days_needed:
            prev_dt = dt - timedelta(days=1)
            while prev_dt.weekday() >= 5:
                prev_dt -= timedelta(days=1)
            prev_close = taiex_close_lookup.get(prev_dt)

            print(f"  計算 {dt}...", end="", flush=True)
            row = compute_row(dt, prev_close, foreign_df)
            if row is None:
                print(" (非交易日/無資料)")
                continue
            row["date"] = pd.Timestamp(dt)
            new_rows.append(row)
            print(f" gap={row['gap_pct']:+.2%} score={row['score_3']:.0f}/3  ratio={row['ratio_910']:.2f}x" if not np.isnan(row.get('ratio_910', float('nan'))) else f" gap={row['gap_pct']:+.2%} score={row['score_3']:.0f}/3")
            time.sleep(0.3)  # 避免 API rate limit

        if new_rows:
            new_df = pd.DataFrame(new_rows).set_index("date")
            # 只保留小漲開盤日
            new_df = new_df[new_df["is_small_up"] == 1].drop(columns=["is_small_up"], errors="ignore")
            hist = pd.concat([hist, new_df]).sort_index()
            # 去重
            hist = hist[~hist.index.duplicated(keep="last")]
            hist.to_csv(HIST_CSV, encoding="utf-8-sig", float_format="%.6f")
            print(f"\n  → 新增 {len(new_rows)} 天，總計 {len(hist)} 筆  → {HIST_CSV}")
        else:
            print("  沒有新資料。")

    print("\n=== 完成 ===")
    print("下一步：git add data/ && git commit -m 'daily update' && git push")
