# -*- coding: utf-8 -*-
"""
大盤減碼信號 — 4 因子框架
小漲開盤（gap +0.1%~+1.5%）時的當日走勢預測
回測基礎：2019-2026 共 1,521 交易日
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from datetime import datetime, date, timedelta

try:
    import pytz
    TZ = pytz.timezone("Asia/Taipei")
except ImportError:
    TZ = None

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

st.set_page_config(
    page_title="大盤減碼信號",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── 回測歷史勝率（固定，2019-2026 回測結果）─────────────────────────────────
WIN_3 = {
    0: dict(n=260, rate=0.296, ret=+0.0038, color="#26c281", label="持倉觀望"),
    1: dict(n=347, rate=0.369, ret=+0.0028, color="#26c281", label="持倉觀望"),
    2: dict(n=59,  rate=0.678, ret=-0.0025, color="#f6c90e", label="考慮減碼 30-50%"),
    3: dict(n=10,  rate=1.000, ret=-0.0046, color="#e74c3c", label="強烈減碼 50-100%"),
}
WIN_4 = {
    0: dict(n=120, rate=0.275, ret=+0.0035, color="#26c281", label="持倉觀望"),
    1: dict(n=311, rate=0.357, ret=+0.0028, color="#26c281", label="持倉觀望"),
    2: dict(n=217, rate=0.396, ret=+0.0027, color="#26c281", label="觀望偏謹慎"),
    3: dict(n=23,  rate=0.870, ret=-0.0036, color="#f6c90e", label="考慮減碼 30-50%"),
    4: dict(n=5,   rate=1.000, ret=-0.0045, color="#e74c3c", label="強烈減碼 50-100%"),
}

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.factor-card {
    background: #1e2530; border-radius:10px; padding:14px 18px;
    margin:6px 0; display:flex; justify-content:space-between; align-items:center;
}
.factor-label { font-size:15px; color:#dce1e7; }
.factor-value { font-size:14px; color:#8899aa; margin-top:2px; }
.badge-green  { background:#1a4a2e; color:#26c281; border-radius:6px; padding:4px 10px; font-weight:700; font-size:14px; }
.badge-red    { background:#4a1a1a; color:#e74c3c; border-radius:6px; padding:4px 10px; font-weight:700; font-size:14px; }
.badge-grey   { background:#2a3040; color:#8899aa; border-radius:6px; padding:4px 10px; font-size:14px; }
.score-big    { font-size:52px; font-weight:800; line-height:1; }
.win-rate-big { font-size:32px; font-weight:700; }
.decision-box { border-radius:12px; padding:20px 24px; margin-top:16px; text-align:center; }
.stMetric     { background:#1e2530; border-radius:8px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ── 資料抓取 ─────────────────────────────────────────────────────────────

def now_tw():
    if TZ:
        return datetime.now(TZ)
    return datetime.utcnow() + timedelta(hours=8)

def today_tw():
    return now_tw().date()

@st.cache_data(ttl=300)
def get_taiex_info():
    """yfinance 抓今日 TAIEX 開盤/現價 vs 前日收盤。"""
    if not HAS_YF:
        return {}
    try:
        df = yf.download("^TWII", period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        today = today_tw()
        if df.index[-1].date() == today:
            today_open  = float(df["Open"].iloc[-1])
            prev_close  = float(df["Close"].iloc[-2])
            today_high  = float(df["High"].iloc[-1])
            today_close = float(df["Close"].iloc[-1])
            gap = (today_open - prev_close) / prev_close
            high_open = (today_high - today_open) / today_open
            return dict(open=today_open, prev_close=prev_close,
                        high=today_high, close=today_close,
                        gap=gap, high_open=high_open, has_today=True)
        else:
            # 非交易日或資料未到
            prev_close = float(df["Close"].iloc[-1])
            return dict(prev_close=prev_close, has_today=False)
    except Exception as e:
        return dict(err=str(e))


@st.cache_data(ttl=300)
def get_foreign_signals():
    """yfinance 抓 N225 / KS11 今日開盤缺口、NQ 前日報酬。"""
    if not HAS_YF:
        return {}
    result = {}
    # N225 & KS11：用 5-min 資料抓今日開盤 vs 昨收
    for name, ticker in [("n225", "^N225"), ("ks11", "^KS11")]:
        try:
            df5 = yf.download(ticker, period="2d", interval="5m",
                              auto_adjust=True, progress=False)
            if df5.empty:
                continue
            if isinstance(df5.columns, pd.MultiIndex):
                df5.columns = df5.columns.get_level_values(0)
            df5.index = pd.to_datetime(df5.index)
            # 轉台灣時區
            if df5.index.tzinfo is not None:
                if TZ:
                    df5.index = df5.index.tz_convert(TZ)
            today = today_tw()
            today_bars = df5[df5.index.date == today]
            prev_bars  = df5[df5.index.date < today]
            if len(today_bars) > 0 and len(prev_bars) > 0:
                today_open = float(today_bars["Open"].iloc[0])
                prev_close = float(prev_bars["Close"].iloc[-1])
                gap = (today_open - prev_close) / prev_close
                result[f"{name}_gap"]  = gap
                result[f"{name}_open"] = today_open
                result[f"{name}_prev"] = prev_close
        except Exception as e:
            result[f"{name}_err"] = str(e)

    # NQ 前日報酬（前日收 vs 前前日收）
    try:
        df_nq = yf.download("NQ=F", period="5d", interval="1d",
                            auto_adjust=True, progress=False)
        if not df_nq.empty and len(df_nq) >= 2:
            if isinstance(df_nq.columns, pd.MultiIndex):
                df_nq.columns = df_nq.columns.get_level_values(0)
            df_nq = df_nq.sort_index()
            # 今日已知的「前日 NQ 報酬」= 昨日相對前日
            closes = df_nq["Close"].values
            result["nq_prev_ret"] = (float(closes[-2]) - float(closes[-3])) / float(closes[-3])
    except Exception as e:
        result["nq_err"] = str(e)

    return result


@st.cache_data(ttl=120)
def get_twse_volume_ratio(date_str: str):
    """
    抓 TWSE MI_5MINS：9:10 累積量 × 8 ÷ 前日總量。
    前日總量用 FMTQIK API 取得。
    """
    # 今日 MI_5MINS
    url = (f"https://www.twse.com.tw/exchangeReport/MI_5MINS"
           f"?response=json&date={date_str}")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        if data.get("stat") != "OK" or "data" not in data:
            return None
        rows = data["data"]
        df = pd.DataFrame(rows, columns=["time", "vol", "amt"])
        df["amt"] = pd.to_numeric(df["amt"].str.replace(",", ""), errors="coerce")
        at_910 = df[df["time"] <= "09:10:00"]
        if at_910.empty:
            return None
        cum_910 = at_910["amt"].iloc[-1]
        total_today = df["amt"].iloc[-1]  # 今日到目前的累積（用於計算比例時更新）
    except Exception:
        return None

    # 前日總量 via FMTQIK（取本月資料）
    prev_total = None
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        url2 = (f"https://www.twse.com.tw/exchangeReport/FMTQIK"
                f"?response=json&date={date_str}")
        r2 = requests.get(url2, headers=headers, timeout=10)
        d2 = r2.json()
        if d2.get("stat") == "OK" and "data" in d2:
            rows2 = d2["data"]
            # 找前一個交易日的成交金額（第3欄）
            # 日期格式 "115/06/23" (民國)
            for row in reversed(rows2):
                try:
                    roc_date = row[0]   # "115/06/23"
                    y, m, d = roc_date.split("/")
                    greg_date = date(int(y)+1911, int(m), int(d))
                    if greg_date < dt.date():
                        amt_str = row[2].replace(",", "")  # 成交金額（千元）
                        prev_total = float(amt_str) * 1000  # 轉換為元
                        break
                except Exception:
                    continue
    except Exception:
        pass

    if prev_total and prev_total > 0 and cum_910 > 0:
        ratio = cum_910 * 8 / prev_total
        return dict(cum_910=cum_910, prev_total=prev_total,
                    ratio=ratio, total_today=total_today)
    return dict(cum_910=cum_910, prev_total=prev_total, ratio=None)


def load_tx_oi():
    """讀 data/tx_oi_latest.json。"""
    fp = DATA_DIR / "tx_oi_latest.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


@st.cache_data(ttl=3600)
def load_history():
    """讀預先計算的歷史因子表。"""
    fp = DATA_DIR / "taiex_factor_history.csv"
    if fp.exists():
        try:
            df = pd.read_csv(fp, parse_dates=["date"], index_col="date",
                             encoding="utf-8-sig")
            return df
        except Exception:
            pass
    return None


# ── Helper: factor badge ──────────────────────────────────────────────────
def badge(triggered: bool | None, true_label="✅", false_label="❌", na_label="⏳ 待資料"):
    if triggered is None:
        return f'<span class="badge-grey">{na_label}</span>'
    elif triggered:
        return f'<span class="badge-green">{true_label}</span>'
    else:
        return f'<span class="badge-red">{false_label}</span>'


def factor_row(label: str, value_str: str, triggered):
    b = badge(triggered)
    return f"""
    <div class="factor-card">
      <div>
        <div class="factor-label">{label}</div>
        <div class="factor-value">{value_str}</div>
      </div>
      {b}
    </div>
    """


# ── Main UI ───────────────────────────────────────────────────────────────
st.title("大盤減碼信號 📉")

col_now, col_refresh = st.columns([4, 1])
with col_now:
    now = now_tw()
    st.caption(f"更新：{now.strftime('%Y-%m-%d %H:%M')} 台灣時間  ·  回測基礎 2019-2026 / 1,521 交易日")
with col_refresh:
    if st.button("🔄 重新整理"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── 抓資料 ────────────────────────────────────────────────────────────────
date_str = now.strftime("%Y%m%d")
taiex    = get_taiex_info()
foreign  = get_foreign_signals()
tx_data  = load_tx_oi()

# ── 判斷小漲條件 ──────────────────────────────────────────────────────────
gap_pct     = taiex.get("gap")
is_small_up = (gap_pct is not None) and (0.001 <= gap_pct <= 0.015)

gap_label = "—"
if gap_pct is not None:
    sign  = "+" if gap_pct >= 0 else ""
    color = "#26c281" if is_small_up else "#8899aa"
    gap_label = f'<span style="color:{color};font-size:22px;font-weight:700">{sign}{gap_pct:.2%}</span>'
    if is_small_up:
        gap_label += ' <span style="color:#26c281;font-size:13px">✅ 小漲條件成立</span>'
    elif gap_pct > 0.015:
        gap_label += ' <span style="color:#8899aa;font-size:13px">缺口過大（>1.5%），不在預測範圍</span>'
    else:
        gap_label += ' <span style="color:#8899aa;font-size:13px">非小漲開盤，信號不適用</span>'

st.markdown(f"### 今日 TAIEX 開盤缺口　{gap_label}", unsafe_allow_html=True)

if not is_small_up and gap_pct is not None:
    st.info("⚠️ 今日開盤缺口不在 +0.1%~+1.5% 範圍，4因子框架不適用。以下數值僅供參考。")

st.write("")

# ── 四因子評分 ────────────────────────────────────────────────────────────
col_factors, col_score = st.columns([5, 3])

with col_factors:
    st.subheader("因子檢查清單")

    # ① 外圍跌數（N225 + KS11 + NQ）
    n225_gap  = foreign.get("n225_gap")
    ks11_gap  = foreign.get("ks11_gap")
    nq_ret    = foreign.get("nq_prev_ret")
    n_down    = sum([
        1 if (n225_gap is not None and n225_gap < 0) else 0,
        1 if (ks11_gap is not None and ks11_gap < 0) else 0,
        1 if (nq_ret   is not None and nq_ret   < 0) else 0,
    ])
    f_foreign = None
    foreign_val = "—"
    if all(v is not None for v in [n225_gap, ks11_gap, nq_ret]):
        f_foreign = n_down >= 2
        parts = []
        if n225_gap is not None:
            parts.append(f"N225 {'+' if n225_gap>=0 else ''}{n225_gap:.2%}")
        if ks11_gap is not None:
            parts.append(f"KS11 {'+' if ks11_gap>=0 else ''}{ks11_gap:.2%}")
        if nq_ret is not None:
            parts.append(f"NQ前日 {'+' if nq_ret>=0 else ''}{nq_ret:.2%}")
        foreign_val = "  ·  ".join(parts) + f"　→ {n_down}/3 跌"
    elif any(v is not None for v in [n225_gap, ks11_gap, nq_ret]):
        parts = []
        if n225_gap is not None: parts.append(f"N225 {n225_gap:+.2%}")
        if ks11_gap is not None: parts.append(f"KS11 {ks11_gap:+.2%}")
        if nq_ret   is not None: parts.append(f"NQ前日 {nq_ret:+.2%}")
        foreign_val = "  ·  ".join(parts) + f"  ({n_down}/3 跌，部分資料待更新)"

    st.markdown(factor_row(
        "① 外圍跌 ≥ 2　（N225+KS11開盤缺口、NQ前日報酬）",
        foreign_val, f_foreign
    ), unsafe_allow_html=True)

    # ② 量能比 at 9:10
    h, m = now.hour, now.minute
    vol_data = None
    f_vol    = None
    vol_val  = "—"
    if (h > 9) or (h == 9 and m >= 12):  # 9:12 後才有意義
        vol_data = get_twse_volume_ratio(date_str)
    if vol_data:
        ratio = vol_data.get("ratio")
        if ratio is not None:
            f_vol   = ratio >= 1.2
            vol_val = f"9:10 累積量 × 8 ÷ 昨日 = {ratio:.2f}x"
        elif vol_data.get("cum_910"):
            vol_val = f"9:10 累積量 {vol_data['cum_910']:,.0f}（前日量待取得）"
    elif h < 9 or (h == 9 and m < 12):
        vol_val = "市場未開盤或 9:10 前，稍後自動更新"

    st.markdown(factor_row(
        "② 開盤爆量　（9:10 累積量 × 8 ÷ 昨日 ≥ 1.2x）",
        vol_val, f_vol
    ), unsafe_allow_html=True)

    # ③ TX 外資期貨淨空
    tx_oi   = tx_data.get("tx_oi") if tx_data else None
    f_fut   = None
    tx_val  = "—"
    tx_note = ""
    if tx_oi is not None:
        f_fut   = tx_oi < -10000
        tx_val  = f"臺股期貨外資淨未平倉 {tx_oi:+,} 口"
        tx_date = tx_data.get("date", "")
        tx_note = f"（資料日期：{tx_date}，需手動更新）"
        if abs(tx_oi) > 50000:
            tx_note += " ⚠️ 極端淨空 > 5萬口，short squeeze 風險，降低信心"

    st.markdown(factor_row(
        "③ TX 外資期貨淨空 < -10,000 口　（前日收盤後公告）",
        f"{tx_val}<br><small style='color:#667'>{tx_note}</small>" if tx_note else tx_val,
        f_fut
    ), unsafe_allow_html=True)

    # ④ 漲不動（High-Open < 0.3%）
    high_open = taiex.get("high_open")
    f_rise    = None
    rise_val  = "—"
    if high_open is not None:
        f_rise   = high_open < 0.003
        rise_val = f"(High-Open)/Open = {high_open:.3%}"
        if not taiex.get("has_today"):
            f_rise  = None
            rise_val = "今日尚未開盤"
    if h < 9 or (h == 9 and m < 30):
        rise_val = "9:10-9:30 盤中觀察：大盤拉高後縮回即觸發"
        f_rise = None

    st.markdown(factor_row(
        "④ 漲不動　（盤中觀察：High-Open < 0.3%，或拉高後收縮）",
        rise_val, f_rise
    ), unsafe_allow_html=True)


# ── 得分與決策 ────────────────────────────────────────────────────────────
with col_score:
    st.subheader("今日得分")

    # 計算得分（None 算缺資料，不計入）
    factors_3 = [f_foreign, f_vol, f_rise]
    factors_4 = [f_foreign, f_vol, f_rise, f_fut]

    known_3 = [f for f in factors_3 if f is not None]
    known_4 = [f for f in factors_4 if f is not None]
    score_3  = sum(known_3)
    score_4  = sum(known_4)
    max_3    = len(known_3)
    max_4    = len(known_4)

    has_tx   = f_fut is not None

    if has_tx and max_4 >= 3:
        # 使用 4 因子框架
        wdata = WIN_4.get(score_4, WIN_4[0])
        st.markdown(
            f'<div class="score-big" style="color:{wdata["color"]}">'
            f'{score_4}<span style="font-size:28px;font-weight:400"> / {max_4}</span>'
            f'</div><div style="color:#8899aa;font-size:13px;margin-top:4px">4 因子框架（含期貨OI）</div>',
            unsafe_allow_html=True
        )
    else:
        # 使用 3 因子框架
        wdata = WIN_3.get(score_3, WIN_3[0])
        st.markdown(
            f'<div class="score-big" style="color:{wdata["color"]}">'
            f'{score_3}<span style="font-size:28px;font-weight:400"> / {max_3}</span>'
            f'</div><div style="color:#8899aa;font-size:13px;margin-top:4px">3 因子框架（期貨OI 待更新）</div>',
            unsafe_allow_html=True
        )

    st.write("")
    rate_str = f"{wdata['rate']:.0%}"
    ret_str  = f"{wdata['ret']:+.2%}"
    n_str    = f"n = {wdata['n']}"
    color    = wdata["color"]
    label    = wdata["label"]

    st.markdown(
        f'<div class="win-rate-big" style="color:{color}">{rate_str} 走低率</div>'
        f'<div style="color:#8899aa;font-size:13px">平均報酬 {ret_str}  ·  歷史樣本 {n_str}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="decision-box" style="background:{color}22;border:2px solid {color}44;margin-top:20px">'
        f'<div style="color:{color};font-size:18px;font-weight:700">{label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if tx_data is None:
        st.caption("⚠️ TX 期貨OI 資料未找到，請執行 `update_market_signal.py` 後推到 GitHub。")

st.divider()

# ── 歷史勝率表 ────────────────────────────────────────────────────────────
col_t3, col_t4 = st.columns(2)

with col_t3:
    st.subheader("3 因子框架 勝率表")
    st.caption("量能 + 外圍跌 + 漲不動")
    rows3 = []
    for s, w in WIN_3.items():
        rows3.append({
            "得分": f"{s}/3",
            "樣本數": w["n"],
            "走低率": f"{w['rate']:.0%}",
            "平均報酬": f"{w['ret']:+.2%}",
            "建議": w["label"],
        })
    df3 = pd.DataFrame(rows3)
    st.dataframe(df3, hide_index=True, use_container_width=True)

with col_t4:
    st.subheader("4 因子框架 勝率表")
    st.caption("量能 + 外圍跌 + 漲不動 + TX外資淨空")
    rows4 = []
    for s, w in WIN_4.items():
        rows4.append({
            "得分": f"{s}/4",
            "樣本數": w["n"],
            "走低率": f"{w['rate']:.0%}",
            "平均報酬": f"{w['ret']:+.2%}",
            "建議": w["label"],
        })
    df4 = pd.DataFrame(rows4)
    st.dataframe(df4, hide_index=True, use_container_width=True)

st.divider()

# ── 歷史訊號明細 ─────────────────────────────────────────────────────────
st.subheader("近期歷史訊號")
hist = load_history()

if hist is not None and not hist.empty:
    disp = hist.copy()
    disp.index = pd.to_datetime(disp.index).strftime("%Y-%m-%d")
    disp = disp.sort_index(ascending=False).head(60)

    rename = {
        "gap_pct":       "開盤缺口",
        "high_open_pct": "High-Open",
        "day_return":    "當日報酬",
        "walked_lower":  "走低",
        "f_vol":         "量能",
        "f_foreign":     "外圍",
        "f_rise":        "漲不動",
        "f_fut_short":   "期貨淨空",
        "score_3":       "3因子分",
        "score_4":       "4因子分",
        "tx_oi":         "TX淨OI",
        "ratio_910":     "量比",
        "n_foreign_down":"外圍跌數",
    }
    show_cols = [c for c in rename if c in disp.columns]
    disp = disp[show_cols].rename(columns=rename)

    # 格式化百分比欄
    for col in ["開盤缺口", "High-Open", "當日報酬"]:
        if col in disp.columns:
            disp[col] = disp[col].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    for col in ["量能", "外圍", "漲不動", "期貨淨空", "走低"]:
        if col in disp.columns:
            disp[col] = disp[col].map(lambda x: "✅" if x == 1 else ("❌" if x == 0 else "—"))
    if "量比" in disp.columns:
        disp["量比"] = disp["量比"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    if "TX淨OI" in disp.columns:
        disp["TX淨OI"] = disp["TX淨OI"].map(lambda x: f"{x:+,.0f}" if pd.notna(x) else "—")

    st.dataframe(disp, use_container_width=True)
    last_update = hist.index.max()
    st.caption(f"資料截至：{last_update}　·　執行 `update_market_signal.py` 並推 GitHub 以更新")
else:
    st.info(
        "歷史資料尚未生成。\n\n"
        "請在本機執行：\n"
        "```bash\n"
        "python disposal-signals/update_market_signal.py\n"
        "```\n"
        "然後 `git add data/ && git commit -m 'update market signal data' && git push`"
    )

st.divider()

# ── 方法說明 ─────────────────────────────────────────────────────────────
with st.expander("📖 因子定義與反直覺發現"):
    st.markdown("""
### 因子定義

| 因子 | 判斷條件 | 可知時間 |
|------|---------|---------|
| **量能** | 9:10 累積成交金額 × 8 ÷ 昨日全日 ≥ **1.2x** | 9:10 |
| **外圍跌** | N225+KS11 今日開盤缺口 < 0（各+1分），NQ 前日報酬 < 0（+1分），≥ **2點** 觸發 | 8:00 |
| **漲不動** | (High-Open)/Open < **0.3%**（日線），盤中觀察拉高後縮回 | 9:10-9:30 |
| **TX外資淨空** | 臺股期貨外資及陸資未平倉口數淨額 < **-10,000 口**（前日收後公告） | 8:00 |

### 反直覺發現（請勿使用以下因子）

- ❌ **外資現貨大量賣超** → 次日走低率 **低於**基準（33.7%）
- ❌ **外資現貨大量買超** → 次日走低率 **高於**基準（46.2%）
- ❌ **TX期貨極端淨空 < -50,000口** → 走低率反而只有 10%（軋空風險）

> 解釋：法人在支撐點買（低位承接），在阻力點賣（高位出貨），是 mean-reversion 參與者，
> 其現貨交易方向不能直接用於預測次日趨勢。

### 最強單因子

| 條件 | n | 走低率 |
|------|---|--------|
| High-Open < 0.2% | 76 | **92.1%** |
| High-Open < 0.3% | 134 | **74.6%** |

### 最強組合

| 組合 | n | 走低率 |
|------|---|--------|
| 量>1.0x + ≥1外圍跌 + 漲不動<0.2% | 28 | **96.4%** |
| 量>1.2x + ≥1外圍跌 + 漲不動<0.2% | 13 | **100%** |
    """)

with st.expander("🔧 TX 外資期貨OI 更新方式"):
    st.markdown("""
TX 外資期貨未平倉口數從 **TAIFEX 官方資料** 透過 finlab 取得。

**每日更新步驟：**

1. 在本機執行 finlab 資料更新：
   ```bash
   python D:/stock/finlab_update.py
   ```

2. 執行信號資料更新腳本：
   ```bash
   python D:/stock/disposal-signals/update_market_signal.py
   ```

3. 推到 GitHub：
   ```bash
   cd D:/stock/disposal-signals
   git add data/tx_oi_latest.json data/taiex_factor_history.csv
   git commit -m "daily: update market signal data"
   git push
   ```

Streamlit Cloud 會在約 1-2 分鐘內自動重新讀取最新資料。

**資料來源：** `D:\\stock\\finlab_db\\futures_inst_net_oi.parquet` → 欄位 `臺股期貨_外資及陸資`
    """)
