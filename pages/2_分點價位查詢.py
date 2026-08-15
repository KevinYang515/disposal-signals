# -*- coding: utf-8 -*-
"""即時查詢 TWSE bshtm 個股分點進出，含價位級距明細。"""
from __future__ import annotations

import re

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="分點價位查詢", page_icon="🔍", layout="wide")

BASE = "https://bsr.twse.com.tw/bshtm/bsMenu.aspx"
CONTENT_URL = "https://bsr.twse.com.tw/bshtm/bsContent.aspx"
HEADERS = {"User-Agent": "Mozilla/5.0"}
ROW_PAT = re.compile(
    r"<tr class='column_value_price_\d+'>\s*"
    r"<td class='column_value_center'>\s*(\d+)</td>\s*"
    r"<td class='column_value_left'>\s*([^<]*?)\s*</td>\s*"
    r"<td class='column_value_right'>\s*([\d,]+\.\d+)</td>\s*"
    r"<td class='column_value_right'>\s*([\d,]*)</td>\s*"
    r"<td class='column_value_right'>\s*([\d,]*)</td>",
    re.S,
)

KNOWN_BRANCHES = {
    "9227": "凱基-城中（City-GA 訊號分點）",
    "5854": "統一-城中（UniCenter 訊號分點）",
    "9600": "富邦（母公司，非任何分店；富邦策略訊號分點）",
    "9B20": "台新-台北（研究監看，未驗證通過）",
    "9268": "凱基-台北（廣泛型造市分點，量體大屬正常，非訊號分點）",
}
SIGNAL_BRANCH_CODES = ["9227", "5854", "9600", "9B20"]


def _extract(html: str, field_id: str) -> str:
    m = re.search(r'id="' + field_id + r'"[^>]*value="([^"]*)"', html)
    return m.group(1) if m else ""


@st.cache_data(ttl=120)
def fetch_pricelevel(stock_id: str) -> tuple[pd.DataFrame, str, str]:
    """Return (price-level rows, trading date, error message-or-empty).

    Two-step flow (matches E:\\stock\\scripts\\live_stage1\\bshtm_lib.py exactly,
    this endpoint requires it): POST the query form first (its response is not
    the data itself, just a redirect-shaped stub containing StkNo/RecCount),
    then GET bsContent.aspx with those two values to get the real table HTML.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        r1 = session.get(BASE, timeout=15)
        payload = {
            "__EVENTTARGET": "",
            "__EVENTARGUMENT": "",
            "__LASTFOCUS": "",
            "__VIEWSTATE": _extract(r1.text, "__VIEWSTATE"),
            "__VIEWSTATEGENERATOR": _extract(r1.text, "__VIEWSTATEGENERATOR"),
            "__EVENTVALIDATION": _extract(r1.text, "__EVENTVALIDATION"),
            "RadioButton_Normal": "RadioButton_Normal",
            "TextBox_Stkno": str(stock_id),
            "btnOK": "查詢",
        }
        r2 = session.post(BASE, data=payload, timeout=15)
        m = re.search(r"StkNo=(\w+)&RecCount=(\d+)", r2.text)
        if not m:
            return pd.DataFrame(), "", f"查無資料或格式異常（股票代號 {stock_id} 是否正確？）"
        r3 = session.get(CONTENT_URL, params={"v": "t", "StkNo": m.group(1), "RecCount": m.group(2)}, timeout=20)
        r3.encoding = "utf-8"
        html = r3.text
    except Exception as error:
        return pd.DataFrame(), "", f"連線失敗：{type(error).__name__}: {error}"

    date_m = re.search(r"id='receive_date'>\s*([\d/]+)</td>", html)
    date = date_m.group(1).replace("/", "-") if date_m else ""

    rows = ROW_PAT.findall(html)
    if not rows:
        return pd.DataFrame(), date, f"查無成交列（股票代號 {stock_id} 今日尚無成交，或資料尚未公告）"

    df = pd.DataFrame(rows, columns=["seq", "broker_raw", "price", "buy", "sell"])
    split = df["broker_raw"].str.split(n=1, expand=True)
    df["code"] = split[0]
    df["name"] = split[1].fillna("") if 1 in split.columns else ""
    df["name"] = df["name"].astype(str).str.replace("　", "", regex=False).str.strip()
    df.loc[df["name"] == "None", "name"] = ""
    code2name = df[df["name"] != ""].drop_duplicates("code").set_index("code")["name"].to_dict()
    df["full_name"] = df["code"].map(code2name).fillna(df["code"])
    df["seq"] = df["seq"].astype(int)
    df["price"] = df["price"].str.replace(",", "").astype(float)
    df["buy"] = df["buy"].str.replace(",", "").replace("", "0").astype(int)
    df["sell"] = df["sell"].str.replace(",", "").replace("", "0").astype(int)
    df["net"] = df["buy"] - df["sell"]
    df = df.sort_values("seq").reset_index(drop=True)
    return df[["code", "full_name", "price", "buy", "sell", "net"]], date, ""


st.title("🔍 分點價位查詢")
st.caption(
    "即時查詢 TWSE bshtm 個股當日分點進出，含每個成交價位的買賣量級距。"
    "⚠️ TWSE 公開資料**沒有真正的時間戳記**，只有「該分點在每個成交價位的累計買賣量」——"
    "價位高低只能配合當天已知的走勢粗略推測「較早／較晚」，不是精確的成交時間序列。"
)

col1, col2 = st.columns([1, 2])
with col1:
    stock_id = st.text_input("股票代號", value="", placeholder="例如 2327")
with col2:
    branch_input = st.text_input(
        "分點代號（可留空看全部，多個用逗號分隔）",
        value="",
        placeholder="例如 9227,9600",
    )
    st.caption(
        "常用代號：" + "　".join(f"{code}={label}" for code, label in KNOWN_BRANCHES.items())
    )

if st.button("🔄 查詢 / 重新整理"):
    st.cache_data.clear()

if not stock_id.strip():
    st.info("請輸入股票代號開始查詢。")
    st.stop()

df, date, error = fetch_pricelevel(stock_id.strip())
if error:
    st.error(error)
    st.stop()

st.caption(f"資料日期：{date}")

st.subheader("📌 已知隔日沖訊號分點今日買賣比例")
signal_rows = []
for code in SIGNAL_BRANCH_CODES:
    sub = df[df["code"] == code]
    buy = int(sub["buy"].sum())
    sell = int(sub["sell"].sum())
    net = buy - sell
    ratio = buy / sell if sell > 0 else float("inf") if buy > 0 else float("nan")
    signal_rows.append(
        {
            "分點代號": code,
            "分點名稱": KNOWN_BRANCHES.get(code, ""),
            "買進": buy,
            "賣出": sell,
            "淨額": net,
            "買賣比（買/賣）": ratio,
        }
    )
signal_df = pd.DataFrame(signal_rows)
st.dataframe(
    signal_df.style.format(
        {"買進": "{:,.0f}", "賣出": "{:,.0f}", "淨額": "{:,.0f}", "買賣比（買/賣）": "{:.2f}"}
    ),
    width="stretch",
    hide_index=True,
)
st.caption("買賣比 > 1 代表買多於賣；今日若某分點完全沒有成交紀錄，買進／賣出／淨額會顯示為 0。")

st.divider()

branch_codes = [b.strip() for b in branch_input.split(",") if b.strip()]
if branch_codes:
    filtered = df[df["code"].isin(branch_codes)].copy()
    if filtered.empty:
        st.warning(f"分點代號 {branch_codes} 今日在此股票沒有成交紀錄。")
        st.stop()
else:
    filtered = df.copy()

daily_net = (
    filtered.groupby(["code", "full_name"])["net"]
    .sum()
    .reset_index()
    .sort_values("net", key=lambda s: s.abs(), ascending=False)
)
daily_net.columns = ["分點代號", "分點名稱", "當日淨買賣（股）"]

st.subheader("📊 當日淨買賣（依絕對值排序）")
st.dataframe(
    daily_net.style.format({"當日淨買賣（股）": "{:,.0f}"}),
    width="stretch",
    hide_index=True,
)

st.subheader("📋 逐價位明細")
sort_order = filtered.groupby("code")["net"].sum().abs().sort_values(ascending=False).index
filtered["code"] = pd.Categorical(filtered["code"], categories=sort_order, ordered=True)
filtered = filtered.sort_values(["code", "price"])
detail = filtered.rename(
    columns={"code": "分點代號", "full_name": "分點名稱", "price": "成交價", "buy": "買進", "sell": "賣出", "net": "淨額"}
)
st.dataframe(
    detail.style.format({"成交價": "{:.2f}", "買進": "{:,.0f}", "賣出": "{:,.0f}", "淨額": "{:,.0f}"}),
    width="stretch",
    hide_index=True,
)
