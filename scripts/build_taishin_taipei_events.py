"""Build the strict-lock/liquid 台新-台北 event snapshot for the Streamlit site.

This is deliberately a monitoring/reference dataset, not a promoted strategy.
The D0 population is the exact-identifier ``台新-台北`` strict-lock/liquid
subset emitted by the 2026-08-13 discovery run.  It excludes the bare labels
``台新`` and ``台新證券`` by equality filtering rather than name matching.

OHLC, disposal, market-cap, TAIEX momentum, and real minute-kbar enrichment
reuse the reviewed City-GA builder helpers.  The D1 day-trade short-suspension
flag reuses the reviewed borrow-availability audit helper.  The output schema
matches data/fubon_branch_events.csv, the other branch-level event snapshot.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
STOCK_ROOT = Path(r"E:\stock")
CACHE_DIR = Path(r"C:\Users\USER\finlab_db")
DISCOVERY_DIR = STOCK_ROOT / "outputs" / "new_branch_discovery_20260813"
SOURCE_STRICT = DISCOVERY_DIR / "d0_strict_lock_large_buy_events.csv"
SOURCE_RATIOS = DISCOVERY_DIR / "taishin_taipei_all_bigbuy_ratios.csv"
OUT_CSV = REPO / "data" / "taishin_taipei_events.csv"
MINUTE_KBAR_CSV = STOCK_ROOT / "data" / "minute_kbars" / "kbars_raw.csv"
BRANCH = "台新-台北"

# This is exactly the branch-level schema used by fubon_branch_events.csv.
FIELDS = [
    "d0", "code", "market", "d1", "net_amt_wan", "influence_pct", "lock_streak",
    "d0_locked", "d1_frozen", "censored", "short_ret_open_to_close_pct", "short_mae_pct",
    "success", "gap_pct", "d0_close", "d1_open", "d1_high", "d1_low", "d1_close",
    "exit_price", "exit_date", "exit_kind", "year", "d1_intraday_close", "has_intraday",
    "d0_disposal", "d1_disposal", "d1_disposal_type", "mktcap_billion",
    "taiex_2day_mom_pct", "day_trade_short_suspended_d1",
]


def load_module(name: str, path: Path) -> ModuleType:
    """Load a reviewed local helper script without copying its data joins."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def source_events() -> pd.DataFrame:
    """Return only the supplied exact-name strict-lock/liquid D0 population."""
    strict = pd.read_csv(SOURCE_STRICT, dtype={"stock": str})
    ratios = pd.read_csv(SOURCE_RATIOS, dtype={"stock": str})

    strict = strict.loc[strict["broker"].eq(BRANCH)].copy()
    ratios = ratios.loc[ratios["broker"].eq(BRANCH)].copy()
    if strict.empty or ratios.empty:
        raise RuntimeError(f"Exact branch {BRANCH!r} was not found in both discovery inputs.")
    if not strict["broker"].eq(BRANCH).all() or not ratios["broker"].eq(BRANCH).all():
        raise RuntimeError("Branch identity check failed; do not merge similarly named brokers.")

    events = ratios.loc[as_true(ratios["strict_lock_liquid"])].copy()
    events = events.drop_duplicates(["d0_date", "stock"])
    strict_keys = set(zip(strict["d0_date"], strict["stock"]))
    event_keys = set(zip(events["d0_date"], events["stock"]))
    if not event_keys.issubset(strict_keys):
        raise RuntimeError("A strict-lock/liquid event is absent from the strict-lock source audit.")
    if len(events) != 250:
        raise RuntimeError(f"Expected the discovery run's ~250 strict-lock/liquid events; got {len(events)}.")
    return events.sort_values(["d0_date", "stock"]).reset_index(drop=True)


def main() -> None:
    if str(STOCK_ROOT) not in sys.path:
        sys.path.insert(0, str(STOCK_ROOT))
    from lib.broker_flow_arrow import date_text, load_minute_kbar_index, strict_tick_limit_up_mask

    city_builder = load_module(
        "citycenter_ga_events_builder",
        STOCK_ROOT / "scripts" / "run_build_citycenter_ga_events_v2.py",
    )
    discovery_builder = load_module(
        "new_branch_discovery_builder",
        STOCK_ROOT / "scripts" / "run_new_branch_discovery_20260813.py",
    )
    borrow_audit = load_module(
        "borrow_availability_risk",
        STOCK_ROOT / "scripts" / "run_borrow_availability_risk_20260812.py",
    )

    source = source_events()
    # The discovery run used this exact named-axis alignment because the local
    # turnover Feather had been refreshed on a different schedule to OHLC.
    # It joins only common (date, stock) cells and never positional-aligns data.
    prices = discovery_builder.load_price_data_aligned(CACHE_DIR)
    strict_locked = strict_tick_limit_up_mask(prices)
    streak = city_builder.lock_streak_matrix(strict_locked)
    mktcap = city_builder.load_market_value(prices)
    taiex_mom = city_builder.load_taiex_2day_mom()
    disposal_intervals = city_builder.load_disposal_intervals()
    market_type = city_builder.load_market_type()
    kbar_index = load_minute_kbar_index(MINUTE_KBAR_CSV) if MINUTE_KBAR_CSV.exists() else {}

    print(f"Exact {BRANCH} strict-lock/liquid source events: {len(source)}")
    print(f"Minute-kbar index: {len(kbar_index)} (code, date) pairs available")

    rows: list[dict] = []
    for event in source.itertuples(index=False):
        day0 = int(event.day_index)
        stock_col = int(event.stock_index)
        code = str(event.stock).zfill(4)
        d0_date = int(prices.dates[day0])
        if date_text(d0_date) != str(event.d0_date) or str(prices.stocks[stock_col]) != code:
            raise RuntimeError(f"Discovery event no longer aligns to price matrix: {event.d0_date} {event.stock}")
        if day0 >= prices.close.shape[0] - 1:
            raise RuntimeError(f"No D1 is available for source event {event.d0_date} {code}")
        if not strict_locked[day0, stock_col]:
            raise RuntimeError(f"Source event is not a strict D0 lock after price join: {event.d0_date} {code}")

        day1 = day0 + 1
        d1_date = int(prices.dates[day1])
        d0_close = float(prices.close[day0, stock_col])
        d1_open = float(prices.open[day1, stock_col])
        d1_high = float(prices.high[day1, stock_col])
        d1_low = float(prices.low[day1, stock_col])
        d1_close = float(prices.close[day1, stock_col])
        if not (
            np.isfinite(d0_close) and np.isfinite(d1_open) and np.isfinite(d1_high)
            and np.isfinite(d1_low) and np.isfinite(d1_close) and d0_close > 0 and d1_open > 0
        ):
            raise RuntimeError(f"Invalid D1 OHLC for source event {event.d0_date} {code}")

        turnover = float(prices.turnover[day0, stock_col])
        if not np.isfinite(turnover) or turnover <= 0:
            raise RuntimeError(f"Invalid D0 turnover for source event {event.d0_date} {code}")
        net_amt_twd = float(event.net_amt_twd)
        d0_disp, _ = city_builder.disposal_lookup(disposal_intervals, code, d0_date)
        d1_disp, d1_minutes = city_builder.disposal_lookup(disposal_intervals, code, d1_date)
        d1_text = date_text(d1_date)
        minute_closes = kbar_index.get((code, d1_text))
        frozen = (
            np.isclose(d1_open, d1_high, atol=0.0001)
            and np.isclose(d1_high, d1_low, atol=0.0001)
            and np.isclose(d1_low, d1_close, atol=0.0001)
        )
        short_ret = (d1_open - d1_close) / d1_open * 100.0
        rows.append({
            "d0": date_text(d0_date),
            "code": code,
            "market": market_type.get(code, "TSE"),
            "d1": d1_text,
            "net_amt_wan": net_amt_twd / 10_000.0,
            "influence_pct": net_amt_twd / turnover * 100.0,
            "lock_streak": int(streak[day0, stock_col]),
            "d0_locked": True,
            "d1_frozen": bool(frozen),
            "censored": bool(frozen),
            "short_ret_open_to_close_pct": short_ret,
            "short_mae_pct": (d1_high - d1_open) / d1_open * 100.0,
            "success": bool(short_ret > 0),
            "gap_pct": (d1_open - d0_close) / d0_close * 100.0,
            "d0_close": d0_close,
            "d1_open": d1_open,
            "d1_high": d1_high,
            "d1_low": d1_low,
            "d1_close": d1_close,
            "exit_price": d1_close,
            "exit_date": d1_text,
            "exit_kind": "d1_close",
            "year": int(pd.Timestamp(str(date_text(d0_date))).year),
            "d1_intraday_close": json.dumps(minute_closes) if minute_closes else "",
            "has_intraday": bool(minute_closes),
            "d0_disposal": bool(d0_disp),
            "d1_disposal": bool(d1_disp),
            "d1_disposal_type": (
                f"{int(d1_minutes)}分鐘"
                if d1_disp and d1_minutes is not None and np.isfinite(d1_minutes)
                else ""
            ),
            "mktcap_billion": (
                None if not np.isfinite(mktcap[day0, stock_col]) else float(mktcap[day0, stock_col])
            ),
            "taiex_2day_mom_pct": taiex_mom.get(d0_date),
        })

    events = pd.DataFrame(rows)
    if len(events) != len(source) or events.duplicated(["d0", "code"]).any():
        raise RuntimeError("The output must be a one-to-one rebuild of the 250 D0 events.")
    events["d0"] = pd.to_datetime(events["d0"])
    events["d1"] = pd.to_datetime(events["d1"])
    # Reuse the audited D1 interval matching; only the schema's D1 flag is emitted.
    borrow_audit.suspension_flags(events, CACHE_DIR / "day_trade_short_suspension.feather")
    events["day_trade_short_suspended_d1"] = events["day_trade_short_suspended_d1"].fillna(False).astype(bool)

    # These are informational flags in the CSV.  Page 17 applies both D1 flags as hard, non-toggleable gates.
    before = len(events)
    after_disposal = int((~events["d1_disposal"]).sum())
    after_both = int((~events["d1_disposal"] & ~events["day_trade_short_suspended_d1"]).sum())
    settled_before = int((~events["censored"]).sum())
    settled_after_both = int((~events["censored"] & ~events["d1_disposal"] & ~events["day_trade_short_suspended_d1"]).sum())
    print(f"Hard-exclusion audit (all rows): {before} -> {after_disposal} after D1 disposal -> {after_both} after D1 suspension")
    print(f"Hard-exclusion audit (settled rows): {settled_before} -> {settled_after_both}")

    events["d0"] = events["d0"].dt.strftime("%Y-%m-%d")
    events["d1"] = events["d1"].dt.strftime("%Y-%m-%d")
    events = events.sort_values(["d0", "code"])
    if list(events.columns.intersection(FIELDS)) != FIELDS:
        raise RuntimeError("Output schema drifted from the branch event schema.")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(events[FIELDS].to_dict("records"))
    print(f"Wrote {len(events)} rows to {OUT_CSV}")
    print(f"D0 range: {events['d0'].min()} .. {events['d0'].max()}")


if __name__ == "__main__":
    main()
