"""Build the data-backed 09:15 add-on inputs for the City-GA research page.

The public Streamlit app deliberately reads only the small CSV produced here.
This script is the reproducible local-only join that derives its fields from:

* ``data/citycenter_ga_events.csv``: the fixed City-GA event universe;
* ``C:\\Users\\USER\\finlab_db\\broker_transactions.feather``: D0 branch nets;
* ``E:\\stock\\data\\minute_kbars\\kbars_raw.csv``: real D1 09:15 minute closes.

For each event, the three bonus flags are: KGI-City is one of the three largest
positive D0 buyers, Uni-President Chengzhong has a positive D0 net, and Fubon
Chiayi has a positive D0 net.  The add-on leg is eligible only when at least
one flag is true and the exact D1 09:15 close is below D1's daily open.

No market data is downloaded or mutated.  Run this script from this repository
with the E:\\stock virtualenv when either upstream local cache is refreshed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS_CSV = REPO_ROOT / "data" / "citycenter_ga_events.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "citycenter_ga_0915_addon_events.csv"
BROKER_FEATHER = Path(r"C:\Users\USER\finlab_db\broker_transactions.feather")
MINUTE_KBARS_CSV = Path(r"E:\stock\data\minute_kbars\kbars_raw.csv")

KGI_CITY = "凱基-城中"
UNICENTER = "統一-城中"
FUBON_CHIAYI = "富邦-嘉義"


def _date_days(values) -> np.ndarray:
    """Return NumPy day integers for an Arrow/Pandas date-like array."""
    return np.asarray(values).astype("datetime64[D]").astype(np.int64)


def _event_lookup(events: pd.DataFrame, day_col: str) -> tuple[dict[int, dict[str, int]], np.ndarray]:
    """Map (calendar day, code) to the unique City-GA event row id."""
    by_day: dict[int, dict[str, int]] = {}
    day_values = _date_days(events[day_col].to_numpy())
    for event_id, (day, code) in enumerate(zip(day_values, events["code"].astype(str))):
        existing = by_day.setdefault(int(day), {}).get(code)
        if existing is not None:
            raise ValueError(f"Duplicate City-GA event key: {day_col}={day}, code={code}")
        by_day[int(day)][code] = event_id
    return by_day, day_values


def load_d0_branch_nets(events: pd.DataFrame, broker_path: Path) -> pd.DataFrame:
    """Stream the 100M+ row cache and retain only D0 rows for City-GA events."""
    by_day, _ = _event_lookup(events, "d0")
    wanted_dates = pa.array(events["d0"].dt.to_pydatetime(), type=pa.timestamp("ns"))
    wanted_codes = pa.array(events["code"].astype(str).unique(), type=pa.string())
    event_parts: list[np.ndarray] = []
    broker_parts: list[np.ndarray] = []
    net_parts: list[np.ndarray] = []
    reader = ipc.open_file(str(broker_path))

    for batch_number in range(reader.num_record_batches):
        batch = reader.get_batch(batch_number)
        # Filter in Arrow before converting dictionary strings to Python.  The
        # complete source contains 100M+ rows but only ~1.5k event pairs.
        source_dates = batch.column(batch.schema.get_field_index("date"))
        source_codes = batch.column(batch.schema.get_field_index("stock_id"))
        target_pair_candidate = pc.and_(
            pc.is_in(source_dates, value_set=wanted_dates),
            pc.is_in(source_codes, value_set=wanted_codes),
        )
        if not pc.any(target_pair_candidate).as_py():
            continue
        selected = batch.filter(target_pair_candidate)
        day_values = _date_days(
            selected.column(selected.schema.get_field_index("date")).to_numpy(zero_copy_only=False)
        )
        codes = np.asarray(
            selected.column(selected.schema.get_field_index("stock_id")).to_pylist(), dtype=str
        )
        event_ids = np.fromiter(
            (by_day[int(day)].get(code, -1) for day, code in zip(day_values, codes)),
            dtype=np.int64,
            count=len(selected),
        )
        matched = event_ids >= 0
        if not matched.any():
            continue
        event_parts.append(event_ids[matched].astype(np.int32))
        broker_parts.append(
            np.asarray(
                selected.column(selected.schema.get_field_index("broker")).to_pylist(), dtype=str
            )[matched]
        )
        buy = selected.column(selected.schema.get_field_index("buy")).to_numpy(zero_copy_only=False)
        sell = selected.column(selected.schema.get_field_index("sell")).to_numpy(zero_copy_only=False)
        net_parts.append((buy[matched].astype(np.int64) - sell[matched].astype(np.int64)))

        if (batch_number + 1) % 100 == 0:
            print(f"Scanned {batch_number + 1}/{reader.num_record_batches} broker batches")

    if not event_parts:
        raise RuntimeError("No City-GA event rows were found in broker_transactions.feather.")

    raw = pd.DataFrame(
        {
            "event_id": np.concatenate(event_parts),
            "broker": np.concatenate(broker_parts),
            "net_lots": np.concatenate(net_parts),
        }
    )
    return raw.groupby(["event_id", "broker"], as_index=False, sort=False)["net_lots"].sum()


def derive_bonus_flags(events: pd.DataFrame, branch_nets: pd.DataFrame) -> pd.DataFrame:
    """Derive all three D0 bonus flags from exact aggregated branch nets."""
    positives = branch_nets[branch_nets["net_lots"] > 0].copy()
    positives.sort_values(["event_id", "net_lots", "broker"], ascending=[True, False, True], inplace=True)
    positives["positive_buy_rank"] = positives.groupby("event_id").cumcount() + 1

    rank_top3 = (
        positives.loc[
            (positives["broker"] == KGI_CITY) & (positives["positive_buy_rank"] <= 3), "event_id"
        ]
        .drop_duplicates()
    )
    uni_cobuy = positives.loc[positives["broker"] == UNICENTER, "event_id"].drop_duplicates()
    fubon_cobuy = positives.loc[positives["broker"] == FUBON_CHIAYI, "event_id"].drop_duplicates()

    result = pd.DataFrame({"event_id": np.arange(len(events), dtype=np.int32)})
    result["city_d0_buy_rank_top3"] = result["event_id"].isin(rank_top3)
    result["unicenter_city_d0_cobuy"] = result["event_id"].isin(uni_cobuy)
    result["fubon_chiayi_d0_cobuy"] = result["event_id"].isin(fubon_cobuy)
    result["bonus_signal_count"] = result[
        ["city_d0_buy_rank_top3", "unicenter_city_d0_cobuy", "fubon_chiayi_d0_cobuy"]
    ].sum(axis=1)

    # Every City-GA event must itself remain a positive KGI-City net after aggregation.
    city_positive = set(positives.loc[positives["broker"] == KGI_CITY, "event_id"])
    missing_city = sorted(set(result.event_id) - city_positive)
    if missing_city:
        raise RuntimeError(
            f"{len(missing_city)} City-GA events lack positive {KGI_CITY} D0 net in the source cache; "
            "do not publish a mixed-source join."
        )
    return result


def load_d1_0915_closes(events: pd.DataFrame, minute_path: Path) -> pd.Series:
    """Read only exact 09:15 bars for the event's D1 code/date keys."""
    by_day, _ = _event_lookup(events, "d1")
    target_keys = {
        f"{pd.Timestamp(events.iloc[event_id].d1).date().isoformat()}\x1f{code}": event_id
        for day in by_day.values()
        for code, event_id in day.items()
    }
    close = pd.Series(np.nan, index=np.arange(len(events)), dtype=float)
    for chunk in pd.read_csv(
        minute_path,
        usecols=["code", "date", "ts", "Close"],
        dtype={"code": str, "date": str, "ts": str, "Close": float},
        chunksize=500_000,
    ):
        at_0915 = chunk["ts"].str.endswith("09:15:00", na=False)
        if not at_0915.any():
            continue
        minute = chunk.loc[at_0915].copy()
        keys = minute["date"] + "\x1f" + minute["code"]
        event_ids = keys.map(target_keys)
        matched = event_ids.notna()
        if matched.any():
            ids = event_ids.loc[matched].astype(int)
            if ids.duplicated().any():
                raise RuntimeError("Minute cache has duplicate exact 09:15 bars for a City-GA event.")
            close.loc[ids.to_numpy()] = minute.loc[matched, "Close"].to_numpy(dtype=float)
    return close


def build(events_path: Path, broker_path: Path, minute_path: Path, output_path: Path) -> pd.DataFrame:
    if not broker_path.exists() or not minute_path.exists():
        raise FileNotFoundError("Both local broker_transactions.feather and kbars_raw.csv are required.")
    events = pd.read_csv(events_path, parse_dates=["d0", "d1"], dtype={"code": str})
    branch_nets = load_d0_branch_nets(events, broker_path)
    addon = derive_bonus_flags(events, branch_nets)
    addon["d1_0915_close"] = load_d1_0915_closes(events, minute_path)
    addon["d1_open"] = events["d1_open"].to_numpy(dtype=float)
    addon["d1_close"] = events["d1_close"].to_numpy(dtype=float)
    addon["has_d1_0915_close"] = addon["d1_0915_close"].notna()
    addon["addon_eligible"] = (
        (addon["bonus_signal_count"] >= 1)
        & addon["has_d1_0915_close"]
        & (addon["d1_0915_close"] < addon["d1_open"])
    )
    addon["addon_ret_0915_to_close_pct"] = np.where(
        addon["has_d1_0915_close"],
        (addon["d1_0915_close"] - addon["d1_close"]) / addon["d1_0915_close"] * 100.0,
        np.nan,
    )
    addon.insert(0, "d1", events["d1"].dt.strftime("%Y-%m-%d"))
    addon.insert(0, "code", events["code"].astype(str))
    addon.insert(0, "d0", events["d0"].dt.strftime("%Y-%m-%d"))
    addon.drop(columns=["event_id", "d1_open", "d1_close"], inplace=True)

    if addon.duplicated(["d0", "code", "d1"]).any():
        raise RuntimeError("Add-on output key is not unique.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    addon.to_csv(output_path, index=False, encoding="utf-8")
    coverage = addon["has_d1_0915_close"].mean() * 100
    print(f"Wrote {len(addon)} rows to {output_path}")
    print(f"Exact D1 09:15 coverage: {coverage:.2f}% ({int(addon.has_d1_0915_close.sum())}/{len(addon)})")
    print(f"Eligible add-on events: {int(addon.addon_eligible.sum())}")
    return addon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=EVENTS_CSV)
    parser.add_argument("--broker", type=Path, default=BROKER_FEATHER)
    parser.add_argument("--minute", type=Path, default=MINUTE_KBARS_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    build(args.events, args.broker, args.minute, args.output)


if __name__ == "__main__":
    main()
