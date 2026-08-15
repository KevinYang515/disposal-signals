"""Offline-only D0 broker context for branch-event CSV builders.

The FinLab broker cache contains more than one hundred million rows.  This
module deliberately streams its Arrow record batches and retains only exact
``(D0, code)`` rows for the supplied event population.  It never downloads
data or performs per-row lookups.  Streamlit pages must only consume the CSV
columns written by ``scripts/build_branch_event_context.py``; they must never
import this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc


CACHE_DIR = Path(r"C:\Users\USER\finlab_db")
BROKER_FILE = CACHE_DIR / "broker_transactions.feather"
TARGET_V1_EVENTS_FILE = Path(
    r"E:\stock\apps\broker_target_backtest\data\broker_target\events.csv"
)

# Exact labels verified against the source constants before this helper was
# introduced:
# - CITYCENTER_BROKER's pre-normalization literal in
#   E:\stock\scripts\live_stage1\citycenter_ga_stage1.py
# - UNICENTER_BROKER's pre-normalization literal in
#   E:\stock\scripts\live_stage1\unicenter_stage1.py
# - FUBON in E:\stock\scripts\run_build_fubon_branch_events.py
# - BRANCH in scripts\build_taishin_taipei_events.py
BRANCH_LABELS = {
    "city_ga": "凱基-城中",
    "unicenter_city": "統一-城中",
    "fubon": "富邦證券",
    "taishin_taipei": "台新-台北",
}


def _date_days(values) -> np.ndarray:
    """Return calendar-day integers for Arrow/Pandas date-like values."""
    return np.asarray(values).astype("datetime64[D]").astype(np.int64)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _event_lookup(events: pd.DataFrame) -> tuple[dict[int, dict[str, int]], np.ndarray]:
    """Map each exact D0/code pair to one event id and reject ambiguous input."""
    by_day: dict[int, dict[str, int]] = {}
    day_values = _date_days(events["d0"].to_numpy())
    for event_id, (day, code) in enumerate(zip(day_values, events["code"].astype(str))):
        existing = by_day.setdefault(int(day), {}).get(code)
        if existing is not None:
            raise ValueError(f"Duplicate branch-event key: d0={day}, code={code}")
        by_day[int(day)][code] = event_id
    return by_day, day_values


def _load_event_branch_nets(events: pd.DataFrame) -> pd.DataFrame:
    """Stream broker cache and aggregate all branches for exact event D0/code keys."""
    if not BROKER_FILE.exists():
        raise FileNotFoundError(f"Missing local broker cache: {BROKER_FILE}")

    by_day, _ = _event_lookup(events)
    # Do not independently filter all event dates and all event codes: for the
    # Fubon population that broad Cartesian prefilter would materialise many
    # millions of non-event rows.  Batches are date-sorted, so build an exact
    # D0-specific code predicate for the one-to-three dates in each batch.
    codes_by_day = {
        day: pa.array(list(code_map), type=pa.string()) for day, code_map in by_day.items()
    }
    event_parts: list[np.ndarray] = []
    broker_parts: list[np.ndarray] = []
    net_parts: list[np.ndarray] = []
    reader = ipc.open_file(str(BROKER_FILE))

    for batch_number in range(reader.num_record_batches):
        batch = reader.get_batch(batch_number)
        source_dates = batch.column(batch.schema.get_field_index("date"))
        source_codes = batch.column(batch.schema.get_field_index("stock_id"))
        start_day = int(_date_days([source_dates[0].as_py()])[0])
        end_day = int(_date_days([source_dates[-1].as_py()])[0])
        candidate = None
        for day, code_values in codes_by_day.items():
            if day < start_day or day > end_day:
                continue
            day_scalar = pa.scalar(
                pd.Timestamp(np.datetime64(day, "D")).to_pydatetime(), type=pa.timestamp("ns")
            )
            exact_pair = pc.and_(
                pc.equal(source_dates, day_scalar), pc.is_in(source_codes, value_set=code_values)
            )
            candidate = exact_pair if candidate is None else pc.or_(candidate, exact_pair)
        if candidate is None:
            continue
        if not pc.any(candidate).as_py():
            continue

        selected = batch.filter(candidate)
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
        net_parts.append(buy[matched].astype(np.int64) - sell[matched].astype(np.int64))

    if not event_parts:
        raise RuntimeError("No matching D0 broker rows were found in broker_transactions.feather.")

    raw = pd.DataFrame(
        {
            "event_id": np.concatenate(event_parts),
            "broker": np.concatenate(broker_parts),
            "net_lots": np.concatenate(net_parts),
        }
    )
    return raw.groupby(["event_id", "broker"], as_index=False, sort=False)["net_lots"].sum()


def _turnover_file() -> Path:
    """Return the local FinLab ``price:成交金額`` cache without a live data call."""
    filename = "price#" + "".join(map(chr, [25104, 20132, 37329, 38989])) + ".feather"
    path = CACHE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing local D0 turnover cache: {path}")
    return path


def _load_d0_turnover(events: pd.DataFrame) -> np.ndarray:
    """Read only the event codes from local ``price:成交金額`` and align by name."""
    path = _turnover_file()
    schema = ipc.open_file(str(path)).schema
    available = set(schema.names)
    event_codes = events["code"].astype(str)
    selected_codes = [code for code in event_codes.unique() if code in available]
    if not selected_codes:
        return np.full(len(events), np.nan, dtype=float)

    turnover = pd.read_feather(path, columns=["date", *selected_codes])
    turnover["date"] = pd.to_datetime(turnover["date"], errors="coerce")
    turnover = turnover.set_index("date")
    row_positions = turnover.index.get_indexer(pd.DatetimeIndex(events["d0"]))
    column_positions = turnover.columns.get_indexer(event_codes)
    values = turnover.to_numpy(dtype=float, copy=False)
    out = np.full(len(events), np.nan, dtype=float)
    valid = (row_positions >= 0) & (column_positions >= 0)
    out[valid] = values[row_positions[valid], column_positions[valid]]
    return out


def _top3_text(branch_nets: pd.DataFrame) -> pd.Series:
    """Format the positive D0 top-three buyers per event in deterministic order."""
    positive = branch_nets.loc[branch_nets["net_amt_wan"] > 0].copy()
    if positive.empty:
        return pd.Series(dtype=object)
    positive.sort_values(
        ["event_id", "net_amt_wan", "broker"], ascending=[True, False, True], inplace=True
    )
    top3 = positive.groupby("event_id", sort=False).head(3).copy()
    top3["text"] = top3.apply(
        lambda row: f"{row['broker']}({row['net_amt_wan']:,.2f}萬)", axis=1
    )
    return top3.groupby("event_id", sort=False)["text"].agg("／".join)


def _load_v1_candidate_keys() -> pd.MultiIndex:
    """Load the established target-price V1 candidate set for D1-exact matching."""
    if not TARGET_V1_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Missing target-price event file: {TARGET_V1_EVENTS_FILE}")
    targets = pd.read_csv(
        TARGET_V1_EVENTS_FILE,
        usecols=["event_date", "ticker", "v1_candidate"],
        dtype={"ticker": str},
    )
    targets["event_date"] = pd.to_datetime(targets["event_date"], errors="coerce").dt.normalize()
    targets["ticker"] = targets["ticker"].astype(str).str.strip().str.zfill(4)
    candidates = targets.loc[
        _as_bool(targets["v1_candidate"]) & targets["event_date"].notna(),
        ["event_date", "ticker"],
    ].drop_duplicates()
    return pd.MultiIndex.from_frame(candidates)


def enrich_branch_events(events: pd.DataFrame) -> pd.DataFrame:
    """Attach D0 branch context and D1-exact target-price V1 overlap to events.

    ``net_amt_wan`` follows the existing branch-builder unit convention:
    net lots * D0 close / 10.  ``influence_pct`` then reuses the exact existing
    formula: net_amt_wan * 10,000 / D0 turnover * 100.
    """
    required = {"d0", "d1", "code", "d0_close"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Branch-event context is missing required columns: {sorted(missing)}")

    result = events.copy()
    result["d0"] = pd.to_datetime(result["d0"], errors="coerce").dt.normalize()
    result["d1"] = pd.to_datetime(result["d1"], errors="coerce").dt.normalize()
    result["code"] = result["code"].astype(str).str.strip().str.zfill(4)
    if result[["d0", "d1", "code"]].isna().any().any():
        raise ValueError("Branch-event context received invalid D0/D1/code values.")
    if result.duplicated(["d0", "code"]).any():
        raise ValueError("Branch-event context requires one row per D0/code event.")

    event_keys = result[["d0", "code", "d0_close"]].reset_index(drop=True)
    branch_nets = _load_event_branch_nets(event_keys)
    d0_turnover = _load_d0_turnover(event_keys)
    branch_nets["net_amt_wan"] = (
        branch_nets["net_lots"].to_numpy(dtype=float)
        * event_keys.loc[branch_nets["event_id"], "d0_close"].to_numpy(dtype=float)
        / 10.0
    )
    turnover_for_row = d0_turnover[branch_nets["event_id"].to_numpy(dtype=int)]
    with np.errstate(divide="ignore", invalid="ignore"):
        branch_nets["influence_pct"] = branch_nets["net_amt_wan"] * 10_000.0 / turnover_for_row * 100.0
    branch_nets.loc[~np.isfinite(branch_nets["influence_pct"]), "influence_pct"] = np.nan

    context = pd.DataFrame({"event_id": np.arange(len(result), dtype=np.int32)})
    for key, label in BRANCH_LABELS.items():
        branch = branch_nets.loc[branch_nets["broker"].eq(label), ["event_id", "net_amt_wan", "influence_pct"]]
        branch = branch.drop_duplicates("event_id")
        context = context.merge(branch, on="event_id", how="left")
        context.rename(
            columns={
                "net_amt_wan": f"{key}_net_amt_wan",
                "influence_pct": f"{key}_influence_pct",
            },
            inplace=True,
        )
        # A missing raw row is no branch activity.  Keep a real but uncomputable
        # influence value as blank instead of pretending it is zero.
        context[f"{key}_net_amt_wan"] = context[f"{key}_net_amt_wan"].fillna(0.0)
        no_raw_branch_row = context[f"{key}_net_amt_wan"].eq(0.0) & context[f"{key}_influence_pct"].isna()
        context.loc[no_raw_branch_row, f"{key}_influence_pct"] = 0.0

    top3 = _top3_text(branch_nets)
    context["d0_top3_net_buy_branches"] = context["event_id"].map(top3).fillna("")
    result = result.reset_index(drop=True).join(context.drop(columns="event_id"))

    result["top3_available"] = result["d0_top3_net_buy_branches"].ne("")

    # V1 candidate overlap uses the predecessor report's confirmed D1-exact
    # anchor convention, rather than D0 or a retrospective window.
    candidate_keys = _load_v1_candidate_keys()
    d1_keys = pd.MultiIndex.from_arrays([result["d1"], result["code"]])
    result["v1_candidate_d1"] = d1_keys.isin(candidate_keys)
    return result


def add_other_branch_activity_count(events: pd.DataFrame, own_branch: str) -> pd.DataFrame:
    """Add a QA-friendly count of other known branches with non-zero D0 net activity."""
    if own_branch not in BRANCH_LABELS:
        raise ValueError(f"Unknown own branch: {own_branch}")
    other_amount_columns = [
        f"{key}_net_amt_wan" for key in BRANCH_LABELS if key != own_branch
    ]
    result = events.copy()
    result["other_branch_active_count"] = (
        result[other_amount_columns].abs().gt(0.0).sum(axis=1).astype(int)
    )
    return result
