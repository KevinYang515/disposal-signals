"""Write offline D0 cross-branch context into the committed event CSVs.

Run this after regenerating any of the three branch-event CSVs.  It is an
offline build step: ``branch_event_context`` reads Kevin's local FinLab cache
and target-price source, while deployed Streamlit pages read only the resulting
committed CSV columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from branch_event_context import add_other_branch_activity_count, enrich_branch_events  # noqa: E402


EVENT_BUILDS = {
    "city_ga": REPO / "data" / "citycenter_ga_events.csv",
    "fubon": REPO / "data" / "fubon_branch_events.csv",
    "taishin_taipei": REPO / "data" / "taishin_taipei_events.csv",
}
MARKET_VALUE_FILE = Path(r"C:\Users\USER\finlab_db\etl#market_value.feather")
CONTEXT_COLUMNS = [
    *(f"{branch}_{measure}" for branch in ("city_ga", "unicenter_city", "fubon", "taishin_taipei")
      for measure in ("net_amt_wan", "influence_pct")),
    "d0_top3_net_buy_branches", "top3_available", "v1_candidate_d1", "other_branch_active_count",
]


def add_city_mktcap_percentiles(events: pd.DataFrame) -> pd.DataFrame:
    """Restore City-GA's existing point-in-time market-cap percentile field.

    This is the same inclusive empirical-CDF convention as the historical
    ``rebuild_citycenter_mktcap_percentiles_20260812.py`` helper: each event
    is ranked only against finite, positive market values on its own D0, with
    every historical panel column retained in the comparison universe.
    """
    if not MARKET_VALUE_FILE.exists():
        raise FileNotFoundError(f"Missing local market-value cache: {MARKET_VALUE_FILE}")

    table = ipc.open_file(MARKET_VALUE_FILE).read_all()
    dates = (
        table.column("date").combine_chunks().to_numpy(zero_copy_only=False)
        .astype("datetime64[D]").astype(np.int64)
    )
    codes = list(table.schema.names[1:])
    values = np.empty((len(dates), len(codes)), dtype=np.float64)
    for position, code in enumerate(codes):
        values[:, position] = table.column(code).combine_chunks().to_numpy(zero_copy_only=False)

    result = events.copy()
    day_index = {int(day): position for position, day in enumerate(dates)}
    code_index = {code: position for position, code in enumerate(codes)}
    percentiles = np.full(len(result), np.nan, dtype=float)
    event_days = result["d0"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    for event_position, (day, code) in enumerate(zip(event_days, result["code"])):
        day_position = day_index.get(int(day))
        code_position = code_index.get(str(code))
        if day_position is None or code_position is None:
            continue
        daily = values[day_position]
        universe = np.sort(daily[np.isfinite(daily) & (daily > 0)])
        event_value = daily[code_position]
        if len(universe) and np.isfinite(event_value) and event_value > 0:
            percentiles[event_position] = np.searchsorted(universe, event_value, side="right") / len(universe) * 100.0
    result["mktcap_pct_rank"] = percentiles
    print(
        f"Restored City-GA mktcap_pct_rank for {int(np.isfinite(percentiles).sum())}/{len(result)} events"
    )
    return result


def enrich_csv(path: Path, own_branch: str) -> int:
    """Atomically replace one event CSV with its offline context columns."""
    if not path.exists():
        raise FileNotFoundError(f"Event CSV does not exist: {path}")

    events = pd.read_csv(path, parse_dates=["d0", "d1"], dtype={"code": str})
    events["code"] = events["code"].str.zfill(4)
    if events.duplicated(["d0", "code"]).any():
        raise ValueError(f"Duplicate D0/code event keys in {path}")
    events = events.drop(columns=CONTEXT_COLUMNS, errors="ignore")

    if own_branch == "city_ga":
        events = add_city_mktcap_percentiles(events)
    enriched = add_other_branch_activity_count(enrich_branch_events(events), own_branch)
    enriched["d0"] = pd.to_datetime(enriched["d0"]).dt.strftime("%Y-%m-%d")
    enriched["d1"] = pd.to_datetime(enriched["d1"]).dt.strftime("%Y-%m-%d")

    temporary = path.with_suffix(path.suffix + ".tmp")
    enriched.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    print(
        f"Wrote {len(enriched)} rows with offline branch context to {path} "
        f"(V1 D1 matches: {int(enriched['v1_candidate_d1'].sum())})"
    )
    return len(enriched)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch",
        choices=[*EVENT_BUILDS, "all"],
        default="all",
        help="CSV to enrich (default: all three)",
    )
    args = parser.parse_args()
    builds = EVENT_BUILDS.items() if args.branch == "all" else [(args.branch, EVENT_BUILDS[args.branch])]
    for own_branch, path in builds:
        enrich_csv(path, own_branch)


if __name__ == "__main__":
    main()
