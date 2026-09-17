# Broker-flow event backfill — 2026-09-17

## Result

The approved City-GA, Fubon, and Taishin-Taipei event builders were run from
the local cache.  Their outputs were enriched with
`scripts/build_branch_event_context.py` and finalized as strict append-only
CSV files.

| Dataset | Before | Added | After | Maximum D0 |
| --- | ---: | ---: | ---: | --- |
| `data/citycenter_ga_events.csv` | 1,653 | 0 | 1,653 | 2026-09-14 |
| `data/fubon_branch_events.csv` | 4,372 | 0 | 4,372 | 2026-09-14 |
| `data/taishin_taipei_events.csv` | 292 | 1 | 293 | 2026-09-10 |

The single appended event is Taishin-Taipei `6173`, D0=2026-09-10 and
D1=2026-09-11.  No City-GA or Fubon key absent from the committed CSV was
available in this local build.

## Build and source notes

- `git pull --no-rebase --no-edit origin master` completed before the build;
  the local `master` was already current and includes the requested
  `8c4ed5d` event-data baseline.
- All inputs came from the local FinLab cache and local minute-kbar cache. No
  remote refresh, VM command, VM file, or scheduled task was used.
- City-GA and Fubon have a non-identical turnover-panel axis in the local
  cache. Their approved price reader was replaced in memory by the reviewed
  key-based alignment helper: only common `(date, stock)` cells were joined;
  no positional alignment, fill, or cache/source edit was performed.
- The external Taishin discovery files were stale.  In a temporary workspace,
  the approved discovery runner's exact-branch history and three-trading-day
  ratio functions rebuilt the two inputs consumed by the approved Taishin
  builder.  This yielded 11,457 exact branch date/stock rows, 2,619 large-buy
  rows, and 293 strict-lock/liquid rows.  The temporary workspace was removed
  after validation.

## Append-only and enrichment verification

Full rebuilds can revise prior numeric fields as upstream panels change.  To
preserve immutability, rows were keyed by `(d0, code)` and only keys absent
from the original CSV were retained.  Context enrichment ran over the
combined data; the exact pre-build bytes of each historical prefix were then
restored before the enriched new rows were appended.

| Dataset | Original SHA-256 / final historical-prefix SHA-256 | Prefix exact |
| --- | --- | --- |
| City-GA | `5a9da4c2bf49811ec91b69a0e10138bab609ac398a6f9c8a56d48f92ad641bcc` | yes |
| Fubon | `a4090c3e50345d4b969294664c7d51f74732d3530d708b30e1401e297001a761` | yes |
| Taishin-Taipei | `346c50af44b0c208f42479e9712efb05b8d674ccef0277b1b313d4eb42058fa8` | yes |

All final CSVs retain a UTF-8-sig BOM and have no duplicate `(d0, code)`
keys.  All twelve branch-context fields are populated for the one added row,
and its `top3_available` value is `true`.

## Final Git checks

- `git diff --check` passed.
- The normalized Git diff is append-only: Taishin-Taipei `1/0`
  added/deleted lines; City-GA and Fubon have no diff.
- Only `data/taishin_taipei_events.csv` and this report are intended for the
  backfill commit.  Earlier untracked backfill reports were left untouched.
