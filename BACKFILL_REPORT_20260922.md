# Broker-flow event backfill — 2026-09-22

## Result

The approved City-GA, Fubon, and Taishin-Taipei event builders were run from
the local data cache.  Their combined append-only populations were enriched
with `scripts/build_branch_event_context.py` and then finalized without
changing the historical byte prefixes.

| Dataset | Before | Added | After | Maximum D0 |
| --- | ---: | ---: | ---: | --- |
| `data/citycenter_ga_events.csv` | 1,653 | 26 | 1,679 | 2026-09-18 |
| `data/fubon_branch_events.csv` | 4,372 | 33 | 4,405 | 2026-09-18 |
| `data/taishin_taipei_events.csv` | 293 | 3 | 296 | 2026-09-16 |

All appended rows have the twelve offline branch-context fields populated;
`top3_available` is `true` for every appended row.

## Build and source notes

- `git pull --no-rebase --no-edit origin master` completed before the build.
  The requested `d688b97` baseline is an ancestor of the synchronized
  `master`; the pull advanced the local checkout to the current remote tip.
- City-GA and Fubon used their approved external builder scripts, with their
  output paths redirected only in process to this repository.  No builder
  source was edited.
- The external Taishin discovery CSVs no longer aligned with the current
  price matrix, and the approved builder correctly rejected them before
  output.  In an operating-system temporary directory, the reviewed discovery
  runner's exact-branch-history and three-day ratio functions regenerated its
  two input CSVs (2,662 ratio rows and 296 strict-lock/liquid rows).  The
  unmodified repository Taishin builder then produced its 296-row full build.
- All inputs came from existing local FinLab and minute-kbar caches.  No
  remote cache refresh, VM action, external discovery overwrite, or website
  asset change was performed.

## Enrichment and append-only verification

`scripts/build_branch_event_context.py` completed for all three combined
full-build datasets:

| Dataset | Enriched rows | V1 D1 matches |
| --- | ---: | ---: |
| City-GA | 1,679 | 1 |
| Fubon | 4,405 | 7 |
| Taishin-Taipei | 296 | 0 |

Full builders can revise historical values as local source panels evolve.  To
enforce immutability, rows were keyed by `(d0, code)`.  Only keys absent from
the pre-build CSV were retained as additions.  Context ran over the combined
population; the exact original bytes were then restored as each final CSV's
historical prefix, followed by only the enriched newly-added CSV rows.

| Dataset | Original SHA-256 / final historical-prefix SHA-256 | Prefix exact |
| --- | --- | --- |
| City-GA | `5a9da4c2bf49811ec91b69a0e10138bab609ac398a6f9c8a56d48f92ad641bcc` | yes |
| Fubon | `a4090c3e50345d4b969294664c7d51f74732d3530d708b30e1401e297001a761` | yes |
| Taishin-Taipei | `a9502587357b9a31ef417d0083d9d3734c7bb0e00fde628a5b183c1241425362` | yes |

Each final CSV retains a UTF-8-sig BOM and has no duplicate `(d0, code)` key.

## Final Git checks

- `git diff --check` passed.
- The normalized Git diff is append-only: City-GA `26/0`, Fubon `33/0`, and
  Taishin-Taipei `3/0` added/deleted lines.
- The intended backfill commit contains only the three event CSVs and this
  report.  The pre-existing untracked 2026-09-09 and 2026-09-16 reports were
  left untouched.
