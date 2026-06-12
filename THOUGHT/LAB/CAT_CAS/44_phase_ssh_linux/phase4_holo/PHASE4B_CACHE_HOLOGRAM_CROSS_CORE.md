# PHASE4B_CACHE_HOLOGRAM_CROSS_CORE

## Verdict

`PHASE4B_CROSS_CORE_PARTIAL_BOUNDARY`

## Objective

Push the Phase 4B cache `.holo` witness from same-core post-restore timing into a cross-core observer/prober split.

Question:

```text
If one core runs the .holo schedule and another core probes the restored tape, does the physical readout still classify the mode?
```

## Harness

```text
session_scripts/phase4_holo/cache_hologram_cross_core.c
```

Protocol:

- writer pinned to core 0,
- observer pinned to core 1,
- writer initializes, flushes, runs reversible `.holo` schedule,
- writer restores logical hash,
- observer probes all 64 lines,
- analyzer uses the matched-null gates from `analyze_cache_hologram_matched_nulls.py`.

The first cross-core run used only the reversible schedule. The second run added a read-only echo pass over the same schedule lines after restoration, then let the observer probe from the other core.

## Results

### Cross-Core Direct

Tracked summary:

```text
phase4_holo/results/phase4b_cache_hologram_cross_core_summary.json
```

Raw CSV is ignored:

```text
phase4_holo/results/phase4b_cache_hologram_cross_core.csv
```

Result:

| Metric | Value |
|---|---:|
| Rows | 3840 |
| Hash restored | 3840/3840 |
| Real accuracy | 0.275000 |
| Real floor | 0.112500 |
| Pseudo declared-match | 0.220312 |
| Real-vs-pseudo floor | 0.459375 |
| Pseudo reject floor | 0.425000 |
| Wrong actual-match | 0.243750 |
| Wrong declared-match | 0.243750 |

Verdict:

```text
PHASE4B_MATCHED_NULLS_PARTIAL
```

### Cross-Core Echo

Tracked summary:

```text
phase4_holo/results/phase4b_cache_hologram_cross_core_echo_summary.json
```

Raw CSV is ignored:

```text
phase4_holo/results/phase4b_cache_hologram_cross_core_echo.csv
```

Result:

| Metric | Value |
|---|---:|
| Rows | 3840 |
| Hash restored | 3840/3840 |
| Real accuracy | 0.245312 |
| Real floor | 0.087500 |
| Pseudo declared-match | 0.248438 |
| Real-vs-pseudo floor | 0.484375 |
| Pseudo reject floor | 0.368750 |
| Wrong actual-match | 0.242188 |
| Wrong declared-match | 0.251563 |

Verdict:

```text
PHASE4B_MATCHED_NULLS_PARTIAL
```

## Interpretation

Cross-core observation is not confirmed with the current timing method.

This does not erase the same-core results:

- `PHASE4B_MATCHED_NULLS_REPEATABLE_PASS`
- `PHASE4B_LAYOUT_HOLDOUT_PASS`
- `PHASE4B_SUBSTRATE_COORDINATE_CONFIRMED`

It sets the current Phase 4B boundary:

```text
The .holo cache witness is readable as a same-core substrate-coordinate timing structure, but does not yet survive a simple cross-core observer split.
```

The failed echo run is useful because it tested a stronger physical afterimage while preserving logical restoration; it still did not recover the matched-null structure from the observer core.

## Claim Boundary

Supported:

```text
PHASE4B_SCALAR_PHYSICAL_HOLO_WITNESS_SAME_CORE
PHASE4B_SUBSTRATE_COORDINATE_CONFIRMED
```

Not supported yet:

```text
PHASE4B_CROSS_CORE_HOLO_LOCKIN_WITNESS
```

## Next Push

- Try writer/observer core pairs beyond `0 -> 1`.
- Add an eviction-set observer instead of direct line timing.
- Add a shared last-level-cache occupancy readout if available.
- Keep cross-core as a separate gate; do not use it to weaken the same-core/layout witness.
