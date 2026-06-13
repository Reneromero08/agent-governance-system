# PHASE5_8_REVERIFY_ARTIFACT_RECHECK

## Verdict

`PHASE5_8_ARTIFACT_EVIDENCE_PARTIAL__BOUNDARY_SCALING_ALIVE`

The Phase 5.8 claim was rechecked against tracked repository artifacts. The
strict `AREA_LAW_CONFIRMED` label is not fully reproducible from the files
currently present in the repo because the full condition-matrix output tree is
not tracked.

This does not erase the hypothesis. It changes the current evidence status:
Phase 5.8 remains a boundary-scaling lead, but strict area-law confirmation
requires either the original matrix artifacts or a fresh rerun.

## Artifact Check

Tracked Phase 5.8 files currently include:

- reports and summaries
- `50_5_8_boundary_scaling/results/freq_locked_cache_probe/condition_order.csv`
- `50_5_8_boundary_scaling/results/freq_locked_cache_probe/freq_locked_cache_probe_summary.csv`
- `50_5_8_boundary_scaling/results/freq_locked_cache_probe/msr_lock_audit.csv`
- `50_5_8_boundary_scaling/results/freq_locked_cache_probe/PHASE5_8_FREQ_LOCKED_CACHE_ARTIFACT_PROBE.md`

Missing from tracked artifacts:

- full 15-condition raw matrix `raw_cycles.csv` files
- per-run `window_features.csv`
- per-run `geometry_stats.csv`
- cross-run `cross_run_area_law_stats.csv`
- `phase5_8_master_verdict.csv`

## R2 Consistency Check

The final report states:

| Model | Thickness R2 | Radius R2 |
|---|---:|---:|
| Volume | 0.4725 | 0.0226 |
| Area | 0.7071 | 0.1493 |
| Log | 0.8812 | 0.0781 |

Area beats volume on both listed metrics, so the reported "strict area wins = 2"
is internally consistent. However, log beats area on thickness. The careful
claim is therefore not "pure area law"; it is:

```text
volume-beating sublinear boundary scaling, with area and log both viable on the
reported Phase 5.8 dataset.
```

Phase 5.9A later found volume-law wins under stress conditions, so Phase 5.8
should not be generalized across stressed regimes.

## Hardened Status

Use:

```text
PHASE5_8_BOUNDARY_SCALING_LEAD_ARTIFACT_PARTIAL
```

until the full condition matrix is restored or rerun.

Acceptance to restore strict status:

- condition matrix artifacts present
- true eigendecomposition artifacts present
- cross-run model comparison present
- area/log split present
- rerun or artifact hashes attached
