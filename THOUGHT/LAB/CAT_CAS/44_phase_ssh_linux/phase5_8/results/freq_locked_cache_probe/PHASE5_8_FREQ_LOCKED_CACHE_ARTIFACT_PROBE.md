# Phase 5.8 Frequency-Locked Cache Artifact Probe

Verdict: `CACHE_ARTIFACT_CLEARED`

Objective: test whether the large-tape CACHE anomaly survives fixed P-state and interleaved NONE/CACHE ordering.

Acceptance:
- `CACHE_ARTIFACT_CLEARED`: CACHE/NONE thickness ratio > 1.05 for all tested large tapes.
- `CACHE_ARTIFACT_PERSISTS_OR_BOUNDARY_CONTRACTS`: any tested large tape has ratio < 0.95.
- `CACHE_EFFECT_FLAT_UNDER_FREQ_LOCK`: all ratios within +/-5%.

## Summary

| Tape | NONE thickness | CACHE thickness | CACHE/NONE | NONE radius | CACHE radius |
|------|----------------|-----------------|------------|-------------|--------------|
| 1024 | 0.683017 | 1.225586 | 1.794370 | 3.376966 | 3.418498 |
| 4096 | 1.238444 | 1.525784 | 1.232016 | 3.300889 | 3.293042 |

Artifacts:
- `freq_locked_cache_probe_summary.csv`
- `msr_lock_audit.csv`
- `condition_order.csv`
