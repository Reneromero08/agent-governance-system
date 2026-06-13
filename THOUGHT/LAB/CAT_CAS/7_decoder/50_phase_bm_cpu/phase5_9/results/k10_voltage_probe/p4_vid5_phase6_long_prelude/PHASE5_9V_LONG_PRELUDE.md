# Phase 5.9V Phase 6 Basin Reproducibility Matrix

Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`

Objective: hold P4 VID+5 and test whether Phase 6-relevant preludes reproducibly select collapsed/mid/high carrier basins.

This is not a Mode C fixed-point crossing claim. The public/shuffled/oracle preludes here are substrate-prelude probes; answer-predictive coupling still requires the Phase 6 fixed-point map integration.

- VID offset: +5
- Decoded voltage: 1.1625V
- Rows analyzed: 50
- Restoration failures: 0
- Requested repeats per selector: 10

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|
| d_oracle_syscall_prelude | 10 | 4 | 2 | 4 | collapsed | 0.400 | 0.600 | 0.600 | 3417.491099 | 9116.320743 |
| public_kb_syscall_prelude | 10 | 5 | 4 | 1 | collapsed | 0.500 | 0.500 | 0.900 | 1907.045112 | 6168.730275 |
| quiet | 10 | 4 | 4 | 2 | mid | 0.400 | 0.600 | 0.800 | 3314.338375 | 16171.902865 |
| shuffled_kb_syscall_prelude | 10 | 5 | 3 | 2 | collapsed | 0.500 | 0.500 | 0.800 | 2328.542861 | 7960.558924 |
| syscall_prelude | 10 | 4 | 4 | 2 | collapsed | 0.400 | 0.600 | 0.800 | 2583.708278 | 12099.345867 |

## Gate Readout

- Restoration: PASS.
- Mode C handoff requires public-prelude basin reproducibility plus answer-predictive invariant separation; this run tests basin reproducibility only.
