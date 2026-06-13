# Phase 5.9V Phase 6 Basin Reproducibility Matrix

Verdict: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`

Objective: hold P4 VID+5 and test whether Phase 6-relevant preludes reproducibly select collapsed/mid/high carrier basins.

This is not a Mode C fixed-point crossing claim. The public/shuffled/oracle preludes here are substrate-prelude probes; answer-predictive coupling still requires the Phase 6 fixed-point map integration.

- VID offset: +6
- Decoded voltage: 1.1500V
- Rows analyzed: 30
- Restoration failures: 0
- Requested repeats per selector: 5

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|
| public_kb_prelude | 5 | 2 | 1 | 2 | collapsed | 0.400 | 0.600 | 0.600 | 3413.562350 | 10128.070585 |
| public_kb_syscall_prelude | 5 | 0 | 4 | 1 | mid | 0.800 | 1.000 | 0.800 | 2679.983638 | 10099.697331 |
| quiet | 5 | 2 | 1 | 2 | high | 0.400 | 0.600 | 0.600 | 6315.335361 | 15986.795370 |
| shuffled_kb_prelude | 5 | 3 | 2 | 0 | collapsed | 0.600 | 0.400 | 1.000 | 996.678675 | 2694.182376 |
| shuffled_kb_syscall_prelude | 5 | 2 | 2 | 1 | collapsed | 0.400 | 0.600 | 0.800 | 2352.125171 | 8868.987733 |
| syscall_prelude | 5 | 3 | 1 | 1 | collapsed | 0.600 | 0.400 | 0.800 | 3593.287259 | 17365.761934 |

## Gate Readout

- Restoration: PASS.
- Public-prelude top rate: 0.400.
- Shuffled-prelude top rate: 0.600.
- Mode C handoff requires public-prelude basin reproducibility plus answer-predictive invariant separation; this run tests basin reproducibility only.
