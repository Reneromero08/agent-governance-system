# Phase 5.9V Phase 6 Basin Reproducibility Matrix

Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`

Objective: hold P4 VID+5 and test whether Phase 6-relevant preludes reproducibly select collapsed/mid/high carrier basins.

This is not a Mode C fixed-point crossing claim. The public/shuffled/oracle preludes here are substrate-prelude probes; answer-predictive coupling still requires the Phase 6 fixed-point map integration.

- VID offset: +5
- Decoded voltage: 1.1625V
- Rows analyzed: 90
- Restoration failures: 0
- Requested repeats per selector: 10

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|
| d_oracle_syscall_prelude | 10 | 3 | 3 | 4 | high | 0.400 | 0.700 | 0.600 | 2529.481847 | 6797.787811 |
| public_kb_branch_prelude | 10 | 4 | 2 | 4 | collapsed | 0.400 | 0.600 | 0.600 | 4118.152243 | 15874.791269 |
| public_kb_cache_prelude | 10 | 2 | 5 | 3 | mid | 0.500 | 0.800 | 0.700 | 2917.464953 | 7644.144694 |
| public_kb_syscall_prelude | 10 | 3 | 4 | 3 | mid | 0.400 | 0.700 | 0.700 | 3210.487857 | 12068.294624 |
| quiet | 10 | 1 | 6 | 3 | mid | 0.600 | 0.900 | 0.700 | 4316.778878 | 14585.337495 |
| shuffled_kb_branch_prelude | 10 | 2 | 7 | 1 | mid | 0.700 | 0.800 | 0.900 | 1413.044666 | 5969.268116 |
| shuffled_kb_cache_prelude | 10 | 4 | 4 | 2 | mid | 0.400 | 0.600 | 0.800 | 2493.757253 | 7769.194427 |
| shuffled_kb_syscall_prelude | 10 | 1 | 6 | 3 | mid | 0.600 | 0.900 | 0.700 | 3373.729340 | 9733.483020 |
| syscall_prelude | 10 | 3 | 5 | 2 | mid | 0.500 | 0.700 | 0.800 | 2312.017814 | 8205.850603 |

## Gate Readout

- Restoration: PASS.
- Mode C handoff requires public-prelude basin reproducibility plus answer-predictive invariant separation; this run tests basin reproducibility only.
