# Phase 5.9V Phase 6 Basin Reproducibility Matrix

Verdict: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`

Objective: hold P4 VID+5 and test whether Phase 6-relevant preludes reproducibly select collapsed/mid/high carrier basins.

This is not a Mode C fixed-point crossing claim. The public/shuffled/oracle preludes here are substrate-prelude probes; answer-predictive coupling still requires the Phase 6 fixed-point map integration.

- VID offset: +6
- Decoded voltage: 1.1500V
- Rows analyzed: 70
- Restoration failures: 0
- Requested repeats per selector: 10

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|
| d_oracle_syscall_prelude | 10 | 8 | 2 | 0 | collapsed | 0.800 | 0.200 | 1.000 | 445.598774 | 3893.938976 |
| public_kb_prelude | 10 | 3 | 6 | 1 | mid | 0.600 | 0.700 | 0.900 | 1547.640509 | 7946.046292 |
| public_kb_syscall_prelude | 10 | 6 | 3 | 1 | collapsed | 0.600 | 0.400 | 0.900 | 1827.666748 | 7451.095621 |
| quiet | 10 | 2 | 6 | 2 | mid | 0.600 | 0.800 | 0.800 | 2601.335684 | 13197.684854 |
| shuffled_kb_prelude | 10 | 1 | 6 | 3 | mid | 0.600 | 0.900 | 0.700 | 3217.462484 | 10323.862580 |
| shuffled_kb_syscall_prelude | 10 | 3 | 5 | 2 | mid | 0.500 | 0.700 | 0.800 | 2548.822479 | 14580.576828 |
| syscall_prelude | 10 | 3 | 4 | 3 | mid | 0.400 | 0.700 | 0.700 | 4184.265053 | 15490.504507 |

## Gate Readout

- Restoration: PASS.
- Public-prelude top rate: 0.600.
- Shuffled-prelude top rate: 0.600.
- Mode C handoff requires public-prelude basin reproducibility plus answer-predictive invariant separation; this run tests basin reproducibility only.
