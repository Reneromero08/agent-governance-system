# Phase 5.9V Phase 6 Target-Coupled Basin Matrix

Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`

Objective: test whether a public Phase 6 target payload can select collapsed/mid/high carrier basins when it drives both prelude dynamics and workload shape.

- VID offset: +5
- Decoded voltage: 1.1625V
- Rows analyzed: 40
- Restoration failures: 0
- Malformed final CSV rows skipped during analysis: 5
- Requested repeats per selector: 5
- Coupled workload: 1

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness | Skipped rows |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|--------------|
| d_oracle_prelude | 5 | 2 | 2 | 1 | collapsed | 0.400 | 0.600 | 0.800 | 3251.380581 | 15704.084949 | 1 |
| d_oracle_syscall_prelude | 5 | 1 | 1 | 3 | high | 0.600 | 0.800 | 0.400 | 6619.516110 | 12785.905042 | 1 |
| public_kb_prelude | 5 | 2 | 1 | 2 | high | 0.400 | 0.600 | 0.600 | 3848.161468 | 12153.177352 | 0 |
| public_kb_syscall_prelude | 5 | 1 | 3 | 1 | mid | 0.600 | 0.800 | 0.800 | 2108.985324 | 5757.777368 | 1 |
| shuffled_kb_prelude | 5 | 3 | 2 | 0 | collapsed | 0.600 | 0.400 | 1.000 | 847.476328 | 2309.041041 | 0 |
| shuffled_kb_syscall_prelude | 5 | 2 | 3 | 0 | mid | 0.600 | 0.600 | 1.000 | 1114.538891 | 3949.500875 | 1 |
| wrong_kb_prelude | 5 | 2 | 2 | 1 | collapsed | 0.400 | 0.600 | 0.800 | 4332.176886 | 16593.655265 | 0 |
| wrong_kb_syscall_prelude | 5 | 0 | 2 | 3 | high | 0.600 | 1.000 | 0.400 | 4127.521261 | 5892.269841 | 1 |

## Gate Readout

- Restoration: PASS.
- Public-prelude top rate: 0.400.
- Shuffled-prelude top rate: 0.600.
- Oracle-control top rate: 0.400; this is a smuggle detector, not crossing evidence.
- Public target coupling did not produce a deterministic public basin selector in this matrix.
- Mode C still requires public selector reproducibility plus answer-predictive invariant separation.
