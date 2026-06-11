# Phase 5.9V Phase 6 Target-Coupled Basin Matrix

Verdict: `PHASE5_9V_SELECTOR_REPRODUCIBLE_NONPUBLIC`

Objective: test whether a public Phase 6 target payload can select collapsed/mid/high carrier basins when it drives both prelude dynamics and workload shape.

- VID offset: +6
- Decoded voltage: 1.1500V
- Rows analyzed: 40
- Restoration failures: 0
- Malformed final CSV rows skipped during analysis: 0
- Requested repeats per selector: 5
- Coupled workload: 1

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness | Skipped rows |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|--------------|
| d_oracle_prelude | 5 | 0 | 3 | 2 | mid | 0.600 | 1.000 | 0.600 | 7340.945802 | 21333.951754 | 0 |
| d_oracle_syscall_prelude | 5 | 2 | 1 | 2 | high | 0.400 | 0.600 | 0.600 | 6043.923426 | 23053.998592 | 0 |
| public_kb_prelude | 5 | 1 | 1 | 3 | high | 0.600 | 0.800 | 0.400 | 5792.890653 | 10534.040645 | 0 |
| public_kb_syscall_prelude | 5 | 3 | 1 | 1 | collapsed | 0.600 | 0.400 | 0.800 | 1367.706621 | 6139.393478 | 0 |
| shuffled_kb_prelude | 5 | 1 | 4 | 0 | mid | 0.800 | 0.800 | 1.000 | 1391.769932 | 3881.159620 | 0 |
| shuffled_kb_syscall_prelude | 5 | 2 | 3 | 0 | mid | 0.600 | 0.600 | 1.000 | 427.027127 | 1683.793256 | 0 |
| wrong_kb_prelude | 5 | 2 | 3 | 0 | mid | 0.600 | 0.600 | 1.000 | 1402.989033 | 2992.936775 | 0 |
| wrong_kb_syscall_prelude | 5 | 1 | 3 | 1 | mid | 0.600 | 0.800 | 0.800 | 3427.452061 | 11610.056876 | 0 |

## Gate Readout

- Restoration: PASS.
- Public-prelude top rate: 0.600.
- Shuffled-prelude top rate: 0.800.
- Oracle-control top rate: 0.600; this is a smuggle detector, not crossing evidence.
- Public target coupling did not produce a deterministic public basin selector in this matrix.
- Mode C still requires public selector reproducibility plus answer-predictive invariant separation.
