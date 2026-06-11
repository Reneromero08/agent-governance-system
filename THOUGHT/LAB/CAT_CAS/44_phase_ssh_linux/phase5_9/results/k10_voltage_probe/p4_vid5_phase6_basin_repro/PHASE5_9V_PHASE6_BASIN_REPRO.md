# Phase 5.9V Phase 6 Basin Reproducibility Matrix

Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`

Objective: hold P4 VID+5 and test whether Phase 6-relevant preludes reproducibly select collapsed/mid/high carrier basins.

This is not a Mode C fixed-point crossing claim. The public/shuffled/oracle preludes here are substrate-prelude probes; answer-predictive coupling still requires the Phase 6 fixed-point map integration.

- VID offset: +5
- Decoded voltage: 1.1625V
- Rows analyzed: 70
- Restoration failures: 0
- Requested repeats per selector: 10

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|
| branch_prelude | 10 | 1 | 5 | 4 | mid | 0.500 | 0.900 | 0.600 | 2837.937163 | 6678.983512 |
| cache_prelude | 10 | 2 | 4 | 4 | high | 0.400 | 0.800 | 0.600 | 4421.491471 | 13773.316998 |
| d_oracle_prelude | 10 | 2 | 4 | 4 | mid | 0.400 | 0.800 | 0.600 | 4061.763750 | 10624.669026 |
| public_kb_prelude | 10 | 3 | 6 | 1 | mid | 0.600 | 0.700 | 0.900 | 1579.791624 | 9687.117247 |
| quiet | 10 | 3 | 6 | 1 | mid | 0.600 | 0.700 | 0.900 | 2487.801538 | 8954.843519 |
| shuffled_kb_prelude | 10 | 6 | 3 | 1 | collapsed | 0.600 | 0.400 | 0.900 | 1346.652327 | 8627.529599 |
| syscall_prelude | 10 | 2 | 1 | 7 | high | 0.700 | 0.800 | 0.300 | 8237.235749 | 15723.055918 |

## Gate Readout

- Restoration: PASS.
- Public-prelude top rate: 0.600.
- Shuffled-prelude top rate: 0.600.
- Oracle-control top rate: 0.400; this is a smuggle detector, not crossing evidence.
- Mode C handoff requires public-prelude basin reproducibility plus answer-predictive invariant separation; this run tests basin reproducibility only.
