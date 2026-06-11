# Phase 5.9V Phase 6 Basin Reproducibility Matrix

Verdict: `PHASE5_9V_DIRECTIONAL_REPRODUCED_NOT_DETERMINISTIC`

Objective: hold P4 VID+5 and test whether Phase 6-relevant preludes reproducibly select collapsed/mid/high carrier basins.

This is not a Mode C fixed-point crossing claim. The public/shuffled/oracle preludes here are substrate-prelude probes; answer-predictive coupling still requires the Phase 6 fixed-point map integration.

- VID offset: +4
- Decoded voltage: 1.1750V
- Rows analyzed: 30
- Restoration failures: 0
- Requested repeats per selector: 5

## Selector Summary

| Selector | n | Collapsed | Mid | High | Top basin | Top rate | Noncollapse | Anti-high | Mean thickness | Max thickness |
|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|----------------|---------------|
| public_kb_prelude | 5 | 2 | 3 | 0 | mid | 0.600 | 0.600 | 1.000 | 1916.581697 | 3873.850191 |
| public_kb_syscall_prelude | 5 | 2 | 2 | 1 | mid | 0.400 | 0.600 | 0.800 | 2691.892934 | 8673.109030 |
| quiet | 5 | 1 | 2 | 2 | high | 0.400 | 0.800 | 0.600 | 5147.568807 | 15247.799884 |
| shuffled_kb_prelude | 5 | 3 | 2 | 0 | collapsed | 0.600 | 0.400 | 1.000 | 487.035952 | 2173.407264 |
| shuffled_kb_syscall_prelude | 5 | 1 | 2 | 2 | high | 0.400 | 0.800 | 0.600 | 3704.719161 | 9106.581028 |
| syscall_prelude | 5 | 0 | 3 | 2 | mid | 0.600 | 1.000 | 0.600 | 5063.547723 | 11692.753616 |

## Gate Readout

- Restoration: PASS.
- Public-prelude top rate: 0.600.
- Shuffled-prelude top rate: 0.600.
- Mode C handoff requires public-prelude basin reproducibility plus answer-predictive invariant separation; this run tests basin reproducibility only.
