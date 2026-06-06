# Phase 2B Update Summary

**Date:** 2026-06-05
**Roadmap:** `SSH_ROADMAP.md`

## What Changed

Phase 2 split into 2A (direct Kuramoto observation, exhausted/unresolved) and 2B (black-box attractor computation). Phase 2B does not attempt to watch phase lock happen. Instead, it asks whether the Phenom can be USED as a black-box attractor computer where the final answer distribution is the only measurement.

## Why Phase 2 Split

Phase 2A tried three coupling channels (power grid, LOCK CMPXCHG, matched-frequency MESI) and found:
- Power-grid coupling detected but below TSC noise floor
- MESI protocol provides consistency, not synchronization
- No Kuramoto phase lock observed through software-visible channels

Phase 2B pivots: instead of watching the hidden dynamics, use them. Encode problems into the substrate, let it evolve, read the final answer. The answer IS the measurement.

## Passive vs Active Distinction

| | 2B-Passive | 2B-Active |
|---|---|---|
| Worker logic | No Ising gradient | May use local-field updates |
| Claim if works | Hidden attractor dynamics | Catalytic/reversible optimization |
| Kuramoto evidence | Statistical, indirect | None (it's software) |
| Belongs to | Phase 2B black-box | Phase 3 bridge/application |

## False-Positive Risks

1. **Hidden gradient**: Worker that accidentally uses J_ij to compute flip direction produces software-biased results that look like attractor dynamics
2. **Software Ising**: Explicit local-field computation with energy minimization is a software solver, not a substrate measurement
3. **Single-null cherry-pick**: Beating one null but not the full hierarchy is not evidence
4. **Active mislabeled as passive**: Labeling a gradient-aware worker as "passive" creates false Kuramoto claims

## Current Status

`PHASE2B_PASSIVE_NULLS_FAILED` — Random-flip passive worker tested. Did not beat nulls. Pivot to 2B.3 correlation-based attractor.

## Next Exact Implementation Task

**Phase 2B.3:** Wormhole-Inspired Correlation Attractor.
1. Workers compute pairwise spin correlations (si * sj) through shared tape slots via atomic XOR
2. Accumulate correlations — no spin flipping in first test, measurement only
3. Compare correlation patterns against J_ij coupling structure
4. Test whether shared-tape correlation patterns match J_ij better than nulls
5. File: `session_scripts/phase2b/wormhole_correlation.c`
