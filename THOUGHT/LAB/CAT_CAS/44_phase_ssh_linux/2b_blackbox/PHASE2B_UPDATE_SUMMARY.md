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

`PHASE2B_PASSIVE_MESI_SPIN_BRANCH_CLOSED` — Binary-spin passive MESI branch exhausted.

`PHASE2B_5A_CLOSED_SUCCESSFUL_PARTIAL` — Phase-oracle port (Exp20) completed (2026-06-06). Energy-ensemble engine (v7+v11, pick lower energy) beats all nulls at N=24/N=32. Advantage shrinks but survives on dense problems at N=32. Not the primary engine (active edge solver dominates). See `PHASE2B_5A_FINAL_STATUS.md`.

Phase 2B remains alive through remaining untested phase-oracle mechanisms (2B.5B-2B.5E: Optical 3-SAT, Bloch Ising, Spectral Classifier, .holo/MERA Bridge).

## Next Exact Implementation Task

**Phase 2B.5A:** Port Exp20 phase-oracle machinery to Phenom II.
1. Extract complex-phase encoding from Exp20 (phase lasing, filter bank, .holo cavity)
2. Implement `session_scripts/phase2b/phase_oracle_port.c`
3. Represent candidate states as complex phase values, not binary spins
4. Encode problem constraints as phase shifts, decode answer from interference pattern

See `PHASE2B_ZIP_MECHANISM_INVENTORY.md` for full untested mechanism inventory.
