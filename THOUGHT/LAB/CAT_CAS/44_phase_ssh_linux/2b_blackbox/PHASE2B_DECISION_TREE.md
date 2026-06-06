# Phase 2B Decision Tree

**Date:** 2026-06-05

## Status Labels

| Label | Meaning |
|---|---|
| `PHASE2B_NOT_TESTED` | No tests run yet |
| `PHASE2B_PASSIVE_ATTRACTOR_CANDIDATE` | Passive shared-substrate beats nulls without gradient-aware worker |
| `PHASE2B_PASSIVE_BASELINE_BEATEN` | Passive result survives full null hierarchy |
| `PHASE2B_PASSIVE_NULLS_FAILED` | Passive result does not beat nulls |
| `PHASE2B_REJECTED_SOFTWARE_EXPLAINS` | Active software baseline explains result |
| `PHASE2B_ACTIVE_CATALYTIC_SOLVER_WORKING` | Active solver works but is not Kuramoto evidence |
| `PHASE2B_ACTIVE_NOT_KURAMOTO_EVIDENCE` | Active solver exists but does not imply physical coupling |
| `PHASE2B_NEGATIVE` | No condition beats nulls |

## Decision Flow

```
Run passive harness (2B.2)
    │
    ├─ Does passive beat Null 0 (random)?
    │   ├─ NO  → PHASE2B_NEGATIVE
    │   └─ YES → Continue
    │
    ├─ Does passive beat Null 1 (single-core)?
    │   ├─ NO  → PHASE2B_PASSIVE_NULLS_FAILED
    │   └─ YES → Continue
    │
    ├─ Does passive beat Nulls 2-6 (shared tape, shuffled, isolated)?
    │   ├─ NO  → PHASE2B_PASSIVE_NULLS_FAILED
    │   └─ YES → Continue
    │
    ├─ Does passive beat Null 7 (active software baseline)?
    │   ├─ NO  → PHASE2B_REJECTED_SOFTWARE_EXPLAINS
    │   └─ YES → PHASE2B_PASSIVE_ATTRACTOR_CANDIDATE
    │
    └─ Independent replication across problem suite?
        ├─ NO  → PHASE2B_PASSIVE_ATTRACTOR_CANDIDATE (single-instance)
        └─ YES → PHASE2B_PASSIVE_BASELINE_BEATEN

Run active solver (2B.3)
    │
    ├─ Does active beat random?
    │   ├─ NO  → Active solver broken
    │   └─ YES → PHASE2B_ACTIVE_CATALYTIC_SOLVER_WORKING
    │
    └─ Does active also beat passive?
        ├─ YES → PHASE2B_ACTIVE_NOT_KURAMOTO_EVIDENCE
        │        (gradient-aware code outperforms, as expected)
        └─ NO  → Investigate passive advantage (unusual)
```

## False-Positive Checklist

Before claiming `PHASE2B_PASSIVE_ATTRACTOR_CANDIDATE`, verify:

- [ ] Passive worker does not call any J_ij-dependent flip logic
- [ ] Passive worker does not compute local fields
- [ ] Passive worker does not minimize energy explicitly
- [ ] No energy gradient is accessible inside the worker loop
- [ ] Full null hierarchy (0-7) tested
- [ ] Multiple problem instances tested (not single cherry-pick)
- [ ] Statistical measures reported (CI, effect size, p-value)
- [ ] Active baseline separately labeled and not confused with passive

## Current Status — ACTIVE THREAD: 2B.5A v9

`PHASE2B_PASSIVE_MESI_SPIN_BRANCH_CLOSED` — Binary-spin passive MESI branch exhausted. Phase 2B remains alive through phase-oracle branch (2B.5A).

**2B.5A v1-v8**: phase oracle beats random/sign-shuffled/misaligned-perm nulls. Edge fidelity partial. Spectral eigenbasis no gain. **Next: v9 MUSIC/super-resolution filter bank.**

**3.14**: Hybrid phase-seeded catalytic Ising — phase seeding no advantage over random init. Active solver dominates. Complete, parked.

**3.15**: Active Core Escape Dynamics — PARKED future work. Do not implement until 2B.5A closed.

**Next exact task:** PHASE2B_5A_V9_MUSIC_SUPER_RESOLUTION_FILTER_BANK
