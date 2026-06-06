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

## Current Status

`PHASE2B_PASSIVE_MESI_SPIN_BRANCH_CLOSED` — Binary-spin passive MESI branch failed current tests. Phase 2B remains alive through unported CAT_CAS phase-oracle/interference branch.

**2B.2** random-flip: negative. **2B.3A** wormhole protocol transfer: PASS (not Ising claim). **2B.3B** P1 ferro-bias FALSIFIED, P2 active edge solver works (shared=null). **2B.4** channel matrix: all 4 CAT_CAS-derived passive channels failed mixed-sign controls.

**Correction:** Previous `CURRENT_PASSIVE_MECHANISMS_CLOSED` was too broad. Only the MESI binary-spin branch is closed. Phase-oracle/interference mechanisms (Exp20, Exp26, Exp07, Exp31, Exp33) were never ported to the Phenom II. Phase 2B is NOT globally dead.

**Next:** 2B.5A Exp20 phase-oracle port to Phenom II (`session_scripts/phase2b/phase_oracle_port.c`). See `PHASE2B_ZIP_MECHANISM_INVENTORY.md` for full untested mechanism inventory.
