# Phase 2B Roadmap Fix Report

**Date:** 2026-06-05

## What Was Wrong

The previous closure labeled Phase 2B as `CURRENT_PASSIVE_MECHANISMS_CLOSED` which could be read as "Phase 2B is done." This was too broad. Phase 2B is a container for ALL black-box attractor approaches. Only one branch (passive MESI binary-spin flips) was tested and exhausted.

## What Branch Is Actually Closed

`PHASE2B_PASSIVE_MESI_SPIN_BRANCH_CLOSED`

The following mechanisms were tested and failed to produce general passive shared-substrate advantage:
- Random binary spin flips through shared cache
- P1 topology-only ferro-bias (falsified by anti-ferro acid test)
- QR orthogonal subspace partition
- Retrocausal 2-pass self-consistency
- Warm-tape fingerprint contention
- DID frequency detuning

## What Branch Remains Alive

`PHASE2B_PHASE_ORACLE_BRANCH_UNTESTED`

The CAT_CAS ZIP contains stronger mechanisms never ported to the Phenom II:
- Exp20: phase lasing, FFT/QFT, MUSIC, phase oracle filter bank, holographic phase oracle, contained .holo phase cavity
- Exp26: optical 3-SAT phase-shifting mirror interference
- Exp07: Bloch-vector phase-angle Ising simulator (1M qubits, O(N) memory)
- Exp31: graph isomorphism spectral classifier
- Exp33: MERA/.holo compression bridge

## New Phase 2B.5 Structure

| Subphase | Name | Source | Priority |
|----------|------|--------|----------|
| 2B.5A | Exp20 Phase-Oracle Port | phase_oracle_port.c | FIRST |
| 2B.5B | Exp26 Optical 3-SAT Port | optical_3sat_phase_port.py | SECOND |
| 2B.5C | Exp07 Bloch/Complex Ising Port | bloch_complex_ising.py | THIRD |
| 2B.5D | Exp31 Spectral Classifier | spectral_problem_classifier.py | SUPPORT |
| 2B.5E | Exp33 .holo/MERA Bridge | TBD | LATER |

## Next Exact Implementation Task

**Phase 2B.5A:** Port Exp20 phase-oracle machinery to Phenom II.
1. Extract complex-phase encoding from Exp20 (phase lasing, filter bank, .holo cavity)
2. Implement `session_scripts/phase2b/phase_oracle_port.c`
3. Represent candidate states as complex phase values, not binary spins
4. Encode problem constraints as phase shifts
5. Decode answer from interference pattern
6. Compare to binary-spin baseline and nulls

## Files Changed

- `ROADMAP.md` — Phase 2B header, closure note, 2B.5 section added
- `PHASE2B_DECISION_TREE.md` — Updated with MESI-vs-oracle branch distinction
- `PHASE2B_UPDATE_SUMMARY.md` — Updated with correction
- `PHASE2B_ZIP_MECHANISM_INVENTORY.md` — NEW
- `PHASE2B_BRANCH_CORRECTION.md` — NEW
- `PHASE2B_ROADMAP_FIX_REPORT.md` — NEW (this file)

## Final Verdict

"Phase 2B passive MESI/binary-spin mechanisms are closed for current tests. Phase 2B remains alive through the unported CAT_CAS phase-oracle/interference branch."
