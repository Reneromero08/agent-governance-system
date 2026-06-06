# CAT_CAS ZIP Mechanism Inventory for Phase 2B Phase-Oracle Branch

**Date:** 2026-06-05

## Mechanisms NOT Yet Ported to Phenom II

| ZIP Exp | Mechanism | Why It Matters | Phase 2B Use | File to Create | Status |
|---------|-----------|---------------|-------------|----------------|--------|
| Exp20 | Phase lasing, FFT/QFT, MUSIC super-resolution, phase oracle filter bank, holographic phase oracle, contained .holo phase cavity | Phase-angle encodes answer in non-binary representation. Interference reveals invariant without enumeration. | Represent spin/candidate as complex phase bucket. Decode from interference pattern. | `phase_oracle_port.c` | UNPORTED |
| Exp26 | Optical 3-SAT: variables as phase paths, clauses as mirrors, satisfying = constructive interference | Instant O(1) SAT solving via physical interference. No spin-flip enumeration needed. | Encode small QUBO/SAT as phase grating. Read constructive vs destructive output. | `optical_3sat_phase_port.py` | UNPORTED |
| Exp07 | 1M-qubit Bloch vector simulator: [N,3] Bloch vectors, O(N) memory, spectral aliasing | Mean-field coupling via global Ising interaction. Phase angles, not bits. | Bloch vectors on tape slots. Phase-coupled update rule. | `bloch_complex_ising.py` | UNPORTED |
| Exp31 | Graph isomorphism spectral signatures: permutation-invariant .holo signatures | Classify problem instances before testing. Avoid overtesting equivalent graphs. | Group Ising instances by spectral class. Canonical phase layout. | `spectral_problem_classifier.py` | UNPORTED |
| Exp33 | MERA/.holo eigenbasis compression: shared SVh + rotation chain + residuals | Bridge phase-oracle output to .holo structure for Phase 4A. | Compress problem/result into .holo format. | TBD | UNPORTED |
| Exp12 | Structured tape acceleration: warm-tape caching, fingerprint checksums, cross-depth transfer | Accelerate already-working mechanism, not create new coupling. | Supporting acceleration for phase-oracle implementations. | — | SUPPORT ONLY |
| Exp13 | QR orthogonal subspace sharing: cross-talk 1.98e-16 | Already tested in channel matrix (C1). No passive advantage shown. | Supporting encoding, not passive evidence. | — | SHALLOW TESTED |
| Exp23 | Retrocausal 2-pass convergence | Already tested in channel matrix (C2). No passive advantage shown. | Supporting convergence pattern, not passive evidence. | — | SHALLOW TESTED |

## Mechanisms Tested and Closed (MESI Binary-Spin Branch)

| Mechanism | Result | Verdict |
|-----------|--------|---------|
| Random binary spin flip (2B.2) | Shared did not beat null | NEGATIVE |
| P1 ferro-bias (2B.3B) | Anti-ferro acid test: 0/200 | FALSIFIED |
| P2 sign-aware edge (2B.3B) | Works but shared=null | ACTIVE SOLVER |
| QR subspace (2B.4 C1) | Failed mixed-sign | CLOSED |
| Retrocausal (2B.4 C2) | Failed mixed-sign | CLOSED |
| Warm-tape fingerprint (2B.4 C3) | Failed mixed-sign | CLOSED |
| DID detuning (2B.4 C4) | Failed mixed-sign | CLOSED |

## Status Labels

```
PHASE2B_PASSIVE_MESI_SPIN_BRANCH_CLOSED
PHASE2B_PHASE_ORACLE_BRANCH_UNTESTED
PHASE2B_EXP20_EXP26_EXP07_PORT_REQUIRED
PHASE2B_ANSWER_AS_MEASUREMENT_RETAINED
```
