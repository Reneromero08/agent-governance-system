# Phase 2B.5A Final Status: Exp20 Phase-Oracle Ising Port

**Date:** 2026-06-06
**Final Status:** PHASE2B_5A_CLOSED_SUCCESSFUL_PARTIAL

## 1. Objective

Port the Exp20 Phase-Oracle Ising approach to Phenom II as a structural inference engine: encode Ising problems as continuous phase angles, optimize via gradient descent on a trigonometric objective, and decode phases to spin configurations. Evaluate as a structural black-box test for Kuramoto/GOE signature detection (or absence thereof).

## 2. Version Summary

| Version | Description | Result |
|---------|-------------|--------|
| v5 | Vertex phase oracle: sin(J * sin(delta)) gradient descent | Beats random null 6/6 |
| v6 | Structure nulls added (edge-rewired, sign-shuffled) | Baseline quality established |
| v7 | Sign fidelity PASS, label alignment PASS, edge fidelity PARTIAL | Core engine validated |
| v8 | Spectral Jacobi eigenbasis initializer | No gain over random init |
| v9 | MUSIC/super-resolution | No gain, worse than v7 |
| v10 | Autocorr/coherence/cepstrum features | Edge coherence signal found, no Ising energy gain |
| v11 | Coherence-guided refinement (edge term + worst-edge fix) | Sparse gain, edge fidelity improved |
| v12 | Adaptive selector (energy + coherence scoring) | Beats v7 and v11 |
| v13 | Selector ablation | v12 win = energy-only ensembling, coherence adds zero selection value |
| v14 | Scale test N=8/12/16 | Ensemble scales, beats edge-rewired with growing margins |
| v15 | Stability N=16, loop bounds fix | Ensemble stable, beats v7/v11/edge-rewired |
| Final | N=24/N=32 kill shot | Ensemble survives but advantage shrinks on dense problems |

## 3. Best Implementation

**Energy ensemble:**
1. Run v7 (vertex phase oracle) from random init
2. Run v11 (coherence-guided) from same random init
3. Decode both phase configurations to spins
4. Compute Ising energies on the original J
5. Select lower energy

This is exactly v13's ablation-confirmed engine. Coherence is used only within v11's descent; the final selection is energy-only.

## 4. What Worked

- **Phase encoding works:** Continuous phase representation captures Ising Hamiltonian structure
- **Sign fidelity works:** Phase angle cos(theta_i) >= 0 decoding preserves edge sign relationships
- **Label alignment works:** Decoded spin configurations maintain correct sign alignment
- **Coherence signal detected:** Edge-specific coherence features exist (v10) but are diagnostic, not selection-valuable (v13)
- **Coherence-guided refinement works:** v11 improves edge fidelity on sparse and frustrated graphs
- **Energy ensembling works:** v7 + v11 + min-energy selection reliably beats either alone
- **Scale stability to N=16 confirmed:** Ensemble maintains advantage at N=16 (v14, v15)
- **N=24 survival confirmed:** Ensemble beats all nulls at N=24; v11 hits planted ground truth (-276)
- **N=32 survival confirmed:** Ensemble beats all nulls at N=32; but advantage over rand-phase-descent collapses on dense problems

## 5. What Did Not Work

- **Spectral eigenbasis (v8):** Jacobi eigenvalue decomposition produces no better initial phase angles than random
- **MUSIC/super-resolution (v9):** Noise subspace decomposition provides no signal for phase initialization
- **Coherence-only selection (v13):** Adding coherence to the selection score adds zero value beyond energy-only
- **Hybrid phase-seeded active solver:** Phase oracle seed did not improve the Phase 3 active edge solver
- **Large-N dense problems:** N=32 planted (496 edges) not solved; best energy -226 vs ground truth -496 (46%)
- **Ensemble advantage scaling:** Advantage over best individual method (rand-phase-descent) collapses from -10.08 (N=24 planted) to -1.07 (N=32 planted)

## 6. Final Verdict

**PHASE2B_5A_CLOSED_SUCCESSFUL_PARTIAL**

The phase-oracle ensemble is a functional structural inference engine. It reliably extracts Ising graph structure beyond random nulls, edge-rewired nulls, and individual phase oracle runs. However:

1. It is not the primary Ising engine (active edge baseline from Phase 3 dominates)
2. Its advantage shrinks on dense problems at N=32
3. It does not reach global optima on planted problems at N=32
4. It has no detectable physical Kuramoto/GOE signature — the phase oracle is purely computational gradient descent

The phase-oracle branch of Phase 2B has been fully explored. The ensemble is the best 2B.5A method. It works. It was worth building. It is not the engine.

## 7. Next Roadmap Pointer

Next work resumes from the roadmap after 2B.5A. The roadmap indicates:
- Phase 2B.5B-2B.5E (Optical 3-SAT, Bloch Ising, Spectral Classifier, .holo/MERA Bridge) remain available
- Phase 3 (Catalytic Computing Ladder) is complete
- Phase 4 Track A (Catalytic Tape) is complete
- Phase 4 Track B (Physical Phase Network) remains pending

PHASE2B_5A_CLOSED_SUCCESSFUL_PARTIAL
