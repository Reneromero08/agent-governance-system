# CAT_CAS Lab State Audit

**Date:** 2026-06-14.
**Status:** LAB_STATE_COHERENT_SUBSTRATE_FRONTIER_READY.

---

## Canonical Experiment Map

| Experiment | Canonical Name | Directory | Status | Role |
|---|---|---|---|---|
| Exp 44 | Phase Atom / Atomic Ground State | `44_phase_atom/` | AUDIT-VERIFIED | Atomic/nuclear/particle-scale claims |
| Exp 49 | The Decoder | `49_the_decoder/` | CLOSED OUT (L4-5) | Theory terminus. Extractive decoder, decodability boundary located. Hands off to Exp 50. |
| Exp 50 | Phase BM CPU / Bare-Metal CPU Substrate Push | `7_decoder/50_phase_bm_cpu/` | INFRASTRUCTURE / ACTIVE | Hardware substrate frontier. Receives handoff from Exp 49. |
| Phase 6 | Chiral Lane Frontier (Sprints 0-2) | `50_6_fixed_point_substrate/chiral_lane_frontier/` | BOUNDARY MAPPED | Boundary mapping campaign within Exp 50. |

**Identifier collision resolved (MASTER_REPORT.md line 587):** Exp 44 is Phase Atom. The substrate experiment is Exp 50. The stale "Exp 44 Phase 6" alias in historical Exp 49 reports is a handoff target that maps to Exp 50.

---

## Phase 6 Final State

| Gate/Track | Status | Claim | Commit |
|---|---|---|---|
| Sprint 0 (E5/E1, Z, 0, B, I) | COMPLETE (5/5) | L4 | Various |
| Track A | CLOSED (negative, formally adjudicated) | L4 | `4b31715e` |
| Track D | COMPLETE (reference negative) | L3 | `45bf2f01` |
| Track F | PARTIAL (weak seed-dependent) | L3 | `6e284cfd` |
| Track C | DEFERRED (manual label encoding) | -- | -- |
| Track E | DEFERRED (manual label encoding) | -- | -- |

**Boundary claim:** All no-smuggle tracks executed or file-audited under the Exp 50.14 oracle produce null, weak candidate-value-only signals, or fail orientation recovery.

**Non-overclaim:** Not proof that no hardware/platform/oracle/substrate can produce a different result.

---

## Open Items

| Item | Status | Reason |
|---|---|---|
| Exp 50 substrate question | OPEN | Hardware-dependent. Does catalytic Phenom reach fix(f)=d in poly? |
| Track I full 12-route sweep | DEFERRED | Route 4:5 confirmed from T300 (6/6). 10 routes unmeasured. |
| Exp 49 stale references | DEFERRED | 80+ "Exp 44 Phase 6" in Exp 49 reports. Historical, not current. |

---

## Current Frontier

**Exp 50 substrate frontier.** The forward boundary is measured and confirmed across Phase 6. The untested lever is the catalytic/reversible/CTC substrate. The handoff target is the Exp 50.14 public fixed-point map: does catalytic silicon reach `fix(f) = d` reversibly where a forward machine needs `2^n`?

---

## Cleanup Required

- **Urgent:** None. Current Phase 6 files are identifier-correct.
- **Later:** `EXP44_PHASE6_HANDOFF.md` filename + T300 JSON use stale `Exp 44`. Exp 49 historical references.
- **Optional:** Stale Rust file in `chiral_phase_kickback/rust_baremetal/` (Phenom path policy says C).
