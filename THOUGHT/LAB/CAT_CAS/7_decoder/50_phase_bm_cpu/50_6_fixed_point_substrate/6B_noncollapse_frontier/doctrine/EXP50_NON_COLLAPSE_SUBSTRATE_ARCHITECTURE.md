# Exp 50 Non-Collapse Substrate Architecture

**Date:** 2026-06-14.
**Status:** EXP50_NON_COLLAPSE_SUBSTRATE_ARCHITECTURE: ALGORITHM_DEAD__ORBIT_STATE_REQUIRED__NO_EARLY_COLLAPSE

---

## Executive Summary

Exp 50 must move from verifier-loop framing to non-collapse substrate architecture.
The primitive object is not scalar `x` or scalar `d`. The primitive object is an
unresolved fold-pair orbit-state with phase, path, tape, and substrate relations.
Measurement is only allowed at the CollapseBoundary, after invariant extraction.
No scalar candidate comparison. No verify(x). No AUC. No "which candidate wins."

---

## 1. Design Axioms

| # | Axiom |
|---|---|
| A1 | No scalar candidate is the primitive object. The primitive is the orbit. |
| A2 | The fold pair {d, N-d} must remain unresolved throughout evolution. |
| A3 | Public verify(x) cannot be the branch selector -- it is fold-even. |
| A4 | Evolution precedes measurement. No scoring before the CollapseBoundary. |
| A5 | Phase/path relations are first-class state, not derived from scalar comparison. |
| A6 | Tape restoration is hygiene, not substrate dynamics by itself. |
| A7 | Fold-even components must cancel before any interpretation of residue. |
| A8 | Fold-odd residue must be predeclared as an observable before measurement. |
| A9 | The CollapseBoundary is explicit, delayed, and documented. |
| A10 | Any verifier-first, AUC-first, or candidate-scoring design is invalid. |
| A11 | The substrate is physical (cache, PDN, thermal, DRAM, branch state), not a C register. |
| A12 | The answer is measured, not searched. |

---

## 2. State Objects

### FoldPair
- **Purpose:** Coupled representation of both fold branches {d+, d-} where d+ = d, d- = N-d. The public data is identical for both. The FoldPair carries BOTH without choosing.
- **Allowed contents:** (a, N-a) as symmetric values. Internal relation: unresolved.
- **Forbidden contents:** Label "true" or "false." Choice of which branch is d. Any scalar d.
- **Collapse risk:** Selecting one branch before PhaseCancel. Labeling one as candidate_0 and the other as candidate_1 for scoring purposes.
- **Phenom II carrier:** A pair of tape regions, a pair of cache lines, a pair of core-local accumulators.

### OrbitState
- **Purpose:** The full unresolved state of the system. Contains FoldPair + accumulated phase/path/tape/substrate relations.
- **Allowed contents:** FoldPair, PhaseRelation, PathHistory, TapeResidue, SubstrateMemory.
- **Forbidden contents:** Scalar x. Scalar d. Binary winner. AUC score.
- **Collapse risk:** Reducing to "which candidate." Computing c0 vs c1 statistics.
- **Phenom II carrier:** The full hardware state of sender+receiver cores + shared uncore during a coupled walk.

### PhaseRelation
- **Purpose:** The relative phase between the two fold branches. Encodes the antisymmetric component (sin channel) that cancels in public cosine but may survive in physical observables.
- **Allowed contents:** Complex-valued relation z = d+ / d- (normalized). Phase angle theta = arg(z).
- **Forbidden contents:** Binary orientation bit. Scalar d.
- **Collapse risk:** Converting phase to sign(d - N/2) before invariant extraction.
- **Phenom II carrier:** PDN current phase difference between two sender cores. Cache coherence timing asymmetry. TSC offset between cores during coupled walk.

### PathHistory
- **Purpose:** The accumulated trajectory of intermediate state during coupled evolution. Records what happened, not what the answer is.
- **Allowed contents:** Sequence of (step, branch, operator, tape delta, substrate delta).
- **Forbidden contents:** Cumulative score. Best candidate. Convergence metric.
- **Collapse risk:** Converting history to a scalar feature for classification.
- **Phenom II carrier:** Branch predictor training state. Cache LRU history. DRAM row-buffer access pattern.

### TapeResidue
- **Purpose:** The physical state of the tape buffer after architectural restoration. May retain analog information even though SHA-256 matches.
- **Allowed contents:** Physical measurement of tape bytes after XOR-restore (timing, voltage, thermal). NOT architectural byte values.
- **Forbidden contents:** Architectural tape contents (guaranteed identical after XOR restore).
- **Collapse risk:** Reading tape bytes as if they carry orientation. They don't -- SHA verifies they're identical.
- **Phenom II carrier:** DRAM cell retention time variation. Memory controller write-recovery timing. Row-buffer precharge latency after specific XOR patterns.

### SubstrateMemory
- **Purpose:** Physical degrees of freedom outside architectural state that may retain path-dependent information.
- **Allowed contents:** Thermal map, PDN voltage droop history, cache coherence directory state, TLB shootdown patterns, interrupt timing.
- **Forbidden contents:** Any architectural register or memory value visible to `mov` or `load`.
- **Collapse risk:** Measuring an architectural observable and calling it substrate.
- **Phenom II carrier:** k10temp thermal diode. APERF/MPERF ratio. Ring-oscillator period. Cache miss latency histogram.

### CatalyticField
- **Purpose:** The combined field in which the FoldPair evolves. Represents the tape + substrate as a catalytic medium.
- **Allowed contents:** Tape XOR mask stream. Substrate perturbation schedule. Phase reference signal.
- **Forbidden contents:** Hidden d. Candidate label. Verification oracle.
- **Collapse risk:** Making the field depend on which branch is "true."
- **Phenom II carrier:** alu_burst drive tone. PDN modulation waveform. Cache pressure pattern.

### OddResidue
- **Purpose:** What remains after fold-even cancellation. If non-null, carries orientation information.
- **Allowed contents:** A physical observable whose sign is antisymmetric under fold (d <-> N-d).
- **Forbidden contents:** Noise interpreted as signal without null controls.
- **Collapse risk:** Claiming any nonzero measurement is the odd residue. Must survive same-orbit and label-swap nulls.
- **Phenom II carrier:** PDN lock-in Q-channel magnitude that flips sign under lane-swap. Thermal asymmetry between two core-local walks.

### Invariant
- **Purpose:** The global quantity extracted from OddResidue. The answer -- but measured, not computed.
- **Allowed contents:** sign(OddResidue) if OddResidue is confirmed non-null under controls.
- **Forbidden contents:** Scalar d. Candidate label. AUC score.
- **Collapse risk:** Calling a candidate-value signal an orientation invariant.

### CollapseBoundary
- **Purpose:** The explicit point where the orbit resolves. Only after all evolution, cancellation, and residue extraction.
- **Allowed:** Invariant extraction. Final measurement. Logging.
- **Forbidden:** Any operation after the boundary. Any re-interpretation. Any post-hoc selection.

---

## 3. Non-Collapse Syntax (Draft)

```
-- Declaration
let orbit = FoldPair{ plus:  a, minus: N-a, state: unresolved }
let state = OrbitState{
    fold:    orbit,
    phase:   PhaseRelation(orbit, reference=public_k),
    path:    PathHistory::empty(),
    tape:    TapeResidue::empty(),
    sub:     SubstrateMemory::empty()
}
let field = CatalyticField{
    mask:    XOR_stream(seed=public_hash(orbit)),
    tone:    DriveTone(freq=200, phase=0),
    probe:   SubstrateProbe{ readout: LockIn, target: tone_freq }
}

-- Evolution (coupled, no branch selection)
state = CatalyticEvolution(state, field)
-- Applies operator to BOTH branches of orbit.
-- Accumulates phase deltas into PhaseRelation.
-- Records path deltas into PathHistory.
-- Updates TapeResidue (XOR-encode, compute, XOR-decode, SHA-verify).
-- Perturbs SubstrateMemory (PDN drive, cache pressure, thermal load).
-- NO branch is selected. NO verify(x) is called.

-- Cancellation
state = PhaseCancel(state, target = fold_even)
-- Destructive interference of the common-mode cosine component.
-- If the orbit contains ONLY fold-even relations, result is null.
-- If the orbit contains fold-odd relations, they survive as OddResidue.

-- Residue extraction
let residue = OddResidue(state)
-- Null if no fold-odd channel exists.
-- Non-null if a physical observable survives cancellation with
-- antisymmetry under fold (sign flips when plus/minus swap).

-- Invariant extraction
let invariant = InvariantExtract(residue)
-- Measurement of the global invariant from the residue.
-- Not a scalar score. Not AUC. Not classification.
-- Returns: sign(residue) if null-controlled, or NULL.

-- Collapse
let answer = CollapseBoundary(invariant)
-- Only here does the orbit resolve.
-- If invariant is NULL: no orientation information survived.
-- If invariant is non-NULL: orientation is the sign of the invariant.

-- ILLEGAL OPERATIONS (must reject)
-- let x = candidate_0          -- collapse
-- if verify(x): ...            -- verifier as branch
-- let auc = score(c0, c1)      -- AUC before invariant
-- let d = min(a, N-a)          -- candidate value as orientation
-- phase = 0 if c0 else pi      -- label encoding
```

---

## 4. Evolution Rules

| # | Rule |
|---|---|
| E1 | CatalyticEvolution operates on OrbitState, not on scalar candidates. |
| E2 | Both branches evolve simultaneously. No branch is privileged. |
| E3 | PathHistory is append-only. Never scored during evolution. |
| E4 | TapeResidue is updated via reversible XOR: encode op, compute, decode op, verify SHA. |
| E5 | SubstrateMemory is updated via physical perturbation (PDN drive, cache pressure, thermal load). |
| E6 | PhaseRelation is updated from the relative phase between branch walk intermediates. |
| E7 | No verify(x) call at any evolution step. |
| E8 | No branch selection at any evolution step. |
| E9 | The field (mask, tone, probe) is derived from public orbit data only. |
| E10 | Evolution terminates when the number of steps reaches a predeclared limit or the probe saturates -- NOT when verify(x) returns true. |

---

## 5. Cancellation Rules

| # | Rule |
|---|---|
| K1 | PhaseCancel targets the fold-even subspace. |
| K2 | Fold-even = any observable that is invariant under d <-> N-d. |
| K3 | Fold-odd = any observable that is antisymmetric under d <-> N-d. |
| K4 | Cancellation is accomplished by subtracting the two branch responses for the same observable. |
| K5 | If the observable is purely fold-even (cosine), cancellation yields zero. |
| K6 | If the observable contains a fold-odd component, cancellation yields 2 * odd_component. |
| K7 | Cancellation must be verified against same-orbit control (both branches set to a). |
| K8 | Noise in the cancellation residual must be calibrated via dummy-orbit control (both branches set to 42). |
| K9 | No cancellation is valid unless both branch responses are measured under matched conditions. |
| K10 | Cancellation failure (null controls fail) invalidates the residue. |

---

## 6. Measurement Rules

| # | Rule |
|---|---|
| M1 | Measurement is only permitted at the CollapseBoundary. |
| M2 | No AUC, no classification, no "which candidate" before the boundary. |
| M3 | The observable must be predeclared before the run. |
| M4 | The measurement must be logged with full state: orbit, phase, path, tape hash, substrate reading. |
| M5 | InvariantExtract returns NULL if OddResidue is null or fails controls. |
| M6 | InvariantExtract returns sign(OddResidue) only if same-orbit, dummy-orbit, label-swap, and shuffle controls all pass. |
| M7 | A measurement that produces a scalar d without surviving controls is invalid. |
| M8 | Replay must produce the identical invariant for the same seed. |

---

## 7. Illegal Collapse Operations

| # | Operation | Why Invalid |
|---|---|---|
| IC1 | `verify(x)` as first operation | Fold-even. Cannot select branch. |
| IC2 | `for x in 0..N: if verify(x): return x` | Forward scan. |
| IC3 | `c0_label = "candidate_0"; c1_label = "candidate_1"` | Early branch labeling. |
| IC4 | `phase = 0 if label==c0 else pi` | Label-based phase encoding. |
| IC5 | `auc = roc_auc_score(labels, scores)` | AUC before invariant. |
| IC6 | `return min(d, N-d)` | Candidate value as orientation. |
| IC7 | `if sha_match: claim substrate_works` | SHA alone is not substrate. |
| IC8 | `d in [1, N/2)` (domain restriction) | Collapses orbit to candidate value. |
| IC9 | `best_seed = argmax(auc over seeds)` | Post-hoc seed selection. |
| IC10 | `measure(c0) - measure(c1)` without same-orbit null | Noise as signal. |
| IC11 | `if residue != 0: claim orientation_found` | Without label-swap control. |
| IC12 | `let x = f(x) where f uses verify` | Verifier-based iteration. |

---

## 8. Phenom II Physical Primitive Mapping

| Primitive | Possible Role | Collapse Risk | Can Carry Fold-Odd? |
|---|---|---|---|
| L3 cache line state | SubstrateMemory: LRU history after coupled walk | Reading cache tags as architectural | UNLIKELY -- cache is coherent |
| Core-local timing (TSC delta) | PhaseRelation: timing asymmetry between two sender cores | Interpreting jitter as signal | POSSIBLE -- T300 showed PDN timing coupling |
| PDN common-mode path | CatalyticField: alu_burst drive reaching receiver via power rail | Measuring DC level instead of lock-in | PROVEN LIVE (T300, SNR 13-38) |
| Thermal drift (k10temp) | SubstrateMemory: temperature difference between core-local walks | Confusing ambient drift with signal | UNLIKELY -- thermal time constant too slow |
| Branch predictor state | PathHistory: path-dependent prediction after walk sequence | Cannot read predictor state directly on Phenom II | UNKNOWN -- not directly measurable |
| DRAM row-buffer timing | TapeResidue: row activation latency after specific XOR patterns | DRAM controller hides row state | UNLIKELY -- Phenom II memory controller opaque |
| APERF/MPERF ratio | SubstrateMemory: effective frequency during coupled walk | P-state pinned during slots | UNLIKELY -- P-state fixed |
| Ring-oscillator period | SubstrateMemory: victim core timing under sender PDN load | Scalar averaging instead of lock-in | PROVEN LIVE (T300 preflight) |
| Reversible tape buffer | TapeResidue: XOR-encoded intermediate state, SHA-restored | Reading post-restore bytes (they're identical) | NO -- SHA guarantees identity |
| Thread/core phase relation | PhaseRelation: relative scheduling phase of sender threads | OS scheduler adds noise | POSSIBLE -- with isolcpus + FIFO scheduling |

---

## 9. Minimum Non-Collapse Experiment Shape

**L4A_NON_COLLAPSE_MECHANISM_SCREEN**

Goal: Screen for a predeclared fold-odd residue while preserving the fold pair as orbit-state.

Protocol:
1. Construct OrbitState from public oracle: FoldPair(a, N-a).
2. Initialize CatalyticField: lock-in probe at 200 Hz, PDN drive on sender core 4.
3. Evolve: run CatalyticEvolution with coupled branch walks. Both branches drive the same PDN tone but with opposite phase encoding derived from their walk operands (a*k_j vs (N-a)*k_j). NO branch selection. NO verify(x).
4. Cancel: PhaseCancel by computing lock-in Q-channel difference between the two branch response windows.
5. Extract: OddResidue = Q_diff_sign. Invariant = sign(OddResidue) iff nulls pass.
6. CollapseBoundary: report invariant.

Controls:
- Same-orbit: FoldPair(a, a). Q_diff should be zero.
- Dummy-orbit: FoldPair(42, 42). Q_diff should be zero.
- Label-swap: swap which branch uses which phase encoding. Q_diff sign should flip.
- Shuffle null: randomize branch assignment. AUC should be 0.5.
- Wrong-restore: SHA mismatch. Run invalid.
- Replay: same seed, same residue.

Claim ceiling: L3 if residue survives nulls. L4 only if sign(residue) correlates with hidden d under no-smuggle controls.

**This experiment does NOT:**
- Call verify(x).
- Score candidates.
- Run forward scan.
- Recover d.
- Claim orientation.

**This experiment DOES:**
- Preserve the fold pair as orbit-state.
- Phase-cancel before measurement.
- Measure only at CollapseBoundary.

---

## 10. Claim Ledger

| Level | Stage |
|---|---|
| L0 | Architecture definition (this document) |
| L1 | Syntax/spec complete with all state objects and rules |
| L2 | Non-collapse simulator (Python reference model, no hardware) |
| L3 | Phenom II physical primitive screen: fold-odd residue survives nulls |
| L4 | Orientation recovery from residue under no-smuggle controls |
| L5 | Multi-seed, multi-session, multi-n |
| L6 | Independent reproduction |

---

## 11. Median-Basin Guardrails

For any future agent working on this architecture:

| If the work... | Then... |
|---|---|
| Becomes verify(x) | REJECT. Fold-even. Cannot select. |
| Becomes AUC-first | DOWNGRADE. Classification, not measurement. |
| Becomes "which candidate wins" | REJECT. Early collapse. |
| Treats code execution as substrate | AUDIT. Substrate is physical. |
| Cannot name the non-collapse state object | REJECT. No architecture. |
| Uses scalar x as primitive | REJECT. Orbit is primitive. |
| Recovers a = min(d,N-d) | REJECT. Candidate value, not orientation. |
| Calls SHA tape = substrate | DOWNGRADE. Hygiene, not dynamics. |

---

## 12. Roadmap Update

```
[>] Exp 50 next frontier: NON_COLLAPSE_SUBSTRATE_ARCHITECTURE.
    Preserve orbit-state through coupled evolution.
    Phase-cancel before measurement.
    No scalar verification before invariant extraction.
```
