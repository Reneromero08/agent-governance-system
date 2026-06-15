# Exp 50 Substrate Frontier Charter

**Date:** 2026-06-14.
**Status:** CHARTER_READY. No code built. No hardware run.
**Claim ceiling:** L4 for first valid result. L6 only after independent reproduction.

---

## 1. Purpose

Test whether a catalytic / reversible / zero-Landauer substrate process on real Phenom II
silicon can preserve path-dependent information through architectural restoration in a way
that forward (non-catalytic) public-data measurement cannot.

Phase 6 proved that forward measurement on a classical Phenom II does not recover the
hidden orientation from the Exp 50.14 public cosine-only oracle. This charter defines
the experiment that tests whether the CATALYTIC substrate -- the "relax, don't construct"
lever that Exp 49 identified as the untested frontier -- changes that answer.

---

## 2. Canonical Location

| Property | Value |
|---|---|
| Experiment name | Exp 50 (Phase BM CPU / Bare-Metal CPU Substrate Push) |
| Directory | `7_decoder/50_phase_bm_cpu/` |
| Substrate frontier subdirectory | `50_6_fixed_point_substrate/` |
| Relation to Exp 49 | Exp 49 (Decoder) is the theory terminus. It proved the forward wall and identified the substrate as the untested lever. Exp 50 inherits the 50.14 public fixed-point map as the target. |
| Relation to Phase 6 | Phase 6A (12_chiral_lane_frontier/) is the boundary-mapping campaign within Exp 50. It built the detector spine, no-smuggle gate, route selection, and measured the forward boundary across 3 hardware + 2 reference architectures. Phase 6A is COMPLETE. Phase 6B (14_noncollapse_frontier) is ACTIVE. |
| Relation to Exp 44 | Exp 44 is Phase Atom (44_phase_atom/). Not the substrate frontier. The stale "Exp 44 Phase 6" alias in historical Exp 49 reports refers to Exp 50. Resolved in MASTER_REPORT.md line 587 and CAT_CAS_LAB_STATE_AUDIT.md. |

---

## 3. Phase 6 Handoff

**What Phase 6 proved:**
- The Exp 50.14 public cosine-only oracle is fold-symmetric at the implementation level (E5/E1).
- The Phenom II PDN coupling exists (T300 slot2, SNR 13-38) and the alu_burst + lock-in measurement path is live.
- No no-smuggle forward measurement on Phenom II recovered the orientation bit (Track A, 3 architectures, 12/12 controls).
- Mathematical reference models (Tracks D, F) found either null or weak seed-dependent candidate-value hints with orientation at chance.
- Manual label encoding (Tracks C, E, A-phase-encoding) was rejected at audit.

**Exact boundary claim:**
"All no-smuggle tracks executed or file-audited under the current Exp 50.14 public-data oracle either produce null, weak candidate-value-only signals, or fail orientation recovery."

**What Phase 6 did NOT prove:**
- That the substrate cannot cross the boundary. The substrate was not tested.
- That PDN coupling is absent. It exists for alu_burst + lock-in (T300).
- That candidate-value signals are physically impossible. They are below Phenom II resolution for the tested workload classes.

**Why the substrate frontier is the next question:**
The forward boundary is measured. The untested lever is the catalytic substrate -- "relax, don't construct" -- identified by Exp 49.14 as the mechanism that collapses `2^n` search to poly fixed-point iteration on a reversible/CTC substrate.

---

## 4. Substrate Question

**Can a catalytic / reversible substrate process preserve path-dependent information through architectural restoration while returning the public tape to a fold-symmetric architectural state?**

In operational terms: can the Phenom II, running a reversible compute loop with SHA-256 verified tape restoration, reach the unique fixed point `d` of the public map `f(x) = x if verify(x) else (x+1) mod N` in polynomial (loop) time, where a forward classical machine requires `O(N) = 2^n` candidate evaluations?

---

## 5. Target Problem

| Property | Value |
|---|---|
| Public input | `{(k_i, b_i)}` with `E[b_i] = cos(2*pi*k_i*d/N)`, `N = 2^n` |
| Hidden target | `d` in `[1, N/2)` (the true secret) |
| Candidate pair | `a = min(d, N-d)` (fold magnitude), `N-a` (fold mirror) |
| Public verifier | `verify(x) = score(x) > M/4` where `score(x) = sum_i b_i * cos(2*pi*k_i*x/N)` |
| Fixed-point map | `f(x) = x if verify(x) else (x+1) mod N` |
| `fix(f) = d` means | The unique fixed point of `f` in `[1, N/2)` equals the hidden secret `d` |
| Success | The catalytic loop terminates at `x = d` with SHA-256 tape restored, loop cost ~poly(n), candidate blinding maintained |
| Failure | Loop cost = `O(2^n)` (forward scan), or tape not restored, or recovered value ≠ `d`, or hidden `d` entered the runtime path |

---

## 6. Hardware Requirements

- Phenom II X6 1090T, Debian 13, Linux 6.12.86.
- Isolated cores 2,3,4,5 (isolcpus). OS on cores 0,1.
- P-state pinned at 1600 MHz during slots.
- constant_tsc + nonstop_tsc.
- k10temp veto at 68 C.
- Userspace only. No firmware writes. No MSR writes (reads acceptable for P-state verification).
- Route 4:5 as adjudication carrier (from Track I T300 data). Route read from config file, not hardcoded.
- C toolchain: `gcc -O2 -pthread -march=amdfam10 -Wall -Wextra -lm`.
- No Rust on Phenom hardware path.
- Tape: SHA-256 hash before and after every catalytic loop. Restoration failure = run invalid.

---

## 7. Software Requirements

| Requirement | Detail |
|---|---|
| Catalytic compute path | Reversible operations on dirty tape. XOR-encode, compute, XOR-decode, verify. |
| Tape lifecycle | `record_initial_hash → XOR_mutate → compute → XOR_restore → verify_final_hash == initial_hash` |
| SHA-256 | Verify before/after every trial. Mismatch = trial invalid. |
| No hidden d in runtime | Runtime reads only `(k, b, N, candidate_0, candidate_1)`. Hidden d used for oracle generation and offline scoring only. |
| No true/false labels in runtime | Runtime uses candidate_0 / candidate_1 only. Offline scorer maps to true/false after run. |
| Deterministic replay | Fixed seed per trial. Replay produces identical tape sequence. |
| Per-trial logs | CSV/JSON with: trial_id, n, seed, mode, loop_iterations, tape_restored, recovered_x, true_d, orientation, run_time_s |
| Control modes | public, same_candidate, dummy_candidate, no_op_identity, forward_only, reverse_only, hidden_positive |

---

## 8. Acceptance Criteria

| Level | Criterion |
|---|---|
| L0 | Charter written. No code. |
| L1 | C code compiles and runs on Phenom II without segfault. |
| L2 | Catalytic loop demonstrates tape restoration (SHA-256 in == out) for at least one trial. |
| L3 | Catalytic loop produces stable fixed-point behavior (converges to same value across seeds for same instance). |
| L4 | Recovers hidden `d` under no-smuggle controls with SHA-restored tape and poly loop cost. All required controls pass. |
| L5 | Repeated across ≥5 seeds, ≥2 n values (n=8, n=10), and across reboot sessions. |
| L6 | Independently reproduced on a second Phenom II or independent C implementation. |

**No result may claim L4 unless tape restoration is verified, controls pass, and loop cost is poly(n) with clear evidence.**

---

## 9. Rejection Criteria

A crossing claim is INVALID if any of the following is true:

- Hidden `d` read by the runtime path.
- True/false labels read by the runtime path.
- Manual phase/sign/orientation assigned by candidate label.
- Candidate label leakage into the runtime.
- Post-hoc scoring without a predeclared observable.
- Tape not restored (SHA-256 mismatch).
- Exponential search disguised as a catalytic loop.
- No null controls run.
- No SHA tape verification.
- No deterministic replay.
- Claim level inflation (L6 claimed from a single hardware run).

---

## 10. Required Controls

| Control | Purpose |
|---|---|
| Same-candidate null | Both candidates set to `a`. Loop should produce identical behavior. |
| Dummy-candidate null | Both candidates set to 42. Loop should produce identical behavior. |
| Label-swap null | Swap which candidate is labeled c0 vs c1. Result should track value, not label. |
| Shuffle null95 | Shuffle orientation labels offline. AUC must stay below null95. |
| No-op identity loop | Run loop with `f(x) = x`. Fixed point should be trivially reached. |
| Forward-only baseline | Run forward scan (non-catalytic) as cost baseline. Cost should be O(2^n). |
| Reverse-only baseline | Run reverse scan as matched baseline. |
| Tape-restore verification | SHA-256 before/after every loop invocation. |
| Hidden positive calibration | Inject known `d`-correlated bias into loop dynamics. Detector must register it. |
| Thermal/power null | Verify observed effect is not thermal drift. |
| Reboot/reseed reproducibility | Rerun after reboot. Same seed → same result. |

---

## 11. Core Experiment Design (minimum)

```
1. Initialize public oracle instance:
   - n = 8 or 10, N = 2^n.
   - Generate d in [1, N/2) (hidden, for offline scoring only).
   - Generate (k_i, b_i) via coset_samples (public).
   - Compute a = min(d, N-d), Na = (N-a).

2. Build candidate pair {candidate_0 = a, candidate_1 = Na}.
   Runtime sees only candidate_0 and candidate_1.

3. Run catalytic substrate loop for candidate_0:
   - Initialize dirty tape. Record SHA-256.
   - XOR-encode candidate into tape.
   - Run iteration: x_{t+1} = f(x_t) on the encoded tape.
   - Detect fixed point (x_{t+1} == x_t or verify(x_t) == True).
   - XOR-decode tape. Verify SHA-256 matches initial.
   - Record loop iterations, recovered x, tape restoration status.

4. Run catalytic substrate loop for candidate_1 (same instance, same tape lifecycle).

5. Offline scorer:
   - Map candidate_0 / candidate_1 to true / false using hidden d.
   - Compare recovered x_t against d for true candidate.
   - Score: true candidate reaches d, false candidate does not (or reaches Na).

6. Run all controls:
   - Same-candidate (both = a).
   - Dummy-candidate (both = 42).
   - Label-swap.
   - No-op identity.
   - Forward scan baseline.
   - Hidden positive.

7. Statistics:
   - Compute recovery rate, loop cost, orientation AUC, null95.
   - Multi-seed sweep (≥5 seeds).
   - Multi-n (n=8, n=10).
```

---

## 12. Relation to Phase 6 Results

| Phase 6 Result | Constraint on This Charter |
|---|---|
| E5/E1: Oracle fold-symmetric | The public data is identical for d and N-d. Any substrate crossing must explain how path information survives the oracle's fold without reading hidden d. |
| Track Z: Schedule invariant | The substrate loop schedule must be invariant under d ↔ N-d. |
| Track 0: Binary transfer function | If the substrate needs an injected odd-lane to cross, the threshold is epsilon > 0 with k=1 access. |
| Track B: I/Q candidate-value only | Q channel separates candidates but not orientation. The substrate must produce a signal that the I/Q receiver alone could not. |
| Track I: Route 4:5 confirmed | Use route 4:5 for Phenom II measurements. |
| Track A: Forward negative | Forward measurement on classical Phenom does not recover orientation. This is the baseline the substrate must beat. |
| Tracks D/F: Reference null/weak | Mathematical commutator and HW accumulation models do not produce robust candidate-value signals. The substrate mechanism must be genuinely different. |
| Tracks C/E: Label-smuggle rejected | No manual assignment of operation/direction/phase to candidate labels. |

---

## 13. Claim Ceiling

- **First valid result:** L4 maximum. Must survive all required controls, demonstrate tape restoration, show poly loop cost, and recover d under no-smuggle conditions.
- **L5:** Requires repeated sessions (≥5 seeds, ≥2 n values, reboot reproducibility).
- **L6:** Requires independent reproduction on a second machine or independent implementation.
- **No metaphysical claims from one hardware run.** "The substrate crosses the boundary" is a physical claim about Phenom II silicon, not an ontological claim.

---

## 14. Current Blockers

| Blocker | Status |
|---|---|
| Hardware readiness | Phenom II is operational (confirmed by T300, Track A runs). |
| Software path readiness | **NOT BUILT.** No catalytic substrate loop exists yet. |
| Identifier cleanup | Current Phase 6 files are clean. Exp 49 historical cleanup deferred. |
| Topology sweep | Route 4:5 confirmed from T300. Full 12-route sweep deferred. |
| Substrate model | **NOT IMPLEMENTED.** The fixed-point iteration on a catalytic tape has not been coded for Phenom II. |

---

## 15. Recommended First Implementation Step

**Build a C program that demonstrates tape XOR-encode → compute → XOR-decode → SHA-256 verify on the Phenom II.**

This is the minimal catalytic lifecycle. No fixed-point search yet. No orientation recovery. Just prove the tape lifecycle (record → mutate → restore → verify) works in a C binary on the target hardware. This is the L2 gate before any substrate claim can be attempted.

---

## 16. Roadmap Update

Add to `PHASE6_CHIRAL_LANE_FRONTIER_ROADMAP_2.md`:

```
## Substrate Frontier (post-Phase 6)

- [ ] L2 gate: Catalytic tape lifecycle C binary on Phenom II.
- [ ] L3 gate: Fixed-point loop produces stable convergence.
- [ ] L4 gate: Recovers d under no-smuggle controls with SHA tape restore.
```
