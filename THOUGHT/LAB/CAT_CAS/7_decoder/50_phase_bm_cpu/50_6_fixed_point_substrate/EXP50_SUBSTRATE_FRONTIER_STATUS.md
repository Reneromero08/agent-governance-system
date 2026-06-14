# Exp 50 Substrate Frontier Status Report

**Date:** 2026-06-14.
**Status:** EXP50_SUBSTRATE_FRONTIER_STATUS: L2_PASS__L3_PASS__L4_BLOCKED_BY_FORWARD_SCAN_AND_FOLD_SYMMETRY

---

## Executive Summary

The Exp 50 substrate frontier tested whether a catalytic tape lifecycle on Phenom II silicon
could support fixed-point recovery of the hidden secret `d` from the Exp 50.14 public
cosine-only oracle.

L2 cleared: the catalytic tape lifecycle (SHA-256 before → XOR mutate → compute →
XOR restore → SHA-256 verify) works on Phenom II hardware.

L3 cleared: a nontrivial contraction map converges to a stable fixed point within
the tape lifecycle loop.

L4 is blocked: the naive map `f(x) = x if verify(x) else (x+1) mod N` is ordinary
forward scan. The verifier `verify(x)` is fold-even and accepts both `d` and `N-d`.
The charter's restricted-domain wording collapses `d` recovery into public
candidate-value recovery. Full `d` recovery requires orientation information not
present in `verify(x)`. The frontier is now a hardware/paradigm boundary, not
another public-verify C loop.

---

## Ladder Table

| Gate | Commit | Status | What It Proves | What It Does Not Prove |
|---|---|---|---|---|
| L2 | `db852286` | **PASS** | Tape lifecycle works on Phenom II. SHA restore, deterministic replay, wrong-mask negative. 50/50. | Fixed-point convergence. Orientation recovery. Substrate crossing. |
| L3 | `c394e30b` | **PASS** | Nontrivial contraction map `f(x)=floor((x+42)/2)` converges in 2-9 iterations. 90/90. Tape hash verified each cycle. | Tape-resident state (state in register). Exp 50.14 oracle recovery. Orientation recovery. |
| L4 | `a2ad5243` | **BLOCKED** | N/A -- no code written. Design audit found three blockers. | N/A |

---

## L2 Summary

**Proved:** A C binary on Phenom II can allocate a dirty tape buffer, fill it
deterministically from a seeded RNG, compute SHA-256, apply a reversible XOR mask,
execute a compute phase, reverse the compute and XOR mask, and verify SHA-256
restoration. Across 50 trials with 4 explicit controls (normal restore,
wrong-mask negative, identity no-op, deterministic replay), all 50/50 pass.
Per-trial CSV log produced. Claim L2.

**Did not prove:** Fixed-point convergence. Orientation recovery. Substrate
crossing. Tape-resident state evolution (state in register only).

---

## L3 Summary

**Proved:** A catalytic loop using the tape lifecycle can converge to a stable
fixed point on a nontrivial contraction map. Map: `f(x) = floor((x+42)/2)`.
Convergence takes 2-9 iterations depending on start state (e.g., 10→26→34→38→40→41→41
in 6 steps; 250→146→94→68→55→48→45→43→42→42 in 9 steps). Tape hash verified each
iteration (90/90). Forward scan baseline finds a fixed point in O(N)=42 iterations;
catalytic loop in O(log N). All 6 controls pass. Claim L3.

**Did not prove:** Tape-resident state (state lives in a C register, tape is
borrowed workspace per cycle). Exp 50.14 oracle integration. Orientation recovery.

---

## L4 Blocker Summary

L4 is blocked by three independent reasons. No code was written.

**Blocker A: Forward scan.** `f(x) = x if verify(x) else (x+1) mod N` enumerates
candidates sequentially until `verify(x)` returns true. This is O(N) linear search,
regardless of whether it runs inside a SHA-verified tape lifecycle. The tape
lifecycle is orthogonal to the search -- wrapping it in SHA is ceremonial.

**Blocker B: Fold-even verify.** `verify(x) = score(x) > M/4` with
`score(x) = sum_i b_i * cos(2*pi*k_i*x/N)`. Cosine is even, so
`verify(d) == verify(N-d)` always. The accepting set is the full orbit {d, N-d}.
A scan restricted to [1, N/2) finds `a = min(d, N-d)`, not `d`. The verifier
carries no orientation information that any loop -- catalytic or otherwise --
can read.

**Blocker C: Target collapse.** The charter restricts `d` to `[1, N/2)`. Under
that restriction, `d = a` (the public fold magnitude), and "recovering d" is
recovering a value already obtainable at AUC 1.000 from public readout. The actual
construction (`construction.py` line 71-77) draws `d` from the full `[1, N)`,
making `d` recovery equivalent to orientation recovery. Phase 6 proved orientation
is information-theoretically absent from the public cosine channel under no-smuggle
constraints.

---

## Corrected Frontier

The remaining meaningful target is not candidate-value recovery (`a` is public).
It is orientation recovery -- determining which half of the fold pair contains the
hidden `d`. This requires information or dynamics not present in `verify(x)`.

The Exp 49.14 handoff identified the untested lever: "on a reversible/CTC
fixed-point substrate, fix(f) is found in poly." The substrate mechanism
that could access orientation without reading `d` from the oracle has not been
defined in implementable form. The frontier is now the DEFINITION of such a
mechanism, not another iteration of public-verify C loops.

---

## Relation to Phase 6

Phase 6 mapped the forward public-data boundary: all no-smuggle tracks under the
Exp 50.14 oracle produce null, weak candidate-value-only signals, or fail
orientation recovery. Tracks A (3 hardware architectures), D, F, B, and the
E5/E1 oracle audit all converged on this boundary.

Exp 50 L2/L3 tested whether the substrate lifecycle mechanics exist on Phenom II.
They do. L4 tested whether public `verify(x)` iteration can cross the boundary.
It cannot -- not because hardware failed, but because `verify(x)` is structurally
fold-even and the sequential map is forward scan.

The next question is not "can we find a better verifier loop?" It is "can we
define a substrate mechanism that accesses orientation without scanning
verify(x) sequentially?" That is the paradigm boundary.

---

## Claim Ledger

| Claim Level | Status |
|---|---|
| L0 (charter) | Achieved |
| L1 (compile/run) | Achieved |
| L2 (tape lifecycle) | **Achieved** (`db852286`) |
| L3 (fixed-point convergence) | **Achieved** (`c394e30b`) |
| L4 (Exp 50.14 oracle recovery) | **Blocked** (design audit: forward scan + fold symmetry + target collapse) |
| L5 (multi-seed/session) | Not attempted |
| L6 (independent reproduction) | Not attempted |

---

## Open Frontier Questions

1. Is there any implementable substrate mechanism that is not public verify iteration?
2. Can a tape-resident state be coupled to physical relaxation rather than sequential C control?
3. Can any measurable hardware degree of freedom carry fold-odd orientation under no-smuggle controls?
4. Is this now outside classical Phenom C and into analog/reversible/CTC-theory territory?

---

## Roadmap Update

```
[!] Exp 50 substrate frontier: L2/L3 cleared; L4 blocked by forward scan,
    fold symmetry, and target collapse. Next work requires a new substrate
    mechanism, not public verify iteration.
```
