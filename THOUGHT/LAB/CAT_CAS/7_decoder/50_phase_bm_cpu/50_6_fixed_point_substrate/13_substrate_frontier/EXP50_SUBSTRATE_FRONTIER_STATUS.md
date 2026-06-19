# Exp 50 Substrate Frontier Status

**Date:** 2026-06-18  
**Status:** HISTORICAL_WARMUP_CORRECTED__L4_PUBLIC_VERIFY_REJECTED  
**Authority:** This file records the pre-noncollapse substrate warmup. The active frontier is `../14_noncollapse_frontier/`.

---

## Executive state

The substrate-frontier branch established two useful engineering facts and one decisive rejection:

1. The XOR/SHA tape lifecycle is a valid **software hygiene mechanism** when its hardware proof packet is available.
2. The original integer contraction source did not establish the claimed unique fixed point: `floor((x+42)/2)` has fixed points `41` and `42`, while the committed baseline required `42`. The earlier `L3 PASS` classification is therefore invalidated pending a rerun of the corrected `signed_distance_halving_v2` source.
3. The proposed L4 map based on `verify(x)` is rejected because it is sequential forward scan over a fold-even predicate. It cannot recover the missing orientation channel.

Nothing in this directory is physical geometric-memory evidence. It is mechanical preparation and a recorded rejection boundary.

---

## Corrected ladder

| Gate | Current status | What is established | What is not established |
|---|---|---|---|
| L2 tape lifecycle | `PROVISIONAL_HARDWARE_EVIDENCE_RETAINED` | Reversible XOR mutation and byte-hash restoration were previously reported 50/50 with negative and replay controls. | Physical-state restoration, geometric memory, orientation, wall crossing. |
| L3 contraction warmup | `SOURCE_CORRECTED__RERUN_REQUIRED` | A corrected map now has one integer fixed point at `42` and the test requires convergence to that exact point. | The earlier 90/90 report is not accepted for the corrected source. |
| L4 public verifier route | `REJECTED` | `verify(d)==verify(N-d)` and the proposed update is forward enumeration. | No conclusion about every possible physical substrate or access model. |

The corrected L3 source is:

```text
fixed_point_convergence.c
map = 42 + (x - 42) / 2
expected unique integer fixed point = 42
verdict = L3_MECHANICAL_WARMUP_PASS/FAIL
```

Execution of that source is intentionally deferred to the final SSH verification batch.

---

## Why the previous L3 classification was invalid

For the old map:

```text
f(41) = floor((41+42)/2) = 41
f(42) = floor((42+42)/2) = 42
```

The loop counted convergence to either fixed point as success, but the forward baseline counted only `42` as success. This contradicted the claimed unique target and made the recorded aggregate unsuitable as evidence. Git history preserves the original source and report; current documents must not cite it as a passing gate.

---

## L4 rejection boundary

The original map

```text
f(x) = x if verify(x) else (x+1) mod N
```

is rejected for three independent reasons:

- it enumerates candidates sequentially;
- the cosine verifier is fold-even and accepts both members of `{d,N-d}`;
- restricting the domain to `[1,N/2)` changes the task into public fold-magnitude recovery.

The authoritative rejection is `EXP50_L4_GATE_DESIGN_AUDIT.md`. No future worker should revive the rejected scalar loop by wrapping it in tape restoration.

---

## Handoff to Phase 6B

The valid handoff is not `fix(f)=d`. It is:

```text
preserve the unresolved orbit
carry complex/relational state
record the path as geometric memory
project only at an explicit boundary
map software objects to physical evidence conservatively
```

See:

- `../14_noncollapse_frontier/doctrine/EXP50_NON_COLLAPSE_SUBSTRATE_ARCHITECTURE.md`
- `../14_noncollapse_frontier/holo_runtime/HOLO_SCHEMA.md`
- `../14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md`

---

## Claim ceiling

```text
L2: software/tape lifecycle hygiene
L3: corrected mechanical fixed-point warmup after rerun
L4: BLOCKED for public verify iteration
```

No orientation, physical restoration, physical geometric memory, or substrate crossing claim is authorized from this directory.
