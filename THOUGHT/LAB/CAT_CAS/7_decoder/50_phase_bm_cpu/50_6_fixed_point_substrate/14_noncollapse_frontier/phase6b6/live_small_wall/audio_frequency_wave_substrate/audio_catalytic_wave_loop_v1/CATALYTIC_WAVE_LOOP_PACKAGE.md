# Catalytic Recursive-Wave Loop Package

**Status:** `SOURCE_CANDIDATE__EXACT_R1_QUALIFICATION_PENDING`  
**Package:** `audio_catalytic_wave_loop_v1`  
**Parent result:** `AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED`  
**Source commit:** `65f656527a8bf63e6d44493154f3db15a8a99b8b`  
**Source Git blob SHA-1:** `1bf96eb8a95c89a82665a95a49d3bd722a14f7d4`  
**Source SHA-256:** must be computed and frozen from exact committed bytes during qualification  
**Operation:** bounded ordinary-software catalytic phase-carrier loop  
**Physical authority:** none

## 1. Purpose

This package tests the first complete software lifecycle in the audio lane:

```text
borrow complex phase carrier
-> displace it through the established complete-tree temporal trajectory
-> apply a predeclared matched carrier query
-> latch one complex relational observable outside the history
-> reverse the carrier operators in exact reverse order
-> recover the exact initial R1 tree through ancestry receipts
-> retain the latch while carrier and ancestry close
```

The result is not a spin, Ising energy, winner, candidate, or optimization score. It is a
query-bound complex relational observable copied before restoration.

## 2. Borrowed Carrier

The borrowed software carrier is the established deterministic R0 complex tape:

```text
tau_0[n] = A[n] * exp(i * phi[n])
```

It has nonzero amplitude at every sample. The carrier is ordinary complex128 software
state. It is not a physical audio, voltage, or silicon-phonon carrier.

The carrier must be measurably displaced before any restoration claim is admitted:

```text
||tau_forward - tau_0||_2 >= 1.0
```

An untouched carrier is a hard failure.

## 3. Forward Operator

For R1 trajectory states `T1`, `T2`, `T3`, rendered beams `B_k`, and frozen circular
shifts:

```text
s = [17, -29, 43]
```

one carrier step is:

```text
F_k(tau) = Roll_s[k](tau * B_k)
```

The full forward path is:

```text
tau_3 = F_3(F_2(F_1(tau_0)))
```

Phase multiplication and circular transport are each reversible. Their composition is
order-sensitive because transport and position-dependent phase multiplication do not
commute in general.

The R1 complete trees remain the operator objects. No scalar diagnostic generates a
carrier step.

## 4. Matched Carrier Query

The public default query is the established R0 hierarchy-A tree. It is selected before
the R1 trajectory and is not derived from the final state.

The matched query carrier is built from the same borrowed carrier:

```text
q_0 = tau_0 * B_query
q_3 = Roll_43(Roll_-29(Roll_17(q_0)))
```

It receives the same public transport schedule but none of the hidden trajectory phase
operators.

The predeclared relational readout is:

```text
z_q = normalized_complex_inner_product(tau_3, q_3)
```

The query operates on the actually displaced carrier. It does not reconstruct `T3`, read
an expected result, or feed `z_q` into native evolution.

A hierarchy-B query is the first wrong-query control. The exact and wrong query latches
must differ by at least:

```text
|z_A - z_B| >= 1e-6
```

This is a bounded query-sensitivity gate, not an optimization or uniqueness claim.

## 5. External Latch

Before any reverse operation, the package copies this immutable record:

```text
RelationalLatch {
    query_tree_digest
    final_tree_digest
    carrier_before_sha256
    carrier_displaced_sha256
    response_real
    response_imag
}
```

The latch contains no spin, energy, expected response, answer, candidate, winner, or
score field.

The latch canonical bytes and digest must remain unchanged through carrier restoration
and ancestry unwind. The final closure object must not retain trajectory states,
receipts, drives, step specifications, or the displaced carrier.

## 6. Carrier Inverse

One inverse step is:

```text
F_k^-1(tau) = Roll_-s[k](tau) * conjugate(B_k)
```

The correct inverse schedule is:

```text
tau_restored = F_1^-1(F_2^-1(F_3^-1(tau_3)))
```

The source implements this by iterating the forward `(state, shift)` pairs in reverse.

Required correct-restoration gate:

```text
max_n |tau_restored[n] - tau_0[n]| <= 1e-12
```

Required wrong-restoration gates:

```text
forward-order inverse error        >= 0.05
wrong-trajectory inverse error     >= 0.05
omitted-step inverse error         >= 0.05
no-restoration displacement        >= 0.05
```

## 7. Restoration Semantics

The complex128 carrier is a continuous software chart. Its frozen R2S restoration law is
an observable-state equivalence region:

```text
metric       max absolute complex sample error
tolerance    1e-12
```

Byte-exact identity is recorded but is not required. The pre- and post-carrier SHA-256
values may therefore differ despite accepted numerical equivalence.

This package must never describe a differing SHA-256 as byte restoration. Its allowed
language is:

```text
software phase-carrier equivalence restoration
```

Exact byte restoration remains required for the canonical R1 ancestry object:

```text
reverse receipt traversal must recover exact committed T0 canonical bytes
```

Qualification must explicitly audit whether this two-channel restoration law is coherent
with the CAT_CAS software claim. If byte-exact restoration of the phase carrier is judged
mandatory rather than equivalence-class restoration, return
`SOL_EXTRA_HIGH_ARCHITECTURE_REQUIRED`; do not silently weaken or relabel the claim.

## 8. Ancestry Unwind

The package reuses the complete established R1 trajectory and receipt validation.

After the carrier is restored:

```text
T3 -> T2 -> T1 -> T0
```

must close through exact receipt-bound predecessor extraction. The recovered T0 canonical
bytes and tree digest must equal the original committed T0 identity.

The closure object keeps only:

```text
latch
carrier before hash
carrier restored hash
byte-exact diagnostic
recovered T0 digest
forward displacement
restoration error
```

No temporal history is returned.

## 9. Three Recursions

This package is the first bounded conjunction of:

```text
depth recursion:
    phase signal inside phase signal

temporal recursion:
    complete phase tree inside each later complete tree

catalytic recursion:
    relational result survives while carrier and temporal ancestry are reversed
```

This is still ordinary software. It does not establish a physical carrier or hardware bit
replacement.

## 10. Candidate Source Tests

The source candidate contains eleven thin tests:

```text
nonzero borrowed-carrier displacement
correct carrier-equivalence restoration
exact T0 ancestry recovery
external latch survival
forward-order inverse rejection
wrong-trajectory inverse rejection
omitted-step inverse rejection
no-restoration rejection
wrong-query latch separation
no spin/energy/winner fields in latch
no temporal history fields in closure object
```

An API-compatible local model of the source produced `11 PASS / 0 FAIL` with approximate
diagnostics:

```text
forward displacement L2       73.1576613427
restore max error             4.74287484027e-16
wrong-order restore error     1.79941674031
wrong-trajectory error        0.959034213823
query-latch change            0.00529698627982
```

These are source-authoring diagnostics only. They were not produced by an independent
clean checkout against the exact committed R0 and R1 packages and are not frozen result
identities.

## 11. Required Qualification Expansion

Before promotion, the local agent must add at minimum:

```text
exact committed source SHA-256 and byte count
strict latch schema and canonical parser
strict shift-schedule or loop-contract schema
committed carrier-before, displaced, and restored fixtures
committed latch and closure receipts
manifest binding every source, parent, carrier, query, schedule, result, and hash
build, self-test, and verify-only modes
committed-byte recomputation
AST proof that scalar latch values cannot reach forward or inverse evolution
mutation tests for query, order, shift, tree, latch, carrier, and receipt custody
four independent reviews
normalized findings and claim adjudication
```

Binary carrier fixtures should use an explicitly frozen little-endian complex format or
another deterministic format. WAV may be provided as a visualization chart, but it is not
required to define the native software carrier.

## 12. No-Smuggle Law

The query tree, shift schedule, thresholds, restoration metric, and wrong-control minimums
must be frozen before exact result generation.

Forbidden inputs to the forward carrier path:

```text
latched response
expected response
spin
energy
winner
candidate
argmin or argmax
verification result
final answer
```

The latch may be computed only after the full displaced carrier exists.

The exact query may be public, but it cannot contain or be selected from the observed
latch value. Query A versus query B is a structural control, not a winner-selection loop.

## 13. Claim Law

After exact parent reproduction, committed-byte custody, restoration/adversary closure,
AST no-feedback proof, four independent reviews, and changed-path qualification, the
package may emit only:

```text
AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED
```

Meaning only:

```text
an ordinary-software complex carrier is nontrivially displaced by the established
complete-tree temporal phase trajectory; a prospectively declared carrier query produces
a latched complex relational observable; the correct reverse operator sequence restores
the carrier to the frozen numerical equivalence region; exact R1 receipt traversal
recovers T0; and the latch survives outside the reversed histories
```

Claim ceiling:

```text
SOFTWARE_CATALYTIC_WAVE_LOOP_REFERENCE_ONLY
```

Forbidden promotion:

```text
no byte-exact complex-carrier claim unless hashes actually match
no physical carrier or physical R2 claim
no Ising coupling, solving, or optimization claim
no algorithmic, energy, or capacity advantage
no physical audio or silicon-phonon computing
no hardware bit replacement
no Small Wall or Big Wall crossing
```

## 14. Failure Classes

```text
BLOCKED:
    useful latch cannot be separated from history before restoration;
    correct reverse sequence cannot restore the carrier equivalence class;
    exact T0 ancestry cannot be recovered;
    wrong/reordered/omitted controls restore as well as the correct sequence;
    scalar output is required inside native evolution

INCONCLUSIVE:
    the mechanism remains coherent but exact source custody, committed fixtures,
    deterministic reproduction, adversary closure, restoration-law review, or repository
    qualification is incomplete
```

## 15. Current State

```text
AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED
AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED
AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED
AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_SOURCE_CANDIDATE
AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_NOT_YET_ESTABLISHED
RECURSIVE_CATALYTIC_ISING_NOT_ESTABLISHED
PHYSICAL_AUDIO_COMPUTING_NOT_ESTABLISHED
PHYSICAL_SILICON_PHONONIC_COMPUTING_NOT_ESTABLISHED
SMALL_WALL_CROSSED_NOT_PROMOTED
```

Contact counts:

```text
audio playback       0
audio recording      0
hardware contact     0
target contact       0
SSH/SCP              0
```

Next exact boundary:

```text
EXACT_R0_R1_INTEGRATION_AND_R2S_RESTORATION_LAW_QUALIFICATION
```
