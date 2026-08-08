# M240 Focused Strict-Scope Review

Decision: `PASS_STRICT_SCOPE`

Verification classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Verified mechanism

The production and standalone sources independently implement exact
`Q(zeta_5)` power-basis arithmetic, the public count-one-through-four program
compiler, a full amplitude transaction, the compiled streamed scalar boundary,
the one-qudit Wigner calculation, and custody/restoration state machines.  The
standalone source imports none of M240, M239, or M237.

At every declared injection count, the exact injection cut equals the product
of the independently cubic-phased data wires and the same coherent `|+>`
syndrome wire.  The public post-injection network connects every data wire to
that syndrome with `SUM` and `CZ`, then uses only Clifford gates.  Dephasing
the shared syndrome changes the selected final data boundary at counts two,
three, and four.  Removing either network gate kind or reordering a declared
noncommuting pair changes the selected boundary or full-state commitment.

Only the selected data probability and one-way commitments cross the boundary.
The actual amplitude and scratch list backings restore exactly by the declared
inverse schedule, generation advances from one to two on descriptor-distinct
reuse, a fresh carrier agrees, and no snapshot or baseline reload is used.
Wrong owner, type, descriptor, generation, stale generation, dirty scratch,
premature syndrome projection, missing inverse, wrong inverse, and reordered
inverse controls reject.

## Exact magic and matched baseline

Both sources reconstruct the single cubic magic state's Gross-Wigner values
and obtain five negative cells and exact l1 norm

```text
1 + 2*sqrt(5)/5.
```

Tensor multiplication and the final Clifford network give the declared exact
count-`m` law `(1 + 2*sqrt(5)/5)^m`.  This is only the product-input
stabilizer-relative magic law.  It is not an optimal stabilizer rank, extent,
classical runtime lower bound, or phase-specific computational resource.

The strongest implemented selected-boundary comparator compiles the injection
identities and streams `5^(m+1)` component/syndrome terms into five exact
boundary accumulators.  It materializes no assignment table and retains no
full amplitude vector.  Its exact scalar boundary and five-value boundary
slice commitment agree with the accepted carrier in all eight cases.  The
accepted carrier retains `25,125,625,3125` field cells plus equal scratch,
while this comparator declares five accumulator and one term field value.
That comparison is field-backing-local; public descriptors, loop coordinates,
phase integers, Python objects, allocation, hashing, serialization, and RSS
are excluded but not zero, and whole-transaction liveness is not claimed.

## Claim ceiling

The evidence establishes a bounded exact direct-process software transaction
with one repeatedly reused unresolved five-valued syndrome, distinct data-wire
injections, a connected Clifford consumer network, final-only data-probability
projection, exact same-backing restoration/reuse, and a causal multiplicative
stabilizer-relative Wigner witness through injection count four.

It does not establish an interaction-generated magic law beyond the tensor
product input resource, an optimal stabilizer decomposition, a computational
advantage, fixed-rank or bounded-width scaling, CATVM custody, inference,
Small Wall crossing, physical waveform execution, physical bit replacement,
or unbounded catalytic computation.  The one-syndrome magic-injection route
should be retired after this result rather than extended to larger counts.
