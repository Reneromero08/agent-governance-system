# Topology-Compiled Bosonic Givens Phase Closure

## Result

Accepted bounded claim:

```text
BOUNDED_TOPOLOGY_COMPILED_BOSONIC_GIVENS_PHASE_CLOSURE_REPLACES_STREAMED_NECKLACE_TRANSITION_PERMANENTS_WITH_POLYNOMIAL_OCCUPATION_SCRATCH_ACTUAL_RESTORATION_AND_REUSE
```

Claim ceiling:

```text
EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_DEPTH8_TESTED_NONZERO_CHIRP_SCHEDULE_COMPLEX128_SOFTWARE_ONLY
```

Evidence:

```text
/tmp/four-rotor-bosonic-givens-accepted.GMfQks
```

## Mechanism

The resident carrier remains 285 unresolved necklace amplitudes. Each
17-mode circulant quadratic free law is compiled into 136 complex Givens
rotations and a diagonal phase. The engine expands only to the 4,845
permutation-symmetric degree-four occupation coefficients, applies each
two-mode rotation through homogeneous polynomial blocks, and closes every
rotation orbit back to its necklace amplitude.

The accepted path retains no `285^2` transition operator, materializes no
83,521-cell labelled wave or labelled assignment expansion, and enumerates
zero permanent assignment terms. It does materialize a 4,845-cell occupation
expansion. That scratch has the same fixed-grid `O(R^16)` carrier law as the
necklace representation, with a factor-17 loss from temporarily releasing the
global-rotation quotient.

## Parity and catalytic lifecycle

The Givens carrier agrees with the exact-cyclotomic streamed-permanent
predecessor within:

```text
one-step weighted state L2 error          2.570e-15
depth-eight final-boundary error          4.663e-15
single-particle decomposition residual    7.340e-16
necklace orbit closure residual           9.537e-17
```

The depth-eight weighted norm error is `7.661e-15`. The actual borrowed
carrier restores within `8.446e-15`. An unrelated second program consumes the
actual restored carrier, reaches restoration generation two, restores within
`9.273e-15`, and agrees with fresh execution within `6.162e-15`. Missing,
wrong, and applicable reordered inverse controls separate by `1.404`, `1.346`,
and `1.075`.

## Accounting and comparison

The accepted path reports:

```text
resident carrier payload                        4,560 bytes
retained public topology                       10,532 bytes
occupation scratch                             77,520 bytes
Givens plan                                     4,624 bytes
conservative polynomial block scratch             211 bytes
conservative compilation explicit payload      19,867 bytes
maximum explicit engine payload                97,447 bytes
maximum explicit wrapper payload              102,119 bytes
comparison harness peak                       106,567 bytes
retained inverse history                            0 bytes
retained transition operator                        0 bytes
```

The predecessor comparison harness enumerates 530,236,800 permanent terms.
The accepted path instead executes 15,917,440 polynomial block terms and
8,434,176 two-mode updates. Their count ratio is `33.312`; these are
heterogeneous operations and the ratio is not a total-work or speedup claim.
In the same warm process, the recorded accepted lifecycle takes 1.645 seconds
versus 4.853 seconds for the predecessor, a descriptive `2.951x` elapsed
reduction.

The best matched classical bosonic Givens simulator is identical. This result
therefore removes factorial permanent enumeration from the accepted phase
path but does not establish a distinct phase resource, computational
advantage, Small Wall crossing, or unbounded computation.

The surviving obstruction is:

```text
POLYNOMIAL_OCCUPATION_SCRATCH_AND_MATCHED_CLASSICAL_BOSONIC_GIVENS_IDENTITY
```

The next experiment should enforce custody of the 4,845-cell intermediate or
find a symmetry-preserving closure that never releases the 285-cell necklace
quotient. Neither action alone would establish leverage against the matched
classical recurrence.
