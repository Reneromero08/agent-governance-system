# F17 cubic-chain adaptive-gauge independent review

Scientific source commit:
`f67319b66aeb8cabef5b871a3d2eda9a54532c0d`

## Decision after pre-seal repair

```text
classification:
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification_level:
    SEPARATE_REFERENCE_PARITY

restoration_class:
    EXACT_ALGEBRAIC_RESTORATION
```

The first source inspection rejected the restoration interpretation because
the draft inverse compared a recomputed transfer with the resident target and
then cleared the target. That was validated erasure, not algebraic
uncomputation. The draft was not sealed or promoted.

The repaired source recomputes the expected public transfer, subtracts every
integer coefficient from the actual resident target, subtracts the exact
power-of-17 scale exponent, and applies offset subtraction to the pivot
register with pivot 16 as the zero identity. Seed release uses the same exact
operation. The equality tests are guards, not the restoration mechanism. The
qualifier rejects any `.clear()` call in the accepted source.

## Verified ceiling

The result is limited to two deterministic public F17 path-program families
at nodes 2, 3, 5, 9, 17, 33, and 65. Factors are unary cubic terms and
nearest-neighbor mixed cubic terms. The message retains 17 latent-value rows,
each in an adaptive 16-coefficient chart of `Z[zeta17]`, plus one exact
integer power-of-17 scale exponent.

The adaptive chart selects the omitted root with minimum signed-bit payload
and lowest-root tie breaking. Exact use of
`1 + zeta + ... + zeta^16 = 0` is independently reconstructed. The separate
fixed-basis oracle consumes public hashed descriptors without importing or
calling the production compiler, transfer, gauge selector, projector, or
inverse. It matches semantic and factorized boundaries for every case,
checks the maximal power-of-17 content and pivot minimum, performs a
retain-all exact inverse, and directly enumerates assignments at nodes 2 and
3.

## Measured reduction and remaining growth

Across nodes 2 through 65, the unfactored fixed-basis reversible peak is:

```text
869  1,837  4,653  13,432  35,191  88,872  216,589 bits
```

The adaptive chart with resident exact 17-content reduces that peak to:

```text
937  1,885  2,851  5,580  12,760  30,652  76,433 bits
```

Maximum coefficient signed width is reduced from
`3, 6, 10, 18, 34, 68, 136` to `3, 5, 5, 9, 14, 26, 50` bits. The stored
content exponents are `0, 0, 1, 2, 5, 10, 21`, matching
`floor((nodes-1)/3)` for the tested cases. Residual coefficient width still
grows, so neither fixed integer width nor constant reversible storage is
established.

The same backing is restored and reused by an unrelated program without
baseline reload or retained inverse history. Missing, wrong, and reordered
inverse controls fail. Resource accounting includes the seed and inverse
expected messages, pivot metadata, the 49-cell combined regauge temporary,
projection and reconstruction buffers, content diagnostics, descriptors,
and the fresh verification carrier. Python object, allocator, recursive
stack, bit-operation, OS, and whole-process peaks remain unbounded.

## Baseline and rejected interpretations

The identical adaptive-gauge and content recurrence is the matched compact
classical implementation. A dense period-17 block would require 73,984
integer cells and at least 4,624 transfer-equivalents to construct, exceeding
all declared single-query streaming cases. It was applicability-gated rather
than executed. The strongest family-specific method is therefore not
established.

This result does not establish fixed coefficient width, arbitrary graph
topology, general non-Gaussian composition, CATVM custody, a distinct phase
resource, computational advantage, Small Wall crossing, catalytic inference,
physical waveform execution, replacement of physical bits with pi, or
unbounded computation.
