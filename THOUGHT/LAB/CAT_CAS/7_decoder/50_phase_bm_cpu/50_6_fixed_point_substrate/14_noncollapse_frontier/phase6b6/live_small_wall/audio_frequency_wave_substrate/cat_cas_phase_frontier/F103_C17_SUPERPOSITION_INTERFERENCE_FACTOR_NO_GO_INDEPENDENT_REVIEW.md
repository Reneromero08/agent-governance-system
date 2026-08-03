# Independent review: F103[C17] superposition interference factorization

## Decision

```text
classification: INDEPENDENTLY_VERIFIED_STRICT_SCOPE
verification_level: INDEPENDENT_ORACLE_REEXECUTION
resident_restoration_class: EXACT_ALGEBRAIC_RESTORATION
transient_restoration_class: NO_RESTORATION_CLAIM
claim_ceiling: F103[C17] rotation, addition, and convolution shears on the declared 51-port rotating-hub topology across 18 cases through depth 1024
```

The bounded mechanism is supported at that ceiling. It is a substantive repair
of M149's one-hot limitation: all 51 logical factors may carry general
17-coordinate finite-field superpositions, exact cancellations occur in the
accepted path, and the shared port is consumed by noncommuting OUT and IN
layers. The stronger resource interpretation is rejected because the whole
declared primitive algebra factorizes exactly into 17 independent classical
modes.

## Independent reconstruction

The oracle imports neither the production module nor the M149 schedule source.
It separately implements:

- all three public families and 18 depth/family cases;
- the public rotating-hub topology, offsets, and phase schedule;
- an exact F103 number-theoretic transform using the order-17 root 72;
- 17 independent modal triangular-shear recurrences;
- inverse transformation to every one of the 867 coefficient cells;
- final 17-coordinate boundary projection and full-state SHA-256 commitments;
- exact forward/inverse restoration and 100 reuse cycles; and
- convolution, order, inverse, topology, null-port, and interference controls.

All 18 final carrier commitments, boundaries, minimum and maximum support
counts, and forward work counts match. All 126 case comparisons pass. Every
spectral inverse returns exactly to the public seed.

## Mechanism and controls

Each of the 51 resident factors is an element of `F103[C17]`. A public phase
operation rotates its orbit coordinates. One hub factor controls 16 reversible
triangular convolution shears, and the updated target factors then control the
reciprocal hub shears through a different slot. Consequently OUT/IN order is
noncommuting. No resident factor is decoded to an angle, exponent, or scalar,
and only the final 17-coordinate boundary is emitted.

The accepted executions contain 36,893 exact cancellation events, and an
explicit two-path cancellation witness is checked independently. Missing,
wrong, and reordered inverses fail restoration. Disabling the port, changing
layer order, or mutating the hub topology changes the boundary. Premature
projection, resident-port projection, wrong ownership, and null carriers are
rejected by production controls.

The same carrier backing restores exactly without retained inverse history,
baseline reload, or snapshot reload. An unrelated depth-613 program agrees
with a fresh carrier in boundary, final commitment, and resource signature.
One hundred depth-16 reuse cycles restore exactly.

## Spectral factor no-go

Because 17 divides 102, `x^17 - 1` splits into 17 distinct roots over F103.
Therefore the declared group algebra is isomorphic to 17 independent F103
coordinates. The independent oracle executes that isomorphism rather than
assuming it.

Both paths retain 867 one-byte field coordinates. At the depth-1024 ceiling,
the coefficient carrier counts 18,939,904 convolution multiplications while
the matched spectral recurrence counts 1,114,112 modal multiplications for the
same shears, a factor of 17 in convolution work. Named warm live storage is
1,243 bytes for the coefficient path and 1,226 bytes for the spectral path.
These totals exclude Python container and allocator overhead, NumPy/native
library internals, and whole-process peak memory.

The spectral baseline is an executed strong matched recurrence, but no claim
of classical optimality is made.

## Claim boundary

Preserved:

- bounded general multi-coordinate F103[C17] phase-orbit superposition;
- exact native convolution interference and cancellation;
- one unprojected resident port consumed by multiple noncommuting modules;
- final-only boundary projection;
- exact same-backing restoration and unrelated reuse; and
- exact full-state parity with the 17-mode spectral recurrence.

Not established:

- complex or physical coherence;
- general phase-relational contraction or CATVM custody;
- a resource beyond the 17-mode recurrence;
- computational advantage or Small Wall crossing;
- physical waveform execution or physical bit replacement; or
- unbounded catalytic computation.

## Reviewed artifacts

```text
production source:
  a8e4dac14366fb4edbeb7335ba938f8675e352216840d3cb8a3a8c5a8f93a86a
oracle source:
  c371e87a2037034228f1f2b244889c824c5d6e14147d2e864566507a6124246d
production result:
  4042f591f2db394293cc73d59b27ca1af80239ace3bcb528d0633dfb8dcd021d
oracle result:
  c9cef44d3dcfe7280360337a9167f242ec0908bb425333e8c63b9564616e8367
qualifier:
  9275c532ec36b92c65050ccb18eb410c63e52ae63c8ed509a52324c69137dcde
M149 public schedule dependency:
  1ec87e427dd3c4548b749c6d431208ff19159528d5371cef4391eb6b02144472
```
