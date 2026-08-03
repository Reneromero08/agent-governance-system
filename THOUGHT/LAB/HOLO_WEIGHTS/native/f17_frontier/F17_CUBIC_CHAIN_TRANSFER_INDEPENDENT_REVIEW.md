# F17 cubic-chain transfer independent review

Scientific source commit:
`8d769d1ca8f7a9da233c5ca6a0aa9422f911c8d7`

## Decision after repair

```text
classification:
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification_level:
    SEPARATE_REFERENCE_PARITY

restoration_class:
    EXACT_ALGEBRAIC_RESTORATION
```

The first review decision was `INCONCLUSIVE` because the production source
hash had changed after its evidence seal. The source, oracle, qualifier, and
both result fixtures are now sealed by
`F17_CUBIC_CHAIN_TRANSFER_PROVENANCE.json`, and the qualifier checks every
sealed hash before execution.

## Verified ceiling

The result is limited to two deterministic public F17 path-program families
at latent-node counts 2, 3, 5, 9, 17, 33, and 65. Every tested edge count is
a power of two. The factors are unary cubic terms and nearest-neighbor
`x_i^2*x_(i+1)`, `x_i*x_(i+1)^2`, and bilinear terms.

The accepted exact message has 17 latent-value rows and the canonical
16-dimensional integer basis of `Z[zeta17]`. The final boundary is the
16-coefficient numerator with public normalization metadata
`17^(-nodes/2)`. It is not merely an unnormalized character sum.

The topology-derived recursive schedule uses 2 through 8 message slots
(544 through 2,176 integer cells) instead of retaining all 2 through 65
messages. Its forward transfer applications are 1, 3, 9, 27, 81, 243, and
729. The global `17**nodes` assignment trace is absent. Each transfer still
enumerates fixed local left-value, right-value, and cyclotomic-basis domains.

The separate oracle does not import or call the production compiler,
transfer, or projection. Its two-message dynamic program and retain-all
inverse match the exact boundary for every declared case. Direct assignment
enumeration additionally matches at nodes 2 and 3.

## Restoration and accounting

The recursive forward schedule leaves only the source and final message. The
actual recursive inverse rematerializes required checkpoints, clears the
final and every scratch message exactly, and removes the source seed.
Generation and lease each advance to 2 after an unrelated second program.
The outer message container, every message container, and every row retain
their backing identity. No baseline is reloaded and no inverse descriptor is
retained.

The durable accounting distinguishes the 272-cell preprojection message from
the 16-coefficient boundary. It reports public factor and descriptor sizes,
boundary bytes, scalar updates, recursive call-frame and scratch-reference
bounds, full-carrier instrumentation scans, one accepted-path carrier, one
additional fresh verification carrier, and a two-carrier verification peak.
Python integer objects, growing-integer bit-operation complexity, container
allocation, recursive call-stack bytes, scratch-tuple bytes, OS state, and
whole-process peak remain unbounded.

Single-coefficient signed width grows from 3 to 136 bits and total logical
message payload grows from 869 to 216,589 bits across the tested depths.
Therefore neither fixed integer width nor constant reversible storage is
established.

## Baseline and rejected interpretations

The identical exact two-message streaming dynamic program uses 544 integer
cells and fewer transfer applications than reversible pebbling for nodes at
least 3. It is the strongest implemented matched streaming baseline, not an
established strongest family-specific method. The factor sequence is periodic
over F17; block-transfer powering remains untested.

This result does not establish arbitrary non-power-of-two schedule behavior,
arbitrary graphs or treewidth, general non-Gaussian composition,
machine-enforced CATVM custody, a distinct phase resource, computational
advantage, Small Wall crossing, catalytic inference, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.
