# F17 period-17 native cyclotomic-module independent review

Scientific source commit:
`bf845d208feb81972f63e5d35bb36596e43d7bc6`

## Decision

```text
classification:
    INDEPENDENTLY_VERIFIED_STRICT_SCOPE

verification_level:
    SEPARATE_REFERENCE_PARITY

restoration_class:
    EXACT_ALGEBRAIC_RESTORATION
```

The sealed source supports an exact degree-17
`Q(zeta17)`-coefficient Cayley-Hamilton annihilator for each of two fixed
public period-17 F17 cubic-chain block operators. The result is a certified
closure over the declared coefficient field, not a minimal-order result and
not a scalar-`Q` recurrence of order 17.

## Independent reconstruction

The separate tuple-arithmetic oracle does not import or call the production
compiler, operator builder, characteristic-polynomial routine, projection,
or inverse. It recompiles both public descriptors, constructs both
17-by-17 `Z[zeta17]` operators, and evaluates each supplied monic
annihilator on every operator entry. Both residual matrices are exactly zero.

The oracle independently reconstructs all PRIMARY and REUSE boundaries at
periods 1, 2, 4, and 8. All eight match both the sealed hashes and the prior
adaptive fixed-basis semantic boundaries.

## Failed modular-dependence lift control

An eight-prime CRT probe combines the prior degree-241 and degree-256 modular
dependence coefficients through a 108-bit modulus. The candidate coefficient
width reaches 108 bits in both families. Direct streamed exact
`Z[zeta17]` evaluation completes below the declared 16,384-bit vector cap,
but every one of the 272 residual cells remains nonzero.

This rejects those two CRT candidates. It does not prove that no other exact
recurrence or quotient exists.

## Restoration and reuse

The runtime uses recursive reversible pebbling and exact coefficient-wise
subtraction of recomputed resident targets. At period 8, PRIMARY restores the
actual borrowed carrier, REUSE consumes that same backing, and the unrelated
REUSE boundary matches a fresh carrier. Generation and lease reach two; all
canonical message cells are zero; retained inverse history and baseline
reload are zero.

Missing, wrong, and reordered inverse controls fail as required. Null carrier
use is rejected, and the public semantic family perturbation changes the
boundary.

## Scope limits

The runtime still applies the dense 17-by-17 cyclotomic block; it certifies
but does not execute the characteristic recurrence. Restriction of scalars
turns every `Q(zeta17)` coefficient multiplication into a 16-by-16
`Q`-linear map, so the coefficient-ring change does not lift the prior
modular dependencies or establish scalar-`Q` order 17.

Logical resource accounting is component-level. It names tuple operators,
characteristics, carriers, expected messages, projection, restoration, and
oracle components. SymPy characteristic internals, coexisting process peaks,
Python objects, allocator and native-library overhead, bit-operation cost,
and whole-process memory remain unbounded.

The identical cyclotomic matrix execution and characteristic identity are
available to compact classical software. The evidence does not establish a
distinct phase resource, computational advantage, Small Wall crossing,
CATVM custody, catalytic inference, physical waveform execution, replacement
of physical bits with pi, fixed integer width, constant reversible storage,
or unbounded computation.
