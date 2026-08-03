# M137 Independent Review: Rank-One Cubic/Walsh Derivative Chart

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident group-algebra carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Compiler, projection, commitment, and oracle buffers:
`NO_RESTORATION_CLAIM`

The reviewed scientific source is commit
`bdbc3a1339b7112c28c78f25425899bda0c3e112`.

## Verified mechanism

The public Boolean topology defines

```text
Q_n(z) = sum_i z_(2i) z_(2i+1) mod 17
```

and every accepted cubic phase derivative is an exact scalar multiple of
that one canonical quadratic signature, plus an explicitly retained scalar
phase.  The production compiler canonicalizes the public Boolean monomials
and row-reduces their coefficients over `F17`; it does not inspect a final
answer or compare sampled assignments.

For this rank-one family, the two unresolved Walsh components close exactly
in `K[C17]^2`, represented by 34 field cells.  Cubic phases are cyclic shifts
of the second row, Walsh mixing maps `(A0,A1)` to
`(A0+A1,A0-A1)`, and the exact inverses are the opposite shift and
`(1/2)H`.  The result covers exact `Q(zeta17)` branch-pair/round cases from
`(1,2)` through `(128,256)` and structural parity over `F103` and `F137`.

An independent second quadratic signature has derivative rank two.  It
therefore expands the canonical full `C17^2` group-algebra chart to
`2*17^2 = 578` cells, and the 34-cell compiler rejects it.  This is a
canonical-chart size and compiler-capacity result, not a universal minimum
over every possible exact representation.

## Independent reconstruction

The oracle does not import the production package.  It separately rebuilds
the public programs and canonical derivative rank, reimplements the
34-coordinate forward and inverse recurrence, evaluates the final character
moments, and reconstructs the strongest classical 17-residue multiplicity
recurrence.  It also directly enumerates Boolean assignments only in twelve
bounded verification cases through six branch pairs.

All twenty production transactions match independently for program
fingerprint, final boundary, hidden-state commitment, exact inverse seed,
and maximum resident payload.  The exact `Q(zeta17)` payload sequence is:

```text
1096, 1120, 1616, 3313, 7607, 16305, 33763, 68588 bits
```

Thus the logical carrier remains 34 cells while exact material width grows.
The oracle independently confirms the rank-two mutation, modulo-17 duplicate
canonicalization, nonmerge of an independent unsampled monomial, wrong-family
inverse failure, and the zero-divisor law for `1-s`.

The repaired strict qualifier was reexecuted read-only and emitted:

```text
QUALIFIED_F17_RANK1_CUBIC_WALSH_DERIVATIVE_GROUP_ALGEBRA_CLOSURE_STRICT_SCOPE
```

The replayed artifacts were byte-identical to the seals:

```text
production  3e67009e713dd2b5e23272ac1ff9eb9ce6e8742953afb5c1205039773a0681f9
oracle      eab1c4a2d7a3b488f2b6060bd639dc062f19bd470e2cff51c8c16b5c15905aae
```

## Restoration, reuse, resources, and ceiling

The accepted transaction performs forward execution, retains the final
group-algebra state, computes one final scalar boundary, applies the actual
inverse, verifies exact seed restoration on the same backing, clears the
seed, and only then returns the result object.  The inverse rematerializes
operations from public topology and retains no inverse history.  Unrelated
reuse agrees with a fresh carrier in boundary and resource signature without
snapshot reload.  Its counter is explicitly package-local and is not an
enforced CATVM custody generation.

The strongest matched classical method is the identical exact 34-coordinate
group-algebra recurrence, equivalently a 17-residue multiplicity/character
recurrence.  It has the same arithmetic and exact-payload law.  Fixed logical
cell count therefore does not establish a distinct phase resource or an
advantage.

The strict result applies only to the declared rank-one pair-product
signature family, one unresolved typed Walsh bit, the three declared exact
fields, and direct-process software.  It does not establish rank-two closure
in 34 cells, a lower bound for arbitrary alternative representations,
general rank-r or arbitrary cubic-hypergraph closure, CATVM custody, a
distinct phase resource, computational advantage, Small Wall crossing,
physical waveform execution, replacement of physical bits with pi, or
unbounded catalytic computation.
