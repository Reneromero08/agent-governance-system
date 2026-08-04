# M180 independent reexecution review

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Verified result

The M179 q=7 rejection applies to its public scalar congruence-stratum
completion family. It is not a rejection of every compact exact descriptor.
M180 changes the carrier representation: it expands the open
`psi(det(X)/t)` relation in multiplicative characters and applies the
Fourier transform to those modes through an exact Gauss-sum law.

For symmetric three-by-three matrices, the independently reconstructed open
determinant-character gamma factor is

```text
Gamma(j) = G(j)^2 G(j+quadratic) G(quadratic)^3.
```

The trivial determinant character has rank-stratified singular terms. The
quadratic determinant character has a rank-one singular term. Every other
determinant character is zero on the singular Fourier boundary. The scale
character has the ordinary single exceptional trivial mode at zero scale.

Production checked the determinant-character formula on all `q^6` boundary
points for every character at q=5, q=7, and q=11. The oracle did not import
production or M179. It instead accumulated a determinant/diagonal-coordinate
histogram over all symmetric matrices and evaluated every diagonal congruence
representative and determinant value for every character. It reproduced 36,
66, and 150 exact representative identities at q=5, q=7, and q=11.
Congruence equivariance carries those representatives to the full symmetric
matrix boundary.

The oracle also rebuilt four complete seven-dimensional transforms with a
different axis-matrix algorithm: two q=5 programs over all 78,125 points and
two q=7 programs over all 823,543 points. Every value matched the descriptor.
Deleting one Mellin channel or changing one gamma factor caused a mismatch in
every declared case. Direct representative sums show that omitting either
exceptional singular law also changes the result.

## Resource law

The accepted descriptor keeps `q-1` resident phase coefficients. Its
materialized Gauss, determinant-gamma, and discrete-log compiler tables bring
the counted peak to `4*q-3` field cells. Boundary evaluation uses `q-1`
character terms and Gauss compilation uses theta-q-squared multiply-adds.

The dense `q^7` source, transform, and predicted arrays are verification-only
baselines, explicitly excluded from the accepted descriptor path. The
strongest matched classical method is the identical Mellin/Gauss recurrence,
with identical state, tables, work, and boundary law.

## Restoration and reuse

The resident coefficient carrier changes to the formula-defined Fourier-image
basis by exact character-index negation. The normalized inverse is the same
involutive permutation. It restores the exact discrete coefficients on the
same backing without retained inverse history or snapshot reload. A public
final-boundary value is evaluated from the actual dual descriptor before the
inverse and survives restoration. An unrelated quadratic character-index
phase shear consumes the restored backing at generation two, matches a fresh
carrier, and restores again. The oracle independently reproduced primary
restoration, same-backing identity, generation-two unrelated reuse, and second
restoration for all four programs.

This is direct-process software. It does not establish CATVM custody or a
machine-enforced hidden intermediate.

## Strict ceiling

The result is limited to prime q=5/F41, q=7/F43, and q=11/F331, the declared
symmetric-three determinant-over-scale open relation, all determinant
characters for the gamma-law checks, and four complete q=5/q=7 phase
programs. It is an exact finite-field trace-function descriptor, not an
implementation of middle-extension sheaf operations.

The descriptor removes dense q-to-the-seventh state and arbitrary singular
stratum fitting, but it is not fixed width: resident state grows as q and
compiler work grows as q squared. Because compact classical software executes
the identical recurrence, the result establishes no distinct phase resource,
computational advantage, Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, or unbounded computation.

## Next obstruction

Determine whether the Gauss-coefficient family admits an exact fixed-state
procedural recurrence under the required phase compositions, or whether its
exact recurrence/Hankel rank grows with q. Any accepted successor must keep
the strongest identical classical recurrence and must not hide the q channels
in an uncounted compiler table, oracle, or projection routine.
