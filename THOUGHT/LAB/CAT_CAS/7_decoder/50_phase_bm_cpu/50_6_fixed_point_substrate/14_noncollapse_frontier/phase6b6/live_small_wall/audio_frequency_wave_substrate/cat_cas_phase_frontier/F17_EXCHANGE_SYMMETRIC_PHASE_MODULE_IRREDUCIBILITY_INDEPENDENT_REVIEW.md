# M128 Independent Review: Exchange-Symmetric Phase-Module Irreducibility

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident grid and orbit phase-carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Transient grid, lifted-shear, compiler, and certificate buffers:
`NO_RESTORATION_CLAIM`

The verified source is commit
`aa641f93a89c6311f469e222ae4f94e1ecad960f`. M128 is a distinct extension
of M127. It does not retroactively establish an irreducibility result for the
unextended M127 primitive set.

## Independent reconstruction

For each declared exchange-symmetric orbit module at `k=1,2,3,4`, the
independent oracle reconstructed the occupation set and independently
implemented the `p1,...,pk` phase characters, both orientations of adjacent
mode shears, the phase transaction, and the irreducibility certificate. The
orbit dimensions are

```text
H(k) = binomial(k+16,16) = 17,153,969,4845.
```

Because `k<17`, Newton denominators are invertible and `p1,...,pk` separate
the declared occupations. Exact character orthogonality therefore generates
every coordinate projector. The bidirectional adjacent-mode shears have
nonzero one-particle transition entries and connect the occupation graph;
projecting their entries and multiplying along the streamed predecessor tree
generates all matrix units. Consequently, a uniform exact linear quotient
that preserves every independently selectable declared character and shear
and the nonzero final occupation functional has dimension at least `H(k)`.

This is a uniform-linear-module result only. It does not reject nonlinear,
program-restricted, integrable, approximate, or other representation laws.

## Execution, restoration, and resources

Independent exact `Q(zeta17)` execution agrees at `k=1` PRIMARY and `k=2`
PRIMARY/REUSE. Independent `F103/F137` execution agrees at `k=1,2,3,4`.
Only the final `(k-1 in mode 0, one in mode 1)` occupation is projected.
Actual reverse execution restores exact zero on the same backing, increments
the restoration generation, and supports an unrelated `k=2` reuse program
whose boundary and complete reported deterministic resource signature agree
with a fresh carrier. No snapshot or inverse history is used.

The accepted carrier contains `48+2H(k)` resident field cells. Character,
shear, and grid work is streamed. Production materializes no character table,
dense `H(k) x H(k)` operator, labelled tensor, relation table, serialized
intermediate, or certificate edge list. Public topology, signature sets,
union-find storage, compiler descriptors, transaction transients, projection,
restoration, reuse, and exact payload heights are reported separately.
Oracle-only cached boundary vectors are not accepted-path resources. Full
exact bit complexity, Python/native containers, allocator and bigint
internals, and whole-process peak are not established.

The strongest matched classical implementation remains the identical
`H(k)`-coordinate exact linear phase-module recurrence.

## Claim ceiling

M128 establishes a bounded linear no-go for the new selectable
power-sum-character and bidirectional adjacent-mode-shear algebra on the
declared `k=1..4` exchange-symmetric modules. It does not establish an
unextended-M127 no-go, a nonlinear or program-restricted lower bound, general
relational closure, CATVM custody, a distinct phase resource, computational
advantage, Small Wall crossing, physical waveform execution, replacement of
physical bits with pi, or unbounded catalytic computation.
