# F103 S3 noncommutative phase relation independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Algebra and scope

The accepted carrier represents an S3-translation-invariant relation
`R(x,y)=f(x^-1 y)` by the six values of `f`. Native relational composition is
nonabelian S3 group convolution and native intersection is coordinatewise
Hadamard product. Unlike the preceding three-class C17 Paley algebra, left
and right composition differ: explicit delta functions for group elements 1
and 2 compose to different products in opposite orders.

Two six-cell relation ports remain resident. In each public step the shared A
port is consumed by right and left composition modules and the shared B port
is consumed by two intersection modules. The two declared module orders are
noncommuting. Twelve cases cover both orders at depths 1, 4, 16, 64, 256, and
1024. Only a final scalar B boundary and a one-way whole-forward-state
commitment are serialized; neither hidden relation is emitted.

This is the exact S3 translation-invariant family, not general relations on
six labels, a general finite-group compiler, or an arbitrary graph result.

## Independent full-relation oracle

The oracle imports neither production nor NumPy. It independently lists the
six permutations, reconstructs each six-coordinate function as a 6 by 6
relation matrix, and performs composition by full matrix multiplication and
intersection by full matrix Hadamard product. All 36 basis compositions equal
the independently calculated S3 group products, all 36 basis intersections
match, and every output remains translation invariant.

The full 72-cell two-register oracle reproduces the boundary and commitment
for all 12 production cases and exactly clears both matrices under the actual
reverse sequence. This is semantic-reference expansion in the oracle only;
the accepted path never materializes either 6 by 6 relation.

## Classical comparison and obstruction

The production package separately executes the identical six-coordinate
group recurrence and an S3 Fourier/irrep recurrence. The latter consists of
one trivial scalar, one sign scalar, and one 2 by 2 standard block: six field
coordinates in total. Independent finite-field elimination finds the Fourier
transform rank is exactly six, all six basis functions round-trip, and all 36
irrep convolution products match the group recurrence. The irrep chart does
not compress information below group order.

The accepted phase path and strongest executed group-coordinate classical
path each retain 12 carrier field cells. At the abstract relation-value level
each uses one six-cell atomic-operation scratch, for an 18-cell working peak.
The implementation also reports a conservative source-liveness count: 70
explicit Python field-value slots for the accepted path versus 58 for the
direct classical recurrence, including current-step public operands and
scalars but excluding Python object headers, allocator metadata, interpreter
state, and whole-process peaks. No inverse-value history or snapshot is
retained.

The noncommutative algebra is strictly broader than the commutative Paley
fixture, but it remains an identical compact classical recurrence. No phase
resource, advantage, or Small Wall crossing follows.

## Controls and restoration

Missing, wrong, and reordered inverses fail exact canonical restoration. The
same public modules in the opposite order change the boundary. Premature and
hidden projection, null carrier, invalid depth, carrier-disabled execution,
and conjugacy-class overmerge controls all behave as required. The order-six
phase root and the two-dimensional representation homomorphism are checked
exactly.

Every valid transaction returns the exact twelve cells, seed-family type, and
idle stage to their pre-forward state on the same backing before incrementing
the restoration generation. An unrelated alternate program consumes the
actual carrier restored by a primary program, reaches generation two, and
matches a fresh carrier without snapshot.

The direct-process package does not establish CATVM custody, OS/hardware
isolation, a distinct phase resource, computational advantage, Small Wall
crossing, physical execution, replacement of physical bits with pi, or
unbounded catalytic computation.
