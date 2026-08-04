# F103 Growing Two-Fiber Paley Coherent Configuration Independent Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION` for the resident carrier;
temporary operands, deltas, enumerated class tables, and dense oracle matrices
carry `NO_RESTORATION_CLAIM`.

## Reconstructed mechanism

Each weighted relation acts on two public fibers over the additive points of a
Paley order. Its twelve coefficients are a two-by-two fiber matrix tensored
with the identity, quadratic-residue, and nonresidue difference classes.
Composition is matrix multiplication over the three-class convolution algebra;
intersection is coefficientwise multiplication because the basis relations
are disjoint. The fiber matrix makes composition noncommutative.

The independent oracle imports no production code. It enumerates every Paley
class and reconstructs each structure table rather than using the production
closed formulas. Its separately written twelve-coordinate recurrence matches
all production boundaries, complete state commitments, checkpoint records,
and exact reversals.

## Executed evidence

- Primary depth-256 runs at Paley orders 5, 13, 17, 29, 37, 53, 73, and 97;
  a primary order-97 depth-1024 run; and an alternate order-37 depth-64 run.
- All 144 ordered composition basis pairs and all 144 intersection basis pairs
  match explicit dense relation semantics at orders 5 and 13.
- Formula structure constants match independently enumerated constants at all
  eight declared orders.
- Missing, wrong, and reordered inverses fail; module reordering changes the
  boundary; canonical fiber-matrix units witness noncommutativity.
- Exact same-backing restoration and cross-family reuse at generation two
  match fresh execution without a snapshot.

## Resource law and ceiling

Each port retains twelve F103 coefficients and the two-port carrier retains
24, independent of the represented vertex count and executed depth. At order
97 those coefficients denote a relation on 194 vertices with 37,636 ordinary
matrix entries per port. The conservative named peak is 72 field cells:
carrier 24, public operand 12, delta 12, and restoration verifier 24. It
excludes Python containers, allocator/runtime state, and whole-process memory.

The strongest compact classical method is the identical twelve-coordinate
recurrence and has the same law and named peak. Dense expansion is an oracle,
not the matched baseline. The result is limited to the eight declared Paley
orders, two fibers, the public generators, ten runs, and direct-process Linux
software. It establishes restricted fixed-rank relational closure, not a
general coherent-configuration compiler, CATVM custody, a distinct phase
resource, computational advantage, a Small Wall crossing, physical waveform
execution, physical-bit replacement, or unbounded catalytic computation.
