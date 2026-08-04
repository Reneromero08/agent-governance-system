# F103 Growing Dihedral Irrep-Sparse Relation Independent Reexecution Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION` for both the resident irrep
carrier and the independent group-coordinate recurrence. Temporary public
operands, Fourier observations, and digest buffers carry
`NO_RESTORATION_CLAIM`.

## Reconstructed mechanism

The production path stores two translation-invariant weighted relations over
each split dihedral group in adaptive nonzero irreducible-representation
blocks. Native composition is noncommutative block matrix multiplication.
Native intersection reconstructs only one group coefficient at a time, applies
the Hadamard product, and immediately accumulates one output block coefficient;
it does not retain a full group-coordinate vector on the accepted path.

The independent oracle imports no production module. It instead uses full
dihedral group coordinates as primitive state, reconstructs the public seeds,
public operands, four-shear program, convolution and Hadamard laws, final
boundary, inverse schedule, and commitments. A separately written exact
Fourier observer measures the active irrep blocks at the declared checkpoints.

## Executed evidence

- Primary depth-16 cases for rotation orders 3, 6, 17, 34, and 51, plus an
  alternate depth-8 order-17 case.
- Exact agreement for every boundary, complete forward-state commitment,
  support-history tuple, and final-support tuple.
- Full representation homomorphism checks and Fourier round trips for every
  group basis vector in all five declared groups.
- Independent exact reversal for every case, with missing, wrong, and reordered
  inverse attacks failing restoration.
- Module reordering changes the tested boundary; production separately checks
  distinct left and right products in a two-dimensional irrep.
- Same-backing production reuse performs a depth-1 primary transaction followed
  by an unrelated depth-8 alternate transaction and matches a fresh carrier.

## Resource law and ceiling

The initially active two-port capacity is eight field cells. In all five
primary cases it grows to the full two-port group-algebra capacity by depth 16:
12, 24, 68, 136, and 204 field cells for group orders 6, 12, 34, 68, and 102.
For the three larger groups the measured histories are respectively
`24 -> 40 -> 68`, `24 -> 40 -> 72 -> 134 -> 136`, and
`24 -> 40 -> 72 -> 136 -> 204` at the applicable checkpoints. Thus adaptive
irrep sparsity does not provide a group-size-independent fixed-rank closure in
this bounded family.

The strongest compact classical comparison is the identical adaptive irrep
recurrence, so it has exactly the same retained-state law. The executed full
group-coordinate recurrence also agrees in every case. Streamed intersection
avoids retaining a group vector but performs `(2n)^2` group-coefficient scans
per output transform. Counts exclude Python containers, allocator and runtime
state, cached public group topology, and whole-process peak memory.

The verified claim is limited to split F103 dihedral translation-invariant
relations, the declared public generators, rotation orders 3/6/17/34/51, the
six executed cases, and direct-process Linux software. It establishes exact
noncommutative relation composition, exact intersection, final-only boundary
projection, restoration, and reuse, together with a bounded fixed-rank no-go.
It does not establish a general finite-group compiler, non-translation-
invariant relations, CATVM custody, a distinct phase resource, computational
advantage, a Small Wall crossing, physical waveform execution, replacement of
physical bits with pi, or unbounded catalytic computation.
