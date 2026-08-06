# Separate-reference review: SU(2)_8 fusion-path MPS rank obstruction

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`
Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Exact scope

For two declared public eight-sweep braid families and even strand counts
`4,6,8,10,12,14,16`, this diagnostic inspects the exact final
`SU(2)_8` fusion-path vectors from M214. It forms each fusion-sector
prefix/suffix flattening and proves the exact `Q(zeta_40)` ranks by finding
maximal nonzero minors after reduction at split primes. The observed maximum
sector-summed Schmidt bond ranks are `2,3,6,10,20,35,70`.

A nonzero minor modulo a prime at which every exact denominator is invertible
is a certificate that the corresponding characteristic-zero minor is
nonzero. Each reduced rank reaches its analytic dimensional upper bound, so
no uncomputed larger rank is possible. This rejects a uniform fixed-bond
exact MPS for the declared growing family.

## Separate reconstruction

Production uses split primes `241,401` and complete modular row reduction.
The separate reference imports no M215 production code, uses distinct split
primes `641,881`, and builds an incremental modular column basis. It
reconstructs every cut of both families. All four primes give identical cut
ranks, maximum bond ranks, and canonical dense sector-MPS allocations.

Both paths consume the already independently checked M214 exact braid
substrate. This review therefore independently checks the M215 rank analysis,
not a second implementation of the underlying braid evolution.

## Resource and transaction result

At 16 strands, the direct fusion-path carrier has 1,430 exact field cells.
The canonical dense sector-block MPS induced by the certified cut ranks
allocates 4,110 field cells and reaches bond rank 70. That 4,110-cell count is
a stated canonical allocation, not a universal parameter lower bound for
every possible exact representation.

The verification path list, prefix/suffix sets, and modular matrices are
research instrumentation, not accepted computational runtime state. The
primary verification retains 1,430 path records (24,310 path-label cells)
and peaks at 1,430 modular matrix cells. Those costs are reported rather than
hidden.

The accepted transaction remains the M214 final-boundary-only direct process.
It uses the actual 1,430-cell backing, retains no inverse history, restores
the exact source algebraically, increments restoration generation to two,
and agrees with a fresh carrier for the unrelated reuse program without a
snapshot reload.

## Claim boundary

This package establishes an exact fixed-bond MPS obstruction for the declared
two-family bounded sequence. It does not establish a lower bound against all
exact representations, a native compact MPS transaction, CATVM custody, a
distinct phase resource, computational advantage, a Small Wall crossing,
physical waveform execution, replacement of physical bits with pi, catalytic
inference, or unbounded computation. The strongest compact classical
comparators remain the identical exact anyon MPS analysis and the smaller
direct fusion-path recurrence.
