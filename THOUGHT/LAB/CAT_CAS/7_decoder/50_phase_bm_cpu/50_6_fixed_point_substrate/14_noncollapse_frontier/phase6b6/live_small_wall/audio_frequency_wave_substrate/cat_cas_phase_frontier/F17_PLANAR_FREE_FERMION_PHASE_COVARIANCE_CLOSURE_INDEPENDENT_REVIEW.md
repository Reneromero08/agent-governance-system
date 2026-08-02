# M123 Independent Review: Planar Free-Fermion Phase Closure

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Resident terminal/scalar carrier restoration:
`EXACT_ALGEBRAIC_RESTORATION`

Dense Pfaffian projection buffers: `NO_RESTORATION_CLAIM`

The exact independent scope is limited to open planar grids at `n=2,3,4`,
with the `n=4` defect cases `k=0,1,2,4`.  The independently reconstructed
binary residue histogram and bounded-face cycle-space recurrence agree with
all six exact `Q(zeta_17)` terminal-Pfaffian boundaries.  The `F103/F137`
executions at `n=5..8` are preserved as `SOURCE_AUDITED_PACKAGE_LOCAL`
structural executions; no independent numerical boundary parity is assigned
to those wider cases.

## Independent reconstruction

The oracle does not import the terminal-Pfaffian implementation.  It rebuilds
the public grid programs and checks each exact boundary in two separate ways:

1. direct binary residue histograms; and
2. the bounded-face cycle basis of the open square grid.

With `s_v = 1 - 2 x_v`, the binary exponent becomes

```text
C + sum_e 13 b_e s_u s_v - sum_v 13 r_v s_v

C   = 9 sum_v a_v + 13 sum_e b_e       (mod 17)
r_v = 2 a_v + sum_{e incident v} b_e   (mod 17)
```

The declared zero-field compiler makes every `r_v` zero.  Expanding each
edge factor as `c_e + d_e s_u s_v` leaves the even-subgraph sum.

Degree-four grid vertices are replaced by public ordered trivalent paths.
The terminal construction assigns `c_e` to a long edge and distributes
`d_e` asymmetrically across its incident city edges.  The orientation solver
uses only public topology and satisfies the constrained face parities.  The
all-long reference matching fixes the global Pfaffian sign.  Direct and
cycle-space boundaries confirm the construction and sign through the exact
scope.

For each even defect subset, a public path chain has that subset as boundary.
Swapping `c_e,d_e` on the chain gives a bijection from the even cycle space to
the requested defect sector.  Independent alternate-path execution agrees.
The accepted implementation streams, rather than retains, the sectors.  Its
executed cost is one sector for `k=0,1` and `2^(k-1)` sectors thereafter.  This
is an implemented upper bound, not a lower bound on all algorithms.

## Restoration and controls

Each sector loads the actual sparse antisymmetric carrier, evaluates a
Pfaffian in explicitly transient work buffers, unloads the same entries, and
adds one value to the resident scalar accumulator.  Only the completed
aggregate boundary is projected.  The inverse rematerializes the commuting
public sector generator, subtracts every contribution, clears the lease, and
increments the restoration generation.  The same backing then executes an
unrelated `REUSE` program and agrees with a fresh carrier without snapshot
reload or retained inverse history.

The focused controls establish:

- Pfaffian-square/determinant parity at `F103,n=3`;
- missing and wrong inverse detection;
- premature boundary-projection rejection;
- null-carrier and undeclared-field rejection;
- odd defect-sector rejection;
- alternate path agreement at `F103,n=4,k=4`;
- sensitivity to a flipped orientation edge; and
- exact fresh/restored reuse with generation `0 -> 1 -> 2`.

Reordered inverse failure is inapplicable because the terminal-entry and
scalar-accumulator additions commute.

## Resource audit

The package reports the actual trivalent and terminal graph sizes, sparse
resident cells, orientation equations, terminal load/unload additions,
scalar updates, Pfaffian pivots and field operations, final boundary payload,
and exact resident payload.

Dense elimination reports the caller matrix, elimination copy, and ten named
logical scalar work slots: `2 D^2 + 10` field cells.  Every named
field-valued Schur temporary is observed.  The reported payload number is an
upper bound over those named logical slots using the largest observed exact
field value.  Python containers and SymPy/native internal workspace are
explicitly excluded.  At exact `n=3`, the independently audited values are:

```text
named logical work cells          1,810
maximum observed field payload   1,518 bits
named logical payload bound      2,747,580 bits
```

## Repairs made during review

- Corrected dense work from one matrix to both coexisting matrices.
- Removed retained even-subset tuples from forward and inverse execution.
- Narrowed a defect-expansion requirement to the implemented upper bound.
- Corrected stale reverse-rematerialization wording.
- Distinguished final boundary projections from internal sector Pfaffian
  evaluations.
- Observed nested Schur products/differences before claiming a logical
  payload upper bound.

## Claim ceiling

This result establishes a bounded exact free-fermion terminal-Pfaffian phase
closure for the declared zero-field family, sparse-defect sector execution,
exact restoration, and restored-carrier reuse in direct-process software.

It does not establish compact dense-field closure, arbitrary matchgate or
holographic reduction, CATVM custody, a phase-native pivot/orientation law, a
distinct phase resource, computational advantage, a Small Wall crossing,
physical waveform execution, replacement of physical bits with pi, or
unbounded catalytic computation.  The strongest matched classical method is
the identical Gaussian/Pfaffian recurrence.
