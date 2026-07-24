# Constraint Relational Trace V1 Report

## Current Result

```text
REFERENCE_AND_MECHANISM_CAMPAIGN_IMPLEMENTED__CET_NATIVE_OPERATOR_NOT_ESTABLISHED
```

This package is a constructive `P = NP` proof attempt inside the CAT_CAS non-collapse
ontology. It now contains exact relational semantics, native parity holonomy,
program-derived inverse carrier actions, a complete topological SAT index, a constant
energy-margin Hamiltonian, an exceptional-point root-amplifier candidate, a direct
clause-local thermal boundary, a terminal-agnostic preparation flow, a public
polynomial-selector dilation of that flow, and explicit resource audits for every
tested shortcut.

The package does not yet implement a polynomial-resource native existential trace. It
does not prove `P = NP`.

## Established Foundation

### Open relational semantics

`ConstraintHolo` stores:

```text
boundary variables
constant-size local clause relations
allowed local rows
equality junctions
explicit unresolved native operator
explicit unresolved restoration law
```

The public record rejects answer-bearing fields. Clause order, literal order, duplicate
clauses, and bijective variable renaming are controlled. The DIMACS compiler is strict
and cannot accept clause data before the public problem declaration.

### Total reference boundary

For at most twenty variables, the exact reference backend returns:

```text
VALID_SAT with one independently verified witness
VALID_UNSAT
INVALID_CARRIER
```

It also reports witness count and the complete materialized provenance ledger. Unknown
or truthy malformed boundary values fail closed as `INVALID_CARRIER`. This backend is a
correctness instrument, not the native proof mechanism.

### Native parity holonomy

The incidence-only compiler selects a spanning forest without evaluating parity.
Borrowed vertex phase lanes execute actual `+1/-1` transport. Every non-tree edge closes
one cycle product:

```text
holonomy(u, v) = phase[u] * edge_transport(u, v) * phase[v].
```

A locally compatible but globally inconsistent triangle produces holonomy `-1`. Reverse
execution of the exact tree transports restores the carrier. The original union-find
shadow was removed.

### Conditional complexity theorem

The package proves:

```text
uniform exact polynomial CET decision
+ deterministic polynomial standard-model simulation
-> 3-SAT in P
-> P = NP.
```

A classical post-boundary self-reduction renders a conventional witness with at most
`1 + 2n` exact decision calls.

## Complete Invariants Found

### Topological determinant winding

For the exact solution projector `Q_F`:

```text
U_F(theta) = I + (exp(i theta)-1) Q_F.
```

The clean-sector determinant winding is:

```text
W_F = rank Q_F = #SAT(F).
```

Therefore `W_F != 0` is a complete SAT invariant. The materialized reference carrier
uses exact rational phase turns and has algebraically exact inverse restoration.

### Clause-Hamiltonian zero mode

The local commuting Hamiltonian:

```text
H_F = sum_j violation_projector(C_j)
```

has ground energy zero exactly for SAT and integer ground energy at least one for UNSAT.
This removes an exponential energy-precision gap at the mathematical level. The
remaining problem is detecting or populating a possibly unique zero mode.

## Exact Compensation and Width Results

### Reversible-oracle determinant compensation

The determinant of the complete compute-phase-uncompute circuit is formula-independent.
The clean sector carries `#SAT`; dirty ancillary sectors carry the exact compensating
winding. A valid sensor must isolate the clean determinant line.

### Historical MPO audit

The old MPO bond index `(control state, current symbol)` omits head position and complete
tape contents. Its winding belongs to a finite projected transition graph, not an exact
configuration carrier.

### Residual-relation width

Exact residual relations can require exponential bond width under one presentation and
constant width under another. Compact MPO or OBDD claims must therefore pass variable
order and presentation-gauge controls. No universal polynomial bond dimension is
established.

### Fermionic interaction wall

A genuine three-variable clause contributes a cubic occupation interaction. Generic
clause geometry therefore leaves free-fermion Gaussian determinant closure. Exact
non-Gaussian evolution or a no-smuggle auxiliary-field transformation is required.

## Sensor Candidates and Their Current Boundaries

### Normalized trace

A unique witness changes normalized trace by `2^(1-n)`. Constant-additive trace
estimation cannot see it.

### Filled determinant line

Known antisymmetric determinant constructions use resources polynomial in assignment
space dimension `N = 2^n`, which is exponential in public variables.

### Rank-one resolvent

A normalized uniform probe has zero-pole residue `#SAT/2^n`. An unnormalized probe gives
constant unique-witness residue only with norm squared `2^n`.

### Ideal zero-mode gain

Exponential gain can make settling time linear in `n`, but it transfers the exponent to
relative gain, pump, mode count, noise floor, or restoration unless a stronger physical
law is established.

### Polynomial selector dilation

The original terminal-agnostic flow uses local minimum operations. The new dilation
replaces each three-way clause minimum and each pairwise gradient minimum with conserved
replicator selectors. The native state has:

```text
n + 11m coordinates
polynomial degree <= 6
public rational coefficients and initial state
no native min, max, division, Heaviside, or hard projection
```

The Euler reference chart initially normalized all six pair-selector coordinates as one
simplex. That instrumentation bug was fixed. The native carrier and reference chart now
preserve three independent two-state pair simplexes per clause.

The selector carrier passed:

```text
all 256 three-variable formulae
all 255 satisfiable formulae with verified witnesses
the sole UNSAT formula with no false witness
distinct-terminal parity SAT and UNSAT controls
distinct-terminal pigeonhole SAT and UNSAT controls
distinct-terminal graph-coloring SAT and UNSAT controls
smooth adaptive parity integration with selector-mass drift near machine precision
```

Comparable adaptive parity observations were:

| core parity variables | public variables | clauses | result by t=5 | function evaluations | maximum long memory |
|---:|---:|---:|---|---:|---:|
| 8 | 24 | 32 | verified at 0.91155 | 536 | 16.46 |
| 16 | 48 | 64 | verified at 1.04221 | 548 | 18.54 |
| 24 | 72 | 96 | verified at 2.03552 | 12,680 | 724.37 |
| 32 | 96 | 128 | no solution by 5.0 | 198,140 | 79,841.87 |

The parity-32 carrier preserved selector masses to about `1e-13`, so the transition is
not a simplex-normalization failure. It exposes unresolved memory growth, stiffness, or
instanton/dwell time. No asymptotic claim follows from this finite series.

The exact unresolved theorem is a formula-uniform polynomial trajectory-length bound
from the declared public seed.

### Clause-local thermal zero-mode latch

For inverse temperature

```text
beta = (n + 2) ln 2,
```

the normalized Gibbs population of the zero-energy sector is at least `4/5` for every
satisfiable instance, including a unique witness. It is exactly zero for UNSAT. The
candidate compiles directly from the public clause Hamiltonian and does not assume a
precomputed `#SAT/2^n` coupling.

The exact missing step is polynomial worst-case preparation of that normalized Gibbs
state, followed by a deterministic total boundary and complete system-plus-bath
restoration.

### Terminal-agnostic self-organizing flow

The public flow uses one voltage per variable and two memory coordinates per clause.
Every term is compiled from one literal occurrence. Satisfying Boolean corners are
projected equilibria, nonsatisfying Boolean corners are not, and an answer-blind public
perturbation reaches small satisfying sections in the reference dynamics.

The smooth interior flow has a canonical cotangent Hamiltonian lift:

```text
H(q,p) = p dot f_F(q).
```

This gives a program-derived negative-time inverse. It also exposes that attractor
contraction in the visible relation can reappear as cotangent expansion, precision, or
environment energy. Global convergence, UNSAT totality, switching-surface reversal, and
polynomial cotangent range remain unresolved.

### Exceptional-point root latch

An order-`n` companion with gain two per transport has cycle product:

```text
2^n * (#SAT/2^n) = #SAT.
```

UNSAT remains at an order-`n` exceptional point. SAT has spectral radius
`#SAT^(1/n) >= 1`, including a unique witness. The symbolic sensor uses `n` modes and
has a constant mathematical margin.

This remains a secondary active mechanism candidate. Its unresolved obligations are:

```text
clean port intensity is 4^-n for a unique witness
deterministic noiseless gain is not established
gain/loss environment is not restored
noise below the pre-gain signal is not established
standard-model polynomial simulation is not established
```

A formula-blind deterministic CPTP amplifier cannot solve this, because trace distance
cannot increase. Formula-native nonlinear or nonstandard CAT_CAS dynamics remain open.

## Exact Missing Boundary

The remaining constructive target is now narrower than “find an invariant.” The
invariants and a constant-population normalized boundary are available.

```text
public clause geometry
-> polynomial selector dilation or another exact native preparation
-> uniform polynomial trajectory-length or equilibration theorem
-> deterministic zero-mode presence boundary
-> totalized SAT/UNSAT/INVALID result
-> native inverse restoration of the complete environment
-> polynomial standard-model transfer
```

Equivalent compact statement:

```text
POLYNOMIAL_SELECTOR_CLAUSE_FLOW_WITH_UNIFORM_DEADLINE_AND_REVERSIBLE_ENVIRONMENT
```

## Current Claim Ceiling

```text
CONSTRAINT_RELATIONAL_TRACE_REFERENCE_ONLY__CET_NATIVE_OPERATOR_NOT_ESTABLISHED
```

## Current Engineering Surface

DevSpace is available in an isolated worktree at the published PR head. Local focused
experiments and tests are now the primary engineering loop; GitHub remains the intended
publication and review surface.
