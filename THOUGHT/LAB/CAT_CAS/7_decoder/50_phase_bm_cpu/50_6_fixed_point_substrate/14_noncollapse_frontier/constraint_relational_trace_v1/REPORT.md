# Constraint Relational Trace V1 Report

## Current Result

```text
REFERENCE_AND_MECHANISM_CAMPAIGN_IMPLEMENTED__CET_NATIVE_OPERATOR_NOT_ESTABLISHED
```

This package is a constructive `P = NP` proof attempt inside the CAT_CAS non-collapse
ontology. It now contains exact relational semantics, native parity holonomy,
program-derived inverse carrier actions, a complete topological SAT index, a constant
energy-margin Hamiltonian, an exceptional-point root-amplifier candidate, and explicit
resource audits for every tested shortcut.

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

### Exceptional-point root latch

An order-`n` companion with gain two per transport has cycle product:

```text
2^n * (#SAT/2^n) = #SAT.
```

UNSAT remains at an order-`n` exceptional point. SAT has spectral radius
`#SAT^(1/n) >= 1`, including a unique witness. The symbolic sensor uses `n` modes and
has a constant mathematical margin.

This is the strongest active mechanism candidate. Its unresolved obligations are:

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
invariants are complete.

```text
prepare or access the clean relational sector
-> enact exact determinant winding or zero-mode presence
-> amplify a unique witness without postselection or hidden exponential resources
-> totalized SAT/UNSAT/INVALID boundary
-> native inverse restoration of the complete environment
-> polynomial standard-model transfer
```

Equivalent compact statement:

```text
POLYNOMIAL_CLEAN_SECTOR_PRESENCE_LATCH_WITH_REVERSIBLE_ENVIRONMENT
```

## Current Claim Ceiling

```text
CONSTRAINT_RELATIONAL_TRACE_REFERENCE_ONLY__CET_NATIVE_OPERATOR_NOT_ESTABLISHED
```

## Environment Limitation

DevSpace rejected this conversation before workspace creation, so local worktree and
Codex CLI execution were unavailable. The branch was constructed and qualified through
GitHub. The dedicated hosted workflow is the executable qualification surface.
