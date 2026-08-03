# Public-Seed Boundary-Stratum Escape Audit

## Status

```text
GENERIC_BOUNDARY_STRATUM_POINTWISE_AUDIT_IMPLEMENTED
KNOWN_SYMMETRIC_NON_SOLUTION_STATIONARY_POINT_REMAINS_CENTER_UNRESOLVED
SHORT_ZERO_SYMMETRIC_STRATUM_HAS_PUBLIC_REPELLING_DIRECTION
SATISFYING_BOOLEAN_SECTION_VERIFIED_INVARIANT
GLOBAL_MOVING_SET_CLOSURE_NOT_ESTABLISHED
PUBLIC_SEED_COMPLETENESS_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Scope

This checkpoint implements the first structural gate from
`AGENT_TASK_PUBLIC_SEED_BOUNDARY_ESCAPE.md`.

It does not enumerate all global combinations of memory classes. It evaluates a proposed
state compositionally, one clause at a time, against the five public-seed-compatible
memory strata:

```text
SHORT_ZERO__LONG_ZERO__C_ARBITRARY
SHORT_ZERO__LONG_CAP__C_ARBITRARY
SHORT_ONE__LONG_CAP__C_ARBITRARY
SHORT_INTERIOR_C_EQ_GAMMA__LONG_CAP
SHORT_ZERO__LONG_INTERIOR_C_EQ_DELTA
```

The executable reference audit is:

```text
boundary_stratum_escape_audit.py
```

with focused controls in:

```text
test_boundary_stratum_escape_audit.py
```

## Exact local quantities

For each clause, the audit evaluates the configured exact product violation

```text
C = truth_gain * d_1 d_2 d_3 / 8
```

and its Lie derivative along the reduced exact-product phase field:

```text
C_dot = sum_occurrences (partial C / partial c_i) c_i_dot.
```

Occurrence contributions are summed separately, so repeated-variable clauses are not
silently treated as clauses with three independent variables.

The four memory-boundary normal exponents are reported explicitly:

```text
short from zero:  beta (C-gamma)
short from one:  -beta (C-gamma)
long from zero:   alpha (C-delta)
long from cap:   -alpha (C-delta).
```

For each excluded selector coordinate on a declared selector face, the normal exponent is

```text
selector_rate * (supported_weighted_cost - excluded_cost).
```

Supported selector costs must agree with their supported weighted cost. A declared
support that omits positive mass or includes a zero-mass coordinate is rejected rather
than repaired.

## Classification contract

A proposed state is classified as one of:

```text
NOT_INVARIANT
INVARIANT_REPELLING_IN_AT_LEAST_ONE_PUBLIC_DIRECTION
INVARIANT_STABLE_OR_CENTER_UNRESOLVED
NON_SOLUTION_ATTRACTOR_CANDIDATE
SATISFYING_INVARIANT_SECTION
```

The classifications have deliberately different strength.

### Stationary point

A stationary-point proposal must satisfy:

```text
memory-stratum membership
memory field zero
C_dot = 0 for every interior-memory clause
selector support and stationarity
complete reduced phase field zero.
```

### Moving invariant set

A moving-set proposal may have nonzero phase velocity, but the audit establishes only
pointwise boundary tangency. It always records:

```text
global moving-set closure not certified by pointwise tangency.
```

A single tangent point is never narrated as a global invariant orbit or manifold.

### Asymptotic numerical approach

A numerical approach is non-certifying. A small residual does not become an exact
invariant through the classification interface.

## Separating controls

### Known symmetric non-solution stationary state

For

```text
(x or y or y) and (~x or ~y or ~y)
```

at

```text
c_x = c_y = 0
z_x = z_y = 1
short memory = 1
long memory = L
clause selectors uniform,
```

the reduced field is stationary and the threshold assignment is not satisfying.

Each clause has

```text
C = 1/2
gamma = 1/4
delta = 1/20.
```

The short-one and long-cap memory normals are strictly stable. However, all three
literal defects are equal, so every point of the supported selector simplex is
stationary. The selector subsystem therefore supplies center directions.

The correct classification is:

```text
INVARIANT_STABLE_OR_CENTER_UNRESOLVED
```

not a certified attractor and not a proof against the declared public seed.

### Same phase geometry on short-zero memory

Keeping the same symmetric phase and selector geometry but placing each short memory at
zero gives

```text
lambda_short_from_zero = beta(1/2-1/4) > 0.
```

The point remains stationary on the boundary, but it is repelling in a public memory
direction. The correct classification is:

```text
INVARIANT_REPELLING_IN_AT_LEAST_ONE_PUBLIC_DIRECTION.
```

This demonstrates why the five memory labels cannot be treated as equivalent copies of
one stationary obstruction.

### Satisfying Boolean section

A satisfying Boolean corner with selector support concentrated on an exactly satisfied
literal has zero phase, memory, and selector field. Terminal public verification accepts
its threshold assignment. The classification is:

```text
SATISFYING_INVARIANT_SECTION.
```

### Repeated-literal moving control

For a repeated clause `(x or x or x)` at the unresolved phase midpoint, the audit sums all
three occurrence derivatives. The boundary memory coordinates can be pointwise tangent
while the phase field moves and `C_dot` is nonzero. A stationary-point proposal is
rejected; a moving-set proposal remains only pointwise tangent with global closure
unresolved.

## Fail-closed boundary

The audit raises `ConstraintHoloError` on:

```text
state dimension mismatch
unsupported stratum labels
NaN or infinity
phase-circle violation
memory coordinates outside their public ranges
non-simplex selector weights
incomplete or inconsistent selector support
unsupported proposal kinds.
```

It does not project, renormalize, infer missing support, or replace invalid data.

## What this establishes

This checkpoint establishes a reusable exact-formula reference surface for testing
candidate non-solution boundary strata. It separates:

```text
failure of pointwise tangency
pointwise repulsion
stationary center obstructions
candidate attraction
satisfying invariant sections
moving-set closure obligations.
```

It also shows that the already known symmetric stationary obstruction is not eliminated
by memory-normal analysis alone because selector center directions remain.

## What remains unproved

The checkpoint does not establish:

```text
that the declared public seed approaches the known symmetric obstruction
that the declared public seed avoids that obstruction
closure or stability of any moving invariant set
exclusion of all five forward-compatible global boundary classes
compactness of every omega limit
formula-uniform neighborhood escape
polynomial escape time
robust terminal margin
public-seed completeness
P = NP.
```

The next structural step is to use the audit in a capped small-formula search and to
analyze whether the declared public seed can lie in the stable set of the surviving
stationary-center obstruction or any other certified candidate.
