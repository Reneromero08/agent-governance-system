# Hybrid Witness-Event Geometry Reduction

## Status

```text
HYBRID_GUARD_DIRECTIONAL_GEOMETRY_ESTABLISHED
CONDITIONAL_TRANSVERSE_MARGIN_AND_DWELL_TRANSFER_ESTABLISHED
HYBRID_EVENT_POLYNOMIAL_ACCELERATION_UPPER_BOUND_ESTABLISHED
NATIVE_SIMPLE_GUARD_GRAZING_STATE_ESTABLISHED
STRUCTURAL_TRANSVERSALITY_FALSE
PUBLIC_SEED_TRANSVERSE_LOWER_BOUND_NOT_ESTABLISHED
PUBLIC_SEED_ACTIVE_SET_GAP_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Semialgebraic guard geometry

For literal margins

```text
a_jr = q_jr c_i
```

and clause margins

```text
M_j(c) = max_r a_jr,
```

the exact hybrid witness guard is

```text
G_F(c) = min_j M_j(c).
```

Let

```text
B(c)   = argmin_j M_j(c)
A_j(c) = argmax_r a_jr(c).
```

For a phase-cosine velocity `dc`, the exact one-sided directional derivative is

```text
D G_F(c;dc)
= min_{j in B(c)} max_{r in A_j(c)} q_jr dc_i.
```

A classical derivative exists when the active clause and its active literal are both
unique. Tied clauses or literals remain directionally differentiable but are nonsmooth.

The executable audit classifies a point as:

```text
STRICT_WITNESS_REGION
STRICT_NON_WITNESS_REGION
SMOOTH_TRANSVERSE_WITNESS_ENTRY
SMOOTH_TRANSVERSE_WITNESS_EXIT
SMOOTH_GRAZING_OR_HIGHER_ORDER_EVENT
NONSMOOTH_DIRECTIONAL_WITNESS_ENTRY
NONSMOOTH_DIRECTIONAL_WITNESS_EXIT
NONSMOOTH_GRAZING_OR_HIGHER_ORDER_EVENT
```

## Active-set separation

At a simple event define:

```text
Delta_clause  = second-smallest clause margin - active clause margin
Delta_literal = active literal margin - second-largest literal margin
Delta         = min(Delta_clause, Delta_literal).
```

If `Delta > 0` and every phase coordinate has speed at most `V`, the same active
clause/literal branch persists for at least the conservative interval

```text
t_gap = Delta / (4 V).
```

Thus nonsmooth min/max switching is separated from ordinary transverse entry whenever
an inverse-polynomial active-set gap is available.

## Conditional margin and dwell transfer

Suppose a simple public-seed witness entry satisfies

```text
G_F(0) = 0
G_dot_F(0) >= kappa > 0
|G_ddot_F| <= A
active-set gap >= Delta > 0
phase-coordinate speed <= V.
```

Then set

```text
t_accel = kappa / (2 A)        when A > 0
tau     = min(t_gap, t_accel).
```

On `[0,tau]`, the active branch remains fixed and

```text
G_dot_F(t) >= kappa/2
G_F(t) >= kappa t / 2.
```

Therefore:

```text
witness dwell time >= tau
witness guard margin at tau >= kappa tau / 2.
```

Consequently, inverse-polynomial lower bounds on `kappa` and `Delta`, together with
polynomial upper bounds on `A` and `V`, imply inverse-polynomial witness margin and dwell
time.

This is a local conditional theorem. It does not prove such lower bounds on every
public-seed SAT trajectory.

## Polynomial upper bounds are available

The reduced carrier already has formula-uniform polynomial state and speed bounds.
Differentiating its explicit field term by term gives polynomial upper bounds on:

```text
clause-violation derivative
exact-gradient derivative
rigidity derivative
relational-force derivative
incident-violation derivative
angular acceleration
phase-cosine acceleration.
```

At a simple guard event the active branch is one signed phase cosine `q_i c_i`, so the
phase-cosine acceleration bound is also a guard-branch acceleration bound.

Thus the upper-bound half of the local hybrid event theorem is established.

## Structural transversality is false

The current native law admits an exact simple grazing control on the odd-parity formula.
Take

```text
c = (0,-1/2,-1/2)
z = (1,sqrt(3)/2,sqrt(3)/2)
short memory = 1 for every clause
long memory = 0 for every clause.
```

The witness guard has:

```text
G_F = 0
unique active clause = the all-positive clause
unique active literal = x1
active-set gap = 1/2.
```

Short memory one suppresses rigidity, long memory zero suppresses the exact-gradient
force, and `c1=0` suppresses boundary release in the active normal direction. Therefore

```text
D G_F(c; f_F(c)) = 0.
```

The complete carrier is nevertheless moving because selector and other phase coordinates
need not be stationary.

Hence transversality is not a structural consequence of the native law. It must be
proved specifically for the declared public-seed witness crossings.

## Exact remaining hybrid theorem

The hybrid lane is reduced to proving that every satisfiable public formula and every
declared presentation gauge has a public-seed event by polynomial time with

```text
transverse guard speed kappa >= 1/poly(|F|)
active-set gap Delta >= 1/poly(|F|).
```

The established polynomial speed and acceleration upper bounds would then transfer these
lower bounds into inverse-polynomial event margin and dwell time.

Still separate:

```text
polynomial event-location simulation
fail-closed handling of nonsmooth or grazing events
recorder write and readout cost
native restoration of event history
P = NP.
```
