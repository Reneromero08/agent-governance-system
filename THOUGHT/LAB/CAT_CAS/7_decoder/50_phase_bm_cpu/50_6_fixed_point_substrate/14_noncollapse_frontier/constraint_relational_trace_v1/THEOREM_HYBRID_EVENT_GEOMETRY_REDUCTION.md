# Hybrid Witness-Event Geometry Reduction

## Status

```text
HYBRID_GUARD_DIRECTIONAL_GEOMETRY_ESTABLISHED
CONDITIONAL_TRANSVERSE_MARGIN_AND_DWELL_TRANSFER_ESTABLISHED
HYBRID_EVENT_POLYNOMIAL_ACCELERATION_UPPER_BOUND_ESTABLISHED
NATIVE_SIMPLE_GUARD_GRAZING_STATE_ESTABLISHED
NONSMOOTH_POSITIVE_WITNESS_CONE_ESTABLISHED
STRUCTURAL_TRANSVERSALITY_FALSE
ACTIVE_SET_GAP_SUFFICIENT_NOT_NECESSARY
PUBLIC_SEED_DIRECTIONAL_CONE_LOWER_BOUND_NOT_ESTABLISHED
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

## Simple-event active-set separation

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

Thus an inverse-polynomial active-set gap is one sufficient route for reducing the
nonsmooth guard to one ordinary signed phase coordinate.

## Conditional simple-event margin and dwell transfer

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

This is a sufficient local theorem, not a necessary event geometry.

## Active-set gap is not necessary

Odd three-bit parity has the exact witness ray

```text
c(epsilon) = (epsilon,-epsilon,-epsilon), epsilon > 0.
```

Every point on the ray is the verified `TFF` witness and satisfies

```text
G_F(c(epsilon)) = epsilon.
```

At the origin, all four clause margins tie at zero and several literal maxima tie. Thus

```text
active-set gap = 0.
```

Nevertheless the radial direction

```text
dc = (1,-1,-1)
```

gives every active clause directional derivative one, so

```text
D G_F(0;dc) = 1.
```

This is an exact nonsmooth directional witness entry. Therefore a positive active-set
gap is sufficient for the simple-event proof but is not necessary for robust directional
entry.

The general nonsmooth route must instead control a neighborhood lower derivative, for
example a lower Dini or equivalent directional-cone bound of the form

```text
D_lower G_F(c(t); f_F(c(t))) >= kappa > 0
```

throughout an interval after entry. Integrating such a bound yields

```text
G_F(c(t)) >= kappa t
```

without selecting one unique clause/literal branch.

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
phase-cosine acceleration bound is also a guard-branch acceleration bound. For a tied
guard, the same coordinate bounds control every finite active branch, although a
uniform lower directional-cone theorem is still missing.

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

The hybrid lane has two sufficient geometric routes.

### Simple-event route

For every satisfiable public formula and every declared presentation gauge, prove a
public-seed event by polynomial time with

```text
transverse guard speed kappa >= 1/poly(|F|)
active-set gap Delta >= 1/poly(|F|).
```

### Nonsmooth-cone route

Allow ties, but prove for an inverse-polynomial interval after entry that

```text
lower directional guard speed >= 1/poly(|F|)
```

uniformly over every active min/max branch that can control the trajectory.

The established polynomial speed and acceleration upper bounds would then transfer
either lower-bound route into inverse-polynomial event margin and dwell time.

Still separate:

```text
polynomial first-passage time
polynomial event-location simulation
fail-closed handling of unresolved grazing events
recorder write and readout cost
native restoration of event history
P = NP.
```
