# Exact-Product Carrier Reduction and Energy Wall

## Status

```text
PAIR_SELECTOR_SUBSYSTEM_DECOUPLED_IN_EXACT_PRODUCT_MODE
ACTIVE_CARRIER_REDUCES_FROM_2N_PLUS_11M_TO_2N_PLUS_5M
CLAUSE_PRODUCT_ENERGY_NOT_A_GLOBAL_LYAPUNOV_FUNCTION
AUGMENTED_SEED_SPECIFIC_FUNCTIONAL_OR_TOPOLOGICAL_ARGUMENT_REQUIRED
P_EQUALS_NP_NOT_PROVEN
```

## Pair-selector reduction

The selector-min predecessor used three two-state pair selectors per clause to
approximate the minimum of the other two literal defects. The exact-product direction
uses the public polynomial clause gradient directly:

```text
C_m = g d_m1 d_m2 d_m3 / 8
-GRAD_i C_m = g q_i d_mj d_mk / 8.
```

Therefore pair-selector coordinates do not enter:

- phase derivatives;
- short-memory derivatives;
- long-memory derivatives;
- clause-selector derivatives.

Changing every pair selector while fixing all other coordinates leaves the complete
active derivative byte-for-byte equal in the reference implementation. The pair
selectors continue to evolve only inside their own disconnected subsystem.

The active exact-product candidate can therefore be reduced from

```text
2n + 11m
```

to

```text
2n + 5m,
```

removing six dead coordinates per clause. This reduction does not establish global
convergence or change the claim ceiling.

## Raw clause energy is not monotone

The exact product gradient by itself descends the total clause-product energy. However,
the public boundary-release term is required to move violated Boolean corners away from
the phase boundary.

For one positive clause and the symmetric phase state

```text
c_x = c_y = c_z = 1/2
s_x = s_y = s_z = sqrt(3)/2
short memory = long memory = 1,
```

the configured clause energy is

```text
E = 1/16.
```

With the public release rate ten, the exact directional derivative satisfies

```text
dE/dt > 0.
```

Thus the raw sum of exact clause-product violations is not a global Lyapunov function
for the complete native carrier. When release is reduced to an arbitrarily small
positive control, the same state has `dE/dt < 0`, confirming that the increase comes
from the release mechanism rather than a sign error in the exact gradient.

## Consequence

A convergence proof cannot rely on either of these false structures:

1. the disconnected pair-selector subsystem;
2. monotone descent of raw clause-product energy.

The next theorem must construct one of:

- an augmented functional including phase release and memory;
- a seed-specific invariant region plus a polynomial transit bound;
- a topological instanton argument with explicit width, dwell, range, and restoration
  bounds.

The satisfiable stationary-point wall independently proves that the theorem cannot hold
for arbitrary initial carrier states. It must be stated for the declared answer-blind
public seed and all presentation gauges.
