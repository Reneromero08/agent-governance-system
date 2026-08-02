# Reduced Phase Forward Resource Transfer

## Status

```text
ACTIVE_EXACT_PRODUCT_CARRIER_2N_PLUS_5M
STATE_RANGE_POLYNOMIAL_WITHOUT_DEADLINE_ASSUMPTION
POLYNOMIAL_DEADLINE_IMPLIES_POLYNOMIAL_FORWARD_TRAJECTORY_LENGTH
POLYNOMIAL_DEADLINE_IMPLIES_POLYNOMIAL_FORWARD_LOGIT_RANGE
PUBLIC_SEED_DEADLINE_NOT_ESTABLISHED
TOTAL_UNSAT_NOT_ESTABLISHED
RESTORATION_BOUND_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```

## Reduced carrier

After removing the dynamically disconnected pair selectors, the active exact-product
carrier contains:

```text
2n phase coordinates
m short-memory coordinates
m long-memory coordinates
3m clause-selector coordinates
```

for a total of

```text
2n + 5m
```

native coordinates. Its gauge-fixed observation chart contains `n + 4m` coordinates.

## Unconditional state-range bound

On the native carrier:

- every phase pair remains on `S1`;
- short memory remains in `[0,1]`;
- long memory remains in `[0,Km]` for the public cap factor `K`;
- every clause selector remains on a three-state simplex.

Therefore native state magnitude is polynomial in the public formula size without any
convergence assumption.

## Speed bound

Every literal defect lies in `[0,2]`. For truth gain `g`:

```text
0 <= C_m <= g
|exact gradient coordinate| <= g/2
|rigidity coordinate| <= 1.
```

A public variable occurs at most `3m` times. The resulting phase angular speed is bounded
by a polynomial in `m`; memory and selector speeds are also polynomially bounded. The
reference module emits an explicit formula-uniform Euclidean speed bound.

Consequently, if one public deadline satisfies

```text
T(F) <= polynomial(|F|),
```

then

```text
trajectory_length <= T(F) * polynomial(|F|)
```

for the forward native carrier.

## Precision bound

In logit and log-ratio coordinates:

```text
d short_logit / dt = beta (C_m - gamma)
d long_logit / dt = alpha (C_m - delta)
d clause_log_ratio / dt = selector_rate * (cost_j - cost_i).
```

The right-hand sides are uniformly bounded because `C_m` and literal defects are bounded.
Thus polynomial deadline implies polynomial logit range and polynomial forward precision
requirements.

## Consequence

For the reduced forward carrier, the main standard-model resource problem is no longer
state dimension, native range, raw speed, or logit precision. Conditional on a
formula-uniform polynomial deadline from the declared public seed, these resources are
polynomial automatically.

The remaining proof obligations are concentrated in:

1. proving the public-seed deadline and robust terminal margin for every satisfiable
   formula and every declared presentation gauge;
2. constructing a deterministic total UNSAT boundary;
3. bounding cotangent or environmental restoration resources.

This theorem is conditional and does not establish the deadline, total decision,
restoration, or `P = NP`.
