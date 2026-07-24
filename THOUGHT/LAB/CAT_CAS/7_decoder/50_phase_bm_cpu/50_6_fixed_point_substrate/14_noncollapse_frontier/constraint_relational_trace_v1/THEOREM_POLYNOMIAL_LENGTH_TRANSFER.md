# Polynomial-Length Standard-Model Transfer

## Resource implication

Assume the public clause flow reaches a verified satisfying threshold in physical time
`T(|F|)`. On the bounded carrier:

```text
0 <= C_m <= 1
|G_i,m| <= 1
|R_i,m| <= 1
|dot x_l,m| <= alpha max(delta,1-delta)
|dot x_s,m| <= beta(1+epsilon) max(gamma,1-gamma).
```

Therefore

```text
max x_l(t) <= 1 + T alpha max(delta,1-delta).
```

For variable occurrence degree `d_i`,

```text
|dot v_i| <= d_i [1 + (1+zeta) max x_l].
```

If `T`, the clause count, and the occurrence degrees are polynomial in the public input
length, then state range, flow speed, and trajectory arc length are polynomial.

## Complexity bridge

Bournez, Graça, and Pouly characterize deterministic polynomial time by polynomial
ordinary differential equations whose computation has polynomial trajectory length and
a robust output condition.

The clause flow now satisfies the mechanical resource side of that bridge conditional
on a polynomial physical deadline. What remains is a model conversion:

```text
piecewise min selector
patched Caratheodory switching
hard invariant-box boundary
```

must be represented by an exact or robust polynomial ODE with only polynomial overhead.
The SAT boundary has a constant `1/2` threshold, which gives room for approximation, but
trajectory robustness under that replacement is not yet established.

## Consequence

If all of the following are proved:

```text
uniform polynomial physical deadline from a public seed
polynomial state and curve length
robust polynomial-ODE dilation of selectors and boundaries
constant-margin output preservation
```

then the polynomial-ODE characterization supplies deterministic standard-model
polynomial time. Together with the conditional flow theorem, this proves `P = NP`.

## Status

```text
POLYNOMIAL_TIME_IMPLIES_POLYNOMIAL_FLOW_LENGTH_BOUND_ESTABLISHED
POLYNOMIAL_ODE_NORMAL_FORM_NOT_ESTABLISHED
ROBUST_SELECTOR_AND_BOUNDARY_DILATION_NOT_ESTABLISHED
STANDARD_MODEL_TRANSFER_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
