# Fermionic Gaussian Interaction Wall

## Occupation Representation

Represent each public Boolean variable by an occupation number `n_i` with Boolean
idempotence:

```text
n_i^2 = n_i.
```

A positive literal is violated by `(1-n_i)`. A negative literal is violated by `n_i`.
The local violation projector for one clause is the product of its three literal
violation factors.

For a genuine three-variable positive clause:

```text
(1-n_a)(1-n_b)(1-n_c)
```

contains the cubic interaction `-n_a n_b n_c`.

## Gaussian Boundary

Free-fermion determinant and Pfaffian methods close under quadratic Hamiltonians. A
quadratic or lower occupation polynomial can remain inside that Gaussian algebra. A
cubic local clause projector is interacting and is not represented by one Gaussian
single-particle determinant without an additional transformation.

Exact auxiliary-field or gadget transformations do not automatically solve the
problem. They must preserve the original relational object and account for every
auxiliary configuration, sign, postselection probability, precision requirement, and
restoration operation. Otherwise the interaction has only been moved into an implicit
sum.

## Local Versus Aggregate Cancellation

Cubic coefficients from different clauses can cancel in the aggregate polynomial. This
does not make the native local process Gaussian. The executable carrier must enact each
public local clause relation or an exactly equivalent transformed object. If the
transformation relies on cancellation among formula-specific interaction terms, it must
be frozen before execution and pass presentation-gauge controls.

The audit therefore records both:

```text
maximum degree of every local clause operator
maximum degree after formal summation
```

Gaussian closure is accepted only when both remain quadratic or lower.

## Consequence

The determinant-line invariant remains exact, but generic three-clause geometry is not
a free-fermion determinant problem. A successful fermionic CAT_CAS mechanism requires:

```text
an exact non-Gaussian native interaction
or
an exact polynomial auxiliary-field transformation with no expanded hidden sum
```

## Current Result

```text
GENERIC_GAUSSIAN_DETERMINANT_CLOSURE_BROKEN_BY_INTERACTIONS
EXACT_AUXILIARY_FIELD_SUM_OR_NON_GAUSSIAN_NATIVE_OPERATOR_REQUIRED
P_EQUALS_NP_NOT_PROVEN
```
