# Rank-One Resolvent Sensor Wall

## Probe Construction

Let `H_F` be the exact clause-violation Hamiltonian. Couple one probe mode through a
rank-one perturbation. The relevant scalar response is a resolvent matrix element:

```text
G_F(z) = <u|(z-H_F)^-1|u>.
```

A zero-energy satisfying subspace produces a pole at `z=0`.

## Normalized Uniform Probe

For the normalized uniform assignment probe:

```text
|u> = (1/sqrt(N)) sum_x |x>,
N = 2^n,
```

the residue of the zero pole is:

```text
Res_0 G_F = #SAT(F) / N.
```

A unique witness therefore has residue `2^-n`.

## Unnormalized Probe

Using the all-ones probe gives residue `#SAT(F)`, so a unique witness produces a
constant pole coefficient. But the probe norm squared is:

```text
<u|u> = N = 2^n.
```

The constant signal has been purchased with exponential carrier norm, source energy,
or an equivalent number of coherent assignment couplings.

## Rank-One Lemma Boundary

The matrix determinant lemma makes a rank-one update cheap only after the base
determinant and the required resolvent element are available. Here the hard object is
exactly:

```text
<u|(z-H_F)^-1|u>
```

across the complete unresolved relation. Caching this element from a conventional solve
would be answer smuggling. Computing it by materializing every basis mode would restore
the exponential carrier.

## Current Result

```text
RANK_ONE_DESCRIPTION_COMPACT
NORMALIZED_ZERO_POLE_RESIDUE_EXPONENTIALLY_SMALL
UNNORMALIZED_CONSTANT_RESIDUE_REQUIRES_EXPONENTIAL_PROBE_NORM
POLYNOMIAL_RESOLVENT_ACCESS_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
