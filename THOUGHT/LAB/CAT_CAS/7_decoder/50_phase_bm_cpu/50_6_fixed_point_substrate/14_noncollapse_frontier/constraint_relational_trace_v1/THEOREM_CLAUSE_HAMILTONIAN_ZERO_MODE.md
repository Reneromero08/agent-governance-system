# Clause-Hamiltonian Zero-Mode Latch

## Local Hamiltonian

For each public clause `C_j`, let `V_j` be the diagonal projector onto assignments that
violate that clause. Define:

```text
H_F = sum_j V_j.
```

Every term is positive semidefinite, diagonal, mutually commuting, and local to at most
three public variables. The public description contains one constant-size term per
clause.

For every assignment `x`:

```text
H_F |x> = number_of_violated_clauses(x) |x>.
```

Therefore:

```text
ground_energy(H_F) = 0  if and only if F is satisfiable
ground_energy(H_F) >= 1 if F is unsatisfiable.
```

Unlike normalized trace, the mathematical decision margin in energy is constant.

## Native Inverse

The phase evolution is exactly reversible:

```text
exp(-i t H_F)^-1 = exp(+i t H_F).
```

The local Hamiltonian and its inverse-time program have polynomial public descriptions.

## Remaining Population Wall

The zero-energy subspace has dimension `#SAT(F)` inside a boundary space of dimension
`2^n`. A neutral state that gives equal weight to all assignments therefore has zero-mode
weight:

```text
p_0 = #SAT(F) / 2^n.
```

For a unique witness, `p_0 = 2^-n`.

An ideal active instability can amplify this component in time proportional to
`log(1/p_0)`, which is linear in `n`. That does not establish polynomial total resources.
The relative intensity gain needed to move one unique zero mode from weight `2^-n` to a
constant fraction is of order `2^n`. A realization may instead materialize `2^n` modes,
use exponentially fine addressing, or supply equivalent unresolved gain resources.

This is a resource ledger, not a universal impossibility theorem for nonlinear
substrates. A valid CAT_CAS zero-mode latch must freeze and measure:

```text
carrier mode count
initial zero-mode seeding law
pump or gain energy
noise amplification
saturation law
settling time
readout margin
restoration after active gain
```

## Current Result

```text
COMMUTING_LOCAL_CLAUSE_HAMILTONIAN_EXACT
CONSTANT_SAT_UNSAT_ENERGY_MARGIN_ESTABLISHED
POLYNOMIAL_ZERO_MODE_POPULATION_OR_DETERMINANT_SENSOR_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
