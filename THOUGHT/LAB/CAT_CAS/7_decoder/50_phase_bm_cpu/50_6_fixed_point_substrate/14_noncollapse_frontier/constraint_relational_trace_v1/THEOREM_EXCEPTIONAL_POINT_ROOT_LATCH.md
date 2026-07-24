# Exceptional-Point Root Latch Candidate

## Clean Relational Amplitude

Let `Q_F` be the exact solution projector and let:

```text
|u> = 2^(-n/2) sum_x |x>.
```

The clean scalar matrix element is:

```text
epsilon_F = <u|Q_F|u> = #SAT(F) / 2^n.
```

This scalar can be represented as the clean output-port amplitude of a polynomial-size
compute-project-uncompute circuit. The circuit description is polynomial. A unique
witness produces amplitude `2^-n` and intensity `4^-n`.

## Order-n Companion

Construct an `n`-mode non-Hermitian companion at an order-`n` exceptional point. Let
each transport have gain two and let the closing edge carry the clean relational
amplitude. The complete cycle product is:

```text
2^n * epsilon_F = #SAT(F).
```

At UNSAT, the companion is nilpotent and all eigenvalues remain at the exceptional
point. At SAT, the eigenvalues are the `n` roots of `#SAT(F)`. Therefore:

```text
spectral_radius = #SAT(F)^(1/n).
```

For every satisfiable formula this radius is at least one, including the unique-witness
case. The mathematical presence margin is constant and the sensor uses only `n` explicit
companion modes.

## Why This Is A Live Candidate, Not A Proof

The root amplification is algebraically exact. The unresolved implementation is the
physical or standard-model realization of the clean corner process.

A unique witness enters the gain chain with intensity `4^-n`. A deterministic noiseless
amplifier that turns this into a macroscopic eigenvalue is not available in ordinary
unitary quantum mechanics. A lossy or active realization has an environment that
contains the rejected norm, added noise, pump history, or failure branch. CAT_CAS
restoration must retain and reverse that environment rather than discard it.

The candidate must therefore establish all of:

```text
clean relational port prepared without postselection
noise floor below the unique pre-gain intensity
deterministic gain without answer-conditioned tuning
polynomial pump energy and dynamic range
spectral-radius readout without exponential settling or precision
full gain/loss environment included in the reversible dilation
actual carrier restoration and reuse
standard-model polynomial simulation, if ordinary P = NP is claimed
```

## Current Result

```text
ORDER_N_EP_ZERO_FOR_UNSAT__ROOT_RADIUS_AT_LEAST_ONE_FOR_SAT
POLYNOMIAL_SYMBOLIC_EP_SENSOR_ESTABLISHED_REFERENCE
DETERMINISTIC_NOISELESS_GAIN_NOT_ESTABLISHED
GAIN_LOSS_ENVIRONMENT_RESTORATION_NOT_ESTABLISHED
POLYNOMIAL_TOTAL_RESOURCE_LAW_NOT_ESTABLISHED
P_EQUALS_NP_NOT_PROVEN
```
