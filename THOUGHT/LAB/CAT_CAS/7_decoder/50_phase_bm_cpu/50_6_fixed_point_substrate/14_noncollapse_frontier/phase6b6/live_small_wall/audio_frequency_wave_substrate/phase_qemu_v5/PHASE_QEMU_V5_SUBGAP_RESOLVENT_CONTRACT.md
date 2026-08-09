# Phase-QEMU V5 subgap boundary-resolvent scattering contract

## Purpose

M263 tests the smallest candidate left after M262 that is both noncentral and
not indexed by a supplied charge or prepared eigenlabel: a stipulated
single-channel boundary law below the excitation threshold of a many-body
target.  The algebraic target model has a growing interacting Krylov space
while the declared stationary boundary is one unit-modulus phase.

This package is a deterministic exact software hardware model.  It is not a
QEMU device execution, a time-domain scattering experiment, or physical
evidence.

## Exact stationary law

Let `H|g> = E_g|g>` with a unique target ground state and let
`<g|O|g> = 0`.  Define

```text
|chi> = O|g>
G(E)  = <g| O_dagger (E + E_g - H)^-1 O |g>
K(E)  = kappa G(E)
S(E)  = (1 - i K(E)) / (1 + i K(E)).
```

Below the target spectrum, `G(E)` and `K(E)` are real.  Therefore `S(E)` has
unit modulus exactly.  The software stipulates the formal one-port stationary
boundary law

```text
|E,in>|g> -> S(E)|E,out>|g>.
```

This is a K-matrix boundary model, not executed scattering.  It stipulates the
stationary input/output symbols; the production program does not evolve an
explicit waveguide, observe transient target population, wait for a packet to
leave, execute a finite-time inverse, or demonstrate restoration.

## Process-object separation

Any later device implementation must keep distinct:

- target carrier and its interacting excitation sector;
- probe waveguide and reference arm;
- preparation source and source-isolation state;
- controller and public Hamiltonian/coupling descriptor;
- detector and retained I/Q boundary;
- environment, loss channels, and restoration residual;
- any response cache or learned scalar lookup.

The guest must never supply `G(E)`, `S(E)`, a Bethe root/eigenphase label, or a
precomputed response table as the target state.  A scalar response table is a
required sham comparator, not a carrier model.

## Exact fixtures

At `E=0`, `kappa=1`, and coupling vector `e_1`, the exact stationary controls
are:

```text
H=[3]:
    G=-1/3
    S=(4+3i)/5

H=[[3,1],[1,3]]:
    G=-3/8
    S=(55+48i)/73

H=[[4,1,0],[1,4,1],[0,1,4]]:
    G=-15/56
    S=(2911+1680i)/3361.
```

Every value is reconstructed in `Q` or `Q(i)`.  Unit modulus and conjugate
inverse products are exact predicates, not floating-point tolerances.

The separate non-unit-`kappa` control checks that the Cayley law uses the
linear product `K=kappa G`, not an accidental `kappa=1` specialization:

```text
G=-1/3
kappa=2
K=-2/3
S=(5+12i)/13.
```

## Growing path control and exact compact shadow

The formula-generated family

```text
H_n = 4 I + adjacency(path_n),
n in {2,4,8,16,32},
|chi> = e_1
```

has exact Krylov rank `n`.  Split-prime rank certificates use `65537` and
`998244353`; a full-rank minor modulo either good prime certifies full rank
over `Q`.

The exact Green function nevertheless has the compact continuant law

```text
D_0=1
D_1=4
D_n=4 D_(n-1)-D_(n-2)
G_n(0)=-D_(n-1)/D_n.
```

Production independently checks dense rational solve, continuant recurrence,
and the scalar Lanczos continued fraction.  Only this executed path family has
the compact streamed `O(n)` moment/continuant comparator.  It establishes that
growing exact Krylov rank alone is not an approximation lower bound; it does
not establish a corresponding compact comparator for the interacting blocks.

## Interacting flagged virtual blocks

The many-body diagnostic uses a product-vacuum flag sector and an interacting
virtual block.  Preparation is

```text
|g> = |flag=0>|0^n>.
```

The flag-one block is

```text
A_n = sum_j X_j
    + 1/2 sum_j Z_j Z_(j+1)
    + sum_j ((j+1)^2/(n+2)) Z_j

J_n = n + (n-1)/2 + sum_j (j+1)^2/(n+2)
D_n = 2 J_n + 1
H_exc,n = D_n I + A_n.
```

The simultaneous transverse field, longitudinal inhomogeneous field, and
`ZZ` interaction remove the tested free-transverse-Ising and reflection
controls.  This is an interacting diagnostic, not a proof of nonintegrability
or classical hardness.

For `n=2..6`, dimensions `4,8,16,32,64`, both split primes certify full exact
Krylov rank.  Direct fraction elimination audits the smaller ranks.  Exact
dense rational resolvents are materialized only through `n=4`; larger exact
resolvents are deliberately not materialized merely to manufacture expensive
evidence.

No approximation-work, moment-generation, or compact-shadow conclusion is
drawn from these interacting ranks.  Their exact rank is a structural
diagnostic only.

## Path-only fixed-margin degree bound

For

```text
H_exc = Delta I + A
||A|| <= J
D = Delta-E > J
q = J/D < 1,
```

the exact Neumann law is

```text
(E-H_exc)^-1 = -1/D sum_(m>=0) (-A/D)^m.
```

With `mu_m=<chi|A^m|chi>`, truncation after order `K` obeys

```text
|G-G_K| <= ||chi||^2/D * q^(K+1)/(1-q)
|S-S_K| <= 2 kappa ||chi||^2/D * q^(K+1)/(1-q).
```

For the declared uniform path bound `J=2`, `D=4`, `q=1/2`, and
`epsilon=2^-20`, the smallest accepted order is `K=19`, or twenty retained
moments.  The exact path Krylov rank continues growing through 32 while this
fixed-margin, fixed-precision effective depth does not.

This is only a degree upper bound for the declared path family.  Production
does not count the work required to generate an arbitrary moment, does not
derive an approximation lower bound from Krylov rank, and does not apply this
path conclusion to the tilted-field blocks or to general interacting Green
functions.  Avoiding attenuation by moving near threshold must pay detuning,
condition number, Wigner delay, probe coherence, precision, loss, and any
eventual restoration latency explicitly.

## Finite-bandwidth control

For the coupled two-mode target, production compares exact energy bins
`E=0` and `E=1/2`.  The second phase is

```text
S(1/2)=(341+420i)/541.
```

For an equal coherent superposition of the two bins, the same-mode amplitude
is `(S(0)+S(1/2))/2`.  Its squared modulus is strictly below one.  Thus a
normalizable finite-bandwidth probe is not generally an unchanged probe mode
times one global phase.  The per-energy ground-sector output is part of the
stipulated stationary boundary law and was not executed as a target return.

No time-domain packet, pulse duration, Wigner delay, or residual target
population is executed here.

## Bethe factorization control

The negative control uses rational XXX pair phases

```text
s(u)=(u+i)/(u-i)
Phi(lambda_1,...,lambda_M)=product_(j<k) s(lambda_j-lambda_k)
lambda_j=j(j+1)
M in {2,4,8}.
```

The exact phase and inverse are public `O(M^2)` products of the rapidities
already needed to identify or prepare the Bethe state.  This kills the
factorized Bethe eigenphase as a resource candidate without claiming that all
interacting scattering is classically compact.

## No-restoration classification and descriptor reuse

The same coupled two-mode descriptor is evaluated at `E=0` and `E=1/2`.  Each
formula contains stipulated ground-sector input/output labels, and the second
evaluation uses no new target descriptor.  The exact scope is

```text
STIPULATED_FORMAL_STATIONARY_ONE_CHANNEL_BOUNDARY_AND_DISTINCT_ENERGY_DESCRIPTOR_REUSE_ONLY
```

Restoration classification:

```text
NO_RESTORATION_CLAIM
```

The stationary boundary law, time-domain restoration, inverse/echo,
same-backing custody, and physical restoration flags are all false.

The package does not establish:

- one resident target allocation or same-backing custody;
- time-domain target excursion and return;
- a finite-time echo or public physical inverse;
- a normalizable unchanged probe wavepacket;
- source-off isolation, losslessness, or physical channel closure;
- physical restoration or observation.

## Strongest honest comparators

The comparator set remains:

- exact scalar Lanczos/Jacobi continued fraction;
- fraction-free Krylov recurrence;
- determinant/cofactor or exact dense rational solve;
- truncated Neumann, Chebyshev, or kernel-polynomial moments;
- sparse shifted linear solvers;
- MPS correction-vector, dynamical-DMRG, and other tensor-network methods;
- free, Gaussian, stabilizer, symmetry, and integrability reductions;
- Bethe T-Q or pair-product recurrence;
- fixed-target/fixed-energy learned scalar response tables;
- the equal-access deterministic forward-only software shadow.

Exact Krylov rank, Hilbert dimension, descriptor coefficient height, matrix
nonzeros, effective precision depth, preparation, bandwidth, energy, latency,
controller state, query count, and rematerialization work must remain separate
resource coordinates.

`coefficient_payload_bits` counts only the nonzero coefficients in the
materialized input matrix.  It is not a descriptor-size, copied-descriptor,
intermediate-height, output-height, or whole-process measure.  The package
does not instrument:

- input exact payload height beyond that reported coefficient sum;
- intermediate and output exact payload heights;
- exact arithmetic operation work;
- formula descriptor size or descriptor copies;
- retained state or retained history;
- rematerialization and per-query/total-query work;
- controller and detector state;
- precision or shot count;
- whole-process liveness, memory, allocator, or serialization state;
- time-domain, bandwidth, energy, preparation, noise, loss, or calibration.

## Verification scope

```text
science:   SEPARATE_REFERENCE_PARITY
theory:    FORMAL_DERIVATION_SOURCE_AUDITED
resource:  PACKAGE_SELF_REVIEW
```

The production program's own assertion status is only
`SOURCE_SELF_CHECK_PASS`; it does not self-award strict independent parity.

## M257 guardrail

M257 remains fully active.  This package is deterministic software.  An
equal-access comparator can evaluate the accepted rational resolvent and
Cayley boundary directly while omitting any emulated transient or inverse.
Neither exact QEMU modeling nor a growing exact Krylov rank establishes a
physical resource advantage.

A valid escape would require an actual physical target or another resource
outside the same-domain deterministic-software assumptions, followed by an
honest comparison that counts target preparation, control, precision,
bandwidth, energy, Wigner delay, loss, and reuse stability.

## Claim, ceiling, and disposition

Claim:

```text
EXACT_RATIONAL_SINGLE_CHANNEL_SUBGAP_CAYLEY_RESOLVENT_DIAGNOSTIC_IMPLEMENTS_A_STIPULATED_FORMAL_STATIONARY_UNIT_MODULUS_BOUNDARY_LAW_AT_DECLARED_FIXTURES_WITH_DISTINCT_ENERGY_DESCRIPTOR_REUSE_GROWING_EXACT_KRYLOV_RANK_AND_PATH_ONLY_FIXED_MARGIN_EFFECTIVE_DEPTH_BOUND_PLUS_TILTED_FIELD_AND_BETHE_FACTORIZATION_CONTROLS
```

Claim ceiling:

```text
EXACT_DETERMINISTIC_SOFTWARE_FINITE_DIMENSIONAL_RATIONAL_ONE_CHANNEL_K_MATRIX_BOUNDARY_MODEL_WITH_FORMAL_STATIONARY_ASYMPTOTIC_RETURN_ONLY
```

Disposition:

```text
GROWING_EXACT_KRYLOV_RANK_ALONE_IS_NOT_AN_APPROXIMATION_LOWER_BOUND_PATH_FIXED_MARGIN_HAS_COMPACT_STREAMED_SHADOW_AND_BETHE_FACTORIZED_EIGENPHASE_IS_PUBLIC_RAPIDITY_PRODUCT_NEAR_THRESHOLD_TIME_DOMAIN_QUALIFICATION_REQUIRED
```

Next mechanism:

```text
NEAR_THRESHOLD_NONINTEGRABLE_BOUNDARY_RESOLVENT_WITH_EXPLICIT_WIGNER_DELAY_FINITE_BANDWIDTH_PRECISION_PREPARATION_AMORTIZATION_AND_TENSOR_NETWORK_RESOURCE_CROSSOVER
```

No QEMU device, CATVM custody, same-backing reuse, physical target, physical
restoration, computational advantage, M257 escape, Small Wall crossing,
unbounded computation, or bit-to-pi replacement is established.

## Primary references

- Xu and Fan, *Input-output formalism for few-photon transport: A systematic
  treatment beyond two photons*: https://doi.org/10.1103/PhysRevA.91.043845
- Van Dyke et al., *Preparing Bethe Ansatz Eigenstates on a Quantum Computer*:
  https://arxiv.org/abs/2103.13388
- Mei and Bolech, *Derivation of matrix product states for the Heisenberg spin
  chain with open boundary conditions*: https://arxiv.org/abs/1609.08045
- Bao et al., *Universal Quantum Computation by Scattering in the
  Fermi-Hubbard Model*: https://arxiv.org/abs/1409.3585
