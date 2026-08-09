# Phase-QEMU V7 Quantum Weyl-Loop Bus Contract

## Scope

M265 tests the smallest history-free geometric-loop law that can transform a
coherent client while returning an oscillator bus.  It changes the M264 open
drain mechanism: the useful client phase and bus return are produced by one
closed Weyl commutator rather than by waiting for a finite packet to leave.

The three qubit clients are a bounded non-Gaussian calibration load.  They are
not a phase-native client architecture, `REPLACE_THE_BIT_WITH_PI`, unbounded
compute, or a claim that the bus supplies the clients' quantum resource.

The production program is deterministic complex128 software emulation.  Stable
NumPy allocation identity establishes only logical resident custody inside one
process.  It does not establish physical same-mode custody, physical execution,
physical restoration, or a resource unavailable to an equal-access software
comparator.

## Exact ideal law and pulse sign

Let

```text
X = (a + a_dagger) / sqrt(2)
P = (a - a_dagger) / (i sqrt(2))
[X,P] = i
Q_A(s) = exp(-i s A tensor X)
R_B(t) = exp(-i t B tensor P)
```

For bounded commuting client operators `A` and `B`, set

```text
C = -i lambda A tensor X
D = -i mu B tensor P
[C,D] = -i lambda mu A B tensor I_bus.
```

The commutator is central.  The Weyl relation, rather than an unqualified
formal BCH expansion on unbounded operators, gives

```text
U_loop
  = Q_A(+lambda) R_B(+mu) Q_A(-lambda) R_B(-mu)
  = exp(-i lambda mu A B) tensor I_bus.
```

Operator products act right-to-left, so the chronological device commands are

```text
R_B(-mu)
Q_A(-lambda)
R_B(+mu)
Q_A(+lambda).
```

Reversing the rectangle returns the ideal bus but changes the client gate to
`exp(+i lambda mu A B)`.  A sign-sensitive `YZ` boundary is mandatory.

Because the complete operator factorizes, the ideal infinite-CCR bus action is
the identity for every normal bus state and, more strongly, for arbitrary
initial client-bus correlations.  This is an ideal mathematical law.  It is not
the finite-cutoff claim.

## Finite-cutoff obstruction and accepted scope

At Fock cutoff `N`, production exponentiates the actually projected matrices
`X_N` and `P_N`.  They obey

```text
[X_N,P_N] = i (I_N - N |N-1><N-1|)
||[X_N,P_N] - i I_N|| = N.
```

No finite-dimensional matrices obey the canonical commutation relation
uniformly: the trace of a commutator is zero.  Consequently V7 must not claim
arbitrary-state finite-cutoff return.  Its numerical promotion is restricted to
the declared fixed, energy-constrained fixtures at `N=128`.  The cutoff sweep
`16,32,64,128`, top-edge population after every pulse, and the adversarial
`|N-1>` input expose this nonuniformity.  The top-Fock input must remain at least
`0.5` away in both client and returned-bus trace distance.

## Programs and client boundary

The client begins in `|+> tensor |+> tensor |+>`.  Two distinct public programs
run consecutively without detaching or replacing the client and without
reloading the bus:

| Program | `A` | `B` | `lambda` | `mu` | `theta=lambda*mu` |
|---|---:|---:|---:|---:|---:|
| A | `Z0` | `Z1` | `1/2` | `pi/4` | `pi/8` |
| B | `Z1` | `Z2` | `2/3` | `pi/4` | `pi/6` |

Program A's boundary is a privileged nondestructive verifier read.  It is not
released to the guest.  Program B consumes the same client+bus process state;
only the final combined boundary is released.  Thus there is no detach/attach
CPTP reconstruction between the claimed transactions.

The direct compiled oracle is derived independently by applying the public
diagonal gates `exp(-i theta Z_a Z_b)` to the client.  The analytic moments are
an additional sign and composition check, never the sole oracle.

Program A:

```text
X0 = 1/sqrt(2)       X1 = 1/sqrt(2)       X2 = 1
Y0 Z1 = 1/sqrt(2)    Z0 Y1 = 1/sqrt(2)    X0 X1 = 1
```

Combined A then B:

```text
X0 = 1/sqrt(2)       X1 = 1/(2 sqrt(2))   X2 = 1/2
Y0 Z1 = 1/sqrt(2)    Z0 Y1 = 1/(2 sqrt(2))
Z1 Y2 = sqrt(3)/2    X0 X1 = 1/2          X1 X2 = 1/sqrt(2)
```

The guest-visible boundary contains only these selected client moments.  Bus
coordinates, ensemble components, and verifier baselines are not guest
boundaries.

The production JSON is a privileged scientific evidence envelope, not the
guest response wire format.  Records named `released_boundary` include return,
factorization, density-integrity, and resource diagnostics for qualification;
only their nested `boundary` map is the declared guest-visible payload.

## Canonical logical backing and evolution

Each fixture supplies one fixed ndarray with axes

```text
(mixed-state ensemble component, client basis, bus Fock level, inert reference).
```

Every component retains the complete coherent client superposition.  Applying
the same conditional bus unitary to each component is an exact ensemble
representation of the mixed density operator, not a stochastic sample.  Every
chronological pulse mutates this backing in place.  Temporary row-update
buffers, cached pulse matrices, the ensemble rank, and the full allocation are
charged.

The backing object and base pointer must remain unchanged across A and B.  The
accepted transaction ledger is:

```text
carrier supplies                         1
programs                                 2
conditional pulses                       8
client detach/replacement                0
post-supply carrier state setting        0
snapshot/reload/reinitialize             0
privileged midpoint boundary reads       1
guest boundary releases                  1
generation receipts                      0 -> 1 -> 2
```

The verifier retains a separate full initial bus density and a separate full
initial bus-reference/process density for every fixture, solely to measure bus
return and complete factorization.  When the inert-reference dimension is one,
the second object is another `N x N` process baseline rather than a distinct
physical reference; for `Phi_4` it is the full `4N x 4N` bus-reference density.
Both objects are counted and neither is readable by pulse execution.  Their
existence prevents a history-free complete-experiment claim.  Production
reports zero retained dynamic trajectory cells and zero inverse-history
entries separately from eight retained public schedule entries, its pulse
cache, and both privileged validation baselines.  Stable software allocation
identity and verifier separation do not establish physical same-mode custody.

## Fixtures and coherence-sensitive return

Every cutoff executes:

```text
vacuum
coherent alpha = 0.65 + 0.20 i
thermal nbar = 0.4
squeezed vacuum r = 0.45, angle = 0.30
Fock |1>
non-Gaussian (|0> + i |3>) / sqrt(2)
|Phi_4>_BR = (1/2) sum_{n=0}^3 |n>_bus |n>_reference
```

The inert reference is never a pulse target.  At `N=128`, the complete returned
bus-reference density must be within trace distance `1e-9` of its input and its
entanglement infidelity must be at most `1e-9`.  This prevents a bus-marginal-only
claim.  The matched dephasing sham preserves the `I_4/4` bus marginal exactly
while producing bus-reference trace distance `3/4` and entanglement fidelity
`1/4`.

For the vacuum, the first chronological conditional displacement supplies a
causal-interaction witness before the loop closes.  With `mu=pi/4`, the two bus
branches have overlap

```text
exp(-mu^2) = exp(-pi^2/16),
```

and the pure client-bus mutual information is

```text
2 h((1 + exp(-pi^2/16))/2)
  = 1.078985698292337 nats.
```

Production requires a numerical value of at least `1.0` nat.  Final return does
not erase the fact that the bus was causally involved at the midpoint.

## Prospective numerical gates

At cutoff `N=128`, every declared energy-constrained fixture must satisfy:

```text
client trace distance to direct compiler, after A        <= 1e-10
client trace distance to direct compiler, after A then B <= 1e-10
bus trace distance to supplied state, after A            <= 1e-9
bus trace distance to supplied state, after A then B     <= 2e-9
selected boundary maximum absolute error                 <= 1e-10
complete joint-to-compiled-client tensor supplied-BR
  Frobenius distance, after A and after A then B          <= 2e-9
density trace error                                      <= 1e-12
density Hermiticity error                                <= 1e-12
minimum density eigenvalue                               >= -1e-12
```

All density-integrity gates are fail-closed for client, bus, and bus-reference
claim-bearing objects.  Uhlmann fidelity is not used; the pure-reference
entanglement fidelity is explicitly `Tr(rho_BR_initial rho_BR_returned)`.

The word `convergence` is restricted to the seven named fixture sequences, not
to a uniform theorem over an energy-bounded ball.  Production and the separate
reference both require the combined complete-joint error at `N=128` to be
strictly smaller than at `N=16` for the coherent, squeezed, thermal, and
`Phi_4` sequences.  The top-Fock family supplies the nonuniform counterexample.
The complete joint factorization metric is evaluated exactly one
bus-reference density block at a time from the ensemble representation.  It
includes every client, bus, and inert-reference coherence, so matching
marginals cannot hide residual correlations.  The largest verifier block is
charged explicitly.

## Controls

The minimum executed controls are:

1. Reverse the rectangle.  The bus returns, while `Y0 Z1` flips sign and differs
   from the accepted boundary by at least `1.0`.
2. Replace every `P` pulse with `X`.  Commuting translations return the bus but
   produce no geometric client phase: the client remains `|+++>` and `Y0 Z1=0`.
3. Omit the final chronological `Q_A(+lambda)` pulse.  Bus trace distance must be
   at least `0.05`.
4. Reload the pre-loop ndarray after the omitted pulse.  This must restore the
   numerical state, record one snapshot and one reload, and remain classified
   `SNAPSHOT_RELOAD`, never native restoration.
5. Multiply only the final chronological Q-pulse area by `1.05`.  Client and bus
   trace distances must each exceed `1e-4`.
6. Insert `F=exp(-i 0.05 n)` between each adjacent pair of pulses.  Client and
   bus trace distances must each exceed `1e-3`.
7. Insert `K=exp(-i 0.02 n(n-1))` between each adjacent pair of pulses.  Client
   trace distance must exceed `1e-4` and bus trace distance `1e-3`.
8. Execute the `|Phi_4>` marginal-dephasing sham and the top-Fock counterexample.

The rotation, Kerr, and pulse-area values are dimensionless synthetic controls.
They are not measurements or calibrated physical noise rates.

## Strongest comparator and resource law

The equal-access direct comparator applies two public compiled `ZZ` phase gates
instead of eight conditional bus pulses.  For the selected moments of a
commuting weighted graph state it can also use compact product formulas such as

```text
<X_i> = product over incident j of cos(2 theta_ij).
```

For a growing family with `E` commuting edges, this bus architecture pays `4E`
conditional pulses while the direct compiler pays `E` client gates.  Generic
client-state and boundary costs remain.  The bus does not supply a computational
advantage in the tested software domain.

The evidence ledger reports:

```text
physical carrier modes in the model
Fock cutoff N
three-client dimension
complex128 precision
mixed-state ensemble rank
canonical allocation cells and bytes
verifier baseline cells
cached pulse matrices and exponential calls
temporary row-update cells
per-pulse mean excitation and top-edge probability
dimensionless maximum rectangle excursion
control traffic and pulse count
process RSS and non-claim wall time
uninstrumented linear-algebra scratch
```

For the maximum-rank mixed fixture, the exact ensemble backing uses `8 N^2`
complex cells.  Cached dense exponentials and matrix multiplication use
quadratic storage and cubic work.  Joules, physical pulse duration, analog
bandwidth, and physical precision are uninstantiated until a hardware mapping
exists; simulator numbers must not be relabeled as those resources.

## Claim and ceiling

Accepted package claim:

`IDEAL_INFINITE_CCR_WEYL_COMMUTATOR_FACTORIZATION_WITH_ARBITRARY_NORMAL_STATE_BUS_IDENTITY_AND_FINITE_ENERGY_CONSTRAINED_TRUNCATED_FOCK_NUMERICAL_CONVERGENCE_ON_A_BOUNDED_THREE_QUBIT_CALIBRATION_LOAD`

Claim ceiling:

`DETERMINISTIC_COMPLEX128_SOFTWARE_EMULATION_WITH_LOGICAL_RESIDENT_ARRAY_CUSTODY_DIRECT_COMPILED_FORWARD_SHADOW_AND_NO_PHYSICAL_SAME_MODE_CUSTODY`

Restoration classification, using the existing claim-registry taxonomy:

`NUMERICAL_PHYSICAL_STATE_RESTORATION`

This taxonomy label concerns numerical process-state return.  It does not
establish physical hardware execution or physical same-mode custody.

Restoration scope:

`ENERGY_CONSTRAINED_COMPLEX128_LOGICAL_RESIDENT_BACKING_BUS_AND_REFERENCE_RETURN_WITH_CLIENT_TRANSFORMATION_AND_COMPLETE_FACTORIZATION_AT_CUTOFF128_WITHOUT_PHYSICAL_SAME_MODE_CUSTODY`

Not established:

- uniform arbitrary-state return at any finite cutoff;
- physical execution, physical same-mode custody, or physical restoration;
- a phase-native client architecture or replacement of physical bits with pi;
- a resource advantage over direct compiled gates or compact moment formulas;
- an escape from M257;
- growing or unbounded compute, Small Wall crossing, or lane completion.

M265 is a restoration-law calibration and obstruction package.  It establishes
that an ideal Weyl commutator can return its bus while transforming a client,
and simultaneously establishes that finite cutoff, control error, noise, and a
strictly cheaper direct forward shadow remain controlling barriers.

The exact negative resource disposition is:

`DIRECT_COMPILED_ZZ_FORWARD_SHADOW_STRICTLY_OMITS_THE_BUS_LOOP_AND_NO_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED`

The selected next mechanism is:

`MULTIMODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_WEYL_LOOP_DIGITAL_TWIN_WITH_HEATING_SPECTATOR_MODE_CLOSURE_CONTROLLER_COST_AND_ENERGY_CONSTRAINED_SAME_MODE_REUSE`
