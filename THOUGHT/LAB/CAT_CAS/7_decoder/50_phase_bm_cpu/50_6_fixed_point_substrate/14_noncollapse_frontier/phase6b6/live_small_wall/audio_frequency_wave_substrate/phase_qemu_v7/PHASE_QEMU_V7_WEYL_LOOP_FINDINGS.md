# Phase-QEMU V7 quantum Weyl-loop bus findings

## Authority and disposition

M265 is a positive restoration-law calibration and a negative computational-
resource result.

The exact package claim is:

```text
IDEAL_INFINITE_CCR_WEYL_COMMUTATOR_FACTORIZATION_WITH_ARBITRARY_NORMAL_STATE_BUS_IDENTITY_AND_FINITE_ENERGY_CONSTRAINED_TRUNCATED_FOCK_NUMERICAL_CONVERGENCE_ON_A_BOUNDED_THREE_QUBIT_CALIBRATION_LOAD
```

The exact claim ceiling is:

```text
DETERMINISTIC_COMPLEX128_SOFTWARE_EMULATION_WITH_LOGICAL_RESIDENT_ARRAY_CUSTODY_DIRECT_COMPILED_FORWARD_SHADOW_AND_NO_PHYSICAL_SAME_MODE_CUSTODY
```

The existing restoration taxonomy class is:

```text
NUMERICAL_PHYSICAL_STATE_RESTORATION
```

Its narrowly accepted scope is:

```text
ENERGY_CONSTRAINED_COMPLEX128_LOGICAL_RESIDENT_BACKING_BUS_AND_REFERENCE_RETURN_WITH_CLIENT_TRANSFORMATION_AND_COMPLETE_FACTORIZATION_AT_CUTOFF128_WITHOUT_PHYSICAL_SAME_MODE_CUSTODY
```

The taxonomy label denotes numerical process-state return. It does not assert
physical hardware execution, physical same-mode custody, or physical
restoration.

The exact negative resource disposition is:

```text
DIRECT_COMPILED_ZZ_FORWARD_SHADOW_STRICTLY_OMITS_THE_BUS_LOOP_AND_NO_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED
```

## The native interaction law

For one ideal oscillator bus,

```text
X = (a + a_dagger) / sqrt(2)
P = (a - a_dagger) / (i sqrt(2))
[X,P] = i
```

and commuting bounded client observables `A` and `B`, define

```text
Q_A(s) = exp(-i s A tensor X)
R_B(t) = exp(-i t B tensor P).
```

The central Weyl commutator gives

```text
Q_A(lambda) R_B(mu) Q_A(-lambda) R_B(-mu)
  = exp(-i lambda mu A B) tensor I_bus.
```

Because operator products act right-to-left, production executes the four
device pulses chronologically as

```text
R_B(-mu), Q_A(-lambda), R_B(+mu), Q_A(+lambda).
```

The bus is not a passive spectator. After the first pulse, the vacuum fixture
has client-bus mutual information `1.078985698292337` nats. Only the completed
phase-space rectangle removes the correlation. Reversing the rectangle returns
the bus but flips the sign of the client phase. Replacing both quadratures with
the same quadrature returns the bus but produces the identity client operation.
Those controls distinguish geometric area from source replay or a phase lookup.

In the ideal infinite canonical-commutation law the factorization is an
operator identity. It therefore returns the bus for every normal bus state and
even for an initially correlated client-bus state. That statement is analytic;
it is not attributed to a finite Fock simulation or physical oscillator.

## Two distinct programs on one logical backing

The bounded client calibration begins in `|+++>` and executes two programs on
the same client and bus backing:

| program | client operators | `lambda` | `mu` | `theta` |
|---|---|---:|---:|---:|
| A | `Z0`, `Z1` | `1/2` | `pi/4` | `pi/8` |
| B | `Z1`, `Z2` | `2/3` | `pi/4` | `pi/6` |

Program A increments the logical generation from zero to one. Its boundary is
a privileged nondestructive verification read, not a guest result. Program B
then consumes the actual A output in the same ndarray allocation and increments
the generation to two. Only the final combined boundary is released.

The full production JSON is a privileged scientific evidence envelope. A
record bearing the name `released_boundary` also contains verifier-only bus,
factorization, density-integrity, and resource diagnostics. Only its nested map
of selected client moments is the guest-visible boundary.

For every accepted fixture, production records:

```text
carrier supply count                       1
program count                              2
conditional pulse count                    8
client detach/replacement count            0
post-supply carrier state-set count        0
snapshot/reload/reinitialize count         0
allocation object/base-pointer changes     0
generation sequence                        0, 1, 2
guest boundary releases                    1
```

This establishes resident logical-array custody in one deterministic process.
It is not a QEMU PCI-device result, a VM migration result, authenticated
custody, or physical identity of one oscillator mode.

The final combined client moments are

```text
X0     = 1/sqrt(2)       X1     = 1/(2 sqrt(2))
X2     = 1/2             Y0 Z1  = 1/sqrt(2)
Z0 Y1  = 1/(2 sqrt(2))   Z1 Y2  = sqrt(3)/2
X0 X1  = 1/2             X1 X2  = 1/sqrt(2).
```

Production obtains these moments from the evolved joint state. A separate
reference independently exponentiates the projected oscillator generators by
Hermitian spectral decomposition, and a direct compiled client oracle applies
the two public `ZZ` phase gates. The three paths agree within the declared
complex128 tolerances.

## Finite cutoff is energy-constrained, never arbitrary-state

At cutoff `N`, the projected quadratures satisfy

```text
[X_N,P_N] = i (I_N - N |N-1><N-1|)
||[X_N,P_N] - i I_N|| = N.
```

No finite matrix pair can represent the canonical commutation relation
uniformly. M265 therefore executes cutoffs `16,32,64,128` and distinguishes two
classes explicitly:

- fixed low-energy fixtures that converge to the ideal Weyl result; and
- the cutoff-dependent state `|N-1>`, which remains a large counterexample.

The accepted fixtures are vacuum, a coherent state, a thermal state, a squeezed
vacuum, `|1>`, the non-Gaussian superposition
`(|0> + i|3>)/sqrt(2)`, and an inert-reference purification
`|Phi_4> = (1/2) sum_n |n>_bus |n>_reference`.

Here `energy-constrained convergence` means only these seven named fixture
sequences. It is not uniform convergence over an energy-bounded ball.
Production and the independent reference both require the complete-joint error
at `N=128` to be smaller than at `N=16` for the coherent, squeezed, thermal,
and `Phi_4` sequences.

At `N=128`, the maximum complete joint client-bus-reference factorization
Frobenius error is `2.336739139525e-15`, and the maximum combined bus-return
trace distance is `9.699833445253e-16`. Density trace, Hermiticity, and
positive-semidefinite tolerances also pass. These are numerical convergence
facts for the declared bounded suite, not exact finite-dimensional identities.

For every tested cutoff the top-Fock client and bus trace distances remain
above `0.5`. The separate reference independently reconstructs the growing CCR
defect and the same nonuniform-return obstruction. Any claim of finite-cutoff
arbitrary-state restoration is therefore false.

## Complete coherence, not a bus marginal

The `|Phi_4>` fixture keeps an inaccessible inert reference. Production checks
the complete returned bus-reference density and the complete
client-bus-reference factorization, rather than only the bus marginal.

A matched sham completely dephases the bus number basis. Its bus marginal
remains exactly `I_4/4`, while bus-reference trace distance becomes `3/4` and
entanglement fidelity falls to `1/4`. This is a decisive negative control:
marginal equality alone is not restoration of an unknown or referenced carrier.

The verifier necessarily retains a supplied-bus baseline and a separate full
bus-reference/process baseline for every fixture to compute these distances
and the complete factorization. With reference dimension one this is a second
`N x N` process baseline; `Phi_4` uses a full `4N x 4N` baseline. Those objects
are privileged validation state,
not available to the pulse dynamics or guest, and are charged explicitly. The
complete experiment is therefore not history-free even though pulse execution
retains no trajectory or inverse history.

## Error controls and physical meaning

The ideal loop is sensitive to closure and oscillator dynamics:

- omitting the closing pulse leaves bus trace distance above `0.05`;
- a five-percent final-pulse area error changes both client and bus states;
- free rotation `exp(-i 0.05 n)` between pulse legs changes both;
- Kerr evolution `exp(-i 0.02 n(n-1))` changes both; and
- snapshot reload after an omitted pulse restores a numerical baseline but is
  classified only as `SNAPSHOT_RELOAD`.

These are dimensionless synthetic controls. They are not measured noise rates,
physical observations, or proof that an experimental oscillator will close.
Finite pulse time, damping, thermal noise, spectator modes, phase-reference
drift, anharmonicity, controller quantization, emitted records, and calibration
history remain physical resources.

## Strongest honest comparator and M257

The strongest matched software comparator does not simulate the bus. It applies

```text
exp(-i theta_A Z0 Z1)
exp(-i theta_B Z1 Z2)
```

directly to the client and omits all bus pulses and all return verification. It
uses two client phase gates where M265 uses eight conditional bus pulses. For
the selected commuting graph-state moments, direct product formulas such as

```text
<X_i> = product_j cos(2 theta_ij)
```

are even smaller than a full client-vector recurrence.

For a growing graph with `E` commuting edges, the bus program uses `4E`
conditional pulses and the direct compiler uses `E` phase gates. The bus may be
a valuable connectivity and restoration mechanism in hardware, but the ideal
commutator itself does not force the classical forward shadow to represent the
bus trajectory. M257 remains controlling.

The bounded three-qubit client is not a phase-native client machine. Universal
non-Gaussian client dynamics would supply ordinary quantum-circuit complexity;
it would not make bus restoration itself a proven advantage. No asymptotic or
unbounded claim follows from the present calibration.

## Resource ledger

The production ledger counts the logical oscillator mode, cutoff, three client
qubits, complex128 component width, ensemble rank, canonical backing cells and
bytes, verifier baselines, dense pulse matrices, exponential calls, row-update
storage, public schedules, per-pulse mean excitation, top-edge population, and
the dimensionless rectangle excursion.

The maximum-rank mixed fixture uses `8 N^2` canonical complex cells. Dense
exponentials use quadratic storage and cubic work. The process RSS and wall
time are diagnostic only and explicitly excluded from claim-bearing seals.
NumPy/SciPy internal workspaces are uninstrumented. Joules, physical pulse
duration, phase-reference bandwidth, cooling, loss, calibration, controller
power, and tomography shots are uninstantiated until a physical backend is
specified.

The finite-cutoff state dimension and floating-point tolerance are paid
resources. A continuous oscillator is not treated as free infinite precision
or free unbounded memory.

## Physical translation and next mechanism

The closest demonstrated physical ingredients are trapped-ion state-dependent
forces on collective motional modes. They supply phase-selectable conditional
displacements and closed phase-space trajectories, but published client-gate
fidelity does not by itself establish arbitrary-state bus return, complete
bus-reference coherence, or two distinct same-mode programs without cooling or
reset. Circuit-QED conditional displacement and resonator-induced phase cycles
are a strong second backend; optomechanical and phononic platforms currently
provide carrier ingredients rather than a complete qualified Weyl loop.

The next Phase-QEMU mechanism is frozen as:

```text
MULTIMODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_WEYL_LOOP_DIGITAL_TWIN_WITH_HEATING_SPECTATOR_MODE_CLOSURE_CONTROLLER_COST_AND_ENERGY_CONSTRAINED_SAME_MODE_REUSE
```

It must model every normal mode, state-dependent force phases, simultaneous
closure constraints, heating, dephasing, residual displacement, spectator
coupling, pulse bandwidth, peak occupation, controller phase history, and
energy. It must retain the direct compiled client gate as the strongest
forward-only comparator. Physical execution or instrument control remains
outside current authority.

## Strict nonclaims

M265 does not establish physical execution, physical same-mode custody,
physical restoration, authenticated carrier custody, a finite-cutoff
arbitrary-state identity, a phase-native client architecture, a distinct
computational resource, total or asymptotic advantage, an M257 escape, Small
Wall crossing, unbounded compute, or replacement of the bit with pi.
