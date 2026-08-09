# Phase-QEMU V8 multimode trapped-ion Weyl-loop findings

## Authority and disposition

M266 is a bounded digital-twin mechanism calibration and a class-level resource
obstruction for the public-descriptor, fixed-spin-axis, linear multimode
state-dependent-force family.

The exact package claim is:

```text
THREE_MODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_NULLSPACE_PULSES_CLOSE_ALL_NOMINAL_MODE_DISPLACEMENTS_AND_IMPLEMENT_TWO_DISTINCT_ZZ_PHASE_PROGRAMS_ON_ONE_LOGICAL_MULTIMODE_BACKING_WHILE_DECLARED_NONZERO_MARKOVIAN_HEATING_MONOTONICALLY_BREAKS_EXACT_INITIAL_MODE_STATE_RETURN_WITHOUT_RECOOLING
```

The exact claim ceiling is:

```text
DETERMINISTIC_COMPLEX128_FLOAT64_GAUSSIAN_MOMENT_SOFTWARE_DIGITAL_TWIN_WITH_DECLARED_LINEAR_HARMONIC_STATE_DEPENDENT_FORCE_AND_MARKOVIAN_ADDITIVE_HEATING_LAWS_NO_PHYSICAL_ION_CUSTODY_AND_DIRECT_COMPILED_CLIENT_CHANNEL_SHADOW
```

Restoration is classified:

```text
NO_RESTORATION_CLAIM
```

at the exact scope:

```text
NOMINAL_ZERO_HEATING_LOGICAL_GAUSSIAN_MODE_RETURN_ONLY_WITH_NONZERO_HEATING_EXACT_SAME_MODE_RETURN_REJECTED_AND_FRESH_MODE_SWAP_OR_RECOOLING_CLASSIFIED_AS_EXTERNAL_RESET
```

The exact negative resource disposition is:

```text
DIRECT_COMPILED_ZZ_AND_DEPHASING_CHANNEL_SHADOW_OMITS_THE_THREE_MODE_CONTROLLER_LOOP_WHILE_HEATING_PREVENTS_CATALYTIC_RETURN_SO_NO_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED
```

The result is not a physical trapped-ion observation. It establishes no
physical same-mode custody, physical restoration, authenticated carrier
identity, resource advantage, Small Wall crossing, unbounded computation, or
replacement of bits with pi.

## Physical process-object retained by the twin

The V8 model does not reduce the machine to one phase scalar. It keeps distinct
records for:

```text
carrier       three collective harmonic modes and their persistent logical arrays
client        three two-level ion coordinates
controller    drive frequency, eight-segment envelopes, phases, and calibration
source        the state-dependent force, isolated between programs
environment   additive heating and its accumulated quanta
detector      final client-observable extraction only
boundary      selected combined client moments
restoration   simultaneous closure of every conditional mode displacement
```

The actual phase coordinate is each mode's complex displacement in the force
phase frame. Useful client phase is the signed phase-space area, not a stored
answer or a controller lookup.

## Exact continuous-segment Magnus law

For fixed commuting client axes, the interaction has the declared form

```text
H(t)/hbar = sum_m c_m(Z,t)
            [a_m exp(-i delta_m t) + a_m_dagger exp(+i delta_m t)].
```

The commutator at two times is client-only and central. Magnus therefore ends
exactly at second order. Production integrates both parts of each constant
segment:

- its net conditional displacement;
- its intra-segment geometric area
  `c^2 (delta dt - sin(delta dt))/delta^2`.

It also accumulates the exact cross-segment area. Treating a segment as only an
instantaneous net displacement would omit the first term and is not accepted.

When every operator-valued displacement coefficient closes, the ideal law is

```text
U(T) = exp(+i sum_ij theta_ij Zi Zj) tensor I_all_modes.
```

This analytic identity is valid in the declared harmonic, fixed-axis,
Lamb-Dicke ideal model. It is not a finite-Fock or physical-hardware claim.

## Public three-mode descriptor and two programs

The bounded public device uses three modes at `1.900`, `1.918`, and `1.930`
MHz, a `1.911` MHz drive center, `120 us` programs, and eight equal `15 us`
segments. The stacked real/imaginary closure matrix has rank six and nullity
two. Its two public pulse shapes are distinct rather than scalar copies.
The near-degenerate frequencies are a synthetic calibration descriptor, not a
claim that the displayed eigenvector patterns and frequencies are the axial
spectrum of one realized harmonic trap.

Program A addresses client edge `(0,1)` and produces the signed Walsh
coefficient `-pi/8`. Program B addresses `(1,2)` and produces `-pi/6`.
Production includes the continuous within-segment area and obtains maximum
nominal branch closure below `1e-12` for both programs. The maximum mid-loop
conditional displacements exceed `0.7` and `0.85`, respectively, so the modes
are causally involved before closure.

The one logical multimode backing executes:

```text
one carrier supply
-> program A
-> privileged nondestructive diagnostic only
-> late program B on the actual A output and the same arrays
-> one final combined client boundary
```

There is no detach, state setting, snapshot, reload, reinitialization,
recooling, or fresh-mode swap between the programs. Stable NumPy object and
base-pointer identity establish logical software lineage only. Black-box
process tests cannot exclude a coherent swap into an isomorphic physical mode;
physical same-mode custody would require trusted out-of-band instrumentation.

## Nominal return and declared heating obstruction

Five named Gaussian moment fixtures cover vacuum, thermal, coherent, squeezed,
and bus-reference-correlated inputs. Under the zero-heating verifier control,
the closed displacement law returns their declared mode means, covariances,
and bus-reference correlations while the clients acquire the two intended
phases.

The claim-bearing environmental track instead declares additive Markovian
heating rates

```text
Gamma = [15, 30, 60] quanta / second.
```

After A and B the accumulated occupations are

```text
Delta n = [0.0036, 0.0072, 0.0144].
```

The corresponding covariance drift is approximately
`0.0233306665142683` in Frobenius norm, and the declared model adds
approximately `3.2097743958157e-29 J` to the three harmonic modes. The exact
seal pins the regenerated values.

The same public trajectory also produces a path-dependent client coherence
factor. With the standard convention
`D[L] rho = L rho L_dagger - {L_dagger L,rho}/2`, the declared generator is
`Gamma_m (D[a_m] + D[a_m_dagger])`; it gives both
`d<n_m>/dt = Gamma_m` and

```text
exp[-sum_m Gamma_m integral |beta_m(s,t)-beta_m(s',t)|^2 dt].
```

For the combined programs this moves the client density from the ideal direct
`ZZ` result by approximately `0.00425` trace distance. Endpoint geometric
closure therefore does not erase the environmental cost. Program B closes its
coherent displacement relative to the actual heated post-A input, but the
physical-mode state does not return to its initial covariance.

Recooling could prepare a known low-energy state again, but it exports entropy
and consumes a bath and controller. It is classified only as
`EXTERNAL_DISSIPATIVE_RESET`. Snapshot reload is `SNAPSHOT_RELOAD`, and a
fresh-mode swap is `CARRIER_REPLACEMENT`. None is catalytic restoration.

## Controls

The executed controls distinguish closure from controller replay or an
answer-bearing pulse table:

- omitting the final segment leaves a large mode displacement;
- optimizing only one target mode leaves spectator modes open;
- a common `+250 Hz` mode error breaks closure;
- twelve-bit controller quantization breaks exact closure;
- a one-percent error on one segment breaks closure;
- reversing the phase-space orientation changes the signed client phase;
- zero force has zero geometric area and no `ZZ` boundary;
- a bus-reference fixture rejects a marginal-only return claim;
- snapshot reload, recooling, and fresh-mode replacement remain explicit shams.

The separate reference reconstructs the displacement, continuous area,
heating exposure, and direct client boundary from the public descriptor without
importing production. Its fixed-step convergence is secondary to the analytic
continuous-segment recurrence and does not establish physical behavior.

## Strongest honest comparator and M257

The strongest exact ideal forward shadow applies

```text
exp(-i pi/8 Z0 Z1)
exp(-i pi/6 Z1 Z2)
```

directly to the client. It omits every mode, pulse, closure check, and return
stage. The strongest declared-heating shadow additionally applies the same
public diagonal coherence multiplier directly to the `8 x 8` client density.
It charges trajectory/exposure compilation but still stores no mode state and
performs no restoration.

More generally, fixed-axis linear SDF evolution closes on a displacement vector
and a client phase matrix. For `q` client coordinates and `M` modes, the exact
classical recurrence stores `O(qM + q^2)` scalars. A generic real segmented
controller pays at least `2M` closure constraints before imposing a nonzero
phase or robustness conditions. For `E` public client edges, the direct client
shadow stores and applies only the resulting edge phases and dephasing data.

This is a useful physical connectivity and restoration architecture, but not a
computational resource separation. Any difficult generic client evolution is
ordinary quantum-client complexity, not leverage supplied by the restorative
mode loop. M257 remains intact.

## Resource and physical accounting

The package reports all three physical modes, frequencies, eigenvectors,
Lamb-Dicke couplings, closure rank/nullity/condition, sixteen total segments,
controller literals and quantization, pulse synthesis work, peak drive,
integrated squared-amplitude proxy, maximum displacement, mode covariance,
heating quanta and modeled energy, client cells, privileged baselines,
descriptor storage, verifier work, custody counters, and guest boundary width.

The integral of squared drive amplitude is only a dimensionful controller
proxy. It is not optical/RF joules or wall-plug energy without a source-transfer
model. Physical calibration bits, laser power, scattering, trap control,
cooling cost, tomography shots, latency, bandwidth, Q, drift, anharmonicity,
and hardware reliability remain uninstantiated or explicitly modeled only as
synthetic controls.

The nearest demonstrated ingredients are multimode trapped-ion geometric gates,
power-optimized null-space pulses, and measured sensitivity to motional errors.
Relevant primary sources include the [exact multimode SDF/AESE treatment](https://arxiv.org/abs/2308.05865),
[power-optimal stabilized gates](https://www.nature.com/articles/s41534-021-00489-w),
[individually addressed multimode gates](https://www.nature.com/articles/s41467-024-53405-z),
and [analytic motional-error laws](https://doi.org/10.1103/PhysRevA.105.022437).
They establish ingredients, not this Phase-QEMU transaction or a physical
restoration result.

## Next mechanism

The linear commuting SDF family is retired as a long-term resource route once
the exact recurrence, direct compiler, controller scaling, and heating
obstruction are sealed. The next selected mechanism is:

```text
CONDITIONAL_GAUSSIAN_CLOSED_LOOP_FORWARD_SHADOW_AND_IRREVERSIBLE_DIFFUSION_NO_RETURN
```

Spin-dependent squeezing is still a conditional Gaussian operation, so it is
not accepted as a materially different mechanism merely because its generators
do not commute. The next package must close the whole public piecewise-
quadratic fixed-axis class: affine-symplectic identity on the bus/reference
implies a direct client-only forward channel, while positive additive diffusion
prevents exact initial bus/reference return without an external purification or
reset. Only after that class obstruction should the lane move to an actual
nonquadratic interaction or an exogenous restricted-access phase-eigenstate
query device.

The lane remains nonterminal.
