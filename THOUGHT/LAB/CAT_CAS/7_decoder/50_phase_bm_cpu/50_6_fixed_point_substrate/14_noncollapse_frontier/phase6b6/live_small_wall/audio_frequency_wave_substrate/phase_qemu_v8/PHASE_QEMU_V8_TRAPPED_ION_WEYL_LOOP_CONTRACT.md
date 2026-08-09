# Phase-QEMU V8 multimode trapped-ion Weyl-loop contract

## Scope and authority

M266 is a deterministic software digital twin of a physically structured
three-ion, three-mode state-dependent-force process. It descends from M265's
ideal single-oscillator Weyl loop, but introduces normal-mode geometry, a
segmented controller, spectator closure, and a declared open-system heating
law. It does not report a physical experiment, QEMU device execution, physical
ion custody, or physical same-mode restoration.

The exact package claim is:

```text
THREE_MODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_NULLSPACE_PULSES_CLOSE_ALL_NOMINAL_MODE_DISPLACEMENTS_AND_IMPLEMENT_TWO_DISTINCT_ZZ_PHASE_PROGRAMS_ON_ONE_LOGICAL_MULTIMODE_BACKING_WHILE_DECLARED_NONZERO_MARKOVIAN_HEATING_MONOTONICALLY_BREAKS_EXACT_INITIAL_MODE_STATE_RETURN_WITHOUT_RECOOLING
```

The exact claim ceiling is:

```text
DETERMINISTIC_COMPLEX128_FLOAT64_GAUSSIAN_MOMENT_SOFTWARE_DIGITAL_TWIN_WITH_DECLARED_LINEAR_HARMONIC_STATE_DEPENDENT_FORCE_AND_MARKOVIAN_ADDITIVE_HEATING_LAWS_NO_PHYSICAL_ION_CUSTODY_AND_DIRECT_COMPILED_CLIENT_CHANNEL_SHADOW
```

The restoration classification is:

```text
NO_RESTORATION_CLAIM
```

The exact restoration scope is:

```text
NOMINAL_ZERO_HEATING_LOGICAL_GAUSSIAN_MODE_RETURN_ONLY_WITH_NONZERO_HEATING_EXACT_SAME_MODE_RETURN_REJECTED_AND_FRESH_MODE_SWAP_OR_RECOOLING_CLASSIFIED_AS_EXTERNAL_RESET
```

The exact negative resource disposition is:

```text
DIRECT_COMPILED_ZZ_AND_DEPHASING_CHANNEL_SHADOW_OMITS_THE_THREE_MODE_CONTROLLER_LOOP_WHILE_HEATING_PREVENTS_CATALYTIC_RETURN_SO_NO_RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED
```

The selected successor is:

```text
CONDITIONAL_GAUSSIAN_CLOSED_LOOP_FORWARD_SHADOW_AND_IRREVERSIBLE_DIFFUSION_NO_RETURN
```

## Process partitions

The model must keep these roles distinct:

```text
carrier:
    three persistent collective-mode Gaussian moments

client:
    three qubits, initialized once in |+++>

source and controller:
    signed drive-minus-mode detunings
    eight real force amplitudes per program
    segment clock and amplitude quantizer

environment:
    three nonzero additive Markovian heating rates

detector and boundary:
    selected final client Pauli moments

restoration process:
    nominal simultaneous zero displacement only
    no snapshot, reload, recooling, or carrier replacement
```

The full JSON printed by production is a privileged scientific envelope. Only
the nested selected client boundary is a candidate guest-visible boundary.
Mode coordinates, covariance baselines, pulse compiler diagnostics, heating
exposures, and controls are verifier information.

## Frozen hardware geometry

The three declared equal-mass collective-mode eigenvector patterns are columns
of

```text
B = [
  [ 1/sqrt(3),  1/sqrt(2),  1/sqrt(6)],
  [ 1/sqrt(3),          0, -2/sqrt(6)],
  [ 1/sqrt(3), -1/sqrt(2),  1/sqrt(6)]
]
```

and `eta = 0.06 B`. The declared frequency and controller table is:

```text
mode frequencies Hz       [1900000, 1918000, 1930000]
drive frequency Hz         1911000
signed drive-mode delta    2 pi [11000, -7000, -19000] rad/s
program duration           120 us
segments                    8
segment duration            15 us
heating Gamma              [15, 30, 60] quanta/s
```

The near-degenerate frequency table is a synthetic collective-mode public
calibration descriptor; it is not asserted to be the axial spectrum of one
harmonic trap.
It could be mapped only after a platform-specific normal-mode calculation.
The explicit signed detuning array is authoritative for this twin. No
implementation may silently change its ordering or infer a different sign
convention.

## Exact nominal segment law

For target ions `i,j`, computational branch values `z_i,z_j in {-1,+1}`,
mode `m`, and real segment amplitude `Omega`, production uses

```text
H/hbar = sum_m c_s,m(t)
  [a_m exp(-i delta_m t) + a_m_dagger exp(+i delta_m t)]

c_s,m = (eta_i,m z_i + eta_j,m z_j) Omega / 2.
```

For one constant segment beginning at `t0` with duration `dt`, define

```text
I_m = integral_(t0)^(t0+dt) exp(+i delta_m t) dt
u_m = -i c_s,m I_m.
```

If `z_m` is the displacement accumulated before that segment, the exact
chronological update is

```text
phase += Im(u_m conjugate(z_m))
phase += c_s,m^2 [delta_m dt - sin(delta_m dt)] / delta_m^2
z_m   += u_m.
```

The second phase contribution is the exact intra-segment Magnus self-area. It
is not optional. Dropping it changes the required waveform normalization and
yields the wrong client phase even when endpoint closure appears successful.

For the four target-spin branches, production decomposes the final phase as

```text
phi(z_i,z_j) = phi_global + theta z_i z_j

theta = [phi(++ ) - phi(+-) - phi(-+) + phi(--)] / 4.
```

All three conditional displacements must close for every branch. In the
zero-heating ideal model this makes the final operation

```text
exp(i theta Z_i Z_j) tensor I_three_modes.
```

This ideal logical identity does not establish a physical return.

## Frozen programs

Program A acts on ions `(0,1)` and must implement `theta=-pi/8` with

```text
[875322.2363528529,
 -1017217.5496960702,
 1149417.5349432244,
 -128702.55695608277,
 -128702.55695606946,
 1149417.5349432132,
 -1017217.549696064,
 875322.2363528487]
```

Program B acts on ions `(1,2)` and must implement `theta=-pi/6` with

```text
[-229497.22240917346,
 725694.6296712102,
 -1293755.9826773852,
 1628861.1574510091,
 -1628861.1574510091,
 1293755.9826774026,
 -725694.6296712208,
 229497.22240918515]
```

These are the physically corrected amplitudes including self-area. Production
must obtain, to float64 tolerance:

```text
program A theta                         -0.39269908169872414
program B theta                         -0.5235987755982986
program A maximum final displacement    <= 1e-12
program B maximum final displacement    <= 1e-12
program A maximum midloop displacement   0.7125305458876734
program B maximum midloop displacement   0.8557657738719043
```

The nonzero midloop displacements are causal carrier-participation witnesses.
Endpoint closure alone may not be implemented as a direct client lookup.

## Standard Markovian additive-heating law

Production uses the standard dissipator convention

```text
D[L](rho) = L rho L_dagger
            - 1/2 {L_dagger L, rho}

mathcal_L_heat = sum_m Gamma_m (D[a_m] + D[a_m_dagger]).
```

Consequently

```text
d<n_m>/dt = Gamma_m
V_m(t) = V_m(0) + Gamma_m t I_2.
```

For two client branches `s,r`, let `Delta beta_m(t)` be their exact conditional
mode displacement separation. The reduced-client coherence multiplier is

```text
exp[-sum_m Gamma_m integral |Delta beta_m(t)|^2 dt].
```

There is no factor `1/2` in this influence exponent. Combining a full
covariance drift with a half-strength exponent mixes incompatible conventions
and is rejected.

The exact path-separation integral is evaluated analytically inside every
constant segment. The primary maximum exponents are approximately

```text
program A  0.008551211055284244
program B  0.004805936264387415
```

After both programs, the mode number increments are

```text
[0.0036, 0.0072, 0.0144].
```

They are strictly positive. Thus every named primary fixture rejects exact
return to its initial mode state. This is why the package classification is
`NO_RESTORATION_CLAIM`, despite exact nominal endpoint closure.

The Gaussian moment model propagates mode means, covariance drift, an inert
reference covariance where declared, and the exact reduced-client influence
channel. It does not retain the complete heated joint spin-mode-environment
state and must not claim complete heated process factorization. In schema
language, `full_joint_heated_gaussian_process_retained=false`: this is a
moment/influence model, not a full joint master-equation integration.

## One logical multimode backing and fixtures

Every fixture is supplied once. Its covariance allocation object and base
pointer remain unchanged while A increments generation `0 -> 1` and B consumes
the actual heated A output and increments `1 -> 2`. The accepted primary path
has zero snapshot, reload, recooling, detach, and carrier replacement events.
Object identity establishes logical resident custody only.

The five named Gaussian fixtures are:

1. `vacuum_product`;
2. `thermal_product_nbar_0p05_0p10_0p20`;
3. `coherent_product_alpha_0p30_plus_0p15i_minus_0p20_plus_0p10i_0p10_minus_0p25i`;
4. `squeezed_mode0_r0p35_phi0p20_with_thermal_spectators`;
5. `tmsv_bus0_inert_reference_r0p30`.

Vacuum covariance is `I_2/2`. A thermal mode uses
`(nbar+1/2)I_2`. The squeezed fixture rotates
`diag(exp(-2r),exp(+2r))/2` through angle `phi/2`. The last fixture retains a
two-mode-squeezed covariance between bus mode zero and an inert reference.
Heating affects only the bus block. Its reference fidelity must decrease, so a
bus-marginal-only result cannot be treated as reference-preserving return.

The primary client begins in `|+++>`. Program A has no guest boundary release.
Program B consumes its actual output. Only the final selected A-then-B client
moments are released.

## Strongest honest comparator

The strongest direct compiled comparator has equal access to every public
implementation law.
It applies two direct diagonal ZZ phases and, for the heated comparison,
multiplies client matrix elements by the two public `8 x 8` dephasing channel
descriptors. It retains no bus coordinates, executes no force segments, and
performs no restoration stage.

Production and the direct ideal shadow must agree in the zero-heating model.
Production and the direct heated-channel shadow must also agree. The heated
primary must differ detectably from the direct ideal client, with the frozen
run giving approximately

```text
combined heated-to-direct-ideal trace distance  0.004249949772572
combined heated client purity                    0.991962603457778
```

The direct channel shadow is not a physical trapped-ion replacement. It is the
strongest equal-access deterministic software comparator, and it defeats a
software resource claim. M257 remains intact.

The physical force is treated as an XX-style interaction mapped to the declared
effective ZZ client contract. The ledger charges four single-qubit basis
rotation pulses per program, eight total. The direct software shadow processes
the public exposure descriptors compiled from all 16 force segments, but it
executes zero controller segments, retains zero bus coordinates, and performs
no restoration.

## Mandatory controls

The package must execute and disclose:

1. `omitted_final_segment` for A and B;
2. `target_middle_mode_only_spectator_omission`;
3. `detuning_plus_250hz_all_modes` for A and B, implemented as
   `delta -> delta - 2 pi 250`;
4. `amplitude_quantization_12bit_fullscale_1p8e6` for A and B;
5. `segment_index3_plus_1_percent` for A and B, where the index is zero-based;
6. `opposite_phase_orientation` by complex-conjugating the nominal paths;
7. `zero_force_zero_area`;
8. `snapshot_reload`;
9. `recooling_external_reset`;
10. `fresh_mode_swap_carrier_replacement`;
11. `tmsv_reference_coherence`.

The regenerated closure receipts include:

| Control | A maximum closure | B maximum closure |
|---|---:|---:|
| omitted final segment | 0.434733850265698 | 0.113981122585134 |
| mode frequencies +250 Hz | 0.073242886359666 | 0.024556956173447 |
| segment 3 area +1% | 0.000639208691392 | 0.008089833127939 |
| 12-bit amplitude quantization | 0.000815882274292 | 0.000473727090978 |

The middle-mode-only waveform must close its target mode while leaving a
spectator displacement above `1e-3`. The opposite orientation must return the
nominal modes but flip each ZZ phase sign. Zero force must have both zero area
and zero displacement.

Snapshot reload is classified only as `SNAPSHOT_RELOAD`. Recooling is an
`EXTERNAL_RESET`. A fresh mode object is `CARRIER_REPLACEMENT`. None is
catalytic restoration or same-mode reuse.

## Resource and precision ledger

Production must report at least:

```text
physical modes and client qubits
mode frequencies, eigenvectors, and eta calibration cells
programs, segments, duration, and pulse descriptors
closure-matrix rank, nullity, and singular values
amplitude bit depth, full scale, and update rate
peak Omega and integral Omega^2 dt
sum of per-mode peak coherent-energy upper bounds
heating delta n and added mechanical energy
Gaussian verifier baselines and inert reference
retained schedule and dynamic-history counts
snapshot, reload, recooling, and replacement counts
direct-shadow channel descriptor size
```

`integral Omega^2 dt` is only a control-power proxy. Optical/RF joules and
wall-plug energy remain uninstantiated without a source transfer model. Zero
net coherent mode-energy change at nominal closure does not mean zero drive
energy, zero dissipated energy, or thermodynamic reversibility.

At finite Fock cutoff `N`, the projected commutator obeys

```text
[a_N,a_N_dagger] = I_N - N |N-1><N-1|.
```

Its norm defect is `N`. M266 therefore makes no arbitrary-state finite-Fock
claim. Any future Fock promotion requires named energy-constrained convergence
and a top-edge counterexample.

## Promotion gates

The internal production checks require:

```text
nominal closure, every program and mode       <= 1e-12
ZZ phase error                                <= 1e-12
nominal client/direct ideal trace distance    <= 1e-12
heated client/direct heated trace distance    <= 1e-12
heated client/direct ideal trace distance     >= 1e-3
every fixture covariance drift                 positive and monotone
allocation object and base pointer             unchanged across A then B
snapshot/reload/recooling/replacement           zero on the primary path
```

Passing these checks is a source self-consistency result only. The production
program must not grant itself independent strict verification.

## Exact interpretation

M266 establishes a physically structured software obstruction:

```text
nominal closed geometric loop
+ correct client phases
+ nonzero causal mode excursion
+ one logical multimode backing across A then B
+ explicit controller and environment costs
- nonzero heating changes every mode covariance
- exact initial-state return is false
- direct client channel shadow omits the loop
= no restoration claim and no M257 escape
```

The material difference from M265 is not a positive advantage. It is the
explicit collision between nominal geometric closure and irreversible open-
system diffusion. Further Gaussian closed-loop variants remain directly
shadowable and cannot repair the no-return result merely by adding Gaussian
squeezing or cosmetic holonomy.
