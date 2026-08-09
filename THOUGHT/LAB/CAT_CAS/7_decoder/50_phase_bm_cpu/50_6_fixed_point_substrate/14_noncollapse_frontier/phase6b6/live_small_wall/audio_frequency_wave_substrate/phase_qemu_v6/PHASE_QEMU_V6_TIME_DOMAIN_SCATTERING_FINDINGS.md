# Phase-QEMU V6 time-domain scattering findings

## Authority and result

M264 is a strict obstruction, not a promotion.

The frozen claim is:

```text
FROZEN_L641_SIGMA50_NEAR_THRESHOLD_TIME_DOMAIN_SCATTERING_EXECUTES_TRANSIENT_BORROW_DRAIN_AND_COMPLETE_RETURNED_DENSITY_HANDOFF_BUT_THE_PUBLIC_T120_ADIABATIC_PREPARATION_AND_FINITE_PACKET_T340_RETURN_FAIL_DECLARED_1E_MINUS_7_MATCHED_FREE_TARGET_TRACE_DISTANCE_GATES
```

The predeclared matched-free target-return gate was trace distance `<= 1e-7`.
The accepted public path fails it for both claim-bearing queries:

| query | accepted path | matched-free target trace distance | gate | result |
|---|---|---:|---:|---|
| A | public `Tprep=120`, finite packet, `T=340` | `8.038931410866084e-05` | `<= 1e-7` | fail |
| B | production returned-density rematerialized reuse | `2.8167715493073277e-05` | `<= 1e-7` | fail |

Neither threshold is loosened after observing the result. `SOURCE_SELF_CHECK_PASS`
means that the implementation and independent-reference invariants pass; it does
not mean that the declared return or reuse gates pass.

The exact ceiling is:

```text
FINITE_COMPLEX128_DETERMINISTIC_SOFTWARE_SINGLE_PROBE_L641_FOUR_SPIN_TIME_DOMAIN_MODEL_WITH_RETURNED_TARGET_DENSITY_REMATERIALIZATION_AND_NO_SAME_BACKING_OR_PHYSICAL_RESTORATION
```

The exact disposition is:

```text
STRICT_PREPARATION_AND_FINITE_PACKET_RESTORATION_OBSTRUCTION_RETAINS_REAL_TRANSIENT_INTERACTION_DRAIN_AND_APPROXIMATE_FUNCTIONAL_HANDOFF_BUT_REQUIRES_A_CHANGED_RETURN_PREPARATION_LAW_NOT_POST_HOC_FIXTURE_TUNING
```

The accepted-path scope is
`FAILED_DECLARED_TIME_DOMAIN_RETURN_OR_REUSE_THRESHOLDS`, and the restoration
classification remains `NO_RESTORATION_CLAIM`.

## What the model did establish

The claim-bearing production model is a deterministic complex128 simulation of
one propagating probe coupled locally to a four-spin interacting target. The
open lead has `L=641`, the packet starts at `x0=320` with `sigma=50`, the
contact is site `0`, and the detector bond is `(192, 193)`. Query A uses
`k=2*pi/5`; query B uses `k=9*pi/20`. The accepted endpoint is `T=340`, with a
predeclared `T=370` continuation used for captured-flux and delay controls.

Within this finite model, the following effects are real and retained:

- the target leaves its initial state during the collision by the declared
  transient-excitation margins;
- the interaction/contact population drains by the endpoint;
- near-contact residual population and packet spectral tails satisfy their
  declared gates;
- the detector sees the outgoing packet, and the near-threshold query has the
  larger stationary delay;
- an above-threshold control retains target excitation/nonreturn, so the
  subgap behavior is not a generic propagation artifact; and
- a complete target density matrix is extracted from query A and used to form
  query B in the production carryover route.

These are transient borrow/drain, finite-packet scattering, delay, and numerical
handoff results. They do not repair the failed target-return gates.

The `sigma=50` fixture is also frozen correctly: it was selected before semantic
execution to make the below-gap spectral-tail condition explicit. It must not be
described as a post-result repair of target return.

## Preparation and finite-packet obstruction

The public preparation is a paid `Tprep=120` adiabatic ramp from `|+>^4`, using
480 midpoint exponentials. That preparation is part of the accepted
computational path and cannot be silently replaced after failure.

The `Tprep=480` and exact-ground-injection arms are diagnostics and shams, not
claim-bearing substitutes. They show that the observed failure is sensitive to
preparation accuracy and the accepted finite-packet path: a much more accurate
ground preparation and an exact-ground query can reduce the return error, and
the exact-ground B diagnostic reaches a matched-free trace distance of about
`1.52e-10`. Those controls identify the obstruction; they do not authorize
changing the public fixture. A later, more expensive preparation law must be
declared and charged before it can support a new claim.

The accepted `T=340` window is likewise fixed. The `T=370` continuation is a
detector-tail and delay control, not a retroactive endpoint replacement. Near a
threshold, packet duration, long dwell time, finite bandwidth, tail leakage,
loss sensitivity, and return error are coupled resources rather than tunable
free parameters.

## Returned-density handoff is rematerialization, not restoration

After query A, production traces out the lead and obtains the complete `16 x 16`
returned target density matrix. It Hermitizes and trace-normalizes that matrix,
diagonalizes it, and propagates all 16 spectral components with query B by
constructing new blocks of the form `packet_B tensor b_r`.

That operation is full returned-density **numerical rematerialization and generic
target re-preparation**. The returned state is represented, decomposed, and
reinstantiated in new numerical storage. It is not continuous custody of a
resident target, not reuse of the same allocation or backing physical state,
not an echo or inverse, and not physical restoration. It does not reload a
saved baseline or inject the exact ground state, but the absence of those two
shams does not turn rematerialization into restoration.

The separate reference independently reconstructs the returned density and
rematerializes the follow-on query with controlled spectral discarded trace
`<= 1e-12`. Production, rather than the reference route, is the authority that
propagates every one of the 16 returned-density components. Reference parity
therefore supports the handoff and obstruction within its tolerance; it must
not be rewritten as same-backing reuse or as an assertion that both engines use
an identical full-component implementation.

The clean-B arm is a distinct fresh-preparation comparator. Snapshot/reload is
only a sham baseline. None of these routes supplies catalytic restoration.

## Local-projector fail-closed invariant

The contact projector is an explicit CSR matrix with exactly one nonzero entry,
at `(0, 0)`. The production model asserts both its support and `nnz == 1`.

This matters because constructing a nominal one-entry diagonal with a scalar
through a generic sparse diagonal helper can broadcast the scalar across the
entire diagonal. That changes a local boundary collision into an everywhere-on
interaction and can manufacture an apparently strong effect. The one-entry
projector assertion is therefore a semantic gate, not an implementation detail.

## Independent reference and numerical integrity

Production uses sparse Krylov action through `expm_multiply`. The separate
reference changes the numerical mechanism: it uses a fourth-order Yoshida
split operator, an orthonormal DST-I lead representation, dense target/contact
exponentials, and a separate continuous-preparation control. Its step-halving
and lead-size controls bound discretization and finite-lead substitutions.

The accepted numerical gates cover norm and density trace, Hermiticity,
positive-semidefinite tolerance, one-shot versus chunked endpoints, flux,
spectral tails, local drain, and production/reference parity. Passing those
gates supports the reported obstruction. It cannot be promoted into physical
validation, restoration, or an advantage claim.

The `T=340` detector delay is a finite-window estimate with residual tail. The
stationary multichannel computation and the `T=370` continuation are the
appropriate ordering and capture controls; no exact infinite-time Wigner-delay
claim is made for the finite packet.

## Strongest honest classical comparison

The production experiment is already an ordinary classical sparse recurrence:
the joint state has `641 * 16 = 10256` complex amplitudes, and the Hamiltonian
has 71,760 stored nonzeros. An equal-access forward shadow can execute the same
sparse time evolution and final boundary projection while omitting return
tests and returned-density reconstruction.

The honest comparator family includes:

- direct sparse Krylov, Chebyshev, or Lanczos time propagation;
- boundary-resolvent, correction-vector, and Krylov/Lanczos quotient methods;
- direct dense or spectral treatment of the fixed 16-channel target;
- real-time MPS and correction-vector MPS; and
- direct execution of the same accepted recurrence and final projection.

The one-particle lead admits an MPS cut bond dimension no greater than 17 for
this fixed target. Thus this fixture does not establish a tensor-network
crossover, growing-state separation, or an asymptotic resource advantage.
Comparator families that were named but not executed as full sweeps remain
future controls, not favorable zero-cost results.

M257 remains fully operative. The virtual dynamics, controller, boundary
projection, and returned-density reconstruction are available to an equal-
access deterministic-software comparator. SciPy or a future QEMU wrapper is
not a physical resource, lawful access restriction, oracle, or exogenous
carrier. This result does not escape the forward-shadow obstruction.

## Resource and error ledger

The reported evidence charges or exposes the following resources:

- **Preparation:** the accepted `Tprep=120` ramp is paid once per prepared
  target; reuse claims must state the amortization
  `C_total_per_query(Q) = C_prep/Q + C_query` rather than hiding preparation.
- **Finite packet:** `sigma=50`, propagation to `T=340`, continuation to
  `T=370`, spectral tails, dwell time, and finite-lead substitution are part of
  the cost and error model.
- **Numerics:** complex128 precision, sparse-Krylov error, split-operator step
  error, endpoint consistency, finite lead, and density-matrix
  rematerialization are explicitly finite approximations.
- **State:** the pure joint vector is about 160.25 KiB. A complete 171-sample
  history would be about 26.8 MiB and is not retained. A full 16-component
  returned-density propagation block is about 2.50 MiB before library and
  solver overhead.
- **Boundary precision and shots:** an interferometric phase estimate with
  visibility `V`, target phase error `delta_phi`, and failure probability
  `alpha` requires at least the declared classical sampling estimate
  `ceil(4 ln(4/alpha)/(V^2 delta_phi^2))` samples per quadrature under unity
  efficiency. Detector inefficiency increases this count. No coherent phase-
  estimation credit is claimed.
- **Noise and loss:** packet attenuation, target decoherence, source leakage,
  detector inefficiency, drift, and near-threshold dwell-time sensitivity are
  not free. The current deterministic run does not qualify robustness to them.
- **Energy and bandwidth:** packet generation, adiabatic preparation, control
  switching, detector bandwidth, long observation time, and any resident
  stabilization must be charged by a physical backend. They are not measured
  by this software experiment.
- **Controller and solver memory:** Python/NumPy/SciPy objects, allocator and
  BLAS/cache overhead, internal Krylov bases, RSS, descriptor copies, and
  controller/detector state are not fully instrumented by the compact array
  counts above.

No arbitrary-precision, physical energy, physical shot, wall-plug, or external
carrier cost has been measured. The compact memory numbers are lower bounds on
the software execution, not a complete machine accounting.

## Controls and untested promotion requirements

The package includes a disconnected `g=0` arm, a static scalar-boundary arm,
clean-B and returned-density B routes, exact-ground and slower-preparation
diagnostics, an above-threshold arm, snapshot/reload sham semantics,
one-shot/chunked endpoint checks, finite-window/continued detector checks, a
stationary 16-channel self-check, and a constructive forward-only shadow.

Matched Gaussian/harmonic calibration, loss and dephasing sweeps, preparation
error sweeps, small-core reductions, Bethe/factorized comparisons, and broader
MPS sweeps have not been completed here. Their absence must not be reported as
a passed physical or asymptotic control.

## Required mechanism change

The successor is frozen as:

```text
RESIDENT_OPEN_DRAIN_OR_ECHO_RETURN_WITH_PAID_GROUND_STATE_SUPPLY_PREDECLARED_FINITE_PACKET_ERROR_AND_SAME_BACKING_TARGET_CUSTODY
```

The next experiment must change the resident return/preparation law. A valid
candidate must keep the target in continuous same-backing custody, predeclare
and pay any ground-state supply, specify finite-packet error before execution,
and implement an open-drain, echo, or other physical return process whose
restoration is tested on the same resident state.

Merely increasing `Tprep`, widening `sigma`, extending the endpoint, loosening
`1e-7`, selecting a favorable spectral window after inspection, or
rematerializing a density matrix again would tune the failed fixture without
changing the blocked mechanism. Those actions cannot promote M264.

## Claim boundary

M264 establishes a bounded software fact: the near-threshold interacting model
executes real transient collision, drainage, delay, and a complete numerical
returned-density handoff, but its accepted preparation and finite packet fail
the predeclared target-return and reuse thresholds.

It does **not** establish same-backing restoration, physical restoration,
permanent or catalytic reuse, a physical observation, source-separated carrier
custody, a QEMU device qualification, an oracle, a restricted-access resource,
an asymptotic advantage, an M257 escape, an unbounded-compute result, a Small
Wall repair, or replacement of the bit with pi.
