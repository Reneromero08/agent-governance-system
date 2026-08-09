# Phase-QEMU V6 near-threshold time-domain scattering contract

## Outcome and authority

M264 is a strict obstruction, not a restoration promotion.  The frozen model
executes a source-off packet collision, resolves nonzero transient target
excitation, drains the interaction site, and reconstructs the complete
returned target density as the input to a distinct second query.  The public
T120 adiabatic preparation and finite packet at T340 nevertheless miss the
predeclared `1e-7` interaction-picture target-return gates.  Those gates were
not loosened after observing the result, and neither the exact-ground nor the
T480 preparation diagnostic replaces the accepted path.

The strict claim is:

```text
FROZEN_L641_SIGMA50_NEAR_THRESHOLD_TIME_DOMAIN_SCATTERING_EXECUTES_TRANSIENT_BORROW_DRAIN_AND_COMPLETE_RETURNED_DENSITY_HANDOFF_BUT_THE_PUBLIC_T120_ADIABATIC_PREPARATION_AND_FINITE_PACKET_T340_RETURN_FAIL_DECLARED_1E_MINUS_7_MATCHED_FREE_TARGET_TRACE_DISTANCE_GATES
```

The claim ceiling is:

```text
FINITE_COMPLEX128_DETERMINISTIC_SOFTWARE_SINGLE_PROBE_L641_FOUR_SPIN_TIME_DOMAIN_MODEL_WITH_RETURNED_TARGET_DENSITY_REMATERIALIZATION_AND_NO_SAME_BACKING_OR_PHYSICAL_RESTORATION
```

The route disposition is:

```text
STRICT_PREPARATION_AND_FINITE_PACKET_RESTORATION_OBSTRUCTION_RETAINS_REAL_TRANSIENT_INTERACTION_DRAIN_AND_APPROXIMATE_FUNCTIONAL_HANDOFF_BUT_REQUIRES_A_CHANGED_RETURN_PREPARATION_LAW_NOT_POST_HOC_FIXTURE_TUNING
```

The active successor is:

```text
RESIDENT_OPEN_DRAIN_OR_ECHO_RETURN_WITH_PAID_GROUND_STATE_SUPPLY_PREDECLARED_FINITE_PACKET_ERROR_AND_SAME_BACKING_TARGET_CUSTODY
```

Restoration classification is `NO_RESTORATION_CLAIM` for the frozen accepted
path.  Approximate numerical return and functional density handoff are real
subclaims, but they do not satisfy the declared restoration threshold.

## Backend boundary

This package is a deterministic complex128 software hardware model using
SciPy sparse Krylov `expm_multiply`.  It is not QEMU device execution, a
physical scattering observation, a physical source-isolation measurement, or
an external carrier experiment.

The model keeps separate:

- the 16-dimensional target carrier;
- the 641-site single-probe lead;
- the source-off initial packet descriptor;
- the localized target/lead interaction;
- the matched `g=0` reference arm;
- detector-current and boundary-mode observables;
- the returned target density and its reconstruction controller;
- sham reload and exact-ground controls.

No durable JSON contains state amplitudes, target eigenvectors, answer tables,
or precomputed scattering responses.  Optional streams contain only aggregate
currents and probabilities.

The access model is fully public and equal-access.  Target coefficients,
preparation schedule, lead geometry, packets, coupling, observation grid, and
thresholds are available to production and comparator alike.  No Green
function, phase, delay, target eigenvector, or answer is supplied to the
propagator.  Query B receives the complete reconstructed `rho_A`; the
comparator may execute the identical sparse recurrence.  There is no oracle,
secret-state, or restricted-access claim.

## Frozen target and preparation

The target is exactly the public four-spin Hamiltonian

```text
H_T = -sum_j X_j
      -3/4 sum_(j=0..2) Z_j Z_(j+1)
      -1/5 Z_0 -2/7 Z_1 -3/11 Z_2 -5/13 Z_3
      -1/2 Z_0 Z_1 Z_2.
```

The software diagonalizes `H_T` only for diagnostics, the exact-ground sham,
the target gap, and matched-free return metrics.  Production queries use the
public adiabatic state generated from `|+>^4` by

```text
H(s) = (1-s)(-sum X) + s H_T
s(u) = 3u^2 - 2u^3
T_p = 120
480 midpoint piecewise-constant exponentials.
```

The 240-step same-duration preparation is a discretization check.  A
`T_p=480`, 1,920-step preparation is executed only as a nonclaim-bearing
diagnostic after the T120 path has been frozen; it cannot repair or replace the
accepted result.  Exact-ground injection is likewise a sham diagnostic and is
never production input.

## Frozen lead and packets

The one-port lead is the open discrete Laplacian

```text
H_L = 2 I - sum_(x=0..639) (|x><x+1| + |x+1><x|).
```

The frozen geometry is:

```text
L                  641
x0                 320
sigma              50
contact site       0
detector bond      (192,193)
primary stop       340
delay control stop 370
observation grid   0,2,...,340,342,...,370
```

The source-off packet is

```text
a_x = C exp(-(x-x0)^2/(4 sigma^2)) exp(-i k (x-x0)).
```

The query momenta are `k_A=2 pi/5` and `k_B=9 pi/20`.  The above-threshold
negative control uses `k=pi/2`.  The `sigma=50` value is a predeclared change
from the original `sigma=48` blueprint: before semantic execution, the open
lead discrete-sine audit found query B's above-gap weight at sigma48 was
approximately `1.23e-9`, just above the `1e-9` gate.  Sigma50 reduces it to
approximately `3.25e-10`.  The packets are effectively, never exactly,
subgap.

## Local interaction and fail-closed projector law

The joint Hamiltonian is

```text
H = H_L tensor I_16
  + I_L tensor (H_T-E_g I_16)
  + |0><0| tensor (3/2 Z_0).
```

The contact projector is constructed as an explicit CSR matrix with one
stored coordinate.  Production asserts before propagation:

```text
P0.nnz == 1
P0.nonzero() == {(0,0)}
support(kron(P0,Z0)) on the lead == {0}.
```

Using `scipy.sparse.diags(([1.],), [0], shape=(L,L))` is forbidden because it
can broadcast the scalar across the full diagonal and turn a local collision
into an everywhere-on interaction.  The claim-bearing source fails closed on
this exact error.

For the frozen model the joint dimension is `641*16=10,256` and the assembled
Hamiltonian has 71,760 stored nonzeros.

## Executed process

### Query A

1. Prepare the T120 adiabatic target.
2. Prepare packet A away from the target.
3. Turn the preparation source off.
4. Propagate the coupled joint state through T340.
5. Continue a copy through T370 only for the detector-flux/delay control.
6. Stream contact population, near-target drain, detector current, target
   excitation, edge-buffer probability, norm, and final density metrics.
7. Trace out the departed lead at T340 to obtain `rho_A`.

### Returned-density handoff into query B

Production hermitizes and trace-normalizes `rho_A`, diagonalizes the complete
16-by-16 density, and propagates all 16 spectral components.  No positive
eigenvalue weight is discarded:

```text
rho_A = sum_r |b_r><b_r|
Psi_(B,r)(0) = packet_B tensor b_r.
```

This is complete functional returned-state reconstruction/rematerialization.
It is generic target state re-preparation from the returned density.  It is
not resident same-backing reuse, physical reuse, or proof that the original
target allocation survived.  The production handoff uses neither the saved
baseline preparation nor the exact ground state, but the lack of those two
specific reloads must not be misread as lack of rematerialization.

The clean-B comparator executes a distinct adiabatic preparation.  The
snapshot/reload sham records what replacing `rho_A` with a cached preparation
would mean, but that sham is not the production reuse path.

## Return metrics

For input target density `rho`, the accepted reference is its matched free
interaction-picture evolution

```text
rho_free(T) = exp(-i(H_T-E_g)T) rho exp(+i(H_T-E_g)T).
```

Production records:

- target ground population;
- density trace, Hermiticity, minimum eigenvalue, and purity;
- Uhlmann density fidelity to `rho_free(T)`;
- exact 16-by-16 trace distance to `rho_free(T)`;
- trace distance to the unevolved input separately;
- final contact and near-target populations;
- maximum transient target excitation.

High ground population alone is not restoration.  The controlling strict gate
is

```text
D(rho_out, rho_free(T)) <= 1e-7
```

for both query A and the returned-density query B.  The first frozen execution
found approximately `8.04e-5` for A and `2.82e-5` for B reuse, so M264 is an
obstruction even though final ground infidelity is only a few parts in
`1e-9` and the contact has drained below `1e-11`.

The nonclaim-bearing T480 preparation reduces ground infidelity from about
`4.34e-9` to `1.40e-11`.  Exact-ground B reaches matched-free trace distance
about `1.52e-10`.  These diagnostics localize the frozen failure to the paid
preparation/combined accepted path rather than authorizing an after-the-fact
replacement of its input.

## Boundary I/Q, mode fidelity, and delay

The matched `g=0` arm begins with the same packet and target density.  For pure
queries, production records ground-channel I/Q.  For mixed-state query B it
records only density-linear cross-arm coherence and probabilities; individual
spectral-branch phases are not reported because density eigenvectors have
arbitrary global phases.

The detector-bond current is

```text
J_d(t) = 2 Im sum_a,r conj(Psi[d,a,r]) Psi[d+1,a,r].
```

Incoming and outgoing centroid times are computed from the negative and
positive current windows.  The relative current-centroid delay subtracts the
matched `g=0` round trip.  T340 is retained as the frozen primary window but
is not treated as a tight Wigner estimator because a small outgoing tail has
not crossed the detector.  The T370 continuation is a declared flux control.

An independent-formula, same-package 16-channel stationary boundary solve
computes the relative reflection phase and its centered finite-difference
Wigner delay.  It is a source self-check, not separate-reference parity.  The
frozen first execution gives approximately:

```text
                    query A       query B
T340 centroid       1.29447       7.55048
stationary Wigner   1.33520       7.55929
```

The near-threshold delay growth is real in this finite model.  It does not
repair failed target restoration.

## Numerical gates and solver controls

The predeclared gates include:

```text
norm/trace error                         <= 1e-9
density Hermiticity error                <= 1e-12
density minimum eigenvalue               >= -1e-12
above-gap packet weight                  <= 1e-9
A maximum transient excitation           >= 1e-2
B maximum transient excitation           >= 5e-2
final contact probability                <= 1e-8
matched-free target trace distance       <= 1e-7
incoming and outgoing detector flux      >= 0.99
one-shot/chunked endpoint L2              <= 1e-8
stationary delay ordering B-A             >= 5.5
discarded positive rho_A spectral weight == 0
```

SciPy `expm_multiply` acts on sparse CSR Hamiltonians through a counting
`LinearOperator`.  The observation grid is not a time-integrator step.  The
primary execution streams in ten-interval chunks, retains no complete
171-state history, and compares query A's endpoint against an independent
one-shot T340 call.  A first run found phase-aligned endpoint L2 near
`7.4e-12`.

`SOURCE_SELF_CHECK_PASS` means these execution-integrity, locality,
convergence, and accounting checks passed.  It does not mean restoration
promotion passed.  The JSON separately records
`restoration_promotion_all=false` and the two failed return gates; a valid
obstruction is a successful source execution, not a process error.

The unhalved L641 geometry is retained.  No second lead size is promoted as a
finite-size proof; edge-buffer probabilities and the T370 drain continuation
are the implemented finite-geometry guards.  A finite open lead necessarily
recurs at sufficiently late time, so permanent restoration is not claimed.

## Controls

Implemented controls are:

- matched `g=0` disconnected arm;
- the scalar static contact potential
  `(3/2)<g|Z0|g> |0><0|`;
- distinct clean-B adiabatic preparation;
- exact-ground-injection sham for B;
- T480 adiabatic preparation diagnostic, not production input;
- above-threshold `k=pi/2`, which retains large real target excitation;
- snapshot/reload counterfactual, explicitly classified as a sham;
- one-shot versus chunked Krylov endpoint;
- T340 versus T370 detector-flux/delay windows;
- 16-channel stationary phase/delay self-check;
- constructive forward-only shadow.

The small-core control is explicitly unimplemented because no nonarbitrary
truncation was frozen.  A Gaussian/quadratic surrogate is explicitly
unimplemented because no comparator matching this target's gap and boundary
spectral measure was predeclared.  These absences are not zeros.

## Strongest classical baselines and M257

The production recurrence is already an ordinary sparse classical program on
10,256 complex amplitudes.  Its strongest honest forward shadow executes the
identical `O(16L)` sparse coordinate evolution and final projection while
omitting return tests and returned-density handoff.  A one-particle lead MPS
has lead-cut bond at most `16+1=17`.  Therefore this fixed target establishes
no tensor-network crossover, asymptotic resource separation, or computational
advantage.

M257 remains fully intact.  Phase-QEMU modeling does not hide implementation
access from the comparator and does not turn QEMU or SciPy into a physical
resource.

## Resource and liveness law

One pure joint state has 10,256 complex128 amplitudes, or 160.25 KiB raw.  A
retained 171-state history would require about 26.8 MiB raw and is not kept.
The complete 16-component returned-density block requires 2.50 MiB raw.
Production records vector-equivalent forward and adjoint matvecs, preparation
work, density factorization, maximum chunk bytes, and cold-start/two-query
amortized runtime.  Wall time is diagnostic and never participates in a claim
decision.

The following remain explicitly uninstrumented:

- Python, NumPy, SciPy, allocator, and object overhead;
- SciPy's internal Krylov basis payload and exact peak RSS;
- BLAS threads and cache traffic;
- arbitrary-precision or exact payload heights beyond complex128;
- whole-process controller/detector state and descriptor copies;
- physical source, control, bandwidth, loss, latency, and energy;
- physical measurement precision and shot count.

No input, intermediate, output, history, rematerialization, controller, or
detector cost may be inferred to be zero because it is not instrumented.

## What M264 does not establish

M264 does not establish:

- QEMU device or external hardware execution;
- physical source isolation or scattering;
- same-backing target custody, restoration, or reuse;
- permanent restoration on a finite lead;
- a physical energy, bandwidth, precision, or shot law;
- a growing-target or asymptotic resource law;
- a tensor-network crossover;
- a distinct phase resource or computational advantage;
- escape from M257;
- a Small Wall crossing, unbounded compute, or physical bit replacement.
