# Phase-QEMU V5 subgap boundary-resolvent findings

## Result

The bounded M263 result is:

```text
EXACT_RATIONAL_SINGLE_CHANNEL_SUBGAP_CAYLEY_RESOLVENT_DIAGNOSTIC_IMPLEMENTS_A_STIPULATED_FORMAL_STATIONARY_UNIT_MODULUS_BOUNDARY_LAW_AT_DECLARED_FIXTURES_WITH_DISTINCT_ENERGY_DESCRIPTOR_REUSE_GROWING_EXACT_KRYLOV_RANK_AND_PATH_ONLY_FIXED_MARGIN_EFFECTIVE_DEPTH_BOUND_PLUS_TILTED_FIELD_AND_BETHE_FACTORIZATION_CONTROLS
```

The claim ceiling is:

```text
EXACT_DETERMINISTIC_SOFTWARE_FINITE_DIMENSIONAL_RATIONAL_ONE_CHANNEL_K_MATRIX_BOUNDARY_MODEL_WITH_FORMAL_STATIONARY_ASYMPTOTIC_RETURN_ONLY
```

The exact fixture evidence records the declared target dimensions, public
rational Hamiltonians, subgap query points, Krylov ranks, Green-function
quotients, Cayley phases, formal stationary-return predicates, descriptor
reuse record, and
fixed-margin Neumann certificates. Those values are sealed in the production
and separate-reference evidence; they are not promoted here into a broader
scaling law.

The scientific algebra is suitable for `SEPARATE_REFERENCE_PARITY` when the
production and standalone reference seals agree. No transaction ordering,
response-release protocol, time-domain scattering, or target-return process
is executed. The restoration classification is therefore:

```text
NO_RESTORATION_CLAIM
```

The only reuse statement has the narrow formal scope:

```text
STIPULATED_FORMAL_STATIONARY_ONE_CHANNEL_BOUNDARY_AND_DISTINCT_ENERGY_DESCRIPTOR_REUSE_ONLY
```

This is not returned-value execution, same-allocation or same-backing reuse.
It is not a QEMU-device or CATVM-custody result and is not algebraic,
time-domain, or physical restoration.

## Exact one-port process law

The digital twin separates:

```text
probe source
reference arm
one-port scattering arm
resident target
virtual target sector
detector / boundary
environment
controller
```

The intended future device contract requires the source to prepare the probe
and be isolated before scattering; M263 does not execute that lifecycle. The
formal scattering arm couples through one declared boundary vector `v` to an
exact rational Hermitian target matrix `H`. For a rational query energy `E`
below the declared spectrum, the boundary Green function is

```text
G(E) = v^dagger (E I - H)^(-1) v.
```

Because `E` lies outside the exact spectrum and the ideal target has no loss,
`G(E)` is real. With positive rational port density/coupling `rho`, the public
Cayley law is

```text
S(E) = (1 - i rho G(E)) / (1 + i rho G(E)).
```

It follows algebraically that `S(E)^* S(E)=1`. In the stipulated stationary
one-channel model, the asymptotic target label is unchanged and the outgoing
monochromatic probe acquires one phase. Production evaluates this formula at
two energies against the same immutable public target descriptor. It does not
execute or verify a transient virtual excursion, source isolation, response
copy, target return, or second use of a resident returned target.

Any future native-return implementation must be materially different from
snapshot reload, VM rollback, process restart, baseline replacement, or a
saved carrier copy. The present formal law does not create a resource
advantage: the equal-access forward-only shadow computes `S(E)` directly, as
required by M257.

## Conditions required for the elastic interpretation

The phase is an accepted one-port elastic boundary only under all of these
conditions:

- the resident target is in the declared nondegenerate eigenstate or exact
  ideal target value;
- the probe energy is below every target-addition, Raman, sideband, and bath
  channel admitted by the model;
- there is one collected asymptotic probe channel, or all other elastic
  channels are explicitly included in the boundary rather than discarded;
- the target Hamiltonian and boundary coupling are Hermitian and lossless;
- the probe source is isolated before the query;
- the detector sees the recombined probe rather than source feedthrough or a
  controller-supplied phase; and
- target factorization is established before response release.

Above threshold, in a degenerate target manifold, at finite temperature, or
with unmodeled loss, the retarded self-energy generally acquires an imaginary
part or multiple outgoing channels. Then `|S|<1` for the selected port, the
probe can carry which-channel information, and the retained catalytic
boundary must lock. A physical implementation must test these conditions; a
software predicate is not a physical observation.

## Boundary Green function and the controlling comparator

Full target dimension is not the controlling representation. Starting from
the boundary vector, define the observable Krylov space

```text
K_r(H,v) = span{v, H v, H^2 v, ..., H^(r-1) v}.
```

Exact Lanczos reduction gives a finite tridiagonal or Hessenberg quotient
`T_r` such that

```text
G(E) = ||v||^2 e_1^T (E I - T_r)^(-1) e_1.
```

Equivalently, `G` is a rational continued fraction whose degree is the
boundary minimal-polynomial degree `r`. Target amplitudes outside this cyclic
subspace are invisible to the selected port. A growing many-body Hilbert
space therefore does not help if the boundary spectral measure retains a
bounded or otherwise compact quotient.

The strongest exact comparator stores the rational Krylov quotient, not a
dense inverse. Fixed margin supplies a polynomial-degree upper bound, not by
itself a smaller state representation. If the target spectrum lies in
`[a,b]`, `mu=(a+b)/2`, and `E<a`, then

```text
(H-EI)^(-1)
  = (mu-E)^(-1)
    sum_{k>=0} (-(H-mu I)/(mu-E))^k,

q = (b-a) / (2 (mu-E)) < 1.
```

At a fixed normalized subgap margin, `q` is bounded away from one, so the
required polynomial degree is logarithmic in inverse error. The current
recurrence vector may still occupy the full target-sector representation, and
generating each moment may still require growing state and work. Only the
declared path family has an executed compact streamed comparator, through its
continuant and local recurrence. The general fixed-margin bound obstructs
inferring approximation depth from growing exact Krylov rank; it does not
show that a generic interacting boundary can avoid dense state or work.

The disclosed comparator family also includes sparse fraction-free solves,
exact characteristic/minimal-polynomial methods, Lanczos continued
fractions, Chebyshev or kernel-polynomial spectral approximation, symmetry
reduction, MPS/DMRG correction-vector methods, and rational model reduction.
Comparator optimality is not claimed.

## Physical-platform interpretation

### P0, classical mechanical, and elastic phononic targets

For quartz, PZT, MEMS, and classical phononic networks, the corresponding
boundary law is the mechanical impedance

```text
G_cl(omega)
  = e_b^T (K - omega^2 M + i omega C)^(-1) e_b.
```

This is a sparse linear solve, transfer matrix, or continued fraction. P0's
source-separated quartz carrier, I/Q sensing, and ringdown remain valuable
process-geometry calibrations for a one-port device, but they do not turn the
classical susceptibility into a many-body computational resource. Strong
classical Duffing operation adds harmonics, history, attractors, and noise;
an equal-access software comparator still integrates the same finite
classical state, while dissipative ringdown is not exact catalytic return.

Atomistic Green-function methods explicitly reduce acoustic and elastic
waveguide, phononic-crystal, and defect scattering to effective Hamiltonians,
Green functions, and reflection/transmission calculations:

- H. Khodavirdi, Z.-Y. Ong, and A. Srivastava,
  *The Atomistic Green's Function method for acoustic and elastic
  wave-scattering problems*, <https://arxiv.org/abs/2301.12259>.

Classical and P0 hardware are therefore reference/calibration backends only.

### Gaussian quantum phononics

Harmonic quantum phonons, beam splitters, and squeezing are quantum physical
processes, but their Green functions and covariances remain polynomial-size
Gaussian representations. A multimode SAW device has demonstrated SQUID
coupling across more than twenty resonator modes, two-mode squeezing, and
four-mode entanglement, yet the reported state family is described by
covariance matrices:

- G. Andersson et al., *Squeezing and multimode entanglement of surface
  acoustic wave phonons*, <https://arxiv.org/abs/2007.05826>.

Gaussian phononics is retained as another calibration and negative-control
backend, not as the M257 escape.

### Interacting quantum-acoustic or circuit-QED target

The future physical target would replace `H` with the populated sector of a
connected non-Gaussian phonon or microwave Bose-Hubbard/Kerr network. The
boundary vector would add one probe quantum, and the phase would sample the
many-body addition resolvent rather than a one-particle normal-mode
susceptibility. Actual quantum dynamics, an exogenous resident target, and
Born-rule sampling would lie outside M257's deterministic-software domain;
the present digital twin does not.

Relevant physical primitives already exist separately:

- strong piezoelectric coupling of a superconducting qubit to multiple bulk
  acoustic modes and coherent qubit-phonon operations:
  Y. Chu et al., *Quantum acoustics with superconducting qubits*,
  <https://arxiv.org/abs/1703.00342>;
- strong-dispersive resolution of individual phonon-number states in a
  multimode acoustic cavity:
  L. R. Sletten et al., *Resolving Phonon Fock States in a Multimode Cavity
  with a Double-Slit Qubit*, <https://arxiv.org/abs/1902.06344>;
- acoustic Mach-Zehnder measurement of frequency-dependent one- and
  two-phonon scattering phase from a transmon:
  H. Qiao et al., *Acoustic phonon phase gates with number-resolving phonon
  detection*, <https://arxiv.org/abs/2503.03898>;
- Green-function scattering calculations for quantum waveguides and
  interacting Kerr/Bose-Hubbard targets:
  M. P. Schneider et al., *Green's Function Formalism for Waveguide QED
  Applications*, <https://arxiv.org/abs/1509.08633>, and
  T. F. See, C. Noh, and D. G. Angelakis, *Diagrammatic Approach to
  Multiphoton Scattering*, <https://arxiv.org/abs/1702.01632>.

These sources establish platform ingredients and mathematical precedent, not
the physical machine proposed here and not an advantage claim.

## Bandwidth, delay, precision, preparation, and energy

Subgap operation does not provide a free continuum of exact answers.

- The usable query band is the intersection of the one-port waveguide band,
  the target's elastic window, the coupler band, and the detector band.
- The Wigner-Smith delay
  `tau_W(E)=d arg(S(E))/dE` grows near a sharp pole or threshold. The same
  feature that increases phase sensitivity increases dwell time, exposure to
  loss, and calibration demand.
- A phase resolved to error `epsilon` needs an explicit shot law. Ordinary
  independent interferometric sampling costs order `epsilon^-2` trials;
  any improved law must count its nonclassical probe preparation.
- Exact rational coefficient height in the digital twin is not free physical
  precision. Frequency reference, detuning, pump phase, linewidth, drift, and
  detector calibration must be counted.
- Target preparation is a first-class cost. A command that simply supplies a
  ground-state vector, eigenphase, resolvent pole, or precomputed Krylov
  quotient is a negative control, not accepted preparation. A future target
  needs a public preparation/cooling/adiabatic law with its gap, time, energy,
  fidelity, and amortization recorded.
- Probe energy, target refrigeration, coupling pumps, detector energy,
  latency, and loss are unmodeled by the exact rational fixture and cannot be
  inferred from coefficient counts.

Near-threshold operation can invalidate the fixed-margin Neumann bound, but
it cannot erase these costs. It must be tested in the time domain with the
long dwell, transient target occupation, finite wavepacket bandwidth, and
loss channels present.

## Factorized-scattering negative control

An interacting label does not guarantee an irreducible many-body phase. In an
integrable or Bethe-factorized model, an `N`-body scattering eigenphase may be
a sum or product of compact one- and two-body phase shifts carried by a list
of rapidities. That is a polynomial descriptor and comparator, not the
growing relational invariant sought here. Likewise, a fixed finite emitter
that interacts sequentially with ordered time bins generates a bounded-bond
matrix-product process unless its physical memory or feedback depth grows.

Sequential cavity-QED state generation is known to have an efficient MPS
description:

- C. Schoen et al., *Sequential Generation of Matrix-Product States in
  Cavity QED*, <https://arxiv.org/abs/quant-ph/0612101>.

The next backend must therefore include nonintegrability controls and compare
the measured boundary Krylov rank against factorized S-matrix, finite-memory
MPS, transfer-matrix, and active-core reductions.

## Resource accounting

For every scaling fixture, durable evidence must distinguish at least:

```text
physical target modes
target-sector dimension
virtual-sector dimension
sparse Hamiltonian cells
boundary Krylov rank
minimal-polynomial / quotient degree
nonzero boundary spectral residues
exact coefficient and denominator height
Neumann or Chebyshev truncation degree at declared error
preparation descriptor, work, gap, and retained state
probe bandwidth and energy
Wigner-Smith delay
phase precision and shots
controller and detector state
loss, thermal occupation, and leakage
```

Component-local Python allocations or exact field cells are not a complete
live-process measure. Object overhead, allocator/RSS, hashing, serialization,
QEMU traffic, physical modes, fabrication, calibration, pumps, cooling,
energy, noise, bandwidth, and latency remain incomplete unless separately
instrumented.

## Disposition and successor

The bounded disposition is:

```text
GROWING_EXACT_KRYLOV_RANK_ALONE_IS_NOT_AN_APPROXIMATION_LOWER_BOUND_PATH_FIXED_MARGIN_HAS_COMPACT_STREAMED_SHADOW_AND_BETHE_FACTORIZED_EIGENPHASE_IS_PUBLIC_RAPIDITY_PRODUCT_NEAR_THRESHOLD_TIME_DOMAIN_QUALIFICATION_REQUIRED
```

The stipulated one-port law survives as a useful machine-model primitive:
source separation, coherent reference/scattering paths, virtual interaction,
factorized target return, final boundary, and reuse are all architecturally
worth preserving for the next executed backend. The path-family realization
is killed as a resource candidate because its exact continuant and streamed
fixed-margin shadow retain the accepted boundary without a dense carrier or
catalytic inverse. The interacting flagged blocks establish growing exact
Krylov rank only. They do not execute moment-generation comparators or prove a
general interacting resource kill.

The selected successor is:

```text
NEAR_THRESHOLD_NONINTEGRABLE_BOUNDARY_RESOLVENT_WITH_EXPLICIT_WIGNER_DELAY_FINITE_BANDWIDTH_PRECISION_PREPARATION_AMORTIZATION_AND_TENSOR_NETWORK_RESOURCE_CROSSOVER
```

Its minimum falsifier must use a populated connected non-Gaussian target,
public preparation, a finite-bandwidth wavepacket, explicit transient virtual
occupation, a one-port elastic-return check, distinct-energy reuse, and
measured growth of the boundary Krylov or correction-vector representation.
It must include harmonic/Gaussian, disconnected-core, Bethe-factorized,
finite-memory MPS, above-threshold leakage, thermal, source-on, and snapshot
controls. Promotion requires a growing boundary spectral resource after
preparation, precision, shots, delay, energy, bandwidth, loss, and controller
costs are charged.

No QEMU device, CATVM custody, actual acoustic or microwave target, physical
waveform, physical source isolation, detector, target return, restoration,
reuse, quantum advantage, M257 escape, Small Wall crossing, unbounded
computation, or replacement of physical bits with pi is established by M263.
