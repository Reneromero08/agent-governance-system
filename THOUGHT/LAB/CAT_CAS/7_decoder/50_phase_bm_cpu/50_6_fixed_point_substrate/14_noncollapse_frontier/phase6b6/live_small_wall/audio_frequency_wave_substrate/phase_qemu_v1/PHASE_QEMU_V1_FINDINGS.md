# Phase-QEMU V1 exact non-Gaussian echo findings

## Result

`PHASE_QEMU_V1_BOUNDED_EXACT_FOUR_MODE_TWO_BOSON_CROSS_KERR_QND_PARITY_POINTER_PUBLIC_ADJOINT_RESTORATION_AND_GENERATION2_REUSE`

An actual QEMU 10.2.4 PCI device executed an exact ideal four-mode,
two-boson backend.  The device retained one homogeneous degree-two carrier in
two computational pointer branches, applied public noncommuting exchange
matchings and a causal cross-Kerr sign, copied only a factorized final parity
bit, unlatched the pointer, derived the reverse public adjoints, and verified
the exact initial carrier before making the response readable.

The primary descriptor

```text
A B K01 A B
```

reached odd mode-3 parity and restored generation 1.  The unrelated reuse
descriptor

```text
B A K03 B A
```

reached odd mode-1 parity and restored generation 2 on the same resident C
coefficient array in the same QEMU process, without preparation, reset,
snapshot, migration, or baseline reload.  A valid even selector and a held-out
descriptor containing a public adjoint also restored and returned the
independently reconstructed boundaries.

Verification classification:

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level:

`SEPARATE_REFERENCE_PARITY`

Restoration classification:

`EXACT_ALGEBRAIC_RESTORATION`

## Pointer and inverse controls

The independent exact oracle reconstructs the QND law

```text
psi_even |0> + psi_odd |1>
```

instead of inferring it from a scalar answer.  The accepted primary, even
selector, reuse, and held-out cases have exactly one populated pointer branch;
the copied bit survives while unlatch plus public adjoints restore the carrier
with fidelity 1.  Replacing the cross-Kerr gate with identity leaves both
pointer branches populated with reduced purity `5/8`.  Copying that mixed
pointer prevents factorized carrier restoration, so the device instead
unlatches, restores result-free, and keeps the boundary locked.

Missing inverse, wrong Kerr edge, and reordered inverse controls leave the
boundary locked, restoration generation unchanged, and the carrier spent.
Wrong nominal owner/program/generation, malformed descriptors, post-seal
mutation, null carrier, premature reads, and snapshot commands reject without
accepted scientific mutation.  The qtest command trace contains no reset and
generation-two reuse issues no second preparation command.  Guest stdout/stderr contains no hidden
amplitudes.  VMState does serialize hidden backend state for migration shams;
that trusted stream is not a guest/controller no-smuggle boundary.  V1's
incoming-lineage rejection is source-audited package-local because this package
does not execute a V1 save/load; V0's positive migration sham is separate.

## Resource result and obstruction

The device allocates 20 signed coefficient cells for the two pointer branches
and a 20-cell gate scratch.  The primary uses five forward gates, two pointer
actions, five inverse gates, and 12 abstract virtual cycles; generation-two
reuse doubles those cumulative counts.  The largest post-canonical resident
integer coefficient uses two signed bits.  The package also declares one
denominator-power scalar, one private boundary-bit cell retained during the
inverse, and 16 allocated public-descriptor words.  These are component-local
receipts, not a whole-process live-payload result: denominator height,
transient arithmetic, canonicalization, factorization, norm/restoration
verification, MMIO traffic, QEMU/Python objects, allocator/RSS, controller
state, physical energy, noise, precision, calibration, bandwidth, and latency
remain uninstrumented and nonzero where applicable.

The strongest comparator for the two pinned fixtures is an O(1) analytic
certificate after descriptor validation.  The strongest transferable exact
comparator is the identical ten-state fixed-number recurrence without the
catalytic inverse.  Kerr-disabled Fock-state boundaries also admit a
single-particle recurrence plus at most 2x2 permanents; Gaussian covariance is
not sufficient for that non-Gaussian Fock input.  Adaptive MPS/TTN or
boundary-only tensor contraction is the controlling growing-family baseline.

V1 therefore advances the virtual machine law but does not escape M257.  Its
factorized QND response is lawful only on parity-eigenstate boundaries, and
the accepted bounded fixtures have smaller exact classical descriptions.  No
physical boson, phonon, nonlinear coupler, detector, restoration, authenticated
custody, phase resource, advantage, Small Wall crossing, unbounded compute, or
physical-bit replacement has been established.

## Growing-family continuation

The only potentially relevant family grows modes and bosons together:

```text
M=n, N=n/2, D_B(n)=binomial(3n/2-1,n/2)
```

Fixed `N=2` or fixed `M=4` is polynomial and is not an asymptotic escape.  The
next experiment must compile a connected non-Gaussian public word family,
measure exact or certified approximate Schmidt/bond growth and parity-pointer
factorization, and compare the accepted device law with exact sparse-sector,
adaptive tensor-network, symmetry, integrability, precision, control-energy,
and boundary-only shadows.  If the public family remains monomial,
bounded-bond, or analytically certifiable, this quantum-acoustic echo route is
retired rather than extended with more four-mode fixtures.

## Durable evidence

- `evidence/PHASE_QEMU_V1_QTEST_RESULT.json`
- `evidence/PHASE_QEMU_V1_SEPARATE_REFERENCE.json`
- `evidence/PHASE_QEMU_V1_BUILD_RECEIPT.json`

The compiled QEMU binary and upstream source/build tree remain in managed
disk-backed Scratch and are not committed as scientific payload.
