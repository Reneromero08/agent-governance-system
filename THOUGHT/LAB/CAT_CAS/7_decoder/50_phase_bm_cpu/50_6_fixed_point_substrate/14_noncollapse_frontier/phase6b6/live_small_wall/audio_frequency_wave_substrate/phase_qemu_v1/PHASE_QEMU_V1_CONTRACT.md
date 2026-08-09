# Phase-QEMU V1 exact non-Gaussian echo contract

## Scope

Phase-QEMU V1 is an exact ideal-unitary backend calibration for a four-mode,
two-boson quantum-acoustic machine model. It is implemented as QEMU virtual
hardware, but the QEMU process and its arithmetic are ordinary deterministic
software. The model therefore remains inside the M257 forward-shadow domain.
It is not evidence of physical bosons, phonons, QND measurement, restoration,
or computational advantage.

The bounded claim under test is:

```text
PHASE_QEMU_V1_BOUNDED_EXACT_FOUR_MODE_TWO_BOSON_CROSS_KERR_QND_PARITY_POINTER_PUBLIC_ADJOINT_RESTORATION_AND_GENERATION2_REUSE
```

The claim ceiling is:

```text
DETERMINISTIC_EXACT_IDEAL_QEMU_4_MODE_2_BOSON_BACKEND_ONLY
```

## Process geometry

The model keeps these domains distinct:

```text
preparation source
four carrier modes in the fixed two-boson sector
two-state QND pointer
public exchange/cross-Kerr couplers
nominal guest command tags
private pending boundary
released boundary
environment and virtual work counters
```

The physical interpretation is a prospective localized quantum-acoustic mode
network. The implemented state is a ten-coefficient homogeneous degree-two
polynomial over exact dyadic rationals. It is emulator bookkeeping, not a
claim that a physical device stores ten digital coefficients.

The pointer is represented explicitly by two ten-coefficient branches. The
backend may not replace pointer dynamics with an amplitude inspection and a
copied answer bit.

## Native gate algebra

For directed exchange `R_ij`:

```text
a_i^dagger -> (a_i^dagger - a_j^dagger) / sqrt(2)
a_j^dagger -> (a_i^dagger + a_j^dagger) / sqrt(2)
```

The disjoint matchings are:

```text
A = R_01 R_23
B = R_03 R_12
```

They do not commute. Because every matching covers all four modes and the
carrier has total degree two, its homogeneous-polynomial action has a common
factor `1/2` and closes in exact dyadic rational coefficients.

The nonlinear phases are:

```text
K_01 = (-1)^(n_0 n_1)
K_03 = (-1)^(n_0 n_3)
```

These cross-Kerr signs are self-adjoint. A hard-core restriction makes them
trivial and is therefore a required route-kill control rather than an accepted
substitute.

From the public source state `|1010>`, the primary word is:

```text
A -> B -> K_01 -> A -> B
```

and the descriptor-distinct reuse word is:

```text
B -> A -> K_03 -> B -> A
```

The backend receives only gate opcodes, the declared parity mode, and nominal
owner/program/generation tags. The descriptor contains no expected boundary,
inverse word, amplitude, lookup table, or answer-bearing fixture.
The tags enforce command consistency inside one emulated device; they are not
authenticated identities and do not isolate the device from a malicious guest
that can write every MMIO register.

## Atomic catalytic law

The first transaction is:

```text
lease generation 1
-> prepare |1010>
-> isolate the preparation source
-> seal the public descriptor
-> execute the exact forward word on the actual resident carrier/pointer
-> couple the declared carrier parity to the actual pointer branches
-> require exact pointer factorization
-> retain only the resulting private parity bit
-> uncompute the pointer coupling
-> derive and execute reversed public adjoints on the same coefficient backing
-> verify exact |1010>, clear pointer, idle source/couplers, and canonical metadata
-> increment restoration generation
-> only then expose the boundary
```

Generation-two reuse clears only the already released response and old public
descriptor. It does not reprepare, reload, migrate, copy, or replace the
carrier. The unrelated reuse descriptor then consumes the actual restored
backing.

Restoration classification for the accepted in-place path is:

```text
EXACT_ALGEBRAIC_RESTORATION
```

QEMU reset, process recreation, migration, and snapshot are never accepted as
restoration. A migration stream would remain `SNAPSHOT_RELOAD` and must mark
snapshot lineage so it cannot enter accepted restored reuse.

## Pointer law and rejection boundary

For a selected mode parity decomposition and a pointer initialized in `|0>`:

```text
psi = psi_even + psi_odd
Q_parity(psi |0>) = psi_even |0> + psi_odd |1>
```

The pointer factorizes exactly only when one parity branch is zero. Accepted
programs must reach such a parity eigenstate through the public dynamics. A
generic superposed boundary entangles the pointer. The backend must then
unlatch, execute the public adjoint, verify result-free restoration, keep the
boundary locked, and return `POINTER_ENTANGLED`.

The production compiler may validate descriptor topology and gate types. It
may not precompute the final parity or accept a public promise obtained by
simulating the answer.

## Controls

The focused gate includes:

- Kerr replaced by identity: pointer entangles and no result is released;
- missing, wrong-edge, and prospectively reordered inverse: restoration fails,
  the boundary remains locked, and the carrier becomes spent;
- wrong owner, program, generation, mode, or post-seal mutation: reject before
  scientific mutation;
- premature boundary reads: return the locked sentinel;
- null carrier and snapshot command: reject;
- valid odd and even parity selectors: release parity bits whose pointer-Z
  values are `-1` and `+1`
  only after restoration;
- descriptor-distinct generation-two reuse on the same QEMU device/backing;
- independent exact operator oracle and strongest matched classical recurrences.

Fault-injection controls are host-configured QOM properties compiled into this
test device binary. They are not guest MMIO capabilities, the accepted backend
runs with fault mode zero, and fault-enabled executions never qualify an
accepted path.

## Resource and scaling law

For `M` modes and `N` indistinguishable bosons, the exact fixed-number sector
has:

```text
D_B(M,N) = binomial(M+N-1,N)
```

V1 materializes 20 coefficient cells for the carrier-pointer branches and 20
scratch cells. These are allocated-backing counts, not a whole-process peak-live
claim. The package separately declares one denominator-power scalar, one
private retained boundary bit during inverse, 16 allocated public-descriptor
words, and the five used primary descriptor words. The device directly counts
all forward and inverse gates plus pointer latch/unlatch. Its coefficient-width
receipt is the largest signed integer width observed in the post-canonical
resident arrays; denominator height, transient live payload, canonicalization,
factorization, norm/restoration verification, MMIO traffic, retained host-side
response objects, Python/QEMU object headers, allocator state, host RSS,
native-library work, physical control energy, cooling, noise, calibration, and
bandwidth are not instrumented. Excluded resources are not zero. The VMState
stream contains hidden backend state for a migration sham; V1 source marks an
incoming stream as sham lineage, but V1 does not execute a migration test. This
source-local guard is not a controller-visible no-smuggle result and does not
inherit V0's positive migration evidence.

The bounded V1 fixture is classically tiny: `D_B(4,2)=10`. Its controlling
comparators are:

1. an O(1) analytic certificate after validating either frozen fixture;
2. the identical exact ten-state sparse recurrence without catalytic inverse;
3. for the Kerr-disabled word, a four-mode single-particle recurrence plus
   at most `2x2` permanents, not Gaussian covariance for the non-Gaussian Fock
   input;
4. adaptive MPS/TTN and boundary-only tensor contraction with measured ranks.

The only serious scaling diagnostic grows both resources:

```text
M=n, N=n/2, D_B=binomial(3n/2-1,n/2)
```

Fixed `N=2` or fixed `M=4` grows only polynomially and cannot support the
long-term claim. Even the extensive family remains a diagnostic until its
actual public words generate growing exact or certified approximate tensor
rank and survive all symmetry, integrability, precision, control, energy,
latency, and decoherence comparisons.

## Strict negative ceiling

V1 does not establish:

```text
physical execution or physical QND extraction
authenticated or malicious-guest custody
a resource unavailable to equal-access classical software
computational advantage or Small Wall crossing
general bosonic or relational closure
bounded-width growth or unbounded computation
replacement of physical bits with pi
```

The mechanism advances the machine law only: a causal non-Gaussian phase can
change a deterministic typed boundary, a modeled QND pointer can factorize on
that boundary, and public-adjoint execution can restore and reuse the actual
emulated carrier without history. The next frontier must test whether any
growing physical family preserves that law while escaping the strongest
classical and control-cost shadows.
