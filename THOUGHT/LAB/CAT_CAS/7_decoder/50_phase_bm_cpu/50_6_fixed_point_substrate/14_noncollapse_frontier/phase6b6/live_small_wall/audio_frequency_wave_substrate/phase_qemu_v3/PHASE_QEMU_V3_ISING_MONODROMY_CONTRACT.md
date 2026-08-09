# Phase-QEMU V3 exact Ising-holonomy diagnostic contract

## Scope

The bounded experiment is:

```text
CONTROLLED_MANY_BODY_EIGENPHASE_HOLONOMY_SCATTERING_PHASE_QEMU_BACKEND
```

The implemented claim is:

```text
IDEAL_EXACT_ISING_MTC_TRIANGULAR_ADJACENT_BRAID_PREPARATION_CONTROLLED_TRANSPORTED_MAJORANA_PAIR_HOLONOMY_FINAL_ONLY_BOUNDARY_FUNCTIONAL_RESTORATION_AND_REUSE_AT_N4_N8_N12_N16_WITH_CONTIGUOUS_CUT_RANKS2_4_8_AND_COMPACT_SIGNED_PAIRING_RESOURCE_KILL
```

The ceiling is:

```text
IDEAL_DETERMINISTIC_EXACT_SOFTWARE_ISING_MTC_CONTROLLED_HOLONOMY_DIAGNOSTIC_AT_N4_N8_N12_N16_ONLY
```

This is an exact software backend diagnostic. It does not execute a QEMU
device or CATVM service and does not model or observe physical Ising anyons,
braiding, interferometry, topological protection, QND detection, or physical
restoration. M257 continues to control the equal-access deterministic-software
comparison.

## Exact Ising representation and public braid family

The scaling family has `n=4r` sigma anyons, represented by `2r` pair-charge
fermion modes in the fixed-even-parity sector. Majoranas are one-based:

```text
gamma_(2a-1) = Z_<(a) X_a
gamma_(2a)   = Z_<(a) Y_a.
```

The public clockwise adjacent exchange is:

```text
C_j = (I - gamma_j gamma_(j+1)) / sqrt(2).
```

Starting from vacuum, the public triangular execution list is the
concatenation, for `t=1...2r-3`, of:

```text
[2t+2, 2t+1, ..., t+3].
```

The accepted scaling words are therefore:

```text
n=8:  [4]
n=12: [4,6,5,8,7,6]
n=16: [4,6,5,8,7,6,10,9,8,7,12,11,10,9,8].
```

All scientific decisions use exact arithmetic in:

```text
Z[i] / sqrt(2)^h  subset  Q(zeta_16).
```

The implementation applies every adjacent Majorana product to the resident
amplitude vector. It does not read a stored charge, expected phase, or answer
table to prepare the state or execute a holonomy.

## Contiguous-cut scaling result

The declared spatial cut is the actual contiguous cut after fermion mode `r`:

```text
left  = modes 1...r
right = modes r+1...2r.
```

Exact Gaussian-rational elimination on the actual amplitude flattening gives:

```text
n=8  -> rank 2
n=12 -> rank 4
n=16 -> rank 8.
```

The fixed-sector fusion dimension is `2^(2r-1)`, while this reference
implementation allocates the full `2^(2r)` occupation array. The support and
cut rank both grow as `2^(r-1)`. These are separate resource quantities. The
rank result is a property of this selected family and cut, not a complexity
lower bound.

## Transported pair-parity holonomy and final-only response

For each scaling fixture the two commuting accepted operators are:

```text
L0 = -i gamma_3 gamma_(2r+1), exact eigenvalue +1
L1 = -i gamma_4 gamma_(2r+2), exact eigenvalue -1.
```

Their precise semantic class is:

```text
BRAID_TRANSPORTED_MAJORANA_PAIR_PARITY_HOLONOMY.
```

Equivalently they are the signed images of local pair-parity stabilizers under
the public triangular braid `W`. For
`S_k=-i gamma_(2k-1) gamma_(2k)`, the exact formulas are:

```text
L0 = W S_2 W^dagger
L1 = -W S_3 W^dagger.
```

They must not be described as arbitrary literal
enclosed-subset Wilson loops unless a future descriptor supplies the required
transport path and framing.

The accepted transaction is:

```text
exact carrier value supplied
-> coherent probe-path Hadamard
-> controlled L_j on every resident amplitude
-> probe-path recombination
-> coherent copy to a private response qubit
-> reverse recombination
-> inverse derived from the public loop descriptor
-> exact carrier/path equality verification
-> only then response release.
```

The production path does not inspect the forward carrier and choose a result.
It evolves an explicit coherent path and response. The permitted public bit is
recognized only after the complete joint state equals the restored carrier
with path zero and one definite response bit.

The independent reference reconstructs the exact braid, holonomy, dense rank,
covariance, probe-factorization, and restoration algebra. It derives the
response-copy law from exact branch states rather than independently executing
the production transaction state machine. Accordingly, scientific algebra,
ranks, and boundaries are `SEPARATE_REFERENCE_PARITY`, while production
transaction ordering is `PACKAGE_SELF_REVIEW_SOURCE_AUDITED`.

Generation 1 uses `L0` and returns bit `0`. Generation 2 consumes the exact
returned carrier value without a second preparation, uses distinct `L1`, and
returns bit `1`. Both execute the same public-derived Hermitian inverse before
release.

## N=4 algebra and lifecycle smoke

The `n=4` fixture is not assigned the scaling cross-cut formulas. It begins in
the two-mode vacuum and tests:

```text
L = -i gamma_1 gamma_2 has eigenvalue +1;
C_2 C_2 maps vacuum to |11> up to exact phase;
the same L then has eigenvalue -1;
C_2^dagger C_2^dagger restores vacuum exactly.
```

Both eigenstates execute a retained-copy transaction and a result-free
latch/unlatch transaction. This is an algebra, orientation, inverse, and
lifecycle smoke test only.

## Restoration and reuse

The implementation uses immutable exact values and establishes:

```text
FUNCTIONAL_EXACT_VALUE_RESTORATION_AND_REUSE_WITHOUT_SAME_BACKING
```

with restoration classification:

```text
EXACT_ALGEBRAIC_RESTORATION.
```

It does not establish same-allocation or same-backing reuse. No snapshot,
saved carrier copy, migration, process reset, or baseline reload performs the
inverse. The verifier retains the input value only to test exact equality; it
does not restore from that value.

## Required controls

### Mixed-loop copy obstruction

For all scaling fixtures:

```text
M = -i gamma_3 gamma_4
```

has exact forward port weights `1/2,1/2`. It is not an eigen-holonomy. Retaining
a coherent response copy prevents a factorized restored response: each fixed
computational-basis response form has fidelity `1/4`, the restored-carrier
marginal weight is `1/2`, and the maximum factorized fidelity when an arbitrary
ancilla is allowed is `1/2`. No response is released. If no result is copied,
the public latch/unlatch sequence restores the exact carrier value. This
separates reversible interaction from irreversible retained-boundary use.

### Same-sector scramble

For all scaling fixtures:

```text
K   = -i gamma_2 gamma_(4r-1), with expectation zero
U_K = (I - gamma_2 gamma_(4r-1)) / sqrt(2).
```

`K` is disjoint from and commutes with `L0,L1`. `U_K` changes the carrier ray
with exact overlap squared `1/2`, while both loop outputs remain unchanged.
The source verifies the direct nonadjacent rotation against the adjacent-braid
synthesis:

```text
P = C_2 C_3 ... C_(4r-3)
U_K = P^dagger C_(4r-2) P
```

in execution-order convention, up to the permitted exact global sign. The
public adjoint restores the pre-scramble carrier exactly. Transactions on the
scrambled returned value also restore that value.

### Fault and coherence controls

The package additionally includes:

- an analytic exact path-dephasing control, which removes the exact
  off-diagonal path blocks and leaves port weights `1/2,1/2`, so no
  deterministic response survives; no density matrix is materialized;
- reverse orientation, which has the same exact output because each accepted
  Hermitian Z2 pair parity is self-adjoint;
- missing inverse on the `-1` loop and wrong-loop inverse, both of which fail
  restoration and lock the response;
- nonzero framing, rejected before mutation;
- exact rank measurement from amplitudes rather than descriptor inference;
- functional generation-two reuse without a second preparation.

There is one involutive holonomy layer per transaction, so a reordered-inverse
failure is not applicable. Missing and wrong-loop inverse attacks provide the
prospective failures.

## Strongest classical representation and route kill

The public triangular braid is a fermionic Gaussian/Ising-Clifford process.
The implemented formula-specific comparator transports an `O(n)` signed
Majorana pairing through the `O(n^2)` public braid list. It exactly predicts
`L0=+1`, `L1=-1`, and that `M` is not a definite pairing. A general Ising braid
state admits an `O(n^2)` Majorana covariance or stabilizer representation with
polynomial updates. Once a signed pair is known, its declared holonomy is an
`O(1)` lookup.

The dense exact occupation vector and ranks `2,4,8` are therefore reference
evidence, not the strongest implementation. They do not establish a distinct
phase resource. The accepted disposition is:

```text
TRIANGULAR_ISING_BRAIDS_AND_TRANSPORTED_PAIR_HOLONOMIES_REMAIN_COMPACTLY_TRACKABLE_BY_AN_O_N_SIZED_SIGNED_MAJORANA_PAIRING.
```

This kills this Ising family as a computational-resource candidate while
preserving it as a machine-law calibration. It is not a general no-go for
holonomy, non-Abelian anyons, phase computing, or physical restricted-access
carriers. A successor needs a factoring eigenphase that depends on a growing
interacting relational invariant not compactly tracked by Gaussian signed
pairing or stabilizer state.

M257 supplies an additional controlling shadow: an equal-access deterministic
software implementation can compute the same public forward boundary while
omitting the positive-cost catalytic inverse.

## Resource scope

The result counts the public preparation braid list, full allocated occupation
cells, the four-times-larger data-plus-path-plus-response transaction arrays,
the dense rank-matrix Gaussian cells, resident support, exact denominator
power and coefficient width, holonomy/inverse descriptors, and zero retained
dynamic inverse history.

It does not completely account for immutable intermediate allocations, Python
objects, hashing, serialization, allocator/RSS, whole-process live payload,
QEMU traffic, physical preparation, energy, spectral gap, noise, leakage,
precision, bandwidth, latency, calibration, detector visibility, or repeated
measurement cost. Resource evidence remains `PACKAGE_SELF_REVIEW`.

## Strict claim ceilings

The package establishes none of:

```text
QEMU device execution
CATVM or authenticated custody enforcement
same-backing restoration or reuse
physical Ising anyons or a coherent anyon interferometer
physical braid, detector, topological protection, or restoration
an arbitrary literal enclosed-subset Wilson loop without transport/framing data
a distinct phase-native computational resource
computational advantage or a complexity lower bound
escape from M257
Small Wall crossing
general holonomy or anyon computation
unbounded computation
replacement of physical bits with pi
```
