# Phase-QEMU V4 finite-MTC closed-probe contract

## Purpose

M262 tests one sharply delimited candidate left open by M261: whether a
single simple topological probe, closed around one disk-like target region,
can expose growing internal fusion multiplicity while returning exactly.

The accepted descriptor class is only:

```text
fixed finite UMTC
+ one disk with definite simple total charge a
+ one canonical vacuum-created simple probe b
+ one boundary-parallel closed encirclement
+ projection to the original vacuum probe channel
```

It is not the full annular or tube algebra and not a general anyon computer.

## Exact law

Ribbon balancing makes full monodromy scalar on every multiplicity copy of an
`a x b -> c` fusion channel:

```text
lambda_c = theta_c / (theta_a theta_b).
```

The normalized vacuum-return amplitude is therefore

```text
M_ab
  = sum_c [N_ab^c d_c / (d_a d_b)] lambda_c
  = S_ab S_00 / (S_0a S_0b),
```

up to conjugation when the orientation convention is reversed.  The weights
are positive and sum to one.  Consequently `|M_ab| <= 1`, with equality if
and only if all supported channel phases align.

The corresponding boundary Wilson operator is

```text
W_b = sum_a (S_ab / S_0a) Pi_a.
```

For a fixed finite category, all simple boundary-parallel loops commute and
span only the finite total-charge projector algebra.  On one definite-charge
block they have observable rank one, independently of the internal fusion
multiplicity.

## Restoration semantics

`|M_ab|^2` is a vacuum-return probability.  It is not, by itself, catalytic
restoration evidence.

The package separately executes bounded exact functional transactions only
for deterministic scalar phases `+1` and `-1`:

```text
prepare one exact carrier value
-> form reference and loop branches
-> recombine to a deterministic port
-> retain one boundary bit
-> undo the recombination
-> apply the public conjugate phase
-> compare exact carrier commitments
-> reuse the returned value with a distinct probe
```

For Fibonacci `tau,tau`, both the vacuum-return and complementary exact
probabilities are nonzero.  Coherently retaining that which-outcome record has
Schmidt rank two.  An inverse acting only on carrier and probe cannot erase
the retained orthogonal response, so no response is released with exact
factorized restoration.  This statement excludes the probability-zero and
probability-one endpoints.  A result-free full unitary adjoint may still
restore.

Restoration classification:

```text
EXACT_ALGEBRAIC_RESTORATION
```

Exact scope:

```text
FUNCTIONAL_EXACT_PLUS_MINUS_ONE_SCALAR_LOOP_RESTORATION_AND_DISTINCT_PROBE_REUSE_WITHOUT_SAME_BACKING
```

No resident allocation identity, QEMU/CATVM custody, or physical restoration
is claimed.

## Fixtures

- Semion: `M_ss=-1`.
- Ising: `M_sigma,psi=-1`, `M_psi,psi=+1`, and
  `M_sigma,sigma=0` for one loop.  Two `sigma,sigma` loops realign the channel
  phases and return with common phase `-i`.
- Fibonacci: `M_tau,tau=-phi^-2=(sqrt(5)-3)/2`, with return probability
  `phi^-4=(7-3sqrt(5))/2`.  Five loops realign the root-of-unity phases and
  return with phase `+1`.
- Ising total-`psi` multiplicity dimensions `1,2,4,8,16` execute `-1` then
  `+1` transactions with one preparation and exact functional reuse.

Repeated-loop controls prevent the false conclusion that a nonunit
single-loop return amplitude can never realign at a later finite power.

## Verification scope

```text
fixture arithmetic, dimensions, character ranks, transactions:
    SEPARATE_REFERENCE_PARITY

general fixed-UMTC centrality/equality theorem:
    FORMAL_DERIVATION_SOURCE_AUDITED

resource accounting:
    PACKAGE_SELF_REVIEW
```

The separate reference imports no production code and reconstructs Semion,
Ising, and Fibonacci values through rational quadratic fields and an
independent `Q(zeta_5)` quotient.

## Strongest honest comparator

For a fixed category with `K` simple objects, the controlling representation
is the exact finite monodromy-character table:

```text
definite a,b query:                    O(1) after validation
single-probe charge distribution:      O(K) exact scalars
charge-sector coherences if required:  O(K^2) exact scalars
sequential charge recurrence:          O(n K^2) arithmetic, O(K) cells
```

Coefficient-height and preparation/input costs remain payable.  This quotient
represents exactly the observable algebra of whole-region central loops; it
does not represent an arbitrary internal many-anyon state.  A growing probe
link network may have hard topology and is not assigned constant work.

## Rejected descriptor classes

The scope classifier must reject:

- noncentral constituent weaves or paths entering the disk;
- tube coupons or matrix units;
- multiple independently encircled regions;
- growing probe link networks;
- adaptive or forced measurement, postselection, and classical correction
  history;
- prepared noncentral eigenstates used as a substitute for uniform
  reference-complete factorization;
- growing MTC families.

The theorem also excludes defects, boundaries, higher genus, non-topological
dynamics, and external physical query advantages.

## Claim and ceiling

Claim:

```text
EXACT_FIXED_FINITE_UMTC_SINGLE_GLOBAL_CLOSED_SIMPLE_PROBE_DIAGNOSTIC_ESTABLISHES_MULTIPLICITY_BLIND_TOTAL_CHARGE_SCALAR_ACTION_DETERMINISTIC_UNIT_MODULUS_BOUNDARIES_AS_CONSTANT_SIZE_SIMPLE_OBJECT_LOOKUPS_AND_STRICTLY_INTERMEDIATE_VACUUM_RETURN_RETAINED_BOUNDARY_OBSTRUCTION_WITH_FUNCTIONAL_EXACT_PLUS_MINUS_ONE_SCALAR_LOOP_RESTORATION_DISTINCT_PROBE_REUSE_AND_SEMION_ISING_FIBONACCI_FIXTURES
```

Claim ceiling:

```text
ABSTRACT_EXACT_FIXED_FINITE_UMTC_SINGLE_SIMPLE_PROBE_GLOBAL_DISK_ENCIRCLEMENT_WITH_DECLARED_TOTAL_CHARGE_AND_SEMION_ISING_FIBONACCI_FIXTURES_ONLY
```

This establishes no physical anyons or interferometer, Phase-QEMU device,
authenticated custody, distinct phase resource, computational advantage,
M257 escape, Small Wall crossing, unbounded compute, or bit replacement.

## Primary references

- Bonderson, Shtengel, and Slingerland, *Interferometry of Non-Abelian
  Anyons*: https://arxiv.org/abs/0707.4206
- Kitaev, *Anyons in an exactly solved model and beyond*:
  https://arxiv.org/abs/cond-mat/0506438
- Bonderson, Freedman, and Nayak, *Measurement-Only Topological Quantum
  Computation via Anyonic Interferometry*:
  https://arxiv.org/abs/0808.1933
- Hardiman and King, *The tube category and representations of tube
  algebras*: https://arxiv.org/abs/1806.01800
- Aharonov and Arad, *The BQP-hardness of approximating the Jones
  polynomial*: https://arxiv.org/abs/quant-ph/0605181
