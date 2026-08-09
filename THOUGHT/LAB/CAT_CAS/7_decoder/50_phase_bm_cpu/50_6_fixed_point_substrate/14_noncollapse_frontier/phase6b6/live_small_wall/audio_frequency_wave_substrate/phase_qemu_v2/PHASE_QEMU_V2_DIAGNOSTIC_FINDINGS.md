# Phase-QEMU V2 growing QND/bond diagnostic findings

## Result

```text
EXACT_GROWING_EVEN_MODE_FIXED_NUMBER_ALTERNATING_MATCHING_PI_CROSS_KERR_DIAGNOSTIC_HAS_SECTOR_DIMENSIONS10_56_330_AND_PEAK_CENTRAL_SCHMIDT_RANKS2_8_18_BUT_PRIMARY_QND_PARITY_FACTORIZATION_FAILS_AT_N6_N8_EXHAUSTIVE_N6_ALL_EDGESET_PROPER_SELECTOR_SEARCH_FINDS_ZERO_KERR_DISTINGUISHING_DETERMINISTIC_BOUNDARIES_AND_THE_DECLARED_DETERMINISTIC_N8_FIXED_CORE_ECHO_CONTROL_IS_DISCONNECTED_WITH_FUNCTIONAL_EXACT_PUBLIC_ADJOINT_RESTORATION_AND_NO_ADVANTAGE
```

The bounded exact diagnostic evaluates the formula-generated family

```text
n = 4, 6, 8
N = n/2
|Omega_n> = |1010...10>
F_n = A_n B_n K_(0,1) A_n B_n
```

in the fixed-number bosonic sector. `A_n` and `B_n` are alternating directed
50:50 exchange matchings and `K_(0,1)` is one exact pi cross-Kerr sign. The
public adjoint of every declared word returns the exact normalized initial
state. This is a functional exact-arithmetic diagnostic: immutable returned
values are reused directly, but no QEMU allocation, same backing, CATVM
custody, or response-ordering boundary is established.

Verification classification:

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level:

`SEPARATE_REFERENCE_PARITY`

Restoration classification:

`EXACT_ALGEBRAIC_RESTORATION`

Restoration scope:

`FUNCTIONAL_NORMALIZED_EXACT_STATE_EQUALITY_AND_RETURNED_VALUE_REUSE_WITHOUT_SAME_BACKING`

Claim ceiling:

```text
EXACT_SOFTWARE_FIXED_NUMBER_BOSONIC_ALTERNATING_MATCHING_SINGLE_PI_CROSS_KERR_QND_PARITY_DIAGNOSTIC_AT_N4_N6_N8_ONLY
```

## Growing-family measurement

Both independent implementations produce:

| modes, bosons | sector cells | final support | peak central Schmidt rank | last-mode even/odd | central-half even/odd |
|---|---:|---:|---:|---|---|
| 4, 2 | 10 | 1 | 2 | 0, 1 | 0, 1 |
| 6, 3 | 56 | 54 | 8 | 95/128, 33/128 | 125/256, 131/256 |
| 8, 4 | 330 | 292 | 18 | 767/1024, 257/1024 | 273/512, 239/512 |

The public geometry remains connected and exact Schmidt demand grows, but the
selected parity pointer is factorized only at four modes. At six and eight
modes both pointer branches are nonzero. Copying either mixed parity result
would leave the retained boundary entangled with the carrier, so result
retention and exact carrier restoration cannot coexist for those primary
fixtures. The exact public adjoint still supplies result-free restoration.

## Exhaustive six-mode search

The production implementation uses an exact 15-bit Walsh-Hadamard transform;
the independent implementation uses a separately written dense integer
coefficient recurrence and independent exact/modular rank checks. Each checks
all 32,767 nonempty subsets of the complete six-mode Kerr graph against all 62
nonempty proper subset-parity selectors.

For `A B K_E A B`, the search finds zero deterministic proper-subset parity
selectors. For `A B K_E B^dagger A^dagger`, it finds 180,162 raw deterministic
proper-subset closures across 16,383 Kerr graphs, but every closure preserves
the Kerr-disabled parity: Kerr-caused parity flips are exactly zero. The
all-six-mode selector is excluded from the scientific hit count because fixed
`N=3` makes it a conserved odd-parity sham for every graph.

The strict measured disposition is:

```text
DETERMINISTIC_QND_PARITY_CLOSURE_AND_CONNECTED_BOND_GROWTH_DO_NOT_COEXIST_IN_THE_TESTED_PI_CROSS_KERR_ALTERNATING_MATCHING_FAMILY
```

This is bounded to the two declared six-mode word forms and one pi cross-Kerr
layer. It is not a general cross-Kerr, bosonic, or complexity no-go.

## Fixed-core control

The declared control `A_n K_{(1,2),(3,4)} A_n^dagger`, with selector `{0,4}`,
is deterministic and Kerr-causal. At six modes its graph is connected, support
is four, and peak/final central rank is four. At eight modes its graph splits
as `[6,2]`, while support stays four and rank falls to two. Removing either
Kerr edge changes the exact state and changes the boundary to `1/2,1/2`.
Reapplying the public word restores the exact initial state. Thus the apparent
eight-mode continuation is a disconnected bounded-core certificate, not a
growing relational mechanism.

## Resource and comparator ceiling

Primary resident coefficient cells are 10, 56, and 330. Each exhaustive word
search generates one verifier-only `56 x 32768 = 1,835,008` integer
coefficient table, with 27,525,120 integer add/subtract operations; the two
tables are produced sequentially. Their measured signed integer widths are six
and eight bits. These tables are verifier artifacts, not accepted carriers or
compilers.

Matching-transition caches, compiler entries, Python objects, allocator/RSS,
hashing, serialization, and whole-process live payload are incomplete or
uninstrumented. Physical energy, noise, precision, bandwidth, calibration,
and latency are not modeled. Resource evidence is `PACKAGE_SELF_REVIEW`.

The honest comparison set includes exact sparse fixed-number evolution,
adaptive U(1)-symmetric MPS, TTN or boundary-only parity contraction,
fixed-number linear-optical recurrences for the Kerr-disabled word, symmetry
and finite-order reductions, and O(1) active-component certificates. The
adaptive comparators are disclosed but not implemented here; optimality is
not established and no resource-advantage comparison is authorized.

## Route disposition

The tested alternating-matching single-pi-Kerr family is killed as the next
Phase-QEMU mechanism: meaningful bond growth coincides with mixed parity,
while deterministic continuations collapse to conserved or bounded-component
certificates. More echo micro-variants are not authorized by this result.

The next materially different backend is:

```text
CONTROLLED_MANY_BODY_EIGENPHASE_HOLONOMY_SCATTERING_PHASE_QEMU_BACKEND
```

Its smallest calibration uses a factorized probe phase supplied by a genuine
many-body eigenstate law rather than forcing generic growing-rank occupation
states into a deterministic parity. Preparation, eigenstate verification,
probe coupling, echo, noise, energy, bandwidth, and reuse remain material.
The first candidate is an ideal Ising-anyon Wilson loop, with the compact
Majorana/topological-charge recurrence as the controlling expected no-go
comparator.

No physical carrier, cross-Kerr interaction, anyon, probe, QND measurement,
restoration, authenticated custody, distinct phase resource, computational
advantage, Small Wall crossing, unbounded computation, or replacement of
physical bits with pi is established.

## Durable evidence

- `evidence/PHASE_QEMU_V2_GROWING_QND_BOND_DIAGNOSTIC.json`
- `evidence/PHASE_QEMU_V2_GROWING_QND_BOND_SEPARATE_REFERENCE.json`
- `tests/qualify_growing_even_mode_qnd_bond_diagnostic.py`

