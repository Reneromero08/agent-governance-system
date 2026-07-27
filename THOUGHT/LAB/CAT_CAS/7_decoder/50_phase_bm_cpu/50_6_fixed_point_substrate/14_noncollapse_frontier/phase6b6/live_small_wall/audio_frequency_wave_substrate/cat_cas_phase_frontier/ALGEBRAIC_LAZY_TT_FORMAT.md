# Bounded Width-Three Lazy Tensor-Train Relation Composition

## Status

This mutable checkpoint establishes:

```text
BOUNDED_WIDTH3_RANK2_PHASE_RESIDENT_LAZY_TT_RELATION_COMPOSITION_WITH_SPARSE_BOUNDARY_INVARIANT
```

for two tested Boolean/F3 relation pairs. It replaces the dense width-three
input representation with genuinely coupled rank-two tensor-train cores and
contracts eight predeclared output coefficients from an exact lazy relational
expression.

It does not materialize or claim a compact full output relation.

## Public source

```text
CATCAS_TT_WIDE3_PAIR 1
WIDTH 3
RANK F 1 2 2 1
CORE F 0 <8 F3 values>
CORE F 1 <16 F3 values>
CORE F 2 <8 F3 values>
RANK G 1 2 2 1
CORE G 0 <8 F3 values>
CORE G 1 <16 F3 values>
CORE G 2 <8 F3 values>
SELECT 0 P0 P1 P2
...
SELECT 7 P0 P1 P2
END
```

Each physical index is:

```text
P_i = left_i + 2 * right_i
```

and every core value is in `{0,1,2}`. The fixed ranks are:

```text
1 -> 2 -> 2 -> 1
```

so one six-variable relation occupies:

```text
1*4*2 + 2*4*2 + 2*4*1 = 32 phase cells
```

instead of 64 dense coefficient cells. A nonzero public coefficient-matrix
minor is required for each fixture relation. This rejects the rank-one
independent-site case.

The eight selectors are public topology. They are sparse coefficient requests,
not answers, witnesses, candidate sets, assignments, or a truth table.

## Native lazy relation law

For resident relations `F(X,Y)` and `G(Y,Z)`, the descriptor is:

```text
SQUARE(F)
SQUARE(G)
ADD(LIFT_XY(SQUARE(F)), LIFT_YZ(SQUARE(G)))
NORM_Y0(...)
NORM_Y1(...)
NORM_Y2(...)
```

with:

```text
N_t(P) = P|t=0 * P|t=1
```

All coefficient products use the branch-native `symbol_product`. Coefficient
addition is phase multiplication, and Boolean polynomial multiplication uses
OR-index convolution. A TT coefficient is contracted directly from the
actual resident cores over the two virtual bonds.

The evaluator memoizes only coefficient requests reached by one sparse
selector, clears that cache before the next selector, and never allocates a
64-coefficient output tensor. Its maximum measured live cache population is
449 phase values; the current fixed sparse-cache capacity is 1,024 entries.
The fixed buffers and compiler-measured nested stack frames are reported
separately and combined into a conservative ceiling. They are not presented
as a space advantage.

The width-one degeneration agrees with the reviewed native `OP_COMPOSE`
implementation to `1.11022302463e-16`.

## Resident transaction and inverse

```text
F rank-two cores       cells  0..31
G rank-two cores       cells 32..63
sparse H messages      cells 64..71
public boundary        cells 72..79
```

The accepted path:

1. encodes public F and G cores into one carrier;
2. seals a topology-only lazy descriptor;
3. evaluates eight selected H coefficients directly from the actual F/G core
   cells into resident message cells;
4. copies the actual resident messages to the boundary;
5. decodes only the boundary;
6. removes that boundary copy from the actual messages;
7. recomputes the same lazy factors while F/G remain resident and applies
   their actual inverse to the message cells;
8. inverses G and F encoding;
9. verifies exact canonical carrier restoration; and
10. runs an unrelated rank-two fixture on that same restored carrier; and
11. completes 32 more alternating transactions on the same allocation,
    gating the maximum accumulated restoration error across 34 generations.

The ordinary accepted path holds a comparison snapshot only for equality
measurement and never loads it. A separate labeled snapshot-reload control
establishes the weaker transactional baseline.

## Demonstrated boundaries

```text
primary [0,0,1,2,2,2,1,2]
reuse   [1,1,2,2,1,0,2,0]
```

The maximum correct in-place restoration error is:

```text
1.57009245868e-16
```

against a predeclared `2e-12` tolerance. The decoded result remains in
ordinary result state after the carrier has been restored.

## Controls

Qualification requires:

```text
wrong sparse-message inverse
omitted sparse-message inverse
reordered G-before-message inverse
rank-one virtual-bond cut
mismatched G virtual-bond permutation
attempted intermediate projection rejection
null-carrier rejection
eager output-TT rejection at the declared rank cap
malformed coefficient and rank rejection
separate snapshot-backed restoration
actual restored-carrier cross-program reuse
34-generation alternating reuse drift sentinel
width-one OP_COMPOSE parity
exact output-key allowlist
preprojection output-sink rejection
dense-relation and assignment-table static gate
fresh-process byte determinism
strict GCC and static analysis
ASan, UBSan, and leak detection
source, fixture, and result hashes
focused independent review
```

Wrong, omitted, and reordered inverses each leave restoration error
`1.73205080757`. The rank-one bond cut changes the primary boundary to:

```text
[1,0,0,0,0,0,0,0]
```

and a mismatched G bond changes it to:

```text
[0,0,2,1,2,1,0,2]
```

Both prospective algebra controls reverse their actual altered histories
cleanly.

## Rank and resource disclosure

Exact eager structural ranks for this construction are:

```text
input rank                         2
OR square rank                     4
lifted intersection rank           8
after NORM_Y0                     64
after NORM_Y1                  4,096
after NORM_Y2             16,777,216
```

The engine therefore fails closed on an eager full-output request. It performs
no value-dependent pivoting, rank inspection, or silent compression.

For one accepted forward-and-inverse transaction:

```text
carrier cells                                      80
live carrier bytes                              2,560
resident sparse message cells                       8
public decoded cells                                 8
maximum live sparse cache entries                  449
TT coefficient contractions                    69,376
symbol products                                668,276
coefficient additions                         470,968
OR-decomposition pairs                        113,268
carrier reads                                 832,528
phase-cell updates                                160
current-ABI material-buffer subtotal          38,688 bytes
compiler-measured nested stack call chain     37,088 bytes
conservative combined accounting ceiling      75,776 bytes
```

The material-buffer subtotal counts carrier baseline and working state, the
comparison snapshot, fixed cache capacity and metadata, selector factors,
descriptor, and two parsed public programs. The GCC `-O2 -fstack-usage`
measurement adds the active `main -> execute -> apply -> evaluate -> four
recursive lazy levels -> square -> TT coefficient` frame chain. The combined
ceiling deliberately double-counts stack-resident objects already present in
the material subtotal, so it is conservative rather than an RSS measurement.
It does not imply computational advantage.

## Claim boundary

The accepted result establishes compact phase-resident rank-two inputs,
nonseparable virtual-bond dependence, exact lazy composition, sparse
phase-resident output messages, boundary-only decoding, actual inverse
restoration, and unrelated reuse of the actual restored carrier.

It does not establish:

```text
a compact full H tensor
bounded-rank TT closure
polynomial time or space scaling in interface width
general factor compression
arbitrary interface width, rank, arity, or topology
wide-interface computational advantage
CATVM enforcement for the width-two or width-three operator
physical waveform, silicon, audio, or phononic execution
Small Wall crossing
unlimited catalytic computation
```

## Reproduction

```bash
evidence_dir=$(mktemp -d /tmp/lazy-tt-qual.XXXXXX)
./qualify_algebraic_lazy_tt_phase.sh . "$evidence_dir" \
    >"$evidence_dir.summary.json"
```
