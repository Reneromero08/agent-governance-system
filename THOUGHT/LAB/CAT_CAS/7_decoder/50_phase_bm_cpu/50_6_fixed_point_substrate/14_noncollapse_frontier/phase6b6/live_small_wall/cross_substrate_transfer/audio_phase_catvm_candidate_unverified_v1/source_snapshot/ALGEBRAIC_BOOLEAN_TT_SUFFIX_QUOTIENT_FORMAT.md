# Boolean-TT Suffix-Bisimulation Quotient

## Status

This mutable checkpoint establishes:

```text
BOUNDED_BOOLEAN_TT_SUFFIX_BISIMULATION_QUOTIENT_REDUCES_PRODUCT_RANK_GROWTH_FROM_EXPONENTIAL_TO_LINEAR_WITH_PHASE_RESIDENT_CLOSURE
```

within:

```text
BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_DEPTHS2_3_4_5_8_NEIGHBOR_AND_OR_FAMILY_SCOPED_QUOTIENT_REFERENCE_ONLY
```

It establishes exact phase-resident quotient composition for homogeneous
repeated neighbor-AND chains and, separately, homogeneous repeated
neighbor-OR chains. It does not establish arbitrary mixed AND/OR layers,
general Boolean-TT minimization, global minimal rank, fixed-rank
unbounded-depth closure, computational advantage, Small Wall crossing,
physical execution, or unlimited catalytic computation.

## Product-state geometry

At composition depth `d`, an unquotiented product bond state is a vertical
column of `d` Boolean layer values. Away from the final word position:

```text
neighbor AND:  111...1100...000
neighbor OR:   000...0011...111
```

The state is determined by a threshold height. Raw product rank `2^d` is
therefore unnecessary for this declared family.

The right boundary requires an additional horizon law. Let a bond have `L`
word sites remaining. Its exact suffix-bisimulation rank is:

```text
r = 1                         at the two outer boundaries
r = 2                         when L = 1
r = min(d+1, L+2)             when L >= 2
```

When `2 <= L < d`, quotient classes are:

```text
height 0 .. L-1      singleton classes
height L .. d-1      one horizon-indistinguishable middle class
height d             one distinct all-leading top class
```

The top class cannot merge with the middle class because only the top can
continue the all-leading history. The first prototype incorrectly omitted
this distinction; independent review caught it before evidence promotion.

For the predecessor stages:

```text
depth 2 H: rank vector [1,3,...,3,2,1], cells 36w-64
depth 3 Z: rank vector [1,4,...,4,2,1], cells 64w-136
```

## Native quotient composition

The quotient plan is derived from public family tag, width, depth, site, and
remaining suffix. It contains no final relation values.

For every output core cell, the native phase operation:

```text
selects one suffix-equivalent left-class representative
ORs every live right-class member
contracts the two local shared Boolean values
reads the actual resident depth-(d-1) quotient stage
reads the actual resident depth-one leaf
writes the depth-d quotient cell directly
```

No raw product tensor is materialized in the accepted phase path. No
width-wide assignment, `4^w` relation table, truth table, witness list, or
candidate set is constructed.

All stages remain phase-resident for inverse execution:

```text
encode leaf
-> H2
-> H3
-> ...
-> Hd
-> copy and decode only final Hd
-> remove final copy
-> inverse Hd ... H2
-> inverse leaf
-> verify restoration
-> reuse actual restored carrier for the other homogeneous family
```

This retained history is deliberately counted. It is not compact inverse
history recursion.

## Independent local certificate

The independent verifier materializes raw product-rank TT cores only inside
the verifier. At width 16/depth 8 its largest final raw stage has 3,672,064
one-byte cells. It does not serialize raw intermediates, enumerate
width-wide assignments, or materialize the dense `4^w` relation.

For every bond and core it computes prefix reachability and suffix
coaccessibility, then verifies:

```text
every live raw state maps to a quotient class
members of a class have identical local suffix signatures
quotient edges equal the OR of live raw edges
initial and terminal acceptance survive
final quotient core hashes, one-counts, and shapes match phase execution
```

This is exponential-in-depth verification of the bounded theorem, not a
resource claim for the accepted phase path.

## Tested scaling

Fifteen `(width, depth)` cases cover all widths at depths two and three plus
`(4,4)`, `(5,5)`, `(8,8)`, `(12,8)`, and `(16,8)`.

At width 16/depth 8:

```text
raw product rank                 256
maximum quotient rank              9
raw final TT cells          3,672,064
quotient final TT cells         3,548
final representation ratio   1,034.967x
retained resident stages       13,740 cells
boundary copy                   3,548 cells
total carrier                  17,288 phase cells
```

For width at least depth, maximum quotient rank grows as `d+1`, not `2^d`.
At fixed width the local quotient saturates no higher than `w+1`. This is a
linear-rank family law, not a fixed-rank closure theorem.

## Controls

The evidence includes:

```text
wrong inverse                    restoration detected
missing deepest inverse          restoration detected
penultimate-before-deepest       restoration detected
snapshot reload                  separate generation-zero path
bad remaining horizon            wrong shape/hash; reference parity fails
wrong Boolean OR phase law       final cells leave Boolean alphabet
intermediate projection          rejected
null carrier                     rejected
depth greater than width         rejected in tested experiment
deterministic replay             exact
analyzer and ASan/UBSan          pass
CATVM Boolean-TT predecessor     passes fresh regression
```

The wrong-horizon control reproduces the discarded overmerge and proves that
restoration alone is not semantic correctness.

## Baseline and next obstruction

The strongest fixture-specialized conventional baseline directly emits the
public quotient cores in `O(final quotient cells)`. The accepted phase path
also retains every stage for inversion, so no speed, memory, fixed-point, or
Small Wall advantage follows.

The next experiment is a matched growing-instance compact-baseline,
snapshot-sham, and in-place phase/CATVM triad. It must count the verifier,
resident inverse history, traffic, projection, restoration, and reuse. If the
triad confirms the direct quotient generator and retained-stage history as
the obstruction, the next mechanism must remove that obstruction or change
the problem family, not add quotient depths.

## Reproduction

From this directory:

```bash
evidence_parent=$(mktemp -d /tmp/boolean-tt-quotient.XXXXXX)
evidence_dir="$evidence_parent/evidence"
bash qualify_algebraic_boolean_tt_suffix_quotient.sh "$evidence_dir"
```

Reviewed evidence:

```text
/tmp/boolean-tt-quotient-fourth
```
