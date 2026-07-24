# Algebraic Two-Hub Relation-Memory Format

Status: mutable CAT_CAS frontier checkpoint

## Purpose

This bounded format is the first process in the lane where the output of one
native relation closure remains resident as a complex relation and becomes an
input to a second native closure before any boundary decoding.

The public typed process is:

```text
A -- U -- V -- C
B --'    '-- D
```

`A`, `B`, `C`, and `D` are open Boolean boundary ports. `U` and `V` are
unresolved Boolean internal hubs. Five public binary relations occupy the
four leaf edges and the `U--V` bridge.

This is deliberately one fixed two-hub tree. It is a mechanism discriminator,
not a general graph language.

## Canonical grammar

```text
CATCAS_ALGEBRAIC_TWO_HUB_TREE 1
TYPE BOOLEAN_F3
HUBS U V
RELATION LEFT_A A U c00 c10 c01 c11
RELATION LEFT_B B U c00 c10 c01 c11
RELATION BRIDGE U V c00 c10 c01 c11
RELATION RIGHT_C C V c00 c10 c01 c11
RELATION RIGHT_D D V c00 c10 c01 c11
END
```

The five `RELATION` records may appear in any order. Either endpoint direction
is accepted; reversed endpoints transpose `c10` and `c01` before execution.
Each role must appear exactly once and must connect its mechanically assigned
ports. Coefficients are one canonical ASCII digit in `{0,1,2}`. Records end
with LF. CR, embedded truncation, blank records, records after `END`, missing
roles, duplicate roles, and noncanonical coefficients reject.

Every public relation is the Boolean zero set of

```text
f(x,y) = c00 + c10*x + c01*y + c11*x*y  over F3.
```

Every relation must be total toward both Boolean ports. This prospective
admission law leaves 17 coefficient signatures representing seven distinct
extensional Boolean relations.

## Native relation memory

Write the left leaf relation as

```text
f_A(a,u) = A_A(a)*u + B_A(a)
```

and transpose the bridge into

```text
g(v,u) = A_G(v)*u + B_G(v).
```

The first native closure eliminates `U`:

```text
M_A(a,v) = B_A(a)*A_G(v) - A_A(a)*B_G(v).
```

The four coefficients of `M_A` are encoded in four carrier-relative cube-root
phases. The engine does not decode them. It directly consumes those phases
with the `RIGHT_C` and `RIGHT_D` phase relations to close `V`. `LEFT_B`
produces and consumes a second phase-resident message in the same way.

The final boundary is the conjunction of six binary factors:

```text
A:B
C:D
A:C
A:D
B:C
B:D
```

The two local factors close one hub directly. Each cross factor is a
two-stage phase resultant whose first-stage result is the resident relation
message.

## Why the six factors are exact here

Fix a public assignment `(a,b,c,d)`. Each leaf relation induces a nonempty
allowed subset of its adjacent Boolean hub.

For subsets of a two-point domain, pairwise intersection implies common
intersection. The local `A:B` and `C:D` factors therefore establish nonempty
common allowed sets at `U` and `V`. If either common set becomes a singleton,
one of the corresponding public leaf constraints is already that singleton;
the relevant cross factor then mechanically checks the exact bridge pair. If
a common set remains both hub values, bridge bi-totality supplies a compatible
partner. Consequently all six path factors hold exactly when some common
`(u,v)` satisfies all five public relations.

This law is special to the admitted Boolean bi-total two-hub tree. It is not a
claim about arbitrary domains, cyclic graphs, or unrestricted relations.

## Phase arithmetic

Each F3 coefficient `k` is represented as `exp(2*pi*i*k/3)` relative to the
borrowed carrier baseline. A fixed roots-of-unity product polynomial computes
coefficient products. Conjugate multiplication performs subtraction. Unit
normalization bounds floating-point radial drift.

The native path is:

```text
borrow 52 complex carrier cells
encode five public relations
derive two four-phase relation messages by closing U
consume those message phases while closing V
latch and decode the six final boundary factors
remove the actual boundary factors
remove the actual relation messages
reverse the five public encodings
restore and reuse the same carrier
```

Carrier layout:

```text
public relation cells          5 * 4 = 20
relation-message cells         2 * 4 =  8
boundary-factor cells          6 * 4 = 24
total complex cells                      52
tuple slots                              0
witness slots                            0
truth-table slots                        0
decoded intermediate coefficients        0
retained inverse factors                 0
```

## Controls

The native executable reports:

- a wrong coefficient-factor boundary inverse;
- a cyclic geometry-scrambled boundary inverse;
- omission of the second relation-message inverse;
- forward bypass of relation memory, which treats the two hubs as one.

Every inverse control is applied only when the exact removed factor differs
from the correct factor. The bypass control compares the complete decoded
boundary, not an energy or selected witness. A legitimately trivial control
is reported as inapplicable.

## Reference separation

`algebraic_two_hub_reference.c` derives the same factor coefficients with
ordinary scalar F3 arithmetic and independently enumerates all 16 public
assignments and four internal `(u,v)` assignments. It is a post-execution
bounded adjudicator and is not linked into the native binary.

`algebraic_two_hub_closure_survey.c` exhausts the complete admitted
five-relation coefficient class. It is also separate from the native binary.

## Claim boundary

A passing checkpoint establishes only:

```text
SOFTWARE_TWO_HUB_RELATION_VALUED_PHASE_MEMORY_REFERENCE_ONLY
```

It shows one finite two-stage native relational closure, an intermediate
relation held and consumed as complex phase, exact bounded projection, actual
inverse traversal, restoration, and reuse.

It does not establish a general process graph, a cycle, unbounded recursion,
computational advantage, physical phase computation, or unlimited catalytic
phase computation.
