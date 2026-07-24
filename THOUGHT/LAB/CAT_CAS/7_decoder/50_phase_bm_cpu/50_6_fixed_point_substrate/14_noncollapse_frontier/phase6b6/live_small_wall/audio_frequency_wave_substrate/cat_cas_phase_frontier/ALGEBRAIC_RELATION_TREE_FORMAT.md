# Algebraic Relation-Tree Phase Format

Status: mutable CAT_CAS frontier checkpoint

## Purpose

This format removes the fixed two-hub topology. It describes a public typed
tree whose edges are open bi-total Boolean relations. The native engine
constructs the relational boundary between every external-leaf pair by
propagating four-phase relation messages along the unique public tree path.

The format is bounded by implementation capacities, but neither topology nor
internal depth is hardcoded.

## Canonical grammar

```text
CATCAS_ALGEBRAIC_RELATION_TREE 1
TYPE BOOLEAN_F3
NODE name EXTERNAL|INTERNAL
...
RELATION name first second c00 c10 c01 c11
...
END
```

All `NODE` records precede all `RELATION` records. Nodes are declaration
ordered. Relation records may be presented in any order and either endpoint
direction. Identifiers match `[A-Z][A-Z0-9_]{0,30}`. Coefficients are one
ASCII digit in `{0,1,2}`. Every record ends in LF; CR, blank records, trailing
records, undeclared endpoints, duplicate identifiers, self edges, and
parallel edges reject.

The admitted graph must be one connected tree. Every `EXTERNAL` node has
degree one. Every `INTERNAL` node has degree at least two. The current
native-engine capacities are:

```text
nodes                  <= 64
edges                  <= 63
external nodes         <= 32
external pairs         <= 496
edge count on one path <= 63
```

The separate scalar adjudicator enumerates internal assignments and therefore
has a narrower explicit bound of at most 20 internal nodes. That reference
bound does not constrain native parsing or execution. The committed exact
reference fixtures use at most four internal nodes; the complete extensional
survey independently exhausts all relation labelings of the three-internal
discriminator. No claim is made that scalar exhaustive adjudication is
practical at the native engine's maximum external or internal capacity.

Each edge is the Boolean zero set of

```text
f(x,y) = c00 + c10*x + c01*y + c11*x*y  over F3.
```

Every edge relation must be total toward both Boolean endpoints. The admitted
coefficient class has 17 signatures and seven distinct extensional
relations.

## Native path composition

For a path

```text
X -- N1 -- N2 -- ... -- Nk -- Y
```

the engine orients the first edge as a relation from `X` to `N1`. Each next
edge is oriented from the next node back toward the shared node. If the
current four-phase relation is

```text
f(x,u) = A(x)*u + B(x)
```

and the next is

```text
g(v,u) = C(v)*u + D(v),
```

the next resident message is

```text
R(x,v) = B(x)*C(v) - A(x)*D(v).
```

Coefficient multiplication is the fixed roots-of-unity product polynomial.
Conjugate phase multiplication performs subtraction. Every non-final
resultant occupies four fresh carrier-relative phases and directly feeds the
next resultant. Only the final four-phase factor for each external pair is
decoded.

For `E` public edges and paths `P`, native residency is:

```text
input cells       = 4E
message cells     = 4 * sum_path(max(path_edges - 2, 0))
boundary cells    = 4P
tuple slots       = 0
witness slots     = 0
truth-table slots = 0
```

The native resource law is `O(E + sum_pair_path_length)`. It does not depend
on `2^(internal nodes)` and does not allocate completed assignments.

## Why the boundary is exact

Fix all external Boolean values. Every leaf sends a nonempty allowed subset
of `{0,1}` inward because all edge relations are bi-total. Propagation across
a bi-total relation preserves nonemptiness.

At a branch, incoming allowed sets are subsets of a two-point domain:

- if their intersection is empty, two incoming singleton sets are opposite;
- if their intersection is a singleton, at least one incoming set is that
  singleton;
- otherwise both values remain allowed.

Therefore any restriction or contradiction at an internal node is traceable
to one leaf or an incompatible pair of leaves. Induction over the tree gives:

```text
one global internal assignment exists
iff
every external-leaf pair is compatible along its unique path.
```

The existing complete 289-pair closure establishes that each admitted
resultant remains in the same bi-total relation class, so the phase operation
may be applied inductively without changing the admission law.

The new complete extensional survey covers all `7^7 = 823,543` relation
labelings of a discriminating three-hub/five-leaf tree and all 32 external
assignments, or `26,353,376` rows, with zero mismatches.

## Catalytic lifecycle

The native lifecycle is:

```text
borrow one complex carrier
encode every public edge relation
construct every unique path through resident four-phase messages
latch the external pair boundary
remove every final boundary factor
remove every path message in reverse dependency order
remove every input relation
restore the borrowed carrier
reuse that exact carrier for a different relation tree
```

The boundary record is copied outside the reversed carrier history before
inversion.

## Controls

The primary fixture requires all four controls to be applicable:

- rotating the coefficient factors used by the boundary inverse leaves at
  least `1e-3` restoration error;
- transposing the final public edge geometry of every path changes the
  boundary, while reversing that exact perturbed traversal restores cleanly;
- omitting one nontrivial phase-message inverse leaves at least `1e-3`
  restoration error;
- bypassing all interior path messages changes the boundary, while reversing
  that exact bypass restores cleanly.

The all-`ANY` fixture legitimately marks all four controls inapplicable and
still proves witness-multiplicity invariance.

## Reference separation

`algebraic_relation_tree_reference.c` is a separate scalar executable. It
composes scalar F3 coefficients, enumerates at most 20 internal nodes
afterward, and compares the factorized boundary with exact existential
closure. It is not linked into the native binary. This explicit enumeration
bound belongs only to adjudication, not to the native resource law.

`algebraic_relation_tree_survey.c` is a separate complete extensional
adjudicator for the three-hub topology. Neither reference source participates
in native recurrence or boundary selection.

## Claim boundary

A passing checkpoint establishes only:

```text
SOFTWARE_GENERIC_BOUNDED_RELATION_TREE_PHASE_CLOSURE_REFERENCE_ONLY
```

It establishes topology-generic finite tree parsing, phase-resident recursive
relation composition, exact bounded tree closure, actual inverse traversal,
restoration, and reuse.

It does not establish cyclic relational trace, unrestricted domains or
relations, an asymptotic advantage over compact classical tree algorithms,
physical phase computation, or unlimited catalytic phase computation.
