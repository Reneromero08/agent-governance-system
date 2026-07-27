# Recursive Affine Module Compiler

## Status

Mutable development result:

```text
BOUNDED_PUBLIC_RECURSIVE_AFFINE_SERIES_INTERSECTION_TREE_COMPILATION
```

Claim ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_AFFINE_TREE_COMPILER_REFERENCE_ONLY
```

This is a topology/compiler and compact-carrier successor to the reviewed
mixed affine module. It is not a CATVM enforcement claim, an arbitrary-DAG
claim, an arbitrary-graph claim, a performance-advantage claim, a physical
execution claim, or a Small Wall crossing.

## Public topology format

`recursive_affine_topology.txt` contains only public nominal geometry:

```text
root NODE_ID
leaf NODE_ID DOMAIN CODOMAIN LEAF_BODY_ID
compose NODE_ID LEFT_NODE_ID RIGHT_NODE_ID
intersect NODE_ID LEFT_NODE_ID RIGHT_NODE_ID
```

Operator signatures are derived by the compiler:

```text
COMPOSE(P -> Q, Q -> R)    -> P -> R
INTERSECT(P -> Q, P -> Q)  -> P -> Q
```

The source declarations are deliberately scrambled. Carrier addresses do not
occur in the manifest and are assigned only after the complete topology has
passed validation.

The compiler requires:

```text
one root
unique node IDs
unique leaf-body ownership
known child IDs
acyclicity by tri-color DFS
one parent per nonroot node
no DAG fanout
no repeated or self operands
all nodes reachable
at least one composition
at least one intersection
derived nominal type compatibility
bounded node and slot counts
```

It emits a deterministic left-before-right postorder. The machine binds each
logical node to an actual carrier lease:

```text
physical slot
logical owner
lease serial
live or reserved state
derived nominal signature
```

A stale, missing, overlapping, aliased, or mistyped lease is rejected before
the phase operator can run.

## Qualified tree

The public tree contains fifteen nodes:

```text
LEFT_A  = COMPOSE(L0,L1)
LEFT_B  = COMPOSE(L2,L3)
LEFT    = INTERSECT(LEFT_A,LEFT_B)

RIGHT_A = COMPOSE(L4,L5)
RIGHT_B = COMPOSE(L6,L7)
RIGHT   = INTERSECT(RIGHT_A,RIGHT_B)

ROOT    = COMPOSE(LEFT,RIGHT)
```

It has:

```text
8 leaves
5 composition nodes
2 intersection nodes
14 edges
depth 3
```

The primary leaf bodies are total affine maps. The left intersection imposes
one nontrivial equality between two composed paths. The right intersection
imposes another equality between two composed paths. At width three the final
boundary is:

```text
x0 = 1
x1 + e1 = 0
x2 + e1 = 0
e0 + e1 = 1
e2 = 0
```

Its complete 49-cell canonical boundary hashes to:

```text
e9f225404bbe6d62
```

An unrelated all-identity reuse program on the actual restored carrier
produces:

```text
x0 + e0 = 0
x1 + e1 = 0
x2 + e2 = 0
```

and hashes to:

```text
6f46809cf9d2cb8e
```

The remaining semantic bodies cover a contradictory parallel branch, a
highest-index input restriction, and explicit-empty propagation.

## Reversible pebble law

Retaining all nodes is a valid native baseline:

```text
encode every leaf
-> execute seven operators in compiled postorder
-> retain all fifteen node relations
-> project ROOT only
-> inverse seven operators in reverse postorder
-> inverse every leaf
```

It needs fifteen working relation blocks plus a dedicated boundary block and
uses fourteen native operator applications.

The accepted compact path instead evaluates a node as:

```text
evaluate actual left child
-> evaluate actual right child
-> allocate a fresh typed output lease
-> apply the native operator to those actual resident children
-> actually inverse/uncompute right child
-> actually inverse/uncompute left child
-> retain only the actual parent
```

Restoration is the exact recursive dual:

```text
reconstruct actual left child
-> reconstruct actual right child
-> apply the native inverse to the actual parent
-> actually inverse/uncompute right reconstruction
-> actually inverse/uncompute left reconstruction
-> release the parent only after its carrier block is baseline-zero
```

No snapshot participates in this accepted restoration path.

For fixed left-before-right evaluation, clean-slot requirements are compiled
bottom-up:

```text
E(leaf) = U(leaf) = 1

E(node) = max(
  E(left),
  1 + E(right),
  3,
  2 + U(right),
  1 + U(left)
)

U(node) = max(
  1 + E(left),
  2 + E(right),
  3,
  2 + U(right),
  1 + U(left)
)
```

The qualified balanced tree compiles to seven working slots. Including the
dedicated boundary, it occupies eight relation blocks rather than sixteen.

If `C(leaf)=0`, one clean evaluation or unevaluation obeys:

```text
C(node) = 2 * (C(left) + C(right)) + 1
```

The root costs twenty-one native calls in either direction, so one accepted
transaction costs forty-two. This is an explicit memory/recomputation
tradeoff, not computational leverage.

## Carrier and resource laws

The inherited complete affine relation and shared workspace sizes are:

```text
B(w) = 4w^2 + 4w + 1
W(w) = 36w^2 + 11w + 2
```

The accepted seven-slot path plus dedicated boundary is:

```text
C_pebble(w) = W(w) + 8B(w)
             = 68w^2 + 43w + 10
```

The retain-all baseline is:

```text
C_retain(w) = W(w) + 16B(w)
             = 100w^2 + 75w + 18
```

At width three:

```text
B = 49
W = 359
C_pebble = 751 cells
pebbled live carrier = 24,032 bytes
C_retain = 1,143 cells
retain-all live carrier = 36,576 bytes
```

At width sixteen:

```text
B = 1,089
W = 9,394
C_pebble = 18,106 cells
pebbled live carrier = 579,392 bytes
C_retain = 26,818 cells
retain-all live carrier = 858,176 bytes
```

The accepted width-sixteen verifier accounts for:

```text
carrier plus restoration-verification snapshot heap  1,158,784 bytes
compiled topology metadata                            1,536 bytes
five public leaf-program bodies                      43,560 bytes
decoded final boundary storage                        4,376 bytes
conservative compiler-measured stack                 78,368 bytes
public manifest                                         472 bytes
conservative accounted execution total            1,287,096 bytes
```

The phase executable, reference executable, controller output, and reference
output are also recorded separately in the qualifier summary.

One accepted transaction has exactly:

```text
21 forward native operations
21 inverse native operations
14 transient forward recomputations
21 inverse-factor recomputations
64 forward leaf encodes
64 inverse leaf encodes
85 lease allocations
85 lease releases
0 outstanding leases
```

The coefficient-oblivious native operator schedule is identical for
composition and intersection. Therefore:

```text
AND = 21 * (288w^3 + 225w^2 - 9w)
XOR = 42 * (288w^3 + 96w^2 - 42w)
```

At width sixteen the accepted primary run measured:

```text
25,979,184 phase ANDs
50,549,184 phase XORs
38,822,772 phase-cell updates
51,739,332 native phase-carrier reads
52,667,985 total logical carrier-cell inspections
```

## Projection, restoration, and reuse

Only the copied final `ROOT` block is decoded. Intermediate relation
coefficients, empty flags, ranks, pivot patterns, hashes, row activity, and
workspace values never cross the process boundary.

The accepted path:

```text
compiles public geometry
-> executes native phase operators through address-bound leases
-> copies and decodes ROOT only
-> removes the boundary copy
-> reconstructs dependencies natively
-> applies actual parent inverses
-> releases every block only after inverse restoration
-> compares the canonical full carrier with its borrowed baseline
-> runs unrelated programs on the same restored carrier
```

Seventeen accepted same-carrier transactions at width three had maximum
restoration error:

```text
8.32667268469e-16
```

The predeclared tolerance is:

```text
2e-12
```

## Controls and adjudication

The qualifier establishes:

```text
5 of 5 default complete boundaries match the independent reference
10 of 10 scaled primary/reuse boundaries match at widths 3,4,8,12,16
strict compiler builds pass
static analyzer passes
ASan/UBSan/leak checks pass
deterministic replay passes
mandatory no-smuggle strace passes
```

Rejected controls cover:

```text
cycle
unknown child
fanout or disconnected tree
nominal type mismatch
duplicate node ID
wrong root selection
multiple roots
duplicate leaf-body ownership
intermediate projection
phase-control projection
null carrier
producer-before-consumer inverse reorder
```

Wrong and missing root inverses each leave restoration error:

```text
1.73205080757
```

The retain-all native phase baseline yields the exact same final boundary and
restores its larger carrier. The snapshot baseline reloads saved state and is
kept explicitly separate from the accepted actual-inverse path.

## Claim boundary

Established:

```text
public manifest parsing
scrambled declaration order
deterministic compiled postorder
derived nominal signatures
address-bound typed leases
bounded recursive composition/intersection trees
actual resident child-to-parent consumption
reversible pebble allocation
boundary-only final projection
actual inverse restoration
actual restored-carrier reuse
width-parametric exact boundaries through width sixteen
```

Not established:

```text
general holographic relational computation
DAG fanout
arbitrary graph topology
compact wide separators beyond complete affine descriptors
computational or asymptotic advantage
CATVM enforcement for this recursive compiler
physical waveform or silicon computation
Small Wall crossing
unlimited catalytic computation
```
