# Bounded compact nested affine DAG execution

## Scope

This mutable experiment replaces retain-all execution for the established
seven-node nested affine DAG with a four-slot reversible schedule. The actual
native-produced shared relations `S` and `T` remain pinned. Internal `I`
remains resident until its last inverse consumer. Only immutable public leaf
bodies are reversibly unencoded and later re-encoded to satisfy exact pending
inverse edges.

This is a fixture-specific schedule. It does not establish a general DAG
compiler, automatic pebbling, arbitrary rematerialization, more than two
shared producers, arbitrary graphs or treewidth, CATVM enforcement for this
compiler, computational advantage, physical execution, Small Wall crossing,
or unlimited catalytic computation.

## Public graph and schedule

The accepted public graph is unchanged:

```text
S    = COMPOSE(F : A->B, G : B->A)
T    = COMPOSE(actual S : A->A, H : A->A)
I    = INTERSECT(actual S : A->A, actual T : A->A)
ROOT = COMPOSE(actual I : A->A, actual same T : A->A)
```

The forward and reverse schedule is:

```text
encode F, G
S = COMPOSE(F, G)
inverse-encode and release G, F

encode H
T = COMPOSE(actual S, H)
inverse-encode and release H

I = INTERSECT(actual S, actual T)
ROOT = COMPOSE(actual I, actual T)
project ROOT only
remove the boundary copy

inverse ROOT
inverse I

re-encode H from its immutable public body
inverse T using actual S and reconstructed H
inverse-encode and release H

re-encode F, G from their immutable public bodies
inverse S using reconstructed F and G
inverse-encode and release G, F
```

No native-produced internal relation is rematerialized and no operator is
recomputed. `S`, `T`, and `I` retain their original resident generations
until their inverse applications. Reconstructed degree-one leaves use fresh
lease generations, which are bound to their exact still-pending inverse
edges. Public leaf reconstruction performs six additional reversible leaf
encodes, so compact execution trades leaf work and allocations for three
fewer working relation blocks.

## Reconstruction obligations

Early leaf release is legal only after its sole forward edge commits. The
runtime records a structural obligation:

```text
public leaf body index
nominal signature
exact pending consumer edge
public-program epoch
```

The obligation contains no decoded relation contents. Reconstruction requires
the same public body, signature, epoch, and still-pending inverse edge. A leaf
can be reconstructed once, must be consumed by that exact inverse edge, and
must then be inverse-encoded and released. Shared or internal relations are
not eligible for this path.

Controls reject wrong public bodies, missing or double reconstruction,
skipped final leaf inverse, early eviction of `S`, `T`, or `I`, and premature
inverse of either shared producer.

## Four-slot lower bound

Under the declared machine law, native operators require distinct live left,
right, and output relation blocks, and shared producer generations cannot be
released or rematerialized before all exact inverse consumer edges complete.
Immediately before forward `ROOT`, `S` must remain pinned because the inverse
of both `T` and `I` still depends on it; actual `T` and actual `I` are the two
`ROOT` inputs; and `ROOT` needs a distinct output block. Therefore this graph
requires at least four simultaneously live working relation blocks:

```text
actual S + actual T + actual I + ROOT output = 4
```

The four-slot execution passes. The same strict source compiled with a
three-slot carrier fails before projection with pool exhaustion. This is a
lower bound only for this exact graph and declared non-aliasing/pinned-shared
law, not a general optimal-pebbling theorem.

## Restoration and reuse

The accepted path removes the sole boundary copy, applies the actual inverse
of `ROOT`, `I`, `T`, and `S` in dependency order, discharges every leaf
obligation, releases all ten leases, and verifies exact discrete reset plus
complex carrier equality at the predeclared tolerance `2e-12`.

Seventeen alternating transactions reuse one carrier allocation. The observed
maximum restoration error is `6.66133814775e-16`. Wrong and missing root
inverse controls leave an error of `1.73205080757`; snapshot reload remains a
separate labelled weaker baseline.

## Bounded resources

Let:

```text
B(w) = 4w^2 + 4w + 1
W(w) = 36w^2 + 11w + 2
```

The phase workspace and two physical boundary blocks are unchanged. Working
relation storage is:

```text
compact schedule       W(w) + 6B(w)  = 60w^2 + 35w + 8
retain unique nodes    W(w) + 8B(w)  = 68w^2 + 43w + 10
occurrence expansion   W(w) + 16B(w) = 100w^2 + 75w + 18
```

At width sixteen the compact path uses 15,928 complex cells and 509,696 live
carrier bytes, compared with 18,106 cells for retain-all and 26,818 cells for
the occurrence expansion. The accepted result records exact measured
resources through width sixteen. Compiler stack evidence records 8,176 bytes
for `cp_execute` and 39,232 bytes for `main`, but no total-memory comparison
combines compact metadata, stack, binary, output, and all baselines. The
established reduction is therefore phase-carrier cells only, not total
memory, performance, or computational advantage.

The operation trade is:

```text
                         compact     retain-all     occurrence tree
working relation slots         4              7                  15
native calls                    8              8                  14
leaf encode calls              12              6                  16
lease allocations              10              7                  15
```

## Qualification

The qualifier hard-gates strict and analyzer builds, deterministic replay,
ASan/UBSan/leak checks, 25 complete phase/reference boundary matches at widths
`3, 4, 8, 12, 16`, exact owner/edge custody, exact schedule counters,
retain-all and occurrence-expanded equality, seventeen same-carrier
transactions, a three-slot capacity failure, restoration controls, and
expanded file/network/IPC/memory no-smuggle tracing.
