# Bounded nested affine DAG multi-message custody

## Scope

This mutable experiment extends the bounded affine DAG substrate from one
shared native-produced relation message to exactly two. It proves a nested
case: the actual resident `S` produces `T`, then `S` and `T` remain live at
the same time and each is consumed by two public typed edges.

It does not establish arbitrary DAGs, compact pebbling of shared DAGs,
unbounded fanout, arbitrary graphs or treewidth, CATVM enforcement,
performance advantage, physical execution, Small Wall crossing, or unlimited
catalytic computation.

## Public graph

The existing topology-only manifest syntax is unchanged. The accepted graph
is:

```text
S    = COMPOSE(F : A->B, G : B->A)
T    = COMPOSE(actual S : A->A, H : A->A)
I    = INTERSECT(actual S : A->A, actual T : A->A)
ROOT = COMPOSE(actual I : A->A, actual same T : A->A)
```

It has seven unique nodes, eight edges, depth four, and exactly two
native-produced fanout nodes of degree two. `S` and `T` have the same nominal
signature, so typed cross-substitution controls distinguish exact generation
identity rather than merely rejecting a type mismatch.

At width three the primary final relation is:

```text
x0 + z0 = 0
x1 + z1 = 0
x2      = 0
z2      = 0
```

Its complete canonical boundary hash is `59f0245207f2a0f1`.

## Per-owner and per-edge custody

Every public edge follows:

```text
DECLARED -> FORWARD_CONSUMED -> INVERSE_CONSUMED
```

Each shared public producer has a separate runtime owner receipt containing
only structural custody state:

```text
actual live lease
materialization and inverse counts
owner allocation and release counts
forward and inverse consumer counts
observed generation-equality state
live-at-projection and live-after-root-inverse markers
structural lifecycle events
```

Each shared outgoing edge separately records its forward slot/serial
generation and requires the inverse observation to match it. The forward
transition is committed only after the native parent operation succeeds.
Inverse readiness checks both pending counts and every outgoing edge token.

For each of public nodes `704` (`S`) and `705` (`T`), the accepted path
observes exactly:

```text
one owner allocation and materialization
two forward binds
two inverse binds
one actual inverse application
one owner release
one slot and one serial generation across all four binds
live at projection
live after ROOT inverse
```

The two owners occupy distinct slots while simultaneously live. Their nested
lifecycle is observed rather than inferred:

```text
birth(S) < birth(T) < projection < inverse(ROOT)
         < inverse(T) < release(T) < inverse(S) < release(S)
```

Only `ROOT` is copied to the boundary. No intermediate relation is copied,
decoded, hashed, serialized, or content-committed.

## Restoration and reuse

The accepted path removes the boundary copy, applies actual dependency-ordered
inverse operations to the actual resident nodes, releases every lease, resets
all owner receipts, edge receipts, edge tokens, and pending counts, and checks
the borrowed carrier against its pre-transaction state at the predeclared
complex tolerance `2e-12`.

Exact discrete postconditions include:

```text
zero live leases
all eight edge tokens reset to DECLARED
all owner and edge receipts cleared
all pending counts zero
empty scheduler queue
serial_after = serial_before + 7
restoration_generation_after = restoration_generation_before + 1
```

Seventeen unrelated alternating transactions reuse the same carrier
allocation. The observed maximum restoration error is
`2.73360717445e-16`. Snapshot reload is tested only as a labelled weaker
baseline.

## Independent reference and occurrence expansion

The conventional GF(2) reference contains no phase-carrier implementation
and memoizes each unique public node once. Complete final boundaries match
the phase path for five semantic variants at widths `3, 4, 8, 12, 16`.

The exact occurrence-expanded tree instantiates the same three immutable leaf
bodies eight times:

```text
                         shared DAG       expanded tree
logical nodes                     7                  15
working relation blocks           7                  15
native calls, forward+inverse      8                  14
leaf toggles, forward+inverse      6                  16
```

It produces the same complete final relation and restores by actual inverse,
but contains no fanout and therefore fails the accepted custody predicate.

With `B(w)=4w^2+4w+1` and `W(w)=36w^2+11w+2`:

```text
accepted carrier = W(w) + 8B(w)  = 68w^2 + 43w + 10
expanded tree    = W(w) + 16B(w) = 100w^2 + 75w + 18
```

At width sixteen the accepted path uses 18,106 complex cells and 579,392
live carrier bytes. These are bounded software resource counts, not an
advantage claim.

## Controls

The qualifier hard-gates:

```text
same-typed S-for-T and T-for-S substitution
equal-content typed clone of each owner
stale serial for each owner
skipped and duplicate consumer transition for each owner
premature inverse of S and T
projection of S, T, or custody state
null carrier
cycle, type, and disconnected topology
one-fanout and zero-fanout accepted-topology rejection
wrong and missing root inverse
semantic branch necessity and empty propagation
snapshot comparison
occurrence-expanded boundary equality
actual restored-carrier reuse
expanded file, network, IPC, and memory no-smuggle tracing
```
