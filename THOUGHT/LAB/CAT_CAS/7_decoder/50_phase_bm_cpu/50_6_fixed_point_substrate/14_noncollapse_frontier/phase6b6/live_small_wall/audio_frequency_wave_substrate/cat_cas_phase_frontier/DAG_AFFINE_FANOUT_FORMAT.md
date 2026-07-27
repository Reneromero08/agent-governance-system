# Bounded affine DAG shared-message format

## Scope

This mutable experiment extends the public recursive affine tree language by
one bounded internal fanout. It proves that one native-produced unresolved
relation block can remain resident and be consumed by two typed parents.

It does not claim a general DAG compiler, arbitrary graphs or treewidth,
CATVM enforcement, performance advantage, physical execution, Small Wall
crossing, or unlimited catalytic computation.

## Public syntax

The accepted manifest uses the existing public declarations:

```text
root ID
leaf ID DOMAIN CODOMAIN LEAF_BODY_ID
compose ID LEFT RIGHT
intersect ID LEFT RIGHT
```

The accepted graph is:

```text
S    = COMPOSE(F : A->B, G : B->C)
L    = COMPOSE(actual S : A->C, H : C->D)
R    = COMPOSE(actual same S : A->C, K : C->D)
ROOT = INTERSECT(actual L : A->D, actual R : A->D)
```

It has eight nodes, eight edges, depth three, and exactly one native-produced
node with two consumers. Declarations are scrambled. The compiler resolves
and validates the whole graph, derives types bottom-up, and only then assigns
addresses.

At width three the primary root relation is:

```text
x0 + z2 = 0
x1 + z0 = 0
x2      = 0
z1      = 0
```

Its complete canonical boundary hash is `630132cdcd942021`.

## Shared custody law

Every public operand edge has a discrete state:

```text
DECLARED -> FORWARD_CONSUMED -> INVERSE_CONSUMED
```

Every resident relation lease contains:

```text
slot
logical owner
monotonic serial
nominal signature
live/reserved state
```

The shared producer is allocated and materialized once. Both consumers must
bind the exact same slot, owner, serial, signature, and carrier allocation.
Equal coefficients or a compatible nominal type cannot substitute another
lease. No intermediate relation copy is permitted.

The accepted forward schedule retains each unique DAG node once. The inverse
schedule is reverse postorder:

```text
ROOT^-1
-> R^-1
-> K^-1
-> L^-1
-> H^-1
-> S^-1
-> G^-1
-> F^-1
```

`S^-1` is illegal while either consumer edge remains live. The sibling
inverses `L^-1` and `R^-1` are not required to reject in the opposite order;
after `ROOT^-1` they read the same resident `S` and act on distinct outputs.

## Projection and restoration

Only the final `ROOT` relation is copied to the dedicated boundary block and
decoded. The accepted path then removes that copy, applies the actual
dependency-ordered inverses to the actual resident blocks, checks every
released block against exact discrete zero state within the predeclared
complex tolerance `2e-12`, and reuses the same carrier allocation.

Canonical post-transaction state is:

```text
all carrier cells restored within 2e-12
workspace cleared
zero live relation leases
all edge tokens reset to DECLARED
shared custody receipt reset
all pending consumer counts zero
empty scheduler queue
serial_after = serial_before + lease_allocations
restoration_generation_after = restoration_generation_before + 1
```

The shared owner allocation/release count, peak live-instance count, and
distinct slot/serial sets are observed at runtime rather than inferred from
topology. Projection and post-root-inverse markers also validate the stored
shared receipt against the live lease table.

Monotonic serial and restoration-generation fields are lawful state
transitions, not host details required to return to their old values.

Snapshot reload is measured separately and is not used by the accepted
restoration path.

## No-smuggle law

The accepted process emits no intermediate:

```text
coefficients
rank or pivots
hash or commitment
aggregate
tuple, assignment, witness, or candidate set
```

Its output contains the final complete root plus public topology, custody, and
resource counts. Expanded syscall tracing rejects write-capable file opens,
non-stdout writes, positioned writes, sockets and sends, shared mappings,
memory-file creation, SysV IPC, splice/sendfile/copy routes, cross-process
writes, ptrace, and ioctl routes. Core dumps are disabled.

## Matched duplicate-tree baseline

The occurrence-expanded tree uses the same four immutable compiled leaf
bodies but instantiates `F`, `G`, and `S=COMPOSE(F,G)` twice. It produces the
same complete final boundary and restores by actual inverse, but it has two
producer owners/serials and therefore fails the accepted sharing predicate.

```text
                         shared DAG        duplicate tree
logical nodes                    8                    11
working relation blocks          8                    11
native calls, forward+inverse     8                    10
leaf toggles, forward+inverse     8                    12
```

The accepted carrier laws are:

```text
B(w) = 4w^2 + 4w + 1
W(w) = 36w^2 + 11w + 2
C(w) = W(w) + 9B(w) = 72w^2 + 47w + 11
```

The duplicate-tree carrier is `W(w)+12B(w)`.

At width sixteen the accepted path uses 19,195 cells and 614,240 live carrier
bytes. It executes 4,948,416 phase ANDs and 9,628,416 phase XORs. The
conservative current-ABI accounting, including the verification copy,
topology, programs, edge custody, boundary, stack, and manifests, is
1,375,336 bytes.

These are bounded software resource results, not a performance-advantage
claim.

## Controls

The qualifier hard-gates:

```text
cycle, type, and disconnected-graph rejection
duplicate tree rejected as the accepted fanout graph
stale shared serial
skipped and duplicate consumer transitions
premature producer inverse with live dependents
attempted intermediate and custody projection
attempted intermediate copy
null carrier
wrong and missing root inverse
left and right semantic branch necessity
empty intersection
snapshot comparison
same-carrier unrelated reuse
expanded no-smuggle trace
```
