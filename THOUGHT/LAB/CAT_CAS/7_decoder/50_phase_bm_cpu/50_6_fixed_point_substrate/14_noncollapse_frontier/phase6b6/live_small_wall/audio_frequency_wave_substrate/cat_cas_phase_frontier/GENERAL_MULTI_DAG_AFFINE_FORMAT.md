# Bounded four-owner retain-all affine DAG custody

## Scope

This mutable experiment generalizes native resident-message custody from the
previous exact two-owner graph to a bounded fifteen-node DAG. The topology
compiler derives four shared native-produced owners with heterogeneous
fanouts `4,3,3,2`; every outgoing edge obtains a forward and inverse receipt
for the exact owner generation.

The wrapper intentionally accepts only this fifteen-node graph law. It is not
a general DAG compiler and does not establish automatic compact scheduling,
arbitrary topology, CATVM enforcement for this compiler, performance
advantage, physical execution, Small Wall crossing, or unlimited catalytic
computation.

## Public graph

All four leaves and all internal values have signature `A -> A`:

```text
S    = COMPOSE(L1, L2)       public 805, fanout 4
T    = COMPOSE(L3, L4)       public 806, fanout 3
U    = COMPOSE(S, T)         public 807, fanout 3
V    = COMPOSE(T, S)         public 808, fanout 2
P    = COMPOSE(S, U)
Q    = COMPOSE(T, U)
R    = COMPOSE(S, V)
K    = COMPOSE(U, V)
J    = INTERSECT(P, Q)
M    = INTERSECT(R, K)
ROOT = INTERSECT(J, M)
```

The manifest is deliberately scrambled before compilation. The compiled
graph has fifteen unique nodes, twenty-two edges, depth five, eight
composition nodes, and three intersection nodes.

At width three the primary complete final relation has canonical hash
`2e2200557469163d`. Five nonempty, nonuniversal semantic variants have
distinct hashes and match a separate conventional GF(2) reference at widths
`3,4,8,12,16`, giving twenty-five complete-boundary matches.

## Exact per-edge custody

Each shared edge follows:

```text
DECLARED -> FORWARD_CONSUMED -> INVERSE_CONSUMED
```

The runtime records a forward slot and serial only after its native consumer
operation succeeds. The inverse observation must use the same slot and
serial. Owner exactness is recomputed from every actual outgoing edge
receipt, the observed owner generation, and exact allocation,
materialization, inverse, release, and consumption counts; it is not accepted
from an initialized summary flag.

The accepted transaction observes:

```text
12 exact forward/inverse shared-edge receipts
4 owner allocations and materializations
12 shared forward consumptions
12 shared inverse consumptions
4 owner inverse applications and releases
1 slot and 1 serial generation per owner
4 simultaneously live owners at projection
6 pairwise-distinct owner-slot pairs at projection
0 intermediate relation copies
```

The exact nested lifetime order is:

```text
birth(S) < birth(T) < birth(U) < birth(V) < projection < inverse(ROOT)
         < inverse(V) < release(V)
         < inverse(U) < release(U)
         < inverse(T) < release(T)
         < inverse(S) < release(S)
```

Only `ROOT` is copied to the public boundary. Edge receipts contain public
producer/consumer identifiers, operand side, and structural booleans only;
they do not contain relation coefficients, hashes, ranks, pivots, or
content-derived commitments.

## Restoration and restored-carrier reuse

After copying `ROOT`, the accepted path applies actual dependency-ordered
inverse operations to the actual resident relations. It releases every lease
and checks exact discrete reset of allocator, edge tokens, owner receipts,
edge receipts, pending counts, scheduler queue, and machine counters.
Carrier equality uses the predeclared complex tolerance `2e-12`.

Each accepted transaction has:

```text
serial_after = serial_before + 15
restoration_generation_after = restoration_generation_before + 1
zero outstanding relation leases
all 22 edge tokens reset
all 12 shared edge receipts reset
```

Seventeen alternating transactions consume one actual carrier allocation.
The maximum observed restoration error is `2.00148302124e-16`. Wrong and
missing root inverses each leave error `1.73205080757`. Snapshot reload is
tested only as an explicitly weaker baseline.

## Independent reference and occurrence baseline

The conventional GF(2) reference shares no phase-carrier implementation and
emits only the complete final relation. For every shared edge, a distinct
same-typed replacement is installed in the public operand identifier before
compilation. Each modified graph independently passes alias, cycle,
connectivity, and fanout validation and changes the final primary boundary to
a nontrivial boundary with a different hash. This establishes that all twelve
shared edges are semantically live without enumerating assignments.

The exact occurrence expansion has 51 nodes, 26 leaf occurrences, 22
composition nodes, three intersections, 50 edges, and no fanout. It produces
the same complete boundary and restores by actual inverse. It is a separate
reference-only baseline, not the accepted carrier path.

For relation block `B(w)=4w^2+4w+1` and workspace
`W(w)=36w^2+11w+2`:

```text
accepted retain-all = W(w) + 16B(w) = 100w^2 + 75w + 18
occurrence expansion = W(w) + 52B(w) = 244w^2 + 219w + 54
```

At width sixteen the accepted path uses 26,818 complex cells and 858,176
live carrier bytes. The occurrence baseline uses 66,022 cells and 2,112,704
bytes. The accepted path performs 22 native calls, eight leaf encodes, and 15
lease allocations, versus 50, 52, and 51. These are bounded software
resource counts and phase-carrier compactness relative to this exact
occurrence expansion, not a performance or computational-advantage claim.

## Controls

The qualifier hard-gates:

```text
all 12 independently evaluated semantic shared edges
all 12 ordered same-typed cross-owner handle substitutions
equal-content clone substitution for each owner
stale, skipped, duplicate, and premature inverse for each owner
missing and stale receipt for every shared edge
cross-owner swapped edge receipts
fanout cap 3 against the degree-4 graph
intermediate and custody projection
null carrier
wrong and missing root inverse
snapshot comparison
17-transaction restored-carrier reuse
strict build, analyzer, sanitizers, deterministic replay
expanded file, network, IPC, memory, ptrace, ioctl, and write no-smuggle trace
```

The equal-content clone control uses a control-only sixteen-slot build. The
accepted executable remains capped at fifteen slots and refuses the control
when no bounded spare slot exists.
