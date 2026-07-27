# Rank-2 recursive affine DAG rematerialization format

## Scope

This bounded successor finishes the current automatic compact affine-DAG
scheduler calibration on the exact public fifteen-node topology. It selects
one maximum-rank internal borrower, public node `813`, and proves that its
actual inverse and later reconstruction can depend on lower operators
`809,810` that are themselves dormant and structurally rematerialized.

The implementation embeds the established phase-relational carrier and
one-layer engine. It does not replace the native carrier with a decoded
model, truth table, candidate set, saved relation copy, or host-computed
answer.

## Public rank law

The topology compiler assigns:

```text
rank 0: public bodies and pinned shared owners
rank 1: nonshared degree-one operators with two pinned shared operands
rank 2: nonshared degree-one operators with two rank-one operands
```

It selects the maximum rank and then lowest public ID. On the accepted
topology the rank-two candidates are `813,814`; the bounded single-borrower
law selects `813`. Node `814` remains resident as the sibling anchor. The
root is never eligible.

This policy is coefficient-oblivious. Rank, eligibility, action order,
activation ordinals, receipt ordinals, and resource bounds derive from public
topology, signatures, and recipes only.

## Compiler-expanded tapes

The forward tape has 28 literal actions. Its established 23-action prefix
materializes the graph and evicts the four rank-one nodes. After root `815`
is resident, the compiler appends:

```text
reconstruct 809 generation 1
reconstruct 810 generation 1
actual inverse 813 generation 0
suspend reconstructed 810 generation 1
suspend reconstructed 809 generation 1
```

The projection resident set is exactly:

```text
805,806,807,808,814,815
```

The compiler derives a literal 28-action reverse. Reversal first reconstructs
`809,810` as generation two, reconstructs `813` as generation one, then
suspends those temporary children. Later reverse actions reconstruct
`809,810` as generation three so the actual inverse of rematerialized `813`
can close. Every native forward activation has one actual native inverse.

The maximum nested rematerialization frame depth is two. Counting the pinned
owner frame, the activation chain depth is three:

```text
pinned owner -> rank-one operator -> rank-two operator -> open root custody
```

## Persistent obligations and activation custody

Each evicted public body or operator has one persistent structural
obligation. An obligation binds:

```text
public node and operator identity
nominal signature
public operand identities
sole outgoing edge
public rank
program epoch
plan hash
dead original slot and serial metadata
reconstruction episode count
```

It contains no relation coefficients, relation block, decoded intermediate,
truth table, assignments, witnesses, candidates, or answer-bearing lookup.
Auxiliary reconstruction/suspension episodes do not overwrite or duplicate
the base obligation.

Activation generation is monotonic per node. The four rank-one shared-input
paths reach generation ordinal three. Every physical input use has a
plan-bound receipt containing the exact public edge, ordinal, consumer
activation, forward producer activation, inverse producer activation, and
forward/inverse action IDs.

Receipt policy belongs to the individual activation, not merely the edge:

```text
EXACT
PUBLIC_LEAF_REBIND
INTERNAL_OPERATOR_REBIND
```

Exact receipts require the same producer activation, slot, and serial.
Rebind receipts require the exact compiler authorization for that receipt,
a different declared producer activation, and a fresh actual serial.
Authorization is consumed once. A changed generation or serial alone is not
sufficient.

There remain 22 logical graph edges, each with one public forward/inverse
custody transition. The physical ledger closes 40 activation receipts:

```text
exact receipts                    29
public-leaf rebind receipts        4
internal-operator rebind receipts  7
multi-activation edges            10
second-or-later activations        18
```

The original shared owners remain pinned at activation zero. Their physical
forward/inverse totals are:

```text
805: 8/8
806: 6/6
807: 10/10
808: 4/4
```

## Boundary, restoration, and reuse

Only root `815` is copied to the dedicated boundary block and decoded. The
boundary survives outside inverse history. The copy is inverse-removed, the
literal reverse tape acts on the actual borrowed carrier, all obligations
and receipts close, exact discrete state clears, and an unrelated next
program consumes the actual restored carrier.

Snapshot reload is emitted only as a separately labelled weaker baseline.

## Bounded resources

The tape reaches nine working relation slots, with ten physical relation
blocks including the boundary. Eight slots fail causally at clean pool
exhaustion.

For:

```text
B(w) = 4w^2 + 4w + 1
W(w) = 36w^2 + 11w + 2
```

the accepted carrier law is:

```text
W(w) + 10B(w) = 76w^2 + 51w + 12
```

At width sixteen this is 20,284 complex cells or 649,088 live carrier bytes.
It is slightly larger than the eight-slot one-layer calibration because the
rank-two proof must hold the sibling anchor and actual recursive operands
simultaneously. It remains smaller than the eleven-slot leaf-only,
fifteen-slot retain-all, and 51-slot occurrence-expanded carriers.

Per transaction:

```text
native forward/inverse operations   20/20
leaf forward/inverse encodes          8/8
lease allocations/releases          28/28
```

This is a bounded carrier-law and custody result, not total-memory or
performance advantage.

## Controls and ceiling

Controls reject deep stale consumer and producer activations, swapped
rebind authorization, activation reuse, a missing nested close, a missing
nested child, tape tampering, all intermediate/debug projections, null
carrier, capacity eight, and wrong or missing root inverse. Expanded
file/network/IPC/shared-memory tracing is the mandatory no-smuggle gate.

The claim is:

```text
BOUNDED_15_NODE_RANK2_RECURSIVE_OPERATOR_REMATERIALIZATION_ESTABLISHED
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_SINGLE_RANK2_BORROWER_28_FORWARD_28_REVERSE_ACTIONS_9_WORKING_SLOTS_10_PHYSICAL_BLOCKS_MAX_ACTIVATION_ORDINAL3_REFERENCE_ONLY
```

It does not establish both rank-two branches, automatic general-DAG
pebbling, arbitrary topology, unbounded activation depth, non-affine
relations, CATVM enforcement for this scheduler, total-memory or performance
advantage, physical execution, Small Wall crossing, or unlimited catalytic
computation.

Per the durable lane guardrail, this finishes the affine scheduler
calibration needed for the next machine-boundary experiment. The next
frontier is CATVM enforcement of this automatically scheduled shared
relational DAG, not a larger affine fixture.
