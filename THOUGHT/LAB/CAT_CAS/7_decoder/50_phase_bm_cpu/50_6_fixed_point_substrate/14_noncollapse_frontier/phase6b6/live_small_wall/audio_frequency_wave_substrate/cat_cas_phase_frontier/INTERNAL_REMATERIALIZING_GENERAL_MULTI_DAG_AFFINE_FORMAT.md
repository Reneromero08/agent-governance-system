# One-layer internal rematerialization with activation custody

## Scope

This mutable experiment extends the established fifteen-node affine DAG
compiler from public-leaf-only reconstruction to actual native
rematerialization of four internal operators. The topology-only eligibility
law selects nonleaf, nonshared, degree-one nodes whose two operands are pinned
shared owners and whose sole consumer is not the root. On this graph those
nodes are:

```text
809 P = COMPOSE(805 S, 807 U)
810 Q = COMPOSE(806 T, 807 U)
811 R = COMPOSE(805 S, 808 V)
812 K = COMPOSE(807 U, 808 V)
```

The four shared owners `805,806,807,808` remain their original resident
generations. The retained joins `813,814` and root `815` are not
rematerialized.

This is an exact one-layer bounded planner law, not recursive or general DAG
pebbling.

## Reversible tape

The compiler emits twenty-three coefficient-oblivious forward steps from the
validated public topology and schedule. It adds `EVICT_OPERATOR` after the
sole consumer of each eligible operator has executed. Literal reverse-tape
execution maps that opcode to native operator reconstruction before the
consumer inverse.

The plan binds:

```text
topology hash  41d917d4a3308fbe
schedule hash  aa5719d149bc55e0
plan hash      f0345b7ae7bfe27d
public node ID, opcode, node, edge, expected live count per step
```

Predicted and observed peak residency are eight working relation slots. The
projection live set has seven values: four pinned shared owners, joins
`813,814`, and root `815`. A seven-slot build fails causally at clean-pool
exhaustion under this exact pinned one-layer schedule.

## Actual early inverse and recomputation

After `813` is materialized, actual native inverses clear and release
`810,809`. After `814`, actual native inverses clear and release `812,811`.
Their obligations retain only node/operator identity, nominal signature,
public operand identities, sole outgoing edge, public-program epoch, plan
hash, and dead generation metadata. They retain no relation coefficients,
copied carrier block, candidate set, truth table, witness, or decoded
intermediate.

Reverse execution allocates new generations and reruns the actual native
operator directly from the still-resident shared operands. Those new
generations feed the actual inverses of `813` and `814`, after which their own
actual inverse operations clear the recomputed values.

Per accepted transaction:

```text
native forward operations                    15
native inverse operations                    15
internal early inverses                       4
internal native recomputations                 4
leaf forward/inverse encodes                8/8
lease allocations/releases                23/23
```

## Logical edges and physical activations

The backend retains one logical custody transition for each of the twenty-two
public edges. A separate activation ledger accounts for every physical use.
The eight input edges of nodes `809-812` have two activation epochs; the
other fourteen edges have one.

```text
public logical edges                         22
physical activation receipts                 30
exact producer-generation activation pairs   22
public-leaf generation rebindings              4
internal-operator generation rebindings        4
multi-activation edges                         8
```

Each activation follows:

```text
UNUSED -> FORWARD_OPEN -> INVERSE_CLOSED
```

The receipt binds consumer activation, producer activation, and the actual
slot/serial generation. A new activation cannot open while its predecessor is
open. Exact policies require identical forward/inverse producer activation
and generation. Leaf and internal reconstruction policies require the
specific structural obligation and a new producer activation and serial.

The four shared owners never change producer activation or generation.
Their physical forward/inverse activation totals by public ID are:

```text
805: 6/6
806: 4/4
807: 6/6
808: 4/4
```

The causal stale-consumer control is stronger than a generation-only test:
it presents activation zero while the second activation is open on
`805->809`; the shared slot and serial are deliberately still identical, and
the activation FSM rejects the stale epoch.

## Boundary, restoration, and resources

Only root `815` enters the dedicated boundary block. The copied boundary is
decoded once, inverse-removed, and retained outside inverse history. Every
actual inverse then runs on the borrowed carrier. Seventeen alternating
transactions consume the same carrier allocation and restore with maximum
error `6.66133814775e-16` under the predeclared `2e-12` tolerance.

With `B(w)=4w^2+4w+1` and `W(w)=36w^2+11w+2`:

```text
internal rematerialization = W(w) + 9B(w)  = 72w^2 + 47w + 11
public-leaf-only           = W(w) + 12B(w) = 84w^2 + 59w + 14
retain-all                 = W(w) + 16B(w) = 100w^2 + 75w + 18
occurrence expansion       = W(w) + 52B(w) = 244w^2 + 219w + 54
```

At width sixteen the accepted carrier has 19,195 complex cells and 614,240
live bytes. The activation ledger is 5,280 current-ABI bytes, the plan 1,416,
obligations 1,560, and execution summary 6,992 at width sixteen. The
optimized compiler reports a 33,600-byte `main` frame and 17,472-byte
concurrent `ir_execute` frame, a 51,072-byte accepted stack floor; 325,448
bytes is the conservative sum of all 105 compiled function frames. These
counts prevent hidden scheduler/inverse state from disappearing behind the
carrier comparison. They do not establish total-memory or performance
advantage.

## Controls and ceiling

The qualifier rejects stale consumer activation, a missing first-epoch
close, stale reconstructed producer activation, missing/double/stale operator
reconstruction, noneligible eviction, tape tampering, capacity seven, all
intermediate projection requests, and null carrier. Wrong and missing root
inverse remain detectable; snapshot is a separate weaker branch. Strict,
analyzer, ASAN/UBSAN/leak, deterministic replay, expanded no-smuggle trace,
widths `3,4,8,12,16`, twenty-five independent GF(2) complete-boundary
comparisons, and the reviewed leaf-only predecessor regression pass.

This establishes:

```text
BOUNDED_15_NODE_FOUR_OWNER_INTERNAL_OPERATOR_REMATERIALIZATION_ESTABLISHED
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_ONE_LAYER_FOUR_INTERNAL_OPERATOR_REMATERIALIZATION_23_STEP_REVERSIBLE_TAPE_8_WORKING_SLOTS_9_PHYSICAL_BLOCKS_MULTI_ACTIVATION_EDGE_CUSTODY_REFERENCE_ONLY
```

It does not establish recursive internal rematerialization, automatic general
DAG pebbling, arbitrary activation depth, arbitrary graph topology, a global
pebbling optimum, CATVM enforcement for this compiler, total-memory or
performance advantage, physical execution, Small Wall crossing, or unlimited
catalytic computation.
