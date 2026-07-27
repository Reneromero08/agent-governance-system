# Automatic public-leaf pebbling for the four-owner affine DAG

## Scope

This mutable experiment compiles and executes a reversible compact schedule
for the established fifteen-node affine DAG. The compiler selects every
public degree-one leaf whose sole forward edge has completed, emits an
explicit nineteen-step tape, inverse-encodes that leaf early, and reconstructs
it from its public body only when the literal reverse tape requires it.

The four native shared owners `805,806,807,808` remain pinned. All internal
operator results retain their original resident generations. This is therefore
automatic public-leaf-only pebbling, not automatic general DAG pebbling or
internal operator rematerialization.

## Compiler-emitted reversible tape

The plan is derived from the validated public topology and compiled schedule.
Its opcodes are:

```text
ENCODE_LEAF
APPLY_OPERATOR
EVICT_LEAF
```

For this graph it emits nineteen forward steps and selects leaves
`801,802,803,804`. Reverse execution consumes the same tape literally. The
cursor is `19` at projection and `0` after restoration. The plan is
coefficient-oblivious and binds:

```text
topology hash       41d917d4a3308fbe
schedule hash       aa5719d149bc55e0
plan hash           627f298bb1d2c4e8
public node identifier per step
opcode, node, edge, and expected live count per step
```

Tape tampering and a ten-slot build fail causally. The accepted schedule's
predicted and observed peak is eleven working relation slots; the ten-slot
control exhausts the clean relation-slot pool.

## Reconstruction obligations and edge custody

Early leaf release creates a structural obligation binding the public leaf
body, nominal relation signature, sole pending edge, public-program epoch,
and exact plan hash. It retains no relation coefficients, candidate set,
truth table, witness list, or decoded relation.

The twenty-two graph edges are divided into:

```text
18 EXACT_RESIDENT_GENERATION edges
 4 PUBLIC_LEAF_RECONSTRUCTION edges
```

Every edge has observed forward and inverse consumption. Exact resident edges
must retain slot and serial generation. Reconstructed public-leaf edges must
consume the correctly rebound obligation in the required reverse-tape
position. The twelve shared-owner edges remain exact generation matches; all
four owners are simultaneously live and pairwise nonaliased at projection.

The accepted path performs zero internal-node rematerializations and zero
operator recomputations.

## Restoration and reuse

Only `ROOT` is copied to the public boundary. Reverse tape execution
reconstructs the four public leaves, applies every actual inverse to the
actual borrowed carrier, clears every obligation and residency entry, and
returns the tape cursor to zero.

Seventeen alternating transactions consume one actual carrier allocation.
The maximum observed restoration error is `7.02715047328e-16`, below the
predeclared `2e-12` tolerance. Wrong and missing root inverses remain
detectable. Snapshot reload is separately identified as a weaker baseline
with no accepted post-projection inverse operations.

## Resource law

With relation block `B(w)=4w^2+4w+1` and workspace
`W(w)=36w^2+11w+2`:

```text
automatic leaf pebbling = W(w) + 12B(w) = 84w^2 + 59w + 14
retain-all              = W(w) + 16B(w) = 100w^2 + 75w + 18
occurrence expansion    = W(w) + 52B(w) = 244w^2 + 219w + 54
```

At width sixteen the accepted carrier has 22,462 complex cells and 718,784
live bytes. Retain-all has 26,818 cells; the occurrence expansion has 66,022.
The accepted path performs 22 native operator calls, 16 leaf encodes, and 19
lease allocations, versus retain-all's `22,8,15` and the occurrence path's
`50,52,51`.

The automatic layer also reports its exact current-ABI plan, obligation,
edge-receipt, residency, node-lease, execution-summary, and machine-state
sizes. GCC `-fstack-usage` at the accepted optimized build reports a
47,936-byte `main` frame and 13,664-byte concurrent `ac_execute` frame, a
61,600-byte accepted call-chain floor. The sum of all 89 compiled function
frames, 273,224 bytes, is retained as a conservative upper bound. These
figures prevent carrier-only accounting from hiding scheduler or temporary
inverse state; they are not combined into a total-memory-advantage claim.

## Controls and ceiling

The qualifier rejects shared or internal eviction, wrong body, stale epoch,
missing/double/skipped reconstruction, reordered shared inverse, stale
internal edge generation, tape tampering, every attempted intermediate
projection, null carrier, wrong/missing root inverse, and capacity ten. It
also checks deterministic replay, analyzer, ASAN/UBSAN/leaks, widths
`3,4,8,12,16`, twenty-five complete boundaries against the independent GF(2)
reference, retain-all and occurrence baselines, and an expanded no-smuggle
system-call trace.

This establishes:

```text
BOUNDED_15_NODE_FOUR_OWNER_AUTOMATIC_PUBLIC_LEAF_PEBBLING_ESTABLISHED
```

with ceiling:

```text
BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_DAG_COMPILER_EMITTED_19_STEP_REVERSIBLE_PUBLIC_DEGREE_ONE_LEAF_ONLY_PEBBLING_11_WORKING_SLOTS_12_PHYSICAL_RELATION_BLOCKS_FOUR_PINNED_SHARED_OWNERS_REFERENCE_ONLY
```

It does not establish automatic general DAG pebbling, internal operator
rematerialization, a global eleven-slot optimum outside the declared
public-leaf-only planner, arbitrary graph topology, CATVM enforcement for
this compiler, total-memory or performance advantage, physical execution,
Small Wall crossing, or unlimited catalytic computation.
