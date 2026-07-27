# Typed Relational Phase Modules

## Status

This mutable checkpoint establishes:

```text
BOUNDED_RECURSIVE_TYPED_RELATIONAL_MODULE_COMPOSITION
```

It composes independently compiled public series-parallel Boolean/F3 modules
through nominally typed, phase-resident outputs. It does not establish compact
recursive execution.

## Source format

```text
CATCAS_TYPED_RELATION_MODULE 1
DOMAIN NAME BOOLEAN_F3
LEAF NAME LEFT_DOMAIN RIGHT_DOMAIN LEAF_FILE.aspr
COMPOSE NAME LEFT_MODULE RIGHT_MODULE
PROJECT ROOT_MODULE
END
```

Domains are nominal: two differently named domains remain incompatible even
when both use `BOOLEAN_F3`. Leaves must precede composites. Composite
declarations are topological, and:

```text
LEFT_MODULE.right_domain == RIGHT_MODULE.left_domain
```

is required exactly. A module cannot compose with itself. Every non-root
module has exactly one owning parent, and only a composite root may be
projected.

Leaf paths are strict local filenames. Each leaf is a valid
`CATCAS_SERIES_PARALLEL_RELATION 1` source and inherits that format's parser,
topology-only compilation, exact phase-native composition and intersection,
and structural gates.

## Resident handoff

A leaf export is:

```text
left nominal domain
right nominal domain
final_start phase address
orientation
```

A composite appends one native `OP_COMPOSE` whose two operand addresses are
exactly its child descriptors' `final_start` addresses. It creates no
persistent export cell, coefficient copy, hash, serialized relation, witness,
candidate set, or truth table.

The native operator necessarily uses bounded transient operand arrays while
evaluating its polynomial. Therefore:

```text
module_export_copy_cells = 0
```

means no persistent or cross-module export artifact. It does not mean zero
temporary operator storage.

Only the root relation is copied to the four-cell public boundary and decoded.
Reverse execution removes that boundary, recomputes and inverses the parent
operations before their child operations, then reverses every public input.
The saved carrier used to measure restoration is comparison-only and is never
loaded into the accepted path.

## Demonstrated programs

The semantic parity case splits the reviewed nested two-cycle graph into:

```text
LEFT:  A -> V
RIGHT: V -> B
ROOT = COMPOSE(LEFT, RIGHT)
```

It has three module descriptors, nine native operations, 80 carrier cells,
and matches the flattened reviewed series-parallel engine exactly:

```text
primary boundary [0,2,1,2]
reuse boundary   [0,1,1,1]
```

The recursive case instantiates one leaf source four times in a balanced
binary module tree:

```text
ROOT(
    PAIR_01(LEAF_0, LEAF_1),
    PAIR_23(LEAF_2, LEAF_3)
)
```

It demonstrates:

```text
nominal domains                    5
module descriptors                 7
module-tree depth                  3
unique leaf sources                1
compiled leaf definition reuse     true
native operation descriptors       19
resident operation messages        19
carrier cells                      160
live carrier complex values        320
live carrier bytes                 5120
verification peak complex values   640
```

All child-to-parent handoffs are actual resident addresses. The primary root
is `[0,2,2,0]`; a different program on the actual restored carrier produces
`[0,1,1,1]`.

## Controls

Qualification requires:

```text
nominal type mismatch rejection
non-root projection rejection
wrong boundary inverse discrimination
omitted resident-module inverse discrimination
leaf-intersection bypass discrimination
ordinary-sum intersection discrimination on the parity case
flattened reviewed-engine parity
fresh-process byte determinism
strict GCC compilation and static analysis
ASan, UBSan, and leak detection
source and fixture hashes
focused independent review
```

Wrong and omitted inverse controls each leave restoration error
`1.73205080757`. Correct recursive primary and cross-program reuse restore
within `2.00148302124e-16`, below the predeclared `2e-12` tolerance.

## Resource and claim boundary

The compiler caches the repeated leaf definition, but each instantiation is
still expanded into a native operation descriptor range and distinct
phase-resident messages. The demonstrated recursive program reports:

```text
compact_definition_reuse              false
expanded_native_operation_descriptors 19
native_operation_descriptor_bytes     912
compiled typed-source storage bytes   8232
compiled leaf-definition bytes        20624
```

This is bounded typed recursive composition with direct phase-resident
handoff. It is not compact recursive closure, a wide-interface relation,
arbitrary graph topology, arbitrary arity or domain, advantage, physical
phase execution, Small Wall crossing, or unlimited catalytic computation.

The next blocker is an interpreter or other lawful execution form that reuses
compiled module bodies without operation inlining, followed by unresolved
interfaces wider than two Boolean ports.

## Reproduction

```bash
evidence_dir=$(mktemp -d /tmp/typed-module-qual.XXXXXX)
./qualify_algebraic_typed_module_phase.sh . "$evidence_dir" \
    >"$evidence_dir.summary.json"
```
