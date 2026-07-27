# Compact Compiled-Body Typed Relation Modules

## Status

This mutable checkpoint establishes:

```text
COMPACT_COMPILED_BODY_TYPED_RELATIONAL_MODULE_EXECUTION
```

for the bounded tested Boolean/F3 module trees. It removes per-instance
native-operation inlining. It does not make carrier history compact and does
not establish wide-interface or general recursive relational computation.

## Execution form

The public source remains `CATCAS_TYPED_RELATION_MODULE 1`. The compact
executor changes how admitted leaf modules run:

```text
one parsed native leaf body
-> many nominally typed module descriptors
-> one transient relocated operation descriptor
-> distinct phase-resident operation messages
-> parent composition from actual child final_start addresses
```

Leaf definitions with the same local path share one parsed `struct process`.
An instance retains its nominal domains, input/message offsets, and final
phase address, but no copied native-operation array. For each forward or
inverse step, the executor reconstructs one transient operation from the
shared definition and relocates its operand and output addresses into that
instance's carrier region.

Composite modules retain typed child references, not native-operation
records. Their native composition descriptor is likewise reconstructed
transiently from the two actual child `final_start` addresses.

## Resident state and reversal

Every executed operation still has a distinct four-cell phase-resident
message. This is necessary for reversing the actual forward history:

```text
encode public leaf relations
-> execute shared leaf bodies into resident instance messages
-> compose actual resident child outputs
-> copy and decode only the root relation
-> remove the public boundary
-> reconstruct and inverse parent operations before children
-> inverse public leaf encodings
-> verify and reuse the actual restored carrier
```

The saved carrier is comparison-only. It is never loaded into the accepted
path. A different source with identical typed geometry executes immediately
after restoration on the same carrier allocation.

`module_export_copy_cells = 0` excludes a persistent cross-module export
artifact. Native phase operators still use bounded transient polynomial
arrays.

## Demonstrated recursive program

The depth-four program instantiates the same four-operation leaf body eight
times and combines the outputs through seven typed composite modules:

```text
module descriptors                              15
leaf instances                                   8
composite modules                                7
unique leaf sources                              1
persistent per-instance native operation records 0
persistent shared leaf operation records          4
persistent composite module descriptors           7
simultaneous transient operation records           1
executed native operations                       39
phase-resident operation messages                39
input relations                                  40
carrier cells                                   320
live carrier complex values                     640
live carrier bytes                            10,240
carrier plus comparison snapshot values        1,280
maximum native-operator workspace values          52
root inverse/control temporary values               8
logical phase-execution peak values             1,340
coexisting typed-source storage bytes          16,464
coexisting leaf-definition storage bytes       41,248
coexisting layout storage bytes                   112
```

The expanded typed backend retains 39 native-operation records for the same
program. Both backends produce the same primary `[0,2,2,0]` and reuse
`[0,1,1,1]` boundaries.

## Controls

Qualification requires:

```text
nominal type mismatch rejection
non-root projection rejection
wrong boundary inverse discrimination
omitted parent inverse discrimination
leaf-intersection bypass discrimination
ordinary-sum intersection discrimination on the shallow parity case
expanded typed-backend parity at depth four
reviewed flat series-parallel parity on the shallow case
fresh-process byte determinism
strict GCC compilation and static analysis
ASan, UBSan, and leak detection at both depths
source and fixture hashes
focused independent review
```

Wrong boundary inversion and omitted parent inversion each leave restoration
error `1.73205080757`. Correct primary and cross-program reuse restore within
`2.00148302124e-16`, below the predeclared `2e-12` tolerance.

The accepted transaction retains one four-complex-value root boundary factor
until that boundary is removed. Per-operation inverse factors are recomputed
from resident operands; they are not retained as history. The resource ledger
counts the root factor, its four-value wrong-inverse control rotation, maximum
native polynomial workspace, both concurrently loaded typed sources and leaf
definitions, both layouts, the carrier, and the comparison-only snapshot.

## Claim boundary

The demonstrated compactness is compiled-body and instruction-descriptor
reuse:

```text
persistent native leaf records: expanded 32 -> compact 4
persistent per-instance native records:       compact 0
```

It is not constant-space recursion. The carrier retains 39 distinct operation
messages, and its 320 cells scale with instantiated inputs and history. The
host still traverses the public module tree and reconstructs transient
descriptors. The result does not establish compact phase history, interfaces
wider than two Boolean ports, arbitrary topology or arity, computational
advantage, physical phase execution, Small Wall crossing, or unlimited
catalytic computation.

The next blocker is a native unresolved separator wider than one Boolean
variable, followed by a representation whose carrier growth is explicit in
separator width rather than hidden assignment expansion.

## Reproduction

```bash
evidence_dir=$(mktemp -d /tmp/compact-module-qual.XXXXXX)
./qualify_algebraic_compact_module_phase.sh . "$evidence_dir" \
    >"$evidence_dir.summary.json"
```
