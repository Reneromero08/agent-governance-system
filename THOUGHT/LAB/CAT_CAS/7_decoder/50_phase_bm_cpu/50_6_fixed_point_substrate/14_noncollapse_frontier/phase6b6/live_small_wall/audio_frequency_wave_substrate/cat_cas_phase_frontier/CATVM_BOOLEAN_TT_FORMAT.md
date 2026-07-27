# CATVM Width-Parametric Boolean-TT Composition

## Status

This mutable checkpoint establishes:

```text
CATVM_ENFORCED_WIDTH_PARAMETRIC_BOOLEAN_TT_RESIDENT_H_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

within:

```text
BOUNDED_LINUX_SAME_UID_SOFTWARE_WIDTHS4_5_8_12_16_BOOLEAN_TT_RANK2_TO_RANK4_TO_RANK8_ATOMIC_TRANSACTION_REFERENCE_ONLY
```

It moves the CATVM machine law from a fixed nonlinear schema to a
width-growing many-to-many relation family. It does not establish exact TT
rank reduction, fixed-rank unbounded-depth closure, arbitrary graph topology,
computational advantage, Small Wall crossing, physical execution, or
unlimited catalytic computation.

## Enforced boundary

`catvm_boolean_tt_service` privately embeds the reviewed Boolean-semiring
phase-TT engine. The controller contains no phase engine, local relation
generator, reference evaluator, expected boundary hash, witness list, or
answer table.

Before accepting custody, the service:

```text
creates one exact-width carrier
seals the width, block layout, program commitments, and baseline carrier
sets RLIMIT_CORE to zero
sets PR_SET_DUMPABLE to zero and clears PR_SET_PTRACER
sets PR_SET_NO_NEW_PRIVS
locks and marks the context DONTDUMP and DONTFORK
```

It admits one same-real-UID `AF_UNIX/SOCK_SEQPACKET` client, unlinks the
socket path, and installs a default-kill seccomp allowlist. After custody it
cannot open or create files, connect another socket, fork, execute, inspect
another process, use ordinary shared memory, or write stdout or stderr.

This is a same-UID userspace boundary. It does not claim protection from host
root, the kernel, binary replacement, or microarchitectural observation.

## Atomic transaction

The production command surface is:

```text
HELLO
EXECUTE 0
EXECUTE 1
SHUTDOWN
```

All intermediate, bond, carrier, debug, dump, and witness projections receive
one fixed denial. Embedded-NUL, oversized, and unknown packets receive a
fixed protocol error. Fault, snapshot, and inert selectors exist only in a
separately compiled test service.

Inside `EXECUTE`, the service runs:

```text
encode actual rank-two F/G/J leaves
-> compose resident rank-four H from actual F/G
-> compose resident rank-eight Z by reading actual H/J
-> copy and decode only the final Z core block
-> remove the boundary copy
-> apply actual Z inverse
-> apply actual H inverse
-> inverse-encode J/G/F
-> verify canonical carrier and custody state
-> advance restoration generation exactly once
-> return the already-latched final receipt
```

H has zero decoded cells, zero serialized cells, and zero second-block
materializations. Composition necessarily reads H cells into ordinary
operands; those reads are included in the carrier-read law. The controller
receives only variant, generation, plan commitment, final-boundary
commitment, final one-count and cell count, and carrier creation count.

## Relation and phase law

The embedded backend represents:

```text
chi_R(X,Y) = OR_bonds AND_i A_i[left,x_i,y_i,right]
```

and composes cores locally:

```text
C_i[x,z,(a,c),(b,d)]
  = OR_y A_i[x,y,a,b] AND B_i[y,z,c,d]
```

Only the local shared bit `y` is contracted. No width-wide assignment list,
dense `4^w` relation, truth table, candidate set, or witness list is
materialized.

The tested neighbor-AND leaf is nonfunctional because its last output bit is
free. Three leaves yield:

```text
H = F ; G       rank 4
Z = H ; J       rank 8
```

For every `i <= w-3`, the primary Z relation contains
`z_i=x_i*x_(i+1)*x_(i+2)*x_(i+3)`, whose fourth Boolean derivative is one.

## State, restoration, and reuse

Discrete equality is exact over width, layout, block topology, program
commitments, carrier cell count, carrier creation count, state, and
restoration generation. The exact baseline array is compared byte-for-byte.
Complex working-state restoration uses the predeclared `2e-12` tolerance.

Each tested width performs:

```text
primary transaction
unrelated neighbor-NAND transaction on the actual restored carrier
32 alternating reuse transactions
```

One service process creates one carrier and completes 34 transactions.
Every transaction begins from the sealed carrier law and returns a final
receipt only after actual inverse restoration.

## Controls

Separate test builds establish:

```text
wrong boundary inverse       -> restoration detected
missing H inverse            -> restoration detected
reordered H/Z inverse        -> restoration detected
snapshot reload              -> weaker path, generation zero
inert carrier-disabled path  -> transport only, no final result
```

The production controller establishes fixed intermediate-projection denial,
malformed-packet rejection, deterministic replay, and denial of:

```text
/proc/<pid>/mem
/proc/<pid>/maps
/proc/<pid>/fd/0
process_vm_readv
ptrace
pidfd_getfd
```

The no-argument service check is labeled only as malformed startup before
carrier creation; it is not called a null-carrier experiment. The direct
backend separately retains its genuine null-carrier rejection.

## Resource boundary

For width `w`:

```text
N2 = 16w - 16
N4 = 64w - 96
N8 = 256w - 448
carrier = 3N2 + N4 + 2N8 = 624w - 1040 phase cells
```

The accepted per-transaction operation law is:

```text
phase ANDs       4(N4+N8)
phase ORs        2(N4+N8)
carrier reads    8N4+11N8
cell updates     6N2+2N4+4N8
final decodes    N8
```

The evidence counts carrier payload arrays, sealed verification state,
transaction comparison state, boundary bytes, protocol buffers, binaries,
traffic, and phase operations. It compares operation-law scope for warm
direct phase execution, isolated inert transport, snapshot reload, and
accepted in-place inverse execution. It is not comprehensive process-RSS or
performance accounting and supports no speed or total-memory advantage.

## Reproduction

From this directory:

```bash
evidence_dir=$(mktemp -d /tmp/catvm-boolean-tt.XXXXXX)
bash qualify_catvm_boolean_tt_phase.sh "$evidence_dir"
```

Reviewed evidence:

```text
/tmp/catvm-boolean-tt-third
```

## Next obstruction

The machine boundary is no longer the immediate blocker for this relation
family. Exact product ranks still grow `2 -> 4 -> 8 -> ...`; no fixed rank
cap is closed under unbounded composition. The next experiment must test
exact relation-preserving rank reduction or expose a growing-instance Small
Wall obstruction. More widths at the same ranks do not advance the mission.
