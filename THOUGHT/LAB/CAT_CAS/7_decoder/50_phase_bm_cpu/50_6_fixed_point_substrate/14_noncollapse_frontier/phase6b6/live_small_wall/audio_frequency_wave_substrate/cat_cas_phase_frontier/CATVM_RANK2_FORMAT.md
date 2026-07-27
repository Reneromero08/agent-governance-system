# CATVM Rank-Two Automatically Scheduled Shared DAG

## Status

This mutable checkpoint establishes:

```text
CATVM_ENFORCED_15_NODE_RANK2_AUTOMATIC_SCHEDULED_SHARED_RELATIONAL_DAG_ESTABLISHED_ON_PHASE_BACKEND
```

within the strict ceiling:

```text
BOUNDED_LINUX_SAME_UID_SOFTWARE_GF2_WIDTH3_EXACT_15_NODE_ATOMIC_RUN_28_FORWARD_28_REVERSE_9_WORKING_SLOTS_REFERENCE_ONLY
```

This is a machine-boundary result for the established affine calibration
class. It does not establish non-affine relational closure, a general CATVM,
computational advantage, Small Wall crossing, physical waveform execution, or
unlimited catalytic computation.

## Enforced machine boundary

`catvm_rank2_service` is the only process linked to the reviewed rank-two
scheduler. The production link privately embeds that exact branch-native
implementation and garbage-collects its standalone entry point and reporting
functions. The controller has no phase header, phase symbol, relation
evaluator, expected boundary, witness table, or answer-bearing lookup.

Before carrier creation the service:

```text
sets RLIMIT_CORE to zero
sets PR_SET_DUMPABLE to zero
clears PR_SET_PTRACER
sets PR_SET_NO_NEW_PRIVS
```

The context, carrier baseline, and actual working carrier are private,
locked, `MADV_DONTDUMP`, and `MADV_DONTFORK`. Future verification allocations
are locked on the accepted build. After accepting one same-UID client the
service unlinks its `AF_UNIX/SOCK_SEQPACKET` endpoint and installs a seccomp
default-kill allowlist. It cannot open files, create another socket, fork,
execute, inspect another process, use shared memory, or emit ordinary output
after custody begins.

The boundary is tested against the ordinary same-UID controller. It does not
claim secrecy from host root, the kernel, binary replacement, or
microarchitectural observation.

## Atomic transaction

The strict production protocol is:

```text
HELLO
EXECUTE <public variant 0..4>
SHUTDOWN
```

Every projection, tape, obligation, node-generation, activation-receipt,
debug, dump, carrier-read, and state-detail request receives only:

```text
ERR E_INTERMEDIATE_PROJECTION_DENIED
```

Embedded NUL, oversized, and unknown packets receive only:

```text
ERR E_PROTOCOL
```

`EXECUTE` selects one of five public program variants compiled inside the
service. It cannot supply a carrier pointer, coefficient buffer, schedule,
inverse mode, fault mode, or debug selector.

The service then performs one indivisible transaction on the same actual
carrier:

```text
compile-bound 28-action forward tape
-> unresolved phase-resident DAG relations
-> root-only boundary latch
-> remove the actual boundary factor
-> literal 28-action reverse tape
-> actual inverse closure of receipts, nodes, obligations, and allocator
-> exact discrete-state verification
-> complex carrier restoration verification
-> final response
```

The final boundary is latched before reversal but no response is constructed
or sent until the actual inverse and restoration law pass. A client
disconnect cannot cancel the inverse because request handling is synchronous.

The response contains only:

```text
public variant
restoration generation
public plan commitment
final boundary commitment
carrier creation count
49 final F3 boundary coefficients
```

It contains no internal relation coefficient, phase cell, intermediate hash,
receipt record, tape action, slot or serial, node generation, obligation,
witness, candidate set, assignment expansion, or truth table.

## Restoration and reuse law

The accepted transaction reuses the reviewed exact scheduler law:

```text
28 forward actions
28 literal reverse actions
9 peak working relation slots
6 live relations at projection
40 forward and 40 inverse physical edge activations
20 forward and 20 inverse native operators
8 forward and 8 inverse leaf encodes
28 allocations and 28 releases
0 intermediate relation copies
```

The service rejects a transaction unless:

```text
all activation receipts close
all node generations close
all reconstruction obligations clear
logical custody restores
the allocator restores
the workspace clears
the actual carrier is within 2e-12 of its pretransaction state
restoration generation advances by exactly one
carrier creation count remains exactly one
snapshot_loaded is false
```

The accepted evidence executes a primary program, an unrelated second
program, and 256 alternating reuse transactions in one service process on one
actual carrier. The second and later transactions consume the carrier
restored by the preceding actual inverse.

## Controls

Production exposes no restore or fault selector. Separately compiled
test-only services establish:

```text
wrong root inverse -> restoration detected
missing root inverse -> restoration detected
dependency-reordered reverse tape -> seccomp fail-closed before payload
snapshot reload -> separate weaker path, generation remains zero
inert boundary -> transport-only comparison
```

The production controller additionally establishes fixed denials for all
internal projection surfaces and denial of:

```text
/proc/<pid>/mem
/proc/<pid>/maps
/proc/<pid>/fd/0
process_vm_readv
ptrace
pidfd_getfd
```

The qualifier requires byte-zero service stdout and stderr, exact controller
output-key allowlists, controller/core symbol separation, discarded
standalone reporting code, direct/reference boundary parity, sanitizer and
analyzer passes, deterministic replay, and a traced post-custody syscall
allowlist.

## Resource accounting

At width three:

```text
service context                         17,888 bytes
compiled topology                       2,664 bytes
activation plan                        13,192 bytes
public program table                    1,960 bytes
machine counters                           24 bytes
execution summary                       1,400 bytes
actual carrier                            849 complex cells / 27,168 bytes
verification baseline                     849 complex cells / 27,168 bytes
request / response buffers                128 / 1,024 bytes
production service binary              86,248 bytes
controller binary                      21,576 bytes
```

The 258-transaction accepted evidence accounts for 273 request and response
packets, 2,703 request bytes, and 39,339 response bytes.

One uncontrolled 1,000-transaction warm comparison measured:

```text
direct process phase          37,014,777,483 ns
isolated inert boundary           14,747,358 ns
snapshot CATVM               18,286,765,309 ns
accepted in-place CATVM      36,957,473,181 ns
```

These timings establish no performance advantage. The comparison includes
the same scheduler's actual work rather than a cold-start-only claim.

## Reproduction

From this directory:

```bash
evidence_dir=$(mktemp -d /tmp/catvm-rank2.XXXXXX)
bash qualify_catvm_rank2_phase.sh "$evidence_dir"
```

The recorded clean evidence is:

```text
/tmp/catvm-rank2-full-fifth
```

## Scientific boundary

The service enforces hidden intermediates, topology-compiled typed
composition, root-only projection, actual restoration, and actual
restored-carrier reuse for one exact width-three GF(2)-affine rank-two DAG.
The next high-value experiment must leave this calibration class or use the
new boundary in a controlled Small Wall experiment; larger affine fixtures
alone do not advance the main mission.
