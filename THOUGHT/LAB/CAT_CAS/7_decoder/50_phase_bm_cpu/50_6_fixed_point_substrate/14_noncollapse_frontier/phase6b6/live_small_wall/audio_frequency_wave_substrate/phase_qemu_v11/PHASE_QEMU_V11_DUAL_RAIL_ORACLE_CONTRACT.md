# Phase-QEMU V11 dual-rail oracle backend contract

## Scope

V11 is the reintegration gate after the standalone Phase-QEMU mechanism-search
packages.  It is an actual QEMU PCI device and a common guest-visible control
plane with swappable backend implementations.  It is not a claim that QEMU,
the exact ideal backend, or the directory name is a physical phase resource.

The canonical bounded claim is:

```text
COMMON_PHASE_QEMU_V11_SWAPPABLE_BACKEND_DEVICE_EXECUTES_TWO_LATE_BOUND_DUAL_RAIL_NUMBER_EIGENSPACE_DISPERSIVE_KICKBACK_QUERIES_ON_ONE_HIDDEN_CARRIER_REFERENCE_STATE_AND_RELEASES_ATOMIC_RECEIPTS_ONLY_AFTER_EXACT_COMPLETE_RETURN_WHILE_OPEN_AND_EXTERNAL_BACKENDS_FAIL_CLOSED
```

Claim ceiling:

```text
COMPILED_QEMU_10_2_4_PCI_DEVICE_WITH_EXACT_QOMEGA_IDEAL_BACKEND_TEST_ONLY_PRIVATE_PROVIDER_AND_ANALYTIC_OPEN_EXTERNAL_STUBS_NO_PHYSICAL_ORACLE_COHERENT_PORT_CUSTODY_QUERY_SEPARATION_OR_ADVANTAGE
```

Restoration classification:

```text
EXACT_ALGEBRAIC_RESTORATION
```

Restoration scope:

```text
EXACT_HIDDEN_96_DIMENSION_QOMEGA_DUAL_RAIL_CARRIER_REFERENCE_AND_TWO_CLIENT_REFERENCE_RETURN_ON_ONE_RESIDENT_QEMU_ALLOCATION_FOR_IDEAL_BACKEND_ONLY_WITH_NO_PHYSICAL_SAME_MODE_CUSTODY
```

Disposition:

```text
COMMON_DEVICE_REINTEGRATION_ESTABLISHES_BACKEND_NEUTRAL_LIFECYCLE_AND_EXACT_IDEAL_MACHINE_LAW_BUT_TEST_ONLY_PRIVATE_BINDING_AND_EQUAL_ACCESS_DIRECT_PHASE_SHADOW_PRECLUDE_PHYSICAL_OR_RESOURCE_PROMOTION
```

The next physical-integration target is:

```text
AUTHENTICATED_EXTERNAL_DUAL_RAIL_DISPERSIVE_ORACLE_ADAPTER_WITH_COHERENT_CLIENT_PORT_REFERENCE_PRESERVATION_LATE_BOUND_PRIVATE_CONTROL_AND_TOTAL_RESOURCE_CERTIFICATION
```

## Architectural discipline

Phase-QEMU has two deliberately different layers.

1. Standalone mechanism-search/digital-twin packages may cheaply falsify a
   carrier law, restoration proposal, or resource claim.
2. A mechanism is eligible for common-machine promotion only after it is
   implemented behind the common Phase-QEMU guest/device/backend contract.

V0 and V1 are compiled devices.  V2 through V10 are bounded mechanism-search
packages and do not become devices by living in a `phase_qemu_*` directory.
V11 is the first explicit reintegration checkpoint.  A successful V11 result
may promote only a compiled formal machine law.  Physical qualification still
requires an external adapter, coherent ports, custody evidence, calibrated
resources, and a separate experiment.

## Device identity and backend boundary

The QOM type is `phase-qemu-v11`.  Its PCI identity is `1234:11fb`, revision
one.  BAR0 is 4096 bytes.  Magic is `PH11` (`0x50483131`) and ABI is
`0x00010000`.  V1 remains independently buildable and unchanged.

The immutable backend identifier selects one implementation behind one ABI:

```text
0x0b01  IDEAL_DUAL_RAIL
0x0b02  OPEN_DUAL_RAIL
0x0b80  EXTERNAL_ADAPTER
```

The device core, not a backend, owns tags, lifecycle, descriptor sealing,
boundary locking, response commit, custody receipts, resource sealing, and
snapshot/migration lineage.  A backend cannot directly set `RESPONSE_READY`.
The core holds `LIFE_EXECUTING` through the backend `execute` callback, then
enters `LIFE_VERIFYING_RETURN` before the distinct `verify_return` callback.

The ideal backend executes exact hidden carrier dynamics.  The present open
backend is only a zero-noise parity path plus a nonzero-parameter
`APPROX_MODEL` classification stub.  It makes no Kraus, Lindblad, physical
noise, or exact noisy-return claim.  The external adapter is unavailable and
must fail through the backend `lease_preflight` callback without lifecycle
mutation.  Core code does not special-case its identifier.  These stubs prove
the common dispatch and failure boundary; they are not promoted science.

## Hidden carrier and exact field

The state space has dimensions

```text
C_A, R_A, C_B, R_B, K, R_K = 2, 2, 2, 2, 3, 2
```

and total dimension 96.  Carrier basis `K` is `vac`, rail `a`, rail `b`.
The secret-independent carrier/reference state is

```text
(|a,0> + |b,1>) / sqrt(2).
```

Both clients are fresh Bell-Choi pairs.  The complete initial state therefore
has eight equal components and density denominator eight.

Exact arithmetic uses

```text
Q(omega),  omega^2 + omega + 1 = 0.
```

The backend stores a full hidden `96 x 96` density matrix as coefficient pairs
of `1` and `omega` with one checked common power-of-two denominator.  It also
owns an independent full scratch matrix.  No amplitude, coefficient,
denominator, phase vector, private residue, expected output, inverse, or state
hash is readable through BAR0.

For query slot `j` and late-bound residue `s_j in Z3`, the actual hidden
matrix is transformed by

```text
Q_s = exp(+i 2*pi*s/3 |1><1|_C tensor N_K),
N_K(vac)=0, N_K(a)=N_K(b)=1.
```

Each density cell is multiplied by

```text
omega ** (s * (c_ket*N_ket - c_bra*N_bra)).
```

This executes the declared client-carrier law on the resident density.  It is
not a direct write of the expected client diagonal.  The accepted dual-rail
support makes the final action factorize as `diag(1, omega^s)` on the client
while returning the full carrier/reference joint state.

## Public descriptor and private binding

The accepted public descriptor is eight fixed 32-bit words:

```text
D0  0x50313144                       P11D
D1  schema 1 and word count 8
D2  TWO_PHASE_QUERY
D3  two queries, phase order three
D4  MODEL_TWO_BELL_CHOIS
D5  DUAL_RAIL_N1_WITH_REFERENCE
D6  COMPLETE_RETURN | HOLD_BOTH_ATOMIC
D7  resource-envelope and boundary schema
```

Reserved bits are zero.  Every ordered secret pair uses byte-identical guest
writes.  A residue, phase table, expected boundary, inverse, secret-derived
commitment, or secret-derived timing choice in this descriptor is
`SECRET_SMUGGLE`.

The test backend has a host-only provider surface.  It is enabled explicitly
for qtest and binds `test-private-a` and `test-private-b` through runtime QOM
properties only after `ARM_PRIVATE`.  Each slot is write-once and bound to the
current generation and arm nonce.  These properties are absent from BAR0,
VMState, receipts, and the public descriptor.  They are test infrastructure,
not an authenticated physical provider.  Any access-restriction or physical
oracle claim remains false until an external adapter replaces them.

Read-only test observer properties may inspect the held port phases and
backing counters.  They are privileged qualification surfaces, never guest
outputs, and must be disabled in a service configuration.

## Lifecycle and atomic response law

The mandatory order is:

```text
LEASE
-> PREPARE carrier/reference once
-> ISOLATE_SOURCE
-> upload and SEAL_DESCRIPTOR
-> ARM_PRIVATE
-> host-only bind A and B
-> EXECUTE_ATOMIC
-> hold both client/reference outputs
-> VERIFY_COMPLETE_RETURN on the same allocation
-> seal resources
-> atomically commit one boundary receipt
-> ACK_RESPONSE
-> BEGIN_REUSE
-> LEASE fresh client ports on the restored carrier without PREPARE
-> ISOLATE_SOURCE again for the new generation
```

The boundary window contains sixteen 64-bit receipt words.  Every word reads
as all ones until both outputs are held, the return class is known, resources
are sealed, and the atomic group commit succeeds.  The coherent client states
remain behind the backend port; MMIO exposes no collapsed phase answer.

`EXACT_FORMAL` is the only return class that increments exact restoration
generation and permits `BEGIN_REUSE`.  `APPROX_MODEL`, `STATISTICAL_ONLY`, or
`FAILED` cannot be relabeled exact and cannot reuse catalytically.

Exact ideal return requires all of:

- full `K,R_K` joint equality to preparation;
- factorization from both complete client/reference pairs;
- exact Choi action `diag(1, omega^s)` for each client;
- trivial/factored environment;
- clear ports and no partial response;
- identical resident allocation and custody epoch;
- sealed resource vector.

Failure before atomic commit releases no boundary or partial client group.
Missing, wrong, duplicated, early, or stale private binding is an error rather
than an alternate result.

## ABI discipline

All registers have one declared width and natural alignment.  The map retains
the useful V1 control offsets through `0x098`, appends capability, return,
custody, and total-resource fields, and places the read-only boundary at
`0x200..0x27f`.  Unknown writes, width mismatch, unaligned access, and writes
to read-only registers return `BAD_ARGUMENT` without mutating lifecycle.

The command values are:

```text
1 LEASE             6 BEGIN_REUSE
2 PREPARE           7 SNAPSHOT (reject)
3 ISOLATE_SOURCE    8 ARM_PRIVATE
4 SEAL_DESCRIPTOR   9 ACK_RESPONSE
5 EXECUTE_ATOMIC   10 ABORT_PREEXEC
```

`CMD_SNAPSHOT` is always rejected.  Every accepted migrated instance is marked
`SHAM`, its response remains locked, private-ready mask is cleared, and
execute/reuse is denied.  The per-instance migration latch is not in VMState
and survives machine reset.  Because PCI reset disables memory decoding, a
test must re-enumerate the device and remap BAR0 before checking that the
post-reset instance is still `SHAM`; reading the stale address is not
evidence.  Hidden client, carrier, private-provider, and environment state is
not promoted across migration.  Migrating an already sanitized `SHAM` lineage
into another instance must yield `SHAM` again rather than fail validation or
revive the carrier.  A future physical adapter must install a migration
blocker.

## Required implementation evidence

Promotion to the bounded compiled-device claim requires evidence from a QEMU
10.2.4 binary built with the package source:

- PCI enumeration of `1234:11fb`, PH11 magic, ABI, and backend ID;
- exact installed source hash and exactly one Kconfig/meson entry;
- all nine ordered private residue pairs under the identical guest descriptor
  on one resident allocation, with one prepare, eight reuses, and fresh source
  isolation on every generation;
- independent exact 96-dimensional reference parity;
- response locked before execution, held through atomic commit, readable only
  after verified return, and withdrawn only by `ACK_RESPONSE`;
- two-generation same-process, same-allocation reuse without a second prepare;
- ideal and zero-noise-open ABI parity;
- nonzero open parameters never reported exact or reusable;
- external adapter unavailable without mutation;
- wrong order, missing/duplicate binding, secret smuggle, carrier mutation,
  port failure, resource-unsealed, and inverse/return fault controls;
- real QMP Unix migration, second-hop SHAM migration, and reset-irreversible
  sham lineage after PCI BAR re-enumeration;
- callback-level rejection and `BAD_ARGUMENT` latching for invalid byte, word,
  and unaligned MMIO accesses;
- a width-correct post-commit BAR mutation attempt that leaves the response
  resource snapshot, digest, boundary, and argument state unchanged;
- callback-dispatched external-adapter lease rejection with no lifecycle
  mutation;
- empty QEMU stdout/stderr and no secret/state path in BAR0 or migration;
- a deterministic separate reference importing no production code or output.

A source-only package, a Python twin, or a passing standalone algebra oracle is
not V11 promotion evidence.

## Resource and comparator law

The device reports a vector rather than a selectively weighted score:

```text
state and scratch cells
coefficient/precision width
secret storage and control words
preparation and certification operations
coherent query applications
duration and output hold time
interaction action and carrier energy
port bandwidth
loss and dephasing parameters
environment history
custody transitions and reuse count
maintenance, discarded trials, and postselection
descriptor/compiler work and whole-process uninstrumented costs
```

Uninstrumented runtime, allocator, build, controller, authentication, and
physical costs remain nonzero unknowns rather than zero.

The implemented ideal receipt charges one resident 9216-cell density and one
independent 9216-cell scratch matrix.  It reports allocated private-provider
state (168 bits in this ABI) separately from the four-bit logical residue-pair
entropy.  The cumulative control count includes every width-correct BAR write
and accepted private QOM binding: 29 words at generation one and 30 additional
words per acknowledged reuse transaction.  Unknown compiler, construction,
duration, bandwidth, hold-time, energy, and physical costs use the explicit
`UINT64_MAX` unknown sentinel; they are never interpreted as measured maxima.
Known counters fail closed before they can wrap or reach that reserved
sentinel, and allocation, arm, and transaction serial exhaustion is rejected
before a receipt can reuse an identifier.

At response commit the device snapshots the cumulative control counter and
reports that frozen value while `RESOURCE_SEALED` remains set.  Later BAR
traffic is still charged to the live cumulative counter, but it cannot mutate
the argument state, sealed response vector, digest, or boundary.  The next
successful `BEGIN_REUSE` unseals the live vector for the new generation.  The
digest covers the resource schema and every reported vector coordinate,
including peak bits, allocated secret storage, and the controller count; an
unknown coordinate is hashed as the explicit sentinel rather than omitted.

The strongest equal-interface quantum comparator performs the identical two
queries.  A direct secret-controlled client phase implementation omits the
carrier and restoration.  A coherent classical-wave phase apparatus is also
charged honestly.  Public tables fall back under M241/M242 and M257.  A
restricted provider changes the access premise; it does not prove advantage.

For a generic `N`-entry `d`-ary phase oracle, operational secret storage is at
least `(N-1) log2(d)` bits modulo global phase.  An exact reusable fixed
processor needs program dimension at least `d^(N-1)`.  Two independent exact
programs pay the product dimension.  Forrelation retains its oracle-table
construction, calibration, and amortization costs.

## Nonclaims

V11 does not establish:

- a physical cavity, photon, coherent client port, authenticated provider, or
  same-mode custody observation;
- physical exact restoration under loss, dephasing, leakage, or environment;
- a secure secret, unknown-state oracle, or coherent black-box separation;
- a unique phase resource, query advantage, total-resource advantage, or
  complexity lower bound;
- an M257 escape, Small Wall crossing, bit replacement, or unbounded compute;
- validity of the open/noisy or external stubs beyond their fail-closed
  lifecycle classification.

M257 remains intact:

```text
EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_COUNTED_AS_A_PHASE_RESOURCE
```
