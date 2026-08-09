# Phase-QEMU V12 authenticated-adapter stub contract

## Scope and authority

M270 is a compiled common-device integration result.  It is a QEMU 10.2.4
PCI device with one guest-visible ABI and swappable ideal, open, and external
backend implementations.  Its external backend is a hardware-disconnected,
test-only protocol stub.  It is not a physical adapter merely because it is
compiled or because its package and QOM names contain `adapter`.

The canonical bounded claim is:

```text
COMPILED_QEMU_10_2_4_PHASE_QEMU_V12_COMMON_GUEST_VISIBLE_PCI_DEVICE_BACKEND_EXERCISES_A_HARDWARE_DISCONNECTED_TEST_ONLY_DETERMINISTIC_21_BIT_INTEGRITY_TWO_SLOT_ASYNCHRONOUS_EXTERNAL_ADAPTER_FOR_ALL_NINE_INTERNAL_ZOMEGA_PAIRS_AND_COMMITS_ONLY_APPROX_MODEL_OUTPUTS_HELD_UNTIL_ACK_THEN_SPENT_WITH_NO_BEGIN_REUSE_WHILE_PENDING_RESET_REAL_QMP_MIGRATION_AND_SERVICE_DEFAULT_FAIL_CLOSED
```

Claim ceiling:

```text
NO_HARDWARE_PHYSICAL_CARRIER_COHERENT_PORT_CRYPTOGRAPHIC_SECURITY_STATISTICAL_PHYSICAL_EVIDENCE_CUSTODY_PHYSICAL_RETURN_RESTORATION_REUSE_OR_ADVANTAGE_ALL_EXTERNAL_PHYSICAL_QUANTITIES_UNKNOWN_AND_FINITE_PHYSICAL_EVIDENCE_MAY_NEVER_BE_EXACT_FORMAL_OR_AUTHORIZE_BEGIN_REUSE
```

Restoration classification:

```text
NO_RESTORATION_CLAIM
```

Scope:

```text
COMPILED_QEMU_10_2_4_PHASE_QEMU_V12_PCI_ABI_HARDWARE_DISCONNECTED_TEST_ONLY_TWO_SLOT_ASYNC_ADAPTER_PROTOCOL_AND_INTERNAL_96_DIMENSION_ZOMEGA_SOFTWARE_MODEL_ONLY
```

Disposition:

```text
V12_ESTABLISHES_COMMON_COMPILED_ASYNC_ADAPTER_LIFECYCLE_INTEGRITY_ORDER_EXPIRY_CANCEL_TIMEOUT_ACK_SPENT_AND_SHAM_CONTROLS_BUT_NOT_PHYSICAL_OR_CRYPTOGRAPHIC_PROMOTION_DIRECT_EQUAL_ACCESS_PHASE_COMPILER_CONTROLS_AND_M257_REMAINS_INTACT
```

Successor:

```text
REAL_AUTHENTICATED_HARDWARE_CONNECTED_DUAL_RAIL_DISPERSIVE_ADAPTER_BEHIND_THE_COMMON_PHASE_QEMU_BACKEND_WITH_BOUNDED_PHYSICAL_RECEIPTS_AND_INDEPENDENT_STATISTICAL_VALIDATION_WITHOUT_EXACT_FORMAL_PHYSICAL_RETURN_OR_REUSE
```

The controlling M257 guardrail remains:

```text
EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_COUNTED_AS_A_PHASE_RESOURCE
```

## Architectural discipline

Phase-QEMU maintains a hard boundary between mechanism search and common-device
promotion.

- V0 and V1 are compiled Phase-QEMU PCI devices.
- V2 through V10 are mechanism-search or digital-twin packages.  They may be
  searched, falsified, killed, or nominated, but cannot promote a mechanism.
- V11 is the first compiled common backend/device reintegration checkpoint.
- V12 is a second compiled device with a distinct PCI/QOM identity and a
  hardware-disconnected asynchronous external-adapter test backend.
- A folder name, Python twin, standalone algebra, or protocol simulator is not
  common-device integration evidence.
- Physical promotion requires a compiled backend behind the common guest ABI,
  a real adapter, bounded physical receipts, and independent statistical
  validation.

V12 is a sibling of frozen V11, not an in-place mutation of V11 evidence.  Its
installer must preserve the installed V0, V1, and V11 sources and integration
entries.  The device core owns request tags, lifecycle, descriptor sealing,
boundary locking, response commit, resource sealing, ACK, reuse policy, and
migration/reset lineage.  Backend callbacks supply prepare, client supply,
execute, poll, return classification, cancel, and sanitize operations.  `LEASE`
dispatches through the selected backend's `lease_preflight`; the core invokes
backend execution first and only after completion enters `VERIFYING_RETURN`
and dispatches the distinct `verify_return` callback.

## Compiled device and ABI

The compiled identity is:

```text
QOM type       phase-qemu-v12
PCI identity   1234:11fc, revision 1
BAR0           4096 bytes
magic          PH12 = 0x50483132
ABI            0x00020000
```

Backend identifiers are immutable after realization:

```text
0x0b01  IDEAL_DUAL_RAIL
0x0b02  OPEN_DUAL_RAIL
0x0b80  EXTERNAL_ADAPTER
```

V12 preserves the V11 register prefix through offset `0x1b8`.  The adapter
ledger occupies the nonoverlapping range `0x1c0..0x1f8`:

```text
0x1c0  ADAPTER_STATE          u32 read-only
0x1c8  AUTH_ACCEPTED         u64 read-only
0x1d0  AUTH_REJECTED         u64 read-only
0x1d8  DISPATCHES            u64 read-only
0x1e0  COMPLETIONS           u64 read-only
0x1e8  CANCELS               u64 read-only
0x1f0  VIRTUAL_TICK          u64 read-only
0x1f8  DEADLINE_TICK         u64 read-only
```

The sixteen-word boundary remains at `0x200..0x278`, uses header
`0x5031324200010080`, and remains all ones until atomic commit.  Resource
schema `0x00020001` uses `UINT64_MAX` for an unknown coordinate.  Every
register has one natural width; invalid byte, word, or unaligned accesses are
handled by the device callback and latch `BAD_ARGUMENT`.

The accepted eight-word descriptor is identical for every secret pair:

```text
50313144 00010008 00000002 00020102
00020011 00030021 00000003 00010001
```

A residue, expected output, inverse, amplitude, state vector, or
secret-derived guest timing choice in that descriptor is `SECRET_SMUGGLE`.

## Internal algebra and backend classifications

The ideal software engine materializes a `96 x 96` density matrix over
`Z[omega]`, represented by two signed 64-bit coefficients with
`omega^2 = -1 - omega`.  The dimensions are

```text
C_A, R_A, C_B, R_B, K(vac,a,b), R_K = 2,2,2,2,3,2.
```

Preparation creates the product of two client/reference Bell-Choi pairs and a
dual-rail carrier/reference Bell pair, with eight pure components.  Each query
is applied through the diagonal controlled-number exponent to the complete
density matrix.  It is not implemented as a direct client-phase shortcut.

The V12 external test backend reuses that internal engine only after its two
asynchronous slots complete.  The qtest covers all nine ordered pairs in
`Z_3 x Z_3`.  Exact closure of this internal software algebra is not a physical
return receipt.  External verification clears same-backing, complete-return,
factorization, and environment-factorization observations and returns only
`APPROX_MODEL`.

The return taxonomy is:

```text
NONE = 0
EXACT_FORMAL = 1
APPROX_MODEL = 2
FAILED = 3
STATISTICAL_ONLY = 4
```

For the external V12 backend:

- every successful result is `APPROX_MODEL`;
- `STATISTICAL_ONLY` is not produced and is not statistical physical evidence;
- `EXACT_FORMAL`, exact return generation, restoration, physical same backing,
  and physical environment factorization remain false;
- finite future physical evidence may be statistical or approximate, but must
  never be relabeled `EXACT_FORMAL` and must never authorize `BEGIN_REUSE`.

The zero-parameter open backend and ideal backend retain their bounded V11
formal classifications.  They are outside the external physical claim.

## Test-only 21-bit integrity envelopes

The external test provider is disabled by default.  It is enabled only when
all of the following realization properties hold:

```text
backend-id=0x0b80
test-provider-enabled=on
test-adapter-enabled=on
```

Service-default external configuration therefore fails `LEASE` through
`lease_preflight` with `BACKEND_UNAVAILABLE` and leaves lifecycle `EMPTY`.
Direct use of `test-private-a` or `test-private-b` is rejected for the external
backend; the two adapter envelope properties are the only accepted test path.

After `ARM_PRIVATE`, host-only write-only QOM properties
`test-adapter-envelope-a` and `test-adapter-envelope-b` accept one 64-bit word
each.  Each word has a 43-bit payload and a 21-bit deterministic integrity tag:

```text
bits  0..1   residue, restricted to 0..2
bit       2  wire slot, A=0 or B=1
bits  3..18  generation, 16 bits
bits 19..26  sequence, A=1 then B=2
bits 27..34  issued virtual tick
bits 35..42  expiry virtual tick
bits 43..63  deterministic 21-bit integrity tag
```

The tag mixes domain `M270`, allocation ID, descriptor fingerprint, arm nonce,
external backend ID, and the 43-bit payload through the source's FNV-derived
test function.  It detects the runner's bad-tag, wrong-slot, stale-generation,
out-of-order, expired, bypass, and replay fixtures.  It is explicitly not a
cryptographic MAC, AEAD construction, confidential channel, secure key store,
or adversarial authentication result.  The QOM/QMP test provider is privileged
test infrastructure and does not protect data from the host or QMP operator.

Both slots are write-once for one arm lineage.  Slot A must precede slot B.
Accepted envelopes increment the authentication ledger; rejected envelopes do
not dispatch a query.  Envelope words, residues, arm nonce, density, scratch,
descriptor, and boundary are absent from VMState.

## Async lifecycle and atomic response law

The required successful external order is:

```text
LEASE
-> PREPARE internal model
-> ISOLATE_SOURCE
-> upload and SEAL_DESCRIPTOR
-> ARM_PRIVATE
-> bind authenticated test envelope A
-> bind authenticated test envelope B
-> EXECUTE_ATOMIC starts WAITING_A
-> POLL_EXTERNAL advances to WAITING_B
-> POLL_EXTERNAL completes the internal model
-> VERIFYING_RETURN classifies APPROX_MODEL
-> seal resource vector and atomically publish one boundary
-> hold modeled outputs until ACK_RESPONSE
-> ACK_RESPONSE clears the boundary and enters SPENT
-> BEGIN_REUSE rejects with REUSE_NOT_QUALIFIED
```

`EXECUTE_ATOMIC` returns while lifecycle is `EXECUTING`.  The external backend
uses `READY`, `WAITING_A`, `WAITING_B`, `COMPLETE`, `CANCELED`, `TIMED_OUT`, and
`SHAM` adapter states.  The completion-mode schedule has two dispatches, two
poll completions, two virtual ticks, and deadline tick two.  No boundary word
is readable before the second poll and complete return classification.

`test-adapter-mode=1` is a deterministic timeout fixture.  At deadline tick
two it cancels once and commits a sealed `FAILED` receipt with
`ADAPTER_TIMEOUT`.  `CANCEL_EXTERNAL` while pending cancels once and commits a
sealed `FAILED` receipt with `ADAPTER_CANCELED`.  In both cases the resource
attempt and sixteen-word failure boundary are held in `RESPONSE_READY` until
`ACK_RESPONSE`, which clears the boundary and enters `SPENT`; late completion
and `BEGIN_REUSE` reject.  Authentication, order, expiry, and replay rejection
before dispatch publish no response.  No failed receipt contains or releases
the modeled successful outputs, and no failure can authorize reuse.

The response-local resource snapshot, its digest, and all sixteen boundary
words are immutable after commit.  A rejected post-commit poll is outside the
sealed controller snapshot.  Post-commit QOM writes to either adapter-envelope
property reject at the resource-sealed guard before changing the live error,
any counter, or any resource state, including the controller count, digest,
and boundary.  For both successful and sealed `FAILED` attempts,
qualification independently
recomputes the low and high resource-digest words and the low and high return
receipts that respectively bind those digest words.  Modeled outputs remain
held through commit and are released only by `ACK_RESPONSE`; because the
external class is nonexact, ACK transitions directly to `SPENT`.

## Reset, snapshot, and migration

`SNAPSHOT` is rejected at the command interface.  A canonical reset is allowed
only before an external private envelope or pending/held transaction exists.

Reset after either envelope is accepted, while either asynchronous slot is
pending, or while a response is held cancels pending work once, sanitizes
hidden state, clears receipts and boundary, removes the carrier, and enters
`SHAM`.  The non-VMState migration latch makes this state reset-irreversible;
a second reset must remain `SHAM` and must not repeat cancellation.  The
pending-reset fixture dynamically commands `BEGIN_REUSE` and observes
`SNAPSHOT_LINEAGE`, never authorization.

V12's actual migration law is sanitized migration, not a migration blocker.
Real QMP Unix live migration of an in-flight source first calls `pre_save`,
which cancels pending work once and burns and sanitizes the source into
reset-irreversible `SHAM` before any bytes are serialized.  Only the narrow
VMState marker/configuration fields transfer.  `post_load` always sanitizes the
destination into `SHAM`.  Migrating that sham to a second destination again
burns the source and produces `SHAM`, and a subsequent system reset cannot
revive it.  Tests must re-enumerate PCI and remap BAR0 after system reset before
reading the result.  The migrated-SHAM fixture dynamically commands
`BEGIN_REUSE` and observes `SNAPSHOT_LINEAGE`, never authorization.  Together
with the direct dynamic rejection checks after timeout and cancellation, these
fixtures cover all four terminal paths.

Explicit device `unrealize` cancels pending adapter work when present, invokes
the backend sanitizer, and clears private envelopes, descriptor, boundary, and
observer state.  The qtest observes only that an adapter was pending before
QMP `quit` and that process teardown exited cleanly with empty captured
streams; QMP teardown cannot expose callback internals.  A separate audit of
the pinned installed source proves that the registered `unrealize` callback
performs cancellation and sanitization.  Qualification must not convert the
clean process observation into a dynamic-unrealize or hot-unplug claim.

No migration result proves physical continuity, custody, carrier return, or a
secure session transfer.  A real hardware-connected successor should block
migration or implement a separately reviewed authenticated session-handoff
law; it must not serialize live secret or physical state as ordinary VMState.

## Resource law

The resource vector is a coordinate ledger, not a scalar advantage score.
For the compiled external model it includes:

- `9216` density cells and an independent `9216`-cell scratch matrix;
- two signed 64-bit `Z[omega]` coefficients per cell;
- an explicit peak floor equal to the resident device object plus the largest
  named stack-local allocation, the 36-entry retained carrier/reference
  marginal; this is not a whole-QEMU allocator, RSS, or process peak;
- allocated private storage for two residues, two bound flags, ready mask,
  arm generation, arm nonce, armed flag, and two envelope words;
- four bits of fixed register storage/code capacity, `ceil(log2(3^2))`, for
  the nine residue pairs; this is not a Shannon-, min-, or measured entropy
  assertion;
- accepted/rejected integrity checks, dispatches, completions, cancellations,
  virtual tick, deadline, control writes, preparations, queries, full-state
  return checks, certification operations, custody transitions, and discarded
  trials;
- response-local resource digests and controller counts frozen at commit.

The internal model applies two queries and reports its formal action coordinate.
That number is not a measured external action.  External physical duration,
port bandwidth, action, energy, photon number, loss, dephasing, environment
history, output-hold time, maintenance, construction, calibration, and custody
are unknown.  In particular, the external carrier-photon, loss, and dephasing
registers return the unknown sentinel after successful classification.  No
unknown coordinate may be interpreted as zero.

Compiler and construction work remain unknown sentinels.  Controller traffic
is counted from BAR and accepted/rejected provider writes, but host scheduling,
QEMU/QMP transport, Python runner work, compilation, and machine maintenance
must be charged separately in an equal-interface comparison.

## Equal-interface comparator and M257

The strongest comparator receives the same two ternary residues, compiles the
same two diagonal phases directly, holds its two outputs until an atomic ACK,
and pays its descriptor, provider, controller, memory, build, and runtime
costs.  It reproduces the internal software phase answers without the adapter
protocol.  The V12 result therefore establishes neither unique query power nor
total-resource advantage.

A restricted provider would change an access premise, but the present
test-provider QOM properties are controlled by the qualification runner.  They
do not escape M257 or establish a black-box secret inaccessible to the direct
phase compiler.

## Installer, build, qtest, and reference evidence

The fail-closed installer must:

- accept only an explicitly passed QEMU 10.2.4 source tree;
- structurally validate the V12 C source, exact descriptor, register
  nonoverlap, VMState exclusions, async callbacks, classifier, and SHAM path;
- reject PCI ID or QOM collisions and partial, malformed, or duplicate V12
  Kconfig/Meson integration;
- reject an unknown nonidentical installed V12 target; update only an exact
  current target or an explicitly hash-allowlisted canonical prior V12;
- before any integration write, reject an existing V12 target that is a
  symlink or any other non-regular filesystem object;
- on a first-install copy failure with no preflight target, roll back
  integration texts and move only the partial regular file created by that
  attempt recoverably to the sibling `.failed-install-recovery` path; never
  move a pre-existing object, silently delete the partial file, or overwrite
  an existing recovery target;
- require the frozen V11 source and canonical V11 integration, and compare
  every named V11 register and offset against the V12 prefix;
- preserve V0, V1, and V11 source hashes and integration counts;
- add one V12 source, one Kconfig entry, and one Meson entry;
- report only source integration.  Installer output must keep compiled-device,
  common-contract-compiled, and reintegration gates false until a separate
  build and qtest receipt establishes them.

The compiled evidence set must contain ledgers for:

1. installer/source integration and preservation checks;
2. QEMU 10.2.4 compilation, linking, PCI/QOM enumeration, and empty captured
   process streams;
3. all nine external model pairs, integrity negatives, timeout, cancellation,
   pending reset, two-hop QMP migration, post-migration reset, and
   service-default unavailability;
4. the independent reference boundary.

The current M270 runner independently implements the envelope packing,
FNV-derived tag, receipt arithmetic, resource-digest recomputation, and direct
equal-access phase compiler without importing the V12 C source.  It imports
the frozen V11 runner only for qtest/QMP transport helpers.  A new standalone
M270 separate-reference file independently implements the exact symbolic
`Z[omega]` law for all nine residue pairs, checks direct-compiler parity, and
states the finite-evidence ceiling through a symbolic family of nonexact
countermodels approaching the ideal map.  This is a symbolic countermodel
argument, not a numerical experiment, fitted limit, or machine-derived
convergence theorem.  That reference does not execute QEMU, exercise the
common guest ABI, or compute or validate the test integrity tag.  The runner
covers the compiled V12 protocol and framing; the separate reference covers
only the internal symbolic science.  Neither is physical evidence.

A durable build receipt now exists at
`evidence/PHASE_QEMU_V12_BUILD_RECEIPT.json`.  It records two byte-identical
deterministic qtest evidence replays, two byte-identical separate-reference
replays, the compiled common-architecture pass, and the explicit absence of
physical promotion.  The strict qualifier will pin and cross-check every
artifact identity; this contract deliberately does not duplicate volatile hash
values.

## Nonclaims

V12 does not establish:

- hardware contact, a real adapter, physical cavity, photon, finite-energy
  eigenstate carrier, or coherent client/reference port;
- cryptographic authentication, confidentiality, key custody, access security,
  or an adversarially secure envelope;
- statistical physical evidence, calibrated error bars, physical sampling, or
  a physical likelihood model;
- physical same-mode custody, physical return, environment factorization,
  restoration, catalytic reuse, or `BEGIN_REUSE` eligibility;
- calibrated duration, bandwidth, action, energy, loss, dephasing, leakage,
  maintenance, construction, or total resources;
- a unique phase resource, query advantage, total-resource advantage,
  complexity separation, M257 escape, Small Wall crossing, unbounded compute,
  or replacement of a bit by a physical phase;
- promotion of V2 through V10 merely because their twins remain searchable;
- physical validity of the ideal/open algebra or external stub beyond their
  explicit software classifications.
