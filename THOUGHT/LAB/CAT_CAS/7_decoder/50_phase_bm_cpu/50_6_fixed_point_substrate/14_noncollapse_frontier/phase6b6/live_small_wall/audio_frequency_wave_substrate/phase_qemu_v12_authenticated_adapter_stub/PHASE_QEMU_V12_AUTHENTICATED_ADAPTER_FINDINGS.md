# Phase-QEMU V12 authenticated-adapter stub findings

## Verdict and frozen authority

The recorded M270 result passes as a compiled QEMU 10.2.4 common guest-visible
PCI device/backend exercise.  The external path is a hardware-disconnected,
test-only deterministic integrity protocol.  Its exact internal software
algebra closes for all nine ordered ternary pairs, but every external response
is classified `APPROX_MODEL`, held until ACK, and then spent.  This is not a
physical observation, cryptographic-security result, restoration result, or
advantage result.

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

## Evidence identity and status

The durable identity ledger is
`evidence/PHASE_QEMU_V12_BUILD_RECEIPT.json`.  It records two byte-identical
deterministic qtest evidence replays, two byte-identical separate-reference
replays, the compiled common-architecture pass, and no physical promotion.
The strict qualifier will pin and cross-check every artifact identity.  This
findings narrative avoids duplicating volatile hash values; authority rests in
the receipt and qualifier checks.

| Evidence item | Current bounded result |
| --- | --- |
| Build | `PASS_COMPILED_QEMU_10_2_4_PHASE_QEMU_V12_COMMON_DEVICE` |
| qtest/QMP | `PASS_M270_PHASE_QEMU_V12_AUTHENTICATED_ADAPTER_QTEST_EVIDENCE_V1` |
| QEMU identity | `phase-qemu-v12`, `1234:11fc`, revision 1, PH12, ABI `0x00020000` |
| External connection | hardware disconnected |
| Integrity scheme | deterministic test-only 21-bit tag; noncryptographic |
| Return class | every successful external result is `APPROX_MODEL` |
| Restoration | `NO_RESTORATION_CLAIM` |
| Reuse | forbidden; ACK enters `SPENT`; `BEGIN_REUSE` rejects |
| Total-resource comparison | `UNDETERMINED` |
| Physical evidence class | `NONE` |

## Architecture finding

The architecture gate is satisfied only at the compiled software-device level:

- V0 and V1 remain actual compiled devices.
- V2 through V10 remain searchable, falsifiable mechanism twins.  A twin may
  kill or nominate a mechanism but cannot promote it.
- V11 remains the frozen compiled common-backend reintegration checkpoint.
- V12 is a distinct compiled common guest-visible PCI device with a new
  asynchronous external-backend test path.
- Physical promotion still requires a real adapter behind the common backend
  and independent statistical validation.  The compiled test stub is not that
  promotion.

The V12 source contains one `PhaseV12BackendOps` table boundary.  The core
retains lifecycle and commit ownership; the external table supplies
`lease_preflight`, prepare, execute, poll, verify-return, cancel, and sanitize
callbacks.  `LEASE` dispatches through the selected backend's
`lease_preflight`.  Core execution and return verification are separate
lifecycle stages: only a completed execute/poll result enters
`VERIFYING_RETURN` and dispatches `verify_return`.  Service-default external
realization leaves both test-provider and test-adapter disabled, so `LEASE`
fails unavailable without leaving `EMPTY`.

## Device and protocol finding

The compiled device exposes a 4096-byte BAR and preserves the V11 register
prefix.  Eight additional read-only adapter-ledger registers occupy
`0x1c0..0x1f8`; the locked sixteen-word response boundary remains
`0x200..0x278`.  The fixed public descriptor is identical for every ordered
pair and contains no residue, expected phase, inverse, or amplitude.

The external provider accepts two ordered 64-bit QOM envelope words after
`ARM_PRIVATE`.  Each word contains a 43-bit payload—residue, slot, generation,
sequence, issued tick, and expiry tick—and a deterministic 21-bit tag bound to
allocation, descriptor fingerprint, arm nonce, backend, and payload.  Slot A
must precede slot B.  The qtest runner independently computes these words.

This tag is an integrity fault-control mechanism, not a cryptographic MAC or
confidential transport.  The QOM/QMP host can see and control test material.
Direct private-residue setters are rejected on the external backend, but that
test boundary does not establish access security against the host.

## All-nine-pair finding

The runner creates a separate QEMU fixture for each pair in
`{0,1,2} x {0,1,2}`.  It does not claim nine generations on one physical or
resident allocation.  In every fixture it performs:

```text
LEASE -> PREPARE -> ISOLATE_SOURCE -> SEAL_DESCRIPTOR -> ARM_PRIVATE
-> envelope A -> envelope B -> EXECUTE_ATOMIC
-> POLL_EXTERNAL -> POLL_EXTERNAL -> RESPONSE_READY
```

The external backend then runs the full internal `96 x 96` `Z[omega]` density
law, including both controlled-number queries and complete internal return
verification.  Test-only observers report the expected two phase residues.
Before commit, external verification deliberately clears physical same-backing,
carrier/reference return, factorization, and environment-factorization claims.
The committed return class is always `APPROX_MODEL`, never `EXACT_FORMAL` or
`STATISTICAL_ONLY`.

Each successful fixture records two accepted envelopes, two dispatches, two
completions, virtual tick two, and deadline tick two.  The response is locked
until completion, its modeled outputs remain held through atomic commit, and a
rejected post-commit poll cannot change the boundary, resource digest, or
sealed controller snapshot.  `ACK_RESPONSE` releases the modeled outputs and
transitions directly to `SPENT`; `BEGIN_REUSE` returns
`REUSE_NOT_QUALIFIED`.  A post-commit QOM write to either adapter-envelope
property rejects before changing the live error, authentication counters,
any other counter, or any resource state, including the controller count,
digest, and boundary.  For every successful fixture, the runner independently
recomputes both digest words and both return-receipt
words that respectively bind them.

## Integrity and asynchronous fault controls

Seven integrity controls reject before dispatch:

| Fixture | Required classification |
| --- | --- |
| bad tag | `ADAPTER_AUTH_BINDING` |
| wrong wire slot | `ADAPTER_SLOT` |
| stale generation | `GENERATION_MISMATCH` |
| slot B before A | `ADAPTER_ORDER` |
| expired envelope | `ADAPTER_EXPIRED` |
| direct private-setter bypass | `ADAPTER_AUTH_BINDING` |
| exact replay | `ADAPTER_REPLAY` |

The timeout fixture reaches the exact virtual deadline at tick two, enters
`TIMED_OUT`, cancels once, commits a sealed `FAILED` receipt and response-local
resource digest, rejects late completion, and requires `ACK_RESPONSE` before
entering `SPENT`.  The cancel fixture cancels once while pending, enters
`CANCELED`, likewise commits a sealed `FAILED` receipt, rejects late
completion, and requires ACK before `SPENT`.  These failure boundaries release
no modeled successful output and never permit reuse.  The runner independently
recomputes both digest words and their bound low/high return receipts for both
timeout and cancellation, then dynamically verifies that `BEGIN_REUSE`
rejects after each ACK-to-`SPENT` transition.

These controls establish deterministic software ordering and fail-closed state
transitions only.  They do not measure network timing, hardware latency,
physical decoherence, malicious transport behavior, or cryptographic strength.

## Reset and migration finding

A QMP `system_reset` while the adapter is pending cancels exactly once,
re-enumerates the PCI BAR, and yields `LIFE_SHAM`, `SNAPSHOT_LINEAGE`, a locked
boundary, no lease, and adapter state `SHAM`.  A second reset remains `SHAM`
without a second cancellation.  The fixture dynamically commands
`BEGIN_REUSE` and observes `SNAPSHOT_LINEAGE`.

The migration fixture uses real QMP Unix live migration rather than a mocked
serialization function.  Source `pre_save` cancels a pending adapter once and
burns and sanitizes the source into reset-irreversible `SHAM` before
serialization.  Destination `post_load` sanitizes the first destination into
`SHAM`.  A second-hop migration burns its source and again produces `SHAM`; a
subsequent QMP reset and PCI re-enumeration do not revive it.  Hidden envelopes,
residues, density, scratch, arm nonce, descriptor, and boundary are not VMState
fields.  The migrated-SHAM fixture dynamically commands `BEGIN_REUSE` after
the second hop and reset and observes `SNAPSHOT_LINEAGE`.

The explicit `unrealize` callback cancels pending adapter work when present,
invokes backend sanitization, and clears private envelopes, descriptor,
boundary, and observers.  Qtest observes only that the adapter was pending
before QMP `quit` and that the process then exited cleanly with empty captured
streams; it cannot observe callback internals.  A separate audit against the
pinned installed C source establishes that the registered callback cancels and
sanitizes.  Neither layer asserts a qtest-observed dynamic unrealize or an
unsupported hot-unplug.

This is reset-irreversible software lineage sanitization.  It is not physical
custody continuity, adapter-session handoff, carrier return, or evidence that a
real adapter can migrate.  The current V12 implementation burns the source in
`pre_save` and forces the destination to `SHAM` in `post_load`; it is not a
migration blocker.

## Installer and build ledger

The installer is narrowly fail closed:

- it accepts exactly QEMU 10.2.4 from an explicit source path;
- it structurally validates the C source, exact descriptor, nonoverlapping 4K
  register map, async poll, external classifier, SHAM path, and exclusion of
  hidden state from VMState;
- it rejects PCI/QOM collisions and partial, malformed, or duplicate V12
  Kconfig/Meson entries, including noncanonical collision targets;
- it rejects an unknown nonidentical V12 target and permits replacement only
  for an exact current target or an explicitly hash-allowlisted canonical
  prior revision;
- before any integration write, it rejects an existing V12 target that is a
  symlink or any other non-regular filesystem object;
- if a first-install copy fails after this preflight creates a partial regular
  target, it rolls back Kconfig/Meson and moves only that attempt-created file
  recoverably to the sibling `.failed-install-recovery` path; it never moves a
  pre-existing object and fails closed rather than overwriting an existing
  recovery file;
- it requires frozen V11, compares the complete named V11 register/offset map
  with the V12 prefix, and snapshots V0/V1/V11 source and integration state;
- it installs one V12 C source and exactly one V12 Kconfig and Meson entry;
- its own receipt leaves `qemu_device_implemented`, compiled common-contract,
  and reintegration gates false, because source installation alone is not
  compilation evidence.

The separate compiled build/qtest result and durable
`evidence/PHASE_QEMU_V12_BUILD_RECEIPT.json` establish the bounded
compiled-device status recorded above.  The strict qualifier will pin the
source, installed source, installer, QEMU binary, runner, deterministic qtest
evidence, separate reference, receipt, contract, and findings as one
cross-checked evidence set.

## Qtest ledger

The headless runner uses QEMU qtest acceleration, Unix qtest/QMP sockets,
`-display none`, no default devices, and no hardware adapter.  Its logical
evidence includes:

- PCI/QOM identity and common BAR contract;
- nine ordered external model pairs;
- two-envelope authentication/order/expiry/replay rules;
- boundary lock, held output, atomic commit, sealed-resource immutability,
  post-commit QOM-envelope rejection before live-state mutation, ACK-to-SPENT,
  and reuse rejection;
- the seven negative integrity fixtures;
- deterministic timeout and cancellation with sealed `FAILED` receipts,
  independently recomputed low/high digests and bound return receipts,
  ACK-to-SPENT, and dynamic `BEGIN_REUSE` rejection;
- pending-reset SHAM, second reset, and dynamic `BEGIN_REUSE` rejection;
- real first-hop and second-hop QMP Unix migration, source pre-save burn,
  destination post-load SHAM, post-migration reset, and dynamic `BEGIN_REUSE`
  rejection;
- clean pending-process QMP teardown as an observation distinct from the
  pinned-source static audit that proves unrealize cancellation/sanitization;
- service-default external unavailability;
- empty captured QEMU stdout and stderr for each fixture;
- direct equal-access comparator and no-advantage classifications.

The evidence JSON also contains wall-clock duration fields.  Those timings are
run metadata, not physical adapter timing and not a deterministic scientific
observable.

## Independent-reference ledger

Reference authority is deliberately split:

| Layer | Independent check | Limit |
| --- | --- | --- |
| Internal `Z[omega]` algebra | new standalone M270 exact symbolic reference for all nine pairs, direct-compiler parity, and a symbolic finite-evidence countermodel argument | no numerical convergence experiment, QEMU execution, guest-ABI, adapter, or physical claim |
| V12 envelope/tag and resource receipt | runner-local independent Python FNV, receipt, tag, packing, resource-digest, and direct-compiler implementation | no cryptographic-security or separate-device claim |
| qtest/QMP transport | frozen V11 transport helpers imported by the V12 runner | transport reuse, not an independent device implementation |
| Physical adapter | none | physical evidence class remains `NONE` |

The M270 package now has a standalone separate-reference file that imports no
project module.  It independently implements only the exact internal symbolic
law; it expressly does not compute or verify the 21-bit integrity tag, execute
QEMU, exercise the common guest-visible device contract, or independently
reimplement the V12 device.  Its finite-evidence result constructs a symbolic
family of nonexact countermodels approaching the ideal map; it is not a
numerical run, fitted extrapolation, or machine-derived convergence theorem.
The runner does not import the V12 C source, but its protocol reference logic
is likewise not a wholly separate V12 device.  Those limits remain inside the
claim ceiling and prevent any broader independence or promotion statement.

## Resource ledger

The response resource digest covers query applications, full-state return
checks, environment operations, sealed control count, preparation and
certification operations, logical queries, all physical sentinels, custody and
discarded-trial counters, adapter authentication/dispatch/completion/cancel
counters, virtual time/deadline/state, allocation dimensions, precision,
logical residue-code capacity, and compiler/controller/construction
coordinates.

| Coordinate | External model classification |
| --- | --- |
| Density cells | `9216` |
| Independent scratch cells | `9216` |
| Coefficient representation | two signed 64-bit `Z[omega]` coefficients |
| Peak floor | resident device object plus largest named stack local, the 36-cell retained marginal; not whole-process peak/RSS |
| Envelope allocation | two 64-bit words resident in device/private storage |
| Residue code capacity | 4 bits = `ceil(log2(9))`; not Shannon or min entropy |
| Successful logical queries | 2 per pair fixture |
| Return/certification scan | complete 9216-cell internal check |
| Adapter ledger | accepted/rejected, dispatch, completion, cancel, tick, deadline |
| Physical photon number | unknown sentinel |
| Physical loss/dephasing | unknown sentinel after external return |
| Physical duration/bandwidth/action/energy/history/hold/maintenance | unknown; any formal model value is not a measurement |
| Compiler/construction | unknown sentinel |
| Total-resource comparison | `UNDETERMINED` |

The source peak field is an explicit floor: `sizeof(device state)` plus its
largest named stack-local allocation, the retained 36-cell marginal.  The
resident device state already includes density, independent scratch, and
envelope words.  It does not establish the whole QEMU process's allocator
high-water mark, RSS peak, or unnamed/transient allocation peak.  Allocated
private storage separately charges residues, bound flags, ready mask, arm
generation, arm nonce, armed flag, and envelope words.  The four-bit coordinate
is only fixed storage/code capacity for nine residue pairs, not an entropy
measurement.  Controller reads after response commit return the frozen
response-local count, and the independently recomputed digest and all sixteen
boundary words remain immutable after commit.

All external physical quantities remain unknown.  The formal two-query action
computed by the internal algebra is not an external physical action.  Unknown
must never be converted to zero, and no scalar ranking is justified while
construction, calibration, host work, physical energy, bandwidth, duration,
loss, and custody are absent.

## Equal-interface finding

The strongest equal-access comparator is a direct phase compiler given the
same two ternary residues.  It emits the same two diagonal software phases,
can hold its modeled outputs until one ACK, and avoids the test-envelope and
async-adapter machinery.  The comparator must still pay its secret programming,
descriptor, controller, memory, compilation, and runtime costs, but it removes
any claim that the V12 software result is a unique phase resource.

The test provider is not a black box against the qualification host.  M257
therefore remains controlling, and V12 proves no query separation, total
resource advantage, or bit replacement.

## Promotion and successor discipline

V12 may be cited only as the compiled protocol/device checkpoint described by
the canonical claim.  It may nominate the real authenticated adapter successor
but cannot itself promote a physical mechanism.

The successor must connect a real dual-rail dispersive adapter behind the same
common backend boundary; replace the 21-bit test tag with reviewed
authentication and key custody; provide bounded physical preparation,
calibration, phase/control precision, energy, duration, bandwidth, loss,
dephasing, environment, port, custody, and disposal receipts; and undergo an
independent statistical experiment.  Finite physical evidence must remain
`STATISTICAL_ONLY` or `APPROX_MODEL`, never `EXACT_FORMAL`, and must never
authorize catalytic `BEGIN_REUSE`.

Kill physical promotion on any hardware-disconnected run presented as
physical, secret-derived guest descriptor or timing, missing resource
coordinate silently encoded as zero, external exact/restoration/reuse flag,
partial boundary before completion, output release before ACK, revived
migration/reset lineage, direct-provider bypass, absent equal-interface
comparator, or M257 escape claim.

## Nonclaims

M270 does not establish:

- contact with hardware or an external physical adapter;
- a physical finite-energy eigenstate carrier, cavity mode, photon, coherent
  client/reference port, phase mask, or topological eigenphase;
- cryptographic authentication, confidentiality, authorization, secure key
  storage, access restriction, or resistance to an adversary;
- statistical physical evidence, calibrated samples, physical confidence
  bounds, or a physical noise channel;
- physical custody, same-mode identity, complete physical return,
  environment factorization, restoration, reuse, or `BEGIN_REUSE`;
- calibrated physical action, energy, precision, duration, bandwidth, loss,
  dephasing, leakage, maintenance, construction, or total resources;
- a unique query resource, query advantage, total-resource advantage,
  complexity lower bound, M257 escape, Small Wall crossing, unbounded compute,
  or replacement of a bit with physical phase;
- promotion of V2 through V10 twins or promotion by package naming alone;
- physical validity of the exact internal ideal algebra, analytic open stub,
  or hardware-disconnected external stub.
