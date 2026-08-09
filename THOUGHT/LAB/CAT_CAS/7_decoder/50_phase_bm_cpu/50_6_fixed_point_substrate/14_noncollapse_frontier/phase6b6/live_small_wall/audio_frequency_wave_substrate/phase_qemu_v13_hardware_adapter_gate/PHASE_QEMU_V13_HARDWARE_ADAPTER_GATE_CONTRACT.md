# Phase-QEMU V13 Hardware-Adapter Gate Contract

## 1. Purpose and authority

V13 is the next compiled Phase-QEMU sibling. It returns hardware promotion
work to the common QEMU device/backend architecture; it is not a standalone
mechanism twin. This milestone implements a fixed symbolic-selector model of
the negative half of the physical adapter boundary: hardware absence plus
predetermined enrollment, attestation, freshness, replay, downgrade,
trust-domain, custody, measurement, and resource-provenance failure states
must fail closed before any physical output can be published. It does not
ingest, parse, authenticate, or appraise real evidence.

The source and installation law do not self-award runtime status.  The package
admits a durable build receipt and runtime seals only through the separate
compiled build/qtest qualification stage.  Their presence is evidence only
when the strict qualifier regenerates both executions, matches the sealed
bytes, and passes every obligation below.  No package artifact is evidence
that a physical device ran.

The architectural rule is:

```text
mechanism search may occur outside QEMU
-> nominated mechanism returns to the common Phase-QEMU PCI/backend contract
-> compiled device and boundary controls are independently exercised
-> only then may architecture promotion be considered
```

V2 through V10 do not qualify merely by living under the Phase-QEMU research
directory. V0, V1, V11, V12, and this V13 source are the compiled-device line.
V13 preserves the common V11/V12 guest-visible register prefix and appends its
typed gate interface; it does not fork a new Python-only architecture.

## 2. Exact claim authority

The exact bounded claim target is:

```text
COMPILED_COMMON_PHASE_QEMU_V13_FIXED_OFFLINE_HARDWARE_ADAPTER_SELECTOR_GATE_REJECTS_ABSENT_UNENROLLED_UNATTESTED_STALE_REPLAYED_DOWNGRADED_AND_TEST_FIXTURE_SESSIONS_WITHOUT_PUBLISHING_A_PHYSICAL_OUTPUT_OR_CAMPAIGN_STATISTICAL_CERTIFICATE_AND_WITH_EVERY_NONDESTRUCTIVELY_COMPLETED_DISPATCHED_ATTEMPT_TERMINAL_ACK_THEN_SPENT
```

The exact claim ceiling is:

```text
HARDWARE_ABSENT_PROTOCOL_CONFORMANCE_ONLY_NO_AUTHENTICATED_LIVE_DEVICE_SESSION_NO_PHYSICAL_SAMPLE_NO_CAMPAIGN_STATISTICAL_CERTIFICATE_NO_CUSTODY_RETURN_RESTORATION_REUSE_ADVANTAGE_OR_M257_ESCAPE
```

Restoration classification:

```text
NO_RESTORATION_CLAIM
```

Exact scope:

```text
COMPILED_QEMU_10_2_4_PHASE_QEMU_V13_COMMON_PCI_BACKEND_HARDWARE_ABSENCE_AND_FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY
```

Exact disposition:

```text
V13_ESTABLISHES_COMMON_BACKEND_FAIL_CLOSED_FIXED_SELECTOR_STATE_MACHINE_CONFORMANCE_FOR_DEVICE_ENROLLMENT_ATTESTATION_REPLAY_DOWNGRADE_AND_FIXTURE_DOMAIN_SEPARATION_WITHOUT_A_LIVE_HARDWARE_SESSION_OR_PHYSICAL_OUTPUT_DIRECT_EQUAL_ACCESS_PROTOCOL_COMPARATOR_CONTROLS_AND_M257_REMAINS_INTACT
```

Exact successor:

```text
USER_AUTHORIZED_PINNED_DEVICE_ENROLLMENT_FOLLOWED_BY_A_PREREGISTERED_BLINDED_DUAL_RAIL_DISPERSIVE_CAPTURE_CAMPAIGN_WITH_DEVICE_SIGNED_RAW_MANIFESTS_INDEPENDENT_MEASUREMENT_AND_FAMILYWISE_STATISTICAL_VALIDATION_BEHIND_THE_COMMON_PHASE_QEMU_BACKEND
```

None of these strings authorizes physical connection, device enrollment,
capture, external account activity, purchasing, USB/PCI changes, or VM GUI
operation. The successor remains a user-authorized future campaign target.

## 3. Compiled-device identity

The device source defines:

| Field | Required value |
|---|---:|
| QOM type | `phase-qemu-v13` |
| PCI vendor | QEMU `0x1234` |
| PCI device | `0x11fd` |
| revision | `0x01` |
| magic | `PH13` / `0x50483133` |
| ABI | `0x00030000` |
| BAR0 | 4 KiB little-endian MMIO |
| production backend | `0x0d80`, hardware absent |
| fixture backend | `0x0df0`, explicit test-only enable |

The production hardware backend is the default and its lease preflight always
returns `ERR_HARDWARE_ABSENT` in V13. There is no property, callback, or
internal model that can turn it into a live backend. A future milestone must
replace that backend deliberately and requalify the device; it must not use a
runtime switch that silently upgrades this artifact.

The V13 anti-substitution constants are all zero:

```text
PHASE_V13_INTERNAL_IDEAL_SUBSTITUTION_ALLOWED
PHASE_V13_LIVE_HARDWARE_AVAILABLE
PHASE_V13_PHYSICAL_OUTPUT_ALLOWED
PHASE_V13_CAMPAIGN_STATISTICAL_CLASS_ALLOWED
```

The hardware path contains no exact `Z(omega)` engine, phase compiler, stored
answer, expected residue, analytic approximation, or call into the V11/V12
software model. Failure of the physical backend cannot select an ideal or open
software backend.

## 4. Frozen common register prefix

`PhaseV13Register` must contain every named `PhaseV12Register` entry at the
same offset, including the 16-word locked boundary at `0x200..0x278`. The
installer compares the two enum maps directly and rejects any missing,
renamed, or moved V12 register.

V13 extensions begin at `0x280`:

| Offset | Width | Register |
|---:|---:|---|
| `0x280` | 32 | gate state |
| `0x284` | 32 | evidence origin |
| `0x288` | 32 | channel security |
| `0x28c` | 32 | device appraisal |
| `0x290` | 32 | measurement class |
| `0x294` | 32 | custody provenance |
| `0x298` | 32 | resource provenance |
| `0x29c` | 32 | trust domain |
| `0x2a0` | 32 | rejection reason |
| `0x2a4` | 32 | gate flags |
| `0x2a8` | 32 | observed security version |
| `0x2ac` | 32 | minimum security version |
| `0x2b0` | 64 | fixture/session epoch label |
| `0x2b8` | 64 | attestation nonce label |
| `0x2c0` | 64 | attestation age ticks |
| `0x2c8` | 64 | maximum attestation age ticks |
| `0x2d0..0x308` | 64 | typed provenance digests |
| `0x310..0x338` | 64 | dispatch/failure/ACK/reject counters |
| `0x340` | 64 | monotonic gate tick |
| `0x348` | 64 | lease expiry tick |

Unknown, wrong-width, unaligned, boundary-write, and read-only writes set
`ERR_BAD_ARGUMENT`. The boundary remains locked until a complete terminal
failure receipt is committed. Reading a locked boundary returns all ones.
Both `MemoryRegionOps.valid` and `MemoryRegionOps.impl` admit 1 through 8 byte
and unaligned accesses into the callbacks; the callbacks then reject every
shape not explicitly assigned to the addressed register. QEMU therefore does
not consume an invalid access before the device can latch `ERR_BAD_ARGUMENT`,
and the rejection does not mutate lifecycle state.

## 5. Typed evidence law

The gate never treats an untyped byte string as proof. A committed failure
receipt carries all of these independent types:

### 5.1 Evidence origin

```text
NONE
LIVE_DEVICE
OFFLINE_STANDARD_VECTOR
```

Only `OFFLINE_STANDARD_VECTOR` is constructible in the V13 fixture path.
`LIVE_DEVICE` is reserved for a future live adapter and is never assigned.

### 5.2 Channel security

```text
NONE
AUTHENTICATED_CONFIDENTIAL
OFFLINE_VECTOR_INTEGRITY
REJECTED
```

An offline standard vector may exercise accepted channel metadata through
`OFFLINE_VECTOR_INTEGRITY`; this is not a cryptographic channel claim.
`AUTHENTICATED_CONFIDENTIAL` is reserved and never assigned by V13.

### 5.3 Device appraisal

```text
NONE
ENROLLED_ATTESTED
UNENROLLED
UNATTESTED
STALE
REPLAYED
DOWNGRADED
FIXTURE_STANDARD_ACCEPTED
```

`FIXTURE_STANDARD_ACCEPTED` denotes only the predetermined accepted-metadata
selector state. No evidence parser or appraisal algorithm produced it. It is
intentionally distinct from `ENROLLED_ATTESTED` and cannot be cast or promoted
into it.

### 5.4 Measurement class

```text
NONE
PHYSICAL_SAMPLE
CAMPAIGN_STATISTICAL_CERTIFICATE
PROTOCOL_CONFORMANCE_ONLY
```

Only `PROTOCOL_CONFORMANCE_ONLY` is assigned. The source validator rejects an
assignment of either physical class anywhere in this V13 implementation.

### 5.5 Custody provenance

```text
NONE
LIVE_DEVICE_SIGNED
OFFLINE_VECTOR
REJECTED
```

V13 fixtures use `OFFLINE_VECTOR`. They cannot assert live device custody,
carrier return, reference return, restoration, or reuse.

### 5.6 Resource provenance

```text
NONE
LIVE_DEVICE_MANIFEST
OFFLINE_VECTOR
UNKNOWN
```

V13 fixtures use `OFFLINE_VECTOR`. Physical duration, bandwidth, energy,
loss, dephasing, carrier photon number, maintenance, construction, and output
hold resources remain `UINT64_MAX` (`UNKNOWN`). No offline fixture value may
be substituted for a measured physical resource manifest.

### 5.7 Trust domain

```text
NONE
PRODUCTION
TEST_FIXTURE
```

The production value is reserved but never assigned. The fixture dispatch
sets `TEST_FIXTURE` unconditionally, and the terminal sealer revalidates the
entire fixture tuple. If any field is inconsistent, it clears the tuple,
reconstructs the offline fixture types, and commits
`ERR_EVIDENCE_TYPE_MISMATCH`.

This is structural separation, not a convention: the fixture backend has no
code path to a production trust assignment, physical measurement class, live
custody value, or live resource manifest.

## 6. Backend and lease law

The common backend interface has four lifecycle callbacks:

```text
lease_preflight
dispatch
cancel
sanitize
```

The `HARDWARE` backend is present as an architectural slot but unavailable:

```text
LEASE -> hardware_absent_lease_preflight -> ERR_HARDWARE_ABSENT
```

This is a pre-dispatch rejection. It creates no allocation, session, physical
output, statistical record, or boundary receipt.

The `OFFLINE_FIXTURE` backend is realizable only when both conditions hold:

```text
backend-id = 0x0df0
test-fixture-enabled = true
```

The inverse relation is also enforced: enabling fixtures with the hardware
backend is a realization error. The selected standard-vector ID is immutable
QOM configuration fixed before realization. No guest write can select a new
fixture after a lease begins.

A successful fixture lease binds nonzero owner and program tags, creates a
software allocation label, and sets a 16-tick lease expiry. All subsequent
commands require exact owner, program, and generation match plus a fresh
lease. Expiry before dispatch is a pre-dispatch failure; it does not fabricate
a terminal hardware receipt.

## 7. Fixed offline symbolic selectors

The C device provides exactly seven predetermined fixture failure selectors:

| ID | Typed appraisal/result |
|---:|---|
| 0 | accepted fixture channel/appraisal metadata, then `ERR_FIXTURE_TRUST_DOMAIN` |
| 1 | `UNENROLLED` / `ERR_DEVICE_UNENROLLED` |
| 2 | `UNATTESTED` / `ERR_DEVICE_UNATTESTED` |
| 3 | `STALE` / `ERR_ATTESTATION_STALE` |
| 4 | `REPLAYED` / `ERR_ATTESTATION_REPLAYED` |
| 5 | `DOWNGRADED` / `ERR_SECURITY_DOWNGRADE` |
| 6 | rejected type tuple / `ERR_EVIDENCE_TYPE_MISMATCH` |

The accepted-metadata selector deliberately ends in fixture-domain rejection.
It exercises a selected symbolic channel/appraisal state without thereby
becoming production evidence. The C device does not parse an encoded vector,
verify a signature, validate a certificate, perform cryptographic channel
authentication, or appraise evidence supplied by a real device.

Fixture digests, epoch labels, and nonce labels are deterministic protocol
vectors. They are not cryptographic signatures, device-generated statements,
secrets, unpredictable challenges, physical observations, or custody records.

## 8. Dispatch, receipt, ACK, and spent law

The ordinary fixture lifecycle is:

```text
EMPTY
-> LEASED
-> PREPARED
-> ISOLATED
-> SEALED
-> PRIVATE_READY (attestation gate armed; no secret injected)
-> EXECUTING (dispatch counter commits first)
-> RESPONSE_READY with RETURN_FAILED
-> ACK_RESPONSE
-> SPENT
```

Before the dispatch counter increments, the device proves capacity for every
counter and receipt serial needed to complete the terminal transition. Every
nondestructively completed ordinary fixture dispatch is forced through
`seal_terminal_failure`; a zero result is converted to
`ERR_NO_PHYSICAL_OUTPUT`. There is no success-return branch. Destructive reset,
migration, or unrealize is not an ACK path and instead sanitizes into `SHAM`.
The compiled gate has no live hardware-disconnect input; the separate symbolic
reference classifies disconnect as the same destructive `SHAM`/no-ACK outcome.

The 16-word committed boundary contains only:

- a V13 failure-receipt header and receipt ID;
- generation and rejection reason;
- packed typed evidence classes and gate flags;
- device, measurement, custody, and resource-provenance digests;
- the protocol resource digest; and
- a final commit cookie.

It contains no phase result, amplitude, residue, physical sample, raw capture,
campaign statistic, secret, key, or reusable session handle.

While the receipt is ready:

- the boundary is readable and immutable;
- a new lease or dispatch is impossible;
- virtual tick advancement is rejected;
- `BEGIN_REUSE` is rejected; and
- only an exact tagged `ACK_RESPONSE` performs the ordinary terminal release.

ACK zeroes the boundary, releases lease tags, increments the ACK counter, and
transitions to `SPENT`. `RETURN_FAILED` never authorizes `BEGIN_REUSE`.

## 9. Reset, migration, and teardown law

A reset is canonical only before any lease or transaction activity. Reset
after lease, preparation, isolation, descriptor sealing, arming, dispatch,
held receipt, spent state, or prior sham calls the common sanitizer and enters
reset-irreversible `SHAM`.

Migration is never a live-session transport:

1. `pre_save` cancels a dispatching backend if necessary.
2. `pre_save` sanitizes descriptors, boundary, typed evidence, all provenance
   digests, epoch/nonce labels, receipts, allocation labels, lease tags, and
   request tags before bytes are serialized.
3. The source becomes `SHAM` before serialization.
4. VMState carries only PCI state, the migration marker, configured backend,
   configured fixture enable, configured vector selector, and generation.
5. `post_load` verifies those immutable configuration values and enters the
   same sanitized, reset-irreversible `SHAM`.
6. A second migration preserves sham rather than reconstructing a lease.

VMState must never include:

```text
lease or live-session state
descriptor or boundary words
typed evidence values
device/measurement/custody/resource digests
session epoch or attestation nonce
preparation/return receipts
owner/program/request tags
lease expiry
```

Unrealize invokes the same cancel/sanitize path. It does not preserve a live
session or a fixture receipt after device destruction.

These exceptional destructive transitions establish sanitation, not physical
return or restoration. Reset, migration, and unrealize enter `SHAM` without an
ACK and never claim the ordinary `ACK -> SPENT` transition. The normal
terminal contract applies only to a nondestructively completed ordinary
fixture dispatch: it seals `RETURN_FAILED`, requires explicit guest ACK, and
then enters `SPENT`.  The separate reference's symbolic disconnect enters
`SHAM` without ACK and is not evidence of compiled disconnect handling.

## 10. Installer preservation and collision law

`apply_to_qemu.py` accepts only exact QEMU `10.2.4`. It requires these frozen
predecessor source hashes:

| Device | SHA-256 |
|---|---|
| V0 | `b79ec06f870b611142f5df5c97db2f8e34027458da5acc933f90b694b2055764` |
| V1 | `8b991d8961a6e108d1a4aa7498172564b017c6e622cb8192c6fa15c33638e362` |
| V11 | `84c2ec576ae54b298046fadcda719dcb4c2e97bbe31aa0ac3c77ae5455027771` |
| V12 | `5fe4f99e9bf2ff78774e75149c532293adb03c01c035fcac3d05e2fe74b6153f` |

All four canonical Kconfig and Meson entries must occur exactly once before
installation. V13 is inserted exactly once after the V12 entries.

The installer scans QEMU `hw/**/*.c` and `hw/**/*.h` for PCI ID `0x11fd` and
QOM type `phase-qemu-v13`. Any noncanonical collision fails closed. A
nonidentical existing V13 target is never overwritten; this initial package
has no prior-revision allowlist.

Both `--check` and installation reject a symlink or nonregular QEMU source
root, `VERSION`, `hw`, `hw/misc`, `Kconfig`, `meson.build`, frozen V0/V1/V11/V12
source, existing V13 target, or V13 recovery target. A source-root argument
whose path contains a symlink component is rejected. Critical reads use
no-follow opens. Kconfig/Meson writes use no-follow file descriptors and check
the opened object is still regular before truncation. A first-install V13
target is created exclusively with no-follow semantics, so the installer never
writes through a target symlink.

Normal installation precomputes and validates both build-file results before
writing. It snapshots frozen predecessor hashes and integration counts. Any
write or post-install validation failure restores the original Kconfig and
Meson bytes. An existing V13 target is restored byte-for-byte. On a failed
first install, a partial target is moved recoverably to
`phase-qemu-v13.c.failed-install-recovery`; it is never permanently deleted.

`--check` performs all source, prefix, collision, frozen-predecessor, and
prospective-integration validation without writing.

## 11. Qualification obligations

The compiled qualification must, at minimum:

1. install into the pinned QEMU 10.2.4 source with the frozen V0/V1/V11/V12
   hashes and one canonical entry each;
2. compile the V13 device into `qemu-system-x86_64`;
3. use headless qtest/QMP only, at reduced CPU and I/O priority;
4. verify PCI identity, magic, ABI, BAR size, and the frozen prefix;
5. prove default hardware lease fails with `ERR_HARDWARE_ABSENT` and no
   dispatch or receipt;
6. exercise all seven fixed offline symbolic selectors;
7. prove the accepted-metadata selector remains `TEST_FIXTURE` and ends in
   `ERR_FIXTURE_TRUST_DOMAIN`;
8. prove every nondestructively completed ordinary fixture dispatch exposes
   one complete immutable failed receipt, requires ACK, then becomes `SPENT`
   with reuse rejected, while destructive reset/migration/unrealize sanitizes
   to `SHAM` without ACK and the separate-reference disconnect case is likewise
   classified as destructive `SHAM` without implying a compiled disconnect input;
9. prove no return class other than `NONE` or `FAILED` is assigned;
10. exercise tag, generation, expiry, wrong-width, unaligned, read-only, and
    locked-boundary negative controls;
11. exercise reset after activity, real QMP migration, second-hop migration,
    and clean teardown sanitation;
12. prove migration never reconstructs a lease, nonce, digest, receipt, typed
    evidence tuple, boundary, or production trust;
13. independently recompute failure receipts and typed resource digests rather
    than trusting device-reported parity alone; and
14. report physical resources as unknown and report zero physical samples and
    zero campaign statistical certificates.

The installer is intentionally source-integration-only and always reports
`qemu_device_implemented=0`, `common_guest_visible_contract_compiled=0`, and
`reintegration_gate_passed=0`.  Only the separately pinned build receipt and
strict runtime qualifier may assert those three compiled-architecture facts.

## 12. Scientific and architectural ceiling

V13 does not establish:

- an authenticated live device session;
- hardware enrollment or a pinned production identity;
- a confidential authenticated production channel;
- a physical dual-rail carrier, dispersive interaction, or sample;
- device-signed raw data or manifest custody;
- independent measurement;
- a preregistered or blinded capture campaign;
- a campaign-level statistical certificate;
- physical return, inverse restoration, or same-substrate reuse;
- a resource or computational advantage; or
- an escape from M257.

The offline vectors test protocol rejection structure only. Direct
equal-access protocol comparators remain controlling. M257 remains intact.
No software fixture, even one with accepted local metadata, may be counted as
a physical phase resource.

The promoted successor must remain behind the common Phase-QEMU backend. It
requires explicit user authorization for exact device enrollment and physical
capture scope, pinned production identity, a real authenticated confidential
channel, device-signed raw manifests, independent measurement, preregistration,
blinding, and familywise statistical validation. Even that future campaign
may not claim exact formal physical return or authorize reuse without separate
physical evidence.
