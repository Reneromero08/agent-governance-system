# Phase-QEMU V13 hardware-adapter gate findings

## Verdict and frozen authority

M271 passes only as a compiled common-device fixed offline hardware-adapter
selector gate exercised without hardware, together with a separate symbolic
state-machine truth table.  The C device selects one of seven predetermined
failure outcomes; it does not parse or appraise real evidence.  The result
establishes bounded fail-closed selector/state-machine and direct-comparator
behavior only.  It does not execute an encoded protocol parser, verify a real
cryptographic signature, establish that an authenticated device session
occurred, show that a physical carrier produced a sample, or show that an
independent campaign certified a statistical result.

The canonical bounded claim is:

```text
COMPILED_COMMON_PHASE_QEMU_V13_FIXED_OFFLINE_HARDWARE_ADAPTER_SELECTOR_GATE_REJECTS_ABSENT_UNENROLLED_UNATTESTED_STALE_REPLAYED_DOWNGRADED_AND_TEST_FIXTURE_SESSIONS_WITHOUT_PUBLISHING_A_PHYSICAL_OUTPUT_OR_CAMPAIGN_STATISTICAL_CERTIFICATE_AND_WITH_EVERY_NONDESTRUCTIVELY_COMPLETED_DISPATCHED_ATTEMPT_TERMINAL_ACK_THEN_SPENT
```

Claim ceiling:

```text
HARDWARE_ABSENT_PROTOCOL_CONFORMANCE_ONLY_NO_AUTHENTICATED_LIVE_DEVICE_SESSION_NO_PHYSICAL_SAMPLE_NO_CAMPAIGN_STATISTICAL_CERTIFICATE_NO_CUSTODY_RETURN_RESTORATION_REUSE_ADVANTAGE_OR_M257_ESCAPE
```

Restoration classification:

```text
NO_RESTORATION_CLAIM
```

Scope:

```text
COMPILED_QEMU_10_2_4_PHASE_QEMU_V13_COMMON_PCI_BACKEND_HARDWARE_ABSENCE_AND_FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY
```

Disposition:

```text
V13_ESTABLISHES_COMMON_BACKEND_FAIL_CLOSED_FIXED_SELECTOR_STATE_MACHINE_CONFORMANCE_FOR_DEVICE_ENROLLMENT_ATTESTATION_REPLAY_DOWNGRADE_AND_FIXTURE_DOMAIN_SEPARATION_WITHOUT_A_LIVE_HARDWARE_SESSION_OR_PHYSICAL_OUTPUT_DIRECT_EQUAL_ACCESS_PROTOCOL_COMPARATOR_CONTROLS_AND_M257_REMAINS_INTACT
```

Successor:

```text
USER_AUTHORIZED_PINNED_DEVICE_ENROLLMENT_FOLLOWED_BY_A_PREREGISTERED_BLINDED_DUAL_RAIL_DISPERSIVE_CAPTURE_CAMPAIGN_WITH_DEVICE_SIGNED_RAW_MANIFESTS_INDEPENDENT_MEASUREMENT_AND_FAMILYWISE_STATISTICAL_VALIDATION_BEHIND_THE_COMMON_PHASE_QEMU_BACKEND
```

The controlling M257 guardrail remains:

```text
EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_NOT_BE_COUNTED_AS_A_PHASE_RESOURCE
```

No volatile artifact hash is copied into this narrative.  Source identities
belong in the build receipt and strict qualifier so one mechanically checked
ledger, rather than prose, controls the evidence set.

## What the result does and does not say

The positive result is narrow: the compiled V13 common PCI/backend boundary
uses a fixed offline selector to choose one of seven predetermined failures:
absent, unenrolled, unattested, stale, replayed, downgraded, or test fixture.
The C device does not derive those outcomes by parsing a request, certificate,
attestation, signature, nonce, replay cache, protocol version, firmware image,
or fixture credential.  Every nondestructively completed dispatched attempt
closes through a terminal receipt, acknowledgement, and `SPENT`.  Reset,
migration, and unrealize instead burn compiled lineage to `SHAM` without ACK;
the separate reference likewise classifies symbolic disconnect as destructive
`SHAM`, not as an ordinary acknowledged failure.

The separate reference exercises symbolic Python dictionaries,
presence/validity booleans, integer freshness fields, and string domains.  It
compares the resulting failure codes and lifecycle fields against a frozen
truth table.  It does not parse TLS, EAT, COSE, TPM, a device certificate, a
signed manifest, or any other encoded cryptographic object.

The hardware-absence result is the scientifically important result of M271.
There was no external adapter, no mutually authenticated live channel, no TPM
quote from a measured device, no fresh physical raw-sample manifest, and no
independent statistical campaign certificate.  Consequently V13 publishes no
physical output.  An unavailable device is not silently replaced by a model,
an old sample, a test vector, or a claimed physical success.

| Question | M271 answer |
| --- | --- |
| Compiled common Phase-QEMU device | yes, V13 in QEMU 10.2.4 |
| Hardware present | no |
| Fixed offline selector/state-machine scope | seven predetermined C failure choices plus a separate symbolic fixture/direct comparator; no encoded parser or cryptographic verification |
| Live device enrolled and authenticated | no |
| Fresh device attestation | no |
| Physical raw sample | none |
| Independent campaign certificate | none |
| Physical result published | no |
| Terminal lifecycle | ordinary nondestructively completed dispatched attempts require terminal receipt, ACK, then `SPENT`; reset/migration/unrealize and the reference's symbolic disconnect enter `SHAM` without ACK |
| Restoration or physical return | `NO_RESTORATION_CLAIM` |
| Reuse authority | none |
| Resource advantage | none |
| M257 escape | none |

## Architecture discipline

The Phase-QEMU name denotes an architectural promotion boundary, not merely a
directory for related experiments:

- V0 and V1 are compiled QEMU devices.
- V2 through V10 are standalone mechanism-search and digital-twin
  experiments.  They can falsify, compare, or nominate a mechanism, but they
  cannot promote one.
- V11 returns the surviving exact mechanism to a compiled guest-visible PCI
  device with a common swappable backend.
- V12 extends that compiled common backend with the asynchronous adapter
  lifecycle while remaining hardware-disconnected and noncryptographic.
- V13 is another compiled common-device step.  It places a fixed offline
  seven-choice selector and modeled state-machine boundaries behind the shared
  Phase-QEMU backend.  It does not yet place a real evidence parser or
  appraiser there.

The promotion invariant is therefore explicit:

```text
mechanism search may happen outside QEMU
-> nomination does not imply promotion
-> a promoted mechanism returns to the common compiled device/backend
-> physical authority additionally requires authenticated raw evidence and an independent campaign
```

V13 satisfies the compiled-device half of that invariant.  It does not satisfy
the physical-evidence half.  A later backend implementation must not move
device policy into an unrelated Python twin or expose a second guest ABI that
bypasses the common lifecycle.

## V12 ideal fallback is forbidden on the physical path

The V12 ideal backend remains useful as a software oracle, but it is not a
recovery path for a failed V13 physical session.  None of the current compiled
selector's seven failure outcomes, nor any of the separate reference's 17
symbolic coordinates, may fall back to V12.  In the future encoded physical
route, the same rule must also cover live transport and cryptographic failures.
Once a request selects the physical backend, all of the following are
forbidden:

- falling through to the V12 ideal or test backend after absence, enrollment,
  attestation, TLS, freshness, replay, downgrade, timeout, or device failure;
- relabeling an `APPROX_MODEL`, cached response, public test vector, or direct
  equal-access computation as a physical sample;
- retrying in a less restrictive protocol version or trust profile;
- publishing a modeled phase while leaving the physical-evidence bit, receipt
  type, or campaign status ambiguous; and
- using an exact software answer to authorize physical return, reuse, or
  advantage.

Failure on the physical route stays on the physical route.  An ordinary
nondestructively completed dispatched attempt becomes a sealed terminal
failure and reaches `SPENT` only after guest acknowledgement.  Reset,
migration, or unrealize instead irreversibly burns compiled lineage to `SHAM`
without ACK.  The reference applies the same destructive classification to a
symbolic disconnect; V13 has no live hardware-disconnect input.  This prevents
availability pressure from converting a missing instrument into fabricated
physical evidence.

## Typed evidence and trust-domain separation

The authority model distinguishes evidence by origin rather than by whether a
symbolic signature-valid field is true:

| Evidence type | What it may establish | What it may never establish |
| --- | --- | --- |
| `EXACT_FORMAL` | an exact internal software law under its declared model | a physical observation or physical return |
| `APPROX_MODEL` | a bounded model or digital-twin response | hardware presence, custody, or measured performance |
| symbolic offline fixture conformance | the frozen field/appraisal truth table and direct protocol-comparator behavior | encoded parsing, cryptographic verification, possession of a deployed device key, a live session, or a physical sample |
| future authenticated live-device evidence | identity- and freshness-bound device statements after enrollment and attestation | by itself, a campaign-level physical conclusion |
| future transaction raw-sample receipt | the exact signed bytes and metadata returned for one physical attempt | population inference, reuse, restoration, or advantage |
| future independent campaign certificate | a preregistered statistical adjudication over an enumerated manifest set | `EXACT_FORMAL` physical return or automatic reuse authority |

The exact scope token says
`HARDWARE_ABSENCE_AND_FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY`.
The compiled portion is a seven-choice selector, not an evidence parser.  The
separate reference fixtures label their signature encoding
`SYMBOLIC_STANDARD_VECTOR_SIGNATURE` and model signature validity,
transcript-hash equality, sequence validity, calibration schema validity, and
resource schema validity as booleans.  There are no key bytes, certificates,
encoded claims, signature operations, raw manifests, or real parser acceptance
tests.  Passing proves only the fixed selector/state-machine and symbolic
field-to-comparator transformations.

The current fixture-domain control compares one credential-domain string with
the frozen fixture and production strings.  It does not exercise an issuer
chain, a distinct device-ID namespace, session-key derivation, manifest
binding, attestation policy, algorithm negotiation, or schema-version parser.
Those are requirements of the future live-device contract below, not current
findings.  In that future system, standard-vector roots, keys, device
identifiers, issuers, and policy domains must be disjoint from production, and
no configuration option may promote a fixture receipt in place.

This distinction follows the separation in the IETF RATS architecture between
Attester, Verifier, Relying Party, Evidence, Reference Values, and Attestation
Results.  A relying party still needs explicit appraisal policy; syntactically
valid Evidence is not self-authorizing ([RFC 9334](https://www.rfc-editor.org/rfc/rfc9334.html)).
Entity Attestation Token claims and COSE protection provide interoperable
containers, not proof that the claimed physical phenomenon occurred
([RFC 9711](https://www.rfc-editor.org/rfc/rfc9711.html),
[RFC 9052](https://www.rfc-editor.org/rfc/rfc9052.html)).

## Fail-closed gate finding

The compiled C device exposes seven predetermined selector failures.  It does
not inspect the evidence predicates suggested by their names.  Independently,
the separate reference freezes exactly 17 symbolic negative coordinates.  The
coordinates all produce no physical output and no campaign certificate; they
do not imply real controls beyond the listed symbolic fields.

| Symbolic coordinate | Frozen field condition | Dispatch state |
| --- | --- | --- |
| absent | `device_present = false` | pre-dispatch rejection |
| unenrolled | `enrollment_record_present = false` | pre-dispatch rejection |
| unattested | attestation-present/appraisal-valid booleans false | pre-dispatch rejection |
| stale nonce | `session_nonce <= last_accepted_nonce` | pre-dispatch rejection |
| replay cache | attempt already present in the replay-cache boolean | pre-dispatch rejection |
| protocol downgrade | integer protocol version below 13 | pre-dispatch rejection |
| firmware downgrade | integer firmware epoch below 7 | pre-dispatch rejection |
| fixture credential domain | fixture-domain string used in production | pre-dispatch rejection |
| bad device signature | signature-valid boolean false | pre-dispatch rejection |
| bad transcript hash | transcript-hash-match boolean false | pre-dispatch rejection |
| bad sequence | integer sequence differs from expected sequence | pre-dispatch rejection |
| bad calibration schema | calibration-schema-valid boolean false | pre-dispatch rejection |
| bad resource schema | resource-schema-valid boolean false | pre-dispatch rejection |
| timeout | symbolic transport event `TIMEOUT` | dispatched terminal failure |
| reset | symbolic transport event `RESET` | destructive lineage exit: compiled state `SHAM`, no ACK |
| migration | symbolic transport event `MIGRATION` | destructive lineage exit: compiled state `SHAM`, no ACK |
| disconnect | symbolic transport event `DISCONNECT` | destructive lineage exit: symbolic state `SHAM`, no ACK; no compiled live-disconnect input exists |

For ordinary nondestructive completion, the symbolic failure receipt binds the
modeled attempt/device IDs, failure code, dispatch boolean, version/epoch,
nonce/sequence, boolean appraisal statuses, `FAILED`, ACK-required, and
terminal `SPENT` fields.  It contains no phase output presented as physical
data.  The compiled lifecycle requires ACK before such a dispatched terminal
failure becomes `SPENT`; ACK does not turn failure into reusable state.  Reset,
migration, and unrealize bypass ACK and irreversibly enter compiled `SHAM`;
the reference's symbolic disconnect does the same without claiming compiled
disconnect coverage.  Rich replay protection for real sessions, manifests,
nonces, and attestation results belongs to the future contract.

TLS 1.3 is an appropriate future secure-channel layer because it authenticates
the server, can authenticate the client, protects record integrity, binds the
negotiated transcript, and includes downgrade protections.  It is not the
physical-evidence layer, and TLS 0-RTT must not carry physical dispatches
because its replay properties are unsuitable for one-shot transactions
([RFC 8446](https://www.rfc-editor.org/rfc/rfc8446.html)).

## Preregistered future live-device contract

The following is a contract for a future user-authorized campaign, not a claim
about M271 execution.

### Enrollment and live-session prerequisites

Before one physical dispatch, freeze and sign an enrollment record containing:

1. exact device and board identities, hardware revision, secure-element or TPM
   identity, manufacturer chain, allowed public keys, and revocation state;
2. measured boot/reference values, firmware, adapter protocol, schema,
   analysis, and policy versions;
3. exact BOM identities, circuit/netlist revision, calibration certificates,
   calibration interval, instrument firmware, and environmental-sensor map;
4. allowed TLS 1.3 profile with mutual authentication, 0-RTT disabled, explicit
   downgrade refusal, and a transcript-bound campaign/session identifier; and
5. a fixture-domain exclusion list that can never share production trust roots.

For each session, a fresh challenge must bind campaign ID, transaction ID,
guest allocation/generation, backend identity, nonce, requested operation,
descriptor digest, protocol/schema versions, deadline, and expected rails.
The proposed evidence path is a TPM 2.0 quote or equivalent measured-device
root, appraised using the RATS roles, carried in an EAT and signed in an
appropriate COSE structure.  The verifier's signed Attestation Result and the
relying-party policy decision must both be preserved.  The TPM library defines
the proposed measured-device primitive; merely naming TPM does not enroll or
attest anything ([TCG TPM 2.0 Library](https://trustedcomputinggroup.org/resource/tpm-library-specification/)).

### Per-transaction authenticated raw samples only

A live adapter must return the raw capture, not just an accepted phase or a
device-generated verdict.  Each immutable device-signed manifest must bind at
least:

- campaign, session, transaction, allocation, generation, and monotonic
  sequence identifiers;
- challenge nonce, TLS transcript/channel binding, device identity,
  attestation-result digest, firmware and measured-boot values;
- request/descriptor digest, preparation commands, rail labels, exact sample
  count, channel map, sample encoding, rate, trigger, clock, gain, range, and
  timestamps;
- cryptographic digests of every raw payload block in byte order;
- source-monitor, witness, empty/dummy/control, environment, supply, and
  calibration streams needed by the preregistered adjudicator;
- dispatch, acquisition, completion, cancellation, and timeout states;
- exclusions or device faults without rewriting the original bytes; and
- manifest schema, signer, signature algorithm, certificate chain, and signing
  time evidence.

The QEMU backend must verify the manifest, sequence, challenge, declared
lengths, payload-block digests, and transaction binding before exposing any
typed sample receipt.  Summary-only responses, reconstructed arrays, unsigned
side files, missing blocks, or post hoc metadata are terminal failures.  One
valid transaction receipt is still not a campaign certificate.

### Independent campaign certificate

Campaign inference must occur in a trust domain separate from device capture
and from the mechanism's implementation team.  The independent adjudicator
receives the blinded, device-signed manifest inventory and the preregistered
analysis, verifies all inclusions and exclusions, replays analysis from raw
bytes, and signs a certificate that binds:

- the complete manifest-set root and explicit missing/invalid inventory;
- enrollment, attestation, calibration, software, and protocol identities;
- randomization and blinding assignments revealed only after data lock;
- endpoint definitions, estimators, equivalence margins, significance levels,
  multiplicity family, stopping rule, and exclusion policy;
- estimates, uncertainty budgets, confidence intervals, exact counts,
  adjusted decisions, negative controls, and protocol deviations; and
- the exact conclusion class, which can be at most bounded statistical
  physical evidence and never `EXACT_FORMAL` physical return.

The capture service cannot issue this certificate for itself.  A device
signature establishes byte origin under the enrolled key; it does not perform
independent statistical adjudication.

## Preregistered analysis plan

The physical campaign must freeze its hypotheses, sample size, stopping rule,
randomization, blinding, controls, estimators, margins, and exclusion rules
before the first outcome-bearing capture.  At minimum:

- Use circular phase statistics on the unit circle.  Freeze the wrapped-error
  convention, mean direction/resultant estimator, degeneracy rule, confidence
  construction, and branch-cut handling.  Never choose an unwrap or rotate the
  origin after examining group outcomes.
- Use two one-sided tests (TOST) for any equivalence claim, with a
  scientifically justified equivalence margin fixed before collection.  A
  failure to reject difference is not evidence of equivalence
  ([Schuirmann's TOST paper](https://pubmed.ncbi.nlm.nih.gov/3450848/)).
- Use exact binomial intervals/tests for finite pass/fail, sign, or control
  counts where the binomial model was preregistered; do not substitute a
  favorable normal approximation after inspection
  ([NIST exact binomial confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)).
- Define one family across ordered phase pairs, rails, endpoints, controls,
  batches, and interim looks.  Apply the preregistered familywise correction,
  with Bonferroni as a conservative valid baseline, and publish both raw and
  adjusted decisions
  ([NIST Bonferroni method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm)).
- Carry Type A and Type B uncertainty, correlations, calibration uncertainty,
  quantization, timebase/phase reference, loading, drift, environment, and
  model sensitivity into a declared combined and expanded uncertainty budget.
  Report the result, combined standard uncertainty, coverage factor, expanded
  uncertainty, and how each was obtained
  ([NIST TN 1297](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-7-reporting-uncertainty),
  [JCGM 100:2008 GUM](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf)).
- Preserve an auditable provenance graph for each raw entity, acquisition and
  analysis activity, responsible agent, derivation, revision, and certificate.
  W3C PROV provides the proposed vocabulary; provenance records do not replace
  signatures or calibration
  ([W3C PROV Primer](https://www.w3.org/TR/prov-primer/)).

Metrological traceability requires a documented, unbroken calibration chain
with stated uncertainties; a vendor name or an instrument's presence is not
traceability by itself
([NIST traceability policy](https://www.nist.gov/calibrations/traceability)).

## Proposed physical resources and comparators

The research reports nominate a plausible bench topology, but it is a proposed
hypothesis until exact parts are procured, inspected, assembled, calibrated,
enrolled, and measured.  The candidate BOM includes an Epson FC-135 32.768 kHz
quartz unit as carrier, an Analog Devices ADG1419 switching element, a Texas
Instruments OPA810 front end, and an NI USB-6366-class acquisition instrument.
The primary vendor materials establish candidate identities and specifications,
not that this apparatus exists or works
([Epson FC-135 datasheet](https://download.epsondevice.com/td/pdf/td_xtal_32khz/FC-135_Q13FC13500004_en.pdf),
[ADG1419 product page](https://www.analog.com/en/products/adg1419.html),
[OPA810 product page](https://www.ti.com/product/OPA810),
[NI USB-6366 product page](https://www.ni.com/en/shop/hardware/voltage/model-usb-6366)).

The eventual resource ledger must include, rather than hide:

- all resonators, switches, amplifiers, relays, terminations, sources,
  digitizers, clocks, controllers, TPM/secure element, sensors, cabling,
  shielding, fixtures, and power supplies;
- calibration, warm-up, preparation, failed/rejected attempts, control arms,
  network and cryptographic overhead, capture storage, analysis, and human or
  automated adjudication time;
- wall time, energy, peak memory/storage, samples, bandwidth, retries, device
  wear, consumables, and uncertainty contributions; and
- loss, dephasing, loading, drift, temperature, vibration, feedthrough,
  switching transient, timing, and disposal/custody states.

All such physical quantities are unknown in M271.  No zero may be inferred
from an absent sensor or absent device.

The comparator family must be fixed before collection and receive equal
information and accounting.  It includes the direct equal-access phase
compiler/protocol shadow required by M257, the best ordinary digital method at
the same interface, and matched physical nulls such as resonator removed,
matched dummy capacitance, zero drive, source left on, source muted rather than
isolated, wrong termination, gate-only/relay-only ablations, wrong phase,
wrong frequency, and timing perturbations.  Favorable hardware output without
these controls cannot support mechanism specificity or advantage.

## Claim exclusions

M271 establishes none of the following:

- hardware possession, connection, enrollment, or a live authenticated session;
- a coherent physical carrier, physical phase sample, or statistical physical
  evidence;
- physical custody, same-backing return, restoration, or reuse;
- a campaign certificate, metrological traceability, or independent
  replication;
- a resource reduction, speedup, energetic advantage, asymptotic advantage,
  or Small Wall crossing; or
- an escape from M257's exact deterministic equal-access forward shadow.

Ordinary nondestructive terminal `ACK -> SPENT` is controller lifecycle
closure, not carrier return.  Reset, migration, and unrealize instead produce
`SHAM` without ACK.  The word “authenticated” must always say what was
authenticated: a symbolic fixture, a transport peer, an attestation claim, a
raw manifest, or an independent campaign certificate.  None may silently
stand in for another.

## Decision

V13 is promoted as a compiled fixed offline selector gate because it returns
the candidate hardware mechanism to the common Phase-QEMU device/backend
architecture and fails closed under hardware absence.  It is not promoted as
an evidence parser, cryptographic verifier, or live adapter.  Physical
promotion is deferred.  The only admissible next step is the frozen successor
token above: explicit user authorization, pinned production enrollment, and
then a preregistered blinded campaign whose device-signed raw manifests and
independent familywise statistical certificate remain behind the same common
backend.
