# Phase-QEMU V13 pre-enrollment campaign contract

## Scope

This package is authorized software preparation below the M272 physical
boundary.  It converts part of the V13 future campaign prose into executable,
encoded, fail-closed offline fixtures.  It does **not** enroll or connect a
device, open a transport, issue or parse a production credential, perform a
capture, authenticate a physical sample, or issue a campaign statistical
certificate.

The classification is exactly
`PRE_ENROLLMENT_SOFTWARE_CONFORMANCE_PREPARATION_OUTSIDE_PHYSICAL_EVIDENCE`.
The authority gate remains
`EXPLICIT_USER_AUTHORIZATION_REQUIRED_BEFORE_DEVICE_ENROLLMENT_CONNECTION_OR_CAPTURE`.
This is not M272 and does not change the M271 authority record.

## Encoded-object profile

The production program implements a deliberately small deterministic CBOR
profile: integers, byte and UTF-8 strings, finite arrays, finite maps,
Booleans, and null.  It rejects tags, floats, indefinite items, non-shortest
arguments, duplicate or unsorted keys, invalid UTF-8, and trailing bytes.
Map keys are ordered by their deterministic encoded bytes.

Offline fixtures use untagged COSE_Sign1 with Ed25519 (`alg=-8`), a protected
`kid`, an empty unprotected map, and a domain-separated external AAD.  The
signed structure is `['Signature1', protected, external_aad, payload]`.
Every fixture private key is deterministically derived from a plainly marked
offline-only seed.  No production key, certificate, trust anchor, or secret
is present.

These rules implement narrow profiles of [RFC 8949](https://www.rfc-editor.org/rfc/rfc8949.html)
and [RFC 9052](https://www.rfc-editor.org/rfc/rfc9052.html).  RATS roles from
[RFC 9334](https://www.rfc-editor.org/rfc/rfc9334.html) shape future role
separation only.  [RFC 9711](https://www.rfc-editor.org/info/rfc9711/) is a
future EAT profile target; no EAT is issued or parsed here.

## Offline roles and bindings

Four distinct fixture key pairs represent the enrollment authority, device,
preregistration owner, and independent adjudicator.  The signed enrollment
record binds one offline device public-key digest while explicitly setting
`production_authorization=false` and excluding fixture domains.

Each raw manifest binds the campaign, session, transaction, allocation,
generation, nonce, descriptor digest, transport-placeholder digest, firmware
digest, monotonic sequence, block index, block length, and block SHA-256.
The replay state is keyed by session and sequence.  Missing blocks, a changed
challenge, a replay, a wrong signature domain, and a changed signature fail.
The two signed manifests are committed by a deterministic Merkle inventory.

Adjudication must not consume a parallel in-memory copy of the fixture values.
After signature and manifest verification, the accepted raw block bytes are
checked against their block receipts, encoding, channel map, sample rate,
declared count, and total length; only the integers decoded from those verified
bytes may enter endpoint analysis.  A changed raw byte or forged sample count
must fail before statistics or replay-state acceptance.

The transport digest, measured boot, attestation-result digest, firmware,
calibration, BOM, board, and device identifiers are conspicuous offline
placeholders.  They are not observations.

## Preregistered synthetic adjudication

The signed plan is locked before the synthetic raw blocks are interpreted.
It fixes two primary endpoints, `n=12` per endpoint, no interim looks, no
optional stopping, a four-decision Bonferroni family, wrapped phase errors,
paired Student-t TOST with an 80,000 microradian equivalence margin, and an
exact one-sided Clopper-Pearson lower bound for the probability of falling
within 40,000 microradians.  The required success probability is 0.65.

Program A and B are synthetic passing fixtures.  A large positive-offset
fixture is a mandatory failing control and is not promoted into the primary
family.  Full binary64 diagnostics remain in the outer JSON evidence.  The
signed adjudication receipt contains only integers and Booleans admitted by
the deterministic CBOR profile.

Synthetic passing evidence validates software and thresholds only.  It may
not be interpreted as measurement, physical equivalence, calibration,
restoration, reuse, advantage, or a statistical certificate.

## Resource and architecture discipline

The output counts encoded object sizes, raw fixture bytes, signatures,
verification operations, manifest inventory and replay state, interpreter and
library versions, and unknown physical energy, bandwidth, latency,
controller, and environment coordinates.  Its strongest honest comparators
are direct verification of the same public schemas/keys and direct
recomputation of the same public synthetic data and locked rules.

This preflight targets the existing compiled common Phase-QEMU V13 backend
contract but neither modifies nor executes QEMU.  It does not qualify a
standalone mechanism or an architecture promotion.  M257 remains intact and
no resource advantage is claimed.

## Qualification

Strict qualification requires:

1. production and independent-reference sources that do not import one
   another;
2. two byte-identical, bytecode-disabled executions of each;
3. empty stderr and successful exit;
4. exact parity for encoded-object digests, fixture analyses, failure
   controls, verified-byte-to-analysis dataflow, resource/nonclaim fields, and
   authority gates;
5. canonical stored seals equal to regenerated stdout bytes; and
6. no QEMU, VM, network transport, hardware, enrollment, or capture action.

Passing this contract authorizes no next physical action.  The M272 successor
still begins only after explicit user authorization.
