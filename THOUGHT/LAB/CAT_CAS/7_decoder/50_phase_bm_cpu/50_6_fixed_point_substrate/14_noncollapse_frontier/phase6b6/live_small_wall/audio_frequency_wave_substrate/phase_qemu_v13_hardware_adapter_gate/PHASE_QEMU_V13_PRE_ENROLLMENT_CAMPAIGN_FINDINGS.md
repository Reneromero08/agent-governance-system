# Phase-QEMU V13 pre-enrollment campaign findings

## Result

The hardware-free pre-enrollment program closes an implementation gap in the
future V13 campaign contract.  It produces and verifies actual deterministic
CBOR byte strings and offline Ed25519 COSE_Sign1 objects rather than treating
parser and signature outcomes as symbolic Booleans.  It also locks the
synthetic statistical plan before manifest interpretation and binds every raw
block into a signed manifest inventory.

This is a software-conformance result only.  All identities, trust roots,
transport bindings, firmware, measured-boot, attestation, calibration, BOM,
raw samples, and keys are offline fixtures.  No live parser input came from a
device, no TLS or TPM operation occurred, and no physical evidence exists.

## Executed boundaries

The production run checks strict deterministic encoding, enrollment and plan
signatures, distinct roles, two challenge-bound manifests, raw-block hashes,
monotonic replay state, and a signed adjudication preflight.  It rejects:

- non-shortest integers, indefinite items, unsorted and duplicate map keys,
  and trailing bytes;
- an altered manifest signature and the same signature under the wrong AAD
  domain;
- a changed challenge, a missing raw block, and a repeated sequence; and
- a bit-changed raw block and a declared sample count inconsistent with the
  signed bytes; and
- post-lock plan mutation through a changed plan digest.

The statistics are now causally downstream of the evidence boundary.  Both
implementations decode the receipt-verified signed raw blocks and analyze only
those decoded integers; neither endpoint analysis consumes the parallel
fixture-construction arrays.  The qualifier pins this source/dataflow property
as well as the resulting value parity.

The preregistered A and B synthetic endpoints pass both the Bonferroni-adjusted
TOST and exact finite-count rule.  The positive-offset control fails.  These
are fixed software vectors, not empirical power, calibration, or physical
performance evidence.

## Trust and evidence boundary

The four offline key roles demonstrate a serializable shape for separation of
duties.  Deterministically derived fixture private keys cannot establish
identity or provenance.  The placeholder transport and attestation digests
cannot establish a protected channel, freshness, measured boot, or device
custody.  RFC 9334 and RFC 9711 therefore remain future enrollment/profile
work, not executed claims.

The signed report is explicitly an `OFFLINE_SYNTHETIC_ADJUDICATION_PREFLIGHT_ONLY`.
It sets physical evidence, authenticated physical sample, campaign statistical
certificate, physical authorization, architecture promotion, and M257 escape
to false.

## Architecture and resource result

This preparation remains attached to the compiled V13 common-backend route:
it is not another mechanism-search twin and not a replacement QEMU device.
QEMU is neither changed nor executed.  The package charges encoded bytes,
raw bytes, signing/verifying work, replay and inventory state, numeric
precision, and software versions; physical energy, bandwidth, latency,
controller state, and environment history remain unknown.

The direct equal-access verifier and direct statistical recomputation consume
the same public bytes, keys, schemas, samples, and plan.  No computational
resource advantage, physical restoration, same-carrier reuse, physical
bit-to-pi replacement, or M257 escape follows.

## Next boundary

After strict software qualification, the next step is still the existing
M272 successor: user-authorized pinned device enrollment followed by a
preregistered blinded dual-rail capture campaign with device-signed raw
manifests and independent familywise adjudication.  This package does not
cross or weaken that authorization boundary.
