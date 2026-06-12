---
schema: verdict/v2
question: Q1
slug: q01_example
date: 2026-06-12
status: UNSUPPORTED
verification: primed
packet_sha256: null
predecessor: null
method_summary: Desk review of the fixture example check plus one recorded smoke run.
registry_ids: [E-GEN-01]
prediction_ids: []
claims:
  - id: C1
    text: The example check passes on the fixture input.
    status: UNSUPPORTED
    falsifier: A run of the example check exits non-zero on the fixture input.
    key_results:
      - "exit code 0"
    evidence: []
evidence_manifest:
  - path: ../outside_evidence.txt
    sha256: 181ba6b6dc35eb47e97e52d7725a5f7d6bbdfd54e90133896c1e619ec2ded53e
verifications:
  - date: 2026-06-12
    mode: primed
    result: UNSUPPORTED
---

## Hypothesis

The example check passes deterministically on the fixture input.

## Claims

- C1: The example check passes on the fixture input.

## Method

Desk review of the example check script plus one recorded smoke run.

## Results

Command:

    python check_example.py --input fixture.txt

Output excerpt:

    OK: example check passed

exit code 0

## Status

**Status:** UNSUPPORTED

## Provenance

- date: 2026-06-12
- mode: primed
- predecessor: none
