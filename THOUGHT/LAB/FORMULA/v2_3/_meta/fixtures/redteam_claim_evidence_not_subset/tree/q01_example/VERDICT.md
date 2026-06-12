---
schema: verdict/v2
question: Q1
slug: q01_example
date: 2026-06-12
status: VERIFIED
verification: primed
packet_sha256: null
predecessor: null
method_summary: Desk review of the fixture example check plus one recorded smoke run.
registry_ids: [E-GEN-01]
prediction_ids: []
claims:
  - id: C1
    text: The example check passes on the fixture input.
    status: VERIFIED
    falsifier: A run of the example check exits non-zero on the fixture input.
    key_results:
      - "exit code 0"
    evidence:
      - evidence/ghost_log.txt
evidence_manifest:
  - path: evidence/run_log.txt
    sha256: 23a39045af1948b1ccbf2ca80ba297722838e1b5d62c67cb506c9d2866cde9e5
verifications:
  - date: 2026-06-12
    mode: primed
    result: VERIFIED
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

**Status:** VERIFIED

## Provenance

- date: 2026-06-12
- mode: primed
- predecessor: none
