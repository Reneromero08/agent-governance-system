---
schema: verdict/v2
question: Q1
slug: q01_test
date: 2026-06-12
status: PARTIALLY_VERIFIED
verification: primed
packet_sha256: null
predecessor: null
method_summary: Ran the fixture harness once on the basic case and logged the run.
registry_ids:
  - R-EMB-01
prediction_ids:
  - P-001
claims:
  - id: C1
    text: The fixture harness exits zero on the basic case.
    status: PARTIALLY_VERIFIED
    falsifier: A nonzero exit code from the fixture harness on the basic case.
    key_results:
      - "exit code 0"
      - "ok 3/3 checks"
    evidence:
      - evidence/run_log.txt
evidence_manifest:
  - path: evidence/run_log.txt
    sha256: "9cf585dfd4b8f25e88f2165b25b7d40d95f1f2c00c3d295341601625a428d667"
verifications:
  - date: 2026-06-12
    mode: primed
    result: PARTIALLY_VERIFIED
---

## Hypothesis

The fixture harness produces a deterministic index for one question.

## Claims

- C1: The fixture harness exits zero on the basic case.

## Method

Ran the fixture harness once on the basic case and logged stdout to
evidence/run_log.txt.

## Results

    cmd: python run_test.py --case basic
    out: ok 3/3 checks
    exit code 0

## Status

**Status:** PARTIALLY_VERIFIED

## Provenance

Predecessor: none. Single primed run on 2026-06-12.
