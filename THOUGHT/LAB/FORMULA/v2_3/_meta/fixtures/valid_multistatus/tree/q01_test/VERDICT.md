---
schema: verdict/v2
question: Q1
slug: q01_test
date: 2026-06-12
status: PARTIALLY_VERIFIED
verification: primed
packet_sha256: null
predecessor: null
method_summary: Ran the fixture harness on the alpha and beta cases and logged both runs.
registry_ids:
  - R-EMB-01
  - E-EMB-01
prediction_ids:
  - P-001
claims:
  - id: C1
    text: The alpha case reproduces alpha = 0.75 exactly.
    status: VERIFIED
    falsifier: An alpha value other than 0.75 from the alpha case run.
    key_results:
      - "alpha = 0.75"
    evidence:
      - evidence/run_log_c1.txt
  - id: C2
    text: The beta case reproduces beta = 0.40 on the fixed seed only.
    status: PARTIALLY_VERIFIED
    falsifier: A beta value other than 0.40 from the beta case run on the fixed seed.
    key_results:
      - "beta = 0.40"
    evidence:
      - evidence/run_log_c2.txt
evidence_manifest:
  - path: evidence/run_log_c1.txt
    sha256: "ca7f96d094daaaae6e0cb37f769bc597b33a184991a8dc36558483844317b013"
  - path: evidence/run_log_c2.txt
    sha256: "02e9d3e65c70ff797841b28a62c7fefb4f3242d43c89343b2feec76065c3e747"
verifications:
  - date: 2026-06-12
    mode: primed
    result: PARTIALLY_VERIFIED
---

## Hypothesis

The fixture harness produces a deterministic index for one question.

## Claims

- C1: The alpha case reproduces alpha = 0.75 exactly.
- C2: The beta case reproduces beta = 0.40 on the fixed seed only.

## Method

Ran the fixture harness twice (alpha and beta cases) and logged stdout
to evidence/run_log_c1.txt and evidence/run_log_c2.txt.

## Results

    cmd: python run_test.py --case alpha
    out: alpha = 0.75
    exit code 0

    cmd: python run_test.py --case beta
    out: beta = 0.40
    exit code 0

## Status

**Status:** PARTIALLY_VERIFIED

## Provenance

Predecessor: none. Single primed session on 2026-06-12.
