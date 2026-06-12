#!/usr/bin/env python
"""_make_fixtures.py - regenerate the built-in fixture trees.

Each fixture is a miniature v2_3 root (tree/) plus expect.json. Evidence
hashes are computed from the actual bytes written, so valid fixtures
genuinely pass. expected_INDEX.md files for the valid fixtures are
produced by running generate_index.py on the tree and reviewing the
output, then copying tree/INDEX.md next to expect.json.

ASCII only. Deterministic. Not a fixture itself (test_generator only
iterates directories).
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write(relpath, text):
    path = HERE / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def expect_ok():
    return json.dumps({"exit_code": 0,
                       "expected_index": "expected_INDEX.md"},
                      indent=2, sort_keys=True) + "\n"


def expect_err(code):
    return json.dumps({"exit_code": 1, "error_code": code},
                      indent=2, sort_keys=True) + "\n"


QUESTIONS = """# Fixture catalog (mini v2_3 root).
- id: Q1
  slug: q01_test
  tier: %d
  hypothesis: "The fixture harness produces a deterministic index for one question."
  predecessor: null
"""

VARIABLES = """# VARIABLES.md - fixture registry

| ID | substrate | definition | source | date added |
|----|-----------|------------|--------|------------|
| R-EMB-01 | EMB | fixture row: phase coherence on Gram eigenvalues | fixture | undefined |
| E-EMB-01 | EMB | fixture row: mean cosine overlap | fixture | undefined |
"""

PREDICTIONS = """# PREDICTIONS.md - fixture ledger

| P-NNN | date | question | registry IDs | predicted quantity | threshold | linked verdict |
|-------|------|----------|--------------|--------------------|-----------|----------------|
| P-001 | 2026-06-12 | Q1 | R-EMB-01 | fixture quantity | >= 0.5 | - |
"""

EV_BASIC = ("cmd: python run_test.py --case basic\n"
            "out: ok 3/3 checks\n"
            "exit code 0\n")
EV_ALPHA = ("cmd: python run_test.py --case alpha\n"
            "out: alpha = 0.75\n"
            "exit code 0\n")
EV_BETA = ("cmd: python run_test.py --case beta\n"
           "out: beta = 0.40\n"
           "exit code 0\n")


def base_tree(name, tier):
    write("%s/tree/_meta/questions.yaml" % name, QUESTIONS % tier)
    write("%s/tree/VARIABLES.md" % name, VARIABLES)
    write("%s/tree/PREDICTIONS.md" % name, PREDICTIONS)


VERDICT_ONE = """---
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
    sha256: "%(h)s"
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
"""

VERDICT_MULTI = """---
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
    sha256: "%(h1)s"
  - path: evidence/run_log_c2.txt
    sha256: "%(h2)s"
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
"""


def desk_verdict(status_line="UNSUPPORTED", body_status=None,
                 extra_claims="", registry_block="registry_ids: []",
                 manifest_block="evidence_manifest: []",
                 falsifier="A logged run that decides the fixture question either way.",
                 key_results_block="key_results: []",
                 claim_status="UNSUPPORTED",
                 claim_evidence="evidence: []",
                 results_body="No runs claimed for this desk review."):
    if body_status is None:
        body_status = status_line
    return """---
schema: verdict/v2
question: Q1
slug: q01_test
date: 2026-06-12
status: %(status)s
verification: primed
packet_sha256: null
predecessor: null
method_summary: Desk review of the fixture question with no runs claimed.
%(registry)s
prediction_ids: []
claims:
  - id: C1
    text: The fixture question remains undecided pending a real run.
    status: %(cstatus)s
    falsifier: "%(falsifier)s"
    %(keyres)s
    %(cevidence)s
%(extra)s%(manifest)s
verifications:
  - date: 2026-06-12
    mode: primed
    result: %(status)s
---

## Hypothesis

The fixture harness produces a deterministic index for one question.

## Claims

- C1: The fixture question remains undecided pending a real run.

## Method

Desk review only; no commands were run.

## Results

%(results)s

## Status

**Status:** %(bstatus)s

## Provenance

Predecessor: none.
""" % {"status": status_line, "bstatus": body_status,
       "cstatus": claim_status, "falsifier": falsifier,
       "keyres": key_results_block, "cevidence": claim_evidence,
       "extra": extra_claims, "registry": registry_block,
       "manifest": manifest_block, "results": results_body}


def main():
    # ---- valid_minimal: catalog only, no verdicts -> all-OPEN index ----
    base_tree("valid_minimal", 0)
    write("valid_minimal/expect.json", expect_ok())

    # ---- valid_one_verdict: one PARTIALLY_VERIFIED verdict --------------
    base_tree("valid_one_verdict", 2)
    write("valid_one_verdict/tree/q01_test/evidence/run_log.txt", EV_BASIC)
    write("valid_one_verdict/tree/q01_test/VERDICT.md",
          VERDICT_ONE % {"h": sha(EV_BASIC)})
    write("valid_one_verdict/expect.json", expect_ok())

    # ---- valid_multistatus: claims V + PV -> verdict PV (MIN rule) ------
    base_tree("valid_multistatus", 1)
    write("valid_multistatus/tree/q01_test/evidence/run_log_c1.txt",
          EV_ALPHA)
    write("valid_multistatus/tree/q01_test/evidence/run_log_c2.txt",
          EV_BETA)
    write("valid_multistatus/tree/q01_test/VERDICT.md",
          VERDICT_MULTI % {"h1": sha(EV_ALPHA), "h2": sha(EV_BETA)})
    write("valid_multistatus/expect.json", expect_ok())

    # ---- bad_status_not_min: fm UNSUPPORTED but MIN(claims) FALSIFIED ---
    base_tree("bad_status_not_min", 0)
    extra = """  - id: C2
    text: The fixture control case fails by construction.
    status: FALSIFIED
    falsifier: "The control case passing on any run."
    key_results: []
    evidence: []
"""
    write("bad_status_not_min/tree/q01_test/VERDICT.md",
          desk_verdict(extra_claims=extra))
    write("bad_status_not_min/expect.json", expect_err("E_STATUS_NOT_MIN"))

    # ---- bad_contradictory_body: body Status != frontmatter status ------
    base_tree("bad_contradictory_body", 0)
    write("bad_contradictory_body/tree/q01_test/VERDICT.md",
          desk_verdict(body_status="VERIFIED"))
    write("bad_contradictory_body/expect.json",
          expect_err("E_BODY_MISMATCH"))

    # ---- bad_missing_manifest_file: manifest path absent on disk --------
    base_tree("bad_missing_manifest_file", 0)
    manifest = """evidence_manifest:
  - path: evidence/ghost.txt
    sha256: "%s\"""" % ("a" * 64)
    write("bad_missing_manifest_file/tree/q01_test/VERDICT.md",
          desk_verdict(manifest_block=manifest))
    write("bad_missing_manifest_file/expect.json",
          expect_err("E_MANIFEST_MISSING"))

    # ---- bad_hash_mismatch: manifest sha256 wrong for a real file -------
    base_tree("bad_hash_mismatch", 0)
    write("bad_hash_mismatch/tree/q01_test/evidence/run_log.txt", EV_BASIC)
    manifest = """evidence_manifest:
  - path: evidence/run_log.txt
    sha256: "%s\"""" % ("0" * 64)
    write("bad_hash_mismatch/tree/q01_test/VERDICT.md",
          desk_verdict(manifest_block=manifest))
    write("bad_hash_mismatch/expect.json", expect_err("E_HASH"))

    # ---- bad_empty_manifest_verified: VERIFIED claim, empty manifest ----
    base_tree("bad_empty_manifest_verified", 0)
    write("bad_empty_manifest_verified/tree/q01_test/VERDICT.md",
          desk_verdict(status_line="VERIFIED", claim_status="VERIFIED"))
    write("bad_empty_manifest_verified/expect.json", expect_err("E_FLOOR"))

    # ---- bad_unknown_registry_id: registry id not in VARIABLES.md -------
    base_tree("bad_unknown_registry_id", 0)
    registry = """registry_ids:
  - R-FAKE-99"""
    write("bad_unknown_registry_id/tree/q01_test/VERDICT.md",
          desk_verdict(registry_block=registry))
    write("bad_unknown_registry_id/expect.json", expect_err("E_REGISTRY"))

    # ---- bad_phantom_keynum: key result not verbatim in Results ---------
    base_tree("bad_phantom_keynum", 0)
    keyres = """key_results:
      - "exit code 0\""""
    write("bad_phantom_keynum/tree/q01_test/VERDICT.md",
          desk_verdict(key_results_block=keyres,
                       results_body="No runs were logged for this desk "
                                    "review."))
    write("bad_phantom_keynum/expect.json", expect_err("E_KEYNUM"))

    # ---- bad_no_falsifier: empty falsifier on a claim -------------------
    base_tree("bad_no_falsifier", 0)
    write("bad_no_falsifier/tree/q01_test/VERDICT.md",
          desk_verdict(falsifier=""))
    write("bad_no_falsifier/expect.json", expect_err("E_FALSIFIER"))

    # ---- bad_unknown_catalog_dir: q-dir slug missing from catalog -------
    base_tree("bad_unknown_catalog_dir", 0)
    write("bad_unknown_catalog_dir/tree/q03_unknown/note.txt",
          "placeholder so the uncataloged question dir exists\n")
    write("bad_unknown_catalog_dir/expect.json", expect_err("E_CATALOG"))

    print("fixtures written under %s" % HERE)


if __name__ == "__main__":
    main()
