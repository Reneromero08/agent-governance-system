# Shared External Frontier Substrate

**Status:** OPEN  
**Purpose:** domain-independent contracts, evidence custody, replay, and transfer infrastructure

---

# 1. What belongs here

Only code and schemas that remain valid across multiple frontiers:

- source freezing;
- specification hashing;
- evidence manifests;
- replay manifests;
- external verifier wrappers;
- claim-ceiling records;
- adjudication records;
- transfer records;
- leakage scans;
- deterministic environment capture;
- generic process-object interfaces.

Do not place field arithmetic, CT geometry, ARC logic, theorem generators, ECC circuits, or hardware models here.

---

# 2. Planned module layout

```text
shared/
  README.md
  contracts/
    source_manifest.schema.json
    frontier_run.schema.json
    evidence_manifest.schema.json
    adjudication_record.schema.json
    transfer_record.schema.json
  source_freeze/
  evidence/
  replay/
  adjudication/
  leakage/
  transfer/
```

Directories are created when executable code lands. Do not commit empty placeholders.

---

# 3. Source manifest contract

Every active experiment must record:

```text
frontier_id
official_title
official_urls
retrieved_at
rules_version
software_repository
software_commit
constants_digest
examples_digest
prize_snapshot
deadline_snapshot
eligibility
license
publication_requirement
compute_restrictions
known_ambiguities
specification_digest
```

A source manifest is immutable after a run begins. A changed source creates a new manifest version.

---

# 4. Frontier run contract

Every run records:

```text
run_id
frontier_id
specification_digest
repo_commit
environment_digest
mechanism_id
process_object_version
input_manifest
seed_manifest
command
start_time
end_time
raw_artifacts
derived_artifacts
controls
verifier_version
verdict
failure_reason
claim_ceiling
```

Runs missing the specification digest or repository commit are not claim-bearing.

---

# 5. Evidence manifest contract

Separate three layers.

## Raw

Direct outputs from the computation, benchmark, instrument, or external evaluator.

## Derived

Features, summaries, plots, models, reductions, or candidate witnesses computed from raw artifacts.

## Narrative

Reports and interpretation.

Narrative cannot substitute for missing raw or derived evidence.

---

# 6. Witness and replay contract

A replay bundle must contain:

- exact witness or artifact;
- specification digest;
- official verifier adapter;
- independent verifier where possible;
- deterministic command;
- environment lock;
- expected outputs;
- hashes;
- accounting rules;
- known nondeterminism;
- complete failure message when rejected.

A benchmark submission that cannot expose private test data still records the local build, submission digest, and official returned score.

---

# 7. Adjudication record contract

```text
submission_id
frontier_id
adjudication_class
submitted_at
artifact_digest
official_status
official_score_or_verdict
official_message
independent_status
dispute_status
administrative_eligibility
scientific_claim_ceiling
```

Pending review remains pending.

---

# 8. Leakage scan contract

Scan for:

- target witness strings;
- private benchmark data;
- challenge labels in control fields;
- hidden answers in fixtures;
- known solutions in prompts or retrieval caches;
- post-hoc metric selection;
- evaluation set reuse;
- omitted resource costs;
- relaxed-verifier substitution;
- manual interventions absent from the run manifest.

Each experiment extends the scan with domain-specific forbidden channels.

---

# 9. Transfer record contract

```text
source_frontier
evidence_level
transfer_level
old_boundary
working_boundary
external_boundary
preserved_object
relation_basis
mechanism
invariant_or_witness
restoration_class
controls_passed
domain_specific_components
reusable_components
next_transfer_target
forbidden_claims
```

---

# 10. First implementation slice

The smallest useful shared implementation is:

1. source manifest loader;
2. SHA-256 bundle digest;
3. run manifest writer;
4. artifact hash walker;
5. replay command runner;
6. adjudication record writer;
7. transfer-record validator.

Build this only after the first active frontier is selected, so the schema is tested against a real object rather than designed indefinitely in abstraction.
