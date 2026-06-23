# Phase 6 V2 Architectural Review

**Status:** `SAME_BYTE_CUSTODY_REPAIR_HOSTED_PROVEN__LOCAL_MATERIALIZATION_PENDING`  
**Authority:** `../../PHASE6_V2_ENGINEERING_QUALIFICATION_ADDENDUM_2026-06-22.md`  
**Execution work package:** `../V2_FINAL_QUALIFICATION_WORK_PACKAGE.md`  
**Scientific acquisition authorized:** false  
**Hardware calibration executed:** false

This lane is forward-only engineering qualification. It does not alter V1, accept Gate R, enter Phase 6B.6, establish a predictive physical operator, authorize restoration, execute target coupling, recover orientation, or cross the Small Wall.

## Preserved scientific state

```text
V1:
PERMANENT_RETROSPECTIVE_NEGATIVE_ADJUDICATION
NO_STABLE_PREDICTIVE_OPERATOR
PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN

T48 carrier:
TRANSFER_EQUIVARIANCE_SUPPORTED under a minimal C0 receiver chart
STRICT_CARRIER_CLOSURE_PARTIAL

V2:
ENGINEERING_QUALIFICATION_ONLY

Gate R:
PENDING

Phase 6B.6:
NOT ENTERED
```

The V2 schedule is ascending-order engineering calibration. It is not the proposed reversed/randomized tone-order scientific control.

## Qualified source mechanisms

The current committed V2 source contains:

- requested-frequency eight-state waveform semantics;
- 1/8, 2/8, and 3/8 amplitude duties;
- strict full-consumption CLI numeric parsing;
- strict complete top-level authorization JSON validation;
- singleton session and route/core bindings;
- source-bundle singleton binding;
- sender-off control tone and theta fields;
- sender-off Nyquist analysis at the declared control tone;
- shared C capture-quality thresholds;
- direct C rejection tests for coverage, empirical sample rate, Nyquist margin, and timestamp gaps;
- exact plan/C threshold identity regression;
- immutable run-root file-set and regular-file enforcement;
- all scientific authorization fields false by default and in analysis output.

GitHub-hosted Ubuntu has passed strict compilation, runner contracts, C/Python waveform equivalence, Slot2 identity, V2 contracts and analyzer tests, ASan, and UBSan for those mechanisms.

## Custody defect found after the previous local report

The previous local sequence produced:

```text
source repair:
b7563e5fe67d267840f4d5a25c776e7504e7dc5e

generated contracts:
93f28c5db29eaeeca7d0375efc5f69da8bea15b8

plan SHA-256:
f67ecbba90368ded107cc1cf5225b27698500c6399d5eb3aecc689b0f1edef18

source-bundle SHA-256:
416d748dd851735b5ada5c5f193ba874424fbf24844f810f901e8b2f889ff48f
```

Those commits and digests remain historical provenance, but they are **superseded for final qualification**.

Round 5 did not land. The committed analyzer still hashed immutable paths and then reopened those paths through `read_text`, `open`, and `np.fromfile`. This left a hash-then-parse custody gap. The generated contracts therefore cannot be the final qualification object.

## Audited same-byte repair

The connector prepared a source-only, digest-locked repair that:

- opens immutable inputs with `O_NOFOLLOW` where supported;
- requires regular files;
- captures each input once;
- hashes the captured bytes;
- parses JSON, JSONL, CSV, telemetry, and raw records from those same bytes;
- rejects file identity changes during the read;
- replaces path-based `np.fromfile` with `np.frombuffer` over captured bytes;
- validates exact plan, window, session, run, manifest, evidence-map, runtime, threshold, and source-commit schemas;
- derives recorded input bindings from captured bytes only;
- rejects symlink traversal, extra run entries, raw trailing records, and manifest drift.

The repair includes regressions for:

- reopening after manifest validation;
- extra manifest fields;
- symlinked run roots;
- raw trailing records;
- plan top-level and nested window schema drift.

GitHub Actions run `28015182641`, run number `15`, passed the complete materialized source-only repair through strict C, runtime contracts, waveform equivalence, Slot2 identity, full analyzer campaign tests, ASan, and UBSan.

The repair is stored in:

```text
APPLY_SAME_BYTE_CUSTODY_SOURCE_ONLY.py
```

It has not yet been materialized into the committed analyzer source. Until that occurs, the V2 lane remains a draft qualification object.

## Required final sequence

```text
materialize audited source-only repair
-> remove both temporary installers
-> remove workflow source mutation
-> commit repaired source and tests
-> regenerate all contracts from that exact source commit
-> prove deterministic regeneration
-> pass GitHub against committed source directly
-> pass full no-write repository gate
-> pass exact-head Phenom II strict and sanitizer lanes
-> bind final evidence and authority documents
-> independent PR review
-> Gate R project-owner decision
```

No earlier source, generated contract, or evidence log may qualify a later head.

## Authorization boundary

```text
hardware_ran=false
authorization_artifact_created=false
calibration_authorized=false
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

No V2 hardware rerun is authorized by this review.