# Phase 6B Coherence Repair Record

**Branch:** `repair/phase6b-coherence`  
**Original base:** `48bf42e159c81c798aee2f61b2e6e2e9eb1f295a`  
**Synchronized main:** `11b9c4fb6f061833d01aab089d922c467242c322`  
**Course-correction merge:** `8f9d35493ff3bb8cbc54da140b10bd41b28eedb2`  
**Date:** 2026-06-18  
**Merge policy:** squash merge after deferred SSH verification

---

## Purpose

Repair the Phase 6B dependency chain before further scientific work. The branch preserves the non-collapse architecture while removing stale authority, invalid evidence classifications, non-identifiable controls, weak serialization semantics, and mutable review binding.

No new physical result is claimed by this branch.

The later `COURSE_CORRECTION.md` directive was incorporated as a real second-parent merge from current `main`, then reconciled into the active navigation and roadmaps. The branch is not allowed to skip its revised physical ladder.

---

## Changes

### Evidence and authority

- Corrected the substrate fixed-point warmup to have one integer fixed point.
- Invalidated the earlier L3 pass pending a run of the corrected source.
- Replaced the original scalar substrate charter and root spec with historical authority stubs.
- Refreshed the Phase 6 lab-state audit, navigation, master roadmap, and Phase 6B roadmap.
- Marked the reorganization record historical rather than current status authority.

### Class B PDN screen

- Invalidated the old W_B residue interpretation.
- Separated T300 channel evidence from operand-response evidence.
- Replaced the design with crossed assignments that separate value response from core bias.
- Replaced the source so public orbit values affect executed switching activity.
- Added real idle and dummy captures.
- Removed hardcoded control passes.
- Outputs complex `R_value` and `R_core` coordinates without adjudication.

### `.holo` runtime integrity

- Added semantic validation of each declared phase-walk transition.
- Added parameter-range and state-continuity checks beyond structural digests.
- Added transactional `CollapseBoundary` rollback.
- Added strict deserialization that compares serialized path, restoration, and boundary lifecycle states with reconstructed state.
- Routed the production L4B witness through semantic validation, atomic boundary crossing, and strict reload.
- Added adversarial tests for self-consistent forgery and lifecycle tampering.

### Observability governance

- Closed input, state, calibration-artifact, gate, falsification, and artifact references.
- Registered fixed schedule-transform controls.
- Required complete unique `G1-G10` and `F1-F10` sets.
- Moved human review into a separate envelope.
- Bound review to the internal design digest and full SHA-256 of serialized design bytes.
- Kept review explicitly non-authorizing.

### Canonical schema

- Reconciled `.holo` documentation to schema `1.4.0`.
- Added a semantic integrity profile.
- Marked `holo_record.*` as legacy L4A scaffold rather than an alternate canonical schema.
- Narrowed invariant wording to exactly what the executable checks prove.

### Canonical gate dependency

The full Contracts run exposed a host-dependent CMP-01 defect outside the Phase 6 subtree: Linux `pathlib.Path` did not recognize Windows-drive and UNC absolute syntax. `CAPABILITY/MCP/validation.py` now evaluates both POSIX and Windows path grammars for absolute and traversal checks.

This is the only non-Phase-6 source change in the final diff. It is required by the canonical push plan and covered by the live CMP-01 validator test.

### Course-correction alignment

The active execution order is now explicit:

```text
repair verification
→ carrier-witness closure
→ external observability-design review
→ observable state/operator identification
→ physical closure/restoration tier
→ target-to-carrier coupling
→ fold-odd invariant at CollapseBoundary
→ repeated Small Wall adjudication
```

A successful software packet does not close the carrier witness. A carrier witness does not establish relational memory. Operator identification does not establish restoration. Restoration does not establish target coupling or a crossing.

---

## Repository CI gate

The branch must show both repository workflows green on its final synchronized head:

```text
Governance [DEPRECATED]
Contracts, including canonical push test plan
```

A temporary diagnostic workflow was used only to expose one truncated pytest failure tail and was deleted. It is absent from the final diff. Exact final run status belongs in PR #5 and the merge proof packet rather than being frozen into this document.

---

## Deferred SSH verification

The GitHub file-editing environment did not execute the target C/hardware paths. Those remain deliberately deferred.

The verification entry point is:

```text
14_noncollapse_frontier/verify_phase6b_repair.sh
```

The default mode performs the corrected software warmup, release tests, sanitizer tests, semantic/tamper tests, artifact regeneration, forbidden-field scans, and SHA-256 recording. The Class B hardware capture is an explicit optional calibration capture; it does not close the full carrier witness.

The carrier-witness closure required by `COURSE_CORRECTION.md` remains a separate physical evidence gate with raw/reconstructable acquisition requirements.

---

## Merge contract

The branch now contains current `main` as an ancestor. The diff contains the Phase 6 repair plus the one CMP-01 cross-platform gate dependency described above.

Preferred integration is a squash merge. This compresses connector-generated file commits into one architectural history entry while retaining the review discussion in the pull request.

Suggested squash commit:

```text
fix(phase6): repair Phase 6B evidence, .holo integrity, and review governance
```

Do not merge until the SSH verification packet records:

- branch head SHA and current-main merge base;
- clean diff check;
- release and sanitizer results;
- corrected L3 outcome;
- Class B build/acquisition status;
- regenerated artifact hashes;
- carrier-witness state: closed or explicitly pending;
- final claim ceiling.

---

## Post-merge gates

After verification and squash merge:

1. close or explicitly scope the physical carrier witness;
2. complete the external `L4B.5B0` human design review;
3. authorize observability acquisition only when both predecessors are satisfied.

Physical restoration, target coupling, and Small Wall adjudication remain unauthorized.
