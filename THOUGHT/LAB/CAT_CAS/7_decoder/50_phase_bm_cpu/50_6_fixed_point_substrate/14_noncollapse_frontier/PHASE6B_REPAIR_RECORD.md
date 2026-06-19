# Phase 6B Coherence Repair Record

**Branch:** `repair/phase6b-coherence`  
**Base:** `48bf42e159c81c798aee2f61b2e6e2e9eb1f295a`  
**Date:** 2026-06-18  
**Merge policy:** squash merge after deferred SSH verification

---

## Purpose

Repair the Phase 6B dependency chain before further scientific work. The branch preserves the non-collapse architecture while removing stale authority, invalid evidence classifications, non-identifiable controls, weak serialization semantics, and mutable review binding.

No new physical result is claimed by this branch.

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
- Added transactional CollapseBoundary rollback.
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

This is the only non-Phase-6 source change in the final diff. It was required for the canonical push plan and is covered by the existing live CMP-01 validator test.

---

## Repository CI

Clean final-diff GitHub Actions results:

```text
Governance [DEPRECATED]  PASS
Contracts                PASS
canonical push plan      PASS
```

A temporary diagnostic workflow was used only to expose a truncated pytest failure tail and was deleted before the clean run. It is absent from the final diff.

---

## Deferred SSH verification

The GitHub file-editing environment did not execute the target C/hardware paths. Those remain deliberately deferred.

The verification entry point is:

```text
14_noncollapse_frontier/verify_phase6b_repair.sh
```

The default mode performs the corrected software warmup, release tests, sanitizer tests, semantic/tamper tests, artifact regeneration, forbidden-field scans, and SHA-256 recording. The Class B hardware capture is an explicit optional mode.

---

## Merge contract

The branch is based directly on `main`. The final diff contains the Phase 6 repair plus the one CMP-01 cross-platform gate dependency described above.

Preferred integration is a squash merge. This compresses connector-generated file commits into one architectural history entry while retaining the review discussion in the pull request.

Suggested squash commit:

```text
fix(phase6): repair Phase 6B evidence, .holo integrity, and review governance
```

Do not merge until the SSH verification packet records:

- branch head SHA;
- clean diff check;
- release and sanitizer results;
- corrected L3 outcome;
- Class B hardware status, executed or explicitly deferred;
- regenerated artifact hashes;
- final claim ceiling.

---

## Post-merge next gate

After verification and squash merge:

```text
L4B.5B0 external human design review
```

Calibration acquisition and physical restoration remain unauthorized.
