# Phase 6B Coherence Repair Record

**Branch:** `repair/phase6b-coherence`
**Original base:** `48bf42e159c81c798aee2f61b2e6e2e9eb1f295a`
**Synchronized main / merge base:** `5579d8be06df1f7fc1203cc9b6a7481fa515324a`
**Course-correction merge:** `8f9d35493ff3bb8cbc54da140b10bd41b28eedb2`
**SSH-verified source head:** `fbacd9ee0092dd2118d5f050592f8f0089852135`
**Date:** 2026-06-18
**Merge policy:** squash merge after review of the completed SSH verification packet

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

## SSH verification result

The source-only payload from `fbacd9ee0092dd2118d5f050592f8f0089852135` was compiled and executed on the target Phenom II X6 1090T host with GCC 14.2.0 and OpenSSL 3.5.6. The default software matrix passed:

- corrected L3 warmup: `90/90` catalytic loops and `10/10` for forward scan, identity, wrong restore, no-fixed-point negative, and replay; every convergent corrected-map run ended at `42`;
- Class B static build: passed with `-march=amdfam10 -Wall -Wextra -Werror`;
- L4B release suite and witness: passed;
- combined ASan/LSan/UBSan suite: passed with exit status zero;
- UBSan-only suite: passed with exit status zero;
- strict semantic, lifecycle-tamper, reference-closure, and external-review tests: passed;
- serialized forbidden-field scan: passed;
- retained-artifact manifest: every entry verified by `sha256sum -c`.

The Class B hardware capture was not executed. `k10temp` and all `/sys/class/hwmon` temperature inputs were unavailable, so the predeclared thermal-safety gate could not be established.

The verification entry point is:

```text
14_noncollapse_frontier/verify_phase6b_repair.sh
```

The default mode performs the corrected software warmup, release tests, sanitizer tests, semantic/tamper tests, artifact regeneration, forbidden-field scans, and SHA-256 recording. The Class B hardware capture is an explicit optional calibration capture; it does not close the full carrier witness.

Retained evidence SHA-256 values from the verified source head:

```text
fp_results.csv                    9be34c6a7a28dc4988024b5ac42a8ebf0a9bd7fbf798fc410248c1887ee1b9de
verification.log                 a4b31a4e3dfffa155b218919d3d76e26aa9a5a4749d193ef46226d938b6ce508
l4b_orbitstate_dry_run.holo      2890ee260023f24d63a31eb85ba481d709352f2e5152e68e27be1ba196202db9
l4b5a_physical_mapping.json      04660630d8fc6caa314a1645c4217feeb8e3aedb995d01ff911028d1ab374c0c
l4b5b0_observability_design.json e42881e243e6168f5fc5518482172f7fb6a7437c5ad109898fd97a6193ca2414
```

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
