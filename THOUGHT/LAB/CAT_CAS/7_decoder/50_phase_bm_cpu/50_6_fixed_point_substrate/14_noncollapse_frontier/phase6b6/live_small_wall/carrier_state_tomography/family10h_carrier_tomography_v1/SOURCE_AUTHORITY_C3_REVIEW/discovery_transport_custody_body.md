# C3 Source-Authority Review

**Role:** discovery transport and custody auditor
**Model:** GPT-5 Codex
**Audited commit:** `55e059bc7acaafee3feacddac2069d7b5e40edd1`
**Branch:** `codex/family10h-tomography-repair`
**Source hashes SHA-256:** `997879a0b69393074762a4b3d2f0957250e640506ba0b2cb05c11cfbdc53517f`
**Source bundle SHA-256:** `b58b0967018fb0e41caa5b4bf494e2bf66542b9770409b4dd763d641467a9b43`
**Runtime SHA-256:** `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`
**Runtime Git blob:** `3c007e278b7c3f2b206708739fd9abab5d3e91e7`

## Evidence Inspected

- Confirmed the requested branch was clean and its exact `HEAD` was C3 before and after review.
- Compared C3 with parent `7647e872122e75edd07bdd423d9493a2796c8fd9`, including the archived C2 blocker and required regression.
- Inspected `family10h_carrier_tomography_target.py:28-46,284-432,2217-2326`.
- Inspected `run_family10h_carrier_tomography_v1.py:684-864,1939-2338,2683-3190,4080-4269`.
- Independently verified all nine source hashes and sizes against the committed source-hash authority.
- Verified the runtime SHA-256, size `22928`, and C3 Git blob against runtime authority.
- Verified the source bundle hash and exact nine-member source keyset.
- Inspected committed controller, target, runtime, deployment, manifest, and offline-validation receipts.
- `git diff --check` reported no whitespace errors.
- Tests were not re-executed because their workflows write artifacts; committed offline evidence and source paths were inspected read-only.

## Attempted Attacks

- Traced missing runtime, source-hash receipt, source bundle, and individual source-file omissions. Each fails the pre-contact transfer plan and target transferred-root validation.
- Traced extra runtime/authority files and transfer keyset drift. Exact missing/extra checks and ordered equality with `DISCOVERY_TRANSFER_FILE_NAMES` reject them.
- Traced runtime byte, size, SHA-256, and Git-blob substitutions. Runtime authority, challenged identity, commit-blob comparison, and target validation reject them.
- Traced source-hash and source-file mutation, including coherent overlay attempts. Receipt digest, challenged source-hash identity, per-file hash/size, committed blobs, and bundle reconstruction reject them.
- Traced source-bundle byte mutation and source-to-bundle reconstruction mismatch. Both actual bundle hash and deterministic reconstruction must equal the challenged bundle hash.
- Traced unexpected target-root authority files. Target validation rejects unexpected files before sensor enumeration.
- Traced invalid copyback, cleanup failure, owner-marker mismatch, and failed absence verification. These prevent successful authority publication.
- No SSH, SCP, ping, target inspection, PMU access, tomography, or live-authority path was invoked.

## Findings

1. C3 repairs the C2 transport defect. `DISCOVERY_TRANSFER_FILE_NAMES` has one authoritative definition in the target module, and the controller references that definition rather than rebuilding the set.

2. Production materializes all 13 transfer files from the audited commit and builds the validated plan before the first SSH command. The plan binds names, sizes, SHA-256 values, local and committed Git blobs, authority classes, and destinations.

3. The real SCP loop now iterates `transfer_plan["records"]`; the previous source-only loop is removed. The runtime binary is therefore transported.

4. Pre-contact checks reject every requested omission, mutation, extra-file, reconstruction, wrong-runtime-blob, and keyset-drift case.

5. Target challenge validation invokes transferred-root validation before `enumerate_temperature_candidates()`. A transfer-time mutation therefore fails before sensor inventory.

6. Cleanup remains nonce-root and owner-marker scoped, runs from `finally`, performs an independent absence probe, and gates all success artifacts. Invalid copyback remains failure-custody only.

7. Committed offline receipts preserve `0/0/0/0` target-contact, sensor-inventory, live-invocation, and PMU counters. Discovery fixtures preserve the authorized future transaction shape `1/1/0/0` without runtime or PMU execution.

No material blocker was found for this role.

## Recommendations

Accept C3 for discovery transport and custody. No repair is required. As evidence hygiene, regenerate the production-plan regression from a clean exact C3-derived authority state so its recorded commit-blob field is non-null.

## Boundary Attestations

- `no_git_write: true`
- `no_file_edits: true`
- `no_checkout_mutation: true`
- `no_target_contact: true`
- `no_live_authority: true`
- `no_pmu: true`

## Final Verdict

NO_MATERIAL_BLOCKER

BODY_COMPLETE
