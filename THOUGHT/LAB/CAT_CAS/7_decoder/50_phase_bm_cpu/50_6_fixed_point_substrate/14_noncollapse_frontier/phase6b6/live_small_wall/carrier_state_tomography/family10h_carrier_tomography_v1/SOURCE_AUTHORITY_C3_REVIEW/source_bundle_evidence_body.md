# C3 Source-Authority Review

**Role:** source/bundle/runtime evidence auditor
**Model:** GPT-5 Codex
**Audited commit:** `55e059bc7acaafee3feacddac2069d7b5e40edd1`

## Evidence Inspected

- Exact clean branch HEAD and C3 commit tree; worktree remained unchanged.
- C2 blocker definitions and the C3 diff in `family10h_carrier_tomography_target.py` and `run_family10h_carrier_tomography_v1.py`.
- Committed source hashes, deterministic source bundle, runtime authority, manifest/sidecar, controller self-test, target self-test, deployment-layout self-test, runtime self-test, and offline-validation receipts.
- Independent, in-memory reconstruction of all hashes and the deterministic tarball. No repository or temporary files were created.
- No SSH, SCP, ping, target inspection, PMU, tomography, runtime launch, or live-authority path was invoked.

## Attempted Attacks

- Tested the committed transfer design against runtime omission, byte mutation, size mutation, wrong Git-blob identity, altered transfer keyset, source/bundle omission, bundle mutation, and unexpected-file cases. The committed regression records every case rejected.
- Traced the S/E0/E1 fixture: E1 changes only `family10h_carrier_tomography_runtime`; the nine text source blobs and remaining authority blobs retain their S identities.
- Removed runtime from the replay comparison through the regression's defective comparator. It reports no changed authority object and no runtime-specific failure, while the production comparator reports the runtime object.
- Recomputed source hashes, bundle bytes, runtime SHA-256/size/Git-blob ID, receipt digests, manifest canonical digest, and sidecar file digest independently.

## Findings

1. Runtime transfer is repaired. `DISCOVERY_TRANSFER_FILE_NAMES` now contains the twelve source/generated authority files plus the runtime binary. The production transfer plan freezes all 13 records, validates runtime hash/size/blob identity, and the actual SCP loop consumes those records. The fresh target-root regression passes before inventory with zero runtime, PMU, or tomography use.

2. Exact-object replay is repaired. `replay_final_exact_objects()` defaults to the complete discovery-transfer authority set, accepts explicit `repo_root` and `package_root`, and compares the runtime blob alongside source authority.

3. The regression now constructs three temporary descendant commits: S `abc0b9aa55c68f7f517025008022c113bd2ce4cf`, E0 `0cf237400c4416666ffaae7e6e55c7f518bcd34b`, and E1 `ab9e6562a25efe875595dc6faa278d14dd6c4ee8`. E1 produces exactly:
   - changed authority: `family10h_carrier_tomography_runtime`
   - replay result: failed
   - specific failure: `evidence overlay changed source/runtime authority blobs: family10h_carrier_tomography_runtime`
   - all nine text source blobs unchanged.

4. Mutation sensitivity is demonstrated. Runtime exclusion misses E1 and emits no runtime-specific failure; the production authority set detects it. This closes the C2 regression blocker that previously used blob inequality without invoking final replay.

5. C3 authority bindings are coherent:
   - source-hashes canonical SHA-256: `997879a0b69393074762a4b3d2f0957250e640506ba0b2cb05c11cfbdc53517f`
   - deterministic bundle SHA-256: `b58b0967018fb0e41caa5b4bf494e2bf66542b9770409b4dd763d641467a9b43`
   - runtime SHA-256: `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`
   - runtime size: `22928`
   - runtime Git blob: `3c007e278b7c3f2b206708739fd9abab5d3e91e7`

   The bundle reconstructed byte-for-byte with exactly nine sorted regular members, mode `0644`, timestamp and UID/GID zero, and empty owner/group names. All inspected receipt and manifest digests recomputed successfully.

## Recommendations

- Non-blocking hardening: pass `review_root=SOURCE_AUDIT_REVIEW_DIR` explicitly from `replay_final_exact_objects()` into `source_audit_quorum()`. Its current default captures the original path at definition time; this does not affect the present S/E0/E1 comparator test but would matter for a future fully populated temporary-evidence replay.
- Make the E1 assertion directly require `changed_source_files_after_c1 == [RUNTIME_BINARY_NAME]` to express the runtime-only invariant in one check.

## Final Verdict

NO_MATERIAL_BLOCKER

BODY_COMPLETE
