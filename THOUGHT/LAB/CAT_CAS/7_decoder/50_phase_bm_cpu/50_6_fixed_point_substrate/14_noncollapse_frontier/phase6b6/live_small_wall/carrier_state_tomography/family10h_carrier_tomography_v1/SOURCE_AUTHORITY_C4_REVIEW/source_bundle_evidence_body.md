# C4 Source-Authority Review

- **role:** source/bundle/runtime evidence auditor
- **model:** GPT-5 Codex
- **agent/thread ID:** not visible
- **final_response:** true

## Scope

Read-only review of branch `codex/family10h-tomography-repair`, commit `092d0a655e94d7c00f69efc1236cf1c8a2896ee1`, limited to the C4 affinity repair, attempt versioning, source authority, bundle/runtime evidence, self-test receipts, manifest integrity, and C3/C4 source-audit routing.

No files or Git state were modified. No checkout, target contact, SSH, SCP, ping, PMU access, live authority, package self-test, offline validator, runtime binary, or tomography execution occurred.

## Evidence Inspected

- Branch and HEAD matched the requested branch and commit. Parent is C3 failure-evidence commit `35844e76317017a73dc0fa83f7e976642b80c66f`.
- The tracked package bytes matched the audited commit.
- All nine source files matched their recorded sizes and SHA-256 values.
- `source_hashes_sha256` independently recomputed as `7f707088d83205d45c53d1560fe5c127e4b9e1194c7291fd6e35228aab038f26`.
- The deterministic source bundle contained exactly the nine expected files, byte-identical to the source set, with ordered members, mode `0644`, UID/GID `0`, blank owner names, and zero tar/gzip timestamps.
- `source_bundle_sha256` independently recomputed as `337718f31ffee3011ddc47e6fcb1606fbae846a4c7d9e62aff743192fef9be34`.
- Runtime blob `3c007e278b7c3f2b206708739fd9abab5d3e91e7`, size `22928`, independently hashed to `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`.
- Target, controller, runtime, deployment, and offline receipts were canonically rehashed. Their recorded digests matched and their stored pass states were true.
- Manifest file hash `105c3f905591d8174baebcedd89dca6b791f4c7c718fbcd19e6f5e85444bfe4b` and canonical hash `082ceeda1e2cf7a8e19a1d6bf17a6a0503f4a7be0e9a5f4ae26935d52df38afb` matched both the manifest and sidecar.
- All four archived C3 artifacts matched their index hashes, sizes, and pre-archive Git blob IDs. The active attempt namespace is empty.
- The affinity repair in [family10h_carrier_tomography_target.py](<D:/CCC 2.0/AI/agent-governance-system-family10h/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/family10h_carrier_tomography_target.py:2119>) isolates operational pin probes in child processes, checks singleton readback and actual CPU where available, and preserves the parent affinity mask.
- C4 additions introduce no SSH, SCP, runtime launch, PMU opening, or tomography-output creation. Existing offline runtime `--self-test` behavior was not introduced by C4.

## Attempted Attacks

- Recomputed every source-file and source-receipt digest.
- Checked bundle member substitution, omission, ordering, metadata, and byte drift.
- Checked runtime replacement against SHA-256, size, and Git blob identity.
- Recomputed manifest file and canonical hashes independently.
- Compared C3 review bindings against C4 hashes and traced discovery, manifest, and final-replay routing. Commit `092d0a...` selects `SOURCE_AUTHORITY_C4_REVIEW`; C3 artifacts are not used as the active review overlay.
- Checked for newly introduced runtime, PMU, tomography, or target-contact paths.
- Tested attempt-versioning logic by source-level mutation analysis: archive deletion, extra attempt insertion, removal or relabeling of C4 accounting fields, and digest resealing.

## Findings

### Material: C4 attempt versioning is recorded but not enforced

[run_family10h_carrier_tomography_v1.py](<D:/CCC 2.0/AI/agent-governance-system-family10h/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/run_family10h_carrier_tomography_v1.py:1156>) does not require the exact four archived C3 files, reject extra attempt records, validate `active_namespace_empty_after_archive`, or bind the history paths to the audited commit. An empty `archived_files` map or an additional non-C3 attempt can be resealed with a new index digest and accepted.

At lines 1252-1301, `attempt_version`, `per_attempt_counters`, `prior_cumulative_lane_counters`, and `cumulative_lane_counters` are added with `setdefault`, but transition and journal replay validation never requires or checks them. Final evidence replay at lines 5315-5340 only checks the older state, counter, digest, cleanup, review, and challenge fields.

Consequently, C3 archive custody can be removed and a C4 journal can be stripped or relabeled, resealed, and still satisfy the current replay path. Source-authority commit verification also excludes the history index and archived-attempt paths, leaving this possible after source review without dirtying the checked source-authority set.

Required repair: enforce an exact history schema and archive label set, reject unknown attempt versions, bind history/archive blobs to the source commit, require `attempt_version == "C4"` and exact per-attempt/prior/cumulative counters on every row, and add negative regressions for each mutation.

### Non-material

The C4 path routing itself is correct, but line 2134 still reports `"source authority C3 review gate failed"` on the C4 acquisition path. This is stale diagnostic wording, not a stale path or digest binding.

## Verdict

**MATERIAL_BLOCKER**

final_response=true
