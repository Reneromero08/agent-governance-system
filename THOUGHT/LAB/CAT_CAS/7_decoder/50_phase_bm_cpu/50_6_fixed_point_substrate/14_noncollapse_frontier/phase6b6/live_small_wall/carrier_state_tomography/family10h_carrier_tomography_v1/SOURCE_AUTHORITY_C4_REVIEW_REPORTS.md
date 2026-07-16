# C4 Source-Authority Review Reports

Audited commit: `092d0a655e94d7c00f69efc1236cf1c8a2896ee1`

Disposition: `FAMILY10H_C4_SOURCE_REVIEW_BLOCKED`

The C4 source-authority review returned three material blockers and one clear claim-boundary review. No C4 acquisition, target contact, PMU access, runtime launch, or tomography execution is authorized from this review.


---

## physical sensor-authority auditor

# C4 Source-Authority Review

- **Role:** physical sensor-authority auditor
- **Model:** OpenAI Codex (GPT-5)
- **Agent/thread ID:** not visible
- **Repository:** `D:\CCC 2.0\AI\agent-governance-system-family10h`
- **Branch:** `codex/family10h-tomography-repair`
- **Audited commit:** `092d0a655e94d7c00f69efc1236cf1c8a2896ee1`
- **Parent commit:** `35844e76317017a73dc0fa83f7e976642b80c66f`

## Scope

Read-only review of the C4 affinity repair and attempt-versioning changes. No files or Git state were modified. No checkout, target contact, SSH, SCP, ping, PMU access, live authority, runtime launch, or tomography execution occurred.

## Material Finding

### C4-SA-01: Per-attempt and cumulative counters are emitted but not authority-enforced

The [controller source](</D:/CCC 2.0/AI/agent-governance-system-family10h/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/run_family10h_carrier_tomography_v1.py:1257>) adds `attempt_version`, `per_attempt_counters`, `prior_cumulative_lane_counters`, and `cumulative_lane_counters` with `setdefault` at lines 1257-1260. The transition and journal validators at lines 1263-1327 do not validate these fields, require C4, make them immutable, or prove their equality to the authoritative state counters.

The freeze path also remains open:

- Invalid or missing history falls back to zero cumulative counters at lines 4613-4619.
- `package_decision` at lines 4678-4684 does not gate on valid attempt history or the expected successful cumulative total `2/1/0/0`.
- `final_evidence_paths()` at lines 4977-4995 omits the attempt-history index and archived C3 records.
- Final replay at lines 5270-5350 checks only C4 active counters `1/1/0/0`; it does not check attempt version, per-attempt mirrors, prior cumulative counters, or cumulative counters.

Consequently, a post-acquisition evidence overlay can alter those accounting fields, recompute the row digests, and still satisfy journal and final-evidence replay. Removing or invalidating the history index can also collapse the manifest cumulative baseline to zero without independently blocking freeze. This defeats the requested per-attempt and cumulative counter authority.

The exact source-stage artifacts at the audited commit currently report truthful values, but the source does not enforce that truth through final evidence qualification.

## Evidence Inspected

- Exact commit metadata, parent diff, branch identity, and complete package tree.
- [Target source](</D:/CCC 2.0/AI/agent-governance-system-family10h/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/family10h_carrier_tomography_target.py:2119>), including operational pinning and discovery ordering.
- Controller source, source-hash receipt, source bundle, implementation manifest, target/controller self-test receipts, C3 history index, C3 attempt journal, challenge, and cleanup custody.
- All nine source Git blobs matched their declared hashes and sizes, and all bundle members matched those blobs.
- `source_hashes_sha256=7f707088d83205d45c53d1560fe5c127e4b9e1194c7291fd6e35228aab038f26`
- `source_bundle_sha256=337718f31ffee3011ddc47e6fcb1606fbae846a4c7d9e62aff743192fef9be34`
- `runtime_binary_sha256=e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`

## Attempted Attacks

- **C3 interpretation:** The old affinity error is reached only after Family 10h identity and CPU 4/5 presence checks. Interpreting C3 as inherited-mask exclusion rather than absent platform CPUs is supported.
- **Operational pinning:** C4 probes CPUs 4 and 5 independently in forked children, requires singleton affinity readback, and rejects `EINVAL`, `EPERM`, readback mismatch, and observed execution-CPU mismatch.
- **Parent restoration:** Per-child and aggregate parent affinity are compared before and after probing; changed parent affinity fails closed.
- **Discovery ordering:** Platform and pin checks occur before temperature candidate enumeration.
- **C3 custody:** Three valid journal states stop at `target_command_invoked`; counters are `1/0/0/0`; cleanup and remote-root absence are recorded as successful. Archived hashes and internal digests matched.
- **C4 namespace:** No active authority, discovery, transport, challenge, attempt, journal, or cleanup receipt exists at the audited commit. Active C4 counters are `0/0/0/0`; cumulative lane counters are `1/0/0/0`.
- **Forbidden activity:** Added operational-probe code contains no runtime, PMU, or tomography invocation. The committed C3 evidence records zero runtime and PMU activity, and C4 has not begun.
- **Counter tampering:** Successful against the validation design as described in C4-SA-01.

## Verdict

MATERIAL_BLOCKER

final_response=true


---

## discovery transport and custody auditor

# C4 Source-Authority Review

**Role:** discovery transport and custody auditor  
**Model:** GPT-5 Codex  
**Agent/thread ID:** not exposed in this runtime  
**Branch:** `codex/family10h-tomography-repair`  
**Audited commit:** `092d0a655e94d7c00f69efc1236cf1c8a2896ee1`  
**Parent commit:** `35844e76317017a73dc0fa83f7e976642b80c66f`

## Findings

1. **Material: the C3 history gate does not require or source-bind the archived evidence.**  
   `validate_attempt_history_index()` accepts any dictionary for `archived_files` and only iterates entries that happen to exist (`run_family10h_carrier_tomography_v1.py:1190-1207`). An empty dictionary therefore bypasses all archive existence and digest checks. The validator also does not require the four labels, validate recorded byte sizes, parse the archived attempt/journal/cleanup objects, cross-check their internal digests and counters, or constrain paths to the package. The self-digest at lines 1217-1219 is freely recomputable, while source-commit dirtiness checks only the transport file set at lines 728-733 and excludes the history index and archives. Consequently, C4 acquisition can accept a locally rewritten index claiming `1/0/0/0` and successful cleanup after the underlying evidence has been removed or substituted.

   The exact C3 failure reason is likewise asserted rather than derived from durable failure output. The final attempt snapshot is written before target execution at lines 2260-2277; the nonzero result raises at lines 2278-2280 without persisting stderr. The archived objects prove a pre-inventory failure, but not uniquely the affinity error recorded by the new index.

2. **Material: cumulative counter accounting fails open and is not validated by journal replay.**  
   `history_cumulative_counters_or_zero()` silently returns all-zero history after any index failure (`run_family10h_carrier_tomography_v1.py:1229-1233`). `enrich_attempt_accounting()` then inserts aggregate fields with `setdefault()` at lines 1252-1260. Neither those fields nor `attempt_version` are checked by `validate_discovery_attempt_transition()` at lines 1263-1289. If history becomes unreadable after the initial acquisition precheck, later C4 receipts can reset the prior target-contact baseline from one to zero and still pass transition replay. Independently, forged per-attempt or cumulative fields pass after recomputing the receipt's self-digest.

These defects directly affect the requested cleanup-custody and cumulative-counter authority. The current committed artifacts are internally consistent, but the production gate does not mechanically preserve that truth for acquisition.

## Scope

Read-only review of the C4 affinity repair, C3-to-C4 attempt versioning, acquisition preconditions, cleanup custody, counter accounting, and prohibited-activity boundary at the exact audited commit. No broader tomography or scientific-result review was performed.

## Evidence Inspected

- Verified worktree, branch, `HEAD`, and branch tip all resolve to the audited commit.
- Confirmed all four C3 active artifacts were moved by exact blob-preserving renames into `SENSOR_AUTHORITY_ATTEMPT_HISTORY/c3_55e059bc_failed_affinity_precheck/`.
- Confirmed all seven active C4 authority, discovery, transport, attempt, journal, challenge, and cleanup paths are absent from the commit tree.
- Recomputed all four archived file SHA-256 values; each matches `ATTEMPT_HISTORY_INDEX.json`.
- Confirmed C3 counters are `1/0/0/0`; cleanup is owner-marker scoped, passed, and independently records remote-root absence.
- Recomputed `source_hashes_sha256` as `7f707088d83205d45c53d1560fe5c127e4b9e1194c7291fd6e35228aab038f26`.
- Recomputed bundle SHA-256 as `337718f31ffee3011ddc47e6fcb1606fbae846a4c7d9e62aff743192fef9be34`; all nine members exactly match source authority.
- Recomputed runtime SHA-256 as `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`.
- Confirmed the C4 operational pin probe tests CPUs 4 and 5 in isolated children before sensor enumeration and rejects syscall, readback, execution-CPU, or parent-affinity failures.
- Confirmed the current manifest remains blocked pending C4 review, with active counters `0/0/0/0` and cumulative counters `1/0/0/0`.

## Attempted Attacks

- Traced inherited-mask exclusion with successful operational pinning; C4 correctly permits this case.
- Traced `EINVAL`, `EPERM`, readback mismatch, execution-CPU mismatch, and parent-affinity drift; each fails before inventory.
- Traced C3 source reuse, occupied active namespace, invalid history, and missing C4 review; each blocks before contact.
- Traced archive omission via an empty `archived_files` map; the current validator incorrectly accepts it after self-digest recomputation.
- Traced history loss between attempt states and forged aggregate counters; transition replay incorrectly accepts the resulting accounting.
- Traced cleanup owner mismatch and residual remote root; the cleanup/absence path correctly prevents success.
- No SSH, SCP, ping, target inspection, PMU access, runtime execution, or tomography was performed.

## Boundary Attestations

- `no_file_edits: true`
- `no_git_write: true`
- `no_checkout_mutation: true`
- `no_target_contact: true`
- `no_ssh_scp_ping: true`
- `no_live_authority: true`
- `no_runtime_execution: true`
- `no_pmu: true`
- `no_tomography: true`
- `tests_reexecuted: false`

## Verdict

MATERIAL_BLOCKER

final_response=true


---

## source/bundle/runtime evidence auditor

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


---

## claim-boundary adjudicator

# C4 Source-Authority Review

- **Date:** 2026-07-16
- **Role:** claim-boundary adjudicator
- **Model:** OpenAI Codex (GPT-5)
- **Agent/thread ID:** not visible
- **Review mode:** read-only static inspection
- **Worktree:** `D:\CCC 2.0\AI\agent-governance-system-family10h`
- **Branch:** `codex/family10h-tomography-repair`
- **Audited commit:** `092d0a655e94d7c00f69efc1236cf1c8a2896ee1`
- **Parent/C3 failure-evidence commit:** `35844e76317017a73dc0fa83f7e976642b80c66f`

## Scope

Reviewed only the C4 affinity-precheck repair, attempt versioning, source-authority review selection, preserved C3 evidence, contact accounting, claim boundaries, and acquisition gating.

No files or Git state were modified. No checkout, target contact, network operation, SSH, SCP, ping, PMU access, live-authority path, package code, self-test, runtime binary, or tomography operation was executed.

## Evidence Inspected

- Exact worktree, branch, local tracking ref, clean status, commit identity, parent, commit delta, and whitespace check.
- `family10h_carrier_tomography_target.py` affinity-precheck implementation and committed regression evidence.
- `run_family10h_carrier_tomography_v1.py` C3/C4 review routing, history validation, counter accounting, acquisition ordering, manifest generation, and exact-object replay paths.
- `CARRIER_TOMOGRAPHY_CONTRACT.md`, implementation manifest and sidecar, source-hash receipt, source bundle, runtime binary, controller/target/offline committed self-test receipts, C3 review records, and attempt-history index.
- All four C3 artifacts before and after archival. Each archived artifact has the identical Git blob object as its former active-path counterpart.
- Source-bundle member list and blob identity for both modified Python sources.

Required authority bindings were confirmed:

```text
source_hashes_sha256=7f707088d83205d45c53d1560fe5c127e4b9e1194c7291fd6e35228aab038f26
source_bundle_sha256=337718f31ffee3011ddc47e6fcb1606fbae846a4c7d9e62aff743192fef9be34
runtime_binary_sha256=e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89
```

## Attempted Attacks

1. **Scope-expansion attack:** Compared the C4 delta against its parent. Behavioral source changes are confined to operational affinity probing and C3/C4 attempt/review versioning. The contract, public schedule, public model, C runtime sources, runtime binary, experimental design, and claim vocabulary are unchanged.
2. **Inherited-mask attack:** Verified that exclusion from the inherited affinity mask is no longer treated alone as failure. Each required CPU must pass a child-process singleton `sched_setaffinity`, affinity readback, optional actual-CPU check, and parent-affinity restoration. `EINVAL`, `EPERM`, readback mismatch, execution-CPU mismatch, and parent-mask mutation fail closed.
3. **C3 replay attack:** The exact C3 source-authority commit is explicitly rejected for another acquisition. Any non-C3 source commit selects separate C4 review paths and remains bound to exact source, bundle, runtime, body, and receipt hashes.
4. **Missing or forged C4 review attack:** Acquisition requires four distinct C4 roles, unique agent and thread identities, final responses, clear verdicts, exact audited commit and hashes, canonical archived bodies, matching detached receipts, and no-write/no-live attestations. C4 review artifacts are absent at the audited commit, so the gate is currently closed before challenge creation or target contact.
5. **C3 evidence laundering attack:** The history validator requires exactly one failed C3 record with the original source commit, failure-evidence commit, failure reason, counters, cleanup result, remote-root absence, archive hashes, and index digest.
6. **Counter-flattening attack:** Active C4 counters are independently zero. The prior and cumulative lane counters retain the C3 target contact instead of resetting history.
7. **Claim-promotion attack:** The package remains `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED`; the claim ceiling remains a route-scoped public carrier-state model; final exact-object verification remains false; and the manifest states `this_task_authorizes_live_execution=false`.
8. **PMU/runtime substitution attack:** The repaired discovery path adds only the affinity capability probe before sensor enumeration. It does not open PMU state, launch the tomography runtime, or create the tomography output root.

## Findings

- The C4 behavioral repair is limited to **AFFINITY-PRECHECK-01** and **ATTEMPT-VERSIONING-01**. Those literal identifiers are not embedded in the tree, but no third behavioral repair family appears in the audited delta.
- There is no general assurance expansion. The operational probe establishes only present pin capability for CPUs 4 and 5; it does not claim runtime success, PMU authority, tomography validity, or a stronger scientific result.
- C3 remains a failed attempt at `target_command_invoked`, with `passed=false`, the original affinity failure, one target contact, zero sensor inventories, zero live invocations, and zero PMU acquisitions.
- C3 review records and claim classifications are unchanged. The C4 manifest does not reuse or reinterpret C3 clearance.
- The active C4 namespace is empty and reports `0/0/0/0`. Cumulative accounting reports exactly one C3 target contact and zero inventory/live/PMU counts.
- No live target contact, PMU acquisition, or tomography execution is authorized by the audited package state. The acquisition capability remains an explicit future operation and is currently blocked by the absent C4 review overlay.
- Non-material wording only: the acquisition exception still says "source authority C3 review gate failed" after C4 path selection. The actual path binding and control flow use C4 and remain fail-closed.
- Final worktree status remained clean after inspection.

## Verdict

NO_MATERIAL_BLOCKER

final_response=true
