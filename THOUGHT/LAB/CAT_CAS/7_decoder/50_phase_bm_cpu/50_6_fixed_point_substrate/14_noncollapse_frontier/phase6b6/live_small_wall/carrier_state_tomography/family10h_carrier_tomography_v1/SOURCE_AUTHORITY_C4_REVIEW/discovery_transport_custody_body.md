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
