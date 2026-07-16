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
