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
