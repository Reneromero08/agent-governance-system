# C3 Source-Authority Review

**Role:** physical sensor-authority auditor
**Model:** GPT-5 Codex
**Audited commit:** `55e059bc7acaafee3feacddac2069d7b5e40edd1`
**Branch:** `codex/family10h-tomography-repair`
**Custody:** READ_ONLY

## Evidence Inspected

- Exact HEAD and clean worktree; no checkout mutation.
- C2 blocker baseline at parent `7647e872122e75edd07bdd423d9493a2796c8fd9`.
- C3 diff: two source files plus regenerated authority and offline evidence.
- `family10h_carrier_tomography_target.py:28-46,284-430,1969-2015,2217-2393,2460-2814,2982-3015`.
- `run_family10h_carrier_tomography_v1.py:684-864,1939-2137,2453-2456,3041-3190,3428-3512,4279-4408,4498-4624`.
- Controller, target, runtime, deployment, offline-validation, manifest, source-hash, and bundle receipts.
- Independent in-memory receipt and artifact verification:
  - Source authority: `997879a0b69393074762a4b3d2f0957250e640506ba0b2cb05c11cfbdc53517f`.
  - Bundle: `b58b0967018fb0e41caa5b4bf494e2bf66542b9770409b4dd763d641467a9b43`, byte-identical deterministic reconstruction.
  - Runtime: SHA-256 `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`, blob `3c007e278b7c3f2b206708739fd9abab5d3e91e7`, size `22928`.
  - All top-level receipt digests and the manifest sidecar recomputed exactly.
- No target program, runtime, package self-test, sensor inventory, PMU, SSH, SCP, ping, or live-authority path was executed.

## Attempted Attacks

- Omitted runtime, bundle, source-hash receipt, and individual source-file cases.
- Runtime byte, size, SHA-256, and Git-blob substitution.
- Extra transfer-root file and transfer-keyset drift.
- Production SCP-list divergence from target-required files.
- Runtime-only evidence-overlay mutation and defective replay with runtime comparison removed.
- Runtime or PMU smuggling through the discovery call graph.
- Mode confusion between discovery and authorized execution.
- Sensor enumeration before transfer-root validation.
- Claim-ceiling, source/receiver CPU, package-decision, and offline-counter drift.

## Findings

1. C3 repairs `C2-PHYS-DISCOVERY-RUNTIME-TRANSFER-GAP`. The production set is the exact 13-file union: contract, three schedule artifacts, public module, target module, runtime C/header, controller module, source-hash receipt, source bundle, normalized findings, and runtime binary.

2. Snapshot materialization, transfer-plan construction, outbound SCP iteration, target validation, and fixtures all consume the shared `DISCOVERY_TRANSFER_FILE_NAMES`. The plan is validated before first contact and includes the runtime binary.

3. Target discovery validates the complete fresh transfer root and challenged source/bundle/runtime identities before platform inspection or sensor enumeration. Discovery then opens only the pinned `k10temp` sysfs temperature input. Runtime execution is confined to the mutually exclusive authorized-execution mode; no PMU path exists in the discovery call graph.

4. The production-shaped fresh-root regression validates all 13 files, rejects omissions, mutations, extras, and runtime blob mismatch, and records no runtime, PMU, or output-root activity. The target fixture independently executes synthetic inventory with `1/1/0/0`, `pmu_open_count=0`, `runtime_launch_count=0`, and no tomography output root.

5. Runtime authority is bound through commit-object verification, source receipt, transfer-plan SHA/size/blob checks, controller challenge, target validation, and final replay. The former runtime-overlay proxy is replaced by actual source/evidence commits and `replay_final_exact_objects()` invocation.

6. Claim boundaries did not widen. The contract, public schedule, public module, runtime sources, and runtime binary are unchanged from C2. The claim ceiling remains route-scoped; source CPU `4` and receiver CPU `5` remain fixed.

7. Offline package state remains `0/0/0/0`, has no approved sensor authority, authorizes no live execution, lacks final exact-object authority, and remains `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED`.

8. Non-material evidence note: the controller regression's `no_runtime_binary_executed` and `no_pmu_path_opened` fields are literal `True`, and its recorded commit-blob check used `source_commit=null` during pre-commit generation. Current isolation is nevertheless established by the reviewed call graph, target fixture, exact C3 object verification, and runtime-overlay replay.

## Recommendations

- Accept C3 for the physical sensor-authority role while preserving the blocked package decision.
- Do not acquire sensor authority or authorize tomography until the full C3 review quorum and later authorization gates are satisfied.
- Harden future evidence by running full synthetic discovery from the production-plan-populated root with instrumented runtime/PMU sentinels and an exact clean-head commit binding.

## Boundary Attestations

`no_git_write=true`
`no_file_edits=true`
`no_checkout_mutation=true`
`no_target_contact=true`
`no_live_authority=true`
`no_pmu=true`

## Final Verdict

NO_MATERIAL_BLOCKER

BODY_COMPLETE
