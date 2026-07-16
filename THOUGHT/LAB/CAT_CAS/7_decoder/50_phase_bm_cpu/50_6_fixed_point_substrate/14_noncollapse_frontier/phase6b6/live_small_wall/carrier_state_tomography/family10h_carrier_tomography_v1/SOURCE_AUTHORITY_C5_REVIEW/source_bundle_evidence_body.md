# C5 Source-Authority Review

**Role:** source/bundle/runtime evidence auditor  
**Model:** GPT-5 Codex  
**Agent/thread ID:** `019f5815-7922-7a53-b022-32f2c8d8c03f`  
**Audited commit:** `ca8f8490e9d2fc9b36debbfe7c927bfe2fde5c5e`  
**Verified parent:** `298049f515612a7a7bb2348cbe55cd86d33380fb`  
**Branch:** `codex/family10h-tomography-repair`  
**Custody:** READ_ONLY; final worktree clean and HEAD unchanged.

## Scope

Reviewed only C5 source hashes, deterministic bundle, runtime binary authority, compile-equivalence evidence, C4 affinity preservation, counter authority, historical metadata boundaries, failure persistence, sensor-inventory isolation, and blocked package state. No target contact, live execution, PMU access, tomography, Git mutation, checkout mutation, or file edits occurred.

## Evidence Inspected

- C5's 11-file commit diff and exact Git blobs.
- `CARRIER_TOMOGRAPHY_SOURCE_HASHES.json`, source bundle, runtime binary, runtime/controller/target self-test receipts, manifest and sidecar.
- Controller functions at `run_family10h_carrier_tomography_v1.py:303`, `:700`, `:1136`, `:1265`, `:1339`, `:2181`, `:2504`, `:2680`, `:2820`, and `:4913`.
- Target source-authority, affinity, inventory, and mode-selection functions at `family10h_carrier_tomography_target.py:293`, `:331`, `:2142`, `:2286`, `:2518`, and `:3338`.
- C3 attempt history and both affinity-observation records.

## Attempted Attacks

- Recomputed canonical JSON digests independently.
- Reconstructed the deterministic tar/gzip bundle entirely in memory.
- Compared all nine bundle members against both the source-hash receipt and exact C5 Git blobs.
- Injected forged active counters and a legacy cumulative field into a valid seven-state journal while recomputing row digests; both were rejected.
- Injected historical counters of `999/999/999/999`; they remained non-authoritative and did not alter active-counter validation.
- Confirmed the legacy historical-counter authority accessor raises.
- Checked omission, mutation, runtime-blob mismatch, nonzero-target-return, and discovery-isolation negative evidence.
- Diffed C3, affinity, target, runtime source/header, and runtime binary paths against the parent; all were unchanged.

## Findings

1. The supplied bindings are coherent:
   - `source_hashes_sha256=0d8889c6c0be3b5c571887c92abbc41cead95f27f6610ca091af74d5631e4797`
   - `source_bundle_sha256=29aec7a6e2d9bbde7850ded261e42fbf5655f21d5899cef579c311a8901f22f0`
   - `runtime_binary_sha256=e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`

   The source-hashes value is the canonical receipt digest; the raw JSON file hash is separately `0fe0f470...`.

2. Bundle reconstruction produced exactly 705,839 bytes and the expected hash. Its nine sorted members have normalized mode `0644`, timestamp `0`, UID/GID `0`, empty owner names, and byte-exact agreement with C5 Git blobs.

3. Runtime authority binds Git blob `3c007e278b7c3f2b206708739fd9abab5d3e91e7`, size 22,928, and the exact source/header hashes. The committed receipt records a strict GCC 9.4 compile with `-std=c11 -Wall -Wextra -Werror -O2`; isolated and committed outputs are byte-identical. Receipt integrity and bindings were independently reproduced, but compilation itself was not rerun because it would create an artifact.

4. C4 operational affinity remains intact. The successful preserved observation binds SHA-256 `a516a56a...`, confirms CPU 4 and CPU 5 singleton pin/readback, and leaves parent affinity unchanged.

5. Active counters are state-machine authority only: `0/0/0/0`, then `1/0/0/0`, then `1/1/0/0`. Legacy cumulative fields are rejected. Historical known counters total `3/0/0/0`, are explicitly reporting-only, and explicitly disclaim a complete cryptographic lane ledger.

6. A future nonzero target return captures full-output hashes plus bounded 8,192-character stdout/stderr records, seals cleanup state, atomically persists the failure receipt, and only then raises. The committed regression records this path passing.

7. Sensor discovery selects exactly one CLI mode, performs source validation, C4 affinity probes, sysfs inventory, and an `O_RDONLY` temperature read. It opens no PMU, launches no runtime, and creates no tomography output root.

8. The manifest remains `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED`. C5 review, sensor authority, active-attempt, target-failure, and final-object artifacts are absent; active counters are `0/0/0/0`, and live execution authorization is false.

## Recommendations

No source repair is required. Preserve the blocked state until the complete distinct C5 review quorum, sensor authority, and final exact-object evidence exist. Do not promote historical reporting counters into active transaction authority.

**Direct verdict:** NO_MATERIAL_BLOCKER

final_response=true
