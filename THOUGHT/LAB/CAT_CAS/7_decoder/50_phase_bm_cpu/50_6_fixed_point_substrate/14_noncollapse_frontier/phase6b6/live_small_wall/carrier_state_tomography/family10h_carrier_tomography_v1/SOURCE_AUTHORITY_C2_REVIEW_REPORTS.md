# Family 10h C2 Source-Authority Review Reports

Audited commit: `0b15890e9a8cad95cee38a2d4a333fda8a00dd50`
Source hashes SHA-256: `489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124`
Source bundle SHA-256: `8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef`
Runtime binary SHA-256: `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`
Runtime binary Git blob: `3c007e278b7c3f2b206708739fd9abab5d3e91e7`
Decision: `FAMILY10H_SOURCE_AUTHORITY_C2_REVIEW_BLOCKED`
Package decision: `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED`
Contact counters: `0/0/0/0`

## Material Blockers

### C2-PHYS-DISCOVERY-RUNTIME-TRANSFER-GAP

Originating agent: `019f6980-eede-71b3-929b-ba9be647dbd4`
Independently reproduced: `true`
Mechanism: Production discovery transfers only SOURCE_AUTHORITY_FILE_NAMES to an initially empty remote root, but target-side challenge validation now requires the separate runtime authority binary to be present and byte-identical.
Consequence: A future authorized discovery would consume target contact and then fail before sensor inventory, leaving source authority unrecoverable for C2.
Minimal repair: Transfer SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES during discovery, or introduce a prospectively reviewed discovery-specific validator that preserves immutable runtime binding without requiring local binary presence.
Required regression: A production-shape fresh remote-source fixture must be populated by the exact outbound transfer list and validate successfully through target source authority before sensor inventory; it must preserve 1/1/0/0 accounting and zero runtime/PMU use.
Parent disposition: `FAMILY10H_SOURCE_AUTHORITY_C2_REVIEW_BLOCKED`

### C2-RUNTIME-OVERLAY-REPLAY-REGRESSION-NOT-END-TO-END

Originating agent: `019f6981-6257-7102-a066-0b91dfb33047`
Independently reproduced: `true`
Mechanism: The regression mutates the runtime binary and observes changed blob identity, but does not construct source/evidence commits and invoke replay_final_exact_objects().
Consequence: The test could stay green if future replay logic stopped checking RUNTIME_AUTHORITY_FILE_NAMES, so C1 runtime immutability closure is not mechanically proven.
Minimal repair: Replace the tautological check with an evidence-overlay regression that mutates only the runtime binary and requires final exact-object replay to fail on the runtime entry.
Required regression: Construct source and descendant evidence commits, preserve source-authority blobs, change only the runtime binary, invoke replay_final_exact_objects(), and assert the runtime entry is reported changed; demonstrate the regression fails if runtime comparison is removed.
Parent disposition: `FAMILY10H_SOURCE_AUTHORITY_C2_REVIEW_BLOCKED`

## Archived Reviewer Bodies and Receipts

## physical sensor-authority auditor

Agent/thread ID: `019f6980-eede-71b3-929b-ba9be647dbd4`
Model: `GPT-5.6 Sol`
Verdict: `MATERIAL_BLOCKER`
Body canonical SHA-256: `3621a61916cab0d5bab1f2bf416774ad80ba1beb8f08444cdd89f034f7ebc496`
Body file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/physical_sensor_authority_body.md`
Receipt file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/physical_sensor_authority_receipt.json`

### Report Body

# C2 Physical Sensor-Authority Review

**Identity**

Role: physical sensor-authority auditor  
Model: GPT-5.6 Sol  
Custody: READ_ONLY  
Audited commit: `0b15890e9a8cad95cee38a2d4a333fda8a00dd50`  
Audited source hashes SHA-256: `489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124`  
Source bundle SHA-256: `8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef`  
Runtime binary SHA-256: `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`  
Runtime binary Git blob: `3c007e278b7c3f2b206708739fd9abab5d3e91e7`

**Verdict**

MATERIAL_BLOCKER

**Determination**

C2 repairs the original `C1-PHYS-RUNTIME-IMMUTABILITY` substitution mechanism. The exact runtime binary is now recorded by blob ID, SHA-256, and size; checked against the C2 commit; bound through the source receipt, reviewer quorum, challenge, manifest, and final replay; and included in the cross-commit immutable-object comparison.

The package nevertheless has a new material physical-authority blocker: the production discovery transport does not upload the runtime binary, while target-side discovery now requires that binary to be present and byte-identical. Consequently, a future authorized one-shot discovery would contact the target and then fail before sensor inventory.

The package correctly remains `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED` before real authority acquisition.

**Evidence Reviewed**

- Exact worktree head and branch were confirmed. Final status had no changed or untracked entries.
- All nine committed source files matched their source-hash records and sizes.
- The canonical source-hash digest independently recomputed to the audited value.
- The source bundle independently matched its audited hash and contained exactly nine normalized source members with matching content and metadata.
- The committed runtime independently matched both the audited SHA-256 and Git blob ID.
- Reviewed the C1 normalized findings and physical reviewer report, C2 commit delta, contract, source authority code, target discovery path, controller transport path, manifest, and committed target/controller self-test receipts.
- No package code, tests, runtime, target command, sensor inventory, or PMU path was executed.

**Attempted Attacks**

- Binary-only substitution is rejected by runtime blob, SHA-256, and size validation.
- Coherent binary plus mutable receipt/manifest substitution is rejected by final replay comparing `SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES` across source and evidence commits.
- Missing runtime authority fails closed.
- Live-environment injection during discovery fails before inventory.
- Target mode confusion fails because exactly one of discovery, authorized execution, or self-test must be selected.
- Counter forgery is constrained by strict integer checks; booleans and nonzero live/PMU counters are rejected.
- Production-shape fresh-target transfer was traced statically and exposed the blocker below.

**Findings**

1. **`C2-PHYS-DISCOVERY-RUNTIME-TRANSFER-GAP` — material.**  
   `run_family10h_carrier_tomography_v1.py:1907-1912` creates a fresh remote root, but `:1945-1949` uploads only `SOURCE_AUTHORITY_FILE_NAMES`. The runtime is deliberately separate in `family10h_carrier_tomography_target.py:39-45`. Target challenge validation calls full source validation at `:2160-2162`, which requires the runtime binary at `:301-306` and `:353-366`. Discovery therefore fails before platform validation and sensor enumeration at `:2238-2244`.

2. The controller records `target_contact_count = 1` before issuing the first SSH operation at `run_family10h_carrier_tomography_v1.py:1917-1937`. Thus this failure consumes target contact while leaving sensor inventory, live invocation, and PMU acquisition at zero. Existing attempt artifacts then reject a second attempt at `:1822-1833`.

3. The committed green fixtures do not model this production file set. `copy_source_fixture()` supplies both source and runtime authority files, while production transport supplies only source authority files.

4. The runtime-overlay regression is weaker than its name: `run_family10h_carrier_tomography_v1.py:3098` infers final-replay rejection solely from unequal blob IDs rather than constructing an evidence overlay and invoking `replay_final_exact_objects()`.

5. Sensor-authority accounting is otherwise correct. Successful discovery is target contact and one sensor-inventory transaction, represented as `1/1/0/0`; it is not a tomography live invocation. Discovery reports zero PMU opens, zero runtime launches, and no tomography output root. Authorized runtime execution is a mutually exclusive mode.

6. The current manifest truthfully records counters `0/0/0/0`, no approved physical sensor identity, no C2 review quorum, no final evidence commit, and `this_task_authorizes_live_execution: false`.

**Recommendations**

1. Keep the package blocked.
2. Make discovery transport and target validation consistent: either upload the exact C2 runtime authority blob without executing it, or use a discovery-specific validator that verifies its immutable C2 binding without requiring local binary presence.
3. Add a production-shape regression using a fresh remote-source fixture populated by the exact upload list; require discovery to reach `1/1/0/0` while preserving zero runtime launches and PMU opens.
4. Add the required commit-level coherent-overlay regression that invokes final exact-object replay and proves a runtime-only evidence mutation fails.
5. Do not acquire real sensor authority until this repair receives a fresh read-only review.

**Boundary Attestations**

- `no_git_write`: true
- `no_file_edits`: true
- `no_checkout_mutation`: true
- `no_target_contact`: true
- `no_live_authority`: true
- `no_pmu`: true
- `sensor_inventory_execution`: false
- `runtime_binary_execution`: false

### Detached Receipt

```json
{
  "schema": "FAMILY10H_SOURCE_AUTHORITY_C2_REVIEWER_RECEIPT_V2",
  "issuer": "codex_subagent_read_only_review",
  "receipt_kind": "detached_review_body_acknowledgment",
  "thread_id": "019f6980-eede-71b3-929b-ba9be647dbd4",
  "agent_id": "019f6980-eede-71b3-929b-ba9be647dbd4",
  "role": "physical sensor-authority auditor",
  "model": "GPT-5.6 Sol",
  "review_body_sha256": "3621a61916cab0d5bab1f2bf416774ad80ba1beb8f08444cdd89f034f7ebc496",
  "review_body_canonicalization": "utf8_lf_single_trailing_newline",
  "audited_commit": "0b15890e9a8cad95cee38a2d4a333fda8a00dd50",
  "source_hashes_sha256": "489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124",
  "source_bundle_sha256": "8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef",
  "runtime_binary_sha256": "e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89",
  "no_git_write": true,
  "no_file_edits": true,
  "no_checkout_mutation": true,
  "no_target_contact": true,
  "no_live_authority": true,
  "no_pmu": true,
  "self_authored": false,
  "evidence_origin": "codex_subagent_detached_receipt"
}
```

## discovery transport and custody auditor

Agent/thread ID: `019f6981-29b2-7841-980c-31bf19b22fd6`
Model: `GPT-5.6 Sol`
Verdict: `MATERIAL_BLOCKER`
Body canonical SHA-256: `59379d0b67cfcb24eccc5cbb1ad0f295c4f136f4d275d708f94e51087d929ec1`
Body file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/discovery_transport_custody_body.md`
Receipt file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/discovery_transport_custody_receipt.json`

### Report Body

# C2 Source-Authority Review

Role: discovery transport and custody auditor  
Model: GPT-5.6 Sol  
Audited commit: `0b15890e9a8cad95cee38a2d4a333fda8a00dd50`  
Audited source hashes: `489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124`  
Source bundle: `8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef`  
Runtime binary SHA-256: `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`  
Runtime binary Git blob: `3c007e278b7c3f2b206708739fd9abab5d3e91e7`  
Verdict: MATERIAL_BLOCKER

**Evidence Reviewed**
- Branch and `HEAD` both resolved to the audited commit. The package had zero tracked differences and zero non-ignored untracked files.
- All nine source blobs matched the committed source-hash receipt by SHA-256 and size.
- The source-hash canonical digest recomputed exactly.
- The source bundle was reconstructed entirely in memory from the nine committed blobs and was byte-identical to the committed bundle.
- The runtime binary’s SHA-256, size, and Git blob ID matched its committed authority record.
- Reviewed the exact committed [controller](</D:/CCC 2.0/AI/agent-governance-system-family10h/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/run_family10h_carrier_tomography_v1.py:1816>) and [target](</D:/CCC 2.0/AI/agent-governance-system-family10h/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/family10h_carrier_tomography_target.py:28>) source, source-authority records, manifest, and committed self-test receipts.
- Committed self-test receipts report passing, but were not executed because their workflows write files.
- The audited manifest remains `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED`, with the C2 review gate failed, final exact-object verification failed, counters `0/0/0/0`, and live authorization false.

**Attempted Attacks**
- Commit, source-hash, bundle, runtime-hash, and runtime-blob substitution were traced through challenge construction and final exact-object replay. Those bindings reject coherent source overlays.
- Forged challenge, wrong nonce, replayed challenge, changed source-review binding, and alternate remote-root scope were checked against exact keysets, canonical digests, nonce ownership, and immutable journal fields.
- Copyback corruption was checked against remote/local SHA-256 equality and full authority-chain validation.
- Cleanup failure, owner-marker mismatch, and failed remote-root absence were checked. They prevent authority publication and leave a sealed failure-custody receipt.
- Journal state skipping, duplication, reordering, post-terminal append, boolean counters, and binding mutation were checked against the exact seven-state replay and final snapshot equality.
- Discovery/live mode confusion was checked. Target CLI modes are mutually exclusive, and discovery performs no PMU acquisition, runtime launch, or tomography output.
- A transfer-set omission attack succeeded statically: the required runtime binary is not present in the production SCP set.

**Findings**
1. The live discovery transport omits `family10h_carrier_tomography_runtime`. C2 separated it into `RUNTIME_AUTHORITY_FILE_NAMES`; snapshot materialization, fixtures, deployment tests, mutation tests, and final replay all use `SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES`. The production SCP loop at controller line 1945 still iterates only `SOURCE_AUTHORITY_FILE_NAMES`.

2. Target validation now requires the runtime binary. `validate_source_file_authority()` calls `validate_runtime_binary_authority()`, which rejects a missing binary, and this runs during challenge validation before sensor enumeration. Therefore, after a future C2 review quorum opens the pre-contact gate, the controller will contact the target and transfer the source set, but target discovery will deterministically fail because the runtime binary was never copied.

3. This failure is authority-safe but operationally blocking. Cleanup and absence verification are attempted, no sensor inventory or live execution occurs, no successful authority can be produced, and second attempts are blocked. However, runtime custody is not preserved through transport, and a successful discovery transaction is impossible.

4. The pre-contact review gate itself is correctly ordered. Exact source commit, source hashes, deterministic bundle, runtime hash, four-role review archives, distinct reviewer/thread identities, and boundary attestations are validated before the first SSH command. The challenge, transport, attempt journal, and final replay retain these bindings.

**Recommendations**
1. Change the production transfer loop to the exact union `SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES`.
2. Add a pre-contact transfer-plan assertion requiring exact equality with the target-required file set, including runtime SHA-256, size, and Git blob ID.
3. Add a regression over the production command-building path that verifies every required outbound SCP destination. Existing fixtures copy both sets and therefore conceal this defect.
4. Validate the transported bundle file itself against the challenged bundle hash for explicit byte-custody evidence.
5. Because repairing the controller changes a source-authority blob and deterministic bundle, create a new source-authority commit, regenerate the source hashes and bundle, and repeat all C2 reviews. Do not reuse evidence bound to this commit.

**Boundary Attestations**
- `no_git_write: true`
- `no_file_edits: true`
- `no_checkout_mutation: true`
- `no_target_contact: true`
- `no_live_authority: true`
- `no_pmu: true`

Only local read-only Git-object inspection and in-memory hashing were used. I did not invoke SSH, SCP, ping, the target program, PMU access, live-authority variables, network access, checkout operations, or Git configuration writes.

### Detached Receipt

```json
{
  "schema": "FAMILY10H_SOURCE_AUTHORITY_C2_REVIEWER_RECEIPT_V2",
  "issuer": "codex_subagent_read_only_review",
  "receipt_kind": "detached_review_body_acknowledgment",
  "thread_id": "019f6981-29b2-7841-980c-31bf19b22fd6",
  "agent_id": "019f6981-29b2-7841-980c-31bf19b22fd6",
  "role": "discovery transport and custody auditor",
  "model": "GPT-5.6 Sol",
  "review_body_sha256": "59379d0b67cfcb24eccc5cbb1ad0f295c4f136f4d275d708f94e51087d929ec1",
  "review_body_canonicalization": "utf8_lf_single_trailing_newline",
  "audited_commit": "0b15890e9a8cad95cee38a2d4a333fda8a00dd50",
  "source_hashes_sha256": "489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124",
  "source_bundle_sha256": "8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef",
  "runtime_binary_sha256": "e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89",
  "no_git_write": true,
  "no_file_edits": true,
  "no_checkout_mutation": true,
  "no_target_contact": true,
  "no_live_authority": true,
  "no_pmu": true,
  "self_authored": false,
  "evidence_origin": "codex_subagent_detached_receipt"
}
```

## source/bundle/runtime evidence auditor

Agent/thread ID: `019f6981-6257-7102-a066-0b91dfb33047`
Model: `GPT-5.6 Sol`
Verdict: `MATERIAL_BLOCKER`
Body canonical SHA-256: `4013016d02836fc24306b9ff490ab365bd4b9f415f2be6374cbdfbf4f806107b`
Body file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/source_bundle_evidence_body.md`
Receipt file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/source_bundle_evidence_receipt.json`

### Report Body

# C2 Source-Authority Review

**Role:** source/bundle/runtime evidence auditor  
**Model:** GPT-5.6 Sol  
**Audited commit:** `0b15890e9a8cad95cee38a2d4a333fda8a00dd50`  
**Audited source hashes SHA-256:** `489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124`  
**Source bundle SHA-256:** `8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef`  
**Runtime binary SHA-256:** `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`  
**Runtime binary Git blob:** `3c007e278b7c3f2b206708739fd9abab5d3e91e7`  
**Verdict:** MATERIAL_BLOCKER

## Evidence Reviewed

- The worktree was clean on `codex/family10h-tomography-repair` at the exact audited commit before and after review.
- Commit-object verification passed for all nine source files, the canonical source-authority digest, deterministic source bundle, runtime SHA-256, runtime size, and runtime Git blob.
- The bundle reconstructed from the audited commit produced the supplied SHA-256 and size `692977`. It contained exactly nine sorted regular-file members. Every member matched its recorded source hash and size, with normalized mode `0644`, timestamp `0`, UID/GID `0`, and empty owner/group names.
- The runtime authority record correctly binds the binary hash/blob/size, runtime C and header hashes, C11 compilation flags, and byte-exact isolated-compile law.
- The committed runtime receipt is internally digest-valid and records GCC `9.4.0`, a successful isolated compile, and byte equality at `22928` bytes with the committed runtime.
- The manifest and sidecar, controller self-test, runtime self-test, and offline-validation receipts passed independent canonical-digest checks. The package remains correctly blocked pending C2 review.
- The detached source-review protocol, final exact-object replay, runtime-overlay regression, and the normalized C1 findings were inspected directly.

## Attempted Attacks

- Mutating only the runtime binary was rejected by runtime authority through Git-blob, SHA-256, and size mismatches.
- Mutating the binary and coherently regenerating its local source-hash authority made the copied source root internally consistent, demonstrating the original C1 attack rather than assuming it away.
- The production replay implementation was inspected for that coherent overlay. It compares every source-authority blob plus the runtime binary and records a failure when any blob differs.
- Detached-review attacks covering the old self-referential schema, missing body or receipt, post-acknowledgment body/receipt mutation, invalid body digest, wrong commit/source/bundle/runtime identity, wrong thread/agent/role, missing fourth reviewer, duplicate reviewer, reused prior reviewer, and asserted parent/self/target provenance were exercised and rejected.
- A fresh isolated compilation was attempted. The audit environment exposed no usable local C compiler, so no independent recompilation result is claimed; assessment of compile equivalence relies on the committed, internally valid receipt and static compile path.

## Findings

1. **Material: the required runtime-overlay replay regression is not end to end.**  
   `runtime_binary_overlay_mutation_regression()` mutates the binary and proves that its blob identity changed, but its `final_replay_policy_rejects_changed_runtime_blob` check merely repeats that blob-inequality expression. It never creates source/evidence commits and never invokes `replay_final_exact_objects()`. The regression would remain green if the production replay later stopped checking `RUNTIME_AUTHORITY_FILE_NAMES`.

   The production guard at `run_family10h_carrier_tomography_v1.py:4499` through `4512` appears correct, but C1 explicitly required a binary-only evidence-overlay mutation followed by failure of final exact-object replay. C2 does not mechanically exercise that acceptance condition.

2. **C1-RECEIPT-SELF-HASH-UNSATISFIED is repaired for this role.**  
   The body and acknowledgment are separate objects. The acknowledgment covers a canonical UTF-8/LF body, the old self-referential field is excluded and rejected, and archived body/acknowledgment bytes are independently replayed from the evidence commit.

3. **C1-PHYS-RUNTIME-IMMUTABILITY is implemented but not closed for this role.**  
   Direct binary authority and the production source-to-evidence blob comparison repair the mechanism. Closure remains blocked by the non-exercising regression described above.

4. **Non-blocking hardening:** the authoritative compiler description is generic while the runtime receipt records the concrete GCC version. Pinning a compiler package or image digest would make future isolated reproduction less environment-dependent. The normalized verdict/finality should also be included in the detached acknowledgment or derived mechanically from the acknowledged body.

## Recommendations

1. Replace the tautological overlay check with a regression that constructs source and descendant evidence commits, changes only the runtime binary, preserves the source-authority blobs, invokes final exact-object replay, and asserts the runtime entry is reported changed.
2. Demonstrate that this regression fails if runtime authority is removed from the replay comparison.
3. Re-run isolated compilation using the pinned GCC toolchain and preserve its concrete toolchain identity.
4. After modifying the controller test, regenerate and verify the source-authority digest and deterministic bundle before requesting another review.

## Boundary Attestations

- `no_git_write: true`
- `no_file_edits: true`
- `no_checkout_mutation: true`
- `no_target_contact: true`
- `no_live_authority: true`
- `no_pmu: true`

### Detached Receipt

```json
{
  "schema": "FAMILY10H_SOURCE_AUTHORITY_C2_REVIEWER_RECEIPT_V2",
  "issuer": "codex_subagent_read_only_review",
  "receipt_kind": "detached_review_body_acknowledgment",
  "thread_id": "019f6981-6257-7102-a066-0b91dfb33047",
  "agent_id": "019f6981-6257-7102-a066-0b91dfb33047",
  "role": "source/bundle/runtime evidence auditor",
  "model": "GPT-5.6 Sol",
  "review_body_sha256": "4013016d02836fc24306b9ff490ab365bd4b9f415f2be6374cbdfbf4f806107b",
  "review_body_canonicalization": "utf8_lf_single_trailing_newline",
  "audited_commit": "0b15890e9a8cad95cee38a2d4a333fda8a00dd50",
  "source_hashes_sha256": "489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124",
  "source_bundle_sha256": "8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef",
  "runtime_binary_sha256": "e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89",
  "no_git_write": true,
  "no_file_edits": true,
  "no_checkout_mutation": true,
  "no_target_contact": true,
  "no_live_authority": true,
  "no_pmu": true,
  "self_authored": false,
  "evidence_origin": "codex_subagent_detached_receipt"
}
```

## claim-boundary adjudicator

Agent/thread ID: `019f6981-9b41-78e2-bc90-54849d98f2ec`
Model: `GPT-5.6 Sol`
Verdict: `MATERIAL_BLOCKER`
Body canonical SHA-256: `fbf3f5d9915a0aff05959e5372a848fac0abbac65b1d27ab07fad043a178aaa6`
Body file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/claim_boundary_body.md`
Receipt file: `THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/carrier_state_tomography/family10h_carrier_tomography_v1/SOURCE_AUTHORITY_C2_REVIEW/claim_boundary_receipt.json`

### Report Body

# C2 Source-Authority Claim-Boundary Review

**Role:** claim-boundary adjudicator  
**Model:** GPT-5.6 Sol  
**Effort:** Max  
**Audited commit:** `0b15890e9a8cad95cee38a2d4a333fda8a00dd50`  
**Branch:** `codex/family10h-tomography-repair`  
**Verdict:** MATERIAL_BLOCKER

## Audited Source Authority

**Canonical source-hashes SHA-256:** `489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124`

All nine source-file sizes and SHA-256 values matched:

- `CARRIER_TOMOGRAPHY_CONTRACT.md`: `3c138cd9b79d58d837ed4e1ea5bf2f9b1f6fda6541669028f9c97faf33823a08`
- `CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.json`: `9658ea725d69ae384c85c5664505e59497d1e3291376128ee2160824ee7e0648`
- `CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.sha256`: `0eb44bbd0e0cd1f5a637fedc747acbd6f4069ad51aa52b73ad7aa86e17f94880`
- `CARRIER_TOMOGRAPHY_PUBLIC_SCHEDULE.tsv`: `b788a89f66eeb1074f5a8a591f669933efaaaa2ccfd7ef9ebb3715104fc25690`
- `family10h_carrier_tomography_public.py`: `873c8a8f936e07d8d72f55c4a67100546591de9e9c70477a048313ba4efdb18b`
- `family10h_carrier_tomography_runtime.c`: `bff8d13a02b375029625164b98fb98e9c1e0f134c3b21a743f1b3a13770f85f8`
- `family10h_carrier_tomography_runtime.h`: `9e9fd1cb3ac1395d76791f9578695e781eaa01467b73801a371bdbceeb0e403e`
- `family10h_carrier_tomography_target.py`: `d52d048c503e4528f9536f97269b949fc01071487a4eeaa1da3cbb25a3c2a625`
- `run_family10h_carrier_tomography_v1.py`: `ad5d7872c2171fb81210f51617bbd417d3d73755006f2fd96caf367038fbf21d`

**Source bundle SHA-256:** `8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef`  
The bundle contained exactly the nine source files, byte-matched them, and retained normalized `0644`, zero UID/GID, and zero-mtime metadata.

**Runtime binary SHA-256:** `e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89`  
**Runtime binary Git blob:** `3c007e278b7c3f2b206708739fd9abab5d3e91e7`  
**Runtime binary size:** `22,928` bytes.

## Evidence Reviewed

- Exact commit objects, source receipt, deterministic bundle, runtime object, contract, manifest and sidecar.
- Target discovery, authority validation, controller transport, mode dispatch, and eventual execution gates.
- Committed target/controller self-tests and their relevant counter, isolation, substitution, and authority regressions.
- C1 review reports and retained prior-review artifacts.
- C2 parent diff from `e08215dc140ae390820d050e947602fdea8c626a`.
- Initial and final worktree status were clean; branch and HEAD remained unchanged.
- No package script, runtime binary, live test, discovery transaction, or tomography path was executed.

## Attempted Attacks

- **Discovery/live-mode conflation:** Rejected. Target mode selection is exclusive, and discovery invokes only `--discover-temperature-sensor-authority`.
- **PMU or runtime smuggling:** Rejected by the discovery call graph. It reads platform/sysfs identity and one temperature sample, with no PMU open, runtime launch, or tomography output root.
- **Scientific-result smuggling:** Rejected. Discovery emits identity, custody, sample, challenge, and transport authority data; it emits no tomography records, feature packet, adjudication, or scientific result class.
- **Authority bypass:** Rejected. The current manifest is blocked, authority artifacts are absent, final exact-object verification is false, and all live-authority variables were absent.
- **Source/bundle/runtime substitution:** Hash, size, bundle-member, runtime SHA-256, and Git-blob bindings matched.
- **Retained-evidence mutation:** Rejected. `SMALL_WALL_STATE.md` retained blob `b13cf4afa5fbcfbd22ee00bf3d5906a5ace9c7e3`; C1 reports, prior subagent findings/reports, and live/checkpoint evidence were unchanged.
- **Production transfer completeness:** Succeeded as an attack and exposes the material blocker below.

## Findings

1. **Material blocker: the production discovery transfer omits the runtime binary.** The controller materializes `SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES`, but its SCP loop transfers only `SOURCE_AUTHORITY_FILE_NAMES`. The target now requires the runtime binary during `validate_source_file_authority`; the nonce-owned remote root starts empty. Consequently, target contact occurs, but validation fails with a missing runtime binary before sensor inventory can execute.

2. **Sensor discovery is target contact.** The controller records contact before its first SSH operation. A completed inventory is correctly modeled as `target_contact_count=1`.

3. **Sensor inventory is not a tomography live invocation.** Its intended completed counters are `1/1/0/0`; discovery neither starts the runtime nor creates tomography evidence.

4. **PMU acquisition remains zero.** Neither the reviewed code path nor this review opened or sampled a PMU.

5. **The package remains blocked before authority acquisition.** The manifest is `FAMILY10H_CARRIER_TOMOGRAPHY_PACKAGE_BLOCKED`, with current counters `0/0/0/0`, no sensor-authority receipt, failed source-review quorum, and failed final exact-object gate.

6. **Sensor inventory emits no scientific result.** The temperature sample is custody/readability evidence only.

7. **Eventual tomography remains separately unauthorized.** Even a repaired sensor-authority transaction would only permit a frozen-awaiting-authorization state. Execution still requires separate live, commit, manifest, runtime, and nonce authority.

8. **State and retained evidence remain unmodified.** C2 changed regenerated package source-authority and offline self-test artifacts only; this review changed nothing.

## Recommendations

1. Transfer `SOURCE_AUTHORITY_FILE_NAMES + RUNTIME_AUTHORITY_FILE_NAMES` during discovery, then regenerate the source authority and obtain fresh exact-commit C2 reviews.
2. Add a no-contact regression that stages the exact production transfer set into an initially empty directory and requires `validate_source_file_authority` to pass.
3. Preserve the boundary vocabulary: discovery is one target contact and one sensor inventory, never a tomography invocation, PMU acquisition, or scientific result.
4. Require a new explicit authorization for any eventual tomography execution.

## Boundary Attestations

- `no_git_write`: true
- `no_file_edits`: true
- `no_checkout_mutation`: true
- `no_target_contact`: true
- `no_live_authority`: true
- `no_pmu`: true

### Detached Receipt

```json
{
  "schema": "FAMILY10H_SOURCE_AUTHORITY_C2_REVIEWER_RECEIPT_V2",
  "issuer": "codex_subagent_read_only_review",
  "receipt_kind": "detached_review_body_acknowledgment",
  "thread_id": "019f6981-9b41-78e2-bc90-54849d98f2ec",
  "agent_id": "019f6981-9b41-78e2-bc90-54849d98f2ec",
  "role": "claim-boundary adjudicator",
  "model": "GPT-5.6 Sol",
  "review_body_sha256": "fbf3f5d9915a0aff05959e5372a848fac0abbac65b1d27ab07fad043a178aaa6",
  "review_body_canonicalization": "utf8_lf_single_trailing_newline",
  "audited_commit": "0b15890e9a8cad95cee38a2d4a333fda8a00dd50",
  "source_hashes_sha256": "489a66f073d111fda9ff09e09ee7aa4a7f5107541ce6bfb42d3b6d25af457124",
  "source_bundle_sha256": "8008ae53870bfd8163711fa13b3000d8296b27369cc506fb665e9fabaccf1cef",
  "runtime_binary_sha256": "e40055465f137c8767b565b93ae10cbfe09f51f75e8f3e5371115c14fa4afe89",
  "no_git_write": true,
  "no_file_edits": true,
  "no_checkout_mutation": true,
  "no_target_contact": true,
  "no_live_authority": true,
  "no_pmu": true,
  "self_authored": false,
  "evidence_origin": "codex_subagent_detached_receipt"
}
```
