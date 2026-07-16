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
