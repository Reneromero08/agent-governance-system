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
