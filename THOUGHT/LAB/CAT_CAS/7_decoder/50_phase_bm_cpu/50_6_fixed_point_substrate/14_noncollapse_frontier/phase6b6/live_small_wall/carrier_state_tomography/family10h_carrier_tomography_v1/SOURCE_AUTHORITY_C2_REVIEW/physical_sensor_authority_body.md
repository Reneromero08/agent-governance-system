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
