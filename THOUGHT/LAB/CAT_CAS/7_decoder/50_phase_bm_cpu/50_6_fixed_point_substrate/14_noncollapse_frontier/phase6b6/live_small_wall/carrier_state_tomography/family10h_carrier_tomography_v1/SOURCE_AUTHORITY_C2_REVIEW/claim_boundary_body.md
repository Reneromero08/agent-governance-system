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
