# Phase 6 V2 Final Qualification Work Package

**Status:** `PHASE6_V2_ENGINEERING_QUALIFICATION_COMPLETE__INDEPENDENT_PR_REVIEW_COMPLETE__GATE_R_NEXT`
**Branch:** `codex/phase6-v1-adjudication-v2`
**Audited source-only repair proof:** GitHub Actions run `28015182641`, run number `15`
**PR:** `#21`, draft and unmerged
**Final source commit:** `21201106a2b4cbd811d396181e733e08c38beb5d`
**Final generated-contract commit:** `a8ff3aa96f7bc3bff005088e63e837da44e8ce41`
**Raw evidence closure:** `4b5817a8741889caf5fadfa49df79fecb2f858a9 (incomplete summary), 69691b8061ea9eef6bf1b0dff44d0f1f2de1b863 (incomplete raw), 05c68281bcafda53381b2f70e4de13c25d1f5c9b (corrected), d0086ad0897cce6027b511c3409ff4ba3d422860 (metadata)`
**Command evidence closure:** `d0086ad0897cce6027b511c3409ff4ba3d422860`
**Review ledger correction:** `14469abb48567dda7c6eeb5c4bf16a8b282be85c`
**Plan SHA-256:** `7b21fa00ae986128f812d7720994d8e168844aa71cf3435b2edfea10497c738a`
**Source-bundle SHA-256:** `11547477f1a41e9b0661bb9f5d3532ab75aba20e0c785d9d14861bea2c57d487`
**Hardware calibration authorized:** false
**Scientific acquisition authorized:** false
**Restoration authorized:** false
**Target coupling authorized:** false
**Small Wall execution authorized:** false

---

## Completion record

This work package is retained as the execution and audit history for a completed engineering qualification.

```text
status: PHASE6_V2_ENGINEERING_QUALIFICATION_COMPLETE__INDEPENDENT_PR_REVIEW_COMPLETE__GATE_R_NEXT
source commit: 21201106a2b4cbd811d396181e733e08c38beb5d
generated-contract commit: a8ff3aa96f7bc3bff005088e63e837da44e8ce41
raw evidence closure: 4b5817a8741889caf5fadfa49df79fecb2f858a9 (incomplete summary), 69691b8061ea9eef6bf1b0dff44d0f1f2de1b863 (incomplete raw), 05c68281bcafda53381b2f70e4de13c25d1f5c9b (corrected), d0086ad0897cce6027b511c3409ff4ba3d422860 (metadata)
command evidence closure: d0086ad0897cce6027b511c3409ff4ba3d422860
review ledger correction: 14469abb48567dda7c6eeb5c4bf16a8b282be85c
unique functional test cases: 86
total unittest executions: 209
independent PR review: complete
independent review record: 4559668654
independent reviewed head: 38bd6cb3423c512137a1e5cbcfae18420bcce996
independent review result: NO_BLOCKING_FINDINGS
Gate R: pending
Phase 6B.6: not entered
```

No hardware calibration or scientific acquisition ran. All scientific authorization fields remain false.

---

## 1. Corrected repository state

The previous local report produced:

```text
source repair: b7563e5fe67d267840f4d5a25c776e7504e7dc5e
generated contracts: 93f28c5db29eaeeca7d0375efc5f69da8bea15b8
```

Those commits are retained as provenance, but `93f28c5d` is **superseded for final qualification**.

Reason: the reported Round 5 repair did not land. The analyzer still hashed immutable inputs and reopened their paths through `read_text`, `open`, and `np.fromfile`. Therefore the generated contracts were produced before same-byte analyzer custody was actually closed.

The connector has since prepared and hosted-tested the missing repair. It is stored as an exact, digest-locked source-only installer:

```text
combined_observability_campaign/v2/APPLY_SAME_BYTE_CUSTODY_SOURCE_ONLY.py
```

The installer:

- verifies the audited full repair payload digest;
- extracts only the analyzer and analyzer-test hunks;
- verifies an independent source-only patch digest;
- requires exact source preimage hashes;
- applies the patch with `git apply --check` and `--whitespace=error-all`;
- verifies exact final source hashes;
- is idempotent after successful application.

GitHub Actions run `28015182641` proved the materialized source-only result through:

- strict C compilation with `-Wall -Wextra -Werror`;
- V2 runner contracts;
- C/Python waveform equivalence;
- Slot2 primitive identity;
- V2 calibration-contract tests;
- the complete V2 analyzer campaign suite;
- ASan and UBSan.

No hardware path ran.

---

## 2. Frozen scientific boundary

The following state must not change during this work:

```text
V1:
PERMANENT_RETROSPECTIVE_NEGATIVE_ADJUDICATION
NO_STABLE_PREDICTIVE_OPERATOR
PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN

T48 carrier:
TRANSFER_EQUIVARIANCE_SUPPORTED under a minimal C0 receiver chart
STRICT_CARRIER_CLOSURE_PARTIAL

V2:
ENGINEERING_QUALIFICATION_ONLY

Gate R:
PENDING

Phase 6B.6:
NOT ENTERED

physical restoration:
NOT ESTABLISHED

target coupling:
NOT ESTABLISHED

fold-odd invariant:
NOT ESTABLISHED

Small Wall crossing:
NOT ESTABLISHED
```

The current V2 plan uses ascending tone order. It is not the proposed reversed/randomized tone-order scientific control.

Forbidden in this task:

- `--hardware`;
- `--engineering-smoke`;
- calibration execution;
- physical acquisition;
- authorization creation;
- restoration execution;
- target coupling;
- Small Wall execution;
- V1 reinterpretation;
- branch deletion;
- PR merge;
- force-push, rebase, squash, or amend.

---

## 3. Start clean

```bash
git switch codex/phase6-v1-adjudication-v2
git pull --ff-only
git status --short
git rev-parse HEAD
git log --oneline --decorate -12
```

Require a clean worktree before continuing. Do not discard unexplained changes.

---

## 4. Materialize the audited custody repair

Run exactly:

```bash
python THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/APPLY_SAME_BYTE_CUSTODY_SOURCE_ONLY.py
```

Required output:

```text
SAME_BYTE_CUSTODY_SOURCE_REPAIR_APPLIED
```

`SAME_BYTE_CUSTODY_SOURCE_REPAIR_ALREADY_APPLIED` is also acceptable only when the final source hashes already match.

Immediately inspect:

```bash
git status --short
git diff --stat
git diff --check
git diff -- \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/analyze_spectral_calibration_v2.py \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/test_spectral_calibration_analyzer.py
```

Expected final SHA-256 values:

```text
analyze_spectral_calibration_v2.py
87e6043633812411941cce67267f2ef6790b71cbbcd1f97bb99073b00084f9d3

test_spectral_calibration_analyzer.py
236aced8eacdbd92ab292ad97b76f47158e8b856e1ee52c3b6ccbc9ddb172626
```

The materialized analyzer must:

- open each immutable input with `O_NOFOLLOW` where supported;
- require regular files;
- read each input once;
- hash the captured bytes;
- parse JSON, JSONL, CSV, telemetry, and raw samples from those same bytes;
- reject file-identity changes across the read;
- replace path-based `np.fromfile` with `np.frombuffer` over captured bytes;
- enforce exact plan, session, run, manifest, evidence-map, runtime, threshold, and source-commit schemas;
- derive all recorded input bindings from the captured bytes;
- reject symlinks, extra entries, trailing raw records, and manifest drift.

Do not substitute an approximate implementation. The installer carries the hosted-tested bytes.

---

## 5. Remove temporary scaffolding

After proving the real source files contain the repair:

```bash
git rm \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/APPLY_SAME_BYTE_CUSTODY_REPAIR.py \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/APPLY_SAME_BYTE_CUSTODY_SOURCE_ONLY.py
```

Remove the ephemeral installer step from:

```text
.github/workflows/phase6-v2-strict-qualification.yml
```

The final workflow must test committed source directly. Its job sequence remains:

```text
checkout
setup Python
install NumPy
strict V2 compile and runtime tests
V2 contracts and analyzer tests
V2 ASan/UBSan tests
```

No workflow may mutate source before testing the final source commit.

Verify:

```bash
git grep -n 'APPLY_SAME_BYTE_CUSTODY' -- . ':!*.log' || true
git diff --check
```

Expected result: no remaining workflow or source dependency on the temporary installers.

---

## 6. Source qualification

Run from the repository root. Save complete stdout, stderr, command, exit code, toolchain, commit SHA, and worktree state.

### V2 runtime

```bash
cd THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/holo_runtime_v2

gcc -std=c11 -O2 -pthread -Wall -Wextra -Werror \
  combined_pdn_runner.c combined_pdn_hardware.c \
  -o combined_pdn_runner -lm

python -m unittest -v \
  test_combined_pdn_runner.py \
  test_waveform_equivalence.py \
  test_slot2_primitive_identity.py
```

### V2 contracts and analyzer

```bash
cd ../combined_observability_campaign/v2

python -m unittest -v \
  test_calibration_contract.py \
  test_receiver_schedule.py \
  test_spectral_calibration_analyzer.py
```

### Sanitizers on a Linux-capable host

```bash
cd ../../holo_runtime_v2

gcc -std=c11 -O1 -g -pthread -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  combined_pdn_runner.c combined_pdn_hardware.c \
  -o combined_pdn_runner -lm

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
python -m unittest -v test_combined_pdn_runner.py test_waveform_equivalence.py
```

Delete only the untracked compiled binary afterward. Do not use `git clean`.

---

## 7. Source commit

Review the complete staged unit:

```bash
git status --short
git diff --stat
git diff --name-status
git diff --check
git diff
```

The source commit must contain only:

- same-byte analyzer custody;
- custody regressions;
- removal of the two temporary installers;
- restoration of strict CI to committed-source testing.

Commit as one unit:

```bash
git add \
  .github/workflows/phase6-v2-strict-qualification.yml \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/analyze_spectral_calibration_v2.py \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/test_spectral_calibration_analyzer.py \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/APPLY_SAME_BYTE_CUSTODY_REPAIR.py \
  THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/APPLY_SAME_BYTE_CUSTODY_SOURCE_ONLY.py

git diff --cached --check
git commit -m "phase6: close same-byte analyzer custody"
```

Record the exact source commit. Do not amend it.

---

## 8. Regenerate contracts from the new source commit

The committed contracts at `93f28c5d` are superseded and must be regenerated from the new source commit.

Inspect the generator CLI first:

```bash
python THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/build_calibration_plan_v2.py --help
```

Use the actual CLI to bind the exact new source commit. Do not hand-edit generated JSON or JSONL.

Require:

```text
4 sessions
672 windows per session
1,344 windows per route
2,688 windows total
8 sender-off controls per tone
read_hz = 8000
amplitude duties = 1/8, 2/8, 3/8
ascending tone order
all source bindings = new source commit
all sidecars = exact bytes
```

Regenerate twice. The second run must produce no diff.

Commit as one generated unit:

```bash
git commit -m "phase6: regenerate V2 qualification contracts after custody repair"
```

Record the new plan and source-bundle SHA-256 values. The previous values are historical only:

```text
old plan: f67ecbba90368ded107cc1cf5225b27698500c6399d5eb3aecc689b0f1edef18
old bundle: 416d748dd851735b5ada5c5f193ba874424fbf24844f810f901e8b2f889ff48f
```

---

## 9. Push and exact-head GitHub qualification

```bash
git push origin codex/phase6-v1-adjudication-v2
```

Do not force-push.

Require the final generated-contract head to pass:

- Phase 6 V2 Strict Qualification;
- Phase 6 Combined Campaign Plan;
- Contracts;
- Governance;
- canonical full no-write repository gate locally.

The strict workflow must not apply a repair script before testing.

If source changes after regeneration, invalidate the generated commit and regenerate again.

---

## 10. Exact-head Phenom II Linux lane

Check out the exact generated-contract head on the target. Prove:

```bash
git rev-parse HEAD
git status --short
```

The SHA must equal the GitHub-qualified generated-contract head and the tree must be clean.

Run:

```bash
cd THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/holo_runtime_v2

cc -std=c11 -O2 -pthread -Wall -Wextra -Werror \
  combined_pdn_runner.c combined_pdn_hardware.c \
  -o combined_pdn_runner -lm

python3 -m unittest -v \
  test_combined_pdn_runner.py \
  test_waveform_equivalence.py \
  test_slot2_primitive_identity.py

cc -std=c11 -O1 -g -pthread -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  combined_pdn_runner.c combined_pdn_hardware.c \
  -o combined_pdn_runner -lm

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
python3 -m unittest -v test_combined_pdn_runner.py test_waveform_equivalence.py
```

Run the V2 Python contract/analyzer suite on the target if NumPy is available in a local environment.

Do not run hardware, engineering smoke, calibration, or acquisition.

---

## 11. Evidence and authority commit

Only after all exact-head lanes pass:

- bind local Windows results;
- bind final GitHub run IDs;
- bind the exact Phenom II SHA and toolchain;
- bind strict C, waveform equivalence, Slot2, capture-quality, ASan, UBSan, and Python results;
- bind the canonical full repository gate;
- regenerate the evidence inventory over the exact committed bytes;
- update `ARCHITECTURE_REVIEW.md`, `PHASE6_ROADMAP.md`, `PHASE6_NAVIGATION.md`, `CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md`, and PR #21 with final hashes.

Record explicitly:

```text
hardware_ran=false
authorization_artifact_created=false
calibration_authorized=false
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

Commit as one evidence-and-authority unit:

```bash
git commit -m "phase6: bind final V2 qualification evidence"
```

Keep PR #21 draft and stop for independent review.

---

## 12. Stop conditions

Stop and report instead of improvising if:

- the installer reports any preimage or digest mismatch;
- the final source hashes differ from the values above;
- strict C, analyzer, equivalence, Slot2, or sanitizer tests fail;
- generated contracts are nondeterministic;
- GitHub tests an ephemeral repair rather than committed source;
- the exact Linux SHA differs from the generated-contract head;
- a source change is needed after regeneration;
- a repair would alter V1 evidence;
- a repair would weaken a fail-closed gate;
- a repair would authorize hardware or cross Gate R.
