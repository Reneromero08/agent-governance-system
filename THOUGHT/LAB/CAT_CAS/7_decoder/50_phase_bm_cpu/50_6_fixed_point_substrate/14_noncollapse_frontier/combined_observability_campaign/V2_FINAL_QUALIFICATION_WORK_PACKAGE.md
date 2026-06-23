# Phase 6 V2 Final Qualification Work Package

**Branch:** `codex/phase6-v1-adjudication-v2`  
**Connector-prepared head:** `6b496762393e574f1ba7aa2c1975df38527c2873`  
**Hosted proof:** Phase 6 Combined Campaign Plan run `28009195670`, run number `91`, passed every staged V2 lane.  
**Scope:** materialize the already-tested source repair, close the remaining strict boundaries, regenerate contracts, run the exact-head Linux lane, and package evidence.

**Forbidden:** hardware calibration, scientific acquisition, restoration execution, target coupling, Small Wall execution, V1 reinterpretation, branch deletion, PR merge, or authorization creation.

The connector has already updated the authority stack, PR surface, roadmaps, architecture review, and GitHub workflow. Do not rewrite those documents except to replace provisional status with final hashes after qualification.

## 1. Start clean

```bash
git switch codex/phase6-v1-adjudication-v2
git pull --ff-only
git status --short
git rev-parse HEAD
```

Require a clean worktree and record the starting SHA.

## 2. Materialize the connector-tested repair

Run both deterministic repair scripts:

```bash
python THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/apply_connector_source_repair.py
python THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/apply_connector_source_repair_round2.py
```

These scripts already passed the complete hosted workflow in an ephemeral checkout. They repair:

- V2 strict compilation;
- full-consumption CLI numeric parsing;
- malformed numeric regression cases;
- sender-off control fields in runtime fixtures and engineering smoke;
- sender-off Nyquist analysis at the declared control tone;
- analyzer capture coverage, empirical sample rate, Nyquist, and gap enforcement from the frozen plan;
- plan/C capture-threshold identity regression;
- run-root directory and symlink rejection;
- plan/authorization campaign-source agreement;
- frozen `read_hz=8000` authorization fixture consistency.

Review the resulting diff. Do not hand-edit generated contracts yet.

## 3. Close the remaining source boundaries

Complete these items in the same source-repair commit.

### 3.1 Strict complete C JSON parsing

Replace the remaining authorization string-search trust boundary with complete JSON document validation before any executor digest or hardware-state operation.

The C gate must reject:

- malformed JSON;
- trailing non-whitespace content;
- duplicate top-level keys;
- unknown top-level keys;
- missing fields;
- wrong field types;
- non-singleton `session_ids`;
- unknown or duplicate `route_cores` entries;
- source-bundle objects containing anything except the exact current singleton session;
- prefix, suffix, or elsewhere session matches.

A small vendored parser or a compact local validator is acceptable. Keep it self-contained and compile it under `-Wall -Wextra -Werror`. Do not shell out to Python from the C authorization boundary.

Add tests for unknown fields, trailing content, malformed syntax, duplicate keys, wrong types, extra source-bundle sessions, and exact singleton acceptance.

### 3.2 Direct C capture-quality regression

The runtime quality gates must be directly testable. Factor the calculations into a pure helper or give the mock backend controlled timestamps so tests can trigger each rejection independently:

```text
CAPTURE_COVERAGE_INSUFFICIENT
EMPIRICAL_SAMPLE_RATE_OUT_OF_BOUNDS
EMPIRICAL_NYQUIST_MARGIN_INSUFFICIENT
PATHOLOGICAL_TIMESTAMP_GAP
```

Do not leave all four gates unreachable under mock tests.

### 3.3 Same-byte analyzer custody

For immutable JSON, JSONL, CSV, telemetry, run-manifest, and raw inputs:

- read bytes once;
- hash those exact bytes;
- parse from those same bytes;
- reject mutation between hash and parse;
- validate exact top-level and nested schemas before scientific computation;
- require plan, authorization, session, run, and campaign source commits to agree.

Preserve exact CSV order, telemetry cross-validation, eight sender-off controls per tone, full tone/amplitude/theta/sign coverage, and all false scientific authorization fields.

## 4. Source-repair commit

Run the focused suites before committing:

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

Then run:

```bash
cd ../combined_observability_campaign/v2
python -m unittest -v \
  test_calibration_contract.py \
  test_receiver_schedule.py \
  test_spectral_calibration_analyzer.py
```

Delete the two temporary `apply_connector_source_repair*.py` scripts after their changes have been materialized. Remove the workflow's ephemeral repair step so CI tests the committed source directly.

Commit one coherent source unit:

```text
phase6: close V2 source and authorization boundaries
```

Do not split it into micro-commits.

## 5. Deterministic generated-contract commit

Regenerate from the exact source-repair commit. Use the generator's actual CLI and do not hand-edit JSON or JSONL.

Require:

- four exact sessions;
- 672 windows per session;
- 1,344 windows per route;
- 2,688 windows total;
- eight sender-off controls per tone;
- exact source-repair commit in every required binding;
- all session manifests matching exact bytes;
- source-bundle manifest matching all session manifests;
- matching SHA-256 sidecars;
- deterministic regeneration with no second-run diff.

Commit one generated unit:

```text
phase6: regenerate V2 qualification contracts
```

## 6. Exact generated-head qualification

Push the two commits and require all GitHub checks to pass. The `Phase 6 Combined Campaign Plan` workflow must test committed source directly, with no repair script running first.

Run the full no-write repository gate locally:

```powershell
.\.venv\Scripts\python.exe CAPABILITY\TOOLS\utilities\ci_local_gate.py --full
```

## 7. Exact-head Linux SSH lane

Archive or check out the exact generated-contract head on the Phenom II target. Do not use a working tree with uncommitted changes.

```bash
cd THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/holo_runtime_v2

cc -std=c11 -O2 -pthread -Wall -Wextra -Werror \
  combined_pdn_runner.c combined_pdn_hardware.c \
  -o combined_pdn_runner -lm

python3 -m unittest -v \
  test_combined_pdn_runner.py \
  test_waveform_equivalence.py \
  test_slot2_primitive_identity.py
```

Sanitizer lane:

```bash
cc -std=c11 -O1 -g -pthread -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  combined_pdn_runner.c combined_pdn_hardware.c \
  -o combined_pdn_runner -lm

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
python3 -m unittest -v test_combined_pdn_runner.py

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
python3 -m unittest -v test_waveform_equivalence.py
```

Run the focused V2 Python suite on Linux as well if NumPy is available.

No `--hardware`, `--engineering-smoke`, calibration, acquisition, or authorization command may run.

## 8. Evidence-only commit

After every lane passes, create one evidence-only commit. Bind:

- source-repair commit;
- generated-contract commit;
- exact tested final head;
- calibration-plan digest;
- source-bundle digest;
- GitHub workflow run ID and conclusion;
- local focused tests;
- full no-write repository gate;
- Linux strict tests;
- Linux sanitizer tests.

Record explicitly:

```text
hardware_ran=false
authorization_artifact_created=false
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

Update final hashes in:

```text
v2/ARCHITECTURE_REVIEW.md
PHASE6_ROADMAP.md
PHASE6_NAVIGATION.md
14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md
PR #21 body
```

Commit:

```text
phase6: bind final V2 qualification evidence
```

Keep PR #21 draft and stop for independent review.

## 9. Stop conditions

Stop and report instead of improvising if:

- generated contracts are nondeterministic;
- GitHub and local results disagree;
- strict and sanitizer results disagree;
- source, plan, session, run, or evidence commits cannot be made identical where required;
- a change would alter historical V1 evidence;
- a change would weaken a fail-closed gate;
- a change would authorize hardware or cross Gate R.
