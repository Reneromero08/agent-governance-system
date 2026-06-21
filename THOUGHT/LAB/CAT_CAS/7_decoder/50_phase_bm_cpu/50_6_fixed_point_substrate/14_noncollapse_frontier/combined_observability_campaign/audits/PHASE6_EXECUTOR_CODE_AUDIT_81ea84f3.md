# Phase 6 Combined-Observability Executor Code Audit

**Audit target:** exact acquisition executor bound to commit  
`81ea84f341b29c41b93667d0e0fb98e0975bcbcf`

**Repository:** `Reneromero08/agent-governance-system`  
**Audit type:** read-only static source audit  
**Acquisition result under review:** `PHASE6_ACQUISITION_COMPLETE__EVIDENCE_RETURNED`  
**Date:** 2026-06-21  
**Status:** `AUDIT_COMPLETE__SCIENTIFIC_CONFORMANCE_DEFECTS_FOUND`

---

# Executive Verdict

The Phase 6 acquisition has two different validity layers.

## Provenance and lifecycle layer

**Verdict: PASS**

The code provides a credible chain from:

```text
exact Git commit
→ committed source extraction
→ source-transfer bundle
→ target build and test evidence
→ final sealed bundle
→ explicit authorization
→ preflight
→ immutable per-session execution
→ run manifests
→ returned inventory
```

The successful acquisition proof packet is consistent with that architecture:

- exact bundle, plan, runner, and authorization digests reproduced on CAT_CAS;
- 12/12 sessions completed;
- 99,456 windows completed;
- no automatic retry;
- all run manifests verified;
- 173 returned files matched the returned inventory;
- host cpufreq and boost state restored;
- physical carrier restoration was not claimed.

No code-audit finding indicates that the returned evidence came from an unknown executable, an unknown plan, or an unauthorized run.

## Scientific implementation layer

**Verdict: CONDITIONAL**

Two high-severity conformance defects were found:

1. **The implemented sender waveform does not have the requested tone as its ideal fundamental.**  
   The eight-state amplitude envelope repeats once every four requested tone periods. The stored lock-in demodulates at the requested tone, which is the fourth harmonic of the actual envelope. For amplitude levels 1, 2, and 3, the ideal rectangular envelope has an exact Fourier zero at that fourth harmonic.

2. **The `scramble` negative control is not physically implemented.**  
   The planner marks scramble rows with `shared_schedule=false`, but the C executor never parses or uses that field. Scramble rows therefore remain ordinary physically driven rows, not the intended unshared or decoy schedule.

A third material issue compounds the first:

3. **The acquisition executor replaced the previously proven Slot2 power-virus primitive with a substantially weaker serial dependency chain**, despite the binding interface requiring reuse of the proven primitive.

These defects do not erase the raw acquisition. The exact executed waveform is deterministic from the source, and raw timestamps and ring observations were preserved. They do, however, prevent treating the stored `computed_I`, `computed_Q`, and `floor` columns as automatically faithful measurements of the originally specified carrier.

The acquisition should therefore enter:

```text
PROVENANCE_VALID
RAW_EVIDENCE_VALID
ORIGINAL_LOCKIN_SEMANTICS_NOT_YET_VALIDATED
SCRAMBLE_NULL_INVALID
SCIENTIFIC_VERDICT_PENDING_DEFECT_AWARE_REANALYSIS
```

---

# 1. Scope

The audit covered the exact source list included by `make_executor_source_bundle.py`:

## Hardware runtime

- `holo_runtime/combined_pdn_hardware.h`
- `holo_runtime/combined_pdn_hardware.c`
- `holo_runtime/combined_pdn_runner.c`
- `holo_runtime/test_combined_pdn_runner.py`
- `holo_runtime/Makefile`
- `holo_runtime/README.md`
- `holo_runtime/make_engineering_smoke_schedule.py`

## Campaign generation and execution

- `combined_observability_campaign/compile_session_schedule.py`
- `combined_observability_campaign/campaign_orders.py`
- `combined_observability_campaign/campaign_plan.py`
- `combined_observability_campaign/generate_campaign_plan.py`
- `combined_observability_campaign/run_combined_campaign.py`
- `combined_observability_campaign/verify_run_manifests.py`

## Evidence and provenance

- `combined_observability_campaign/catcas_preflight.py`
- `combined_observability_campaign/collect_target_engineering_evidence.py`
- `combined_observability_campaign/make_executor_source_bundle.py`
- `gate_r/verify_combined_plan_binding.py`

## Tests

- `test_campaign_plan.py`
- `test_session_schedule.py`
- `test_session_determinism.py`
- `test_orchestrator.py`
- `test_run_manifests.py`
- `test_executor_source_bundle.py`
- `test_catcas_preflight.py`
- `test_target_engineering_evidence.py`
- `test_combined_pdn_runner.py`

## Binding contracts reviewed

- `CAMPAIGN_CONTRACT.md`
- `ANALYSIS_CONTRACT.md`
- `HARDWARE_EXECUTOR_INTERFACE.md`
- `EXECUTOR_OUTPUT_CONTRACT.md`
- acquisition authorization artifact

---

# 2. Audit Limitations

This is a static code audit against the exact committed source.

It does not yet include:

- direct inspection of the 3.35 GB returned evidence tree;
- re-execution of the C runner;
- independent compilation;
- binary disassembly comparison against source;
- sample-level spectral reconstruction;
- verification of all 173 returned files;
- scientific model fitting.

Those belong to the next evidence-audit phase.

The prior engineering audit already reported:

- strict and sanitizer builds;
- Python, strict C, and sanitizer tests passing;
- exact binary reproduction;
- target evidence and bundle hash closure.

This report does not repeat those execution claims as independently rerun by this auditor.

---

# 3. Architecture Reviewed

```text
frozen campaign source
        │
        ▼
campaign_plan.json
        │
        ▼
12 compiled session bundles
        │
        ▼
source-transfer bundle
        │
        ▼
target strict/sanitizer build and engineering evidence
        │
        ▼
sealed final bundle
        │
        ▼
owner authorization
        │
        ▼
read-only preflight
        │
        ▼
run_combined_campaign.py
        │
        ▼
combined_pdn_runner
        │
        ▼
combined_pdn_hardware
        │
        ▼
12 immutable run directories
        │
        ▼
return inventory
```

The architecture correctly separates:

```text
engineering readiness
acquisition authorization
host control-state restoration
physical carrier restoration
scientific interpretation
```

That separation is one of the strongest parts of the implementation.

---

# 4. Findings Summary

| ID | Severity | Domain | Finding | Effect on completed acquisition |
|---|---|---|---|---|
| P6-CODE-001 | HIGH | Scientific waveform | Sender cycle is four requested tone periods; requested lock-in is an ideal spectral zero | Stored I/Q cannot be assumed faithful to intended drive |
| P6-CODE-002 | HIGH | Negative controls | `scramble` is metadata only and does not alter physical execution | Scramble rows are not a valid scramble null |
| P6-CODE-003 | MEDIUM-HIGH | Physical stimulus | Proven Slot2 drive primitive was replaced by a weaker serial chain | Prior carrier sensitivity does not transfer automatically |
| P6-CODE-004 | MEDIUM | Authorization | Authorization does not cryptographically bind runtime numeric parameters or force all 12 sessions in code | Actual run complied, but gate was broader than document scope |
| P6-CODE-005 | MEDIUM | Manifest closure | Per-run manifest verifier ignores unlisted extra files | Final return inventory compensates, but `RUN_MANIFESTS_VERIFIED` is not total-tree closure |
| P6-CODE-006 | MEDIUM | Semantic verification | Orchestrator checks hashes and return code but not all scientific run fields | Independent proof packet compensates; future verifier should be stricter |
| P6-CODE-007 | MEDIUM | Test coverage | Tests and smoke validate lifecycle, not waveform frequency, phase, amplitude, or scramble behavior | Explains why high-severity scientific defects passed engineering tests |
| P6-CODE-008 | LOW | Parser robustness | Hand-written JSON parser is positional/substr-based and silently defaults some malformed fields | Bound generated inputs make exploitation unlikely |
| P6-CODE-009 | LOW | Thermal semantics | Temperature is vetoed before each window but not immediately after a crossing | Maximum observed temperature stayed far below threshold |
| P6-CODE-010 | LOW | Path containment | C authorization path containment is lexical rather than canonical/symlink-aware | Exact preflight and controlled root reduce practical risk |

---

# 5. Finding P6-CODE-001

## Requested Tone Is Not the Implemented Fundamental

**Severity:** HIGH  
**Class:** scientific conformance  
**Affects provenance:** no  
**Affects stored lock-in interpretation:** yes  
**Raw evidence salvageable:** yes

## 5.1 Intended historical drive

The previously proven Slot2 implementation used:

```c
half_ticks = 0.5 * tsc_hz / tone_hz
halfidx = floor(elapsed / half_ticks)
on = ((halfidx & 1) == 0)
```

This produces:

```text
ON for half a period
OFF for half a period
repeat every 1 / tone_hz
```

It is a 50%-duty square wave whose fundamental is the declared physical tone.

The sign is encoded as a half-period phase shift and theta as a fractional phase shift of the same period.

## 5.2 Acquisition implementation

The combined executor uses:

```c
sender->half_ticks = 0.5 * tsc_hz / tone_hz
period = 2 * half_ticks
half = floor(offset / half_ticks)
quadrant = half mod 8
drive when quadrant < 2 * amplitude_level
```

Each state still lasts one half-period of the requested tone.

Because there are eight states, the complete gate pattern repeats every:

```text
8 × half_period
= 8 × (1 / (2f))
= 4 / f
```

Therefore:

```text
actual envelope fundamental = f / 4
```

not `f`.

## 5.3 Duty levels

The three amplitude levels create ideal envelope duty fractions:

```text
level 1 → 2 / 8 = 1/4
level 2 → 4 / 8 = 1/2
level 3 → 6 / 8 = 3/4
```

The C lock-in demodulates at `f`.

Relative to the actual `f/4` fundamental, this is harmonic:

```text
k = 4
```

For an ideal rectangular pulse train with duty fraction `D`, the harmonic coefficient is proportional to:

```text
sin(π k D) / (π k)
```

At `k = 4`:

```text
D = 1/4 → sin(π)  = 0
D = 1/2 → sin(2π) = 0
D = 3/4 → sin(3π) = 0
```

So the intended lock-in frequency is an exact ideal Fourier zero for every amplitude level used by the trajectory stage.

## 5.4 Phase semantics

The executor stores:

```c
phase = phase_index / 8
offset = phase * requested_period
```

But the actual gate cycle lasts four requested periods.

At the actual `f/4` envelope fundamental:

```text
one theta index step = π / 16
negative code sign (+4 indices) = π / 4
```

not:

```text
one theta index step = π / 4
negative code sign = π
```

At the requested `f` harmonic, the sign shift is nominally π, but that harmonic is ideally canceled.

## 5.5 What may still appear at `f`

Real CPU power draw is not an ideal two-level rectangle.

Possible residual energy at `f` may come from:

- transition transients;
- ALU burst startup and cessation;
- nonuniform busy-state draw;
- scheduling jitter;
- rail nonlinearities;
- thermal and electrical nonlinear response;
- sampling leakage;
- finite-window effects.

Such a residual may still carry phase, but it is not automatically the intended clean tone.

## 5.6 Effect

The stored columns:

```text
computed_I
computed_Q
magnitude
floor
```

must be treated as:

```text
demodulation of the observed ring signal at requested f
```

not as:

```text
validated measurement of the intended f carrier
```

until raw evidence demonstrates a coherent spectral line at `f`.

## 5.7 Required defect-aware analysis

For every driven window, reconstruct and test:

1. requested frequency `f`;
2. actual envelope fundamental `f/4`;
3. harmonics `f/2`, `3f/4`, `f`, `5f/4`, and others within Nyquist;
4. the exact binary gate reference implied by:
   - origin TSC;
   - TSC calibration;
   - tone index;
   - theta index;
   - code sign;
   - amplitude level;
   - first-drive timestamp.

The strongest coordinate should be selected only on training sessions and frozen before final test sessions.

Because this coordinate recovery is prompted by an implementation defect, it must be labeled:

```text
IMPLEMENTATION_RECOVERY_ANALYSIS
```

and separated from the original predeclared lock-in analysis.

---

# 6. Finding P6-CODE-002

## Scramble Negative Control Is Not Physically Implemented

**Severity:** HIGH  
**Class:** control validity  
**Affects provenance:** no  
**Affects Stage B adjudication:** yes

## 6.1 Planner behavior

For `family == "scramble"`, `campaign_orders.py` sets:

```python
shared_schedule = False
```

but otherwise leaves:

```text
actual_mode
theta_idx
tone_execution_indices
codeword_bin_permutation
drive_on
```

as an ordinary driven symbol.

## 6.2 Compiler behavior

`compile_session_schedule.py` emits:

```json
"shared_schedule": false
```

into `windows.jsonl`.

## 6.3 Executor behavior

The C `Window` structure contains no `shared_schedule` member.

The parser does not read it.

The drive implementation does not create:

- an unshared sender permutation;
- an independently keyed sender schedule;
- a hidden decoy route;
- a randomized codeword unknown to the receiver;
- any other physical scramble.

Therefore, `scramble` changes only the family label and the random seed used to choose ordinary mode/theta fields.

## 6.4 Effect

Scramble rows are not a valid implementation of the intended negative control.

They must not be analyzed as evidence that a decoder fails when the sender and receiver schedule are unshared.

## 6.5 Required treatment

For this completed campaign:

```text
scramble rows = driven rows with invalid control label
```

They may be retained for descriptive or exploratory modeling under their actual executed controls.

They must be excluded from:

- scramble-null pass/fail;
- claims requiring an unshared schedule;
- claims that actual schedule agreement was isolated by this control.

The remaining controls still exist:

- wrong declaration;
- pseudo codeword permutation;
- order-label sham;
- silent;
- random physical order;
- sender-off windows.

The frozen analysis contract must record the missing scramble null as a limitation. It may lower or block the strongest Stage B claim depending on the exact adjudication logic.

---

# 7. Finding P6-CODE-003

## Proven Slot2 Power Drive Was Replaced

**Severity:** MEDIUM-HIGH  
**Class:** carrier sensitivity and historical transfer  
**Affects provenance:** no  
**Affects expected SNR and comparability:** yes

## 7.1 Previous proven drive

The Slot2 code described and implemented a register/L1-only power virus using:

- eight floating-point dependency chains;
- four integer chains;
- several distinct multipliers and constants;
- independent instruction-level work intended to saturate execution units.

## 7.2 Combined executor drive

The acquisition executor uses:

- one floating-point accumulator;
- one integer recurrence;
- one serial loop.

This is not the same primitive.

It has lower instruction-level parallelism and may produce a materially different:

- power amplitude;
- dI/dt;
- frequency response;
- thermal profile;
- rail coupling;
- transient structure.

## 7.3 Contract conflict

`HARDWARE_EXECUTOR_INTERFACE.md` required reuse of the proven Slot2:

```text
affinity
TSC
drive
capture
thermal veto
telemetry
P-state restoration
immutable writer
```

The affinity, timing, capture, thermal, telemetry, and restoration ideas were reused.

The drive primitive was not reused verbatim.

## 7.4 Effect

Earlier claims that the Slot2 drive produced a measurable cross-core PDN signal cannot be imported as calibration of this executor.

The completed engineering smoke established:

- thread lifecycle;
- real hardware touch;
- raw capture;
- timing;
- cleanup.

It did not establish a minimum spectral SNR for the new drive.

---

# 8. Finding P6-CODE-004

## Authorization Scope Is Broader in Code Than in the Artifact

**Severity:** MEDIUM  
**Class:** governance and authorization  
**Actual run compliant:** yes

## 8.1 Authorization artifact says

```text
FROZEN_12_SESSION_COMBINED_OBSERVABILITY_ACQUISITION_ONLY
```

and binds:

- executor commit;
- campaign plan hash;
- final bundle hash;
- output root.

## 8.2 Code permits

`run_combined_campaign.py` accepts repeated:

```text
--session
```

arguments and can execute a subset of plan sessions.

The authorization verifier does not require:

```text
sessions_requested == all 12 frozen sessions
```

The following runtime values are also command-line parameters rather than explicit authorization fields:

- pin frequency;
- driven slot duration;
- sender-off duration;
- read rate;
- thermal veto;
- selected session set.

## 8.3 Actual acquisition

The proof packet demonstrates that the executed run used:

- all 12 sessions;
- 1,600,000 kHz pin request;
- 0.5 s slots;
- 0.5 s sender-off windows;
- 4,000 Hz read rate;
- 68°C veto;
- no retry.

Therefore the completed run complied with the intended scope.

## 8.4 Future repair

A future authorization schema should bind:

```json
{
  "sessions": ["all exact IDs"],
  "pin_khz": 1600000,
  "slot_s": 0.5,
  "off_window_s": 0.5,
  "read_hz": 4000,
  "temp_veto_c": 68.0,
  "automatic_retry": false
}
```

The orchestrator and C runner should enforce them.

---

# 9. Finding P6-CODE-005

## Per-Run Manifest Verification Does Not Close the Directory

**Severity:** MEDIUM  
**Class:** evidence closure  
**Final inventory compensates:** yes

`verify_run_manifests.py` verifies every file listed in each run manifest.

It does not compare:

```text
actual run-directory file set
```

against:

```text
manifest file set
```

Therefore unlisted files are ignored.

The C run manifest intentionally binds eight core files:

- `run.json`
- `session.json`
- `windows.jsonl`
- `window_results.csv`
- `raw_samples.bin`
- `telemetry.csv`
- `stdout.log`
- `stderr.log`

The orchestrator later adds:

- `orchestrator_stdout.log`
- `orchestrator_stderr.log`

These are not included by the C manifest.

## Effect

The statement:

```text
RUN_MANIFESTS_VERIFIED count=12
```

proves the listed core files matched their bindings.

It does not by itself prove that each run directory contained no additional files.

The returned 173-file inventory is the stronger total-return closure and should remain the authoritative outer manifest.

---

# 10. Finding P6-CODE-006

## Orchestrator Semantic Verification Is Incomplete

**Severity:** MEDIUM  
**Class:** defense in depth  
**Actual proof packet compensates:** yes

After a session returns zero and its manifest verifies, the orchestrator explicitly checks:

- executor commit;
- physical carrier restoration not claimed.

It does not explicitly require all of:

- `exit_status == COMPLETE`;
- `hardware_executed == true`;
- `scientific_acquisition_authorized == true`;
- authorization artifact hash equals the frozen expected digest;
- `host_control_state_restored == true`;
- route and session match the requested session;
- run plan hash matches the frozen plan;
- run session-manifest hash matches the generated session.

The C runner’s zero exit path and manifest creation imply several of these, and the returned proof packet independently checked them.

Future verification should make these requirements explicit.

---

# 11. Finding P6-CODE-007

## Test and Smoke Coverage Misses Scientific Semantics

**Severity:** MEDIUM  
**Class:** quality assurance

The test suite strongly covers:

- manifest schemas and hashes;
- output collisions;
- path rejection;
- sender-off invariants;
- route/core mapping;
- authorization presence;
- commit syntax;
- failure cleanup;
- late sender failure;
- restoration failure;
- immutable outputs;
- validation-only behavior.

It does not test:

- spectral energy at the requested tone;
- actual waveform fundamental;
- phase mapping from theta;
- π sign flip;
- monotonic amplitude response;
- relationship between `amplitude_level` and Fourier amplitude;
- comparison to the proven Slot2 primitive;
- physical or synthetic SNR;
- scramble changing actual execution;
- `shared_schedule` being consumed.

The engineering smoke validates timing and lifecycle only.

This is why all tests could pass while the scientific drive semantics remained wrong.

---

# 12. Lower-Severity Findings

## 12.1 Hand-written JSON parser

The C runner uses substring searches rather than a full JSON parser.

Risks include:

- first matching key wins;
- escaped strings are not decoded;
- numeric parsing accepts trailing characters;
- some missing fields default rather than fail;
- duplicate keys are not detected.

The risk is controlled because:

- the session files are generated internally;
- they are exact-hash bound;
- the source bundle is sealed;
- the runner refuses hash mismatches.

For future generalized use, replace this parser or validate canonical line schemas more strictly.

## 12.2 Range validation

The runner does not strictly reject every invalid generated field:

- theta range;
- tone-index range;
- amplitude-level range;
- actual-mode vocabulary.

Some values are clamped or become default behavior.

The frozen plan contains valid values, so this is not an observed-run defect.

## 12.3 Thermal veto

Temperature is vetoed before each window.

A post-window temperature exceeding the threshold is logged but is not immediately converted to failure until the next window’s pre-check.

The observed maximum, 52.125°C, is well below 68°C.

## 12.4 Symlink-aware output containment

The C authorization check uses lexical containment.

The Python preflight requires exact output root equality and the actual target paths were controlled, reducing practical risk.

---

# 13. Strong Components

The audit also found substantial strengths.

## 13.1 Exact committed-source custody

The source-bundle builder:

- requires HEAD equal to the requested commit;
- requires a clean tree;
- extracts each source with `git show`;
- compares working and committed blob hashes;
- compiles all 12 sessions;
- rejects Python bytecode;
- hashes the complete bundle.

## 13.2 Final bundle construction

The sealing code:

- verifies source transfer;
- verifies target evidence;
- verifies validation-only runs;
- verifies target runner hash;
- copies source and target evidence without collisions;
- produces complete file bindings;
- keeps acquisition and restoration authorization false in the engineering bundle.

## 13.3 Acquisition preflight

Preflight validates:

- exact final bundle file set;
- runner hash;
- source-transfer binding;
- campaign plan and manifest;
- canonical binding;
- all 12 compiled sessions;
- strict, sanitizer, and Python evidence;
- target host evidence;
- constant and nonstop TSC;
- k10temp;
- MSR access;
- cpufreq control;
- free space;
- cleanup;
- exact acquisition authorization;
- unused output path;
- restoration prohibition.

## 13.4 Fail-closed execution

The runner refuses:

- existing output;
- missing authorization;
- invalid route/core pairing;
- schedule hash mismatch;
- sender-off plus drive;
- invalid sender-off measurement;
- thermal veto;
- cpufreq failure;
- MSR failure;
- sender creation failure;
- sender timing skew;
- receiver timing skew;
- short capture;
- deadline overflow;
- raw write failure;
- control-state restoration failure.

## 13.5 Immutable evidence

Scientific outputs use exclusive creation.

The runner stores:

- copied exact input schedules;
- raw timestamps and doubles;
- detailed window metadata;
- telemetry;
- run metadata;
- file hashes.

## 13.6 Claim ceiling

The code explicitly records:

```text
physical_carrier_restoration_claimed = false
restoration_authorized = false
automatic_retry = false
```

This is correct and should remain unchanged.

---

# 14. Impact on the Scientific Claim Ladder

## Still supported

The completed acquisition proves:

- the exact authorized executor ran;
- the exact frozen plan was used;
- all sessions and windows completed;
- raw evidence was returned;
- provenance and outer inventory are intact;
- sender threads were absent in declared sender-off windows according to lifecycle evidence;
- host control state restored;
- the campaign can now be scientifically audited.

## Not yet supported

The code audit prevents immediately promoting:

- stored I/Q into intended tone response;
- stored floor into a validated off-bin noise floor;
- theta indices into the intended phase rotations;
- negative code signs into intended π phase flips;
- scramble rows into a valid schedule-scramble null;
- prior Slot2 sensitivity into this campaign’s drive sensitivity.

## Potentially recoverable from raw evidence

- actual spectral response at `f/4` and harmonics;
- matched response to the exact executed gate;
- tone identity under the actual waveform;
- order dependence;
- silent controls;
- sender-off persistence;
- trajectory predictability;
- route differences;
- temperature/frequency context;
- actual phase geometry induced by the implemented waveform.

---

# 15. Required Evidence Audit Sequence

## Gate A: Outer custody

Verify:

- returned inventory hash;
- all 173 paths;
- all file sizes and hashes;
- no duplicates;
- no path traversal;
- exact authorization copy;
- exact plan and source-bundle copies.

## Gate B: Run closure

For each of 12 sessions:

- run directory exists;
- session ID, route, and seed correct;
- core files match run manifest;
- outer inventory covers all run files;
- no unexpected or missing session IDs;
- no duplicate windows.

## Gate C: Raw-size reconciliation

For each run:

```text
sum(sample_count) × 16
==
size(raw_samples.bin)
```

Then verify:

- each raw timestamp is strictly increasing inside its window;
- every sample double is finite;
- first and last timestamps match CSV;
- all slices consume the file exactly;
- no trailing or missing records.

## Gate D: Schedule-to-result conformance

For all 99,456 windows:

- input and copied schedule rows agree;
- `window_index` contiguous;
- route/session/stage/block/family agree;
- tone and codeword source agree;
- drive and sender-off flags agree;
- theta and amplitude agree;
- declared and executed metadata remain separate.

## Gate E: Lifecycle and safety

Verify:

- sender timestamps on every driven window;
- no sender timestamps on sender-off windows;
- first drive inside deadline;
- receiver epoch inside skew limit;
- sample range within deadline allowance;
- temperature and frequency telemetry;
- APERF, MPERF, and COFVID continuity;
- all window statuses `OK`.

## Gate F: Waveform reconstruction

Construct the exact expected gate:

```text
half_ticks = 0.5 × tsc_hz / f
phase_index = (theta + sign_offset) mod 8
phase_offset = phase_index / 8 × (1/f)
gate(t) = 1 when floor((t-origin-phase_offset)/half_ticks) mod 8 < 2L
```

Account for:

- first-drive truncation;
- capture origin;
- sender stop;
- finite sampling;
- amplitude level.

## Gate G: Spectral audit

For each training window, compute:

- spectrum or nonuniform Fourier transform;
- response at `f/4`, `f/2`, `3f/4`, `f`;
- matched-filter response to exact gate;
- off-frequency controls;
- silent distributions.

Freeze the corrected coordinate before validation, stress, and test partitions.

## Gate H: Control reclassification

Use:

```text
valid controls:
wrong
pseudo
order_sham
silent
random tone orders
sender_off

invalid as intended:
scramble
```

## Gate I: Frozen and recovery analyses

Maintain two ledgers:

### Original frozen analysis

Uses stored I/Q as originally specified.

### Implementation-recovery analysis

Uses raw samples and the exact executed waveform.

Do not merge them silently.

---

# 16. Can the Evidence Be Audited Under the 512 MB Upload Limit?

Yes.

Each session is approximately:

```text
8,288 windows
× approximately 2,000 samples
× 16 bytes per sample
≈ 265 MB raw
```

Therefore each of the 12 session run directories should fit under 512 MB when packaged separately, even if the binary raw data compresses poorly.

Recommended transfer format:

```text
phase6_audit_index/
  outer_inventory_and_manifests.zip
  session_v2s3_seed0.zip
  session_v2s3_seed1.zip
  ...
  session_v4s5_seed5.zip
```

Before uploading sessions, create a compact local audit packet containing:

- complete file tree;
- SHA-256 inventory;
- execution.json;
- preflight;
- authorization;
- campaign plan and manifest;
- all run.json files;
- all run manifests;
- all session manifests;
- all `window_results.csv`;
- all `telemetry.csv`;
- a raw file size/index table.

That metadata packet should be far below the limit.

Scientific raw analysis can then proceed one session at a time without losing the global hash chain.

---

# 17. Remediation for a Future Executor Version

Do not alter the historical `81ea84f3` source or claim it produced different data.

Create a new executor version.

## 17.1 Correct amplitude/phase waveform

Use an eight-state cycle whose complete period is `1/f`:

```c
step_ticks = tsc_hz / (8.0 * tone_hz);
cycle_ticks = 8.0 * step_ticks;
phase_offset = phase_index * step_ticks;
state = floor((now - origin - phase_offset) / step_ticks) mod 8;
on = state < 2 * amplitude_level;
```

This yields:

- fundamental `f`;
- phase step `2π/8`;
- sign offset of 4 states = π;
- duty levels 1/4, 1/2, 3/4.

## 17.2 Restore or explicitly requalify the power-virus primitive

Either:

- reuse the proven Slot2 drive verbatim; or
- define a new drive, measure its spectral power/SNR, and qualify it independently.

## 17.3 Implement scramble physically

The session must carry an explicit sender-side decoy mapping, for example:

```json
{
  "receiver_codeword_source_index": 4,
  "sender_codeword_source_index": 9,
  "scramble_key_digest": "...",
  "shared_schedule": false
}
```

The C executor must use the sender field while analysis uses only the receiver-visible field.

## 17.4 Bind authorization parameters

Bind and enforce:

- exact sessions;
- exact route set;
- core mapping;
- pin frequency;
- slot durations;
- read rate;
- thermal veto;
- runner digest;
- no retry.

## 17.5 Add scientific conformance tests

Add synthetic tests that assert:

- dominant spectral line is at requested `f`;
- sign flip changes phase by π;
- theta changes phase by `π/4`;
- amplitude levels have predeclared response;
- lock-in recovers synthetic injected waveform;
- scramble differs physically from real;
- silent and sender-off create no sender activity.

## 17.6 Add drive-edge evidence

Record or reconstructably bind:

- gate transition timestamps;
- drive state counts;
- effective on-time;
- phase state;
- per-window drive-reference digest.

---

# 18. Final Disposition

```text
PROVENANCE:
PASS

AUTHORIZATION:
PASS FOR ACTUAL EXECUTION
CODE SCOPE BROADER THAN DOCUMENTED AUTHORIZATION

HOST LIFECYCLE:
PASS

IMMUTABLE RAW EVIDENCE:
SUPPORTED

SCIENTIFIC WAVEFORM CONFORMANCE:
FAIL

SCRAMBLE CONTROL CONFORMANCE:
FAIL

PRIOR SLOT2 DRIVE TRANSFER:
NOT ESTABLISHED

ORIGINAL STORED I/Q:
REQUIRES VALIDATION AGAINST RAW

RAW DATASET:
VALID FOR DEFECT-AWARE AUDIT

SMALL WALL:
NOT ADJUDICATED

RERUN:
NOT AUTHORIZED
NOT YET JUSTIFIED
```

The correct next state is:

```text
PHASE6_ACQUISITION_COMPLETE
PROVENANCE_VALID
IMPLEMENTATION_DEFECTS_IDENTIFIED
RAW_EVIDENCE_AUDIT_REQUIRED
SCIENTIFIC_CLAIM_CEILING_UNCHANGED
```

---

# 19. Claim Ceiling After Code Audit

Allowed:

> The exact authorized Phase 6 executor completed all 12 sessions and returned a provenance-bound raw physical evidence set. Static code audit identified scientific conformance defects in the sender waveform and scramble control. The raw data remain auditable because exact timestamps, schedules, telemetry, and ring observations were retained.

Not allowed yet:

- the intended carrier was driven at the declared tones;
- stored I/Q faithfully represent the intended control;
- the scramble null passed;
- a stable operator was identified;
- persistence was established;
- target coupling exists;
- a fold-odd invariant exists;
- orientation was recovered;
- the Small Wall was crossed.

---

# 20. Immediate Next Action

Do not interpret aggregate scientific results yet.

First create and inspect:

```text
PHASE6_ACQUISITION_METADATA_AUDIT_PACKET_81ea84f3
```

Then audit one complete session, preferably a training session, including its `raw_samples.bin`.

The first scientific question is now:

```text
What carrier response exists under the waveform that was actually executed?
```

not:

```text
Did the originally intended lock-in coordinates win?
```
