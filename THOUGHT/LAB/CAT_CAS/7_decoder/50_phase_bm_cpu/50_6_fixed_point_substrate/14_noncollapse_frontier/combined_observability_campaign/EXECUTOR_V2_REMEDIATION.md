# Phase 6 Executor V2 Remediation Contract

**Status:** `FORWARD_ONLY`  
**Historical executor:** `81ea84f341b29c41b93667d0e0fb98e0975bcbcf`  
**Historical evidence:** immutable and not to be rewritten  
**Purpose:** define the minimum code and qualification work for any future acquisition

---

# 1. Non-Destructive Rule

Do not edit, amend, replace, or reinterpret the historical acquisition object.

The completed campaign remains bound to:

```text
executor commit:
81ea84f341b29c41b93667d0e0fb98e0975bcbcf

final bundle:
5c6588a51ce6b806e1b7b269bafd1981256795653415e012592ad3b6313fdaca

campaign plan:
eb5a46d0a37d66910649467cf0d4e3cf947dee11fab94a36e9bdfed388455e53

strict runner:
0fc846a52d34b2395e254cc5a2db0bb715d1cd1ede77f8a5f6a2e940dab63037

authorization:
e39fb0c6ebfb106c33a0b90b8d52d193a32833103388ebc4c6bd0cad451a0d73
```

A V2 executor is a new experiment object with a new commit, bundle, target qualification, authorization, and evidence chain.

---

# 2. Confirmed V1 Defects

## 2.1 Waveform period

V1 uses:

```c
half_ticks = 0.5 * tsc_hz / tone_hz;
quadrant = floor(offset / half_ticks) mod 8;
on = quadrant < amplitude_level * 2;
```

The complete envelope period is:

```text
8 × half_ticks = 4 / tone_hz
```

Therefore the physical envelope fundamental is `tone_hz / 4`.

The V1 lock-in demodulates at `tone_hz`, an ideal null for amplitude levels 1, 2, and 3.

## 2.2 Scramble control

The planner emits:

```json
"shared_schedule": false
```

but the C executor does not parse or use the field.

Scramble rows therefore do not alter physical execution.

## 2.3 Drive primitive

V1 replaced the prior Slot2 power-virus primitive with a single floating-point and single integer dependency chain.

The new drive was never independently qualified for carrier SNR before acquisition.

## 2.4 Authorization scope

The authorization artifact does not bind every runtime parameter or the exact twelve-session set in the C and Python execution gates.

## 2.5 Scientific conformance tests

The existing engineering tests verify lifecycle and custody. They do not verify:

- requested spectral fundamental;
- theta phase increment;
- code-sign phase flip;
- amplitude response;
- physical scramble divergence;
- carrier SNR.

---

# 3. Correct V2 Waveform

The complete eight-state cycle must last one requested tone period.

Use:

```c
double step_ticks = tsc_hz / (8.0 * tone_hz);
double phase_ticks = phase_index * step_ticks;
long state = floor((now - origin - phase_ticks) / step_ticks);
int cycle_state = ((state % 8) + 8) % 8;
int on = cycle_state < amplitude_level * 2;
```

This gives:

```text
complete cycle = 8 × step_ticks = 1 / tone_hz
one phase index = 2π / 8 = π / 4
four phase indices = π
```

Required semantics:

```text
theta_idx:
0..7 maps to 0, π/4, π/2, ..., 7π/4

code sign:
+1 adds 0
-1 adds 4 phase indices = π

amplitude_level:
1 gives 2/8 duty
2 gives 4/8 duty
3 gives 6/8 duty
```

The phase origin used by the sender, raw evidence, and lock-in must be identical.

---

# 4. Drive Primitive Requirement

Choose exactly one route.

## Route A: restore the proven Slot2 primitive

Reuse the previously qualified register/L1-only drive exactly:

- eight floating-point dependency chains;
- four integer chains;
- no shared-memory traffic inside the drive;
- identical compiler flags and optimization assumptions.

## Route B: qualify a new primitive

A new drive must pass a separate calibration campaign before scientific acquisition.

Minimum calibration:

- real sender-on and sender-off windows;
- at least two routes;
- all twelve tones;
- amplitude levels 1, 2, and 3;
- requested-frequency spectral line;
- phase recovery;
- minimum predeclared SNR or driven/null separation;
- thermal and frequency stability;
- repeated sessions and reboot.

Do not import Slot2 sensitivity into a different primitive.

---

# 5. Physical Scramble Contract

Scramble must alter the sender's physical control while leaving the receiver-side declared reference unchanged.

The schedule should contain explicit fields such as:

```json
{
  "family": "scramble",
  "receiver_codeword_source_index": 4,
  "sender_codeword_source_index": 9,
  "receiver_theta_idx": 2,
  "sender_theta_idx": 7,
  "shared_schedule": false,
  "scramble_key_digest": "<sha256>"
}
```

The executor must:

1. parse both sender and receiver fields;
2. drive only the sender fields;
3. record both in raw evidence;
4. prevent analysis code from reading sender-private fields during ordinary decoding;
5. bind the scramble mapping before acquisition;
6. prove in tests that real and scramble produce different physical gate digests.

The current `codeword_source_index` field is insufficient for both roles.

---

# 6. Authorization V2

The V2 authorization artifact must bind:

```json
{
  "executor_commit": "<40 hex>",
  "executor_sha256": "<64 hex>",
  "source_bundle_sha256": "<64 hex>",
  "campaign_plan_sha256": "<64 hex>",
  "session_ids": ["all exact session IDs"],
  "route_cores": {
    "v4s5": [4, 5],
    "v2s3": [2, 3]
  },
  "pin_khz": 1600000,
  "slot_s": 0.5,
  "off_window_s": 0.5,
  "read_hz": 4000,
  "temperature_veto_c": 68.0,
  "automatic_retry": false,
  "restoration_authorized": false
}
```

Both the orchestrator and C runner must reject any mismatch.

Subset execution requires a separate authorization artifact.

---

# 7. Run Manifest V2

The run manifest must bind the complete run directory after orchestration.

Required files include:

- `run.json`;
- `session.json`;
- `windows.jsonl`;
- `window_results.csv`;
- `raw_samples.bin`;
- `telemetry.csv`;
- `stdout.log`;
- `stderr.log`;
- `orchestrator_stdout.log`;
- `orchestrator_stderr.log`.

Verification must compare:

```text
actual directory file set
==
manifest file set
```

not only verify listed files.

The outer acquisition inventory remains required.

---

# 8. Frequency Settling Gate

The first window of all twelve V1 sessions crossed a reported cpufreq settling transition.

V2 must not begin scientific capture immediately after writing frequency controls.

Add an explicit settling gate:

1. write min, max, and boost;
2. verify policy readback;
3. poll `scaling_cur_freq` on relevant cores;
4. require the requested value for a predeclared consecutive duration or sample count;
5. record the settling evidence;
6. only then establish the first scientific origin.

If `scaling_cur_freq` cannot be trusted as an exact hardware-frequency indicator, define and bind an alternative APERF/MPERF criterion.

No gauge row may double as the frequency-settling probe.

---

# 9. Scientific Conformance Tests

The following tests are mandatory before target acquisition.

## 9.1 Pure waveform tests

For every amplitude level:

- dominant ideal Fourier component at requested `f`;
- near-zero error against the reference waveform;
- one theta step changes phase by `π/4`;
- four phase steps change phase by `π`;
- gate duty matches 2/8, 4/8, or 6/8.

## 9.2 Synthetic lock-in tests

Generate timestamps and a synthetic carrier from the reference gate.

Require:

- requested-frequency lock-in recovery;
- exact theta recovery;
- exact sign recovery;
- amplitude ordering;
- off-bin rejection.

## 9.3 Scramble tests

Require:

- `shared_schedule=true` makes sender and receiver gate digests identical;
- `shared_schedule=false` makes them different;
- the receiver-visible schedule cannot reconstruct the sender gate;
- scramble still preserves matched workload statistics.

## 9.4 C to Python equivalence

For frozen fixtures, compare C and `analysis/waveform_reference.py` on:

- tones;
- codebook;
- phase indices;
- gate state at every timestamp;
- lock-in I and Q.

Bit equality is preferred. Explicit numerical tolerance must otherwise be frozen.

## 9.5 Real hardware calibration

Engineering smoke must include a scientific waveform check, not only lifecycle.

Require:

- requested `f` response above a predeclared null boundary;
- correct sign and theta phase;
- sender-off null;
- no first-window frequency transition.

---

# 10. V1 Evidence Analysis Rules

The V1 evidence is not rerun data.

Analyze it under the waveform that actually executed.

Required coordinates:

- `f/4`;
- `f/2`;
- `3f/4`;
- `f`;
- `5f/4`;
- `3f/2`;
- exact gate matched filter.

Keep two ledgers:

```text
ORIGINAL_FROZEN_ANALYSIS
IMPLEMENTATION_RECOVERY_ANALYSIS
```

Do not silently replace the frozen coordinate.

The implementation-recovery lane may use training sessions to freeze the corrected state representation, then apply that representation unchanged to validation, stress, and final test sessions.

Scramble rows are reclassified as:

```text
DRIVEN_ROWS_WITH_INVALID_SCRAMBLE_LABEL
```

They may not enter the scramble-null pass condition.

Window zero of each session must be excluded from ordinary stationary gauge estimation or modeled as a separate settling transition.

---

# 11. Rerun Gate

Do not authorize a rerun merely because V1 has defects.

A rerun becomes justified only after one of these outcomes:

```text
V1 raw evidence cannot support the declared transport question
V1 corrected coordinate fails cross-session and cross-route transfer
missing scramble null blocks the only remaining claim
V2 asks a materially new question after V1 adjudication
```

The first raw training session already supports:

- strong driven/null separation at `f/4`;
- exact gate phase coherence;
- complete actual-mode recovery in Stage B;
- high theta recovery.

Therefore:

```text
IMMEDIATE_RERUN_NOT_AUTHORIZED
FULL_V1_RAW_AUDIT_FIRST
```

---

# 12. Agent Work Package

The local agent should implement one architectural change set:

```text
Phase 6 V2 executor and conformance qualification
```

Minimum deliverables:

- corrected C waveform;
- restored or newly qualified drive primitive;
- physical scramble fields and execution;
- authorization V2;
- total run-manifest closure;
- frequency-settling gate;
- C to Python waveform equivalence tests;
- synthetic spectral tests;
- target engineering calibration;
- new source-transfer bundle;
- new audit packet.

Do not split this into micro-commits.

The historical V1 acquisition remains unchanged and auditable.
