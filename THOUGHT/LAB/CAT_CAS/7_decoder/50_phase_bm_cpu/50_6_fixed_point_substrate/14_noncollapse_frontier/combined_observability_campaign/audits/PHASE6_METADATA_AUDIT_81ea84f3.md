# Phase 6 Acquisition Metadata Audit

**Executor:** `81ea84f341b29c41b93667d0e0fb98e0975bcbcf`  
**Inputs:** `Runs_no_bin.zip`, `Sessions.zip`  
**Audit mode:** read-only, no raw binary samples  
**Status:** `METADATA_AUDIT_PASS__RAW_CONTENT_PENDING`

---

# Executive Verdict

The uploaded metadata archives are internally coherent.

```text
ARCHIVE SAFETY: PASS
SESSION SET: 12/12
WINDOW SET: 99,456/99,456
SCHEDULE TO RESULT AGREEMENT: PASS
TELEMETRY TO RESULT AGREEMENT: PASS
RUN MANIFEST BINDINGS: PASS FOR ALL PRESENT FILES
RAW SIZE RECONCILIATION: PASS
RAW CONTENT HASH VERIFICATION: PENDING
SCIENTIFIC WAVEFORM VALIDATION: PENDING
SCRAMBLE NULL: INVALID BY CODE AUDIT
```

No metadata mismatch was found across the 12 sessions.

The uploaded run copies of `session.json` and `windows.jsonl` are byte-identical to the independently uploaded compiled session bundles. Every session-manifest binding verifies. Every present file listed by each run manifest verifies. The only missing run-manifest member is the intentionally removed `raw_samples.bin`.

The sample counts imply exactly:

```text
198,912,000 raw records
× 16 bytes per record
= 3,182,592,000 bytes
```

This equals the sum of all twelve raw sizes declared by the original run manifests.

The dataset therefore has strong structural closure before the raw bytes are inspected.

---

# Archive Identity

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `Runs_no_bin.zip` | 18,775,994 | `119b4224324058d46b08a475dfd11825ff0db81b3c7459a0e81d334707f27bf8` |
| `Sessions.zip` | 1,411,303 | `0e2c4fe3493f6dc9a9f3c46f083a8253625fa115f220e12c512fc84292c4a863` |

Both ZIP archives contain:

- no absolute paths;
- no parent-directory traversal;
- no duplicate archive members.

---

# Global Provenance Bindings

Every run records exactly one common value for each binding:

```text
authorization:
e39fb0c6ebfb106c33a0b90b8d52d193a32833103388ebc4c6bd0cad451a0d73

campaign plan:
eb5a46d0a37d66910649467cf0d4e3cf947dee11fab94a36e9bdfed388455e53

executor commit:
81ea84f341b29c41b93667d0e0fb98e0975bcbcf

campaign source commit:
f5b6079a5748bb6138ab19d1c22d79c74734dddf
```

This independently supports the proof packet’s claim that all twelve runs belong to one frozen execution object.

---

# Session Summary

| Session | Partition | Windows | Samples | Min °C | Max °C | Raw SHA-256 |
|---|---|---:|---:|---:|---:|---|
| v2s3_seed0 | train | 8,288 | 16,576,000 | 46.500 | 52.125 | 6d7b6d61ca2b653e48cba63eec5830adcb7915755a997c99462af9c208d7f3b7 |
| v2s3_seed1 | train | 8,288 | 16,576,000 | 50.000 | 52.000 | 6cd56279f3c4eef2bf572d04dd9f7002c64ae41819a952f9b85cc07aea16581e |
| v2s3_seed2 | train | 8,288 | 16,576,000 | 49.000 | 52.000 | 693c4a51eed736118f2970fb712943e5ebfde7daf4b6efc166af7e3e0bee61a2 |
| v2s3_seed3 | validation | 8,288 | 16,576,000 | 48.000 | 50.000 | 624bbf6d0b58769ee1a6769b04a8832c7f2e221c7a38785abd0f7662e755e879 |
| v2s3_seed4 | stress | 8,288 | 16,576,000 | 46.000 | 49.500 | 14817f963dbe05b67a37f026b14ae73159e7c9005c5ed3c3c76bdf739943b9d0 |
| v2s3_seed5 | test | 8,288 | 16,576,000 | 46.000 | 48.500 | 6522e31954b80b026252b5e85a578007beccb301e84e1a6f9af6b48e10568085 |
| v4s5_seed0 | train | 8,288 | 16,576,000 | 41.000 | 47.000 | 5a6d018535bf883cac036e0b19e65c6ea31db0b6fdcb6a439d979d9da66ae6f5 |
| v4s5_seed1 | train | 8,288 | 16,576,000 | 41.000 | 47.000 | 654473a77f57c0c3a86e2a14ff28f5ace7c7dd5da4ec0510f2f65e476620fd52 |
| v4s5_seed2 | train | 8,288 | 16,576,000 | 41.000 | 47.000 | 0d92e6e19585e38a69f85241b908f96b8c05283d7f98cdf9f142c1aae30398ac |
| v4s5_seed3 | validation | 8,288 | 16,576,000 | 41.000 | 47.625 | abf608c54ef9ed8116e2f5f79780280929f4ed157a949aa189b4e0f1e5285fb4 |
| v4s5_seed4 | stress | 8,288 | 16,576,000 | 41.000 | 48.000 | b2fbdc0b5f603b7049a6fe4e7c102bf8cb17d1888758b5fe6d80dad8c37fc39b |
| v4s5_seed5 | test | 8,288 | 16,576,000 | 41.000 | 48.000 | a567f369789af1e25884cd634f349ee8604672fa102b9c4d6bbf54df47e9d20f |

Each session contains:

```text
8,288 windows
16,576,000 records
265,216,000 raw bytes
```

Every window contains exactly 2,000 raw records.

---

# Exact Campaign Composition

## Stage counts

```json
{
  "A_GAUGE": 1152,
  "B_TONE_ORDER": 41472,
  "C_PERSISTENCE_OFF": 6144,
  "C_PERSISTENCE_PREPARE": 13824,
  "D_TRAJECTORY": 36864
}
```

## Family counts

```json
{
  "gauge_preamble": 1152,
  "impulse": 7680,
  "order_sham": 6912,
  "pseudo": 6912,
  "real": 6912,
  "scramble": 6912,
  "silent": 6912,
  "step": 12288,
  "trajectory": 36864,
  "wrong": 6912
}
```

## Amplitude counts

```json
{
  "0": 22272,
  "1": 8844,
  "2": 9132,
  "3": 59208
}
```

The exact total is:

```text
77,184 driven windows
22,272 non-driven windows
99,456 total windows
```

The metadata contain:

```text
6,912 scramble-labeled rows
6,912 rows with shared_schedule=false
```

Those are the same rows. The prior static code audit established that the executor never consumes `shared_schedule`, so these labels do not correspond to a physically scrambled schedule.

---

# Schedule and Result Reconciliation

For all 99,456 windows, the following fields agree exactly between the frozen schedule and the returned CSV:

- window index;
- session;
- stage;
- block;
- family;
- actual mode;
- declared mode;
- executed tone order;
- declared tone order;
- physical tone index;
- codeword source index;
- drive state;
- sender-off requirement;
- measurement mode;
- amplitude level;
- theta index.

All window indices are contiguous within their sessions.

No result row is missing, duplicated, reordered, or attached to the wrong session.

---

# Telemetry Reconciliation

For every window, `telemetry.csv` agrees exactly with the matching `window_results.csv` row for:

- temperature before and after;
- current frequency before and after;
- APERF before and after;
- MPERF before and after;
- COFVID before and after.

No non-finite numeric values were found in the available result columns.

Exactly 6,144 rows contain null I/Q. These correspond to the `C_PERSISTENCE_OFF` raw-ring-only stage.

---

# Sender Lifecycle Audit

All driven windows satisfy the recorded lifecycle contract:

- sender ready before origin;
- sender epoch within the 5 ms alignment limit;
- first drive between origin and deadline;
- sender started;
- sender remained alive during capture;
- sender stopped and joined.

All non-driven windows satisfy:

- no sender-ready timestamp;
- no sender epoch;
- no first-drive timestamp;
- no sender thread started;
- no sender alive during capture.

This includes both explicitly sender-off windows and silent controls.

## Timing distributions

```json
{
  "capture_span": {
    "max": 499.8812699558927,
    "mean": 499.81745621500903,
    "median": 499.82029443307425,
    "min": 499.75962924323017,
    "p95": 499.83434980801974,
    "p99": 499.8404434656797
  },
  "first_drive_delay": {
    "max": 43.369253718237594,
    "mean": 5.38386240040146,
    "median": 1.6151428998602513,
    "min": 8.523045761580469e-05,
    "p95": 24.782461376283138,
    "p99": 37.173637936825074
  },
  "last_sample_minus_deadline": {
    "max": 0.13140048083842754,
    "mean": 0.0675957662252053,
    "median": 0.07042727915352112,
    "min": 0.009772043958981064,
    "p95": 0.08448735055179354,
    "p99": 0.09059377178908606
  },
  "receiver_epoch_skew": {
    "max": 0.01299982458096526,
    "mean": 6.675304002285973e-05,
    "median": 6.626063614627595e-05,
    "min": 4.603688127295768e-05,
    "p95": 8.429728896335595e-05,
    "p99": 8.678576981836277e-05
  },
  "sender_epoch_skew": {
    "max": 0.008613254753228564,
    "mean": 6.813332272939654e-05,
    "median": 6.781605014032e-05,
    "min": 4.603688127295768e-05,
    "p95": 8.5547769672422e-05,
    "p99": 8.802998191824099e-05
  }
}
```

The largest sender-epoch skew is far below 5 ms.

The largest observed last-sample deadline overrun is approximately 0.1314 ms, far below the 20 ms allowed overflow guard.

---

# Newly Identified Frequency-Settling Anomaly

The first window of every session reports a current frequency transition during capture:

| Session | Window | Stage | Before kHz | After kHz |
|---|---:|---|---:|---:|
| v2s3_seed0 | 0 | A_GAUGE | 800,000 | 1,600,000 |
| v2s3_seed1 | 0 | A_GAUGE | 800,000 | 1,600,000 |
| v2s3_seed2 | 0 | A_GAUGE | 800,000 | 1,600,000 |
| v2s3_seed3 | 0 | A_GAUGE | 800,000 | 1,600,000 |
| v2s3_seed4 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |
| v2s3_seed5 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |
| v4s5_seed0 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |
| v4s5_seed1 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |
| v4s5_seed2 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |
| v4s5_seed3 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |
| v4s5_seed4 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |
| v4s5_seed5 | 0 | A_GAUGE | 3,200,000 | 1,600,000 |

All twelve affected rows are:

```text
window_index = 0
stage = A_GAUGE
family = gauge_preamble
actual_mode = basis
theta_idx = 0
physical_tone_index = 0
```

After window zero, all remaining 99,444 windows report:

```text
frequency_before_khz = 1,600,000
frequency_after_khz  = 1,600,000
```

## Scientific consequence

The first gauge observation in each session was captured across a reported frequency transition.

Because the analysis contract estimates session gauge only from `A_GAUGE`, these twelve rows must not silently enter the gauge estimator as ordinary stationary samples.

Required treatment:

```text
exclude window 0 from each session gauge
or
model it as a distinct frequency-settling transition
```

The treatment must be declared before outcome analysis.

---

# Raw Binary Reconciliation

Although the `.bin` files were removed from the upload, their exact expected identities remain in the run manifests.

| Raw file | Bytes | SHA-256 |
|---|---:|---|
| `v2s3_seed0/raw_samples.bin` | 265,216,000 | `6d7b6d61ca2b653e48cba63eec5830adcb7915755a997c99462af9c208d7f3b7` |
| `v2s3_seed1/raw_samples.bin` | 265,216,000 | `6cd56279f3c4eef2bf572d04dd9f7002c64ae41819a952f9b85cc07aea16581e` |
| `v2s3_seed2/raw_samples.bin` | 265,216,000 | `693c4a51eed736118f2970fb712943e5ebfde7daf4b6efc166af7e3e0bee61a2` |
| `v2s3_seed3/raw_samples.bin` | 265,216,000 | `624bbf6d0b58769ee1a6769b04a8832c7f2e221c7a38785abd0f7662e755e879` |
| `v2s3_seed4/raw_samples.bin` | 265,216,000 | `14817f963dbe05b67a37f026b14ae73159e7c9005c5ed3c3c76bdf739943b9d0` |
| `v2s3_seed5/raw_samples.bin` | 265,216,000 | `6522e31954b80b026252b5e85a578007beccb301e84e1a6f9af6b48e10568085` |
| `v4s5_seed0/raw_samples.bin` | 265,216,000 | `5a6d018535bf883cac036e0b19e65c6ea31db0b6fdcb6a439d979d9da66ae6f5` |
| `v4s5_seed1/raw_samples.bin` | 265,216,000 | `654473a77f57c0c3a86e2a14ff28f5ace7c7dd5da4ec0510f2f65e476620fd52` |
| `v4s5_seed2/raw_samples.bin` | 265,216,000 | `0d92e6e19585e38a69f85241b908f96b8c05283d7f98cdf9f142c1aae30398ac` |
| `v4s5_seed3/raw_samples.bin` | 265,216,000 | `abf608c54ef9ed8116e2f5f79780280929f4ed157a949aa189b4e0f1e5285fb4` |
| `v4s5_seed4/raw_samples.bin` | 265,216,000 | `b2fbdc0b5f603b7049a6fe4e7c102bf8cb17d1888758b5fe6d80dad8c37fc39b` |
| `v4s5_seed5/raw_samples.bin` | 265,216,000 | `a567f369789af1e25884cd634f349ee8604672fa102b9c4d6bbf54df47e9d20f` |

For every session:

```text
sum(window sample_count) × 16
==
run_manifest raw_samples.bin size
==
265,216,000 bytes
```

This proves structural size closure, not content closure.

Raw content remains unverified until at least one original `.bin` is uploaded and hashed.

---

# Findings Carried Forward from Static Code Audit

## 1. Waveform semantics remain unresolved

The implemented eight-state sender envelope has an actual cycle of four requested tone periods. The stored lock-in demodulates at the requested frequency, which is the fourth harmonic of the envelope.

The raw timestamps and observations are therefore required to determine:

- whether measurable energy exists at the requested frequency;
- whether the dominant response occurs at `f/4` or another harmonic;
- whether exact-gate matched filtering recovers the executed control;
- what phase relation was physically transmitted.

## 2. Scramble is not a valid negative control

The metadata contain 6,912 scramble rows, but the executor did not physically scramble them.

They must be reclassified as:

```text
ordinary driven rows carrying an invalid scramble-control label
```

They may not support a claim that unshared schedule information destroys decoding.

## 3. Prior Slot2 sensitivity does not automatically transfer

The combined executor used a different and weaker ALU dependency chain than the prior proven Slot2 drive. The acquired raw data must establish its own carrier response.

---

# What This Audit Proves

The current upload independently proves:

- all twelve session schedule objects are present;
- every session schedule is hash-valid;
- the run copies exactly match the compiled schedules;
- all 99,456 expected result rows are present;
- all result metadata agree with the schedules;
- all telemetry rows agree with the results;
- all window statuses are `OK`;
- sender lifecycle metadata are internally consistent;
- raw sample counts close exactly to the original raw file sizes;
- all runs bind one executor, plan, campaign source, and authorization digest.

---

# What This Audit Cannot Yet Prove

Without the raw binaries and outer acquisition files, this audit cannot independently prove:

- the raw file contents match their run-manifest hashes;
- timestamp monotonicity inside each raw window;
- raw sample finiteness;
- exact binary slicing;
- spectrum at `f/4`, `f/2`, `3f/4`, `f`, or other harmonics;
- matched response to the exact executed gate;
- the outer 173-file inventory digest;
- the content of `execution.json`;
- the copied authorization, preflight, plan, and source bundle at the acquisition root.

---

# Next Required Upload

Upload this single training-partition file first:

```text
v2s3_seed0/raw_samples.bin
```

Expected identity:

```text
size:
265,216,000 bytes

SHA-256:
6d7b6d61ca2b653e48cba63eec5830adcb7915755a997c99462af9c208d7f3b7
```

Why this file:

- it is under the 512 MB upload limit;
- it is a training session;
- it uses the historically established `v2s3` route;
- it permits waveform recovery without opening validation, stress, or test outcomes.

The next raw audit will verify:

```text
16,576,000 binary records
8,288 exact window slices
timestamp monotonicity
sample finiteness
CSV boundary agreement
requested-frequency spectrum
actual-envelope spectrum
exact-gate matched filtering
silent and sender-off controls
```

Also upload the small outer files when convenient:

```text
execution.json
campaign_plan.json
campaign_manifest.json
source_bundle.json
ACQUISITION_AUTHORIZATION.json
the preflight JSON
the returned 173-file inventory
```

These are not needed for the first spectral reconstruction, but they are required to complete independent outer custody verification.

---

# Current Canonical State

```text
PHASE6_ACQUISITION_COMPLETE
PROVENANCE_VALID
METADATA_INTEGRITY_PASS
SCHEDULE_RESULT_CONFORMANCE_PASS
RAW_SIZE_CLOSURE_PASS
RAW_CONTENT_PENDING
FIRST_GAUGE_WINDOW_NONSTATIONARY
SCRAMBLE_NULL_INVALID
WAVEFORM_RECOVERY_PENDING
SCIENTIFIC_VERDICT_PENDING
SMALL_WALL_NOT_ADJUDICATED
```
