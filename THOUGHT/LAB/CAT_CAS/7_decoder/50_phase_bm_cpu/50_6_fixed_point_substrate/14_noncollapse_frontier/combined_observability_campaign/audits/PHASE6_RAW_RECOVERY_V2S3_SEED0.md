# Phase 6 Raw Recovery Audit: v2s3_seed0

**Executor:** `81ea84f341b29c41b93667d0e0fb98e0975bcbcf`  
**Session:** `v2s3_seed0`  
**Partition:** `train`  
**Raw SHA-256:** `6d7b6d61ca2b653e48cba63eec5830adcb7915755a997c99462af9c208d7f3b7`  
**Status:** `RAW_RECOVERY_SUPPORTED__FULL_CAMPAIGN_PENDING`

# Executive Verdict

The first raw session is authentic, structurally complete, and scientifically useful.

```text
RAW IDENTITY: PASS
RAW RECORD CLOSURE: PASS
TIMESTAMP CLOSURE: PASS
STORED LOCK-IN REPRODUCTION: PASS

REQUESTED f AS PRIMARY COORDINATE: REJECTED
ACTUAL f/4 CARRIER: STRONGLY SUPPORTED
EXACT EXECUTED GATE PHASE: STRONGLY SUPPORTED
SCRAMBLE NULL: INVALID
IMMEDIATE RERUN: NOT REQUIRED
FULL 12-SESSION RAW AUDIT: REQUIRED
```

The static code audit predicted that the sender's eight-state envelope repeats every four requested tone periods. The raw data confirm that prediction directly.

At the true envelope fundamental `f/4`, driven windows exceed the non-driven median by approximately:

```text
14.24×
```

At the originally stored `f` coordinate:

```text
driven median / non-driven median = 0.732×
```

The requested frequency does not carry the intended primary response.

The exact executed gate, reconstructed from each window's tone, code sign, theta, amplitude, origin TSC, and TSC calibration, matches the observed complex phase at `f/4` with:

```text
circular concentration R = 0.982395
median centered phase residual = 0.081277 rad
```

This is a strong deterministic carrier under the waveform that actually ran.

# 1. Raw Identity and Structure

```text
size:
265,216,000 bytes

SHA-256:
6d7b6d61ca2b653e48cba63eec5830adcb7915755a997c99462af9c208d7f3b7

records:
16,576,000

windows:
8,288

records per window:
2,000
```

Every raw window satisfies:

- first timestamp equals `first_sample_tsc`;
- last timestamp equals `last_sample_tsc`;
- timestamps increase strictly;
- all ring-period samples are finite;
- all records are consumed exactly;
- no trailing bytes remain.

# 2. Stored Lock-In Reproduction

The original C lock-in was independently reproduced from raw timestamps and samples.

```text
maximum I error:
8.882e-16

maximum Q error:
8.327e-16
```

The stored `computed_I` and `computed_Q` columns faithfully represent the implemented C calculation. The mismatch is in the waveform contract, not evidence corruption.

# 3. Harmonic Response

| Coordinate | Driven median | Non-driven median | Median ratio |
|---|---:|---:|---:|
| `f/4` | 0.107040856 | 0.007516730 | 14.240× |
| `f/2` | 0.072808936 | 0.006809049 | 10.693× |
| `3f/4` | 0.035661707 | 0.006694850 | 5.327× |
| `f` | 0.004907746 | 0.006708264 | 0.732× |
| `5f/4` | 0.021778023 | 0.006333186 | 3.439× |
| `3f/2` | 0.024343974 | 0.006828735 | 3.565× |
| `2f` | 0.004847927 | 0.006458819 | 0.751× |

The harmonic ladder follows the executed rectangular envelope:

```text
f/4     strongest
f/2     strong
3f/4    present
f       ideal null and empirically absent
5f/4    present
3f/2    present
2f      ideal null and empirically absent
```

# 4. Exact Gate Match

The exact executed binary gate was reconstructed independently for every driven window.

```text
median weighted regression beta = 0.223468
median matched-gate correlation = 0.477692
circular phase concentration R = 0.982395
```

The phase relation remains strong across all twelve physical tones. The response is not produced by one isolated bin.

# 5. Stage B Control Recovery

A diagnostic recovery was performed on the training session only.

Procedure:

1. estimate a complex transfer coefficient for each physical tone from `A_GAUGE`;
2. exclude session window zero because it crosses the cpufreq settling transition;
3. retain the exact executed `f/4` gate;
4. test all 32 `(mode, theta)` candidates for each Stage B symbol;
5. select the minimum complex residual.

This is an implementation-recovery diagnostic, not final frozen campaign adjudication.

| Family | Symbols | Mode accuracy | Theta accuracy | Joint accuracy |
|---|---:|---:|---:|---:|
| real | 48 | 1.000 | 0.938 | 0.938 |
| wrong | 48 | 1.000 | 0.958 | 0.958 |
| pseudo | 48 | 1.000 | 0.958 | 0.958 |
| order_sham | 48 | 1.000 | 0.917 | 0.917 |
| scramble | 48 | 1.000 | 0.917 | 0.917 |
| silent | 48 | 0.271 | 0.250 | 0.083 |

Across 240 driven Stage B symbols:

```text
mode accuracy = 1.000
theta accuracy = 0.938
joint accuracy = 0.938
```

For `wrong` rows:

```text
best recovered mode equals declared wrong mode = 0.000
```

The physical carrier follows actual execution rather than wrong declaration metadata.

The `scramble` rows recover like ordinary driven rows because no physical scramble occurred. This confirms the code-audit finding.

# 6. Frequency-Settling Row

Only the first window reports a current-frequency transition:

```text
frequency before = 800,000 kHz
frequency after = 1,600,000 kHz
```

All later windows report 1.6 GHz before and after.

Required treatment:

```text
exclude window 0 from gauge estimation
or
model it as a cpufreq-settling transition
```

# 7. Rerun Decision

A rerun is not justified yet.

The current raw session demonstrates:

- strong driven/non-driven separation;
- exact gate phase coherence;
- complete actual-mode recovery;
- high theta recovery;
- valid raw custody;
- a deterministic correction coordinate.

Run the same audit on all twelve sessions before deciding whether V2 hardware acquisition is necessary.

A corrected executor is still required before any future acquisition because `f` must be the actual fundamental, scramble must alter physical execution, the drive primitive must be restored or requalified, runtime parameters must be authorization-bound, and waveform conformance must be tested before hardware acquisition.

# 8. Claim Ceiling

Allowed:

> The first returned raw Phase 6 session is byte-valid and structurally complete. The implemented sender created a strong, phase-coherent cross-core carrier at the actual envelope fundamental `f/4`. The exact executed mode was recovered for every driven Stage B symbol in this training session. The originally stored `f` lock-in is not the correct primary coordinate, and the scramble control was not physically implemented.

Not yet allowed:

- the full twelve-session campaign passes Stage B;
- post-drive persistence is established;
- a stable predictive operator is identified;
- cross-route transfer is established;
- target coupling exists;
- a fold-odd invariant exists;
- orientation was recovered;
- the Small Wall was crossed.

# 9. Canonical State

```text
PHASE6_ACQUISITION_COMPLETE
PROVENANCE_VALID
METADATA_INTEGRITY_PASS
V2S3_SEED0_RAW_IDENTITY_PASS
V2S3_SEED0_RAW_STRUCTURE_PASS
REQUESTED_F_COORDINATE_REJECTED
F_OVER_4_CARRIER_SUPPORTED
EXACT_GATE_PHASE_SUPPORTED
TRAINING_MODE_RECOVERY_SUPPORTED
SCRAMBLE_NULL_INVALID
IMMEDIATE_RERUN_NOT_REQUIRED
ALL_SESSION_RAW_AUDIT_REQUIRED
SCIENTIFIC_VERDICT_PENDING
```
