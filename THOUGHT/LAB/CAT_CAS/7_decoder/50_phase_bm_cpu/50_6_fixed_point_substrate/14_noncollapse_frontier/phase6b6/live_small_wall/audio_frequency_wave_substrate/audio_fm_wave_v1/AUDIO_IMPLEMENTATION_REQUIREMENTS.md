# Audio Implementation Requirements

Status: `IMPLEMENTED_OFFLINE_REFERENCE`

## Package Boundary

All authored and generated content stays in `audio_fm_wave_v1/`. No main, Family 10h,
Small Wall global state, OrbitState evidence, stash, hardware, or network surface is
part of this implementation.

## Reference Runtime

`audio_wave_reference.py` is the single offline implementation. It uses Python plus
NumPy and imports no audio-device or network package. Operations:

```text
build      regenerate deterministic fixtures, manifest, test freeze, and results
verify     parse and hash committed outputs without regeneration
self-test  build then verify
```

The exact reference runtime is Python `3.11.6` with NumPy `1.26.4`. The engine fails
closed on a different runtime. Reported metrics are canonicalized to 12 significant
digits before JSON serialization so platform-level last-bit noise does not become an
unstated identity variable.

## Frozen Envelope

```text
sample_rate_hz          48000
duration_seconds        2.0
sample_count            96000
primary_carrier_hz      8000
baseband_limit_hz       1000
absolute_sample_ceiling 0.95
carrier_amplitude       0.90
wav_encoding            IEEE_FLOAT32_LE
default_edge_samples    4096 per side
FM k                    420 Hz/unit
PM k                    1.15 rad/unit
```

## Required Operations And Tests

The engine must implement and test:

```text
FM encode/recover
PM encode/recover
FFT analytic signal
conjugate phase subtraction
ordinary phase addition
complex multitone state
circular and zero-filled sample delay
complex phase rotation
complex filter-bank projection
normalized and unnormalized correlation
matched filtering
full FFT convolution
controlled polynomial nonlinear mixing
```

The frozen test file defines 29 tests and their exact metrics. Test order and identity
must match the result file exactly.

Every test that names a WAV input consumes samples parsed back from that committed
float32 file. Verification recomputes the complete manifest, test freeze, observations,
comparators, summary, and claim token; stored `PASS` strings are never trusted.

## Fixture Law

Each manifest record binds:

```text
fixture id and path
semantic role
generator parameters
sample rate and count
duration
channel count and dtype
peak and RMS amplitude
byte count and SHA-256
RIFF chunk list
```

Committed WAVs contain exactly `fmt ` and `data`. Float32 quantization occurs only at
serialization; calculations use float64/complex128.

## Determinism

- No random seed or clock is used.
- JSON is sorted, two-space indented, ASCII, newline-terminated.
- WAV chunk order and encoding are fixed.
- Frequencies in exact projection tests occupy integer-cycle bins.
- Result metrics are quantized to 12 significant decimal digits for stable JSON.
- The frozen Python and NumPy versions are bound into manifest, test, and result JSON.
- Temporary output is replaced atomically.

## Validation Commands

The lane validation must include:

```text
Python syntax compilation
reference-engine self-test
reference-engine verify-only pass
all JSON parse
all WAV parse
fixture hash verification
every named reference/adversary test
cross-document identity scan
git diff --check
governance critic (requested integration check only)
ci_local_gate.py --full (requested integration check only)
```

The last two are requested by the bootstrap despite the LAB-local exemption; unrelated
failures must be reported without changing non-audio files.

## Future Physical Freeze Requirements

Before one carrier is frozen, the integration owner must add an exact hardware packet
with:

```text
schematic or mechanical geometry
component identities and tolerances
source/query/read port separation
break-before-make disconnect truth table
maximum energy, voltage/current or acoustic limit
buffer-drain proof
baseline and ring-down campaign
measurement-disturbance bound
restoration mechanism and R2 metric
carrier-off, wrong-query, wrong-inverse, and natural-relaxation controls
```

No implementation in this package supplies or authorizes those items.
