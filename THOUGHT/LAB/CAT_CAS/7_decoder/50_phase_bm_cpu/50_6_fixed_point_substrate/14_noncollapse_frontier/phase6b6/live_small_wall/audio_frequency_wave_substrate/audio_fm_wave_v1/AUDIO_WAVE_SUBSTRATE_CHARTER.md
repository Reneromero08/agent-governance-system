# Audio Wave Substrate Charter

Status: `OFFLINE_ARCHITECTURE_IMPLEMENTED_AWAITING_INTEGRATION_REVIEW`

## Mission

This package establishes a deterministic audio-frequency wave algebra and a reviewed
architecture for a later physical carrier. It does not play audio, record audio, or
contact any hardware. Its maximum positive scientific result is:

```text
AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED
```

The offline engine is deliberately reproducible by ordinary software. A WAV file is
serialized state and Python is the computer. That fact is a required adversary result,
not a defect to hide.

## Objects Preserved

The primary objects are:

```text
real waveform x[n]
analytic complex state z[n]
complex spectral coefficient c_k
frequency and phase trajectory phi[n]
receiver query q
relative state z_state * conjugate(z_query)
delay, correlation, convolution, and nonlinear operators
```

Magnitude-only summaries are controls, not the ontology of the lane.

## Frozen Offline Envelope

```text
sample rate                 48000 Hz
duration                    2.0 s
sample count                96000
primary carrier             8000 Hz
baseband bandwidth          <= 1000 Hz
absolute sample ceiling     <= 0.95
committed carrier amplitude 0.90
WAV encoding                IEEE float32 little-endian
channels                    mono, except explicit stereo I/Q fixtures
default analytic edge crop  4096 samples at each end
```

The exact metric, comparator, tolerance, input identity, and edge convention for every
test are authoritative in `AUDIO_WAVE_REFERENCE_TESTS.json`. Fixture identity is
authoritative in `AUDIO_WAVE_FIXTURE_MANIFEST.json`.

## Claim Boundary

Allowed:

```text
AUDIO_FM_WAVE_ALGEBRA_ESTABLISHED
offline ordinary-DSP replay established
finite-answer-cache equivalence reproduced
compressed-generator equivalence reproduced
physical carrier candidates mechanically specified for review
```

Forbidden:

```text
AUDIO_POST_SOURCE_STATE_OBSERVED
PHYSICAL_AUDIO_COMPUTING_ESTABLISHED
RELATIONAL_CARRIER_ESTABLISHED
PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED
CATALYTIC_BORROWING_ESTABLISHED
SMALL_WALL_CROSSED
```

## No-Smuggle Boundary

Every offline adversary receives all source-visible inputs, all generator parameters,
all committed WAVs, the manifest, public source, and the realized query at scoring
time. No filename, RIFF chunk, manifest field, label alias, or schedule index may carry
a hidden expected answer. Public generator parameters are explicitly classified as a
surviving replay side channel. The scored matched pair uses neutral aliases, and the
committed WAVs contain only `fmt ` and `data` chunks.

## Hardware Boundary

This package contains no playback, microphone, ADC, DAC, network, PMU, controller, or
live-authority code. Hardware contact counts are frozen to zero. A later prompt must
separately authorize a physical prototype.

## Exit Condition

The lane is ready for integration review only when the reference suite passes, all
fixtures parse and hash, exactly four independent reviews are archived, all material
findings are normalized, and no report exceeds the offline claim ceiling.
