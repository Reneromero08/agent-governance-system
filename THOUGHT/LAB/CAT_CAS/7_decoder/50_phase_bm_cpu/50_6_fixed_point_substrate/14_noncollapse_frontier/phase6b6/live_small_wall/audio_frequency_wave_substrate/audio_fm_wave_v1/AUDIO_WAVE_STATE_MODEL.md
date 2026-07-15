# Audio Wave State Model

Status: `FROZEN_OFFLINE_MODEL`

## Digital State

For sample index `n`, sample period `T_s = 1/48000`, and record length `N = 96000`:

```text
x[n] in R                       real mono waveform
z[n] = x[n] + i H{x}[n]        analytic complex waveform
c_k = (2/N) sum x[n]e^-i2pi f_k nT_s
```

The factor `2/N` applies to a real signal projected onto a positive-frequency bin.
For an analytic or already-complex signal the factor is `1/N`. Frequencies used for
exact coefficient tests complete an integer number of cycles in the two-second record.

An explicit two-axis fixture stores `I = Re(z)` in channel 0 and `Q = Im(z)` in channel
1. Stereo never means two unrelated scalar examples.

## Analytic-Signal Convention

The reference uses an FFT Hilbert construction:

```text
DC multiplier       1
positive bins       2
Nyquist multiplier  1 for even N
negative bins       0
```

This convention is periodic at the FFT boundary. FM and PM recovery exclude 4096
samples at each record edge. No other test silently discards samples.

## Operator Algebra

```text
relative phase:  z_rel = z_state * conjugate(z_query)
phase addition:  z_sum = z_a * z_b
phase rotation:  R_theta(z) = z * exp(i theta)
sample delay:    D_d(x)[n] = x[(n-d) mod N] for circular fixtures
projection:      P_f(x) = scale * sum x[n]exp(-i2pi f nT_s)
```

The committed delayed multitone uses a 37-sample circular delay. The engine also
implements a zero-filled causal delay, but no fixture confuses its boundary with the
circular law.

## Multitone State

The frozen multitone is:

```text
f (Hz)  magnitude  phase (rad)
6300    0.22       0.31
8000    0.29      -0.72
9700    0.17       1.11
```

Its complex coefficients are state coordinates. Spectral magnitude alone is an
intentionally incomplete projection.

## Serialized State Versus Physical State

The committed WAV encodes samples and persists because a filesystem persists. Reading
it later proves only serialized-state replay. Operating-system buffers or an audio
interface queue would be the same class unless independently excluded.

A future physical state must instead name:

```text
energy-bearing degrees of freedom
source preparation operation
mechanical source disconnect
post-source lifetime distribution
fresh receiver query operation
observable and measurement disturbance
active restoration operation
accepted restoration equivalence
```

The offline model establishes none of those physical facts.

## Observable State Versus Total State

For every future candidate:

```text
observed distinguishable wave states != total physical state capacity
```

Transfer-function bins, modal amplitudes, modal phases, and ring-down samples are
operational observables. They do not count unmeasured microstates or prove independent
bits.

## Numerical Edge Laws

- WAV samples are quantized once to float32, parsed back, and only then promoted to
  float64/complex128 for every manifest-bound score.
- A normalized correlation with either zero norm returns complex zero.
- Phase errors use the principal angle of `observed * conjugate(expected)`.
- Full convolution assumes both finite inputs are zero outside support.
- PM removes the mean interior phase because a real carrier does not identify an
  absolute DC phase offset.
- NaN and infinity are forbidden in fixtures.
