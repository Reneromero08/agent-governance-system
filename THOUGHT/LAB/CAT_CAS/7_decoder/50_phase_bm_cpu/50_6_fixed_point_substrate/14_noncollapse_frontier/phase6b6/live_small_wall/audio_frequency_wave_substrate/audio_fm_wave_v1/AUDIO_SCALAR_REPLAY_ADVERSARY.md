# Audio Scalar And DSP Replay Adversary

Status: `FROZEN_AND_EXECUTED_OFFLINE`

## Adversary Input

The attacker receives:

```text
every source-visible input and receipt
all committed WAV files
the fixture manifest and reference-test law
public generator and receiver source
all generator parameters
the realized query at scoring time
```

Nothing in the offline positive result depends on obscurity.

## Attack Registry

| Attack | Construction | Offline outcome | Consequence |
| --- | --- | --- | --- |
| Finite cache | Store all four phase-query responses | Exact replay | Delayed finite query is non-identifying |
| Compressed generator | Store amplitude and phase; answer `A cos(phi-q)` | Exact held-out replay | Continuous query alone is non-identifying |
| Ordinary DSP | FFT Hilbert, unwrap, derivative, complex projection | Meets all frozen tolerances | Python is the offline computer |
| Linear filter | Public FFT convolution | Exact numerical replay | Linear response is ordinary |
| Nonlinear filter | Public `x+0.8x^2` law | Replay within frozen float32 serialization tolerance | Intermodulation is ordinary |
| Spectral energy | Compare matched target and time-reversed sham | Cannot explain matched gap | Energy-only leakage killed for this fixture |
| Phase label | Rename aliases without changing samples | Strict null | Label strings are not phase evidence |
| File metadata | Add and strip RIFF LIST chunk | Strict invariant | Metadata is not required |
| Manifest parameters | Reconstruct multitone coefficients from public parameters | Exact replay | Public non-sample side channel survives |
| Query preselection | Give source `q` before closure | Exact answer | Custody violation is answer-smuggleable |
| File persistence | Reopen committed WAV | Observable reproduced | Persistence is serialized, not physical |
| Interface buffer | Treat queued samples as state | Admissible until mechanically drained | Future prototype must exclude it |

## Leakage Laws

The manifest records semantic roles and generator parameters required to reproduce a
fixture. It must not contain expected answers, truth labels, winning labels, or scored
query results. Committed WAV chunks are exactly `fmt ` and `data`.

Magnitude-spectrum equality is verified for neutral pair members A and B while their
matched-filter responses remain separated. This attacks energy leakage only for the
declared matched-filter fixture; it does not prove all energy attacks impossible.

The manifest intentionally exposes generator parameters for reproducibility. The
reference therefore executes a manifest-parameter replay and records that it succeeds.
Fixture aliases and the matched pair use neutral names; changing aliases is a true
bijective map in the test harness and cannot alter content-bound scoring.

## Finite And Compressed Generators

The finite-cache test uses:

```text
Q = {0, pi/2, pi, 3pi/2}
```

The held-out generator test uses phase queries absent from that cache. Both attacks
succeed. The second is the important correction: a high-resolution query family may
still have a tiny answer program.

## Physical Successor Requirement

A future carrier claim must replace unbounded software replay with a measured bounded
physical channel, freeze the adversary class before evidence, and demonstrate a
capacity or intervention separation. Failure of one trained model is not enough.

## Offline Adjudication

Ordinary replay survival is required for success of this package. Interface-buffer
persistence is not exercised: it remains a future physical drain/source-off control,
separate from the executed serialized-file reopen test. If the reference
results claimed ordinary replay failed, the offline claim would be overstated or the
adversary implementation would be defective.
