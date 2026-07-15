# Audio Channel Capacity Model

Status: `MODEL_FROZEN_NUMERIC_CAPACITY_NOT_ESTABLISHED`

## Variables

```text
B                usable bandwidth after physical filtering
T_hold           independently measured post-disconnect lifetime
SNR(f,t)         query-relevant signal-to-noise distribution
M                number of independently resolved persistent modes
Q_k              quality factor of mode k
p_k              accepted response precision of mode k
C_prep           source-preparation capacity surviving disconnect
B_answer_raw     |Q| times response bits
B_answer_min     minimum ordinary answer-equivalent code length
B_relation       declared relation-state code length
S_side           public and hidden side-information budget
```

An AWGN-style upper comparison may use `B*T_hold*log2(1+SNR)`, but no future claim may
treat that ideal bound as a measured number. Persistent modal capacity is further
limited by mode correlation, read disturbance, calibration precision, damping, drift,
and restoration requirements.

## Required Separation

```text
B_answer_min > C_prep
B_relation <= C_prep
```

The minimization for `B_answer_min` includes:

```text
finite tables
compressed tables
closed-form response generators
low-rank spectral bases
complex coefficient sets
linear time-invariant filters
bounded nonlinear filters
public seeds and schedules
decoder computation and public side information
```

Raw WAV bytes, raw sample count, FFT bin count, or nominal ADC resolution are not
capacity lower bounds.

## Per-Candidate Measurements

A physical campaign must independently estimate:

- ring-down amplitude and phase versus delay;
- modal covariance and effective rank;
- response repeatability at fixed preparation;
- smallest stable query-relevant displacement;
- measurement disturbance from each query;
- source-off leakage and interface-buffer lifetime;
- formula/compression complexity of observed responses;
- state remaining after the declared restoration operation.

## Side-Information Accounting

File names, RIFF metadata, hardware routes, gain settings, query order, clock time,
buffer contents, interface latency, and decoder constants all count as side information
when available to an adversary. A capacity argument that ignores them is invalid.

## Offline Decision

The offline lane assigns no numeric `C_prep`, `B_answer_min`, or physical SNR. Its files
are ordinary serialized state with ample software replay capacity. Therefore:

```text
CAPACITY_SEPARATION_NOT_ESTABLISHED
```
