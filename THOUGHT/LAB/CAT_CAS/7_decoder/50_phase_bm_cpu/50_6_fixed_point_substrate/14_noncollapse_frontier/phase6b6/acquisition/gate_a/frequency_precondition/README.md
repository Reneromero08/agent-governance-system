# Gate A Frequency-Precondition Review

**Status:** `READ_ONLY_QUALIFICATION_IMPLEMENTED__TARGET_OBSERVATION_NOT_AUTHORIZED`

## Adjudicated evidence

The second owner-authorized Gate A attempt is sealed on `main` at
`26576400cb10c3dfb2968f44cc7066f4b143463a`.

It failed closed before physical runtime:

```text
temperature = 38.5 C
core 4 scaling_cur_freq = 800000 kHz
core 5 scaling_cur_freq = 800000 kHz
required observed frequency = 1600000 kHz
runtime starts = 0
sender starts = 0
capture starts = 0
retries = 0
```

The packet is valid. The authority is consumed. This result does not authorize
a retry or a weaker frequency veto.

Historical read-only target evidence from June 12, 2026 recorded cores 4 and 5
at `1600000 kHz`. The target can therefore expose the required state, but the
state is not stable across sessions.

## Frozen contract interpretation

The Gate A run plan states that the observed frequency must **already** equal
`1600000 kHz`. Gate A may observe and veto, but may not write frequency,
voltage, or MSR state.

Therefore:

- one idle `scaling_cur_freq` sample at `800000 kHz` is a valid veto;
- a sender or capture warmup cannot be smuggled into preflight;
- a temporary load that raises frequency does not prove the sender-absent idle
  slots begin at the required state;
- the Gate A executor must not gain a frequency-control surface;
- any future frequency preparation or restoration operation must be designed,
  reviewed, authorized, and evidenced separately from the smoke.

## Read-only qualification

`gate_a_frequency_precondition_probe.py` diagnoses the target cpufreq state
without changing it.

It reads only the frozen cores 4 and 5 and records:

- cpufreq policy identity;
- driver and governor;
- CPU and scaling frequency bounds;
- available governors and frequencies when exposed;
- affected and related CPU policy membership;
- a bounded paired `scaling_cur_freq` observation window;
- exact sample timing, values, and consecutive required-frequency runs.

It performs:

```text
network operations = 0
filesystem writes = 0
frequency writes = 0
voltage writes = 0
MSR reads = 0
MSR writes = 0
sender starts = 0
capture starts = 0
```

Observation outcomes are:

```text
PASS_STATIC_PRECONDITION_OBSERVED
    Every paired sample on cores 4 and 5 equals 1600000 kHz.

INCONCLUSIVE_DYNAMIC_PRECONDITION
    The required paired state appears but is not stable across the window.

FAIL_REQUIRED_FREQUENCY_NOT_OBSERVED
    The required paired state never appears.

FAILED_CLOSED_UNOBSERVABLE
    The cpufreq surface or required metadata cannot be closed.
```

Only the first outcome could support consideration of another Gate A owner
decision. It does not itself authorize one.

## Why this is separate from Gate A

This lane is diagnostic. It does not enter the 21-file Gate A execution bundle,
does not contact the target automatically, does not create an execution
authority, and cannot invoke the smoke.

The first target use, if separately owner-authorized, should execute only this
read-only probe and preserve its stdout as a receipt. A non-pass result stops
the lane.

If the target cannot maintain the required state read-only, the next boundary
is a separately reviewed frequency preparation/restoration mechanism with
snapshot, exact write scope, verification, bounded lifetime, and restoration.
That mechanism must remain outside the Gate A execution bundle.

## Current boundary

```text
source review complete = false
read-only target contact authorized = false
frequency preparation writes authorized = false
third Gate A attempt authorized = false
Gate B authorized = false
```

Next boundary:

```text
INDEPENDENT_EXACT_HEAD_REVIEW_FOR_GATE_A_FREQUENCY_PRECONDITION_PROBE
```
