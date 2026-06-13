# EXP50 PHASE 5.1-5.5 - LIVE RUNBOOK

**Purpose:** collect the live artifacts needed to extend the Phenom-run
foundation into physical 5.1-5.5 evidence.

**Current foundation artifact:** `PHASE5_1_5_FOUNDATION_REPORT.md`

**Current target observability:** `PHASE5_1_5_TARGET_OBSERVABILITY.md`

**Primary evidence run:** `results/phase5_1_5_target_summary.json`

**Host run status:** `HOST_SMOKE_TEST_ONLY`; do not use host timing values as
Exp50 evidence.

## Read-Only Target Checks

Run on the Phenom target only when the lab host link is stable:

```bash
uname -a
lscpu
ls /sys/class/hwmon
find /sys/class/hwmon -maxdepth 2 -type f -name 'temp*_input' -o -name 'power*_input' -o -name 'energy*_input'
ls /dev/cpu/0/msr
command -v rdmsr
command -v perf
command -v taskset
```

These checks only discover observability channels. They do not change voltage,
firmware, clocks, or board state.

The 2026-06-12 target check found `rdmsr`, `/dev/cpu/0/msr`,
`k10temp/temp1_input`, and cpufreq readback. It did not find a package energy
counter or powercap channel.

## 5.1 Energy Trace

Needed artifact:

- wall/EPS12V joule trace aligned to the probe start/end timestamp, or
- calibrated package-energy counter with documented AMD K10 validity, plus
- temperature trace over the same window.

Acceptance input:

```text
start_timestamp_ns,end_timestamp_ns,joules,temp_c,calibration_note
```

Without this artifact, Phase 5.1 stays at:
`PHASE5_1_ZERO_LOGICAL_ERASURE_CONFIRMED__ENERGY_TRACE_REQUIRED`.

## 5.2 Physical Capacity Trace

Needed artifact:

- the same energy/time trace from 5.1,
- die/package geometry assumptions used for the bound,
- explicit mapping from reversible byte touches to physical information
  throughput.

Acceptance input:

```text
geometry_source,observation_window_s,reversible_touches,bound_model,ratio
```

Without this artifact, Phase 5.2 stays at:
`PHASE5_2_CYCLIC_THROUGHPUT_ACCOUNTED__PHYSICAL_BOUND_TRACE_REQUIRED`.

## 5.3 Timing Asymmetry

Already run on the Phenom target. Current primary artifact:
`results/phase5_1_5_target_summary.json`.

Reusable harness command from the lab folder:

```bash
python 50_5_1_limit_violations/src/phase5_1_5_foundation_probe.py
```

Live strengthening artifact:

- pinned core run,
- cache/PMU event counters if available,
- repeated forward/reverse windows under quiet and loaded conditions.

## 5.4 Rank-1 Physical Control

Needed live data:

- six phase/timing channels,
- one declared master/reference channel,
- coupling-on and coupling-off runs,
- shuffled-reference null.

Acceptance input:

```text
run_id,channel,window,phase_or_timing_value,coupling_mode,reference_id
```

Without this artifact, Phase 5.4 stays at:
`PHASE5_4_RANK1_CONTROL_MODEL_PASS__LIVE_OSCILLATOR_TRACE_REQUIRED`.

## 5.5 Noise-Only Physical Candidate

Needed live data:

- no deliberate phase programming,
- no coupling lever applied,
- fixed capture windows,
- shuffled-window null,
- same detector threshold declared before analysis.

Acceptance input:

```text
run_id,window,channel,phase_or_jitter_value,detector_threshold,null_label
```

Without this artifact, Phase 5.5 stays at:
`PHASE5_5_NOISE_ONLY_TRANSIENT_LOCK_MODEL_CANDIDATE__LIVE_NOISE_TRACE_REQUIRED`.

## Do Not Do

- Do not flash BIOS.
- Do not write voltage or clocks.
- Do not use decoded VID as physical voltage without readback.
- Do not call software model results physical-limit violations.
- Do not treat restored hash alone as a catalytic primitive.
