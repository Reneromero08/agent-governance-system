# PHASE2_DEEP_3_EXTERNAL_MEASURE

## Verdict

EXTERNAL_OBSERVABILITY_PLAN_READY

## Goal

Measure whether Core3/Core4/Core5 workload state produces a waveform correlated with workload markers, while separating the fixed 2.67 MHz VRM artifact from true workload-correlated structure.

## Non-Invasive Measurement Points

Preferred safe points:

1. CPU VRM output capacitor top pad, measured with a high-impedance probe and short ground spring.
2. CPU 12 V input rail at an accessible decoupling capacitor.
3. Motherboard ground reference at a nearby mounting or shield point.
4. Optional fan tach or audio input only as a timing reference, not as a CPU signal.

Do not lift pins, cut traces, solder wires, or force a probe under IC packages.

## Instrument Requirements

- Oscilloscope or logic analyzer with at least 20 MS/s for marker timing.
- For the 2.67 MHz VRM component, use at least 25 MS/s; 100 MS/s or higher is preferred.
- Capture windows of 5-20 seconds.
- Save raw waveform, not only screenshots.

## Marker Alignment

Use `phase2_marker_harness.c`:

```sh
gcc -O2 -pthread phase2_marker_harness.c -o phase2_marker_harness
./phase2_marker_harness 256 50000 > phase2_marker_log.csv
```

The CSV gives segment index, TSC, state word, edge counter, and Core3/Core4/Core5 counters. Use state transitions as expected waveform boundaries.

## Artifact Separation

Analysis must compare:

- Raw waveform spectrum.
- Marker-aligned average waveform per state.
- Shuffled-state null.
- Idle marker run.
- Core3-only, Core4-only, Core5-only marker runs.

Accept only if:

- A component changes with marker state and repeats across runs.
- The same component is not present in shuffled-state nulls.
- The component is not simply the fixed 2.67 MHz line with constant amplitude.

## Output Needed

- Instrument model and sample rate.
- Probe point photo.
- Raw waveform file.
- Marker CSV.
- Alignment script output.
- Temperature and P4 readback after run.

