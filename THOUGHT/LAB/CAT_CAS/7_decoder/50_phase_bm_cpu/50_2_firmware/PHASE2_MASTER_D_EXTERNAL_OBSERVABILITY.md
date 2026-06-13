# PHASE2_MASTER_D_EXTERNAL_OBSERVABILITY

## Verdict

`EXTERNAL_MEASUREMENT_READY`

Route D is the best next non-destructive path. The software-only routes are exhausted, firmware is not byte-ready, and the lab already has marker harnesses suitable for non-invasive correlation.

## Existing Inputs

- `50_2_phase_locked_network/PHASE2_DEEP_3_EXTERNAL_MEASURE.md`
- `50_2_phase_locked_network/PHASE2_DEEP_4_MARKERS.md`
- `50_2_phase_locked_network/src/phase2_marker_harness.c`
- `50_2_phase_locked_network/src/phase2_probe.c`
- `50_2_phase_locked_network/src/phase2_external_align.py`

## Measurement Objective

Prove or reject marker-correlated physical behavior by correlating Core3/Core4/Core5 workload state with external waveform capture.

The fixed 2.67 MHz component remains rejected unless its amplitude, phase, or sideband structure is marker-modulated and survives null tests.

## Non-Invasive Setup

Allowed measurement points:

- CPU VRM output capacitor top pad with high-impedance probe and short ground spring.
- CPU 12 V input rail at accessible decoupling capacitor.
- Motherboard ground reference at mounting/shield point.
- Optional Pi GPIO or logic analyzer channel as a timing marker only if connected externally and safely.

Do not solder, lift pins, cut traces, force probes under ICs, or write unknown PCI/MSR/voltage values.

## Marker Harness Command

On the Phenom host, from the lab script directory:

```sh
gcc -O2 -pthread phase2_marker_harness.c -o phase2_marker_harness
./phase2_marker_harness 256 50000 > phase2_marker_log.csv
```

Expected CSV header:

```text
segment,tsc,state,edge,c3,c4,c5
```

## Capture Plan

1. Start scope or logic-analyzer capture first.
2. Start the marker harness.
3. Capture at least 5-20 seconds.
4. Sample at 25 MS/s minimum for the 2.67 MHz line; 100 MS/s preferred.
5. Save raw waveform, not only screenshots.
6. Save marker CSV from the same run.
7. Record instrument model, sample rate, probe point, and temperature.

## Offline Alignment Command

After a waveform CSV exists, run the offline analyzer:

```sh
python3 phase2_external_align.py \
  --marker phase2_marker_log.csv \
  --wave scope_waveform.csv \
  --segment-us 50000 \
  --out-csv phase2_external_summary.csv \
  --out-report phase2_external_alignment_report.md
```

If the waveform CSV has sample index/value but no time column, provide the sample rate:

```sh
python3 phase2_external_align.py \
  --marker phase2_marker_log.csv \
  --wave scope_waveform.csv \
  --sample-rate 100000000 \
  --value-column ch1 \
  --segment-us 50000 \
  --out-csv phase2_external_summary.csv \
  --out-report phase2_external_alignment_report.md
```

The analyzer emits per-state waveform means, a deterministic shuffled null, and a Goertzel amplitude check at 2.67 MHz.

## Nulls Required

| Null | Purpose |
|---|---|
| Idle marker run | Reject waveform unrelated to workload |
| Shuffled-state analysis | Reject post-hoc alignment artifacts |
| Core3-only | Separate PPU-A contribution |
| Core4-only | Separate PPU-B contribution |
| Core5-only/reference | Separate master/reference contribution |
| No-marker idle | Identify fixed board/VRM background |
| Analyzer shuffled null | Reject state labels that only work because of post-hoc ordering |

## Acceptance Gate

Accept external phase evidence only if:

- marker-aligned waveform changes by state,
- the same change repeats across runs,
- shuffled/null runs do not reproduce it,
- the fixed 2.67 MHz component is not merely constant amplitude infrastructure noise,
- the evidence correlates with Core3/Core4/Core5 marker states rather than wall-clock time alone.
- `phase2_external_align.py` output shows aligned state separation stronger than its shuffled null.

## Human Approval Boundary

Human approval is required before physical probing because the next action involves connecting external measurement equipment to the board.

This is not a firmware or voltage approval. It is limited to non-invasive measurement setup.

## Route D Outcome

`EXTERNAL_MEASUREMENT_READY`

Exact next human action:

Set up the scope/logic analyzer and run the marker capture above, then place the raw waveform and marker CSV in the lab and run `50_2_phase_locked_network/src/phase2_external_align.py`.
