# EXP50 PHASE 5.1-5.5 - FOUNDATION REPORT

**Status:** `PHASE5_1_5_SOFTWARE_FOUNDATION_COMPLETE__PHYSICAL_INSTRUMENTATION_REQUIRED`

## Scope

This pass finishes the runnable Phenom-side foundation for Phase 5.1 through
5.5 and records the exact physical artifacts still needed for stronger claims.
The evidence run is the Phenom II target run at `root@192.168.137.100`; the
workstation run is retained only as a harness smoke test. The probe does not
write voltage, touch firmware, flash BIOS, or modify hardware state.

## Results

| Phase | Label | Evidence |
|---|---|---|
| 5.1 Landauer gate | `PHASE5_1_ZERO_LOGICAL_ERASURE_CONFIRMED__ENERGY_TRACE_REQUIRED` | 96/96 reversible cycles restored; max logical bits erased = 0; irreversible control bit-erasure median = 4105.0; nonzero rate = 0.989583. |
| 5.2 Bekenstein gate | `PHASE5_2_CYCLIC_THROUGHPUT_ACCOUNTED__PHYSICAL_BOUND_TRACE_REQUIRED` | 147415 reversible byte touches across cycles; model phase capacity = 48.0 bits; throughput/model-capacity ratio = 24569.167. |
| 5.3 Arrow gate | `PHASE5_3_FORWARD_REVERSE_TIMING_ASYMMETRY_MEASURED` | Phenom median forward = 610196.0 ns; median reverse = 608799.5 ns; median reverse/forward = 0.996875. |
| 5.4 Schmidt/rank-1 gate | `PHASE5_4_RANK1_CONTROL_MODEL_PASS__LIVE_OSCILLATOR_TRACE_REQUIRED` | master-correlation floor = 0.999986; residual-ratio ceiling = 0.005377; null residual-ratio floor = 0.183985. |
| 5.5 Noise-only gate | `PHASE5_5_NOISE_ONLY_TRANSIENT_LOCK_MODEL_CANDIDATE__LIVE_NOISE_TRACE_REQUIRED` | spontaneous lock windows = 12/512; best order = 0.996119; threshold = 0.96. |

## Proxy Hardening Push

A follow-up Phenom target run pushed the remaining software-observable angle
for 5.3-5.5:

- `PHASE5_3_PINNED_TIMING_HARDENED_PROXY`
- `PHASE5_4_REFERENCE_TO_MULTICHANNEL_PROXY_MEASURED`
- `PHASE5_5_NOISE_JITTER_SHUFFLE_NULL_MEASURED`
- `RESTORATION_INTACT`
- `RANK1_PROXY_PARTIAL`
- `NOISE_TEMPORAL_STRUCTURE_NOT_SEPARATED_FROM_SHUFFLE`

Summary:

- 5.3 pinned all-core median reverse/forward ratio: `0.997884`
- 5.4 reference-to-channel abs correlation floor: `0.720324`
- 5.4 rank-1 explained energy: `0.556533`
- 5.5 real noise-order median: `0.389339`
- 5.5 shuffled noise-order median: `0.389339`
- 5.5 median delta: `0.000000`

Proxy hardening artifact:
`50_5_1_limit_violations/PHASE5_1_5_PROXY_HARDENING.md`

## Physical Artifacts Still Needed

5.1 needs an aligned joule trace or calibrated package-energy counter plus
temperature. 5.2 needs the same physical trace plus explicit die/package
geometry assumptions and an accepted tape-throughput to physical-capacity map.
5.4 and 5.5 need live oscillator phase traces from six channels with
coupling-on/off and shuffled-window controls.

## Phenom Target Evidence Run

The foundation probe was executed on `root@192.168.137.100` using a temporary
target copy. This is the evidentiary run for Phase 5.1-5.5.

- restore rate: `1.000000`
- logical bits erased, reversible max: `0`
- median forward time: `610196.0 ns`
- median reverse time: `608799.5 ns`
- median reverse/forward ratio: `0.996875`
- rank-1 model: `PASS`
- noise-only transient candidate: `PASS`
- target artifacts available to the probe: `/sys/class/hwmon`, `/dev/cpu/0/msr`
- package-energy counter: absent

Compact target summary:
`50_5_1_limit_violations/results/phase5_1_5_target_summary.json`

The Phenom target exposed `/sys/class/hwmon` and `/dev/cpu/0/msr`, but did not
expose a package-energy artifact:
`rapl_energy_uj_present=False`, `msr_device_present=True`.

## Host Smoke Test

The initial workstation-side run is retained only as `HOST_SMOKE_TEST_ONLY`.
It verified that the harness generated internally consistent reports before the
target run, but it is not used as proof for Exp50.

Host smoke artifact:
`50_5_1_limit_violations/results/phase5_1_5_summary.json`

## Artifacts

- `50_5_1_limit_violations/results/phase5_1_5_forward_reverse_cycles.csv`
- `50_5_1_limit_violations/results/phase5_1_5_target_summary.json` primary evidence
- `50_5_1_limit_violations/results/phase5_1_5_summary.json` host smoke test only
- `50_5_1_limit_violations/results/proxy_hardening/phase5_1_5_proxy_hardening_summary.json`
- `50_5_1_limit_violations/src/phase5_1_5_foundation_probe.py`
- `50_5_1_limit_violations/src/phase5_1_5_proxy_hardening.py`

## Claim Boundary

Accepted now: reversible logical zero-erasure accounting, cyclic throughput
accounting, forward/reverse timing asymmetry measurement, rank-1 control model,
noise-only transient-lock candidate model, and proxy hardening inside current
software observability.

Not accepted from this pass alone: physical Landauer violation, physical
Bekenstein violation, physical oscillator control, physical noise computation,
or thermodynamic claim. The proxy hardening specifically caps 5.5: current
software-visible noise ordering does not separate from the shuffled-window null.
