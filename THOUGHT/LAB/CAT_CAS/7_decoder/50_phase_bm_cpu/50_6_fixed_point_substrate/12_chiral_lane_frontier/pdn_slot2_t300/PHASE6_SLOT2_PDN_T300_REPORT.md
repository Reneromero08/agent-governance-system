# Phase 6 Slot2 PDN T300 Report

Status: `PDN_CARRIER_LIVE_CONFIRMED_ON_PHENOM_ROUTE_4_5`

Claim level: `L1_DETECTOR_LIVE`

Verdict:

```text
PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS
```

## Executive Summary

The Phenom II T300 run confirms that the cross-core PDN/timing carrier can transport a bound phase lane on real silicon when the sender owns the drive phase. This is detector/transport evidence, not a public lane-generation result and not a wall crossing.

The result is route-sensitive. Core route `4:5` passes all six seeds. Core route `2:3` preserves real-mode accuracy and phase recovery but misses the strict real-vs-pseudo floor on four seeds.

## Imported Artifacts

Local result directory:

```text
chiral_lane_frontier/pdn_slot2_t300/results/
```

Tracked compact artifacts:

```text
result_slot2_pdn_t300.json
aggregate_t300.json
aggregate_t300.txt
per_run/matrix_v2s3_seed0.json
per_run/matrix_v2s3_seed1.json
per_run/matrix_v2s3_seed2.json
per_run/matrix_v2s3_seed3.json
per_run/matrix_v2s3_seed4.json
per_run/matrix_v2s3_seed5.json
per_run/matrix_v4s5_seed0.json
per_run/matrix_v4s5_seed1.json
per_run/matrix_v4s5_seed2.json
per_run/matrix_v4s5_seed3.json
per_run/matrix_v4s5_seed4.json
per_run/matrix_v4s5_seed5.json
```

Local ignored logs retained for operator audit but not committed:

```text
t300.log
control_silent.log
control_scramble.log
```

Raw matrix CSV captures were not imported into git. The compact JSON summaries preserve the scored evidence; raw captures remain on the Phenom run directory unless separately archived.

## Run Configuration

Source host:

```text
root@192.168.137.100:/root/slot2_pdn
```

Machine:

```text
AMD Phenom II X6 1090T
Debian 13, Linux 6.12.86
isolcpus=2,3,4,5
constant_tsc + nonstop_tsc
userspace-only run
P-state pinned at 1600 MHz during slots
k10temp veto threshold: 68 C
```

Experiment:

```text
EXP44_SLOT2_PDN_DRIVEN_LOCKIN_CROSS_CORE_HOLO_TRAVERSAL
```

Method:

```text
sender/receiver userspace processes pinned to isolated cores
shared absolute TSC origin
register/L1-only drive
victim ring-osc timing lock-in
12 non-harmonic tones in [20,1500] Hz
0.5 s slots
4000 Hz readout
300 trials
two core routes: 2:3 and 4:5
six seeds per route
silent and scramble controls
```

## Aggregate Result

Final aggregate:

```text
PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS
pair 2:3: 2/6 pass  real_acc=1.000-1.000  rvp=0.910-0.962  phase_d=0.980-1.032  fail_seeds=[1, 3, 4, 5]
pair 4:5: 6/6 pass  real_acc=0.953-1.000  rvp=0.954-0.985  phase_d=0.978-1.033  fail_seeds=[]
control silent: real_acc=0.193 rvp=0.463 wr_act=0.213 phase_d=-0.068 witness_gates_pass=False -> control_ok(null)=True
control scramble: real_acc=0.253 rvp=0.412 wr_act=0.253 phase_d=-0.063 witness_gates_pass=False -> control_ok(null)=True
PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS
```

## Gate Readout

Required reproducibility gate:

```text
6/6 seeds:
all_rows_restore
real_accuracy >= 0.60
real_vs_pseudo floor >= 0.95
pseudo_reject >= 0.95
wrong_actual_match >= 0.60
wrong_declared_match <= 0.20
phase_corr_true - phase_corr_null > 0.30
```

Route `4:5` passes this gate for every seed.

Route `2:3` fails as a full reproducibility route because the real-vs-pseudo floor drops below `0.95` on seeds `1`, `3`, `4`, and `5`. It is not a null route: real accuracy remains `1.000-1.000`, restoration is true, and phase delta remains `0.980-1.032`.

Both controls behave as nulls. Neither silent nor scramble passes witness gates.

## Interpretation

This result strengthens the Phase 6 fixed fact:

```text
PDN_CARRIER_LIVE
```

More precisely:

```text
PDN_CARRIER_LIVE_ON_PHENOM_ROUTE_4_5
TOPOLOGY_ROUTE_SENSITIVITY_CONFIRMED
CONTROLS_NULL
```

It does not establish:

```text
PUBLIC_CHIRAL_LANE_GENERATED
DIHEDRAL_ORIENTATION_RECOVERED
PUBLIC_ROUTE_CROSSING
LATTICE_WALL_BROKEN
```

The sender owns the drive phase in this run. Therefore the result calibrates and validates a physical carrier/detector route. It does not show public generation of the missing fold-odd lane.

## Roadmap Impact

Track I is promoted from optional topology preparation to required operating procedure for all public lane-generation attempts.

The immediate experimental implication is:

```text
Use route 4:5 as the current Phenom adjudication carrier when possible.
Treat route 2:3 as route-sensitive/partial, not as a failed detector.
Require topology chirality mapping before Track A/B/C/D/F/G/H/K route selection.
```

Track 0 remains required because this run proves a strong hidden/bound carrier but does not measure the minimum detectable fold-odd amplitude.

## Next Action

Run the Phase 6 detector spine in this order:

```text
Track Z: Orientation conservation audit
Track I: topology chirality map using the T300 result as seed evidence
Track 0: odd-lane transfer function
Track B: I/Q receiver base layer
Track A: dual-lane even cancellation on route 4:5 first
```

The public frontier remains:

```text
Can public execution geometry generate a fold-odd physical carrier when the published data itself is fold-even?
```
