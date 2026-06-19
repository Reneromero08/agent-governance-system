# Phase 6B.5 Carrier-Witness Closure Execution Record

**Date:** 2026-06-18
**Branch:** `phase6b/carrier-witness-closure`
**Starting commit:** `a34bb33770c392f9bdce6e4e9a67aa579c351a31`
**Host:** `catcas` (`root@192.168.137.100`)
**Status:** `PARTIAL`

## Repository and software result

The existing Slot 2 receiver now supports immutable raw witness capture below
the lock-in reduction boundary. The sender and receiver share one effective
drive calculation for matrix, silent, and scramble conditions. The receiver
serializes the exact schedule and runtime `t0`, records per-window thermal,
frequency, and P-state telemetry, retains absolute TSC/ring-period samples, and
propagates raw writer failures. Run finalization refuses incomplete or existing
outputs and verifies its generated manifest.

The source-only payload was compiled on the target with GCC 14.2.0. Both the
full test target and the actual `-march=amdfam10` Slot 2 build passed with
`-Wall -Wextra -Werror`.

The combined ASan/LSan/UBSan and UBSan-only integration matrices passed with
no sanitizer findings or leaks.

## Historical evidence audit

Generated artifact:

```text
t300_existing_evidence_audit.json
SHA-256: 0e74419b8bb86a7dc81041d0f5daf4f0bde867731074c921f3eec4a27dc55dba
```

The audit found 12 historical matrix CSVs, 12 matrix logs, two control CSVs,
121 files under `/root/slot2_pdn`, and no complete raw candidate directory.
Neither `raw_samples.bin` nor `windows.csv` survived. A read-only search of
`/root`, `/var/tmp`, and `/tmp` found historical Slot 2 trees and compact CSV
evidence, but no retained `t_tsc[]`/`ro_period[]` arrays.

## Instrumentation diagnosis

The target is an AMD Phenom II X6 1090T running Linux
`6.12.86+deb13-amd64`. Kernel flags include `constant_tsc` and `nonstop_tsc`;
the boot configuration isolates cores `2,3,4,5`. The required thermal source
is readable at `/sys/class/hwmon/hwmon0/temp1_input` and identifies as `k10temp`.
The initial shell inventory used `find -type f`, which did not follow the hwmon
class symlink; the generated audit and the acquisition code both resolved it.

The first pre-acquisition invocation stopped on the separate P-state check and
preserved `/root/cw_thermal_gate_a34bb337` with its logs. It captured no raw
windows. The implementation was then corrected to verify acpi-cpufreq policy
readback on every core while retaining current frequency as a window telemetry
proxy. The failed run remains preserved and was not reused.

## Smoke result

Campaign `phase6b5_smoke_d32b1bed` completed one route `4:5`, seed `777`,
four-trial run. It is structurally valid with 192 windows and 291,564 raw
records. Maximum reconstruction error is `1.1518563880486e-15` absolute and
`5.1000782556901936e-11` relative. Its expected campaign verdict is `PARTIAL`.

## Frozen T48 campaign

Campaign `phase6b5_t48_d32b1bed_20260619` ran 14 immutable bundles: route
`4:5` seeds 0-5, silent seed 900, scramble seed 901, and route `2:3` seeds 0-5.
All process exits were zero; every manifest verifies; all raw windows,
summaries, and analyses reconstruct.

```text
source commit: d32b1bed0deae1b907a07eeed018b924244e9ea2
binary SHA-256: 75024a02982e853336558468e550e824e6ad17a01e1ddcfad85610e138a81f59
raw records: 37,757,575
windows: 24,864
raw bytes: 604,121,200
maximum absolute reconstruction error: 1.7486012637846216e-15
maximum relative reconstruction error: 1.0740479543420315e-10
temperature range: 40.375 C to 44.0 C
route 4:5: 1/6 scientific passes
route 2:3: 2/6 scientific passes
silent control: null
scramble control: null
closure status: PARTIAL
```

Corrected seven-gate closure report SHA-256:
`51c44f451273496bc9915a3ebdbf64199ff205beffbf040b98eb705d1ceae2b2`.
Campaign manifest SHA-256:
`cbcd2a19d6dd3bc478244f77888aa87eb043003a7685caa17ff13fe4d47e6487`.

## Exact boundary

```text
software integration: complete and target-built
historical raw recovery: absent
thermal instrumentation: readable k10temp
smoke campaign: structurally valid, PARTIAL as expected
full T48 campaign: structurally valid, scientific PARTIAL
closure status: PARTIAL
```

The existing compact T300 route `4:5` result remains channel-summary evidence.
It is not independently reconstructable from retained physical timing samples.
