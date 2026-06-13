# EXP50 PHASE 5.1-5.5 - TARGET OBSERVABILITY

**Target:** `root@192.168.137.100`

**Date:** 2026-06-12

## Read-Only Probe Result

The Phenom target was reachable over SSH. Read-only observability channels found:

- OS: `Linux catcas 6.12.86+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.86-1 (2026-05-08) x86_64 GNU/Linux`
- CPU: `AMD Phenom(tm) II X6 1090T Processor`
- Cores: `6`
- `rdmsr`: `/usr/sbin/rdmsr`
- MSR device: `/dev/cpu/0/msr`
- k10 temperature: `/sys/class/hwmon/hwmon0/temp1_input`
- current k10 temperature sample: `43625` millideg C
- cpufreq current samples:
  - cpu0 `1607460`
  - cpu1 `1607458`
  - cpu2 `1600000`
  - cpu3 `1600000`
  - cpu4 `1600000`
  - cpu5 `1600000`
- all-core `MSRC001_0071` readback:
  - cpu0 `180000140032c00`
  - cpu1 `180000140032c00`
  - cpu2 `180000140032c00`
  - cpu3 `180000140032c00`
  - cpu4 `180000140032c00`
  - cpu5 `180000140032c00`

## Missing Physical Channels

No usable package-energy or power counter was found in `/sys/class/powercap` or
the hwmon scan. The target can provide temperature, cpufreq, and MSR readback,
but Phase 5.1 and 5.2 still require an external or calibrated energy artifact.

## Consequence For 5.1-5.5

- 5.1 can run the logical zero-erasure/timing probe and record temperature, but
  cannot close a physical energy claim without a joule trace.
- 5.2 can account cyclic throughput, but cannot close a physical capacity claim
  without energy/time/geometry inputs.
- 5.3 can be strengthened on-target with pinned timing windows.
- 5.4 and 5.5 still need live oscillator/jitter phase traces and null controls.

## Foundation Probe Run On Target

The foundation probe was run on the target from a temporary copy after the
observability check.

- verdict: `PHASE5_1_5_SOFTWARE_FOUNDATION_COMPLETE__PHYSICAL_INSTRUMENTATION_REQUIRED`
- restore rate: `1.000000`
- median forward time: `610196.0 ns`
- median reverse time: `608799.5 ns`
- median reverse/forward ratio: `0.996875`
- rank-1 model: `PASS`
- noise-only transient candidate: `PASS`
- compact summary: `results/phase5_1_5_target_summary.json`

## Working Read-Only Commands

```bash
uname -a
lscpu
command -v rdmsr
ls -la /dev/cpu/0/msr
cat /sys/class/hwmon/hwmon0/name
cat /sys/class/hwmon/hwmon0/temp1_input
for c in 0 1 2 3 4 5; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; done
for c in 0 1 2 3 4 5; do rdmsr -p "$c" 0xC0010071; done
```

## Do Not Do

- Do not write MSRs.
- Do not change cpufreq settings.
- Do not write voltage.
- Do not flash firmware.
- Do not classify decoded VID as physical Vcore.
