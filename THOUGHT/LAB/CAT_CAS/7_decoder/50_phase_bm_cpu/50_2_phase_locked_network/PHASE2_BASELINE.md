# PHASE2_BASELINE

## Verdict

PHASE2_BASELINE_READY

## Evidence

```text
hostname: catcas
kernel: Linux catcas 6.12.86+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.86-1 (2026-05-08) x86_64 GNU/Linux
cmdline: isolcpus=2,3,4,5 nohz_full=2,3,4,5 rcu_nocbs=2,3,4,5 processor.max_cstate=1 idle=poll amd_pstate=disable
isolated cores: 2-5
baseline k10temp: +47.5 C
rdmsr: /usr/sbin/rdmsr
wrmsr: /usr/sbin/wrmsr
setpci: /usr/bin/setpci
gcc: /usr/bin/gcc
python3: /usr/bin/python3
```

P-state and COFVID snapshot:

```text
core 2 PCTL 3 P4 8000013540003440 COFVID 180000140042440
core 3 PCTL 3 P4 8000013540003440 COFVID 180000140042440
core 4 PCTL 0 P4 8000013540003440 COFVID 180000140012410
core 5 PCTL 3 P4 8000013540003440 COFVID 180000140042440
NB PCI: a01a0800 03001315 0067641a 0207df19 80000000
```

Scripts used:

- `50_2_phase_locked_network/src/phase2_probe.c`
- `50_2_phase_locked_network/src/phase2_analyze_fast.py`
- Existing oscillator and TSC scripts in `50_2_phase_locked_network/src/`.

No firmware flash, physical modification, unknown PCI write, or voltage sweep was performed.

