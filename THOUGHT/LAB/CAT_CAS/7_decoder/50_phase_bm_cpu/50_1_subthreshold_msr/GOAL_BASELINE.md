# GOAL_BASELINE

## Verdict

BASELINE_READY

## Read-Only Snapshot

Command:

```sh
ssh root@192.168.137.100 'hostname; uname -a; cat /proc/cmdline; cat /sys/devices/system/cpu/isolated; sensors; command -v rdmsr; command -v wrmsr; command -v setpci; command -v gcc; command -v python3; rdmsr/setpci snapshot'
```

Raw evidence:

```text
== identity ==
catcas
Linux catcas 6.12.86+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.86-1 (2026-05-08) x86_64 GNU/Linux
BOOT_IMAGE=/boot/vmlinuz-6.12.86+deb13-amd64 root=UUID=e97cf328-66c7-423e-882e-1dbfe4291252 ro quiet isolcpus=2,3,4,5 nohz_full=2,3,4,5 rcu_nocbs=2,3,4,5 processor.max_cstate=1 idle=poll amd_pstate=disable
== isolated cores ==
2-5
== temp ==
k10temp-pci-00c3
Adapter: PCI adapter
temp1:        +37.1°C  (high = +70.0°C)
                       (crit = +80.0°C, hyst = +75.0°C)
== tools ==
/usr/sbin/rdmsr
/usr/sbin/wrmsr
/usr/bin/setpci
/usr/bin/gcc
/usr/bin/python3
== pstate core 4 ==
0xc0010062 4
0xc0010068 8000013540003440
0xc0010070 40043440
0xc0010071 180000140042c40
== all p4/cofvid ==
core 0 P4 8000013540003440
core 0 COFVID 180000140032c00
core 1 P4 8000013540003440
core 1 COFVID 180000140042c40
core 2 P4 8000013540003440
core 2 COFVID 180000140042c40
core 3 P4 8000013540003440
core 3 COFVID 180000140042c40
core 4 P4 8000013540003440
core 4 COFVID 180000140042c40
core 5 P4 8000013540003440
core 5 COFVID 180000140043440
== nb pci focused ==
a01a0800
03001315
0067641a
0207df19
80000000
```

## Decode

- Core 4 P4 `0x8000013540003440`: FID `0x00`, DID `1`, CpuVid `0x1A`, NbVid `0x20`.
- Core 4 COFVID_STS `0x180000140042c40`: FID `0x00`, DID `1`, CpuVid `0x16`, NbVid `0x20`.
- k10temp was below the 60 C limit.
- P4 was restored to stock before route work.

## Scripts Present

The per-phase `src/` dirs (e.g. `50_1_subthreshold_msr/src/`, `50_2_phase_locked_network/src/`) contain the active phase, oscillator, Kuramoto, P-state, VID, and catalytic tape harnesses. `50_2_firmware/cpu_hack/` contains BIOS dump/report/disassembly material.

