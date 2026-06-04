# RUNTIME_VID_DECIDER_PACK

## A. EXECUTIVE VERDICT

RUNTIME_VID_CLAMPED

Meaning: Core 4 accepted the runtime P4 definition write to `0x80000135400040c0`, but after P-state cycling COFVID_STS reported CpuVid `0x1A`, not the requested CpuVid `0x20`.

Runtime undervolt path alive: no, not through this runtime K10 P-state MSR path.

Clamp confirmed: yes.

## B. PRECHECKS

- SSH reachable: yes.
- Hostname: `catcas`.
- Kernel: `Linux catcas 6.12.86+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.86-1 (2026-05-08) x86_64 GNU/Linux`.
- `/proc/cmdline`: `BOOT_IMAGE=/boot/vmlinuz-6.12.86+deb13-amd64 root=UUID=e97cf328-66c7-423e-882e-1dbfe4291252 ro quiet isolcpus=2,3,4,5 nohz_full=2,3,4,5 rcu_nocbs=2,3,4,5 processor.max_cstate=1 idle=poll amd_pstate=disable`.
- Isolated cores: `2-5`.
- k10temp baseline: `+36.0 C`.
- `rdmsr` availability: yes, `/usr/sbin/rdmsr`.
- `wrmsr` availability: yes, `/usr/sbin/wrmsr`.
- Current P4 value on Core 4 before write: `8000013540003440`.
- Current COFVID_STS on Core 4 before write: `1800001400434c0`.
- PSTATE_CTL value on Core 4 before write: `4`.

## C. READ-ONLY SNAPSHOT

Connection/package remediation command:

```sh
ssh root@192.168.137.100 '
set -eux
hostname
ip route || true
cat /etc/resolv.conf || true
apt update
apt install -y msr-tools
command -v rdmsr
command -v wrmsr
'
```

Raw output:

```text
catcas
default via 192.168.137.1 dev enp3s0 onlink
192.168.137.0/24 dev enp3s0 proto kernel scope link src 192.168.137.100
nameserver 8.8.8.8
nameserver 1.1.1.1
Get:1 http://security.debian.org/debian-security trixie-security InRelease [43.4 kB]
Hit:2 http://deb.debian.org/debian trixie InRelease
Get:3 http://deb.debian.org/debian trixie-updates InRelease [47.3 kB]
Get:4 http://security.debian.org/debian-security trixie-security/main amd64 Packages [207 kB]
Get:5 http://security.debian.org/debian-security trixie-security/main Translation-en [125 kB]
Fetched 423 kB in 0s (1,530 kB/s)
Reading package lists...
Building dependency tree...
Reading state information...
10 packages can be upgraded. Run 'apt list --upgradable' to see them.
Reading package lists...
Building dependency tree...
Reading state information...
Installing:
  msr-tools

Summary:
  Upgrading: 0, Installing: 1, Removing: 0, Not Upgrading: 10
  Download size: 9,700 B
  Space needed: 45.1 kB / 110 GB available

Get:1 http://deb.debian.org/debian trixie/main amd64 msr-tools amd64 1.3+git20220805.7d78c80-1 [9,700 B]
Fetched 9,700 B in 0s (68.3 kB/s)
Selecting previously unselected package msr-tools.
Preparing to unpack .../msr-tools_1.3+git20220805.7d78c80-1_amd64.deb ...
Unpacking msr-tools (1.3+git20220805.7d78c80-1) ...
Setting up msr-tools (1.3+git20220805.7d78c80-1) ...
/usr/sbin/rdmsr
/usr/sbin/wrmsr
```

Read-only snapshot command:

```sh
ssh root@192.168.137.100 '
set -eu
echo "== identity =="
hostname
uname -a
cat /proc/cmdline

echo "== isolated cores =="
cat /sys/devices/system/cpu/isolated || true

echo "== temperature =="
sensors | sed -n "/k10temp/,+5p" || true

echo "== core 4 MSR snapshot =="
CORE=4
for m in 0xc0010061 0xc0010062 0xc0010064 0xc0010065 0xc0010066 0xc0010067 0xc0010068 0xc0010070 0xc0010071; do
  printf "%s " "$m"
  rdmsr -p "$CORE" "$m" || true
done

echo "== all-core P4 and COFVID_STS =="
for c in 0 1 2 3 4 5; do
  echo "-- core $c --"
  printf "P4 "; rdmsr -p "$c" 0xc0010068 || true
  printf "COFVID_STS "; rdmsr -p "$c" 0xc0010071 || true
done

echo "== NB PCI readback =="
setpci -s 00:18.3 a0.l d8.l dc.l e8.l 1fc.l || true
'
```

Raw output:

```text
== identity ==
catcas
Linux catcas 6.12.86+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.86-1 (2026-05-08) x86_64 GNU/Linux
BOOT_IMAGE=/boot/vmlinuz-6.12.86+deb13-amd64 root=UUID=e97cf328-66c7-423e-882e-1dbfe4291252 ro quiet isolcpus=2,3,4,5 nohz_full=2,3,4,5 rcu_nocbs=2,3,4,5 processor.max_cstate=1 idle=poll amd_pstate=disable
== isolated cores ==
2-5
== temperature ==
k10temp-pci-00c3
Adapter: PCI adapter
temp1:        +36.0°C  (high = +70.0°C)
                       (crit = +80.0°C, hyst = +75.0°C)

nouveau-pci-0100
== core 4 MSR snapshot ==
0xc0010061 30
0xc0010062 4
0xc0010064 8000019e40000c14
0xc0010065 8000019f40002410
0xc0010066 8000017540002808
0xc0010067 8000015440002c00
0xc0010068 8000013540003440
0xc0010070 40043440
0xc0010071 1800001400434c0
== all-core P4 and COFVID_STS ==
-- core 0 --
P4 8000013540003440
COFVID_STS 180000140042c40
-- core 1 --
P4 8000013540003440
COFVID_STS 180000140032c00
-- core 2 --
P4 8000013540003440
COFVID_STS 180000140042c40
-- core 3 --
P4 8000013540003440
COFVID_STS 180000140042c40
-- core 4 --
P4 8000013540003440
COFVID_STS 1800001400434c0
-- core 5 --
P4 8000013540003440
COFVID_STS 180000140043440
== NB PCI readback ==
a01a0800
03001315
0067641a
0207df19
80000000
```

## D. SAFETY GATE

Safety gate result: passed before write.

- Core 4 P4 equals exactly `8000013540003440`: passed.
- k10temp baseline is available and not above 55 C: passed, `+36.0 C`.
- SSH and `rdmsr`/`wrmsr` are stable: passed.
- Target core is Core 4: passed.
- Lab host identity: passed, hostname `catcas`.
- Unexpected hardware identity: none observed in command output.

## E. HUMAN-APPROVED TEST WRITE

Command run:

```sh
ssh root@192.168.137.100 '
set -eu
CORE=4
P4=0xc0010068
PCTL=0xc0010062
CSTS=0xc0010071
ORIG_EXPECTED=8000013540003440
TEST=0x80000135400040c0

echo "== pre-write temp =="
sensors | sed -n "/k10temp/,+5p" || true

ORIG=$(rdmsr -p "$CORE" "$P4")
echo "Original P4 on core $CORE: $ORIG"

test "$ORIG" = "$ORIG_EXPECTED" || {
  echo "ABORT: unexpected P4 value: $ORIG"
  exit 1
}

echo "Writing test P4 on core $CORE: $TEST"
wrmsr -p "$CORE" "$P4" "$TEST"

echo "P4 readback after write:"
rdmsr -p "$CORE" "$P4"

echo "Cycling P0 -> P4"
wrmsr -p "$CORE" "$PCTL" 0
sleep 0.2
wrmsr -p "$CORE" "$PCTL" 4
sleep 0.5

echo "COFVID_STS after transition:"
rdmsr -p "$CORE" "$CSTS"

echo "== post-test temp =="
sensors | sed -n "/k10temp/,+5p" || true
'
```

Raw output:

```text
== pre-write temp ==
k10temp-pci-00c3
Adapter: PCI adapter
temp1:        +35.8°C  (high = +70.0°C)
                       (crit = +80.0°C, hyst = +75.0°C)

nouveau-pci-0100
Original P4 on core 4: 8000013540003440
Writing test P4 on core 4: 0x80000135400040c0
P4 readback after write:
80000135400040c0
Cycling P0 -> P4
COFVID_STS after transition:
1800001400434c0
== post-test temp ==
k10temp-pci-00c3
Adapter: PCI adapter
temp1:        +36.2°C  (high = +70.0°C)
                       (crit = +80.0°C, hyst = +75.0°C)

nouveau-pci-0100
```

## F. IMMEDIATE ROLLBACK

Command run immediately after the test:

```sh
ssh root@192.168.137.100 '
set -eu
CORE=4
P4=0xc0010068
PCTL=0xc0010062
CSTS=0xc0010071
ROLLBACK=0x8000013540003440

echo "Rolling back P4 on core $CORE to $ROLLBACK"
wrmsr -p "$CORE" "$P4" "$ROLLBACK"

wrmsr -p "$CORE" "$PCTL" 0
sleep 0.2
wrmsr -p "$CORE" "$PCTL" 4
sleep 0.5

echo "P4 after rollback:"
rdmsr -p "$CORE" "$P4"

echo "COFVID_STS after rollback:"
rdmsr -p "$CORE" "$CSTS"

echo "== rollback temp =="
sensors | sed -n "/k10temp/,+5p" || true
'
```

Raw output:

```text
Rolling back P4 on core 4 to 0x8000013540003440
P4 after rollback:
8000013540003440
COFVID_STS after rollback:
180000140043440
== rollback temp ==
k10temp-pci-00c3
Adapter: PCI adapter
temp1:        +36.9°C  (high = +70.0°C)
                       (crit = +80.0°C, hyst = +75.0°C)

nouveau-pci-0100
```

## G. DECODE

Decode rule used:

- `CpuFid = value & 0x3f`
- `CpuDid = (value >> 6) & 0x7`
- `CpuVid = (value >> 9) & 0x7f`
- `NbVid = (value >> 25) & 0x7f`
- approximate SVI voltage = `1.55 - VID * 0.0125`

P4 readback before write:

- Raw: `0x8000013540003440`
- CpuFid: `0x00`
- CpuDid: `1`
- CpuVid: `0x1A`
- CpuVid approximate SVI voltage: `1.225 V`
- NbVid: `0x20`
- NbVid approximate SVI voltage: `1.150 V`

P4 readback after write:

- Raw: `0x80000135400040c0`
- CpuFid: `0x00`
- CpuDid: `3`
- CpuVid: `0x20`
- CpuVid approximate SVI voltage: `1.150 V`
- NbVid: `0x20`
- NbVid approximate SVI voltage: `1.150 V`

COFVID_STS before transition:

- Raw: `0x1800001400434c0`
- CpuFid: `0x00`
- CpuDid: `3`
- CpuVid: `0x1A`
- CpuVid approximate SVI voltage: `1.225 V`
- NbVid: `0x20`
- NbVid approximate SVI voltage: `1.150 V`

COFVID_STS after transition:

- Raw: `0x1800001400434c0`
- CpuFid: `0x00`
- CpuDid: `3`
- CpuVid: `0x1A`
- CpuVid approximate SVI voltage: `1.225 V`
- NbVid: `0x20`
- NbVid approximate SVI voltage: `1.150 V`

COFVID_STS after rollback:

- Raw: `0x180000140043440`
- CpuFid: `0x00`
- CpuDid: `1`
- CpuVid: `0x1A`
- CpuVid approximate SVI voltage: `1.225 V`
- NbVid: `0x20`
- NbVid approximate SVI voltage: `1.150 V`

PSTATE_CTL before write:

- Raw: `0x4`

COFVID_CTL before write:

- Raw: `0x40043440`
- CpuFid: `0x00`
- CpuDid: `1`
- CpuVid: `0x1A`
- CpuVid approximate SVI voltage: `1.225 V`
- NbVid: `0x20`
- NbVid approximate SVI voltage: `1.150 V`

## H. DECISION LOGIC

Observed condition:

- P4 accepted `0x80000135400040c0`.
- COFVID_STS after P-state cycling reported CpuVid `0x1A`, not requested CpuVid `0x20`.

Decision:

- Verdict = `RUNTIME_VID_CLAMPED`.
- Meaning: definition writes work, but hardware output VID is clamped.

## I. NEXT ACTION

Do not run the incremental sweep. The runtime VID path did not accept CpuVid `0x20`, so further lower-voltage runtime MSR sweep values are not justified by this result.

Recommended escalation:

1. P4-specific AGESA patch design, not the global `JBE -> JAE` branch change.
2. VRM controller chip identification photos.
3. Optional frequency-detuning fallback for analog instability.

## J. FINAL PACK

- Raw command output: included.
- Decoded MSR values: included.
- Verdict: `RUNTIME_VID_CLAMPED`.
- Runtime undervolt path alive: no, not through this runtime K10 P-state MSR route.
- Clamp confirmed: yes, COFVID_STS stayed at CpuVid `0x1A` after P4 was redefined to CpuVid `0x20` and P-state cycling was performed.
- Exact next action: pursue P4-specific AGESA patch design or VRM controller identification; use frequency detuning only as a non-VID fallback.
- BIOS flash actions: none.
- ACPI modifications: none.
- Physical board modifications: none.
- Broad exploration: stopped after the decisive runtime VID test.
