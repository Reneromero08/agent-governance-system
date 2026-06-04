# Undervolt Pathway 2: NB PCI Config And Runtime VID Control Route

## Status

`NB_PCI_PATHWAY_READONLY_ONLY`

Candidate quality: `STRONG_CANDIDATE` only when paired with runtime P-state MSR control. Pure `F3xA0`/`F3xDC` writes are not proven as a safe active-core undervolt mechanism.

Risk tags: `HUMAN_WRITE_ONLY`

## Evidence

Local CPU NB PCI functions:

- `00:18.0` Family 10h Processor HyperTransport Configuration
- `00:18.1` Address Map
- `00:18.2` DRAM Controller
- `00:18.3` Miscellaneous Control
- `00:18.4` Link Control

Local raw function 3 dump around `F3xA0`:

```text
000000a0: 0008 1aa0 ef0f 0c2f ...
```

Interpreted as little-endian dword at offset `0xA0`: `0xA01A0800`. This contains `0x1A`, matching the observed approximate VID floor.

Local raw function 3 dump around `F3xDC`:

```text
000000d0: 0000 0000 260f 81c8 1513 0003 1a64 6700
```

Interpreted as little-endian dword at offset `0xDC`: `0x0067641A`. This also contains `0x1A`, but the field meaning is not proven from local evidence alone.

Local `setpci` summary conflict:

```text
0xa0: c00a0000
```

That conflicts with the raw config dump. This must be reconciled before any write.

## BKDG Interpretation

The AMD Family 10h BKDG supports the following conclusions:

- Internal VID fields are 7-bit.
- Hardware filters requested VID through MinVid/MaxVid limits before generating SVI/PVI commands.
- In SVI mode, `F3xA0[PsiVid]` is a PSI_L threshold, not a general active-core voltage command.
- `F3xDC` participates in `PstateMaxVal` and AltVid/C5 behavior.
- AltVid is documented as a C5-state mechanism, not an active P-state undervolt knob.

Therefore, direct `F3xA0` writes are read-only research until a live dump proves the exact field and effect.

## Read-only Command Set

Human-run, read-only:

```bash
ssh root@192.168.137.100 '
set -eu
echo "== identity =="
hostname
uname -a
cat /proc/cmdline

echo "== NB devices =="
lspci -nn -s 00:18.0
lspci -nn -s 00:18.1
lspci -nn -s 00:18.2
lspci -nn -s 00:18.3
lspci -nn -s 00:18.4

echo "== setpci reads =="
for fn in 0 1 2 3 4; do
  echo "-- 00:18.$fn --"
  setpci -s 00:18.$fn 40.l 64.l 80.l 84.l a0.l a8.l d8.l dc.l e8.l 1fc.l 2>/dev/null || true
done

echo "== raw config function 3 =="
xxd -g 1 -l 0x120 /sys/devices/pci0000:00/0000:00:18.3/config
'
```

## Human-approved Test Write

No pure NB PCI write is recommended yet. The only prepared write test is a runtime P-state MSR test, because that is the documented K10 control path and has rollback.

Use only after read-only dumps are saved and after confirming the current P4 value is exactly `0x8000013540003440`.

```bash
ssh root@192.168.137.100 '
set -eu
CORE=4
P4=0xc0010068
PCTL=0xc0010062
CSTS=0xc0010071

ORIG=$(rdmsr -p "$CORE" "$P4")
echo "Original P4 on core $CORE: $ORIG"
test "$ORIG" = "8000013540003440" || { echo "Unexpected P4, aborting"; exit 1; }

# P4 test: FID 0, DID 3 (about 200 MHz), CpuVid 0x20 and NbVid 0x20.
# This is a mild below-floor test compared with 0x3A and is intended to prove clamp behavior.
TEST=80000135400040c0
echo "Writing human-approved test P4: $TEST"
wrmsr -p "$CORE" "$P4" "0x$TEST"

wrmsr -p "$CORE" "$PCTL" 0
sleep 0.2
wrmsr -p "$CORE" "$PCTL" 4
sleep 0.5

echo "COFVID_STS after transition:"
rdmsr -p "$CORE" "$CSTS"
sensors | sed -n "/k10temp/,+3p"
'
```

## Rollback

Human-run rollback:

```bash
ssh root@192.168.137.100 '
set -eu
CORE=4
P4=0xc0010068
PCTL=0xc0010062
CSTS=0xc0010071

wrmsr -p "$CORE" "$P4" 0x8000013540003440
wrmsr -p "$CORE" "$PCTL" 0
sleep 0.2
wrmsr -p "$CORE" "$PCTL" 4
sleep 0.5
echo "P4 restored:"
rdmsr -p "$CORE" "$P4"
echo "COFVID_STS:"
rdmsr -p "$CORE" "$CSTS"
'
```

## Decision

Pure NB PCI register writes are not yet a safe pathway. The NB evidence is valuable because it explains the floor, but the next executable route is runtime P-state MSR testing with both CpuVid and NbVid controlled and with immediate rollback.
