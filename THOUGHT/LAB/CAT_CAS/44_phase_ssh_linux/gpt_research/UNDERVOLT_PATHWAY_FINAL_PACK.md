# Undervolt Pathway Final Pack

## A. Executive Verdict

`TWO_OR_MORE_UNDERVOLT_PATHWAYS_FOUND`

Two evidence-backed voltage-control routes were found:

1. Runtime K10 P-state MSR control: best non-flash candidate.
2. BIOS/AGESA firmware patch: real route, but the existing one-byte patch is not flash-ready.

A third route, frequency detuning without VID reduction, is verified for analog instability work but does not lower voltage.

## B. Candidate Table

| # | Pathway | Status | Evidence | Risk | Next Human Action |
|---|---|---|---|---|---|
| 1 | Runtime K10 P-state MSR control | `STRONG_CANDIDATE` | Local DID writes work; K10 P-state MSRs decoded; k10ctl/TurionPowerControl precedent | Human-only MSR writes | Run VID `0x20` P4 test with rollback |
| 2 | BIOS/AGESA firmware patch | `STRONG_CANDIDATE`, not flash-ready | `AmdProcessorInitPeim` uncompressed; branch and checksum offsets verified | `BRICK_RISK`, `PROGRAMMER_REQUIRED` | Do not flash; derive safer P4/table-specific patch |
| 3 | NB PCI config | `READONLY_ONLY` | `F3xA0`/`F3xDC` dumps contain VID-like `0x1A`; BKDG documents VID filtering | Unsafe until field mismatch reconciled | Re-dump only |
| 4 | VRM/SVI board path | `WEAK_CANDIDATE` | SMBus negative; board has split-plane/VRM evidence; chip ID missing | `PHYSICAL_MOD_REQUIRED` | Photograph VRM PWM controller |
| 5 | ACPI `_PSS` route | `LOW_VALUE` | SSDT exposes only selectors 0-3, no visible VID values | Low | Preserve tables, do not patch |
| 6 | Frequency detuning without VID reduction | `VERIFIED_READY` | DID 0-4 verified, 100-3200 MHz range | No voltage reduction | Use as analog fallback |

## C. Best Pathway

Runtime K10 P-state MSR control.

Reason:

- It avoids BIOS flashing.
- It directly tests the hardware clamp.
- It uses the documented Family 10h P-state control surface.
- It has a simple rollback.
- Local evidence already proves frequency writes and P-state cycling work.

Required artifacts:

- Current read-only MSR dump.
- Current temperature baseline.
- Confirmation current P4 is `0x8000013540003440`.

Exact next command:

Use the human-approved write block from [UNDERVOLT_PATHWAY_2_NB_PCI_CONFIG.md](UNDERVOLT_PATHWAY_2_NB_PCI_CONFIG.md). Test value:

```text
0x80000135400040c0
```

Recovery plan:

- Roll back P4 to `0x8000013540003440`.
- Cycle core 4 P0 -> P4.
- If core 4 stalls, keep SSH on housekeeping core, kill workload, and restore P4 from another core if MSR access remains available.
- If system becomes unstable, reboot; P-state MSR edits are runtime only and should not persist across cold boot.

## D. Second-Best Pathway

BIOS/AGESA firmware patch.

Reason:

- The target function and byte offsets are real.
- The module is uncompressed.
- The patch changes a branch inside a loop that scans `MSRC001_0064` through `MSRC001_0068` and writes P-state MSRs.
- The checksum offset is verified in the local dump.

Required artifacts:

- External SPI programmer and verified stock backup.
- Official FD BIOS archive extracted and compared to local dump.
- Corrected patch design. Current global `JBE -> JAE` appears to affect all P-states, not just P4.
- UEFITool parse of patched image.

Exact next step:

Do not flash. First build or derive a P4/table-specific firmware patch, or reduce high-frequency P-state risk before any global branch reversal.

Recovery plan:

- Treat external programmer as required.
- Keep stock dump hash `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`.
- Verify the SPI chip externally before modifying firmware.
- Do not rely solely on DualBIOS.

## E. Rejected Or Low-value Pathways

- Pure `F3xA0` write: rejected for now. BKDG indicates PSI/VID threshold behavior and hardware filtering, not a proven active undervolt command.
- `F3xDC` direct write: rejected for now. BKDG ties it to `PstateMaxVal` and C5/AltVid behavior; not proven for active-core undervolt.
- ACPI `_PSS` override: low value. The table exposes selectors, not VID values.
- SMBus/PMBus VRM write: rejected for now. No CPU VRM endpoint found on SMBus scans.
- Blind physical mod: rejected. Needs chip ID and datasheet first.

## F. Safety / Recovery

- BIOS backup status: one local 4 MB dump exists; second read/compare still needed.
- Flash recovery method: external SPI programmer required before firmware experiments.
- External programmer needed: yes, for BIOS route.
- Rollback plan:
  - Runtime MSR path: restore P4 to `0x8000013540003440`, reboot if needed.
  - BIOS path: reflash verified stock dump externally.
- Thermal limit: keep k10temp below 60 C.
- Voltage floor risk: hardware may clamp lower-VID requests regardless of software path.
- Brick risk: firmware path can fail before SSH or OS boot.

## G. Commands Prepared But NOT Run

Read-only verification:

```bash
ssh root@192.168.137.100 '
set -eu
hostname
uname -a
cat /proc/cmdline
for c in 0 1 2 3 4 5; do
  echo "== core $c =="
  for m in 0xc0010061 0xc0010062 0xc0010064 0xc0010065 0xc0010066 0xc0010067 0xc0010068 0xc0010070 0xc0010071; do
    printf "%s " "$m"; rdmsr -p "$c" "$m" || true
  done
done
setpci -s 00:18.3 a0.l d8.l dc.l e8.l 1fc.l || true
sensors | sed -n "/k10temp/,+3p"
'
```

Human-approved runtime VID test:

```bash
ssh root@192.168.137.100 '
set -eu
CORE=4
P4=0xc0010068
PCTL=0xc0010062
CSTS=0xc0010071
ORIG=$(rdmsr -p "$CORE" "$P4")
test "$ORIG" = "8000013540003440" || { echo "Unexpected P4: $ORIG"; exit 1; }
wrmsr -p "$CORE" "$P4" 0x80000135400040c0
wrmsr -p "$CORE" "$PCTL" 0
sleep 0.2
wrmsr -p "$CORE" "$PCTL" 4
sleep 0.5
rdmsr -p "$CORE" "$CSTS"
sensors | sed -n "/k10temp/,+3p"
'
```

Rollback:

```bash
ssh root@192.168.137.100 '
set -eu
CORE=4
wrmsr -p "$CORE" 0xc0010068 0x8000013540003440
wrmsr -p "$CORE" 0xc0010062 0
sleep 0.2
wrmsr -p "$CORE" 0xc0010062 4
sleep 0.5
rdmsr -p "$CORE" 0xc0010068
rdmsr -p "$CORE" 0xc0010071
'
```

## H. Missing Artifacts

- Photos:
  - PCB revision marking.
  - CPU VRM controller chip marking.
  - SPI flash chip marking.
- Dumps:
  - Second BIOS read and hash.
  - Current MSR dump for all six cores.
  - Current `setpci` and raw config dumps for `00:18.0` through `00:18.4`.
  - Current ACPI `DSDT.dat`, `SSDT.dat`, and decompiled `.dsl`.
- Logs:
  - Output of the mild VID `0x20` P4 test.
  - Any QFlash acceptance/rejection screen for a non-flashed modified image.

## I. Next Session Script

1. SSH into the Phenom:

```powershell
ssh root@192.168.137.100
```

2. Run the read-only verification block in section G.

3. Confirm P4 on core 4 is exactly `8000013540003440`.

4. Run the human-approved runtime VID `0x20` test.

5. Immediately run rollback.

6. Decode `COFVID_STS`:

- If VID reports `0x20`, continue with incremental VID sweep.
- If VID clamps to `0x1A`, firmware or physical route is required.
- If the core stalls but SSH remains alive, reboot or restore from a responsive core.

7. After skating, provide:

- Command output.
- VRM photos.
- Board revision.
- Confirmation whether `/tmp/bios_patched.bin` exists.
