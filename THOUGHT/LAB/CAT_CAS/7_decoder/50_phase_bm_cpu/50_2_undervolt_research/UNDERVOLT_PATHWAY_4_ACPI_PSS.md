# Undervolt Pathway 4: ACPI _PSS / BIOS Table Route

## Status

`ACPI_PATHWAY_LOW_VALUE`

Candidate quality: `WEAK_CANDIDATE`

## Evidence

Local ACPI extraction log:

- DSDT table length: `0x52C7` / 21,191 bytes.
- SSDT table length: `0x0E10` / 3,600 bytes.
- SSDT signature/vendor context: `AMD POWERNOW`.
- `_PSS` exposes four P-states, not five.

Extracted `_PSS` package values:

| P-state | Core frequency field | Power field | Control | Status |
|---|---:|---:|---:|---:|
| P0 | `0x00000C80` / 3200 MHz | `0x0000524B` | `0x00000000` | `0x00000000` |
| P1 | `0x00000960` / 2400 MHz | `0x00003B6A` | `0x00000001` | `0x00000001` |
| P2 | `0x00000640` / 1600 MHz | `0x000029D6` | `0x00000002` | `0x00000002` |
| P3 | `0x00000320` / 800 MHz | `0x0000195C` | `0x00000003` | `0x00000003` |

`XPSS` buffers likewise carry control/status selectors `0`, `1`, `2`, `3`; no explicit VID bytes are visible in the extracted context.

Local ACPI scan found `RVID` strings, but no complete voltage-control method was preserved in the current logs.

## Interpretation

The ACPI path only exposes OS-visible P-state selectors. It does not define the underlying VID values for the Family 10h P-state MSRs. The firmware/AGESA path programs those values before Linux uses `_PSS`.

Linux can request ACPI P-states, but this table does not appear to provide a direct lower-VID control route. A DSDT/SSDT override might expose an additional P-state selector, but it would still depend on the underlying `MSRC001_0064`-`MSRC001_0068` definitions and hardware VID filtering.

## Read-only Commands

Human-run:

```bash
ssh root@192.168.137.100 '
set -eu
mkdir -p /tmp/acpi_readonly
cp /sys/firmware/acpi/tables/DSDT /tmp/acpi_readonly/DSDT.dat
cp /sys/firmware/acpi/tables/SSDT /tmp/acpi_readonly/SSDT.dat 2>/dev/null || true
iasl -d /tmp/acpi_readonly/DSDT.dat /tmp/acpi_readonly/SSDT.dat 2>&1 | tee /tmp/acpi_readonly/iasl.log
grep -nEi "_PSS|XPSS|_PCT|_PPC|_PSD|RVID|VID|Voltage|PowerNow|C0010070|C0010062" /tmp/acpi_readonly/*.dsl || true
'
```

## No Automatic Patch

No ACPI override is prepared. A useful override would need:

- Full `DSDT.dsl` and `SSDT.dsl`.
- Confirmation of current Linux cpufreq driver (`acpi-cpufreq` vs direct MSR tooling).
- Proof that a new selector maps to a valid MSR P-state.
- Rollback boot entry without the override.

## Decision

ACPI is useful for documenting the OS-visible P-state table, but it is not a primary undervolt pathway for this board. It should not be pursued until runtime MSR and firmware routes are exhausted.
