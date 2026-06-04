# Undervolt Pathway 3: VRM, SVI, And Board-level Route

## Status

`VRM_PATHWAY_NEEDS_CHIP_ID`

Candidate quality: `WEAK_CANDIDATE`

Risk tags: `PHYSICAL_MOD_REQUIRED` if pursued beyond identification.

## Evidence

Local SMBus/I2C scan:

- AMD SBx00 SMBus controller at `00:14.0`, driver `piix4_smbus`.
- I2C buses `0` through `19` were listed.
- Bus 0 showed only RAM SPD devices at `0x50`-`0x53`.
- Other SB PIIX4 ports showed no CPU VRM-like device.
- GPU/display I2C buses produced many expected display/GPU artifacts, not a CPU VRM controller.
- `hwmon` devices found: `k10temp`, `nouveau`, and `hidpp_battery_0`.
- No PMBus/VRM sensor endpoint was found.

Local board evidence:

- DMI product: `970A-DS3P`.
- DMI board revision field: `To be filled by O.E.M.`.
- BIOS report includes IT8728-related modules (`It8728SmmFeatures`, `IT8728SioAcBackSmm`), but local Linux `hwmon` did not expose voltage-control hooks through Super I/O.

Vendor evidence:

- Gigabyte rev. 2.x page confirms AMD 970 + SB950 and UEFI DualBIOS.
- Gigabyte rev. 1.0 page states split power plane and 4+1 phase VRM for AM3 Phenom II/Athlon II. The lab board is rev. 2.x or unknown from DMI, so do not assume the exact same PWM controller or component layout.

## Interpretation

The CPU voltage regulator is likely controlled by the AMD SVI/PVI interface from the CPU/NB rather than by an exposed PMBus controller. The local SMBus result means a software PMBus undervolt path is not currently available.

A board-level route remains possible only after identifying the physical VRM PWM controller and its feedback/SVI pins. Without that chip ID, any electrical modification is speculative.

## Human Visual Inspection Checklist

Needed photos:

- Full motherboard front, readable PCB revision text.
- CPU socket + VRM area, high resolution.
- Close-up of the small PWM controller IC near the CPU VRM chokes/MOSFETs.
- Close-up of all chips between the CPU socket, 4-pin ATX12V connector, and VRM phases.
- Close-up of BIOS SPI flash chip marking.

What to record:

- Exact PWM controller top marking.
- Package pin count.
- Whether there are one or two PWM controllers for CPU/NB planes.
- SVI/SVC/SVD or PVI VID pin labels if visible from datasheet.
- Feedback (`FB`) pin and resistor divider network, if identifiable.
- Any nearby zero-ohm links or test pads tied to feedback or VID/SVI lines.

Likely controller vendors to check after chip ID:

- Intersil/Renesas
- Richtek
- uPI Semiconductor
- International Rectifier/Infineon
- ON Semiconductor
- Anpec

## Alternate Bus Probing

Safe read-only checks after boot:

```bash
ssh root@192.168.137.100 '
set -eu
modprobe i2c-dev || true
i2cdetect -l
for b in /dev/i2c-*; do
  n=${b#/dev/i2c-}
  echo "== bus $n =="
  i2cdetect -y "$n" || true
done
find /sys/class/hwmon -maxdepth 2 -type f -name name -print -exec cat {} \;
find /sys -iname "*voltage*" -o -iname "*pmbus*" -o -iname "*vrm*" 2>/dev/null | head -200
'
```

No blind `i2cset` or PMBus writes are prepared.

## Possible Physical Route

If the PWM controller datasheet confirms an analog feedback divider:

- Route type: feedback-divider offset to lower requested Vcore.
- Status: `PHYSICAL_MOD_REQUIRED`.
- Requirements:
  - External DMM on Vcore.
  - Bench power supply or current-limited startup if available.
  - Stock BIOS and known-good boot path.
  - External SPI programmer.
  - Reversible resistor/potentiometer plan.
  - Start with tiny offsets only.

Do not proceed to physical modification without chip ID, datasheet, pinout, and measured stock Vcore points.

## Decision

The VRM/SVI board route is real in principle but not executable yet. The next human action is chip identification, not electrical modification.
