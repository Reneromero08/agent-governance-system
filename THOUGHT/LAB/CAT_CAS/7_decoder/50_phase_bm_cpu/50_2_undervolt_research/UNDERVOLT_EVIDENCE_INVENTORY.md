# Undervolt Evidence Inventory

## Target

- CPU: AMD Phenom II X6 1090T, Thuban, Family 10h, 45 nm SOI.
- Board: Gigabyte GA-970A-DS3P, AMD 970 northbridge + SB950 southbridge.
- BIOS: American Megatrends Inc. FD, release date 02/26/2016, BIOS revision 4.6 from local `dmidecode`.
- OS: Debian 13, kernel `6.12.86+deb13-amd64`, host `catcas`, SSH target `root@192.168.137.100`.
- Local board evidence: [REPORT.md](REPORT.md), [ROADMAP.md](ROADMAP.md), [cpu_hack/catcas_vrm_probe.log](cpu_hack/catcas_vrm_probe.log).

## External References Used

- Gigabyte GA-970A-DS3P rev. 2.x support/spec page: https://www.gigabyte.com/Motherboard/GA-970A-DS3P-rev-2x/support
- Gigabyte GA-970A-DS3P rev. 1.0 overview showing UEFI DualBIOS and split power plane/4+1 VRM wording: https://www.gigabyte.com/sg/Motherboard/GA-970A-DS3P-rev-10
- AMD Family 10h BKDG Rev. 3.62 mirror: https://device.report/m/80554786942e004cc15c2f27231c0df309aa4961bcb3b1a81b4bc801ea590780
- K10 Linux software precedent: k10ctl package summary says it targets AMD Family 10h P-state frequency/voltage modification: https://t2linux.com/packages/k10ctl
- TurionPowerControl precedent for K10/Phenom II voltage and frequency control: https://www.phoronix.com/news/ODY2MA

## BIOS Artifacts

- Local dump: [cpu_hack/bios_dump.bin](cpu_hack/bios_dump.bin)
- Size: 4,194,304 bytes.
- SHA-256: `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`.
- Flash chip from local flashrom log: Macronix `MX25L3205(A)`, 4096 KB, SPI, mapped at `0x00000000ffc00000`.
- Official Gigabyte support page lists BIOS `FD`, size 2.74 MB, date Mar 2, 2016, with a BIOS flashing risk warning.
- UEFI report shows `AmdProcessorInitPeim` as an uncompressed PEI module:
  - File offset `0x00340048`, size `0x000563D2`, GUID `DE3E049C-A218-4891-8658-5FC0FA84C788`.
  - PE32 image section at `0x00340088`, size `0x00056364`.
- GUID map also identifies `DualBiosDxe`, `DualBiosPei`, `DualBiosSMM`, `ACPI`, `AmdAgesaDxeDriver`, and `AmdProcessorInitPeim`.

## MSRs Decoded

From [ROADMAP.md](ROADMAP.md) and [cpu_hack/catcas_vrm_probe.log](cpu_hack/catcas_vrm_probe.log):

| MSR | Meaning | Observed value / decode |
|---|---|---|
| `0xC0010015` | HWCR | `0x0000000001000011`, invariant/nonstop TSC confirmed locally |
| `0xC0010061` | P-state current/limit | Roadmap says write-locked, value `0x30` |
| `0xC0010062` | PSTATE_CTL | Local log showed `0x0000000000000003` during probe |
| `0xC0010064` | P0 | `0x8000019e40000c14`, FID `0x14`, DID `0`, CpuVid `0x06`, NbVid `0x20` |
| `0xC0010065` | P1 | `0x8000019f40002410`, FID `0x10`, DID `0`, CpuVid `0x12`, NbVid `0x20` |
| `0xC0010066` | P2 | `0x8000017540002808`, FID `0x08`, DID `0`, CpuVid `0x14`, NbVid `0x20` |
| `0xC0010067` | P3 | `0x8000015440002c00`, FID `0x00`, DID `0`, CpuVid `0x16`, NbVid `0x20` |
| `0xC0010068` | P4 | `0x8000013540003440`, FID `0x00`, DID `1`, CpuVid `0x1A`, NbVid `0x20` |
| `0xC0010070` | COFVID_CTL | `0x0000000040043440`; roadmap says writes are overridden by controller |
| `0xC0010071` | COFVID_STS | `0x0180000140042440`; used for readback of current FID/DID/VID |

The local decode formula used in scripts is:

- `CpuFid = value & 0x3f`
- `CpuDid = (value >> 6) & 0x7`
- `CpuVid = (value >> 9) & 0x7f`
- `NbVid = (value >> 25) & 0x7f`
- approximate SVI voltage: `1.55 - VID * 0.0125`

## PCI Config Evidence

Relevant CPU northbridge device functions from local `lspci`:

- `00:18.0` HyperTransport Configuration
- `00:18.1` Address Map
- `00:18.2` DRAM Controller
- `00:18.3` Miscellaneous Control
- `00:18.4` Link Control

Local NB function 3 raw config dump around `0xA0` shows bytes:

```text
000000a0: 0008 1aa0 ef0f 0c2f ...
```

Interpreted little-endian as a dword at `F3xA0`, that is `0xA01A0800`, containing `0x1A` in the expected VID-related byte position. However, the same probe log also has a `setpci` summary line reporting `0xa0: c00a0000`. This is a missing-evidence item: before any write, re-dump `F3xA0` with multiple methods and reconcile the endian/value mismatch.

The local raw dump around `F3xDC` includes:

```text
000000d0: 0000 0000 260f 81c8 1513 0003 1a64 6700
```

The roadmap says `F3xDC` contains P-state voltage parameters. The AMD BKDG specifically describes `F3xDC` as containing fields including `PstateMaxVal` and C5/AltVid-related fields; it is not proven to be a general active-core undervolt command register.

## VID Floor Observations

From [ROADMAP.md](ROADMAP.md):

- Attempted VID `0x3A`, approx `0.825 V`: rejected by hardware.
- Attempted VID `0x20`, approx `1.150 V`: rejected by hardware.
- Observed floor: approximately VID `0x1A`, about `1.225 V`.
- Three enforcement layers were inferred: P-state MSR definitions, NB PCI config `F3xA0`, and SVI hardware clamping.
- SMBus scan found only RAM SPD devices at `0x50`-`0x53`; no accessible CPU VRM/PMBus controller.

Important correction: the AMD BKDG describes `F3xA0[PsiVid]` as a PSI_L threshold and also documents hardware filtering of input VID to output VID using MinVid/MaxVid limits reported in `MSRC001_0071`. Therefore `F3xA0` may be evidence of a threshold/floor, but it is not proven to be the direct undervolt knob.

## Existing AGESA Patch Notes

Roadmap-stated patch:

- BIOS offset `0x00366E3E`: byte `0x76` (`JBE`) -> `0x73` (`JAE`).
- FFS checksum byte offset `0x00340059`: `0x8E` -> `0x91`.
- Claimed purpose: reverse voltage-normalizer comparison so higher numeric VID/lower voltage is selected.

Raw local bytes at `0x00366E20` include:

```text
... 83 E0 7F 83 E2 7F C1 EF 09 3B C2 74 41 76 02 8B C2 ...
```

Raw local FFS header byte at `0x00340059` is `0x8E`. The checksum arithmetic for changing `0x76` to `0x73` is internally consistent for an 8-bit checksum correction, but the boot-safety of the logic is not verified.

## Safety And Recovery Assumptions

- Board has Gigabyte UEFI DualBIOS per vendor page and local BIOS GUIDs include DualBios modules.
- DualBIOS is not a guaranteed recovery path for an early PEI undervolt failure.
- External SPI recovery should be treated as required before any firmware flash:
  - CH341A-class programmer or equivalent.
  - SOIC-8 clip or chip removal plan.
  - Known-good stock dump verified by hash.
- Thermal envelope from local tests: k10temp stayed around 42-46 C during DID/frequency work; roadmap uses less than 60 C as the lab limit.

## Missing Artifacts

- Current live read-only dump of `rdmsr -a` for `0xC0010061` through `0xC0010071`.
- Current live `setpci` and `/sys/.../config` dumps for `00:18.0` through `00:18.4`, especially `F3xA0`, `F3xD8`, `F3xDC`, `F3xE8`, and `F3x1FC`.
- Current ACPI table files and decompiled DSL, not only extraction logs.
- Physical board revision printed on PCB.
- High-resolution photo of the CPU VRM area and readable PWM controller marking.
- Confirmation whether `/tmp/bios_patched.bin` still exists on `catcas`; no local patched image was found in this lab directory.
- A second BIOS backup read, compared byte-for-byte with [cpu_hack/bios_dump.bin](cpu_hack/bios_dump.bin).
