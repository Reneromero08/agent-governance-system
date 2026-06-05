# PHASE2_DONOR_DIFF_REPORT

## Verdict

`PUBLIC_MOD_DONOR_DIFFED`

Route 4 advanced. An exact GA-970A-DS3P rev. 1.0 F2j stock BIOS package and a public F2j NVMe donor image were acquired inside `cpu_hack/mod_donors/`, hashed, parsed, and diffed for workflow study only.

No donor image is a flash candidate.

## Sources

| Artifact | Source | Local path |
|---|---|---|
| Official stock F2j package | `https://download.gigabyte.com/FileList/BIOS/mb_bios_ga-970a-ds3p_f2j.zip?v=a7771fc10dad45586d4c1f6e67316f68` | `cpu_hack/mod_donors/gigabyte_rev1_stock_F2j.zip` |
| Public NVMe donor package | `https://winraid.level1techs.com/uploads/short-url/hrKqsZfCFqq7lq1zus0TomVE58v.zip` | `cpu_hack/mod_donors/winraid_970ADS3PNVME.zip` |
| Donor thread | `https://winraid.level1techs.com/t/offer-gigabyte-ga-970a-ds3p-rev-1-0-nvme-mod/33089` | URL evidence only |

## Hashes

| File | SHA-256 |
|---|---|
| `gigabyte_rev1_stock_F2j.zip` | `5ADA05ED5697A2C972ED47B282ABB2311A8EC36A2C7D5F1FB61943937C7A0F9E` |
| `970ADS3P.F2j` | `FB4018464DE6D60AF10E8B9BB853D2A3FDEE1353AA742FD86F09806B5E0B307A` |
| `winraid_970ADS3PNVME.zip` | `74B0AD7B57928C9D4C4950F5718CBA1A02C61BAA5E46C348569761A4EDD180D6` |
| `970ADS3PNVME.F2j` | `BF86BCAF4787B7C908F197DDB1F1FE7B0FA124C5B20EEC7A14DE11FAA5B5F157` |

## Parse Artifacts

| Image | UEFIExtract report |
|---|---|
| Stock F2j | `cpu_hack/mod_donors/gigabyte_rev1_stock_F2j_extracted/970ADS3P.F2j.report.txt` |
| NVMe donor F2j | `cpu_hack/mod_donors/winraid_970ADS3PNVME_extracted/970ADS3PNVME.F2j.report.txt` |

Both reports were produced by local `UEFIExtract`.

## Structural Diff

The donor operation is a single DXE insertion into existing free space in the main DXE firmware volume.

| Region | Stock | Donor |
|---|---|---|
| `0x002C58A0` | Volume free space, size `0x0005A760` | New DXE driver file |
| New file | absent | `NvmExpressDxe_4` |
| New file GUID | absent | `5BE3BDF4-53CF-46A3-A6A9-73C34A6E5EE3` |
| New file size | absent | `0x00005160` |
| New PE32 section | absent | raw `0x002C58B8`, size `0x00005124` |
| New UI section | absent | raw `0x002CA9DC`, size `0x00000024` |
| Remaining free space | `0x0005A760` | `0x00055600` at `0x002CAA00` |

Stock report excerpt:

```text
Free space | 002C58A0 | 0005A760 | Volume free space
Volume     | 00320000 | 00020000 | 8C8CE578-8A3D-4F1C-9935-896185C32DD3
```

Donor report excerpt:

```text
File       | DXE driver | 002C58A0 | 00005160 | 5BE3BDF4-53CF-46A3-A6A9-73C34A6E5EE3 | NvmExpressDxe_4
Section    | PE32 image | 002C58B8 | 00005124 | PE32 image section
Section    | UI         | 002CA9DC | 00000024 | UI section
Free space |            | 002CAA00 | 00055600 | Volume free space
```

## Byte Diff Envelope

| Metric | Value |
|---|---:|
| Stock size | `4,194,304` bytes |
| Donor size | `4,194,304` bytes |
| First changed offset | `0x002C58A0` |
| Last changed offset | `0x002CA9FF` |
| Changed byte count | `20,281` |
| Changed byte ranges | `423` contiguous ranges inside the inserted-file envelope |
| Pre-insert bytes changed | `0` |
| Remaining-free-space bytes changed | `0` |
| Bytes after `0x00320000` changed | `0` |

The donor did not shift later volumes or rewrite unrelated image regions. The structural lesson is that the successful workflow consumed existing free space and preserved the rest of the image byte-for-byte.

## Workflow Lessons

1. Aptio/AMI module insertion can be limited to a free-space region without moving later volumes.
2. The parse-clean donor report shows the main integrity concern is valid FFS file insertion and volume checksum handling, not global image reshuffling.
3. This donor supports the rebuild workflow route, not the voltage route directly. It does not identify an AGESA P4 VID source.
4. A P4-safe candidate still requires a proven target in the owned `AmdProcessorInitPeim` route and a parse-clean no-op rebuild.

## Actionability

`PUBLIC_MOD_DONOR_DIFFED` is met.

`BYTE_READY_HUMAN_REVIEW` is not met.

The donor route teaches safe structural constraints for future module replacement or no-op rebuild validation, but it does not provide a P4-safe VID edit and must not be flashed.

## Do-Not-Do

- Do not flash the donor image.
- Do not treat the donor as board-state-compatible evidence.
- Do not derive P4 voltage bytes from the NVMe module insertion.
- Do not produce patch bytes from this donor diff.
