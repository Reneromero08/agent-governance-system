# Undervolt Pathway 1: BIOS AGESA Firmware Patch

## Status

`BIOS_PATHWAY_NEEDS_MORE_EVIDENCE`

Candidate quality: `STRONG_CANDIDATE`, but the existing one-byte patch is not ready for human flash.

Risk tags: `BRICK_RISK`, `PROGRAMMER_REQUIRED`

## Evidence

- BIOS dump exists locally at [50_2_firmware/cpu_hack/bios_dump.bin](50_2_firmware/cpu_hack/bios_dump.bin), 4,194,304 bytes.
- SHA-256: `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`.
- BIOS version from local DMI: `FD`, release date `02/26/2016`.
- Gigabyte support page lists rev. 2.x BIOS `FD`, dated `Mar 2, 2016`, and warns that inadequate BIOS flashing may cause malfunction.
- UEFI report shows `AmdProcessorInitPeim`:
  - File offset `0x00340048`
  - Size `0x000563D2`
  - GUID `DE3E049C-A218-4891-8658-5FC0FA84C788`
  - PE32 image at `0x00340088`
  - No compressed section wrapper on this PEI module in the report.
- Local GUID map contains `DualBiosDxe`, `DualBiosPei`, and `DualBiosSMM`.

## Branch Context

Local bytes around BIOS offset `0x00366E20`:

```text
8B D6 0F AC D0 19 C1 EA 19 8B D1 8B FE 0F AC FA
09 83 E0 7F 83 E2 7F C1 EF 09 3B C2 74 41 76 02
8B C2 FF 75 08 83 E0 7F 33 D2 8B F8 8B DA 0F A4
FB 10 C1 E7 10 0B F8 0B DA 0F A4 FB 09 8D 45 F4
81 E1 FF 01 FF 01 50 FF 75 FC C1 E7 09 0B F9 0B
DE 89 7D F4 89 5D F8 E8 15 E0 FD FF 83 C4 0C FF
45 FC 81 7D FC 68 00 01 C0 0F 86 67 FF FF FF 5F
```

Relevant disassembly from [50_2_firmware/cpu_hack/agesa_trace/pstate_targeted_disasm.txt](50_2_firmware/cpu_hack/agesa_trace/pstate_targeted_disasm.txt):

```text
fff66e3a: cmp      eax, edx
fff66e3c: je       0xfff66e7f
fff66e3e: jbe      0xfff66e42
fff66e40: mov      eax, edx
...
fff66e77: call     0xfff44e91
fff66e82: cmp      dword ptr [ebp - 4], 0xc0010068
fff66e89: jbe      0xfff66df6
```

The loop scans `MSRC001_0064` through `MSRC001_0068` and calls a WRMSR wrapper when it changes the P-state definition.

## Patch Re-check

Roadmap patch:

- `0x00366E3E`: `0x76` (`JBE`) -> `0x73` (`JAE`)
- `0x00340059`: `0x8E` -> `0x91`

Local raw bytes confirm:

- Offset `0x00366E3E` is currently `0x76`.
- Offset `0x00340059` is currently `0x8E`.

Checksum math:

- Changing `0x76` to `0x73` decreases byte sum by `0x03`.
- Changing FFS checksum byte `0x8E` to `0x91` increases byte sum by `0x03`.
- This is consistent for the local FFS checksum byte, but it does not prove platform flash acceptance or boot safety.

## Logic Result

The function extracts two VID-like fields from each P-state MSR and writes one selected value back into both positions. On the locally decoded stock P-states:

| P-state | Raw MSR | CpuVid | NbVid |
|---|---|---:|---:|
| P0 | `0x8000019e40000c14` | `0x06` | `0x20` |
| P1 | `0x8000019f40002410` | `0x12` | `0x20` |
| P2 | `0x8000017540002808` | `0x14` | `0x20` |
| P3 | `0x8000015440002c00` | `0x16` | `0x20` |
| P4 | `0x8000013540003440` | `0x1A` | `0x20` |

Original `JBE` logic selects the lower numeric VID in the compared pair. Lower numeric VID means higher voltage. The proposed `JAE` replacement appears to select the higher numeric VID instead. That does reverse the comparison in the desired undervolt direction.

The problem: it appears global, not P4-only. With the stock table, a global max-numeric selection would tend to push P0-P4 toward `0x20`, including high-frequency P0. That could produce a no-POST condition if P0 is initialized at too low a voltage.

## Capsule / Signature Layers

No local evidence of a modern signed capsule enforcement layer was found. This AMI 4 MB image has FFS checksums and Gigabyte DualBIOS/QFlash modules, but that is not enough to assume a modified image will be accepted.

Missing checks:

- Compare local dump with the official FD image after extracting the official archive.
- Run UEFITool parse on the patched image and confirm no checksum errors.
- Confirm QFlash accepts the modified image name/format without flashing.
- Confirm external programmer can erase/write/verify the exact flash chip.

## Pre-flash Checklist

Do not flash until all items pass:

1. Confirm board revision printed on the PCB.
2. Read the SPI chip twice and verify identical SHA-256 hashes.
3. Preserve the stock dump offline.
4. Confirm chip identity with flashrom using an explicit chip name.
5. Confirm CH341A/SOIC recovery works by reading the chip externally.
6. Rebuild the patch from the current stock dump, not from stale `/tmp` paths.
7. Parse patched image with UEFITool and verify checksums.
8. Prefer a P4-only or table-specific patch over the current global branch reversal.
9. If using the global `JBE -> JAE` patch anyway, first lower P0 frequency in the table or expect no-POST risk.
10. Have the stock image and external programmer ready before first boot.

## Human-only Next Step

Read-only verification on `catcas`:

```bash
ssh root@192.168.137.100 '
set -eu
sha256sum /tmp/bios_dump.bin 2>/dev/null || true
ls -l /tmp/bios_dump.bin /tmp/bios_patched.bin 2>/dev/null || true
flashrom -p internal -c "MX25L3205(A)" -r /tmp/bios_verify_1.bin
flashrom -p internal -c "MX25L3205(A)" -r /tmp/bios_verify_2.bin
sha256sum /tmp/bios_verify_1.bin /tmp/bios_verify_2.bin
cmp -l /tmp/bios_verify_1.bin /tmp/bios_verify_2.bin >/dev/null && echo "BIOS reads match"
'
```

No flash command is prepared here because the current patch is not flash-ready.

## Rejection Of Existing Flash Action

Do not flash `/tmp/bios_patched.bin` based only on the roadmap. No local patched image was found in this lab directory, and the patch appears to affect all enabled P-states rather than only the desired low-frequency state.
