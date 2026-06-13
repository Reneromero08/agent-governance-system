# AGESA Gate 3 No-Op Replace Workflow

Status: `NOOP_REBUILD_PROVEN`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## What Is Proven

The target PE32 body is already extracted inside the lab:

`cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin`

Identity proof:

| Item | Value |
|---|---|
| BIOS raw section start | `0x0034008C` |
| PE32 body length | `0x56360` |
| PE32 body SHA-256 | `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A` |
| BIOS slice SHA-256 | `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A` |
| Match | yes |

Available local extraction binaries:

- `cpu_hack/tools/uefitool/UEFIExtract.exe`
- `cpu_hack/tools/uefitool_A74/UEFIExtract.exe`

Both local binaries are extraction/report tools. Their help output supports report, dump, unpack, and GUID extraction modes. They do not provide replace-body and save-image functionality.

Rebuild/save proof was later produced through a temporary Qt/qmake build of public LongSoft `old_engine` UEFIReplace with identical-body force-save behavior enabled.

## Required No-Op Workflow

The no-op workflow must prove the editor/rebuilder path before any logic change:

1. Load `cpu_hack/bios_dump.bin`.
2. Locate file GUID `DE3E049C-A218-4891-8658-5FC0FA84C788`.
3. Locate the PE32 image section at raw `0x00340088`, body start `0x0034008C`.
4. Replace the PE32 image section body with the identical `body.bin`.
5. Save rebuilt image inside this lab folder.
6. Parse the rebuilt image with UEFIExtract.
7. Verify:
   - image parses without structure errors,
   - target FFS file remains present,
   - target PE32 body hash remains `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`,
   - stock vs rebuilt diff is limited to expected checksum bytes, or ideally no bytes differ for identical replacement.

## Accepted Output

`NOOP_REBUILD_PROVEN`

Accepted files:

- `cpu_hack/noop_replace/bios_noop_rebuilt.bin`
- `cpu_hack/noop_replace/bios_noop_rebuilt.bin.report.txt`
- `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`

Verification:

- stock and rebuilt image SHA-256 match,
- `fc /b` reports no differences,
- UEFIExtract report mode exits 0,
- rebuilt target PE32 body hash remains `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`.

Gate 3 is proven. This does not create a P4-safe candidate; Gate 5 remains blocked by missing P4-only edit-source proof.
