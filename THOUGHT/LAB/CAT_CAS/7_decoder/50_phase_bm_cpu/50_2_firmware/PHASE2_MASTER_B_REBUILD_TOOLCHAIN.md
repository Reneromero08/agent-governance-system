# PHASE2_MASTER_B_REBUILD_TOOLCHAIN

## Verdict

`NOOP_REBUILD_PROVEN`

Route B now has a parse-clean identical no-op rebuilt image for the AmdProcessorInitPeim PE32 body.

This proves rebuild/save mechanics. It does not make AGESA byte-ready because the current P4 provenance still resolves to runtime `MSRC001_0068`, not an editable static P4-only source.

## Tooling Found And Used

Found inside `50_2_firmware/cpu_hack/tools/uefitool_rebuild/`:

| Tool | SHA-256 |
|---|---|
| `UEFIReplace.exe` | `AB05D53FCAC19651818F4EE4505813B10BADEC7A10D141836FFF3BBA8964ED8B` |
| `UEFITool.exe` | `26F85D22712361595C70EE982B1BB8CFFDCC0CA4CE1FB5049B9F7A0115A7EDED` |

The stock Windows CLI has no force-save option for identical replacements. Source review showed two suppressors:

```text
ffsengine.cpp: identical body returns ERR_NOTHING_TO_PATCH
UEFIReplace/uefireplace.cpp: reconstructed == buffer returns ERR_NOTHING_TO_PATCH
```

To prove the gate, public LongSoft/UEFITool `old_engine` source was fetched into ignored local tool tree:

```text
50_2_firmware/cpu_hack/tools/UEFITool_repo/
```

A temporary Qt/qmake build on the Linux target compiled a force-save UEFIReplace variant with only those suppressors removed.

## Accepted No-Op Replacement

Only identical body replacement was performed:

```text
UEFIReplace bios_dump.bin DE3E049C-A218-4891-8658-5FC0FA84C788 10 body.bin -o bios_noop_rebuilt.bin
```

Accepted artifact:

```text
50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin
```

## Verification

Image hashes:

```text
B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91  50_2_firmware/cpu_hack/bios_dump.bin
B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91  50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin
```

Binary compare:

```text
fc /b 50_2_firmware/cpu_hack/bios_dump.bin 50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin
FC: no differences encountered
```

Parser:

```text
UEFIExtract 50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin report
exit code 0
report: 50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin.report.txt
```

Target PE32 body:

```text
50_2_firmware/cpu_hack/noop_replace/rebuilt_AmdProcessorInitPeim_PE32_body.bin/body.bin
BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A
```

## Byte-Difference Table

| Compared files | Difference count | Explanation |
|---|---:|---|
| `50_2_firmware/cpu_hack/bios_dump.bin` vs `50_2_firmware/cpu_hack/noop_replace/bios_noop_rebuilt.bin` | 0 | Force-saved no-op output is byte-identical to stock. |

There are no changed offsets, no changed FFS checksums, and no changed target PE32 body bytes.

## Current Blocker After Route B

`AGESA_P4_SAFE_ROUTE_NOT_BYTE_READY`

The missing artifact is no longer the no-op rebuild image. The remaining missing proof is:

- editable P4-only source or edit target,
- P0-P3 unchanged proof,
- P4-only effect proof,
- offsets/bytes/checksum proof,
- clean parse proof after a non-no-op candidate.

## Safety

- No firmware was flashed.
- No voltage writes were performed.
- No patch bytes were produced for firmware behavior.
- No P0-P3 or P4 bytes were modified in an accepted image.
