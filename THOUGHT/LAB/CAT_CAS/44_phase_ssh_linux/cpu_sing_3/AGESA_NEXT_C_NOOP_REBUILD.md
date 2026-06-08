# AGESA Next C No-Op Rebuild

Status: `NOOP_REBUILD_PROVEN`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Tool Search

Searches and follow-up actions performed:

- Local lab `cpu_hack/tools` searched for UEFITool/UEFIPatch/AMI replacement tools.
- Existing UEFIExtract tools were treated as extraction-only and not counted as rebuild tools.
- LongSoft classic 0.28.0 tooling was found under `cpu_hack/tools/uefitool_rebuild/`.
- Public LongSoft/UEFITool `old_engine` source was fetched into ignored local tool tree `cpu_hack/tools/UEFITool_repo/`.
- Qt5 qmake/g++ toolchain on the Linux target was used to compile a temporary force-save UEFIReplace variant.

## No-Op Replacement Performed

Only identical-body replacement was performed:

```text
UEFIReplace bios_dump.bin DE3E049C-A218-4891-8658-5FC0FA84C788 10 body.bin -o bios_noop_rebuilt.bin
```

Accepted output:

```text
cpu_hack/noop_replace/bios_noop_rebuilt.bin
```

The rebuilt image SHA-256 matches stock:

```text
B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91
```

`fc /b` reported no differences between `cpu_hack/bios_dump.bin` and `cpu_hack/noop_replace/bios_noop_rebuilt.bin`.

## Parser And Body Verification

UEFIExtract report mode parsed the rebuilt image with exit code 0:

```text
cpu_hack/noop_replace/bios_noop_rebuilt.bin.report.txt
```

The target PE32 body was extracted from the rebuilt image and hashed:

```text
cpu_hack/noop_replace/rebuilt_AmdProcessorInitPeim_PE32_body.bin/body.bin
BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A
```

## Gate C Decision

`NOOP_REBUILD_PROVEN`

This proves a parse-clean force-saved no-op rebuild path exists for this target PE32 body.

It does not prove a P4-safe edit target. The firmware route is still blocked from byte-ready review until the P4-only source/edit target is proven with P0-P3 unchanged, P4-only effect, offsets/bytes/checksums, and clean parse evidence.

## Safety

- No BIOS flash.
- No voltage writes.
- No board modification.
- No firmware behavior patch bytes.
- No P0-P3 or P4 bytes modified in the accepted image.
