# AGESA Next C No-Op Rebuild

Status: `NOOP_REBUILD_BLOCKED_REPLACER_TOOL_MISSING`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Tool Search

Searches performed:

- Local lab `cpu_hack/tools` for UEFITool/UEFIPatch/AMI replacement tools.
- Local repo read-only search for exact rebuild-capable tool names.
- PATH command lookup for UEFITool, UEFIPatch, Ghidra/IDA/rizin-style tools.

Result:

- Rebuild-capable `UEFITool.exe`, `UEFITool_NE`, `UEFIPatch.exe`, `MMTool`, `AMIBCP`, `UEFIReplace`, or AMI replacement/save-image tool: not found.
- Existing tools found only:
  - `cpu_hack/tools/uefitool/UEFIExtract.exe`
  - `cpu_hack/tools/uefitool_A74/UEFIExtract.exe`
- These are extraction/report tools only and do not count as rebuild tools.

## No-Op Replacement Status

No no-op replacement was performed because no local replacer/save-image tool exists.

The verified target PE32 body remains:

| Item | Value |
|---|---|
| Body path | `cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin` |
| Body hash | `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A` |
| BIOS raw body start | `0x0034008C` |
| Body length | `0x56360` |

## Required Missing Tool

Exact missing tool/artifact:

`cpu_hack/tools/uefitool_rebuild/UEFITool.exe`

Acceptable equivalent:

- a local UEFITool NE/A-series GUI or CLI build that supports replacing the PE32 body/section and saving the rebuilt image, or
- `cpu_hack/tools/uefipatch/UEFIPatch.exe` plus a documented no-op patch descriptor capable of emitting a rebuilt image.

## Required Outputs After Tool Exists

Only after a real replacer exists:

- `cpu_hack/noop_replace/bios_noop_rebuilt.bin`
- `cpu_hack/noop_replace/bios_noop_rebuilt.report.txt`
- `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`

`NOOP_DIFF_SUMMARY.txt` must explain every stock-vs-rebuilt byte difference and verify that the target PE32 body hash remains `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`.

## Gate C Decision

`MISSING_ARTIFACT_BLOCKER`

No-op rebuild cannot progress further from current local tools.
