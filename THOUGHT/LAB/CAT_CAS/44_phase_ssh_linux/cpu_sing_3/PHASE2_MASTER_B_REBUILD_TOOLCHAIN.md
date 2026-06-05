# PHASE2_MASTER_B_REBUILD_TOOLCHAIN

## Verdict

`MISSING_REBUILD_TOOLCHAIN`

Route B was rechecked. The lab contains extraction tools but no local replace/save-image tool.

## Local Search Result

Found:

- `cpu_hack/tools/uefitool/UEFIExtract.exe`
- `cpu_hack/tools/uefitool_A74/UEFIExtract.exe`
- `cpu_hack/tools/uefitool/UEFIExtract_NE_A72_win64.zip`
- `cpu_hack/tools/uefitool_A74/UEFIExtract_NE_A74_win64.zip`

Not found:

- `UEFITool.exe`
- `UEFITool_NE` GUI/editor binary
- `UEFIPatch.exe`
- `UEFIReplace.exe`
- `MMTool`
- `AMIBCP`
- Any local tool capable of replace-body plus save-image

`UEFIExtract` remains extraction-only and does not satisfy the no-op rebuild gate.

## No-Op Replacement Status

Not performed. The required replacer is absent.

Expected no-op target if the tool is acquired:

- Input image: `cpu_hack/bios_dump.bin`
- Target extracted body: `cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin`
- Required output: `cpu_hack/noop_replace/bios_noop_rebuilt.bin`
- Required diff report: `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`
- Required target body SHA-256 after rebuild: `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`

## Exact Public Tool Needed

Preferred acquisition:

- `UEFITool.exe` from LongSoft UEFITool classic release `0.28.0`
- Public source/release family: `https://github.com/LongSoft/UEFITool`
- Release page checked: `https://github.com/LongSoft/UEFITool/releases/tag/0.28.0`

Acceptable equivalent:

- UEFITool NE/A-series GUI build that explicitly supports replacing a PE32 section body and saving the rebuilt image.
- `UEFIReplace.exe` plus a documented descriptor workflow.
- `UEFIPatch.exe` only if a no-op descriptor can replace the identical PE32 body and save a parsed image.
- AMI tool capable of module body replacement and image save, used only for no-op replacement first.

## Install Checklist Inside Lab

1. Create `cpu_hack/tools/uefitool_rebuild/`.
2. Place rebuild-capable `UEFITool.exe` there.
3. Record tool filename, version, SHA-256, and source URL in a markdown note.
4. Run only no-op PE32 body replacement first.
5. Parse the rebuilt image with local extraction tooling.
6. Verify the target PE32 body hash remains `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A`.
7. Create `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt` explaining every byte difference between stock and rebuilt images.

## Route B Outcome

`NOOP_REBUILD_PROVEN` is not met.

Route B is alive but blocked by one exact artifact:

`cpu_hack/tools/uefitool_rebuild/UEFITool.exe`

No firmware bytes were changed.
