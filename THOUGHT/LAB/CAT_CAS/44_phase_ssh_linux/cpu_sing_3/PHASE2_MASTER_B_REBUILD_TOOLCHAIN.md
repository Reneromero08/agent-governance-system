# PHASE2_MASTER_B_REBUILD_TOOLCHAIN

## Verdict

`TOOLCHAIN_ACQUIRED_FORCE_SAVE_BLOCKED`

Route B was rechecked against the current Exp44 filesystem. Rebuild-capable LongSoft classic tools are now present, but the required parse-clean identical no-op rebuilt image still does not exist.

## Local Tooling Found

Found inside `cpu_hack/tools/uefitool_rebuild/`:

| Tool | SHA-256 |
|---|---|
| `UEFIReplace.exe` | `AB05D53FCAC19651818F4EE4505813B10BADEC7A10D141836FFF3BBA8964ED8B` |
| `UEFITool.exe` | `26F85D22712361595C70EE982B1BB8CFFDCC0CA4CE1FB5049B9F7A0115A7EDED` |

Also present:

- `UEFIReplace_0.28.0_win32.zip`
- `UEFITool_0.28.0_win32.zip`
- `source_0.28.0/`

The earlier `MISSING_REBUILD_TOOLCHAIN` state is obsolete.

## No-Op Replacement Status

`NOOP_REBUILD_PROVEN` is not met.

Required output still missing:

```text
cpu_hack/noop_replace/bios_noop_rebuilt.bin
```

Existing rejected outputs:

| File | SHA-256 | Status |
|---|---|---|
| `cpu_hack/noop_replace/bios_noop_rebuilt_asis.bin` | `5D2442DB5B7733D9E6A34EB453E5936F6C270451F48B0CD27BD9B2503BDEC85A` | rejected parser output |
| `cpu_hack/noop_replace/bios_noop_rebuilt_all__asis.bin` | `5D2442DB5B7733D9E6A34EB453E5936F6C270451F48B0CD27BD9B2503BDEC85A` | rejected parser output |
| `cpu_hack/noop_replace/AmdProcessorInitPeim_PE32_section.bin` | `3E3DB0BCF0CBC7C5267CA231634FC5B70E655FE7577D6985AF7EC881C2B3513C` | generated full PE32 section attempt |

`cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt` remains the authoritative failed no-op attempt log.

## Source-Level Finding

The local LongSoft source explains why the identical CLI no-op did not save an accepted image.

Evidence:

```text
UEFIReplace_uefireplace_main.cpp:34
Usage: UEFIReplace image_file guid section_type contents_file [-o output] [-all] [-asis]

UEFIReplace_uefireplace.cpp:104
patched = result == ERR_SUCCESS;

UEFIReplace_uefireplace.cpp:121
return patched ? ERR_SUCCESS : ERR_NOTHING_TO_PATCH;

ffsengine.cpp:4812
if (body != model->body(index)) { ... }

ffsengine.cpp:4818
return ERR_NOTHING_TO_PATCH;
```

Interpretation:

- `UEFIReplace.exe` has no force-save option.
- Identical body replacement is treated as `ERR_NOTHING_TO_PATCH`.
- The CLI only writes a successful output when the object actually changes.
- Passing `-asis` with body bytes creates rejected images because the body is interpreted as a full section object.
- Passing a generated full PE32 section still did not produce an accepted no-op image for this target.

## Build Environment Check

Local build tools found:

- `cmake.exe`

Not found in the lab command environment:

- `qmake`
- `mingw32-make`
- Qt build environment needed to compile a modified LongSoft `UEFIReplace`

No local process remained after probing `UEFITool.exe`.

## Exact Current Blocker

`NOOP_REBUILD_FORCE_SAVE_BLOCKED`

The missing artifact is exact:

```text
cpu_hack/noop_replace/bios_noop_rebuilt.bin
```

The exact acceptable ways to produce it are:

1. Manual GUI force-save with classic `cpu_hack/tools/uefitool_rebuild/UEFITool.exe`, replacing the AmdProcessorInitPeim PE32 body with the identical extracted `body.bin`, then saving the image.
2. A Qt/qmake build environment capable of compiling a modified `UEFIReplace` that treats an identical matched replacement as a force-save success.
3. Another public CLI replacer with documented force-output behavior for identical replacements.

## Required Acceptance After Artifact Exists

When `cpu_hack/noop_replace/bios_noop_rebuilt.bin` exists:

1. Parse the rebuilt image with local UEFI extraction tooling.
2. Verify the target PE32 body hash remains:

```text
BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A
```

3. Create a fresh byte-difference table against stock.
4. Keep the verdict as `NOOP_REBUILD_PROVEN` only if the image is parse-clean and the target body hash is unchanged.

## Safety

- No firmware was flashed.
- No voltage writes were performed.
- No patch bytes were produced.
- No P0-P3 or P4 bytes were modified in an accepted image.
