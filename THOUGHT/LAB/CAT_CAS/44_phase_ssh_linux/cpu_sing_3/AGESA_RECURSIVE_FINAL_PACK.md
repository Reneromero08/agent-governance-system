# AGESA Recursive Final Pack

Status: `MISSING_ARTIFACT_BLOCKER`

Scope: owned local Phenom II X6 1090T / GA-970A-DS3P firmware route research. No flash command. No hardware-changing command.

## Verdict

The P4-safe AGESA firmware route is still alive but not actionable from current artifacts.

No `BYTE_READY_HUMAN_REVIEW` patch candidate exists yet.
No `TABLE_TARGET_FOUND` result exists yet.
No `CODE_CAVE_OR_REPLACE_WORKFLOW_READY` result exists yet.
No hard impossibility proof exists yet.

The current stop condition is:

`MISSING_ARTIFACT_BLOCKER`

## Gate Table

| Gate | Output artifact | Result | Deepest concrete progress |
|---|---|---|---|
| Gate 1 CFG reconstruction | `AGESA_GATE1_CFG_PSEUDOCODE.md` | `GATE1_CFG_RECONSTRUCTED` | Normalizer, helper, and constructor pseudocode produced. Constructor path identifies 0x18 stride and fields at `+0x10`, `+0x14`, `+0x1C`, `+0x20`. |
| Gate 2 backing table hunt | `AGESA_GATE2_TABLE_HUNT.md` | `TABLE_TARGET_NOT_FOUND_CURRENT_BYTES_NEXT_XREF_TASK_DEFINED` | Direct xref scan, MSR constant scan, and 0x18-stride static structure scan performed. No P4 record plus P0-P3 siblings proven. |
| Gate 3 no-op replace workflow | `AGESA_GATE3_NOOP_REPLACE.md` | `NOOP_REPLACE_WORKFLOW_BLOCKED_BY_REPLACER_TOOL_MISSING` | PE32 body extracted and proven byte-identical to BIOS slice. Local tools are extraction-only; no replacer/rebuilder present. |
| Gate 4 code cave / safe injection | `AGESA_GATE4_INJECTION_PLAN.md` | `CODE_CAVE_OR_REPLACE_WORKFLOW_NOT_READY` | Executable cave scan performed. Largest executable fill run is 15 bytes; minimum safe P4-only trampoline is about 31 bytes. |
| Gate 5 patch candidate | `AGESA_GATE5_PATCH_CANDIDATE.md` | `NO_BYTE_READY_PATCH_CANDIDATE_MISSING_ARTIFACT_BLOCKER` | Table route preferred but unproven; code route blocked by missing no-op workflow and no cave; prior global branch edit remains rejected. |

## Deepest Progress Reached

Gate 1 is complete enough to drive further work.

Gate 2 reached the exact next reverse-engineering boundary: the containing function and caller/xref provenance for `0xFFF737A3` must be decompiled or annotated. The byte scans did not find a static table target.

Gate 3 reached the exact tool boundary: the extracted PE32 body is proven, but the local toolset lacks a replacement/rebuild tool.

Gate 4 reached the exact injection boundary: there is no known executable cave large enough for the minimum P4-only trampoline, and replace-body workflow is not proven.

## Exact Remaining Blocker

`MISSING_ARTIFACT_BLOCKER`

Required files/tools inside this lab:

1. `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt`
   - Must include the full function containing `0xFFF737A3`, not only the middle block.
   - Must show assignments feeding `[ebp-8]`, `ecx`, and `esi`.
   - Must show the base structure behind `+0x0B`, `+0x0F`, `+0x10`, `+0x14`, `+0x1C`, and `+0x20`.
2. `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt`
   - Must include all callers and data/control xrefs to the containing function.
3. `cpu_hack/tools/uefitool_rebuild/UEFITool.exe` or equivalent local replacement/rebuild tool.
   - Must support replace-body or replace-section and save-image.
4. `cpu_hack/noop_replace/bios_noop_rebuilt.bin`
   - Must be built by replacing the target PE32 body with identical bytes.
5. `cpu_hack/noop_replace/NOOP_DIFF_SUMMARY.txt`
   - Must prove parseability and explain every stock-vs-rebuilt byte difference.

## Exact Next Command / Tool / File Needed

Next RE command after decompiler artifact exists:

```powershell
rg -n "fff737a3|fff73a90|ebp-8|\\+0x0b|\\+0x0f|\\+0x10|\\+0x14|\\+0x1c|\\+0x20|0x18|C0010068" cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt
```

Next workflow command after replacer tool exists:

```powershell
New-Item -ItemType Directory -Force cpu_hack/noop_replace
```

Then use the replacement tool interactively or by documented CLI to replace the PE32 body at file GUID `DE3E049C-A218-4891-8658-5FC0FA84C788` with:

`cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin`

All outputs must remain inside `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux`.

## Do-Not-Do List

- Do not repeat the global `0x00366E3E: 76 -> 73` branch edit.
- Do not flash.
- Do not issue a flash command.
- Do not change voltage.
- Do not modify hardware.
- Do not claim P4-safe status without P0-P3 byte/logic equivalence proof.
- Do not use non-executable padding as code storage.
- Do not use a code cave smaller than the proven P4-only trampoline footprint.
- Do not use any output folder outside this lab.

## Route Alive?

Yes. The route remains alive because the constructor path strongly implies a per-P-state structure and no hard impossibility proof exists.

It is not actionable until the missing decompiler/xref artifacts and no-op replacement workflow exist.

