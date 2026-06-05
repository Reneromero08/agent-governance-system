# AGESA Next Gate Final Pack

Status: `MISSING_ARTIFACT_BLOCKER`

Scope: owned local Phenom II X6 1090T / GA-970A-DS3P firmware route research. No flash command. No hardware-changing command.

## Verdict

The AGESA firmware route is still alive, but not actionable.

This pass advanced both live blockers:

- Constructor path: full containing function identified as `0xFFF7371A`; `0xFFF737A3` is an internal block.
- No-op rebuild: local tool search completed; only extraction tools exist, no replacer/save-image tool found.

The current stop condition is exact:

`MISSING_ARTIFACT_BLOCKER`

## Gate Table

| Gate | Artifact | Result |
|---|---|---|
| A constructor decompile/xrefs | `cpu_sing_3/AGESA_NEXT_A_CONSTRUCTOR_DECOMPILE.md` | `CONSTRUCTOR_FUNCTION_IDENTIFIED_SOURCE_NOT_FULLY_PROVEN` |
| A export | `cpu_hack/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt` | Function entry, stack frame, base flow, and field provenance recovered. |
| A xrefs | `cpu_hack/AmdProcessorInitPeim_fff737a3_xrefs.txt` | No direct call xrefs; `.dG3_DXE` pointer xref at `0xFFF8D11E`. |
| B table reopen | `cpu_sing_3/AGESA_NEXT_B_TABLE_REOPEN.md` | Runtime/dispatch-selected source indicated; P4 static record not found. |
| C no-op rebuild | `cpu_sing_3/AGESA_NEXT_C_NOOP_REBUILD.md` | Rebuild blocked; no local replacer/save-image tool. |
| D actionability | `cpu_sing_3/AGESA_NEXT_D_ACTIONABILITY.md` | `MISSING_ARTIFACT_BLOCKER`. |

## New Artifacts Produced

- `cpu_hack/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt`
- `cpu_hack/AmdProcessorInitPeim_fff737a3_xrefs.txt`
- `cpu_sing_3/AGESA_NEXT_A_CONSTRUCTOR_DECOMPILE.md`
- `cpu_sing_3/AGESA_NEXT_B_TABLE_REOPEN.md`
- `cpu_sing_3/AGESA_NEXT_C_NOOP_REBUILD.md`
- `cpu_sing_3/AGESA_NEXT_D_ACTIONABILITY.md`
- `cpu_sing_3/AGESA_NEXT_GATE_FINAL_PACK.md`

## Deepest Progress

The constructor base is now understood at local function level:

```text
arg_0C -> edi
selected_base = edi + 8
helper 0xFFF4CF55 may update selected_base
0xFFF737A3 loads ecx = selected_base
multi-entry path uses selected_base + pstate*0x18
```

The function pointer to this constructor function is present in `.dG3_DXE`:

```text
0xFFF8D11E -> 0xFFF7371A
```

This makes the next source hunt narrower: recover the `.dG3_DXE` dispatch table consumer and the argument source for `arg_0C`.

## Exact Remaining Blocker

`MISSING_ARTIFACT_BLOCKER`

Missing constructor-source artifact:

`cpu_hack/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`

Missing no-op rebuild tool:

`cpu_hack/tools/uefitool_rebuild/UEFITool.exe`

Acceptable rebuild equivalent:

`cpu_hack/tools/uefipatch/UEFIPatch.exe` plus a documented no-op descriptor that emits a rebuilt image.

## Exact Next Command / Tool / File

After the dispatch consumer artifact exists:

```powershell
rg -n "FFF8D11E|FFF8D0EC|FFF7371A|arg_0C|selected_base|\\+0x0B|\\+0x0F|\\+0x10|\\+0x14|\\+0x1C|\\+0x20|0x18" cpu_hack/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt
```

After a rebuild-capable tool exists:

```powershell
New-Item -ItemType Directory -Force cpu_hack/noop_replace
```

Then perform only no-op replacement of:

`cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin`

and save:

`cpu_hack/noop_replace/bios_noop_rebuilt.bin`

## AGESA Route Still Alive?

Yes.

It is alive because the constructor path clearly consumes per-P-state records through a selected base and a 0x18 stride. It is not actionable because the selected base is not yet tied to editable P4/P0-P3 static records, and no no-op replace workflow is proven.

## Do-Not-Do List

- Do not produce patch bytes from this checkpoint.
- Do not repeat the rejected global branch edit.
- Do not flash.
- Do not issue a flash command.
- Do not write voltage.
- Do not modify hardware.
- Do not claim P4-safe status without P0-P3 equivalence proof.
- Do not count UEFIExtract as a rebuild tool.
- Do not write outputs outside `THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux`.

