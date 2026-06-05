# AGESA Next D Actionability

Status: `MISSING_ARTIFACT_BLOCKER`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Choice

`MISSING_ARTIFACT_BLOCKER`

## Why Not The Other Choices

| Choice | Status |
|---|---|
| `TABLE_TARGET_FOUND` | Not met. P4 record and P0-P3 siblings are not proven. |
| `NOOP_REBUILD_PROVEN` | Not met. No rebuild-capable local tool exists. |
| `BOTH_LIVE_GATES_ADVANCED` | True as progress, but not the actionability verdict because exact missing artifacts still block both routes. |
| `HARD_IMPOSSIBILITY_PROOF` | Not met. Constructor path remains a live route. |

## Current Actionability

The AGESA route is still alive but not actionable.

Gate A advanced from "constructor block only" to the full containing function `0xFFF7371A` and a `.dG3_DXE` function-pointer table reference at `0xFFF8D11E`.

Gate B advanced the table-source model: the constructor consumes a dispatch/runtime-selected `selected_base` and uses `0x18` records, but no editable static P4 record is proven.

Gate C is blocked by absent local replacement/save-image tooling. Existing UEFIExtract binaries are extraction-only.

## Exact Remaining Blockers

1. `cpu_hack/AmdProcessorInitPeim_dG3_DXE_dispatch_table_consumer_decompile.txt`
   - Needed to prove how the `.dG3_DXE` table reaches `0xFFF7371A` and what source feeds `arg_0C`.
2. `cpu_hack/tools/uefitool_rebuild/UEFITool.exe`
   - Needed to perform no-op PE32 body replacement and save a rebuilt image.

