# AGESA Recursive Final Pack

Status: `NOOP_REBUILD_PROVEN_P4_SOURCE_MISSING`

Scope: owned local Phenom II X6 1090T / GA-970A-DS3P firmware route research. No flash command. No hardware-changing command.

## Verdict

The P4-safe AGESA firmware route is still alive but not actionable from current artifacts.

No `BYTE_READY_HUMAN_REVIEW` patch candidate exists yet.
No `TABLE_TARGET_FOUND` result exists yet.
No-op replace/rebuild workflow is now proven. No P4-only byte-ready target exists yet.
No hard impossibility proof exists yet.

The current stop condition is:

`NOOP_REBUILD_PROVEN_BUT_P4_STATIC_SOURCE_MISSING`

## Gate Table

| Gate | Output artifact | Result | Deepest concrete progress |
|---|---|---|---|
| Gate 1 CFG reconstruction | `AGESA_GATE1_CFG_PSEUDOCODE.md` | `GATE1_CFG_RECONSTRUCTED` | Normalizer, helper, and constructor pseudocode produced. Constructor path identifies 0x18 stride and fields at `+0x10`, `+0x14`, `+0x1C`, `+0x20`. |
| Gate 2 backing table hunt | `AGESA_GATE2_TABLE_HUNT.md` | `TABLE_TARGET_NOT_FOUND_CURRENT_BYTES_NEXT_XREF_TASK_DEFINED` | Direct xref scan, MSR constant scan, and 0x18-stride static structure scan performed. No P4 record plus P0-P3 siblings proven. |
| Gate 3 no-op replace workflow | `AGESA_GATE3_NOOP_REPLACE.md` | `NOOP_REBUILD_PROVEN` | PE32 body extracted, force-saved no-op rebuilt image produced, parse report clean, stock-vs-rebuilt byte diff zero, target body hash preserved. |
| Gate 4 code cave / safe injection | `AGESA_GATE4_INJECTION_PLAN.md` | `CODE_CAVE_OR_REPLACE_WORKFLOW_NOT_READY` | Executable cave scan performed. Largest executable fill run is 15 bytes; minimum safe P4-only trampoline is about 31 bytes. |
| Gate 5 patch candidate | `AGESA_GATE5_PATCH_CANDIDATE.md` | `NO_BYTE_READY_PATCH_CANDIDATE_P4_SOURCE_MISSING` | Table route preferred but unproven; code route blocked by no P4-only edit source and no cave; prior global branch edit remains rejected. |

## Deepest Progress Reached

Gate 1 is complete enough to drive further work.

Gate 2 reached the exact next reverse-engineering boundary: the containing function and caller/xref provenance for `0xFFF737A3` must be decompiled or annotated. The byte scans did not find a static table target.

Gate 3 is complete: the extracted PE32 body is proven and a force-saved identical rebuilt image is parse-clean and byte-identical.

Gate 4 reached the exact injection boundary: there is no known executable cave large enough for the minimum P4-only trampoline.

## Exact Remaining Blocker

`NOOP_REBUILD_PROVEN_BUT_P4_STATIC_SOURCE_MISSING`

Required proof inside this lab:

1. `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt`
   - Must include the full function containing `0xFFF737A3`, not only the middle block.
   - Must show assignments feeding `[ebp-8]`, `ecx`, and `esi`.
   - Must show the base structure behind `+0x0B`, `+0x0F`, `+0x10`, `+0x14`, `+0x1C`, and `+0x20`.
2. `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt`
   - Must include all callers and data/control xrefs to the containing function.
3. Editable P4-only source or edit target.
4. P0-P3 unchanged proof and P4-only effect proof.
5. Offsets/bytes/checksum proof plus clean parse proof after a non-no-op candidate.

## Exact Next Command / Tool / File Needed

Next RE command after decompiler artifact exists:

```powershell
rg -n "fff737a3|fff73a90|ebp-8|\\+0x0b|\\+0x0f|\\+0x10|\\+0x14|\\+0x1c|\\+0x20|0x18|C0010068" cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt
```

No-op rebuild workflow is proven; do not repeat it unless validating a future non-no-op candidate. All outputs must remain inside `THOUGHT/LAB/CAT_CAS/50_phase_bm_cpu`.

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

It is not actionable until P4-only edit-source proof exists.
