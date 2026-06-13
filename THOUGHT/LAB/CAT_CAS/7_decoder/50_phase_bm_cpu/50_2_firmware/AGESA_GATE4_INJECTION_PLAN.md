# AGESA Gate 4 Code Cave / Safe Injection Plan

Status: `CODE_CAVE_OR_REPLACE_WORKFLOW_NOT_READY`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Required P4-Safe Logic

Any code-route patch must preserve P0-P3 byte/logic behavior and only alter P4 behavior. Minimum condition:

```asm
cmp dword ptr [ebp-4], 0xC0010068
jne original_stock_selector_path
; alternate behavior only for P4
```

The rejected global byte at raw `0x00366E3E` changes selector behavior for every P-state in the loop and is not P4-safe.

## Minimum Trampoline Footprint

At the normalizer selector area:

```text
0x00366E3A: 3B C2          cmp eax, edx
0x00366E3C: 74 41          je  0xFFF66E7F
0x00366E3E: 76 02          jbe 0xFFF66E42
0x00366E40: 8B C2          mov eax, edx
0x00366E42: FF 75 08       push dword ptr [ebp+8]
```

A safe detour must preserve or reproduce the stock selector and the overwritten `push [ebp+8]`. A practical P4-specific detour needs at least:

- 5 bytes for near jump out of the normalizer path,
- 7 bytes for `cmp [ebp-4], 0xC0010068`,
- 2 bytes for conditional branch back to stock behavior,
- stock selector bytes for non-P4,
- alternate selector bytes for P4,
- 3 bytes to preserve `push [ebp+8]`,
- 5 bytes to jump back to `0xFFF66E45`.

Minimum realistic payload: about 31 bytes before alignment and validation.

## Executable Cave Scan

Executable sections from the corrected PE32 body:

| Section | Raw start | Raw size |
|---|---:|---:|
| `.text` | `0x003403CC` | `0x00006780` |
| `.tG1_PEI` | `0x00346B4C` | `0x00022A40` |
| `.tG2_PEI` | `0x0036958C` | `0x00009120` |
| `.tG3_DXE` | `0x003726AC` | `0x0000B240` |

Largest executable fill runs found:

| Fill | Largest run | Location |
|---|---:|---|
| `0x00` | 15 bytes | `.tG1_PEI`, raw `0x0036957D-0x0036958B` |
| `0x00` | 12 bytes | `.tG2_PEI`, raw `0x003726A0-0x003726AB` |
| `0xCC` | 10 bytes | `.text`, raw `0x00345762-0x0034576B` |
| `0xCC` | 8 bytes | `.text`, raw `0x00345CC4-0x00345CCB` |
| `0xFF` | none >= 8 bytes | none |
| `0x90` | none >= 8 bytes | none |

No executable run is large enough for the minimum P4-only selector trampoline.

## Other Injection Options

| Option | Status | Reason |
|---|---|---|
| In-place selector rewrite | Rejected | Cannot test `[ebp-4] == 0xC0010068`; affects P0-P3. |
| Small executable cave | Blocked | Largest cave is 15 bytes; minimum safe logic is about 31 bytes. |
| Dead code reuse | Not proven | Requires decompiler xrefs proving a target block is unreachable in all boot paths. |
| Section expansion | Not ready | Requires a proven no-op replace/rebuild workflow first. |
| Replace-body workflow | Not ready | Gate 3 replacer/rebuilder tool is missing. |
| Table edit | Preferred but not found | Gate 2 has no proven P4 record and sibling P0-P3 records. |

## Blocker

`MISSING_ARTIFACT_BLOCKER: no executable cave large enough, no dead-code proof, and no proven replace-body workflow.`

Exact next artifact needed:

- Gate 3 no-op rebuilt image proof, or
- decompiler/xref proof that an executable block of at least 31 bytes in `AmdProcessorInitPeim` is unreachable and can be safely reused, or
- a replacement/rebuild workflow that can expand or replace the PE32 body while preserving firmware structure.

Until one of those artifacts exists, Gate 4 cannot produce `CODE_CAVE_OR_REPLACE_WORKFLOW_READY`.

