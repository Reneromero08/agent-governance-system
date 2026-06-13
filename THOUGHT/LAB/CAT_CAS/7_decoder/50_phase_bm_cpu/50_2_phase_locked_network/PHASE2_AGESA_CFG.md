# PHASE2_AGESA_CFG

## Verdict

AGESA_CFG_MAPPED_P4_DISTINCTION_ONLY_AT_LOOP_COUNTER

## Evidence Base

- BIOS dump SHA-256: `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`
- Target module: `AmdProcessorInitPeim`
- GUID: `DE3E049C-A218-4891-8658-5FC0FA84C788`
- FFS file offset: `0x00340048`
- UEFI report PE32 section offset: `0x00340088`
- Corrected MZ image start: `0x0034008C`
- Image base: `0xFFF4008C`
- Normalizer raw range: `0x00366DE6-0x00366E93`
- Normalizer VA range: `0xFFF66DE6-0xFFF66E93`

## Corrected PE32 Layout

The UEFI report lists the PE32 section at `0x00340088`. The section begins with four bytes before the `MZ` header. The image used for VA/RVA mapping starts at `0x0034008C`.

Executable sections:

| Section | Raw start | Raw size | VA start | Characteristics |
|---|---:|---:|---:|---|
| `.text` | `0x003403CC` | `0x00006780` | `0xFFF403CC` | code, execute, read |
| `.tG1_PEI` | `0x00346B4C` | `0x00022A40` | `0xFFF46B4C` | code, execute, read |
| `.tG2_PEI` | `0x0036958C` | `0x00009120` | `0xFFF6958C` | code, execute, read |
| `.tG3_DXE` | `0x003726AC` | `0x0000B240` | `0xFFF726AC` | code, execute, read |

The normalizer is inside `.tG1_PEI`.

## Control Flow

Relevant instructions from `50_2_firmware/cpu_hack/agesa_trace/pstate_targeted_disasm.txt`:

```text
fff66de6: push     ebp
fff66de7: mov      ebp, esp
fff66de9: sub      esp, 0xc
fff66dec: push     ebx
fff66ded: push     esi
fff66dee: mov      dword ptr [ebp - 4], 0xc0010064
fff66df5: push     edi
fff66df6: push     dword ptr [ebp + 8]
fff66df9: lea      eax, [ebp - 0xc]
fff66dfc: push     eax
fff66dfd: push     dword ptr [ebp - 4]
fff66e00: call     0xfff44e76
fff66e05: mov      esi, dword ptr [ebp - 8]
fff66e08: mov      eax, esi
fff66e0a: shr      eax, 0x1f
fff66e0d: add      esp, 0xc
fff66e10: xor      ecx, ecx
fff66e12: cmp      eax, 1
fff66e15: jne      0xfff66e7f
fff66e17: test     ecx, ecx
fff66e19: jne      0xfff66e7f
fff66e1b: mov      ecx, dword ptr [ebp - 0xc]
fff66e1e: mov      eax, ecx
fff66e20: mov      edx, esi
fff66e22: shrd     eax, edx, 0x19
fff66e26: shr      edx, 0x19
fff66e29: mov      edx, ecx
fff66e2b: mov      edi, esi
fff66e2d: shrd     edx, edi, 9
fff66e31: and      eax, 0x7f
fff66e34: and      edx, 0x7f
fff66e37: shr      edi, 9
fff66e3a: cmp      eax, edx
fff66e3c: je       0xfff66e7f
fff66e3e: jbe      0xfff66e42
fff66e40: mov      eax, edx
fff66e42: push     dword ptr [ebp + 8]
fff66e45: and      eax, 0x7f
fff66e48: xor      edx, edx
fff66e4a: mov      edi, eax
fff66e4c: mov      ebx, edx
fff66e4e: shld     ebx, edi, 0x10
fff66e52: shl      edi, 0x10
fff66e55: or       edi, eax
fff66e57: or       ebx, edx
fff66e59: shld     ebx, edi, 9
fff66e5d: lea      eax, [ebp - 0xc]
fff66e60: and      ecx, 0x1ff01ff
fff66e66: push     eax
fff66e67: push     dword ptr [ebp - 4]
fff66e6a: shl      edi, 9
fff66e6d: or       edi, ecx
fff66e6f: or       ebx, esi
fff66e71: mov      dword ptr [ebp - 0xc], edi
fff66e74: mov      dword ptr [ebp - 8], ebx
fff66e77: call     0xfff44e91
fff66e7c: add      esp, 0xc
fff66e7f: inc      dword ptr [ebp - 4]
fff66e82: cmp      dword ptr [ebp - 4], 0xc0010068
fff66e89: jbe      0xfff66df6
fff66e8f: pop      edi
```

## Basic Blocks

| Block | Raw range | Role | Exit |
|---|---:|---|---|
| B0 | `0x00366DE6-0x00366DF5` | Prologue, initialize `[ebp-4] = 0xC0010064` | B1 |
| B1 | `0x00366DF6-0x00366E15` | Read current P-state MSR and test enable bit | B6 when disabled, else B2 |
| B2 | `0x00366E17-0x00366E3C` | Extract compared VID-like fields into `eax` and `edx` | B6 when equal, else B3 |
| B3 | `0x00366E3E-0x00366E40` | Stock selector: keep `eax` when `eax <= edx`, otherwise set `eax = edx` | B4 |
| B4 | `0x00366E42-0x00366E7C` | Rebuild MSR value and call WRMSR wrapper | B6 |
| B6 | `0x00366E7F-0x00366E89` | Increment loop counter and continue while `[ebp-4] <= 0xC0010068` | B1 or epilogue |
| B7 | `0x00366E8F-0x00366E93` | Epilogue | return |

## P4 Distinction

The visible normalizer has no separate P4 case before the selector. P4 is distinguishable only when `[ebp-4] == 0xC0010068`.

The loop sequence is:

```text
0xC0010064 -> P0
0xC0010065 -> P1
0xC0010066 -> P2
0xC0010067 -> P3
0xC0010068 -> P4
```

Therefore any P4-safe branch design must test `[ebp-4] == 0xC0010068` before changing VID selection behavior. A branch-only change at `0x00366E3E` does not satisfy that condition.

## CFG Decision

The normalizer can be mapped, but the mapped CFG does not contain an in-place P4-only decision point before the selector. A safe design would require either a proven P4-only table source or additional executable space for a branch that tests `[ebp-4]`.

