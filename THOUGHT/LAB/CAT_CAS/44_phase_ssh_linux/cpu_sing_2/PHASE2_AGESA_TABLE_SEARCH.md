# PHASE2_AGESA_TABLE_SEARCH

## Verdict

TABLE_TARGET_NOT_FOUND_FROM_CURRENT_ARTIFACTS

## Search Inputs

Read-only binary search was run against:

- `cpu_hack/bios_dump.bin`
- SHA-256: `B7C0C725C4B6F50F399A208E5CAD6938BAACDD8FA1BBC795098CA393083FBC91`
- Length: `4,194,304` bytes
- Target PEIM corrected MZ start: `0x0034008C`
- Target PEIM size used for search: `0x00056360`

## MSR Constant Hits

Little-endian 32-bit constants in the 4 MB BIOS:

| Constant | Count | First hits |
|---|---:|---|
| `0xC0010064` | 28 | `0x0034EAE1`, `0x0034EDFD`, `0x0034EFB8`, `0x0035953B`, `0x00359ECF`, `0x003668D4`, `0x00366BA6`, `0x00366D91`, `0x00366DF1`, `0x00367490`, `0x0036756B`, `0x0036786C`, `0x003678A8` |
| `0xC0010065` | 10 | `0x00366B94`, `0x00366BB7`, `0x00366CA2`, `0x00366D2D`, `0x00367577`, `0x00367594`, `0x00367836`, `0x0037386B`, `0x0037583D`, `0x003BF727` |
| `0xC0010066` | 1 | `0x003BF731` |
| `0xC0010067` | 1 | `0x003BF73B` |
| `0xC0010068` | 6 | `0x00366D6C`, `0x00366E85`, `0x0036732E`, `0x003675B2`, `0x00367910`, `0x003BF745` |

The constants prove multiple P-state code paths, but not a static table record for P4.

## Stock P-state Qword Search

Exact little-endian 8-byte searches:

| P-state | Stock qword | Count |
|---|---|---:|
| P0 | `0x8000019E40000C14` | 0 |
| P1 | `0x8000019F40002410` | 0 |
| P2 | `0x8000017540002808` | 0 |
| P3 | `0x8000015440002C00` | 0 |
| P4 | `0x8000013540003440` | 0 |
| P4 test qword | `0x80000135400040C0` | 0 |

Stride search for the five stock qwords with record strides from 8 through 64 bytes found no match.

## VID Constant Search

Target PEIM byte counts:

| Byte | Meaning in this lab | Count in target PEIM |
|---|---|---:|
| `0x1A` | stock P4 CpuVid | 327 |
| `0x20` | stock NbVid and requested lower-voltage VID | 1739 |

The counts are too broad to identify a P4-only table record. Compact sequence searches also failed:

| Pattern | Count |
|---|---:|
| `06 12 14 16 1A` | 0 |
| five `0x20` bytes | 0 |
| `64 00 01 C0 65 00 01 C0 66 00 01 C0 67 00 01 C0 68 00 01 C0` | 0 |

## Normalizer Window Hit

The normalizer byte sequence is unique:

```text
pattern count: 1
hit: 0x00366E20
```

Window `0x00366DF0-0x00366E8F`:

```text
FC 64 00 01 C0 57 FF 75 08 8D 45 F4 50 FF 75 FC
E8 71 E0 FD FF 8B 75 F8 8B C6 C1 E8 1F 83 C4 0C
33 C9 83 F8 01 75 68 85 C9 75 64 8B 4D F4 8B C1
8B D6 0F AC D0 19 C1 EA 19 8B D1 8B FE 0F AC FA
09 83 E0 7F 83 E2 7F C1 EF 09 3B C2 74 41 76 02
8B C2 FF 75 08 83 E0 7F 33 D2 8B F8 8B DA 0F A4
FB 10 C1 E7 10 0B F8 0B DA 0F A4 FB 09 8D 45 F4
81 E1 FF 01 FF 01 50 FF 75 FC C1 E7 09 0B F9 0B
DE 89 7D F4 89 5D F8 E8 15 E0 FD FF 83 C4 0C FF
45 FC 81 7D FC 68 00 01 C0 0F 86 67 FF FF FF 5F
```

## Executable Cave Search

The corrected PE image has these executable sections:

| Section | Raw start | Raw size |
|---|---:|---:|
| `.text` | `0x003403CC` | `0x00006780` |
| `.tG1_PEI` | `0x00346B4C` | `0x00022A40` |
| `.tG2_PEI` | `0x0036958C` | `0x00009120` |
| `.tG3_DXE` | `0x003726AC` | `0x0000B240` |

Search result:

- No all-zero run of 32 bytes or larger in any executable section.
- No all-`0xFF` run of 32 bytes or larger in any executable section.
- No all-`0xCC` run of 32 bytes or larger in any executable section.
- No all-`0x90` run of 32 bytes or larger in any executable section.

## Table Search Decision

A P4-only table target was not found in the current dump, report, and disassembly artifacts. The constructor-like paths strongly suggest per-P-state structures exist at runtime, but the static backing bytes are not identified well enough for a table edit.

