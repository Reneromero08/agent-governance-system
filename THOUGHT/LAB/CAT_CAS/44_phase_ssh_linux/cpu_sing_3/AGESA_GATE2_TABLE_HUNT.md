# AGESA Gate 2 Table Hunt

Status: `TABLE_TARGET_NOT_FOUND_CURRENT_BYTES_NEXT_XREF_TASK_DEFINED`

Scope: owned local firmware route research. No flash command. No hardware-changing command.

## Inputs Consumed

- Gate 1 constructor pseudocode.
- `cpu_hack/agesa_trace/pstate_targeted_disasm.txt`
- `cpu_hack/agesa_trace/pstate_mask_hits.txt`
- `cpu_hack/bios_parse/bios_dump.bin.report.txt`
- `cpu_hack/bios_dump.bin`
- `cpu_hack/bios_dump.bin.dump/.../AmdProcessorInitPeim/1 PE32 image section/body.bin`

## Direct Results

### PE32 Body Identity

The extracted PE32 body is byte-identical to the BIOS slice at raw `0x0034008C`.

| Item | Value |
|---|---|
| PE32 body length | `0x56360` |
| PE32 body SHA-256 | `BF92A1321B98908E7D74299A6C1E629EC3583599F164DEC6E774BFF040FBDF2A` |
| MZ header | `4D 5A` |

### Constructor Xref Scan

Direct `E8 rel32` call scan inside the target PEIM found:

| Target | Direct call xrefs |
|---|---:|
| Normalizer entry `0xFFF66DE6` | 0 |
| Helper entry `0xFFF6788D` | 0 |
| Constructor block entry `0xFFF737A3` | 0 |
| Constructor-adjacent window `0xFFF73700-0xFFF73B00` | 1 call: raw `0x00373BDC`, VA `0xFFF73BDC`, target `0xFFF73A94` |

Interpretation: the requested constructor block is likely inside a larger function entered earlier than `0xFFF737A3`, or reached by fall-through / indirect control flow. A direct call-xref-only method cannot prove the caller and struct base.

### Immediate Constant Scan In Target PEIM

| Immediate | Count | First raw hits |
|---|---:|---|
| `0xC0010064` | 23 | `0x0034EAE1`, `0x0034EDFD`, `0x0034EFB8`, `0x0035953B`, `0x00359ECF`, `0x003668D4`, `0x00366BA6`, `0x00366D91`, `0x00366DF1`, `0x00367490`, `0x0036756B`, `0x0036786C`, `0x003678A8`, `0x0036B308`, `0x0037052A`, `0x003734CD`, `0x003737BC`, `0x00373857`, `0x0037398E`, `0x003757BD` |
| `0xC0010065` | 9 | `0x00366B94`, `0x00366BB7`, `0x00366CA2`, `0x00366D2D`, `0x00367577`, `0x00367594`, `0x00367836`, `0x0037386B`, `0x0037583D` |
| `0xC0010066` | 0 | none as full immediate |
| `0xC0010067` | 0 | none as full immediate |
| `0xC0010068` | 5 | `0x00366D6C`, `0x00366E85`, `0x0036732E`, `0x003675B2`, `0x00367910` |

The constants prove multiple P-state writer/reader paths but do not identify a static P4 table record.

### 0x18-Stride Structure Scan

The scan tested candidate five-sibling static structures with record stride `0x18`, record `+0x10` as a 0/1 enable flag, record `+0x1C` as a 7-bit byte field, and record `+0x20` as a two-bit field.

Results:

- Strict clusters with five sibling records and plausible `+0x10`, `+0x1C`, `+0x20`: `0`.
- Known sequence at `record + 0x1C` for stock P4 fragments: `0`.
- Known sequence at `record + 0x1C` for CpuVid-like values `06 12 14 16 1A`: `0`.
- Loose clusters: 2, but both have all-zero `+0x1C` values and high unrelated-looking `+0x20` dwords:
  - raw `0x0038C9E9`, VA `0xFFF8C9E9`
  - raw `0x0038D751`, VA `0xFFF8D751`

The loose clusters are not valid P0-P4 sibling proof because they do not contain a P4-specific field, do not match known VID/qword fragments, and are not tied to the constructor base register.

## Current Table Verdict

`TABLE_TARGET_NOT_FOUND`

No editable P4 record and sibling P0-P3 records are proven from the current BIOS image, extracted PE32 body, disassembly, and byte scans.

This is not the endpoint. The blocker becomes the next RE task below.

## Exact Next Xref / Decompiler Task

Required missing artifact:

`MISSING_ARTIFACT_BLOCKER: decompiler database or annotated disassembly for the full function containing 0xFFF737A3, including function entry, stack frame, all callers, and register provenance for [ebp-8] / ecx / esi at 0xFFF737A3.`

Exact next local task:

1. Open `cpu_hack/bios_dump.bin.dump/5 8C8CE578-8A3D-4F1C-9935-896185C32DD3/0 AmdProcessorInitPeim/1 PE32 image section/body.bin` in Ghidra or IDA as PE32/i386 with image base `0xFFF4008C`.
2. Define or recover the function that contains `0xFFF737A3`.
3. Export decompiler pseudocode and xrefs for:
   - Function entry through `0xFFF73A90`.
   - All callers of the containing function.
   - All assignments to `[ebp-8]` before `0xFFF737A3`.
   - The provenance of the base pointer whose fields are read at `+0x0B`, `+0x0F`, `+0x10`, `+0x14`, `+0x1C`, and `+0x20`.
4. Name the exported artifact inside this lab:
   - `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_containing_function_decompile.txt`
   - `cpu_hack/agesa_trace/AmdProcessorInitPeim_fff737a3_xrefs.txt`

Gate 2 can reopen only when that artifact proves either:

- a static table address with P4 record and sibling P0-P3 records, or
- a runtime-only structure with no static editable backing table in this image.

